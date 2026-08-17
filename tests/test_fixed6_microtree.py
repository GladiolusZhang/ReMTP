from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from remtp.fixed6_microtree import (
    BranchState,
    commit_selected_state,
    get_topology,
    materialize_branch_states,
    sample_without_replacement,
    strict_tree_verify,
)
from remtp.fixed6_vllm import (
    _RUNTIME,
    _processed_target_probs,
    _selected_target_hidden_row,
)


class Fixed6TargetProbabilityTests(unittest.TestCase):
    def test_selected_tree_leaf_maps_to_its_actual_target_hidden_row(self) -> None:
        previous = _RUNTIME.selected_nodes
        try:
            _RUNTIME.selected_nodes = ()
            self.assertEqual(_selected_target_hidden_row(), 0)
            _RUNTIME.selected_nodes = (0, 2, 5)
            self.assertEqual(_selected_target_hidden_row(), 6)
        finally:
            _RUNTIME.selected_nodes = previous

    def test_processed_target_logits_are_not_temperature_scaled_twice(self) -> None:
        raw_logits = torch.tensor([[0.0, 1.0, -0.5]])
        temperature = torch.tensor([0.5])
        metadata = SimpleNamespace(temperature=temperature)
        processed_logits = raw_logits / temperature[0]

        actual = _processed_target_probs(processed_logits, metadata)
        expected = torch.softmax(raw_logits / temperature[0], dim=-1)
        double_scaled = torch.softmax(
            raw_logits / temperature[0] / temperature[0], dim=-1
        )
        torch.testing.assert_close(actual, expected)
        self.assertFalse(torch.allclose(actual, double_scaled))

    def test_processed_target_logits_keep_greedy_one_hot_semantics(self) -> None:
        metadata = SimpleNamespace(temperature=torch.tensor([0.0]))
        actual = _processed_target_probs(
            torch.tensor([[0.5, 2.0, 1.0]]), metadata
        )
        torch.testing.assert_close(actual, torch.tensor([[0.0, 1.0, 0.0]]))


class Fixed6TopologyTests(unittest.TestCase):
    def test_static_layouts_are_exactly_six_breadth_first_nodes(self) -> None:
        expected = {
            "6-chain": (6,),
            "4+2": (4, 2),
            "3+2+1": (3, 2, 1),
        }
        for name, lengths in expected.items():
            topology = get_topology(name)
            self.assertEqual(topology.branch_lengths, lengths)
            self.assertEqual(len(topology.nodes), 6)
            self.assertEqual(sum(map(len, topology.branch_nodes)), 6)
            self.assertEqual(topology.branch_major_to_bfs, tuple(
                topology.branch_major.index(i) for i in range(6)
            ))

    def test_4_plus_2_is_not_a_flat_chain(self) -> None:
        topology = get_topology("4+2")
        self.assertEqual(
            topology.choices,
            ((0,), (1,), (0, 0), (1, 0), (0, 0, 0), (0, 0, 0, 0)),
        )
        mask = topology.causal_tree_mask()
        # Input rows 1 and 2 are sibling roots and cannot see one another.
        self.assertFalse(bool(mask[1, 2]))
        self.assertFalse(bool(mask[2, 1]))
        # The final long-branch node sees its own ancestors, not branch 1.
        self.assertTrue(bool(mask[6, 1]))
        self.assertTrue(bool(mask[6, 3]))
        self.assertFalse(bool(mask[6, 2]))
        self.assertFalse(bool(mask[6, 4]))

    def test_positions_parents_and_recurrent_rows(self) -> None:
        topology = get_topology("3+2+1")
        self.assertEqual(topology.node_position_offsets, (1, 1, 1, 2, 2, 3))
        self.assertEqual(topology.parent_logit_rows, (0, 0, 0, 1, 2, 4))
        self.assertEqual(
            topology.state_index_matrix((11, 12, 13, 14, 15, 16)),
            ((11, 14, 16), (12, 15, -1), (13, -1, -1)),
        )


class Fixed6StateTests(unittest.TestCase):
    @staticmethod
    def _root() -> BranchState:
        return BranchState(
            kv=torch.tensor([1.0, 2.0]),
            gdn_conv=torch.tensor([3.0, 4.0]),
            gdn_ssm=torch.tensor([5.0, 6.0]),
            mtp=torch.tensor([7.0, 8.0]),
            position=10,
        )

    @staticmethod
    def _step(state: BranchState, token: int, node) -> BranchState:
        # Mutate the received state on purpose. Correct branch cloning must keep
        # this mutation local to one branch.
        state.kv.add_(token + node.depth)
        state.gdn_conv.mul_(1.0 + token / 100.0)
        state.gdn_ssm.add_(state.kv * 0.01)
        state.mtp.copy_(state.mtp.roll(1) + token)
        state.position += 1
        return state

    def test_branch_states_are_isolated_and_match_independent_execution(self) -> None:
        topology = get_topology("3+2+1")
        tokens = (2, 3, 5, 7, 11, 13)
        states = materialize_branch_states(topology, self._root(), tokens, self._step)

        for branch in topology.branch_nodes:
            reference = self._root()
            for node_index in branch:
                reference = self._step(
                    reference, tokens[node_index], topology.nodes[node_index]
                )
                actual = states[node_index]
                for actual_tensor, expected_tensor in zip(
                    actual.tensors(), reference.tensors(), strict=True
                ):
                    torch.testing.assert_close(actual_tensor, expected_tensor)
                self.assertEqual(actual.position, reference.position)

        before = states[1].kv.clone()
        states[0].kv.add_(999)
        torch.testing.assert_close(states[1].kv, before)

    def test_commit_is_a_copy_of_only_the_selected_leaf(self) -> None:
        topology = get_topology("4+2")
        tokens = (2, 3, 5, 7, 11, 13)
        root = self._root()
        states = materialize_branch_states(topology, root, tokens, self._step)
        selected = topology.path_to(topology.leaves[1])
        committed = commit_selected_state(root, states, selected)
        expected = states[selected[-1]]
        for actual_tensor, expected_tensor in zip(
            committed.tensors(), expected.tensors(), strict=True
        ):
            torch.testing.assert_close(actual_tensor, expected_tensor)
            self.assertNotEqual(
                actual_tensor.untyped_storage().data_ptr(),
                expected_tensor.untyped_storage().data_ptr(),
            )


class StrictTreeVerificationTests(unittest.TestCase):
    def test_all_accepted_returns_one_branch_plus_bonus(self) -> None:
        topology = get_topology("4+2")
        vocab = 8
        draft_ids = torch.tensor([0, 1, 2, 3, 4, 5])
        draft_probs = torch.nn.functional.one_hot(
            draft_ids, num_classes=vocab
        ).to(torch.float32)
        target = torch.full((7, vocab), 1e-9)
        # Make branch 0 deterministic, including its bonus.
        for node_index in topology.branch_nodes[0]:
            row = topology.parent_logit_rows[node_index]
            target[row, draft_ids[node_index]] = 1.0
        target[topology.branch_nodes[0][-1] + 1, 7] = 1.0
        target /= target.sum(dim=-1, keepdim=True)
        result = strict_tree_verify(topology, draft_ids, draft_probs, target)
        self.assertEqual(result.accepted_node_indices, topology.branch_nodes[0])
        self.assertEqual(result.output_token_ids[-1], 7)
        self.assertEqual(result.terminal, "bonus")
        self.assertEqual(result.target_forward_calls, 1)
        self.assertEqual(result.target_validation_nodes, 6)

    def test_root_multi_candidate_residual_preserves_target_distribution(self) -> None:
        # The verifier is stochastic; compare the first committed token against
        # the target categorical distribution over many deterministic seeds.
        topology = get_topology("3+2+1")
        target_root = torch.tensor([0.42, 0.27, 0.18, 0.09, 0.04])
        draft_root = torch.tensor([0.30, 0.25, 0.20, 0.15, 0.10])
        counts = torch.zeros(5)
        trials = 12000
        for seed in range(trials):
            generator = torch.Generator().manual_seed(seed)
            root_ids, root_q = sample_without_replacement(
                draft_root, len(topology.roots), generator
            )
            draft_ids = torch.zeros(6, dtype=torch.int64)
            draft_probs = torch.zeros((6, 5))
            for root_offset, node_index in enumerate(topology.roots):
                draft_ids[node_index] = root_ids[root_offset]
                draft_probs[node_index] = root_q[root_offset]
            # Child rows cannot affect the first output token.
            for node_index in set(range(6)) - set(topology.roots):
                draft_ids[node_index] = 0
                draft_probs[node_index] = torch.tensor([1, 0, 0, 0, 0])
            target = target_root.repeat(7, 1)
            result = strict_tree_verify(
                topology, draft_ids, draft_probs, target, generator
            )
            counts[result.output_token_ids[0]] += 1
        empirical = counts / trials
        torch.testing.assert_close(empirical, target_root, atol=0.015, rtol=0.0)


if __name__ == "__main__":
    unittest.main()
