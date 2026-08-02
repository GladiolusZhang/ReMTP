import itertools
import unittest

import torch

from remtp.block_verification import (
    block_verification_state,
    longest_accepted_prefix,
)


class BlockVerificationTest(unittest.TestCase):
    def test_toy_example_matches_paper_subblock_probabilities(self) -> None:
        target = torch.tensor([[1.0 / 3.0, 2.0 / 3.0]] * 2)
        draft = torch.tensor([[2.0 / 3.0, 1.0 / 3.0]] * 2)

        aa = block_verification_state(target, draft, torch.tensor([0, 0]))
        torch.testing.assert_close(
            aa.prefix_joint_probability,
            torch.tensor([0.5, 0.25]),
        )
        torch.testing.assert_close(
            aa.subblock_acceptance_probability,
            torch.tensor([0.0, 0.25]),
        )

        ab = block_verification_state(target, draft, torch.tensor([0, 1]))
        torch.testing.assert_close(
            ab.subblock_acceptance_probability,
            torch.tensor([0.0, 1.0]),
        )

        ba = block_verification_state(target, draft, torch.tensor([1, 0]))
        torch.testing.assert_close(
            ba.subblock_acceptance_probability,
            torch.tensor([1.0, 0.5]),
        )

        bb = block_verification_state(target, draft, torch.tensor([1, 1]))
        torch.testing.assert_close(
            bb.subblock_acceptance_probability,
            torch.ones(2),
        )

    def test_toy_example_expected_acceptance_is_eleven_ninths(self) -> None:
        target = torch.tensor([[1.0 / 3.0, 2.0 / 3.0]] * 2)
        draft = torch.tensor([[2.0 / 3.0, 1.0 / 3.0]] * 2)
        q = draft[0]
        expected = 0.0
        for first, second in itertools.product(range(2), repeat=2):
            ids = torch.tensor([first, second])
            state = block_verification_state(target, draft, ids)
            # Exact expectation of max accepted length for two independent
            # subblock tests: P(L>=1) + P(L>=2).
            h1, h2 = state.subblock_acceptance_probability.tolist()
            fixed_path_expectation = h1 * (1.0 - h2) + 2.0 * h2
            expected += q[first].item() * q[second].item() * fixed_path_expectation
        self.assertAlmostEqual(expected, 11.0 / 9.0, places=6)

    def test_correction_scale_tracks_committed_prefix(self) -> None:
        target = torch.tensor([[0.4, 0.6], [0.7, 0.3], [0.2, 0.8]])
        draft = torch.tensor([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
        state = block_verification_state(
            target,
            draft,
            torch.tensor([0, 0, 1]),
        )
        torch.testing.assert_close(
            state.correction_scale,
            torch.cat(
                (
                    torch.ones(1),
                    state.prefix_joint_probability[:-1],
                )
            ),
        )

    def test_longest_prefix_can_skip_an_earlier_failed_test(self) -> None:
        accepted = longest_accepted_prefix(
            torch.tensor([0.1, 0.9, 0.8]),
            torch.tensor([0.5, 0.2, 0.3]),
        )
        self.assertEqual(accepted.item(), 3)


if __name__ == "__main__":
    unittest.main()
