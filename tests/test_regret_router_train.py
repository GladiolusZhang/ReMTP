from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

import torch

from remtp.regret_router_model import load_regret_router_checkpoint
from remtp.regret_router_train import RouterTraceDataset, train


def synthetic_record(seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    heads, vocab, top_k, hidden = 2, 7, 3, 8
    p = torch.softmax(torch.randn(heads, vocab, generator=generator), dim=-1)
    q = torch.softmax(torch.randn(heads, vocab, generator=generator), dim=-1)
    p_probs, p_ids = p.topk(top_k, dim=-1)
    q_probs, q_ids = q.topk(top_k, dim=-1)
    direction = torch.randn(hidden, generator=generator)
    output = torch.randn(vocab, hidden, generator=generator)
    p_delta = (output[p_ids] * direction).sum(dim=-1)
    q_delta = (output[q_ids] * direction).sum(dim=-1)
    draft_ids = q.argmax(dim=-1)
    rows = torch.arange(heads)
    p_y = p[rows, draft_ids]
    q_y = q[rows, draft_ids]
    allocated = torch.minimum(
        (q_y - p_y).clamp_min(0.0), torch.full_like(p_y, 0.1)
    )
    strict = torch.minimum(torch.ones_like(p_y), p_y / q_y)
    relaxed = torch.minimum(torch.ones_like(p_y), (p_y + allocated) / q_y)
    return {
        "root_hidden": torch.randn(hidden, generator=generator).to(torch.float16),
        "regret_direction": direction.to(torch.float16),
        "regret_debt": torch.tensor([0.05 + 0.01 * seed]),
        "source_entropy": torch.tensor([0.5]),
        "source_margin": torch.tensor([0.2]),
        "target_entropy": torch.tensor([0.4, 0.6]),
        "target_margin": torch.tensor([0.3, 0.1]),
        "head_reliability": torch.tensor([1.0, 0.5]),
        "p_top_ids": p_ids.to(torch.int32),
        "p_top_probs": p_probs,
        "q_top_ids": q_ids.to(torch.int32),
        "q_top_probs": q_probs,
        "p_on_q_top": p.gather(1, q_ids),
        "q_on_p_top": q.gather(1, p_ids),
        "p_direction_delta": p_delta,
        "q_direction_delta": q_delta,
        "draft_token_ids": draft_ids.to(torch.int32),
        "target_candidate_probs": p_y,
        "draft_candidate_probs": q_y,
        "allocated_tv": allocated,
        "strict_acceptance": strict,
        "relaxed_acceptance": relaxed,
        "accepted": torch.ones(heads, dtype=torch.bool),
        "reallocation_fraction": torch.zeros(heads),
        "expected_regret": allocated * (relaxed - strict),
    }


class RegretRouterTrainTest(unittest.TestCase):
    def test_compact_support_is_normalized(self) -> None:
        example = RouterTraceDataset([synthetic_record(1)])[0]
        torch.testing.assert_close(
            example["p_compact"].sum(dim=-1), torch.ones(2)
        )
        torch.testing.assert_close(
            example["q_compact"].sum(dim=-1), torch.ones(2)
        )

    def test_tiny_cpu_training_writes_loadable_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = [synthetic_record(index) for index in range(8)]
            torch.save(
                {"format_version": 1, "records": records},
                root / "router_shard_000000.pt",
            )
            output = root / "router.pt"
            args = argparse.Namespace(
                data_dir=root,
                output=output,
                device="cpu",
                epochs=1,
                batch_size=4,
                learning_rate=3e-4,
                weight_decay=1e-4,
                validation_fraction=0.25,
                seed=42,
                rank=2,
                width=8,
                max_direction_strength=0.03,
                max_logit_scale=0.08,
                max_budget_reduction=0.5,
                intervention_weight=0.25,
                entropy_weight=0.1,
                acceptance_weight=1.0,
                risk_weight=2.0,
                debt_reference=0.05,
                grad_clip=1.0,
            )
            report = train(args)
            model, metadata = load_regret_router_checkpoint(output)
            self.assertEqual(model.architecture.hidden_size, 8)
            self.assertEqual(model.architecture.num_heads, 2)
            self.assertEqual(metadata["records"], 8)
            self.assertEqual(report["metadata"]["records"], 8)


if __name__ == "__main__":
    unittest.main()
