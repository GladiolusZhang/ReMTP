import unittest
from types import SimpleNamespace

import torch

from remtp.adaptive_scheduler import (
    shrink_deferred_speculation,
    shrink_runner_schedule,
)


class _StructuredOutputManager:
    @staticmethod
    def should_advance(request: object) -> bool:
        return False


class _Request:
    is_prefill_chunk = False

    def __init__(self) -> None:
        self.num_computed_tokens = 107
        self.num_output_placeholders = 7

    @staticmethod
    def is_finished() -> bool:
        return False


class AdaptiveSchedulerTest(unittest.TestCase):
    def test_gpu_only_async_path_uses_actual_proposal_width(self) -> None:
        runner = SimpleNamespace(
            _draft_token_ids=torch.tensor([[11, 12, 13, 14]]),
            input_batch=SimpleNamespace(prev_req_id_to_index={"req": 0}),
        )
        output = SimpleNamespace(
            has_structured_output_requests=False,
            scheduled_spec_decode_tokens={"req": [-1] * 6},
            num_scheduled_tokens={"req": 7},
            total_num_scheduled_tokens=7,
        )

        removed = shrink_runner_schedule(runner, output)

        self.assertEqual(removed, {"req": 2})
        self.assertEqual(output.scheduled_spec_decode_tokens["req"], [-1] * 4)
        self.assertEqual(output.num_scheduled_tokens["req"], 5)
        self.assertEqual(output.total_num_scheduled_tokens, 5)
        self.assertEqual(output._remtp_adaptive_removed_drafts, {"req": 2})

    def test_short_draft_repairs_deferred_async_accounting(self) -> None:
        request = _Request()
        scheduler = SimpleNamespace(
            requests={"req": request},
            structured_output_manager=_StructuredOutputManager(),
        )
        drafts = SimpleNamespace(
            req_ids=["req"],
            draft_token_ids=[[11, 12, 13, 14]],
        )
        output = SimpleNamespace(
            scheduled_spec_decode_tokens={"req": [-1] * 6},
            num_scheduled_tokens={"req": 7},
            total_num_scheduled_tokens=7,
            num_invalid_spec_tokens=None,
        )

        changed = shrink_deferred_speculation(scheduler, drafts, output)

        self.assertTrue(changed)
        self.assertEqual(output.scheduled_spec_decode_tokens["req"], [11, 12, 13, 14])
        self.assertEqual(output.num_scheduled_tokens["req"], 5)
        self.assertEqual(output.total_num_scheduled_tokens, 5)
        self.assertEqual(request.num_computed_tokens, 105)
        self.assertEqual(request.num_output_placeholders, 5)
        self.assertEqual(output.num_invalid_spec_tokens, {})

    def test_full_width_draft_is_untouched(self) -> None:
        request = _Request()
        scheduler = SimpleNamespace(
            requests={"req": request},
            structured_output_manager=_StructuredOutputManager(),
        )
        drafts = SimpleNamespace(
            req_ids=["req"],
            draft_token_ids=[[11, 12, 13, 14, 15, 16]],
        )
        output = SimpleNamespace(
            scheduled_spec_decode_tokens={"req": [-1] * 6},
            num_scheduled_tokens={"req": 7},
            total_num_scheduled_tokens=7,
            num_invalid_spec_tokens=None,
        )

        changed = shrink_deferred_speculation(scheduler, drafts, output)

        self.assertFalse(changed)
        self.assertEqual(output.scheduled_spec_decode_tokens["req"], [-1] * 6)
        self.assertEqual(output.num_scheduled_tokens["req"], 7)
        self.assertEqual(request.num_computed_tokens, 107)


if __name__ == "__main__":
    unittest.main()
