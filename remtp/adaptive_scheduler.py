"""Async-scheduler support for variable-width speculative drafts.

vLLM's asynchronous scheduler reserves the configured maximum number of
speculative positions before the proposer result is available. Upstream pads
shorter proposer results with ``-1``. Qwen3.5's recurrent GDN attention cannot
consume those invalid positions, so adaptive-depth ReMTP shrinks the deferred
scheduler output before it reaches the model runner.

The patch is intentionally narrow: it is installed only for the
``adaptive_depth`` experiment, single-request text generation, and leaves the
proposal tokens and verifier unchanged.
"""

from __future__ import annotations

from typing import Any

import torch


_REMOVED_ATTR = "_remtp_adaptive_removed_drafts"


def shrink_runner_schedule(runner: Any, scheduler_output: Any) -> dict[str, int]:
    """Shrink the normal GPU-only async path to the prior proposal width."""

    drafts = getattr(runner, "_draft_token_ids", None)
    if not torch.is_tensor(drafts) or drafts.ndim != 2:
        return {}
    if scheduler_output.has_structured_output_requests:
        return {}
    previous_rows = runner.input_batch.prev_req_id_to_index
    if previous_rows is None:
        return {}

    proposal_width = int(drafts.shape[1])
    removed_by_request: dict[str, int] = {}
    for req_id, placeholders in scheduler_output.scheduled_spec_decode_tokens.items():
        previous_row = previous_rows.get(req_id)
        if previous_row is None or previous_row >= drafts.shape[0]:
            continue
        removed = len(placeholders) - proposal_width
        if removed <= 0:
            continue
        current_scheduled = scheduler_output.num_scheduled_tokens[req_id]
        if current_scheduled <= removed:
            raise RuntimeError(
                "adaptive runner shrink would remove the target token"
            )
        scheduler_output.scheduled_spec_decode_tokens[req_id] = placeholders[
            :proposal_width
        ]
        scheduler_output.num_scheduled_tokens[req_id] = current_scheduled - removed
        scheduler_output.total_num_scheduled_tokens -= removed
        removed_by_request[req_id] = removed

    if removed_by_request:
        setattr(scheduler_output, _REMOVED_ATTR, removed_by_request)
    return removed_by_request


def shrink_deferred_speculation(
    scheduler: Any,
    draft_token_ids: Any,
    scheduler_output: Any,
) -> bool:
    """Replace fixed placeholders with shorter drafts and repair accounting.

    Returns ``True`` when at least one request was shortened. Structured output
    requests are deliberately left to the upstream implementation, because
    grammar filtering uses ``-1`` padding as an explicit contract.
    """

    shortened = False
    scheduled = scheduler_output.scheduled_spec_decode_tokens
    for req_id, proposed in zip(
        draft_token_ids.req_ids,
        draft_token_ids.draft_token_ids,
        strict=True,
    ):
        request = scheduler.requests.get(req_id)
        if request is None or request.is_finished() or request.is_prefill_chunk:
            continue
        placeholders = scheduled.get(req_id)
        if not placeholders or len(proposed) >= len(placeholders):
            continue
        if scheduler.structured_output_manager.should_advance(request):
            continue

        proposal = list(proposed)
        removed = len(placeholders) - len(proposal)
        if removed <= 0:
            continue

        current_scheduled = scheduler_output.num_scheduled_tokens[req_id]
        if current_scheduled <= removed:
            raise RuntimeError(
                "adaptive draft shrink would remove the target token"
            )
        if request.num_computed_tokens < removed:
            raise RuntimeError("adaptive draft shrink underflowed computed tokens")
        if request.num_output_placeholders < removed:
            raise RuntimeError("adaptive draft shrink underflowed placeholders")

        scheduled[req_id] = proposal
        scheduler_output.num_scheduled_tokens[req_id] = current_scheduled - removed
        scheduler_output.total_num_scheduled_tokens -= removed
        request.num_computed_tokens -= removed
        request.num_output_placeholders -= removed
        shortened = True

    if shortened:
        # These positions are intentionally absent, not grammar-invalid drafts.
        scheduler_output.num_invalid_spec_tokens = {}
    return shortened


def install_adaptive_async_scheduler() -> None:
    """Teach vLLM's deferred async path to preserve a short draft width."""

    from vllm.v1.core.sched.async_scheduler import AsyncScheduler
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    current = AsyncScheduler.update_draft_token_ids_in_output
    if getattr(current, "_remtp_adaptive_depth", False):
        return

    def update_draft_token_ids_in_output(
        self: Any,
        draft_token_ids: Any,
        scheduler_output: Any,
    ) -> None:
        if shrink_deferred_speculation(self, draft_token_ids, scheduler_output):
            return
        current(self, draft_token_ids, scheduler_output)

    update_draft_token_ids_in_output._remtp_adaptive_depth = True  # type: ignore[attr-defined]
    AsyncScheduler.update_draft_token_ids_in_output = (  # type: ignore[method-assign]
        update_draft_token_ids_in_output
    )

    current_update = AsyncScheduler.update_from_output
    if not getattr(current_update, "_remtp_adaptive_depth", False):

        def update_from_output(
            self: Any,
            scheduler_output: Any,
            model_runner_output: Any,
        ) -> Any:
            removed_by_request = getattr(scheduler_output, _REMOVED_ATTR, {})
            for req_id, removed in removed_by_request.items():
                request = self.requests.get(req_id)
                if request is None or request.is_finished():
                    continue
                if request.num_computed_tokens < removed:
                    raise RuntimeError(
                        "adaptive output repair underflowed computed tokens"
                    )
                if request.num_output_placeholders < removed:
                    raise RuntimeError(
                        "adaptive output repair underflowed placeholders"
                    )
                request.num_computed_tokens -= removed
                request.num_output_placeholders -= removed
            return current_update(self, scheduler_output, model_runner_output)

        update_from_output._remtp_adaptive_depth = True  # type: ignore[attr-defined]
        AsyncScheduler.update_from_output = update_from_output  # type: ignore[method-assign]

    current_execute = GPUModelRunner.execute_model
    if not getattr(current_execute, "_remtp_adaptive_depth", False):

        def execute_model(
            self: Any,
            scheduler_output: Any,
            intermediate_tensors: Any = None,
        ) -> Any:
            shrink_runner_schedule(self, scheduler_output)
            return current_execute(self, scheduler_output, intermediate_tensors)

        execute_model._remtp_adaptive_depth = True  # type: ignore[attr-defined]
        GPUModelRunner.execute_model = execute_model  # type: ignore[method-assign]
