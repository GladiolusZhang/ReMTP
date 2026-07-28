"""Read-only token tracing hooks for vLLM speculative decoding.

The hook supports both the newer GPU-worker rejection sampler and the legacy
V1 rejection sampler.  It intentionally relies only on public Python tensor
operations and leaves vLLM's outputs untouched.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import os
import sys
from collections.abc import Sequence
from types import ModuleType
from typing import Any


_GPU_REJECTION_MODULE = "vllm.v1.worker.gpu.spec_decode.rejection_sampler"
_LEGACY_REJECTION_MODULE = "vllm.v1.sample.rejection_sampler"
_TARGET_MODULES = {_GPU_REJECTION_MODULE, _LEGACY_REJECTION_MODULE}

_round_number = 0
_limit_reported = False
_tokenizer: Any | None = None
_tokenizer_attempted = False
_trace_error_reported = False


def _as_int_list(value: Any) -> list[int]:
    """Copy a small tensor/array/sequence to a plain list of ints."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (int, float)):
        return [int(value)]
    return [int(item) for item in value]


def _nested_int_lists(value: Any) -> list[list[int]]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [[int(item) for item in row] for row in value]


def _get_tokenizer() -> Any | None:
    global _tokenizer, _tokenizer_attempted
    if _tokenizer_attempted:
        return _tokenizer
    _tokenizer_attempted = True

    model = os.getenv("REMTP_MODEL", "Qwen/Qwen3.5-4B")
    try:
        from transformers import AutoTokenizer

        # vLLM has already downloaded the tokenizer before the first real
        # decode round. local_files_only prevents tracing from doing I/O.
        _tokenizer = AutoTokenizer.from_pretrained(
            model,
            local_files_only=True,
            trust_remote_code=True,
        )
    except Exception as exc:
        print(
            f"[ReMTP] tokenizer unavailable; showing token IDs only: {exc}",
            file=sys.stderr,
            flush=True,
        )
        _tokenizer = None
    return _tokenizer


def token_label(token_id: int) -> str:
    """Return a compact ``id(decoded-piece)`` label."""
    tokenizer = _get_tokenizer()
    if tokenizer is None:
        return str(token_id)
    try:
        piece = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        return f"{token_id}({piece!r})"
    except Exception:
        return str(token_id)


def format_round(
    round_number: int,
    request_id: str,
    previous_token: int | None,
    draft_tokens: Sequence[int],
    target_tokens: Sequence[int],
    emitted_tokens: Sequence[int],
    accepted_count: int,
) -> str:
    """Format one greedy speculative-verification round."""
    draft_tokens = list(draft_tokens)
    target_tokens = list(target_tokens)
    emitted_tokens = list(emitted_tokens)
    num_drafts = len(draft_tokens)

    lines = [f"[ReMTP][round {round_number:03d}][request {request_id}]"]
    if previous_token is not None:
        lines.append(f"  TARGET previous : {token_label(previous_token)}")

    draft_text = "  ".join(
        f"D{index}={token_label(token)}"
        for index, token in enumerate(draft_tokens)
    )
    lines.append(f"  MTP draft       : {draft_text or '(none)'}")

    verify_targets = target_tokens[:num_drafts]
    target_text = "  ".join(
        f"T{index}={token_label(token)}"
        for index, token in enumerate(verify_targets)
    )
    if len(target_tokens) > num_drafts:
        target_text += f"  bonus={token_label(target_tokens[num_drafts])}"
    lines.append(f"  TARGET verify   : {target_text or '(none)'}")

    checks: list[str] = []
    mismatch_seen = False
    for index, (draft, target) in enumerate(zip(draft_tokens, verify_targets)):
        if mismatch_seen:
            checks.append(f"D{index} skipped")
        elif index < accepted_count and draft == target:
            checks.append(f"D{index}==T{index} ✓")
        else:
            checks.append(f"D{index}!=T{index} ✗")
            mismatch_seen = True
    check_text = "  ".join(checks) or "(no draft)"
    lines.append(
        f"  VERIFY          : {check_text}  "
        f"-> accepted {accepted_count}/{num_drafts}"
    )
    lines.append(
        "  COMMIT          : "
        + (" ".join(token_label(token) for token in emitted_tokens) or "(none)")
    )
    return "\n".join(lines)


def _next_round() -> int | None:
    global _round_number, _limit_reported
    limit = max(0, int(os.getenv("REMTP_TRACE_MAX_ROUNDS", "64")))
    if _round_number >= limit:
        if not _limit_reported:
            print(
                f"[ReMTP] trace limit ({limit} rounds) reached; generation continues.",
                flush=True,
            )
            _limit_reported = True
        return None
    _round_number += 1
    return _round_number


def _emit_round(
    request_id: str,
    previous_token: int | None,
    draft_tokens: Sequence[int],
    target_tokens: Sequence[int],
    emitted_tokens: Sequence[int],
    accepted_count: int,
) -> None:
    round_number = _next_round()
    if round_number is None:
        return
    print(
        format_round(
            round_number,
            request_id,
            previous_token,
            draft_tokens,
            target_tokens,
            emitted_tokens,
            accepted_count,
        ),
        flush=True,
    )


def _trace_gpu_result(
    logits: Any,
    input_batch: Any,
    sampler_output: Any,
) -> None:
    """Trace vLLM's GPU-worker rejection sampler (v0.18+ nightlies)."""
    # Each request owns K draft positions plus one target bonus position.
    boundaries = _as_int_list(input_batch.cu_num_logits)
    model_inputs = _as_int_list(
        input_batch.input_ids[input_batch.logits_indices]
    )
    target_candidates = _as_int_list(logits.argmax(dim=-1))
    sampled_rows = _nested_int_lists(sampler_output.sampled_token_ids)
    sampled_counts = _as_int_list(sampler_output.num_sampled)
    request_ids = list(getattr(input_batch, "req_ids", ()))

    for index in range(len(boundaries) - 1):
        start, end = boundaries[index], boundaries[index + 1]
        num_drafts = max(0, end - start - 1)
        if num_drafts == 0:
            continue
        previous = model_inputs[start]
        drafts = model_inputs[start + 1 : end]
        targets = target_candidates[start:end]
        count = max(0, min(sampled_counts[index], len(sampled_rows[index])))
        emitted = sampled_rows[index][:count]
        # A speculative round always emits one recovery/bonus token in
        # addition to the accepted prefix.
        accepted = max(0, min(num_drafts, count - 1))
        request_id = str(request_ids[index]) if index < len(request_ids) else str(index)
        _emit_round(request_id, previous, drafts, targets, emitted, accepted)


def _patch_gpu_rejection_sampler(module: ModuleType) -> None:
    rejection_sampler = getattr(module, "RejectionSampler", None)
    if rejection_sampler is None or getattr(rejection_sampler, "_remtp_traced", False):
        return

    original_call = rejection_sampler.__call__

    def traced_call(self: Any, logits: Any, input_batch: Any, *args: Any, **kwargs: Any):
        output = original_call(self, logits, input_batch, *args, **kwargs)
        try:
            _trace_gpu_result(logits, input_batch, output)
        except Exception as exc:
            _report_trace_error(exc)
        return output

    rejection_sampler.__call__ = traced_call
    rejection_sampler._remtp_traced = True
    print(f"[ReMTP] tracing {_GPU_REJECTION_MODULE}", flush=True)


def _trace_legacy_result(
    metadata: Any,
    logits: Any,
    sampler_output: Any,
    sampling_metadata: Any,
) -> None:
    """Trace the legacy V1 rejection sampler used by older nightlies."""
    output_histories = getattr(sampling_metadata, "output_token_ids", None)
    if output_histories is not None and all(
        len(history) == 0 for history in output_histories
    ):
        # vLLM warms the rejection sampler with synthetic empty histories.
        return

    target_logits = logits[metadata.target_logits_indices]
    target_candidates = _as_int_list(target_logits.argmax(dim=-1))
    drafts = _as_int_list(metadata.draft_token_ids)
    counts = [int(item) for item in metadata.num_draft_tokens]
    sampled_rows = _nested_int_lists(sampler_output.sampled_token_ids)

    offset = 0
    for index, num_drafts in enumerate(counts):
        if num_drafts == 0:
            continue
        request_drafts = drafts[offset : offset + num_drafts]
        request_targets = target_candidates[offset : offset + num_drafts]
        offset += num_drafts
        emitted = [token for token in sampled_rows[index] if token >= 0]
        accepted = max(0, min(num_drafts, len(emitted) - 1))
        if accepted == num_drafts and emitted:
            # The last emitted token is target-only when every draft is
            # accepted. The legacy API does not expose its logits separately.
            request_targets.append(emitted[-1])
        previous = None
        if output_histories is not None and index < len(output_histories):
            history = output_histories[index]
            if history:
                previous = int(history[-1])
        _emit_round(
            str(index),
            previous,
            request_drafts,
            request_targets,
            emitted,
            accepted,
        )


def _patch_legacy_rejection_sampler(module: ModuleType) -> None:
    rejection_sampler = getattr(module, "RejectionSampler", None)
    if rejection_sampler is None or getattr(rejection_sampler, "_remtp_traced", False):
        return

    original_forward = rejection_sampler.forward

    def traced_forward(
        self: Any,
        metadata: Any,
        draft_probs: Any,
        logits: Any,
        sampling_metadata: Any,
    ):
        output = original_forward(
            self,
            metadata,
            draft_probs,
            logits,
            sampling_metadata,
        )
        try:
            _trace_legacy_result(metadata, logits, output, sampling_metadata)
        except Exception as exc:
            _report_trace_error(exc)
        return output

    rejection_sampler.forward = traced_forward
    rejection_sampler._remtp_traced = True
    print(f"[ReMTP] tracing {_LEGACY_REJECTION_MODULE}", flush=True)


def _report_trace_error(exc: Exception) -> None:
    global _trace_error_reported
    if _trace_error_reported:
        return
    _trace_error_reported = True
    print(
        f"[ReMTP] trace disabled after a non-fatal compatibility error: {exc}",
        file=sys.stderr,
        flush=True,
    )


def _patch_module(fullname: str, module: ModuleType) -> None:
    if fullname == _GPU_REJECTION_MODULE:
        _patch_gpu_rejection_sampler(module)
    elif fullname == _LEGACY_REJECTION_MODULE:
        _patch_legacy_rejection_sampler(module)


class _PatchLoader(importlib.abc.Loader):
    def __init__(self, fullname: str, wrapped: importlib.abc.Loader):
        self.fullname = fullname
        self.wrapped = wrapped

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType | None:
        create_module = getattr(self.wrapped, "create_module", None)
        return create_module(spec) if create_module is not None else None

    def exec_module(self, module: ModuleType) -> None:
        self.wrapped.exec_module(module)
        _patch_module(self.fullname, module)


class _PatchFinder(importlib.abc.MetaPathFinder):
    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        if fullname not in _TARGET_MODULES:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is None or spec.loader is None:
            return spec
        if not hasattr(spec.loader, "exec_module"):
            return spec
        spec.loader = _PatchLoader(fullname, spec.loader)
        return spec


def install_import_hook() -> None:
    """Install the lazy hook without importing torch or vLLM in the API process."""
    if any(isinstance(finder, _PatchFinder) for finder in sys.meta_path):
        return

    # Handle an unusual embedding that imported vLLM before sitecustomize.
    for fullname in _TARGET_MODULES:
        module = sys.modules.get(fullname)
        if module is not None:
            _patch_module(fullname, module)

    sys.meta_path.insert(0, _PatchFinder())
