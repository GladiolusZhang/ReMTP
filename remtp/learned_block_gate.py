"""Runtime and calibration wrapper for the minimal logistic block gate."""

from __future__ import annotations

import atexit
import importlib
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any, TextIO

import torch

from remtp.block_gate import (
    LogisticBlockGate,
    block_gate_features,
    counterfactual_gate_record,
    logistic_gate_probability,
)
from remtp.target_anchored_mtp import (
    TargetAnchoredConfig,
    boost_candidate_logits,
    target_anchored_distribution,
)


_REJECTION_MODULE = "vllm.v1.sample.rejection_sampler"
_COLLECT_HANDLE: TextIO | None = None
_COLLECT_REQUEST_GROUP = -1
_COLLECT_LAST_OUTPUT_LENGTH: int | None = None
_COLLECT_LAST_REQUEST_FINGERPRINT: tuple[tuple[int, int, int], ...] | None = None
_DIAGNOSTIC_EMITTED = False


def _close_collect_handle() -> None:
    global _COLLECT_HANDLE
    if _COLLECT_HANDLE is not None:
        _COLLECT_HANDLE.close()
        _COLLECT_HANDLE = None


def _request_group(sampling_metadata: Any) -> tuple[int, int]:
    global _COLLECT_REQUEST_GROUP, _COLLECT_LAST_OUTPUT_LENGTH
    global _COLLECT_LAST_REQUEST_FINGERPRINT
    output_rows = getattr(sampling_metadata, "output_token_ids", None)
    output_length = len(output_rows[0]) if output_rows else 0
    generators = getattr(sampling_metadata, "generators", {})
    fingerprint = tuple(
        sorted(
            (
                int(index),
                int(generator.initial_seed()),
                id(generator),
            )
            for index, generator in generators.items()
        )
    )
    new_request = False
    if fingerprint:
        new_request = fingerprint != _COLLECT_LAST_REQUEST_FINGERPRINT
        _COLLECT_LAST_REQUEST_FINGERPRINT = fingerprint
    elif _COLLECT_LAST_OUTPUT_LENGTH is None:
        new_request = True
    elif output_length < _COLLECT_LAST_OUTPUT_LENGTH:
        new_request = True
    if new_request:
        _COLLECT_REQUEST_GROUP += 1
    _COLLECT_LAST_OUTPUT_LENGTH = output_length
    return _COLLECT_REQUEST_GROUP, output_length


def _block_candidates(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    draft_token_ids: torch.Tensor,
    config: TargetAnchoredConfig,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    result = target_anchored_distribution(
        target_probs,
        draft_probs,
        draft_token_ids,
        config,
        assume_normalized=True,
        construct_probs=False,
    )
    p_y = result.target_candidate_probs
    q_y = result.draft_candidate_probs
    shield_h = result.boosted_candidate_probs
    cactus_h = (p_y + result.cactus_tv).clamp(max=1.0)
    return p_y, q_y, result.target_log_gaps, cactus_h, shield_h


def _learned_block_gate_rejection_sample(
    draft_token_ids: torch.Tensor,
    num_draft_tokens: list[int],
    max_spec_len: int,
    cu_num_draft_tokens: torch.Tensor,
    draft_probs: torch.Tensor | None,
    target_logits: torch.Tensor,
    bonus_token_ids: torch.Tensor,
    sampling_metadata: Any,
) -> torch.Tensor:
    global _DIAGNOSTIC_EMITTED

    original = getattr(
        _learned_block_gate_rejection_sample,
        "_remtp_original",
    )
    if (
        sampling_metadata.all_greedy
        or target_logits.shape[0] == 0
        or draft_probs is None
    ):
        return original(
            draft_token_ids,
            num_draft_tokens,
            max_spec_len,
            cu_num_draft_tokens,
            draft_probs,
            target_logits,
            bonus_token_ids,
            sampling_metadata,
        )
    if len(num_draft_tokens) != 1:
        raise RuntimeError("minimal block gate requires --max-num-seqs 1")

    logits = target_logits.to(torch.float32)
    target_probs = logits.softmax(dim=-1, dtype=torch.float32)
    mode = getattr(_learned_block_gate_rejection_sample, "_remtp_mode")
    config = getattr(_learned_block_gate_rejection_sample, "_remtp_config")

    if mode == "infer":
        fast_route = getattr(
            _learned_block_gate_rejection_sample,
            "_remtp_fast_route",
        )
        p_y, chosen_h, probability, shield_route = fast_route(
            target_probs,
            draft_probs,
            draft_token_ids,
        )
    else:
        p_y, q_y, log_gap, cactus_h, shield_h = _block_candidates(
            target_probs,
            draft_probs,
            draft_token_ids,
            config,
        )
        record = counterfactual_gate_record(
            target_probs=target_probs,
            draft_probs=draft_probs,
            draft_token_ids=draft_token_ids,
            target_candidate_probs=p_y,
            draft_candidate_probs=q_y,
            target_log_gaps=log_gap,
            cactus_candidate_probs=cactus_h,
            shield_candidate_probs=shield_h,
            surplus_spend_fraction=getattr(
                _learned_block_gate_rejection_sample,
                "_remtp_surplus_spend_fraction",
            ),
        )
        request_group, output_length = _request_group(sampling_metadata)
        record.update(
            {
                "request_group": request_group,
                "output_length_before_block": output_length,
            }
        )
        assert _COLLECT_HANDLE is not None
        _COLLECT_HANDLE.write(json.dumps(record, separators=(",", ":")) + "\n")
        action = getattr(
            _learned_block_gate_rejection_sample,
            "_remtp_collect_action",
        )
        shield_route = torch.tensor(
            action == "shield",
            device=target_probs.device,
        )
        probability = shield_route.to(torch.float32)
        chosen_h = torch.where(shield_route, shield_h, cactus_h)

    verification_logits = boost_candidate_logits(
        logits,
        draft_token_ids,
        p_y,
        chosen_h,
    )
    if (
        not _DIAGNOSTIC_EMITTED
        and os.getenv("REMTP_BLOCK_GATE_DIAGNOSTICS", "0") == "1"
    ):
        print(
            "[ReMTP][BlockGate][diagnostic] "
            f"mode={mode} probability={probability.item():.6f} "
            f"route={'shield' if shield_route.item() else 'cactus'} "
            f"p={p_y.tolist()} h={chosen_h.tolist()}",
            flush=True,
        )
        _DIAGNOSTIC_EMITTED = True
    return original(
        draft_token_ids,
        num_draft_tokens,
        max_spec_len,
        cu_num_draft_tokens,
        draft_probs,
        verification_logits,
        bonus_token_ids,
        sampling_metadata,
    )


def install_learned_block_gate() -> None:
    """Install calibration collection or JSON logistic inference."""
    global _COLLECT_HANDLE

    mode = os.getenv("REMTP_BLOCK_GATE_MODE", "infer")
    if mode not in {"collect", "infer"}:
        raise ValueError("REMTP_BLOCK_GATE_MODE must be collect or infer")
    config = replace(
        TargetAnchoredConfig.from_env(),
        variant="tv_block_shield",
    )
    config.validate()
    collect_action = os.getenv("REMTP_BLOCK_GATE_COLLECT_ACTION", "cactus")
    if collect_action not in {"cactus", "shield"}:
        raise ValueError("collect action must be cactus or shield")
    surplus_spend_fraction = float(
        os.getenv("REMTP_BLOCK_GATE_SURPLUS_SPEND_FRACTION", "0.5")
    )
    if not 0.0 <= surplus_spend_fraction <= 1.0:
        raise ValueError("surplus spend fraction must be in [0,1]")

    fast_route = None
    if mode == "collect":
        path_raw = os.getenv("REMTP_BLOCK_GATE_DATA")
        if not path_raw:
            raise ValueError("REMTP_BLOCK_GATE_DATA is required in collect mode")
        path = Path(path_raw)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_mode = "a" if os.getenv("REMTP_BLOCK_GATE_APPEND", "0") == "1" else "w"
        _COLLECT_HANDLE = path.open(file_mode, encoding="utf-8", buffering=1)
        atexit.register(_close_collect_handle)
        model_description = f"data={path} action={collect_action}"
    else:
        model_path = os.getenv("REMTP_BLOCK_GATE_MODEL")
        if not model_path:
            raise ValueError("REMTP_BLOCK_GATE_MODEL is required in infer mode")
        model = LogisticBlockGate.load(model_path)
        mean_values = model.mean
        scale_values = model.scale
        weight_values = model.weight
        bias_value = model.bias
        threshold = model.threshold

        def route(
            target_probs: torch.Tensor,
            draft_probs: torch.Tensor,
            draft_token_ids: torch.Tensor,
        ) -> tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]:
            p_y, q_y, log_gap, cactus_h, shield_h = _block_candidates(
                target_probs,
                draft_probs,
                draft_token_ids,
                config,
            )
            features = block_gate_features(
                p_y,
                q_y,
                log_gap,
                cactus_h,
                shield_h,
            )
            mean = torch.tensor(
                mean_values,
                device=features.device,
                dtype=torch.float32,
            )
            scale = torch.tensor(
                scale_values,
                device=features.device,
                dtype=torch.float32,
            )
            weight = torch.tensor(
                weight_values,
                device=features.device,
                dtype=torch.float32,
            )
            bias = torch.tensor(
                bias_value,
                device=features.device,
                dtype=torch.float32,
            )
            probability = logistic_gate_probability(
                features,
                mean,
                scale,
                weight,
                bias,
            )
            shield_route = probability >= threshold
            chosen = torch.where(shield_route, shield_h, cactus_h)
            return p_y, chosen, probability, shield_route

        if os.getenv("REMTP_BLOCK_GATE_COMPILE", "1") == "1":
            route = torch.compile(route, fullgraph=True, dynamic=False)
        fast_route = route
        model_description = (
            f"model={model_path} threshold={model.threshold:.6f}"
        )

    module = importlib.import_module(_REJECTION_MODULE)
    current = module.rejection_sample
    if not getattr(current, "_remtp_learned_block_gate", False):
        wrapper = _learned_block_gate_rejection_sample
        wrapper._remtp_learned_block_gate = True
        wrapper._remtp_original = current
        wrapper._remtp_mode = mode
        wrapper._remtp_config = config
        wrapper._remtp_collect_action = collect_action
        wrapper._remtp_surplus_spend_fraction = surplus_spend_fraction
        wrapper._remtp_fast_route = fast_route
        module.rejection_sample = wrapper
    print(
        "[ReMTP][BlockGate] "
        f"mode={mode} {model_description} features=50 "
        f"surplus_spend_fraction={surplus_spend_fraction:g}",
        flush=True,
    )
