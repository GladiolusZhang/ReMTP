"""MiMo physical-MTP-layer routing for vLLM 0.18.

Stock vLLM 0.18 accepts MiMo MTP but asserts that every draft forward uses
``spec_step_idx=0``.  Consequently ``num_speculative_tokens>1`` recursively
reuses layer 0 even when a checkpoint contains layers 1 and 2.  This module
adds an explicit, auditable routing policy without changing model weights.

``layer0`` is the stock-compatible control. ``physical`` maps proposal calls
0, 1, 2 to physical MTP layers 0, 1, 2 and cycles only when more than three
tokens are requested.  The physical policy is experimental: Xiaomi publishes
the extra layers as pretrained weights and has not validated them with its
post-trained checkpoints; vLLM itself does not yet implement this route.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import importlib
import os
from typing import Any, Iterator


_PROPOSAL_ACTIVE: ContextVar[bool] = ContextVar("remtp_mimo_proposal_active", default=False)
_FORCED_LAYER: ContextVar[int | None] = ContextVar("remtp_mimo_forced_layer", default=None)
_INSTALLED = False
_PREFILL_DIAGNOSTIC_EMITTED = False


@dataclass
class MTPDepthRouter:
    """Deterministically map MTP calls/depths to physical layer indices."""

    num_layers: int
    mode: str = "layer0"
    call_index: int = 0

    def __post_init__(self) -> None:
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.mode not in {"layer0", "physical"}:
            raise ValueError("MiMo MTP layer mode must be layer0 or physical")

    def reset(self) -> None:
        self.call_index = 0

    def next(self, forced_layer: int | None = None) -> int:
        if forced_layer is not None:
            if forced_layer < 0 or forced_layer >= self.num_layers:
                raise ValueError(
                    f"forced MTP layer {forced_layer} outside [0,{self.num_layers})"
                )
            return forced_layer
        if self.mode == "layer0":
            layer = 0
        else:
            layer = self.call_index % self.num_layers
        self.call_index += 1
        return layer


def _router(model: Any) -> MTPDepthRouter:
    value = getattr(model, "_remtp_mimo_router", None)
    if value is None:
        layers = int(
            getattr(
                model,
                "_remtp_physical_mtp_layers",
                getattr(model.model, "num_mtp_layers", 1),
            )
        )
        mode = os.getenv("REMTP_MIMO_MTP_LAYER_MODE", "layer0").lower()
        if mode == "physical" and layers < 2:
            raise RuntimeError(
                "physical MiMo MTP routing requires a checkpoint with multiple layers"
            )
        if mode == "physical" and os.getenv("REMTP_ALLOW_EXPERIMENTAL_MIMO_MTP3") != "1":
            raise RuntimeError(
                "physical MiMo MTP layers are experimental; set "
                "REMTP_ALLOW_EXPERIMENTAL_MIMO_MTP3=1 after reading docs/mimo_mtp_tree.md"
            )
        value = MTPDepthRouter(layers, mode)
        model._remtp_mimo_router = value
    return value


def _mimo_init(
    self: Any,
    *,
    vllm_config: Any,
    prefix: str = "",
) -> None:
    """Instantiate physical layers separately from vLLM's logical tree width."""
    original = getattr(_mimo_init, "_remtp_original")
    config = vllm_config.model_config.hf_config
    logical_layers = int(getattr(config, "num_nextn_predict_layers", 1))
    physical_layers = int(
        getattr(config, "remtp_physical_mtp_layers", logical_layers)
    )
    if physical_layers <= 0 or physical_layers > logical_layers:
        raise RuntimeError(
            f"invalid MiMo physical/logical MTP layers: {physical_layers}/{logical_layers}"
        )
    config.num_nextn_predict_layers = physical_layers
    try:
        original(self, vllm_config=vllm_config, prefix=prefix)
    finally:
        config.num_nextn_predict_layers = logical_layers
    self._remtp_physical_mtp_layers = physical_layers
    self._remtp_logical_tree_nodes = logical_layers


@contextmanager
def force_mimo_mtp_layer(layer: int) -> Iterator[None]:
    """Force one physical layer while constructing a known tree depth."""
    token = _FORCED_LAYER.set(int(layer))
    try:
        yield
    finally:
        _FORCED_LAYER.reset(token)


def _mimo_forward(
    self: Any,
    input_ids: Any,
    positions: Any,
    hidden_states: Any,
    intermediate_tensors: Any = None,
    inputs_embeds: Any = None,
    spec_step_idx: int = 0,
) -> Any:
    global _PREFILL_DIAGNOSTIC_EMITTED

    # Outside a proposal (startup profiling and memory sizing), retain stock
    # layer-0 behavior. During a proposal, use the explicitly selected route.
    if _PROPOSAL_ACTIVE.get():
        router = _router(self)
        forced_layer = _FORCED_LAYER.get()
        selected = router.next(forced_layer)
    else:
        router = None
        forced_layer = None
        selected = int(spec_step_idx)
    output = self.model(
        input_ids,
        positions,
        hidden_states,
        inputs_embeds,
        selected,
    )
    # vLLM 0.18's integrated Eagle proposer performs only one first-pass MTP
    # forward. That is sufficient for a single MiMo layer, but physical layers
    # 1/2 would otherwise enter recursive drafting without any KV history for
    # the accepted target prefix. Xiaomi's vLLM 0.7.3 fork explicitly runs one
    # speculative-prefill pass per MTP layer. Mirror that invariant here: the
    # first unforced layer-0 call updates every extra physical layer with the
    # same real-prefix token/target-hidden rows; only layer-0's output is used
    # to propose D0. Later calls consume layers 1/2 sequentially as before.
    prefill_all = os.getenv("REMTP_MIMO_PREFILL_ALL_LAYERS", "1") == "1"
    if (
        router is not None
        and router.mode == "physical"
        and forced_layer is None
        and selected == 0
        and prefill_all
    ):
        for layer in range(1, router.num_layers):
            self.model(
                input_ids,
                positions,
                hidden_states,
                inputs_embeds,
                layer,
            )
        if not _PREFILL_DIAGNOSTIC_EMITTED:
            print(
                "[ReMTP][MiMo] prefilling accepted-prefix KV for physical "
                f"MTP layers 0..{router.num_layers - 1}",
                flush=True,
            )
            _PREFILL_DIAGNOSTIC_EMITTED = True
    return output


def _proposal_with_layer_route(self: Any, *args: Any, **kwargs: Any) -> Any:
    original = getattr(_proposal_with_layer_route, "_remtp_original")
    router = _router(self.model)
    router.reset()
    token = _PROPOSAL_ACTIVE.set(True)
    try:
        return original(self, *args, **kwargs)
    finally:
        _PROPOSAL_ACTIVE.reset(token)


def install_mimo_mtp_runtime() -> None:
    """Patch vLLM once and print the effective physical-layer policy."""
    global _INSTALLED
    if _INSTALLED:
        return
    mimo = importlib.import_module("vllm.model_executor.models.mimo_mtp")
    eagle = importlib.import_module("vllm.v1.spec_decode.eagle")

    model_cls = mimo.MiMoMTP
    if not getattr(model_cls.__init__, "_remtp_mimo_physical_init", False):
        _mimo_init._remtp_mimo_physical_init = True
        _mimo_init._remtp_original = model_cls.__init__
        model_cls.__init__ = _mimo_init
    if not getattr(model_cls.forward, "_remtp_mimo_routed", False):
        _mimo_forward._remtp_mimo_routed = True
        _mimo_forward._remtp_original = model_cls.forward
        model_cls.forward = _mimo_forward

    proposer_cls = eagle.EagleProposer
    if not getattr(proposer_cls.propose, "_remtp_mimo_routed", False):
        _proposal_with_layer_route._remtp_mimo_routed = True
        _proposal_with_layer_route._remtp_original = proposer_cls.propose
        proposer_cls.propose = _proposal_with_layer_route

    _INSTALLED = True
    print(
        "[ReMTP][MiMo] physical-layer route installed "
        f"mode={os.getenv('REMTP_MIMO_MTP_LAYER_MODE', 'layer0')}",
        flush=True,
    )
