from __future__ import annotations

import pytest
from types import SimpleNamespace

from remtp.mimo_mtp import (
    MTPDepthRouter,
    _PROPOSAL_ACTIVE,
    _mimo_forward,
    _mimo_init,
)


def test_stock_control_always_uses_layer_zero() -> None:
    router = MTPDepthRouter(3, "layer0")
    assert [router.next() for _ in range(6)] == [0, 0, 0, 0, 0, 0]


def test_physical_route_cycles_all_layers() -> None:
    router = MTPDepthRouter(3, "physical")
    assert [router.next() for _ in range(7)] == [0, 1, 2, 0, 1, 2, 0]
    router.reset()
    assert router.next() == 0


def test_tree_depth_can_force_a_physical_layer() -> None:
    router = MTPDepthRouter(3, "physical")
    assert router.next(forced_layer=2) == 2
    assert router.call_index == 0
    with pytest.raises(ValueError, match="outside"):
        router.next(forced_layer=3)


def test_logical_tree_width_instantiates_only_physical_layers() -> None:
    config = SimpleNamespace(
        num_nextn_predict_layers=14,
        remtp_physical_mtp_layers=3,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=config)
    )
    instance = SimpleNamespace()

    def original(self, *, vllm_config, prefix=""):
        self.model = SimpleNamespace(
            num_mtp_layers=vllm_config.model_config.hf_config.num_nextn_predict_layers
        )

    previous = getattr(_mimo_init, "_remtp_original", None)
    _mimo_init._remtp_original = original
    try:
        _mimo_init(instance, vllm_config=vllm_config)
    finally:
        if previous is None:
            delattr(_mimo_init, "_remtp_original")
        else:
            _mimo_init._remtp_original = previous

    assert instance.model.num_mtp_layers == 3
    assert instance._remtp_physical_mtp_layers == 3
    assert instance._remtp_logical_tree_nodes == 14
    assert config.num_nextn_predict_layers == 14


def test_physical_first_pass_prefills_every_layer(monkeypatch) -> None:
    monkeypatch.setenv("REMTP_MIMO_MTP_LAYER_MODE", "physical")
    monkeypatch.setenv("REMTP_ALLOW_EXPERIMENTAL_MIMO_MTP3", "1")
    monkeypatch.setenv("REMTP_MIMO_PREFILL_ALL_LAYERS", "1")
    calls: list[int] = []

    class Predictor:
        num_mtp_layers = 3

        def __call__(
            self,
            input_ids,
            positions,
            hidden_states,
            inputs_embeds,
            layer,
        ):
            calls.append(layer)
            return f"layer-{layer}"

    instance = SimpleNamespace(
        model=Predictor(),
        _remtp_physical_mtp_layers=3,
    )
    proposal = _PROPOSAL_ACTIVE.set(True)
    try:
        first = _mimo_forward(instance, None, None, None)
        second = _mimo_forward(instance, None, None, None)
    finally:
        _PROPOSAL_ACTIVE.reset(proposal)

    assert first == "layer-0"
    assert second == "layer-1"
    assert calls == [0, 1, 2, 1]
