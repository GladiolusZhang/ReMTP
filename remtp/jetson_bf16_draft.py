"""Keep FastMTP's small physical MTP layer in BF16 with a 4-bit target.

vLLM 0.18 passes the target's quant_config into the MiMo draft constructor
even when speculative_config selects a separate BF16 draft load format. The
temporary override is limited to construction of the draft layer.
"""

from __future__ import annotations

from typing import Any


def install_bf16_mimo_draft() -> None:
    from vllm.model_executor.models.mimo_mtp import MiMoMTP

    original = MiMoMTP.__init__
    if getattr(original, "_remtp_jetson_bf16_draft", False):
        return

    def init(self: Any, *, vllm_config: Any, prefix: str = "") -> None:
        quant_config = vllm_config.quant_config
        vllm_config.quant_config = None
        try:
            original(self, vllm_config=vllm_config, prefix=prefix)
        finally:
            vllm_config.quant_config = quant_config

    init._remtp_jetson_bf16_draft = True  # type: ignore[attr-defined]
    MiMoMTP.__init__ = init
    print("[ReMTP][Jetson] MiMo draft BF16 constructor installed", flush=True)
