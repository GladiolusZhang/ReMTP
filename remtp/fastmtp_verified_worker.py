"""Auditable FastMTP workers used by the verified comparison scripts."""

from __future__ import annotations

import os
from typing import Any

from vllm.v1.worker.gpu_worker import Worker


def _install_fastmtp_runtime() -> None:
    """Make the one physical trained head explicit and recursively reuse it."""
    from remtp.target_numerics import install_target_numerics
    install_target_numerics()
    from remtp.mimo_mtp import install_mimo_mtp_runtime

    install_mimo_mtp_runtime()
    if os.environ.get("REMTP_JETSON_BNB_DRAFT") == "1":
        from remtp.jetson_bf16_draft import install_bf16_mimo_draft

        install_bf16_mimo_draft()
    from remtp.generation_guard import install_head_batching
    install_head_batching()


def _install_optional_profile() -> None:
    from remtp.target_diagnostics import install_target_diagnostics
    install_target_diagnostics()
    from remtp.generation_guard import install_generation_observer
    install_generation_observer()
    if os.environ.get("REMTP_RUNTIME_PROFILE_JSONL"):
        from remtp.runtime_profile import install_runtime_profile

        install_runtime_profile()


class FastMTPVerifiedTargetWorker(Worker):
    def init_device(self, *args: Any, **kwargs: Any) -> Any:
        from remtp.target_numerics import install_target_numerics
        install_target_numerics()
        from remtp.target_diagnostics import install_target_diagnostics
        install_target_diagnostics()
        return super().init_device(*args, **kwargs)


class FastMTPVerifiedNativeWorker(Worker):
    def init_device(self, *args: Any, **kwargs: Any) -> Any:
        from remtp.probabilistic_mtp import install_probabilistic_mtp

        _install_fastmtp_runtime()
        install_probabilistic_mtp()
        _install_optional_profile()
        result = super().init_device(*args, **kwargs)
        print("[ReMTP][FastMTPVerified] method=native worker=active", flush=True)
        return result


class FastMTPVerifiedCactusWorker(Worker):
    def init_device(self, *args: Any, **kwargs: Any) -> Any:
        from remtp.cactus_mtp import install_cactus_mtp
        from remtp.probabilistic_mtp import install_probabilistic_mtp

        _install_fastmtp_runtime()
        install_probabilistic_mtp()
        install_cactus_mtp()
        _install_optional_profile()
        result = super().init_device(*args, **kwargs)
        print("[ReMTP][FastMTPVerified] method=cactus worker=active", flush=True)
        return result


class FastMTPVerifiedSpecCascadeWorker(Worker):
    def init_device(self, *args: Any, **kwargs: Any) -> Any:
        from remtp.probabilistic_mtp import install_probabilistic_mtp
        from remtp.speculative_cascade import install_speculative_cascade

        _install_fastmtp_runtime()
        install_probabilistic_mtp()
        install_speculative_cascade()
        _install_optional_profile()
        result = super().init_device(*args, **kwargs)
        print("[ReMTP][FastMTPVerified] method=spec_cascade worker=active", flush=True)
        return result


class FastMTPVerifiedDynamicTreeWorker(Worker):
    def init_device(self, *args: Any, **kwargs: Any) -> Any:
        from remtp.dynamic_tree_vllm import install_mimo_dynamic_tree

        _install_fastmtp_runtime()
        install_mimo_dynamic_tree()
        _install_optional_profile()
        result = super().init_device(*args, **kwargs)
        print(
            "[ReMTP][FastMTPVerified] method=dynamic_tree worker=active "
            "physical_mtp_route=repeat_0",
            flush=True,
        )
        return result
