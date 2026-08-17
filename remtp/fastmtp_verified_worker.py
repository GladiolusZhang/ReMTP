"""Auditable FastMTP workers used by the verified comparison scripts."""

from __future__ import annotations

from typing import Any

from vllm.v1.worker.gpu_worker import Worker


def _install_fastmtp_runtime() -> None:
    """Make the one physical trained head explicit and recursively reuse it."""
    from remtp.mimo_mtp import install_mimo_mtp_runtime

    install_mimo_mtp_runtime()


class FastMTPVerifiedNativeWorker(Worker):
    def init_device(self, *args: Any, **kwargs: Any) -> Any:
        from remtp.probabilistic_mtp import install_probabilistic_mtp

        _install_fastmtp_runtime()
        install_probabilistic_mtp()
        print("[ReMTP][FastMTPVerified] method=native worker=active", flush=True)
        return super().init_device(*args, **kwargs)


class FastMTPVerifiedCactusWorker(Worker):
    def init_device(self, *args: Any, **kwargs: Any) -> Any:
        from remtp.cactus_mtp import install_cactus_mtp
        from remtp.probabilistic_mtp import install_probabilistic_mtp

        _install_fastmtp_runtime()
        install_probabilistic_mtp()
        install_cactus_mtp()
        print("[ReMTP][FastMTPVerified] method=cactus worker=active", flush=True)
        return super().init_device(*args, **kwargs)


class FastMTPVerifiedSpecCascadeWorker(Worker):
    def init_device(self, *args: Any, **kwargs: Any) -> Any:
        from remtp.probabilistic_mtp import install_probabilistic_mtp
        from remtp.speculative_cascade import install_speculative_cascade

        _install_fastmtp_runtime()
        install_probabilistic_mtp()
        install_speculative_cascade()
        print("[ReMTP][FastMTPVerified] method=spec_cascade worker=active", flush=True)
        return super().init_device(*args, **kwargs)


class FastMTPVerifiedDynamicTreeWorker(Worker):
    def init_device(self, *args: Any, **kwargs: Any) -> Any:
        from remtp.dynamic_tree_vllm import install_mimo_dynamic_tree

        _install_fastmtp_runtime()
        install_mimo_dynamic_tree()
        print(
            "[ReMTP][FastMTPVerified] method=dynamic_tree worker=active "
            "physical_mtp_route=repeat_0",
            flush=True,
        )
        return super().init_device(*args, **kwargs)
