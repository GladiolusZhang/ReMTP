"""MiMo-specific vLLM workers for the repository's chain baselines.

Every worker installs the physical-layer router before the architecture-
independent verifier.  Keeping these classes separate from ``remtp.worker``
also prevents the MiMo experiment from changing frozen Qwen3.5 runs.
"""

from __future__ import annotations

from typing import Any, Callable

from vllm.v1.worker.gpu_worker import Worker


def _mimo_first(installer: Callable[[], None]) -> None:
    from remtp.mimo_mtp import install_mimo_mtp_runtime

    install_mimo_mtp_runtime()
    installer()


class MiMoProbabilisticMTPWorker(Worker):
    def init_device(self, *args: Any, **kwargs: Any) -> Any:
        from remtp.probabilistic_mtp import install_probabilistic_mtp

        _mimo_first(install_probabilistic_mtp)
        return super().init_device(*args, **kwargs)


class MiMoCactusMTPWorker(Worker):
    def init_device(self, *args: Any, **kwargs: Any) -> Any:
        from remtp.cactus_mtp import install_cactus_mtp
        from remtp.probabilistic_mtp import install_probabilistic_mtp

        from remtp.mimo_mtp import install_mimo_mtp_runtime

        install_mimo_mtp_runtime()
        install_probabilistic_mtp()
        install_cactus_mtp()
        return super().init_device(*args, **kwargs)


class MiMoSpecCascadeMTPWorker(Worker):
    def init_device(self, *args: Any, **kwargs: Any) -> Any:
        from remtp.probabilistic_mtp import install_probabilistic_mtp
        from remtp.speculative_cascade import install_speculative_cascade

        from remtp.mimo_mtp import install_mimo_mtp_runtime

        install_mimo_mtp_runtime()
        install_probabilistic_mtp()
        install_speculative_cascade()
        return super().init_device(*args, **kwargs)


class MiMoBlockVerificationMTPWorker(Worker):
    def init_device(self, *args: Any, **kwargs: Any) -> Any:
        from remtp.block_verification import install_block_verification
        from remtp.probabilistic_mtp import install_probabilistic_mtp

        from remtp.mimo_mtp import install_mimo_mtp_runtime

        install_mimo_mtp_runtime()
        install_probabilistic_mtp()
        install_block_verification()
        return super().init_device(*args, **kwargs)


class MiMoCactusBlockVerificationMTPWorker(Worker):
    def init_device(self, *args: Any, **kwargs: Any) -> Any:
        from remtp.block_verification import install_block_verification
        from remtp.cactus_mtp import install_cactus_mtp
        from remtp.probabilistic_mtp import install_probabilistic_mtp

        from remtp.mimo_mtp import install_mimo_mtp_runtime

        install_mimo_mtp_runtime()
        install_probabilistic_mtp()
        install_block_verification()
        install_cactus_mtp()
        return super().init_device(*args, **kwargs)


class MiMoProposalCalibratedMTPWorker(Worker):
    def init_device(self, *args: Any, **kwargs: Any) -> Any:
        from remtp.probabilistic_mtp import install_probabilistic_mtp
        from remtp.proposal_calibration import install_proposal_calibration

        from remtp.mimo_mtp import install_mimo_mtp_runtime

        install_mimo_mtp_runtime()
        install_probabilistic_mtp()
        install_proposal_calibration()
        return super().init_device(*args, **kwargs)


class MiMoReMTPWorker(Worker):
    """Frozen chain ReMTP (schemes 1+2) on the MiMo proposal path."""

    def init_device(self, *args: Any, **kwargs: Any) -> Any:
        from remtp.probabilistic_mtp import install_probabilistic_mtp
        from remtp.risk_entropy_mtp import install_risk_entropy_mtp

        from remtp.mimo_mtp import install_mimo_mtp_runtime

        install_mimo_mtp_runtime()
        install_probabilistic_mtp()
        install_risk_entropy_mtp()
        return super().init_device(*args, **kwargs)


class MiMoTreeMTPWorker(Worker):
    def init_device(self, *args: Any, **kwargs: Any) -> Any:
        from remtp.mimo_mtp import install_mimo_mtp_runtime
        from remtp.mimo_tree_vllm import install_mimo_microtree

        install_mimo_mtp_runtime()
        install_mimo_microtree()
        return super().init_device(*args, **kwargs)


class MiMoDynamicTreeMTPWorker(Worker):
    """Dynamic MTP-only proposal tree + target-dominant relaxed verifier."""

    def init_device(self, *args: Any, **kwargs: Any) -> Any:
        from remtp.dynamic_tree_vllm import install_mimo_dynamic_tree
        from remtp.mimo_mtp import install_mimo_mtp_runtime

        install_mimo_mtp_runtime()
        install_mimo_dynamic_tree()
        return super().init_device(*args, **kwargs)
