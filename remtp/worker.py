"""vLLM worker that installs the ReMTP observer inside EngineCore.

vLLM deliberately sanitizes parts of the API server environment when it
spawns EngineCore.  Loading the observer from the worker lifecycle guarantees
that the rejection sampler is patched in the process that actually owns the
GPU and performs speculative verification.
"""

from __future__ import annotations

import os
from typing import Any

from vllm.v1.worker.gpu_worker import Worker

from remtp.trace import install_import_hook


class ReMTPWorker(Worker):
    """Standard CUDA worker with a read-only MTP trace observer."""

    def init_device(self, *args: Any, **kwargs: Any) -> Any:
        os.environ["REMTP_MODEL"] = self.model_config.tokenizer
        install_import_hook()
        return super().init_device(*args, **kwargs)


class ProbabilisticMTPWorker(Worker):
    """CUDA worker that samples MTP drafts from and exposes the full q."""

    def init_device(self, *args: Any, **kwargs: Any) -> Any:
        from remtp.probabilistic_mtp import install_probabilistic_mtp

        install_probabilistic_mtp()
        return super().init_device(*args, **kwargs)


class SpecCascadeMTPWorker(Worker):
    """CUDA worker that installs speculative-cascade MTP verification."""

    def init_device(self, *args: Any, **kwargs: Any) -> Any:
        from remtp.probabilistic_mtp import install_probabilistic_mtp
        from remtp.speculative_cascade import install_speculative_cascade

        install_probabilistic_mtp()
        install_speculative_cascade()
        return super().init_device(*args, **kwargs)


class CactusMTPWorker(Worker):
    """CUDA worker that installs Cactus over probabilistic MTP."""

    def init_device(self, *args: Any, **kwargs: Any) -> Any:
        from remtp.cactus_mtp import install_cactus_mtp
        from remtp.probabilistic_mtp import install_probabilistic_mtp

        install_probabilistic_mtp()
        install_cactus_mtp()
        return super().init_device(*args, **kwargs)


class TargetAnchoredMTPWorker(Worker):
    """CUDA worker for target-anchored exact-TV MTP verification."""

    def init_device(self, *args: Any, **kwargs: Any) -> Any:
        from remtp.probabilistic_mtp import (
            install_aligned_hidden_capture,
            install_probabilistic_mtp,
        )
        from remtp.target_anchored_mtp import install_target_anchored_mtp

        install_probabilistic_mtp()
        install_aligned_hidden_capture()
        install_target_anchored_mtp()
        return super().init_device(*args, **kwargs)


class RegretFeedbackMTPWorker(Worker):
    """CUDA worker for target-anchored MTP plus cross-block regret."""

    def init_device(self, *args: Any, **kwargs: Any) -> Any:
        from remtp.probabilistic_mtp import (
            install_aligned_hidden_capture,
            install_probabilistic_mtp,
        )
        from remtp.regret_feedback import install_regret_feedback
        from remtp.target_anchored_mtp import install_target_anchored_mtp

        install_probabilistic_mtp()
        install_aligned_hidden_capture()
        install_target_anchored_mtp()
        install_regret_feedback()
        return super().init_device(*args, **kwargs)
