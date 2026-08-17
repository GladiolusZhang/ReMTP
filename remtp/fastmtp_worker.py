"""FastMTP worker for TencentBAC/FastMTP model.

FastMTP is a Qwen2-based model with a trained MTP layer that uses position-shared
weights across multiple recursive draft steps. It achieves 2.03x speedup with
lossless quality.

Key differences from MiMo:
- Base: Qwen2 (not MiMo architecture)
- MTP: 1 trained layer (not 3 pretrained-only layers)
- Design: Position-shared weights + dual-input fusion
- Performance: 82% better than vanilla MTP

References:
- Model: https://huggingface.co/TencentBAC/FastMTP
- Paper: https://arxiv.org/abs/2509.18362
"""

from __future__ import annotations

from typing import Any

from vllm.v1.worker.gpu_worker import Worker


class FastMTPWorker(Worker):
    """Native FastMTP worker with trained MTP."""

    def init_device(self, *args: Any, **kwargs: Any) -> Any:
        # FastMTP uses custom trust_remote_code for modeling_mimo.py
        # No special runtime adaptation needed - vLLM should load it directly
        return super().init_device(*args, **kwargs)


class FastMTPProbabilisticWorker(Worker):
    """FastMTP with probabilistic MTP sampling."""

    def init_device(self, *args: Any, **kwargs: Any) -> Any:
        from remtp.probabilistic_mtp import install_probabilistic_mtp

        install_probabilistic_mtp()
        return super().init_device(*args, **kwargs)


class FastMTPCactusWorker(Worker):
    """FastMTP with Cactus relaxation."""

    def init_device(self, *args: Any, **kwargs: Any) -> Any:
        from remtp.cactus_mtp import install_cactus_mtp
        from remtp.probabilistic_mtp import install_probabilistic_mtp

        install_probabilistic_mtp()
        install_cactus_mtp()
        return super().init_device(*args, **kwargs)


class FastMTPSpecCascadeWorker(Worker):
    """FastMTP with SpecCascade TokenV3."""

    def init_device(self, *args: Any, **kwargs: Any) -> Any:
        from remtp.probabilistic_mtp import install_probabilistic_mtp
        from remtp.speculative_cascade import install_speculative_cascade

        install_probabilistic_mtp()
        install_speculative_cascade()
        return super().init_device(*args, **kwargs)


class FastMTPProposalCalibratedWorker(Worker):
    """FastMTP with proposal calibration."""

    def init_device(self, *args: Any, **kwargs: Any) -> Any:
        from remtp.probabilistic_mtp import install_probabilistic_mtp
        from remtp.proposal_calibration import install_proposal_calibration

        install_probabilistic_mtp()
        install_proposal_calibration()
        return super().init_device(*args, **kwargs)


class FastMTPDynamicTreeWorker(Worker):
    """FastMTP with dynamic tree attention and target-dominant relaxation."""

    def init_device(self, *args: Any, **kwargs: Any) -> Any:
        from remtp.dynamic_tree_vllm import install_mimo_dynamic_tree
        from remtp.probabilistic_mtp import install_probabilistic_mtp

        install_probabilistic_mtp()
        install_mimo_dynamic_tree()
        return super().init_device(*args, **kwargs)
