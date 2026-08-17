"""FastMTP dynamic tree worker adapter.

Adapts dynamic tree attention to work with FastMTP (MiMo-based).
"""

from __future__ import annotations

from typing import Any

from vllm.v1.worker.gpu_worker import Worker


class FastMTPDynamicTreeWorker(Worker):
    """FastMTP with dynamic tree attention and target-dominant relaxation."""

    def init_device(self, *args: Any, **kwargs: Any) -> Any:
        # Import dynamic tree components
        from remtp.dynamic_mtp_tree import install_dynamic_mtp_tree
        from remtp.dynamic_tree_vllm import install_dynamic_tree_vllm
        from remtp.probabilistic_mtp import install_probabilistic_mtp

        # Install in order
        install_probabilistic_mtp()
        install_dynamic_mtp_tree()
        install_dynamic_tree_vllm()

        return super().init_device(*args, **kwargs)
