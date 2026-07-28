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
