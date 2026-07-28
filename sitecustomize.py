"""Enable the ReMTP observer in every vLLM subprocess.

Python imports ``sitecustomize`` automatically when this repository is on
``PYTHONPATH``.  The shell launcher sets that path and ``REMTP_TRACE=1``.
"""

from __future__ import annotations

import os
import sys


if os.getenv("REMTP_TRACE") == "1":
    try:
        from remtp.trace import install_import_hook

        install_import_hook()
    except Exception as exc:  # The observer must never stop vLLM from starting.
        print(f"[ReMTP] failed to install trace hook: {exc}", file=sys.stderr)
