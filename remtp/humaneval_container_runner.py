"""Minimal in-container HumanEval executor.

The host evaluator launches this file in a restricted, disposable Docker
container.  Do not invoke it on untrusted code outside that container.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import traceback
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print(json.dumps({"status": "runner_error", "detail": "missing job"}))
        return 2
    job = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    namespace: dict[str, object] = {}
    captured_stdout = io.StringIO()
    captured_stderr = io.StringIO()
    try:
        code = compile(job["source"], "<humaneval-candidate>", "exec")
        with contextlib.redirect_stdout(captured_stdout), contextlib.redirect_stderr(
            captured_stderr
        ):
            exec(code, namespace, namespace)
    except AssertionError as exc:
        result = {"status": "failed", "detail": str(exc)[:1000]}
    except SyntaxError as exc:
        result = {
            "status": "syntax_error",
            "detail": f"{exc.msg} (line {exc.lineno})"[:1000],
        }
    except BaseException as exc:
        result = {
            "status": "runtime_error",
            "detail": "".join(
                traceback.format_exception_only(type(exc), exc)
            ).strip()[:1000],
        }
    else:
        result = {"status": "passed", "detail": ""}
    sys.__stdout__.write(json.dumps(result, ensure_ascii=False) + "\n")
    sys.__stdout__.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
