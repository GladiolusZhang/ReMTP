"""Write tree-native acceptance metrics beside a benchmark run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from remtp.mimo_tree_report import load_records, markdown, summarize


def write_metrics(run_dir: Path, audit: Path) -> dict[str, object]:
    tree = summarize(load_records(audit))
    payload: dict[str, object] = {
        "source_audit": str(audit.resolve()),
        "tree": tree,
    }
    output = run_dir / "tree_metrics.json"
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (run_dir / "tree_metrics.md").write_text(markdown(tree), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    write_metrics(args.run_dir.resolve(), args.audit.resolve())
    print((args.run_dir / "tree_metrics.json").resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
