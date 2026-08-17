"""Protocol-checked RL-0530 quality and MTP-route comparison report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


METHODS = (
    ("target_only", "Target only (quality baseline)"),
    ("native_012", "Strict MTP, physical 0-1-2"),
    ("native_000", "Strict MTP, repeated 0-0-0"),
    ("tree_012", "Relaxed dynamic tree, physical 0-1-2 + EOS"),
    ("tree_000", "Relaxed dynamic tree, repeated 0-0-0 + EOS"),
)

PROTOCOL_KEYS = (
    "samples",
    "sample_seed",
    "temperature",
    "generation_seed",
    "max_tokens",
    "data_sha256",
    "model",
    "system_prompt",
)


def _read(directory: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    results = payload.get("results")
    if not isinstance(results, list) or len(results) != 1:
        raise ValueError(f"expected one result in {directory / 'summary.json'}")
    return dict(payload["config"]), dict(results[0])


def _manifest(directory: Path) -> str:
    return hashlib.sha256(
        (directory / "sample_manifest.json").read_bytes()
    ).hexdigest()


def _load_dataset(root: Path, dataset: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    reference_config: dict[str, Any] | None = None
    reference_manifest: str | None = None
    reference_quality: float | None = None
    for method, label in METHODS:
        directory = root / dataset / method
        config, result = _read(directory)
        manifest = _manifest(directory)
        if dataset == "humaneval":
            if result.get("evaluation_status") != "complete":
                raise ValueError(f"HumanEval evaluation is incomplete: {directory}")
            quality = result.get("pass_at_1")
        else:
            quality = result.get("accuracy")
        if quality is None:
            raise ValueError(f"quality is missing: {directory}")

        if reference_config is None:
            reference_config = config
            reference_manifest = manifest
            reference_quality = float(quality)
        else:
            assert reference_config is not None
            assert reference_manifest is not None
            if manifest != reference_manifest:
                raise ValueError(f"sample manifest differs for {dataset}/{method}")
            mismatch = [
                key
                for key in PROTOCOL_KEYS
                if config.get(key) != reference_config.get(key)
            ]
            if mismatch:
                raise ValueError(
                    f"protocol differs for {dataset}/{method}: "
                    + ", ".join(mismatch)
                )

        mal = result.get("mean_acceptance_length")
        accepted = result.get("accepted_draft_tokens_per_round")
        nodes: float | None = None
        if method.startswith("tree_"):
            tree_payload = json.loads(
                (directory / "tree_metrics.json").read_text(encoding="utf-8")
            )
            tree = tree_payload["tree"]
            mal = tree["mean_acceptance_length"]
            accepted = tree["mean_accepted_depth"]
            nodes = tree["target_nodes_per_round"]
        elif method != "target_only":
            draft_rounds = float(result.get("draft_rounds") or 0)
            if draft_rounds > 0:
                nodes = float(result["draft_tokens"]) / draft_rounds

        assert reference_quality is not None
        rows.append(
            {
                "method_id": method,
                "method": label,
                "samples": int(result["samples"]),
                "quality": float(quality),
                "quality_delta_pp": 100.0 * (float(quality) - reference_quality),
                "truncation_rate": result.get("truncation_rate"),
                "decode_tok_s": result.get("decode_tok_s"),
                "e2e_tok_s": result.get("e2e_output_tok_s"),
                "mal": None if mal is None else float(mal),
                "accepted_per_round": None if accepted is None else float(accepted),
                "candidate_nodes": nodes,
            }
        )
    return rows


def _number(value: Any, digits: int = 3) -> str:
    return "-" if value is None else f"{float(value):.{digits}f}"


def _percent(value: Any) -> str:
    return "-" if value is None else f"{100.0 * float(value):.1f}%"


def _table(rows: list[dict[str, Any]], quality_name: str) -> list[str]:
    lines = [
        f"| method | {quality_name} | vs target | truncation | decode tok/s | "
        "e2e tok/s | MAL | accepted drafts/round | candidate nodes/round |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {_percent(row['quality'])} | "
            f"{row['quality_delta_pp']:+.1f} pp | "
            f"{_percent(row['truncation_rate'])} | "
            f"{_number(row['decode_tok_s'])} | {_number(row['e2e_tok_s'])} | "
            f"{_number(row['mal'])} | {_number(row['accepted_per_round'])} | "
            f"{_number(row['candidate_nodes'])} |"
        )
    return lines


def build_report(root: Path) -> tuple[str, dict[str, Any]]:
    gsm8k = _load_dataset(root, "gsm8k")
    humaneval = _load_dataset(root, "humaneval")
    lines = [
        "# MiMo-7B-RL-0530: physical 0-1-2 vs repeated 0-0-0",
        "",
        "All rows use an explicit empty system message, temperature 0.6, the "
        "same sampled tasks and generation seeds. Tree rows enable target-side "
        "EOS protection.",
        "",
        f"## GSM8K ({gsm8k[0]['samples']} tasks)",
        "",
    ]
    lines.extend(_table(gsm8k, "accuracy"))
    lines.extend(["", f"## HumanEval ({humaneval[0]['samples']} tasks)", ""])
    lines.extend(_table(humaneval, "pass@1"))
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `Target only` establishes checkpoint quality without speculative decoding.",
            "- `physical 0-1-2` uses the posttrained checkpoint's built-in MTP layer 0, "
            "then the separately published pretrained layers 1 and 2.",
            "- `repeated 0-0-0` reuses the posttrained checkpoint's layer 0 at all three "
            "draft depths; it is an ablation, not a claim that the repeated states are calibrated.",
            "- Tree MAL is recomputed from the selected connected path in the tree audit.",
            "- Quality should first be compared with `Target only`; 012/000 acceptance "
            "comparisons are meaningful only after that baseline is credible.",
            "",
        ]
    )
    return "\n".join(lines), {"gsm8k": gsm8k, "humaneval": humaneval}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root.resolve()
    markdown, payload = build_report(root)
    (root / "comparison.md").write_text(markdown, encoding="utf-8")
    (root / "comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print((root / "comparison.md").resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
