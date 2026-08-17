"""Protocol-checked report for the audited FastMTP comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


METHODS = (
    ("target", "Target only"),
    ("native", "Native FastMTP"),
    ("cactus", "Cactus + FastMTP"),
    ("spec_cascade", "SpecCascade TokenV3 + FastMTP"),
    ("dynamic_tree", "Dynamic tree + target relaxation"),
)

REQUIRED_METHOD_IDS = {"native", "cactus", "spec_cascade", "dynamic_tree"}

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


def _load_summary(directory: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    rows = payload.get("results")
    if not isinstance(rows, list) or len(rows) != 1:
        raise ValueError(f"expected one result row: {directory / 'summary.json'}")
    return dict(payload["config"]), dict(rows[0])


def _manifest_hash(directory: Path) -> str:
    return hashlib.sha256((directory / "sample_manifest.json").read_bytes()).hexdigest()


def _runtime(directory: Path, expected: str) -> dict[str, Any]:
    path = directory / "method_runtime.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        value.get("method") != expected
        or not value.get("marker_verified")
        or not value.get("adapter_verified")
    ):
        raise ValueError(f"unverified runtime marker: {path}")
    return value


def _load_dataset(root: Path, dataset: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    reference_config: dict[str, Any] | None = None
    reference_manifest: str | None = None
    native_quality: float | None = None
    native_e2e: float | None = None
    available_methods = [
        (method, label)
        for method, label in METHODS
        if (root / dataset / method / "summary.json").is_file()
    ]
    available_ids = {method for method, _ in available_methods}
    missing = REQUIRED_METHOD_IDS - available_ids
    if missing:
        raise ValueError(
            f"required methods missing for {dataset}: {', '.join(sorted(missing))}"
        )
    for method, label in available_methods:
        directory = root / dataset / method
        config, result = _load_summary(directory)
        _runtime(directory, method)
        manifest = _manifest_hash(directory)
        quality = result.get("accuracy" if dataset == "gsm8k" else "pass_at_1")
        if quality is None:
            raise ValueError(f"quality is missing: {directory}")
        if dataset == "humaneval" and result.get("evaluation_status") != "complete":
            raise ValueError(f"HumanEval evaluation incomplete: {directory}")

        if reference_config is None:
            reference_config = config
            reference_manifest = manifest
        else:
            assert reference_config is not None and reference_manifest is not None
            if manifest != reference_manifest:
                raise ValueError(f"sample manifest differs for {dataset}/{method}")
            mismatch = [
                key for key in PROTOCOL_KEYS
                if config.get(key) != reference_config.get(key)
            ]
            if mismatch:
                raise ValueError(
                    f"protocol differs for {dataset}/{method}: {', '.join(mismatch)}"
                )

        mal = result.get("mean_acceptance_length")
        accepted = result.get("draft_token_acceptance_rate")
        nodes: float | None = None
        rescue_rate: float | None = None
        ordinary_depth: float | None = None
        extension_mal_gain: float | None = None
        verification_mode: str | None = None
        if method == "dynamic_tree":
            tree_payload = json.loads(
                (directory / "tree_metrics.json").read_text(encoding="utf-8")
            )
            tree = tree_payload["tree"]
            verification_mode = tree.get("verification_mode")
            mal = tree.get("mean_acceptance_length")
            nodes = tree.get("target_nodes_per_round")
            rescue_rate = tree.get("selected_rescue_round_rate")
            ordinary_depth = tree.get(
                "pre_rescue_mean_accepted_depth",
                tree.get("ordinary_mean_accepted_depth"),
            )
            extension_mal_gain = tree.get(
                "rescue_unlocked_mal_gain",
                tree.get("rescue_extension_mal_gain"),
            )
            node_total = float(tree.get("target_nodes_per_round") or 0.0)
            accepted = (
                float(tree.get("mean_accepted_depth") or 0.0) / node_total
                if node_total > 0
                else None
            )
        elif method != "target":
            rounds = float(result.get("draft_rounds") or 0.0)
            if rounds > 0:
                nodes = float(result.get("draft_tokens") or 0.0) / rounds

        e2e = float(result["e2e_output_tok_s"])
        if method == "native":
            native_quality = float(quality)
            native_e2e = e2e
        rows.append(
            {
                "method_id": method,
                "method": label,
                "samples": int(result["samples"]),
                "quality": float(quality),
                "decode_tok_s": float(result["decode_tok_s"]),
                "e2e_tok_s": e2e,
                "truncation_rate": result.get("truncation_rate"),
                "mal": None if mal is None else float(mal),
                "draft_acceptance": None if accepted is None else float(accepted),
                "nodes": None if nodes is None else float(nodes),
                "rescue_rate": None if rescue_rate is None else float(rescue_rate),
                "ordinary_depth": (
                    None if ordinary_depth is None else float(ordinary_depth)
                ),
                "extension_mal_gain": (
                    None
                    if extension_mal_gain is None
                    else float(extension_mal_gain)
                ),
                "verification_mode": verification_mode,
            }
        )

    assert native_e2e is not None and native_quality is not None
    for row in rows:
        row["speedup_vs_native"] = row["e2e_tok_s"] / native_e2e
        row["quality_delta_vs_native_pp"] = 100.0 * (
            row["quality"] - native_quality
        )
    return rows


def _number(value: Any, digits: int = 3) -> str:
    return "-" if value is None else f"{float(value):.{digits}f}"


def _percent(value: Any) -> str:
    return "-" if value is None else f"{100.0 * float(value):.1f}%"


def _table(rows: list[dict[str, Any]], quality_name: str) -> list[str]:
    lines = [
        f"| method | {quality_name} | vs Native | decode tok/s | e2e tok/s | "
        "speed vs Native | MAL | ordinary depth | rescue MAL | draft acceptance | "
        "nodes/round | rescue rounds | truncation |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {_percent(row['quality'])} | "
            f"{row['quality_delta_vs_native_pp']:+.1f} pp | "
            f"{_number(row['decode_tok_s'])} | {_number(row['e2e_tok_s'])} | "
            f"{row['speedup_vs_native']:.3f}x | {_number(row['mal'])} | "
            f"{_number(row['ordinary_depth'])} | "
            f"{_number(row['extension_mal_gain'])} | "
            f"{_percent(row['draft_acceptance'])} | {_number(row['nodes'])} | "
            f"{_percent(row['rescue_rate'])} | "
            f"{_percent(row['truncation_rate'])} |"
        )
    return lines


def build_report(root: Path) -> tuple[str, dict[str, Any]]:
    gsm8k = _load_dataset(root, "gsm8k")
    humaneval = _load_dataset(root, "humaneval")
    lines = [
        "# Verified FastMTP comparison",
        "",
        "All methods use the same TencentBAC/FastMTP checkpoint, sampled tasks, "
        "temperature, generation seed, explicit empty system message, eager mode "
        "and single-request protocol. Runtime worker markers are checked before "
        "results enter this report.",
        "",
        f"## GSM8K ({gsm8k[0]['samples']} tasks)",
        "",
    ]
    lines.extend(_table(gsm8k, "accuracy"))
    lines.extend(["", f"## HumanEval ({humaneval[0]['samples']} tasks)", ""])
    lines.extend(_table(humaneval, "pass@1"))
    tree_modes = {
        row.get("verification_mode")
        for row in (*gsm8k, *humaneval)
        if row["method_id"] == "dynamic_tree"
    }
    if tree_modes == {"residual_hit_anchor"}:
        tree_semantics = (
            "- This run uses exact residual-hit anchoring: Cactus samples the "
            "correction before tree lookup; a branch is reused only for the same "
            "token ID, then one unmodified target token is appended. The tree adds "
            "no token-choice approximation beyond the Cactus primary verifier."
        )
    elif tree_modes <= {
        "sampled_primary_shadow",
        "residual_hit_strict",
    }:
        tree_semantics = (
            "- This run uses an exact-residual route; consult the recorded "
            "`verification_mode` for whether the tree is shadow-only or continues "
            "with strict `p/q` verification."
        )
    else:
        tree_semantics = (
            "- Dynamic tree is explicitly approximate: its path selector changes "
            "the output distribution and must be judged by quality as well as MAL."
        )
    lines.extend(
        [
            "",
            "## Method semantics",
            "",
            "- Native, Cactus and SpecCascade recursively reuse FastMTP's one trained "
            "physical MTP layer for three proposal positions.",
            "- Cactus and SpecCascade use the exact full proposal distribution captured "
            "from the same sampled candidates used by verification.",
            "- Dynamic tree uses one trained physical layer recursively, a logical tree "
            "node view, `TREE_ATTN`, and one target tree forward per round.",
            tree_semantics,
            "- The published FastMTP speed includes SGLang kernels and vocabulary "
            "compression not present in this vLLM adapter; this report therefore "
            "shows local speed relative to Native FastMTP.",
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
