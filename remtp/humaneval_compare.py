"""Build a protocol-checked HumanEval speed/quality comparison table."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


PROTOCOL_KEYS = (
    "samples",
    "sample_seed",
    "temperature",
    "generation_seed",
    "max_tokens",
    "mtp_tokens",
    "data_sha256",
    "answer_metric",
    "evaluation_image",
    "evaluation_image_id",
    "evaluation_timeout_seconds",
)

PROFILE_LABELS = {
    "native_mtp": "Native probabilistic MTP",
    "cactus": "Cactus + MTP",
    "cactus_regret": "Cactus + residual regret feedback",
    "spec_cascade": "SpecCascade TokenV3 + MTP",
    "native_block": "Native MTP + Block Verification",
    "cactus_block": "Cactus + Block Verification",
    "debt_balanced": "Current-block debt (balanced)",
    "target_surplus": "Cactus-dominant target surplus",
    "tv_head": "Exact-TV + head calibration",
    "tv_hidden_veto": "Exact-TV + head/hidden + future veto",
    "exact_tv": "Exact-TV + target-only future veto",
    "exact_tv_regret_fixed": "Exact-TV + fixed regret feedback",
    "exact_tv_regret_router": "Exact-TV + learned expected-regret Router",
    "regret_calibrated_block": "Regret-Calibrated Block Relaxation",
    "target_mode_regret": "Target-Mode Rescue + within-block regret",
    "target_mode_identity": "Fused strict-MTP identity control",
    "target_mode_p05_b060": "p>=0.5 baseline (block=0.60, debt=0.50)",
    "target_mode_p05_b075": "p>=0.5 mild (block=0.75, debt=0.35)",
    "target_mode_p05_b090": "p>=0.5 aggressive (block=0.90, debt=0.20)",
    "target_mode_p05_b095": "p>=0.5 max rescue (block=0.95, debt=0.00)",
    "target_mode_top1_m000": "Target top-1 (margin>=0.00, block=0.75)",
    "target_mode_top1_m010": "Target top-1 (margin>=0.10, block=0.75)",
    "target_mode_top1_m025": "Target top-1 (margin>=0.25, block=0.75)",
    "target_mode_top1_m050": "Target top-1 (margin>=0.50, block=0.75)",
    "target_band_r2_g025": "Target band (rank<=2, gap<=0.25)",
    "target_band_r4_g050": "Target band (rank<=4, gap<=0.50)",
    "target_band_r4_g075": "Target band (rank<=4, gap<=0.75)",
    "target_band_r8_g100": "Target band (rank<=8, gap<=1.00)",
    "prefix_credit_token_cap": "Target top-1 + joint verify (token cap)",
    "prefix_credit_atomic": "Prefix-Credit MTP (atomic repair)",
    "prefix_credit": "Prefix-Credit MTP (gain/TV>=0.50)",
    "prefix_credit_g025": "Prefix-Credit MTP (gain/TV>=0.25)",
    "prefix_credit_g050": "Prefix-Credit MTP (gain/TV>=0.50)",
    "prefix_credit_g100": "Prefix-Credit MTP (gain/TV>=1.00)",
    "prefix_credit_b060": "Prefix-Credit MTP (block=0.60)",
    "prefix_credit_b075": "Prefix-Credit MTP (block=0.75)",
    "prefix_credit_b090": "Prefix-Credit MTP (block=0.90)",
    "scheme1": "Scheme 1: cumulative marginal-entropy relaxation",
    "scheme2": "Scheme 2: sentinel + target anchor + risk debt",
    "scheme2_relaxed": "Scheme 2: stronger soft-sentinel relaxation",
    "scheme2_strong": "Scheme 2: high-relaxation soft sentinel",
    "scheme2_ultra": "Scheme 2: ultra-relaxation soft sentinel",
    "scheme3": "Scheme 3: entropy-aware adaptive chain fallback",
    "scheme12": "Scheme 1 + Scheme 2",
    "scheme12_joint": "Joint Scheme 1+2 risk-credit allocation",
    "scheme12_anchored": "Target-anchored dual-pool Scheme 1+2",
    "remtp": "ReMTP (margin-calibrated prefix relaxation)",
    "remtp_block": "ReMTP-Block (target prefix certificate)",
}


def _load_run(directory: Path) -> tuple[dict[str, Any], dict[str, Any], str]:
    payload = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    results = payload.get("results")
    if not isinstance(results, list) or len(results) != 1:
        raise ValueError(f"expected one result in {directory / 'summary.json'}")
    config = payload.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"missing config in {directory / 'summary.json'}")
    if results[0].get("evaluation_status") != "complete":
        raise ValueError(f"HumanEval evaluation is incomplete: {directory}")
    manifest = hashlib.sha256(
        (directory / "sample_manifest.json").read_bytes()
    ).hexdigest()
    return dict(config), dict(results[0]), manifest


def _generation_fingerprint(directory: Path) -> list[tuple[Any, ...]]:
    """Load only deterministic generation fields for identity attribution."""

    records: list[tuple[Any, ...]] = []
    for line in (directory / "requests.jsonl").read_text(
        encoding="utf-8"
    ).splitlines():
        row = json.loads(line)
        records.append(
            (
                row.get("task_id"),
                row.get("seed"),
                row.get("output_tokens"),
                row.get("finish_reason"),
                row.get("candidate"),
                row.get("raw_output"),
            )
        )
    return records


def validate_identity_control(run_root: Path, profiles: list[str]) -> None:
    """Require the zero-TV fused verifier to reproduce native MTP exactly."""

    if "target_mode_identity" not in profiles:
        return
    if "native_mtp" not in profiles:
        raise ValueError("identity attribution requires native_mtp")
    native = _generation_fingerprint(run_root / "native_mtp")
    identity = _generation_fingerprint(run_root / "target_mode_identity")
    if native != identity:
        raise ValueError(
            "fused strict-MTP identity control differs from native MTP output"
        )


def compare(run_root: Path, profiles: list[str]) -> list[dict[str, Any]]:
    if "cactus" not in profiles:
        raise ValueError("profiles must include the Cactus reference")
    validate_identity_control(run_root, profiles)
    baseline_config, baseline, baseline_manifest = _load_run(
        run_root / "cactus"
    )
    rows: list[dict[str, Any]] = []
    for profile in profiles:
        config, summary, manifest = _load_run(run_root / profile)
        if manifest != baseline_manifest:
            raise ValueError(f"sample manifest differs for {profile}")
        mismatched = [
            key
            for key in PROTOCOL_KEYS
            if config.get(key) != baseline_config.get(key)
        ]
        if mismatched:
            raise ValueError(
                f"benchmark protocol differs for {profile}: "
                + ", ".join(mismatched)
            )
        rows.append(
            {
                "directory": profile,
                "result_source": (
                    "historical" if (run_root / profile).is_symlink() else "new"
                ),
                "method": PROFILE_LABELS.get(profile, profile),
                "samples": summary["samples"],
                "pass_at_1": summary["pass_at_1"],
                "pass_at_1_delta_pp": 100.0
                * (summary["pass_at_1"] - baseline["pass_at_1"]),
                "decode_tok_s": summary["decode_tok_s"],
                "e2e_tok_s": summary["e2e_output_tok_s"],
                "e2e_delta_pct": 100.0
                * (
                    summary["e2e_output_tok_s"]
                    / max(baseline["e2e_output_tok_s"], 1e-30)
                    - 1.0
                ),
                "mean_acceptance_length": summary["mean_acceptance_length"],
                "mal_delta": summary["mean_acceptance_length"]
                - baseline["mean_acceptance_length"],
                "draft_acceptance": summary["draft_token_acceptance_rate"],
                "truncation_rate": summary["truncation_rate"],
                "timeouts": int(summary.get("evaluation_counts", {}).get("timeout", 0)),
                "manifest_sha256": manifest,
            }
        )
    return rows


def write_outputs(run_root: Path, rows: list[dict[str, Any]]) -> None:
    (run_root / "comparison.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (run_root / "comparison.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# HumanEval MTP comparison",
        "",
        "Quality is official-test `pass@1`; speed excludes Docker evaluation time.",
        "All rows use the same tasks, prompts, sampling parameters, and container policy.",
        "",
        "| method | source | pass@1 | vs Cactus | decode tok/s | e2e tok/s | "
        "vs Cactus | MAL | draft acceptance | truncation | timeouts |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['result_source']} | "
            f"{100.0 * row['pass_at_1']:.1f}% | "
            f"{row['pass_at_1_delta_pp']:+.1f} pp | "
            f"{row['decode_tok_s']:.3f} | {row['e2e_tok_s']:.3f} | "
            f"{row['e2e_delta_pct']:+.2f}% | "
            f"{row['mean_acceptance_length']:.3f} | "
            f"{100.0 * row['draft_acceptance']:.1f}% | "
            f"{100.0 * row['truncation_rate']:.1f}% | "
            f"{row['timeouts']} |"
        )
    (run_root / "comparison.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("profiles", nargs="+")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = compare(args.run_root, args.profiles)
    write_outputs(args.run_root, rows)
    print((args.run_root / "comparison.md").resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
