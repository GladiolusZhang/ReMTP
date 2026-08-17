"""Compare relaxed-MTP experiments against the Cactus reference."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Any


_PROTOCOL_KEYS = (
    "samples",
    "sample_seed",
    "temperature",
    "generation_seed",
    "max_tokens",
    "mtp_tokens",
    "data_sha256",
    "answer_metric",
)


def _load_run(directory: Path) -> tuple[dict[str, Any], dict[str, Any], str]:
    path = directory / "summary.json"
    manifest_path = directory / "sample_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing sample manifest: {manifest_path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    results = payload.get("results")
    if not isinstance(results, list) or len(results) != 1:
        raise ValueError(f"expected one result in {path}")
    config = payload.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"missing benchmark config in {path}")
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    return dict(config), dict(results[0]), manifest_hash


def _profile_label(directory: str) -> str:
    labels = {
        "native_mtp": "Native probabilistic MTP",
        "cactus": "Cactus + MTP",
        "spec_cascade": "SpecCascade TokenV3 + MTP",
        "debt_conservative": "Current-block debt (conservative)",
        "debt_balanced": "Current-block debt (balanced)",
        "debt_cactus_fallback": "Current-block debt (Cactus fallback)",
        "debt_scale": "Current-block debt + posterior scale",
        "debt_top1": "Current-block debt + nonnegative top-1 bias",
        "top1_surplus": "Cactus-dominant top-1 surplus",
        "target_surplus": "Cactus-dominant target surplus",
        "risk_swap": "Target-anchored Cactus risk swap",
        "target_recovery": "Target-surplus + target recovery",
        "native_block": "Native MTP + Block Verification",
        "cactus_block": "Cactus + Block Verification",
        "risk_swap_block": "Target-anchored risk swap + Block Verification",
        "block_shield": "Block-surplus-shielded risk control",
        "event_shield": "Event-triggered Block Shield",
        "risk_gated_block": "Target-Risk-Gated Block Relaxation",
        "regret_calibrated_block": "Regret-Calibrated Block Relaxation",
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
    return labels.get(directory, directory)


def _request_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        record.get("task"),
        record.get("question_id"),
        record.get("sample_index"),
        record.get("seed"),
    )


def _load_requests(directory: Path) -> dict[tuple[Any, ...], dict[str, Any]]:
    path = directory / "requests.jsonl"
    rows: dict[tuple[Any, ...], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            key = _request_key(record)
            if key in rows:
                raise ValueError(f"duplicate request key in {path}: {key}")
            rows[key] = record
    if not rows:
        raise ValueError(f"no request records in {path}")
    return rows


def _request_components(
    record: dict[str, Any],
) -> tuple[float, float, float, float, float]:
    metrics = record.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("request record is missing metrics")
    return (
        float(bool(record.get("correct"))),
        float(record["output_tokens"]),
        float(record["client_seconds"]),
        float(metrics["vllm:spec_decode_num_accepted_tokens_total"]),
        float(metrics["vllm:spec_decode_num_drafts_total"]),
    )


def _aggregate_components(
    rows: list[tuple[float, float, float, float, float]],
    indices: list[int],
) -> tuple[float, float, float]:
    correct = output_tokens = seconds = accepted = rounds = 0.0
    for index in indices:
        row = rows[index]
        correct += row[0]
        output_tokens += row[1]
        seconds += row[2]
        accepted += row[3]
        rounds += row[4]
    count = len(indices)
    if count == 0 or seconds <= 0.0 or rounds <= 0.0:
        raise ValueError("paired bootstrap requires non-empty timed requests")
    return (
        correct / count,
        output_tokens / seconds,
        1.0 + accepted / rounds,
    )


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def paired_bootstrap_intervals(
    run_root: Path,
    profiles: list[str],
    *,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 20260802,
) -> dict[str, dict[str, tuple[float, float]]]:
    """Return paired request-bootstrap intervals against Cactus.

    Accuracy is paired by request. E2E throughput is recomputed as the ratio
    of resampled output-token and wall-time sums. MAL is recomputed as
    ``1 + accepted_drafts / draft_rounds`` rather than averaging request MALs.
    """
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    baseline_map = _load_requests(run_root / "cactus")
    keys = sorted(baseline_map, key=repr)
    baseline_rows = [_request_components(baseline_map[key]) for key in keys]
    result: dict[str, dict[str, tuple[float, float]]] = {
        "cactus": {
            "accuracy_delta_pp": (0.0, 0.0),
            "e2e_delta_pct": (0.0, 0.0),
            "mal_delta": (0.0, 0.0),
        }
    }
    for profile in profiles:
        candidate_map = _load_requests(run_root / profile)
        if set(candidate_map) != set(baseline_map):
            raise ValueError(
                f"request identities differ between cactus and {profile}"
            )
        candidate_rows = [
            _request_components(candidate_map[key]) for key in keys
        ]
        rng = random.Random(bootstrap_seed)
        accuracy_deltas: list[float] = []
        e2e_deltas: list[float] = []
        mal_deltas: list[float] = []
        for _ in range(bootstrap_samples):
            indices = [rng.randrange(len(keys)) for _ in keys]
            base_accuracy, base_e2e, base_mal = _aggregate_components(
                baseline_rows,
                indices,
            )
            cand_accuracy, cand_e2e, cand_mal = _aggregate_components(
                candidate_rows,
                indices,
            )
            accuracy_deltas.append(100.0 * (cand_accuracy - base_accuracy))
            e2e_deltas.append(100.0 * (cand_e2e / base_e2e - 1.0))
            mal_deltas.append(cand_mal - base_mal)
        result[profile] = {
            "accuracy_delta_pp": (
                _quantile(accuracy_deltas, 0.025),
                _quantile(accuracy_deltas, 0.975),
            ),
            "e2e_delta_pct": (
                _quantile(e2e_deltas, 0.025),
                _quantile(e2e_deltas, 0.975),
            ),
            "mal_delta": (
                _quantile(mal_deltas, 0.025),
                _quantile(mal_deltas, 0.975),
            ),
        }
    return result


def _attach_intervals(
    rows: list[dict[str, Any]],
    intervals: dict[str, dict[str, tuple[float, float]]],
) -> None:
    for row in rows:
        profile = row["directory"]
        values = intervals[profile]
        for metric, (lower, upper) in values.items():
            row[f"{metric}_ci95_low"] = lower
            row[f"{metric}_ci95_high"] = upper


def compare(
    run_root: Path,
    profiles: list[str],
    *,
    eligible_profiles: set[str] | None = None,
) -> list[dict[str, Any]]:
    baseline_config, baseline, baseline_manifest = _load_run(
        run_root / "cactus"
    )
    rows: list[dict[str, Any]] = []
    for directory in ["cactus", *profiles]:
        config, summary, manifest = _load_run(run_root / directory)
        if manifest != baseline_manifest:
            raise ValueError(
                f"sample manifest differs between cactus and {directory}"
            )
        mismatched = [
            key
            for key in _PROTOCOL_KEYS
            if config.get(key) != baseline_config.get(key)
        ]
        if mismatched:
            raise ValueError(
                f"benchmark protocol differs for {directory}: "
                + ", ".join(mismatched)
            )
        accuracy_delta = summary["accuracy"] - baseline["accuracy"]
        mal_delta = (
            summary["mean_acceptance_length"]
            - baseline["mean_acceptance_length"]
        )
        e2e_delta = (
            summary["e2e_output_tok_s"]
            - baseline["e2e_output_tok_s"]
        )
        eligible = (
            directory not in {"cactus", "native_mtp"}
            if eligible_profiles is None
            else directory in eligible_profiles
        )
        pareto_pass = (
            eligible
            and accuracy_delta > 0.0
            and mal_delta > 0.0
            and e2e_delta >= 0.0
        )
        rows.append(
            {
                "directory": directory,
                "manifest_sha256": manifest,
                "method": _profile_label(directory),
                "samples": summary["samples"],
                "accuracy": summary["accuracy"],
                "accuracy_delta_pp": 100.0 * accuracy_delta,
                "decode_tok_s": summary["decode_tok_s"],
                "e2e_tok_s": summary["e2e_output_tok_s"],
                "e2e_delta_pct": (
                    100.0
                    * e2e_delta
                    / max(baseline["e2e_output_tok_s"], 1e-30)
                ),
                "mean_acceptance_length": summary[
                    "mean_acceptance_length"
                ],
                "mal_delta": mal_delta,
                "draft_acceptance": summary[
                    "draft_token_acceptance_rate"
                ],
                "truncation_rate": summary["truncation_rate"],
                "pareto_pass": pareto_pass,
            }
        )
    return rows


def _write_outputs(run_root: Path, rows: list[dict[str, Any]]) -> None:
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
        "# Relaxed-MTP Pareto comparison",
        "",
        "Pass condition: Accuracy > Cactus, MAL > Cactus, and E2E >= Cactus.",
        "",
        "| method | accuracy | delta | e2e tok/s | delta | MAL | delta | "
        "draft acceptance | truncation | pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {100.0*row['accuracy']:.1f}% | "
            f"{row['accuracy_delta_pp']:+.1f} pp | "
            f"{row['e2e_tok_s']:.3f} | {row['e2e_delta_pct']:+.2f}% | "
            f"{row['mean_acceptance_length']:.3f} | "
            f"{row['mal_delta']:+.3f} | "
            f"{100.0*row['draft_acceptance']:.1f}% | "
            f"{100.0*row['truncation_rate']:.1f}% | "
            f"{'YES' if row['pareto_pass'] else 'NO'} |"
        )
    if "accuracy_delta_pp_ci95_low" in rows[0]:
        lines.extend(
            [
                "",
                "## Paired request-bootstrap 95% intervals",
                "",
                "These intervals are diagnostic and do not replace the "
                "locked point-estimate gate or second-seed replication.",
                "",
                "| method | accuracy delta | E2E delta | MAL delta |",
                "|---|---:|---:|---:|",
            ]
        )
        for row in rows:
            lines.append(
                f"| {row['method']} | "
                f"[{row['accuracy_delta_pp_ci95_low']:+.1f}, "
                f"{row['accuracy_delta_pp_ci95_high']:+.1f}] pp | "
                f"[{row['e2e_delta_pct_ci95_low']:+.2f}%, "
                f"{row['e2e_delta_pct_ci95_high']:+.2f}%] | "
                f"[{row['mal_delta_ci95_low']:+.3f}, "
                f"{row['mal_delta_ci95_high']:+.3f}] |"
            )
    (run_root / "comparison.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _select(rows: list[dict[str, Any]]) -> dict[str, Any]:
    feasible = [row for row in rows if row["pareto_pass"]]
    winner = None
    if feasible:
        winner = max(
            feasible,
            key=lambda row: (
                row["accuracy"],
                row["mean_acceptance_length"],
                row["e2e_tok_s"],
            ),
        )
    return {
        "pareto_pass": winner is not None,
        "winner": None if winner is None else winner["directory"],
        "selection_order": ["accuracy", "mean_acceptance_length", "e2e_tok_s"],
        "criterion": {
            "accuracy": "strictly greater than Cactus",
            "mean_acceptance_length": "strictly greater than Cactus",
            "e2e_tok_s": "greater than or equal to Cactus",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("profiles", nargs="+")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260802)
    parser.add_argument(
        "--eligible",
        action="append",
        default=None,
        help="profile eligible for selection; repeat for multiple profiles",
    )
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    rows = compare(
        run_root,
        args.profiles,
        eligible_profiles=(
            None if args.eligible is None else set(args.eligible)
        ),
    )
    intervals = paired_bootstrap_intervals(
        run_root,
        args.profiles,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    _attach_intervals(rows, intervals)
    _write_outputs(run_root, rows)
    selection = _select(rows)
    selection_text = (
        json.dumps(selection, ensure_ascii=False, indent=2) + "\n"
    )
    (run_root / "selection.json").write_text(
        selection_text,
        encoding="utf-8",
    )
    # Compatibility for historical experiment scripts.
    (run_root / "gate_selection.json").write_text(
        selection_text,
        encoding="utf-8",
    )
    print((run_root / "comparison.md").read_text(encoding="utf-8"))
    print("SELECTION=" + json.dumps(selection, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
