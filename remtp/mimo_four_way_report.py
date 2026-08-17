"""Generate four-way comparison report for MiMo experiments.

Compares:
1. Native probabilistic MTP (baseline)
2. Cactus + MTP
3. SpecCascade TokenV3 + MTP
4. Dynamic tree with target-dominant relaxation

All methods use the same MTP route (repeated layer 0) for fair comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_summary(path: Path) -> dict:
    """Load summary.json from a method run directory."""
    if not path.exists():
        raise FileNotFoundError(f"Missing summary: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_tree_metrics(path: Path) -> dict | None:
    """Load tree_metrics.json if available."""
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    args = parser.parse_args()

    run_root: Path = args.run_root
    methods = ["native_mtp", "cactus", "spec_cascade", "dynamic_tree"]
    datasets = ["gsm8k", "humaneval"]

    results = {}
    for dataset in datasets:
        results[dataset] = []
        for method in methods:
            method_dir = run_root / dataset / method
            summary = load_summary(method_dir / "summary.json")
            tree_metrics = load_tree_metrics(method_dir / "tree_metrics.json")

            # Extract key metrics
            result_row = summary["results"][0]
            quality_key = "accuracy" if dataset == "gsm8k" else "pass_at_1"

            row = {
                "method": method,
                "quality": result_row[quality_key],
                "truncation_rate": result_row.get("truncation_rate", 0.0),
                "decode_tok_s": result_row["decode_tokens_per_second"],
                "e2e_tok_s": result_row["e2e_output_tokens_per_second"],
                "mal": result_row.get("mean_acceptance_length"),
            }

            if tree_metrics:
                row["tree_mal"] = tree_metrics["tree_mal"]
                row["accepted_per_round"] = tree_metrics["mean_accepted_depth"]
                row["candidate_nodes"] = tree_metrics["mean_validation_nodes"]
                row["useful_node_ratio"] = tree_metrics.get("useful_node_ratio")

            results[dataset].append(row)

    # Generate markdown report
    md_lines = ["# MiMo Four-Way Comparison (50 samples, 000 head only)", ""]
    md_lines.append("All methods use repeated layer 0 for fair comparison.")
    md_lines.append("")

    for dataset in datasets:
        title = "GSM8K" if dataset == "gsm8k" else "HumanEval"
        quality_label = "accuracy" if dataset == "gsm8k" else "pass@1"
        md_lines.append(f"## {title} (50 tasks)")
        md_lines.append("")

        # Build table header
        headers = ["method", quality_label, "truncation", "decode tok/s", "e2e tok/s", "MAL"]
        if any(r.get("candidate_nodes") for r in results[dataset]):
            headers.extend(["accepted/round", "nodes/round"])

        md_lines.append("| " + " | ".join(headers) + " |")
        md_lines.append("|" + "|".join(["---:" if i > 0 else "---" for i in range(len(headers))]) + "|")

        # Build table rows
        for row in results[dataset]:
            method_name = {
                "native_mtp": "Native probabilistic MTP",
                "cactus": "Cactus + MTP",
                "spec_cascade": "SpecCascade TokenV3 + MTP",
                "dynamic_tree": "Dynamic tree + relaxation",
            }[row["method"]]

            cells = [
                method_name,
                f"{row['quality']*100:.1f}%",
                f"{row['truncation_rate']*100:.1f}%",
                f"{row['decode_tok_s']:.2f}",
                f"{row['e2e_tok_s']:.2f}",
                f"{row['mal']:.3f}" if row['mal'] else "-",
            ]

            if any(r.get("candidate_nodes") for r in results[dataset]):
                if row.get("candidate_nodes"):
                    cells.append(f"{row['accepted_per_round']:.3f}")
                    cells.append(f"{row['candidate_nodes']:.2f}")
                else:
                    cells.extend(["-", "-"])

            md_lines.append("| " + " | ".join(cells) + " |")

        md_lines.append("")

    # Add interpretation
    md_lines.extend([
        "## Interpretation",
        "",
        "- **Native MTP**: Baseline strict verification with standard rejection sampling",
        "- **Cactus**: Constrained acceptance with delta=1.0, increases candidate probability",
        "- **SpecCascade TokenV3**: Cascaded verification with alpha=0.5",
        "- **Dynamic tree**: Variable-width tree with target-dominant relaxation (tau_relax=0.7)",
        "",
        "All chain methods (native, cactus, spec_cascade) verify 3 draft tokens per round.",
        "Dynamic tree verifies a variable number of nodes (avg ~4-5) depending on MTP entropy.",
        "",
        "Quality should be compared within each dataset. MAL differences indicate acceptance efficiency.",
    ])

    # Write markdown
    md_path = run_root / "comparison.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    # Write JSON
    json_path = run_root / "comparison.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Report written to {md_path}")


if __name__ == "__main__":
    main()
