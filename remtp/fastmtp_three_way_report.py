"""Generate three-way comparison report for FastMTP experiments.

Compares:
1. Native probabilistic MTP (trained MTP baseline)
2. Cactus + MTP
3. SpecCascade TokenV3 + MTP

All methods use TencentBAC/FastMTP (Qwen2-based with trained MTP).
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    args = parser.parse_args()

    run_root: Path = args.run_root
    methods = ["native_mtp", "cactus", "spec_cascade"]
    datasets = ["gsm8k", "humaneval"]

    results = {}
    for dataset in datasets:
        results[dataset] = []
        for method in methods:
            method_dir = run_root / dataset / method
            summary = load_summary(method_dir / "summary.json")

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

            results[dataset].append(row)

    # Generate markdown report
    md_lines = ["# FastMTP Three-Way Comparison (50 samples)", ""]
    md_lines.append("Model: TencentBAC/FastMTP (Qwen2-based, 8B params, trained MTP)")
    md_lines.append("")

    for dataset in datasets:
        title = "GSM8K" if dataset == "gsm8k" else "HumanEval"
        quality_label = "accuracy" if dataset == "gsm8k" else "pass@1"
        md_lines.append(f"## {title} (50 tasks)")
        md_lines.append("")

        headers = ["method", quality_label, "truncation", "decode tok/s", "e2e tok/s", "MAL"]
        md_lines.append("| " + " | ".join(headers) + " |")
        md_lines.append("|" + "|".join(["---:" if i > 0 else "---" for i in range(len(headers))]) + "|")

        for row in results[dataset]:
            method_name = {
                "native_mtp": "Native MTP (trained)",
                "cactus": "Cactus + MTP",
                "spec_cascade": "SpecCascade TokenV3 + MTP",
            }[row["method"]]

            cells = [
                method_name,
                f"{row['quality']*100:.1f}%",
                f"{row['truncation_rate']*100:.1f}%",
                f"{row['decode_tok_s']:.2f}",
                f"{row['e2e_tok_s']:.2f}",
                f"{row['mal']:.3f}" if row['mal'] else "-",
            ]

            md_lines.append("| " + " | ".join(cells) + " |")

        md_lines.append("")

    md_lines.extend([
        "## Key Advantages of FastMTP",
        "",
        "FastMTP uses a **trained MTP layer** (not pretrained-only like MiMo):",
        "- Single MTP head with position-shared weights across 3 recursive steps",
        "- Trained on self-distilled data with MTP-inference alignment",
        "- Language-aware vocabulary compression",
        "- **Published performance**: 2.03× speedup, 82% better than vanilla MTP",
        "",
        "## Interpretation",
        "",
        "- **Native MTP**: Baseline with trained MTP - should have much higher acceptance rates than MiMo's untrained layers",
        "- **Cactus**: Constrained acceptance with delta=1.0 on top of trained MTP",
        "- **SpecCascade**: TokenV3 cascaded verification with alpha=0.5",
        "",
        "Compare MAL and quality carefully. If native trained MTP already achieves high MAL,",
        "relaxation methods may not provide significant additional benefit.",
        "",
        "## References",
        "",
        "- Model: [TencentBAC/FastMTP](https://huggingface.co/TencentBAC/FastMTP)",
        "- Paper: [FastMTP on arXiv](https://arxiv.org/abs/2509.18362)",
    ])

    md_path = run_root / "comparison.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    json_path = run_root / "comparison.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Report written to {md_path}")


if __name__ == "__main__":
    main()
