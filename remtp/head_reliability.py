"""Estimate conditional MTP-head reliability from a vLLM calibration log."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


METRIC_PATTERN = re.compile(
    r"Drafted:\s+(?P<drafted>\d+)\s+tokens,\s+"
    r"Per-position acceptance rate:\s+"
    r"(?P<rates>[0-9., ]+),\s+Avg Draft acceptance rate:"
)


def estimate_head_reliability(
    text: str,
    num_heads: int,
) -> tuple[list[float], list[float], int]:
    """Return conditional head reliability, prefix rates, and draft rounds."""
    if num_heads < 1:
        raise ValueError("num_heads must be positive")
    prefix_counts = [0.0] * num_heads
    total_rounds = 0.0
    for match in METRIC_PATTERN.finditer(text):
        rates = [
            float(value)
            for value in match.group("rates").split(", ")
        ]
        if len(rates) != num_heads:
            continue
        rounds = int(match.group("drafted")) / num_heads
        total_rounds += rounds
        for index, rate in enumerate(rates):
            prefix_counts[index] += rate * rounds
    if total_rounds == 0:
        raise ValueError(
            f"no {num_heads}-position speculative metrics found"
        )

    prefix_rates = [count / total_rounds for count in prefix_counts]
    conditional = [prefix_rates[0]]
    for index in range(1, num_heads):
        previous = prefix_rates[index - 1]
        conditional.append(
            prefix_rates[index] / previous if previous > 0 else 0.0
        )
    return conditional, prefix_rates, int(total_rounds)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Convert vLLM per-position acceptance logs into conditional "
            "MTP-head reliability values."
        )
    )
    parser.add_argument("server_log", type=Path)
    parser.add_argument("--mtp-tokens", type=int, default=4)
    args = parser.parse_args()

    reliability, prefix_rates, rounds = estimate_head_reliability(
        args.server_log.read_text(encoding="utf-8", errors="replace"),
        args.mtp_tokens,
    )
    formatted = ",".join(f"{value:.4f}" for value in reliability)
    prefix = ",".join(f"{value:.4f}" for value in prefix_rates)
    print(f"calibration_rounds={rounds}")
    print(f"prefix_acceptance={prefix}")
    print(f"HEAD_RELIABILITY={formatted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
