from __future__ import annotations

import torch

from remtp.adaptive_chain_mtp import AdaptiveChainConfig, select_chain_depth


def test_adaptive_chain_selects_long_medium_and_short_depths() -> None:
    config = AdaptiveChainConfig(
        max_depth=6,
        medium_depth=4,
        short_depth=2,
        top_m=3,
        high_margin=1.5,
        medium_margin=0.5,
        low_entropy=0.45,
        high_entropy=0.85,
        high_mtp_confidence=0.3,
        medium_mtp_confidence=0.1,
    )

    logits = torch.tensor([4.0, 1.0, 0.0, -1.0])
    depth, _, _ = select_chain_depth(
        logits.softmax(dim=-1), logits, torch.tensor(0.6), config
    )
    assert int(depth) == 6

    logits = torch.tensor([2.0, 1.2, 0.5, -1.0])
    depth, _, _ = select_chain_depth(
        logits.softmax(dim=-1), logits, torch.tensor(0.2), config
    )
    assert int(depth) == 4

    logits = torch.tensor([1.0, 0.95, 0.9, 0.85])
    depth, _, _ = select_chain_depth(
        logits.softmax(dim=-1), logits, torch.tensor(0.05), config
    )
    assert int(depth) == 2
