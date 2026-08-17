from __future__ import annotations

import json
from pathlib import Path

from remtp.fastmtp_residual_hit_report import build_report


def _row(method_id: str, quality: float, mal: float) -> dict[str, object]:
    return {
        "method_id": method_id,
        "method": method_id,
        "samples": 2,
        "quality": quality,
        "decode_tok_s": 100.0,
        "e2e_tok_s": 90.0,
        "mal": mal,
        "nodes": 3.0 if method_id != "dynamic_tree" else 9.0,
    }


def test_build_report_combines_routes_and_residual_hit_audit(tmp_path: Path) -> None:
    routes = ["shadow", "strict"]
    (tmp_path / "suite_manifest.json").write_text(
        json.dumps({"routes": routes}), encoding="utf-8"
    )
    baseline = [
        _row("native", 0.8, 3.0),
        _row("cactus", 0.8, 3.4),
        _row("spec_cascade", 0.85, 3.2),
    ]
    for route in routes:
        route_root = tmp_path / route
        (route_root / "route_manifest.json").parent.mkdir(parents=True)
        (route_root / "route_manifest.json").write_text(
            json.dumps(
                {
                    "label": route,
                    "support_mode": (
                        "sampled_primary_shadow"
                        if route == "shadow"
                        else "residual_hit_strict"
                    ),
                    "max_nodes": 9,
                    "max_children": 3,
                }
            ),
            encoding="utf-8",
        )
        dynamic = _row("dynamic_tree", 0.8, 3.5 if route == "strict" else 3.4)
        (route_root / "comparison.json").write_text(
            json.dumps({"gsm8k": baseline + [dynamic], "humaneval": baseline + [dynamic]}),
            encoding="utf-8",
        )
        for dataset in ("gsm8k", "humaneval"):
            audit = route_root / dataset / "dynamic_tree" / "tree_rounds.jsonl"
            audit.parent.mkdir(parents=True)
            audit.write_text(
                json.dumps(
                    {
                        "nodes": [
                            {
                                "node": 0,
                                "trunk_rejected": True,
                            },
                            {
                                "node": 1,
                                "correction_hit": True,
                                "rescued": route == "strict",
                                "rescue_unlocked_tokens": 2,
                            }
                        ]
                    }
                )
                + "\n",
                encoding="utf-8",
            )

    markdown, payload = build_report(tmp_path)
    assert "Exact-residual-hit tree exploration" in markdown
    assert "strict" in markdown
    strict = payload["gsm8k"]["routes"][1]
    assert strict["audit"]["correction_hit_rate"] == 1.0
    assert strict["audit"]["mean_unlocked_per_reused_hit"] == 2.0
