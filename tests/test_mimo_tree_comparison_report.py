from remtp.mimo_tree_comparison_report import build_report, markdown


def test_report_computes_deltas() -> None:
    native = {
        "samples": 50,
        "pass_at_1": 0.7,
        "mean_acceptance_length": 2.5,
        "decode_tok_s": 100.0,
        "e2e_output_tok_s": 90.0,
    }
    cactus = {
        "samples": 50,
        "pass_at_1": 0.6,
        "mean_acceptance_length": 3.0,
        "decode_tok_s": 110.0,
        "e2e_output_tok_s": 95.0,
    }
    dynamic = {
        "samples": 50,
        "pass_at_1": 0.5,
        "mean_acceptance_length": 2.0,
        "decode_tok_s": 80.0,
        "e2e_tok_s": 75.0,
        "average_tree_nodes": 6.0,
        "humaneval": {"draft_token_acceptance_rate": 0.2},
    }
    report = build_report(native, cactus, dynamic)
    assert report["rows"][1]["delta_mean_acceptance_length_vs_native"] == 0.5
    assert report["rows"][2]["delta_e2e_output_tok_s_vs_native"] == -15.0
    assert "Dynamic MTP Tree" in markdown(report)
