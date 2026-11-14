import json
from types import SimpleNamespace

import pytest
import torch

from remtp import round_logger


@pytest.fixture()
def log_path(tmp_path, monkeypatch):
    path = tmp_path / "rounds.jsonl"
    monkeypatch.setenv("REMTP_ROUND_LOG_JSONL", str(path))
    monkeypatch.setattr(round_logger, "_STREAM", None)
    monkeypatch.setattr(round_logger, "_FAILED", False)
    monkeypatch.setattr(round_logger, "_COUNTERS", {"chain": 0, "tree": 0})
    monkeypatch.setattr(round_logger, "_PENDING", None)
    yield path
    if round_logger._STREAM is not None:
        round_logger._STREAM.close()


def test_chain_inputs_gathers_target_and_draft_probabilities():
    logits = torch.log(torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]]))
    draft = torch.tensor([[0.5, 0.4, 0.1], [0.2, 0.2, 0.6]])
    ids = torch.tensor([1, 2])
    packed = round_logger.chain_inputs(ids, draft, logits)
    expected = torch.tensor([[0.2, 0.7, 0.0, 0.4, 0.5, 0.0],
                             [0.6, 0.6, 2.0, 0.6, 0.6, 2.0]])
    assert torch.allclose(packed, expected, atol=1e-6)


def test_chain_inputs_without_draft_probs_has_three_columns():
    packed = round_logger.chain_inputs(torch.tensor([0]), None, torch.zeros(1, 4))
    assert packed.shape == (1, 3)


def test_chain_round_record_counts_accepted_drafts(log_path):
    logits = torch.log(torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6], [0.3, 0.3, 0.4]]))
    draft = torch.full((3, 3), 1 / 3)
    ids = torch.tensor([0, 2, 1])
    packed = round_logger.chain_inputs(ids, draft, logits)
    output = torch.tensor([[0, 2, 0, -1]])  # two drafts accepted, then a correction
    round_logger.log_chain_round(ids, [3], torch.tensor([3]), packed, output)
    record = json.loads(log_path.read_text().splitlines()[0])
    assert record["kind"] == "chain" and record["round"] == 0
    assert record["draft"] == [0, 2, 1]
    assert record["emitted"] == [0, 2, 0]
    assert record["accepted"] == 2
    assert record["p"] == pytest.approx([0.7, 0.6, 0.3], abs=1e-6)
    assert record["q"] == pytest.approx([1 / 3] * 3, abs=1e-6)
    assert record["p_top_id"] == [0, 2, 2]


def test_tree_round_record_keeps_verifier_diagnostics(log_path, monkeypatch):
    from remtp import fixed6_vllm as fixed
    runtime = SimpleNamespace(diagnostic_request="req-1", generated_history=(5, 6, 7))
    monkeypatch.setattr(fixed, "_RUNTIME", runtime)
    topology = SimpleNamespace(nodes=(SimpleNamespace(path=(0,)), SimpleNamespace(path=(0, 0))))
    target = torch.tensor([[0.6, 0.4], [0.1, 0.9], [0.5, 0.5]])
    result = SimpleNamespace(output_token_ids=(1, 0), accepted_node_indices=(0,), terminal="correction",
                             evaluated_node_statuses=((0, "SELECTED"), (1, "REJECT")),
                             node_diagnostics=({"node": 0, "target_probability": 0.4, "accept_random": 0.1},
                                               {"node": 1, "target_probability": float("nan")}))
    round_logger.log_tree_round((topology, torch.tensor([1, 1]), None, target), result)
    record = json.loads(log_path.read_text().splitlines()[0])
    assert record["kind"] == "tree" and record["request"] == "req-1"
    assert record["generated_length"] == 3
    assert record["node_paths"] == [[0], [0, 0]]
    assert record["emitted"] == [1, 0] and record["accepted_nodes"] == [0]
    assert record["target_top_id"] == [0, 1, 0]
    assert record["nodes"][0]["accept_random"] == 0.1
    assert record["nodes"][1]["target_probability"] == "nan"   # non-finite kept as text


def test_logging_failure_disables_logger_without_raising(log_path):
    round_logger.log_chain_round(torch.tensor([0]), [1], torch.tensor([1]), "not a tensor", torch.tensor([[0, -1]]))
    assert round_logger._FAILED is True
    round_logger.log_chain_round(torch.tensor([0]), [1], torch.tensor([1]),
                                 torch.zeros(1, 6), torch.tensor([[0, -1]]))
    assert not log_path.exists() or log_path.read_text() == ""
