from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.run_mtp_repaired_regression import Repaired
from scripts.run_adaptive_round4 import GateRejected


def test_failed_new_probability_gate_stops_before_training(tmp_path):
    c = object.__new__(Repaired)
    c.root = tmp_path
    c.protocol = {'source_sha256':'fresh'}
    c.report = lambda:None
    statuses = []
    c.status = lambda value:statuses.append(value)
    with pytest.raises(GateRejected):
        c.gate('short_probability', {'passed':False})
    assert statuses == ['STOPPED_SHORT_PROBABILITY']
    assert 'fresh' in (tmp_path/'short_probability_gate.json').read_text()


def test_report_discloses_short_probe_and_no_formal_claim(tmp_path):
    c = object.__new__(Repaired)
    c.root = tmp_path
    c.args = SimpleNamespace()
    c.report()
    text = (tmp_path/'report.md').read_text()
    assert '32token' in text and '128 token' not in text
    assert '未运行独立留出或三基准正式测试' in text
