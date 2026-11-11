"""Protect data identity, recovery, and scientific scoring in long formal runs."""
import json
import pytest
from scripts.run_raven_full_benchmark import read_checkpoint, quality_result, variants
from scripts.run_raven_dominance_search import variants as screen_variants


def test_power_loss_preserves_completed_data_and_the_torn_tail(tmp_path):
    path = tmp_path / 'requests.jsonl'
    complete = {'sample_id': 'HumanEval/3', 'seed': 72, 'output': 'saved'}
    path.write_bytes((json.dumps(complete) + '\n').encode() + b'{"sample_id":')
    assert read_checkpoint(path, [('HumanEval/3', 72), ('HumanEval/5', 92)]) == [complete]
    assert json.loads(path.read_text()) == complete
    assert next(tmp_path.glob('*.partial.*')).read_bytes() == b'{"sample_id":'


@pytest.mark.parametrize('records', [
    [{'sample_id': 5, 'seed': 93}],  # Changed generation seed.
    [{'sample_id': 17, 'seed': 212}],  # Changed sample ordering.
    [{'sample_id': 5, 'seed': 92}, {'sample_id': 5, 'seed': 92}],  # Duplicate output.
])
def test_resume_refuses_mixed_completed_records(tmp_path, records):
    path = tmp_path / 'requests.jsonl'
    path.write_text(''.join(json.dumps(row) + '\n' for row in records))
    with pytest.raises(ValueError, match='registered sample/seed prefix'):
        read_checkpoint(path, [(5, 92), (17, 212)])


def test_corrupt_complete_record_is_never_discarded(tmp_path):
    path = tmp_path / 'requests.jsonl'
    path.write_text('{bad json}\n')
    with pytest.raises(json.JSONDecodeError):
        read_checkpoint(path, [(5, 92)])
    assert path.read_text() == '{bad json}\n'


@pytest.mark.parametrize('status', ['container_error', 'runner_error', 'unexpected'])
def test_infrastructure_failure_cannot_be_scored_as_a_wrong_answer(status):
    with pytest.raises(RuntimeError):
        quality_result({'status': status, 'detail': 'test failure'})


def test_formal_candidate_preserves_the_chosen_screen_and_original_is_distinct():
    formal, screen = variants(), screen_variants()
    assert formal['deep4_conservative'] == screen['deep4_conservative']
    assert formal['adaptive_original']['protect'] is False
    assert screen['B']['protect'] is True
    assert formal['adaptive_original']['delta'] == screen['B']['delta']
    assert formal['adaptive_original']['mix'] == screen['B']['mix']
