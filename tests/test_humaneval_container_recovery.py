"""An execution limit must not halt the suite; an unknown kill must still halt it."""
import pytest
from scripts.humaneval_container_recovery import classify_exit


def classify(code, stdout='', stderr='', state=None, timed_out=False):
    return classify_exit(code, stdout, stderr, state, timed_out=timed_out, timeout=8., memory='512m')


def test_confirmed_candidate_oom_is_a_runtime_failure():
    result = classify(137, state={'OOMKilled': True, 'ExitCode': 137})
    assert result['status'] == 'runtime_error'
    assert result['oom_killed'] is True


@pytest.mark.parametrize('state', [None, {}, {'OOMKilled': False, 'ExitCode': 137},
                                    {'OOMKilled': True, 'ExitCode': 1}])
def test_a_kill_without_matching_oom_evidence_stays_an_infrastructure_error(state):
    assert classify(137, state=state)['status'] == 'container_error'


def test_timeout_keeps_the_original_semantics():
    assert classify(137, state={'OOMKilled': False, 'ExitCode': 137}, timed_out=True) == {
        'status': 'timeout', 'detail': '> 8s'}


@pytest.mark.parametrize('status', ['passed', 'failed', 'runtime_error', 'syntax_error'])
def test_normal_runner_status_is_preserved(status):
    assert classify(0, stdout='{"status": "' + status + '", "detail": "proof"}\n') == {
        'status': status, 'detail': 'proof'}
