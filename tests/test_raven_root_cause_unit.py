from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import time

import pytest
import torch

from remtp.decision_trace import SummaryRecorder
from remtp.dynamic_mtp_tree import DynamicTreeTopology
from remtp.fixed6_microtree import TreeVerifyResult
from scripts.analyze_raven_root_causes import validate_summary, request_folder, budget_counterexample
from scripts.run_raven_root_cause_unit import RootCauseUnit, natural_frames, LIMITS, unique_prefix_cases
from scripts.run_raven_review_unit import Ledger, NumericRejected


def result(tokens=(4, 5, 6)):
    detail = dict(node=0, proposal_role='sampled_cactus_primary',
        target_probability=.1, draft_probability=.5, verification_probability=.4,
        base_effective_relaxation_delta=1., effective_relaxation_delta=.5,
        accept_random=.3, accept_probability=.8, normal_survival=True,
        trunk_position=1)
    return TreeVerifyResult(output_token_ids=tokens, accepted_node_indices=(0,),
        selected_branch=0, terminal='bonus', target_validation_nodes=1,
        evaluated_node_statuses=((0, 'SELECTED'),), node_diagnostics=(detail,),
        surviving_paths=1, selected_path_score=1.)


def test_summary_budget_skips_warmup_preserves_rng_and_counts_actual_requests(tmp_path):
    recorder = SummaryRecorder(tmp_path/'s.jsonl', max_rounds=1, max_requests=1)
    runtime = SimpleNamespace(diagnostic_request='warmup', diagnostic_prompt_ids=tuple(range(12)),
                              generated_history=(2,))
    topology = DynamicTreeTopology.from_paths([(0,)])
    torch.manual_seed(29); state = torch.random.get_rng_state().clone()
    recorder.save(runtime, topology, torch.tensor([4]), result())
    assert not recorder.path.exists() and not recorder.counts
    runtime.diagnostic_request='actual'; runtime.diagnostic_prompt_ids=tuple(range(65))
    recorder.save(runtime, topology, torch.tensor([4]), result())
    assert torch.equal(state, torch.random.get_rng_state())
    assert recorder.counts == {'actual': 1}
    with pytest.raises(RuntimeError, match='round budget'):
        recorder.save(runtime, topology, torch.tensor([4]), result())
    runtime.diagnostic_request='unexpected'
    with pytest.raises(RuntimeError, match='request budget'):
        recorder.save(runtime, topology, torch.tensor([4]), result())
    assert len(recorder.path.read_text().splitlines()) == 1


def test_exact_enumeration_disproves_block_TV_as_a_root_quality_guarantee():
    check=budget_counterexample()
    assert check['all_blocks_spend_no_more_TV']
    assert check['root_marginal_shift_ratio']==pytest.approx(1.5,rel=2e-6)
    assert sum(check['uniform_root_output'])==pytest.approx(1.,abs=1e-6)
    assert sum(check['adaptive_root_output'])==pytest.approx(1.,abs=1e-6)
    risky=[b for b in check['blocks'] if b['path'][0]==1]
    assert all(b['root_effective_delta']==pytest.approx(2.25,rel=2e-6) for b in risky)


def test_exact_duplicate_prefixes_share_one_fresh_call_without_dropping_natural_cases():
    cases=[dict(label='seed43_first',prefix_ids=[1,2,3]),
           dict(label='seed42_first',prefix_ids=[1,2,3]),
           dict(label='seed42_last',prefix_ids=[1,2,4])]
    unique,mapping=unique_prefix_cases(cases)
    assert [c['label'] for c in unique]==['seed43_first','seed42_last']
    assert mapping==dict(seed43_first='seed43_first',seed42_first='seed43_first',
                         seed42_last='seed42_last')
    assert len(cases)==3


def test_real_trace_wrapper_calls_original_once_without_new_rng_or_sampling(tmp_path, monkeypatch):
    import remtp
    from remtp import decision_trace as module
    topology = DynamicTreeTopology.from_paths([(0,)])
    runtime = SimpleNamespace(diagnostic_request='chatcmpl-x', diagnostic_prompt_ids=tuple(range(65)),
                              generated_history=(2,))
    actual = torch.Generator().manual_seed(17); expected = torch.Generator().manual_seed(17)
    calls = []
    def verify(topology, ids, draft, target, generator):
        calls.append(torch.rand((), generator=generator))
        return result()
    fixed = SimpleNamespace(_RUNTIME=runtime, strict_tree_verify=verify)
    dynamic = SimpleNamespace(_run_dynamic_tree_nodes=lambda *a, **k: None)
    eagle = SimpleNamespace(propose_tree=lambda *a, **k: None)
    monkeypatch.setattr(remtp, 'fixed6_vllm', fixed, raising=False)
    monkeypatch.setattr(remtp, 'dynamic_tree_vllm', dynamic, raising=False)
    monkeypatch.setitem(sys.modules, 'vllm.v1.spec_decode.eagle', SimpleNamespace(EagleProposer=eagle))
    monkeypatch.setattr(module, '_INSTALLED', False)
    monkeypatch.setenv('REMTP_DECISION_TRACE_CHECKPOINTS', '32')
    monkeypatch.setenv('REMTP_DECISION_TRACE_EARLY_PROPOSALS', '3')
    monkeypatch.setenv('REMTP_DECISION_TRACE_SUMMARY_JSONL', str(tmp_path/'summary.jsonl'))
    module.install_decision_trace(tmp_path/'trace')
    for length in (1, 32):
        runtime.generated_history=(2,)*length
        value = fixed.strict_tree_verify(topology, torch.tensor([4]), torch.ones(1,6),
                                         torch.ones(2,6), actual)
        assert value == result()
        assert torch.equal(calls[-1], torch.rand((), generator=expected))
    assert len(calls) == 2 and torch.equal(actual.get_state(), expected.get_state())
    assert len(list((tmp_path/'trace').glob('*/verify_*.pt'))) == 1
    assert len((tmp_path/'summary.jsonl').read_text().splitlines()) == 2


def summary_fixture():
    out = [2, 4, 5, 6, 7, 8]
    rows = [dict(request='chatcmpl-x', round=i, generated_length=n, history_suffix=out[:n],
                 candidate_ids=[4], result=asdict(result(tokens)))
            for i, n, tokens in ((0,1,(4,5,6)), (1,4,(7,8,9)))]
    # JSON arrays are the persisted format.
    return json.loads(json.dumps(rows)), dict(output_token_ids=out, output_tokens=len(out))


def test_live_id_validation_accepts_only_final_budget_clipping_and_marks_local_counterfactual():
    rows, record = summary_fixture()
    check = validate_summary(rows, record)
    assert check['counts']['rounds'] == 2
    assert check['counts']['extra_accept_over_strict'] == 2
    assert check['counts']['accept_differs_from_uniform'] == 0


@pytest.mark.parametrize('fault', ['prefix', 'emitted', 'commit_gap', 'count'])
def test_live_id_validation_rejects_missing_or_misaligned_evidence(fault):
    rows, record = summary_fixture()
    if fault == 'prefix': rows[0]['history_suffix']=[99]
    if fault == 'emitted': rows[0]['result']['output_token_ids'][0]=99
    if fault == 'commit_gap': rows[1]['generated_length']=5
    if fault == 'count': record['output_tokens']=7
    with pytest.raises(ValueError):
        validate_summary(rows, record)


def test_folder_pairing_requires_exact_api_id_boundary_and_unique_engine_id(tmp_path):
    (tmp_path/'chatcmpl-x-longer').mkdir()
    assert request_folder(tmp_path, 'chatcmpl-x').name == 'chatcmpl-x-longer'
    (tmp_path/'chatcmpl-xy').mkdir()
    with pytest.raises(ValueError):
        request_folder(tmp_path, 'chatcmpl')
    (tmp_path/'chatcmpl-x').mkdir()
    with pytest.raises(ValueError):
        request_folder(tmp_path, 'chatcmpl-x')


def coupled_fixture(tmp_path):
    prompt = [0]*65; output = [2, 0, 1, 2]
    trace = tmp_path/'decision_trace/chatcmpl-x-0'
    probe = tmp_path/'target_probe/chatcmpl-x-0'
    trace.mkdir(parents=True); probe.mkdir(parents=True)
    logits = torch.tensor([3.,2.,1.,0.])
    p = torch.softmax(logits/.6,-1)
    for i, n in enumerate((1,2)):
        frame = dict(prompt_ids=tuple(prompt), history=tuple(output[:n]),
            topology=((0,),), ids=torch.tensor([0]), draft=torch.ones(1,4)/4,
            target=torch.stack([p,p]))
        raw = dict(prompt_ids=tuple(prompt), output_ids=tuple(output[:n]), shape_probes={},
            topology=((0,),), draft_ids=torch.tensor([0]), logits_rows=torch.stack([logits,logits]),
            first_logits=logits, actual_input_ids=(output[n-1],0), actual_positions=(65+n-1,65+n))
        torch.save(frame,trace/f'verify_{i:02d}.pt')
        torch.save(raw,probe/f'logits_{i:02d}.pt')
    record = dict(response_id='chatcmpl-x', prompt_token_ids=prompt, output_token_ids=output)
    return record, probe


def test_natural_frame_gate_checks_actual_cache_prefix_position_and_raw_P(tmp_path):
    record, _ = coupled_fixture(tmp_path)
    checks = natural_frames(tmp_path, record)
    assert len(checks) == 2
    assert all(c['actual_raw_to_P_cpu_tv'] == 0 for c in checks)


def test_natural_target_probe_skips_warmup_and_prefill_without_extra_head_calls(tmp_path,monkeypatch):
    from remtp import target_diagnostics as module
    import remtp
    calls=[]
    class Model:
        def compute_logits(self,hidden):
            calls.append(hidden.shape[0])
            return torch.ones(hidden.shape[0],4)
    class Runner:
        def _prepare_inputs(self,*a):return 'original'
    monkeypatch.setitem(sys.modules,'vllm.model_executor.models.mimo',SimpleNamespace(MiMoForCausalLM=Model))
    monkeypatch.setitem(sys.modules,'vllm.v1.worker.gpu_model_runner',SimpleNamespace(GPUModelRunner=Runner))
    runtime=SimpleNamespace(target_tree_active=True,topology=SimpleNamespace(choices=((0,),)),
                            draft_token_ids=torch.tensor([0]))
    monkeypatch.setattr(remtp,'fixed6_vllm',SimpleNamespace(_RUNTIME=runtime),raising=False)
    for key,value in {
        'REMTP_TARGET_DIAGNOSTIC_DIR':str(tmp_path/'probe'),'REMTP_TARGET_PROBE_MIN_PROMPT_TOKENS':'65',
        'REMTP_TARGET_PROBE_MIN_GENERATED':'1','REMTP_TARGET_PROBE_MAX_FRAMES':'8',
        'REMTP_TARGET_PROBE_CHECKPOINTS':'0,4','REMTP_TARGET_PROBE_SHAPE_ROWS':'',
        'REMTP_TARGET_PROBE_PREFIX_MANIFEST':''}.items():monkeypatch.setenv(key,value)
    monkeypatch.delenv('REMTP_TARGET_STATE_AUDIT',raising=False)
    module.install_target_diagnostics()
    runner=Runner(); runner.query_start_loc=SimpleNamespace(np=[0,2])
    runner.input_ids=SimpleNamespace(gpu=torch.tensor([2,0]))
    runner.positions=SimpleNamespace(gpu=torch.tensor([65,66]))
    model=Model(); hidden=torch.ones(2,3)
    for request,prompt,output in [('warmup',[0]*30,[2]),('actual',[0]*65,[]),('actual',[0]*65,[2])]:
        runner.input_batch=SimpleNamespace(req_ids=[request])
        runner.requests={request:SimpleNamespace(prompt_token_ids=prompt,output_token_ids=output)}
        assert runner._prepare_inputs()=='original'
        model.compute_logits(hidden)
    frames=list((tmp_path/'probe').glob('*/*.pt'))
    assert calls==[2,2,2] and len(frames)==1
    frame=torch.load(frames[0],weights_only=True)
    assert frame['request']=='actual' and frame['output_ids']==(2,)
    assert frame['shape_probes']=={}


@pytest.mark.parametrize('fault', ['position','input','shape_probe','P','duplicate'])
def test_natural_frame_gate_rejects_observer_alignment_faults(tmp_path, fault):
    record, probe = coupled_fixture(tmp_path)
    path = probe/'logits_00.pt'; raw = torch.load(path,weights_only=True)
    if fault=='position': raw['actual_positions']=(64,65)
    if fault=='input': raw['actual_input_ids']=(99,0)
    if fault=='shape_probe': raw['shape_probes']={1:torch.ones(4)}
    if fault=='P': raw['first_logits']=torch.zeros(4)
    torch.save(raw,path)
    if fault=='duplicate': torch.save(raw,probe/'logits_02.pt')
    with pytest.raises(NumericRejected):
        natural_frames(tmp_path, record)


def test_actual_root_cause_startup_uses_tested_MRO_and_trace_only_env(tmp_path,monkeypatch):
    import scripts.run_adaptive_round2 as base
    from scripts.run_adaptive_round6 import configs
    c = object.__new__(RootCauseUnit)
    c.root=tmp_path; c.args=SimpleNamespace(port=8000,decision_trace=True,decision_trace_only=True,
        decision_trace_early_proposals=3,decision_trace_checkpoints=(0,4),target_probe=True,
        target_probe_shape_rows=(),target_probe_min_generated=1,target_probe_min_prompt_tokens=65)
    c.current={**configs()['old_eos01'],'probability_ratio_cap':0.}
    c.diagnostic=False; c.server=None; c.base_url='http://fake'; c.eos_ids=[3]
    c.ledger=Ledger(tmp_path/'ledger.json',LIMITS); c.state={}; c.work={}; c.forensics=[]
    c.started=time.monotonic(); c.runtime_limit=14400
    c.stop_server=lambda:None; captured={}
    class Socket:
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def connect_ex(self,*a):return 1
    class Response:
        status=200
        def __enter__(self):return self
        def __exit__(self,*a):pass
    def popen(*args,**kwargs):
        captured.update(kwargs['env'])
        kwargs['stdout'].write('[ReMTP][FastMTPVerified] method=dynamic_tree worker=active\n'
            '[ReMTP][HeadBatch] input_rows=1\nrelax_max_probability_ratio=0 lazy_residual=0\n')
        kwargs['stdout'].flush()
        return SimpleNamespace(poll=lambda:None)
    monkeypatch.setattr(base.socket,'socket',Socket)
    monkeypatch.setattr(base.subprocess,'run',lambda *a,**k:SimpleNamespace(stdout='15W'))
    monkeypatch.setattr(base.subprocess,'Popen',popen)
    monkeypatch.setattr(base.urllib.request,'urlopen',lambda *a,**k:Response())
    monkeypatch.setattr(base,'chat_completion',lambda *a:dict(usage={'completion_tokens':8}))
    directory=tmp_path/'natural/B'
    c.start_server('B',directory)
    assert captured['REMTP_PREFIX_DIAGNOSTIC_JSONL']==''
    assert captured['REMTP_DECISION_TRACE_DIR']==str(directory/'decision_trace')
    assert captured['REMTP_DECISION_TRACE_SUMMARY_JSONL']==str(directory/'decision_summary.jsonl')
    assert captured['REMTP_DECISION_TRACE_EARLY_PROPOSALS']=='3'
    assert captured['REMTP_TARGET_PROBE_SHAPE_ROWS']==''
    assert captured['REMTP_TARGET_PROBE_MIN_PROMPT_TOKENS']=='65'
    assert captured['REMTP_RUNTIME_PROFILE_JSONL']==''
    assert c.ledger.counts()['starts']['completed']==1
    assert c.ledger.counts()['warmup']['completed']==1
