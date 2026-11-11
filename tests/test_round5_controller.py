import pytest
import json
from types import SimpleNamespace
from scripts.run_adaptive_round5 import Round5, validate_reference, output_signature
from scripts.run_adaptive_round4 import NUMERICAL_KEYS
from scripts.run_adaptive_round2 import Round2


def reference():
    row={'question_id':1,'gold_answer':'2'}
    config={'depth':3}
    job=dict(settings=config,pairs=[[1,52]],limit=2048,diagnostic=False)
    record=dict(question_id=1,seed=52,output='#### 2',output_tokens=4,finish_reason='stop',
                gold_answer='2',predicted_answer='2',answer_source='hash',correct=True)
    return row,config,job,record


@pytest.mark.parametrize('fault',['partial','seed','budget','diagnostic','settings','gold','scoring'])
def test_reference_reuse_rejects_incompatible_or_incomplete_evidence(fault):
    row,config,job,record=reference();records=[record]
    if fault=='partial':records=[]
    elif fault=='seed':record['seed']=53
    elif fault=='budget':job['limit']=128
    elif fault=='diagnostic':job['diagnostic']=True
    elif fault=='settings':job['settings']={'depth':4}
    elif fault=='gold':record['gold_answer']='3'
    elif fault=='scoring':record['correct']=False
    with pytest.raises(ValueError):validate_reference(records,job,config,[(row,42)],{1:row})


def test_reference_reuse_keeps_original_evaluator_and_replay_checks_stopping():
    row,config,job,record=reference()
    validate_reference([record],job,config,[(row,42)],{1:row})
    assert output_signature(record)==output_signature({**record,'client_seconds':1000})
    assert output_signature(record)!=output_signature({**record,'finish_reason':'length'})


def test_controller_start_path_dispatches_to_base_and_records_effective_options(tmp_path,monkeypatch):
    # Exercise the real wrapper before a costly GPU launch. Borrowing this
    # method onto a sibling class fails its zero-argument super() dispatch.
    controller=object.__new__(Round5)
    controller.current={'protect':True,'eos_threshold':.01}
    controller.protocol={key:None for key in NUMERICAL_KEYS}
    controller.args=SimpleNamespace(target_probe=False)
    controller.diagnostic=True
    calls=[]
    monkeypatch.setattr(Round2,'start_server',lambda self,*args:calls.append((self,args)))
    controller.start_server('candidate',tmp_path)
    assert calls==[(controller,('candidate',tmp_path,False))]
    assert json.loads((tmp_path/'runtime_options.json').read_text())['settings']['eos_threshold']==.01
