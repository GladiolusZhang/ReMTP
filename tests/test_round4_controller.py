import json

import pytest
import torch

from scripts.run_adaptive_round4 import exact_reference,compare_raw,numerical_source_check,ROOT


def test_chunked_echo_picks_actual_prompt_logprob_instead_of_sampling():
    prefix=list(range(2049));eos=9999
    records=[dict(request='r',prompt_ids=tuple(prefix)+(eos,),actual_positions=(2048,2049),
        rows=1,calls=1,last_logits=torch.tensor([10.,0.])),
        dict(request='r',prompt_ids=tuple(prefix)+(eos,),actual_positions=(2048,2049),
        rows=1,calls=2,last_logits=torch.tensor([0.,10.]))]
    assert exact_reference(records,prefix,eos)['calls']==2
    wrong=[{**records[1],'actual_positions':(2047,2048)}]
    with pytest.raises(ValueError,match='Missing final Target'):exact_reference(wrong,prefix,eos)
    with pytest.raises(ValueError,match='Duplicate Target'):
        exact_reference(records+[{**records[1],'request':'other'}],prefix,eos)


def test_gate_checks_actual_draft_even_if_not_in_either_top_eight():
    reference=torch.arange(20,dtype=torch.float32)
    changed=reference.clone();changed[1]+=.25
    result=compare_raw(changed,reference,[1],0)
    assert 1 in result['compared_ids'] and not result['passed']
    assert compare_raw(reference,reference,[],0)['passed']


def test_echo_boundary_root_is_in_previous_full_chunk_before_eos_only_chunk():
    prefix=list(range(2048));eos=9999
    base=dict(request='r',prompt_ids=tuple(prefix)+(eos,))
    records=[dict(**base,actual_positions=tuple(range(2048)),rows=1,calls=1),
        dict(**base,actual_positions=tuple(range(2048)),rows=2048,calls=2),
        dict(**base,actual_positions=(2048,),rows=1,calls=3)]
    assert exact_reference(records,prefix,eos)['calls']==2


def test_single_final_row_replay_ignores_earlier_chunk_and_generated_rows():
    prefix=list(range(2049))
    base=dict(request='r',prompt_ids=tuple(prefix),rows=1,output_ids=())
    records=[dict(**base,actual_positions=tuple(range(2048)),calls=1),
        dict(**base,actual_positions=(2048,),calls=2),
        {**base,'actual_positions':(2049,),'output_ids':(99,),'calls':3}]
    assert exact_reference(records,prefix,9999)['calls']==2


def test_repair_cannot_reuse_old_norm_protocol(tmp_path):
    old=dict(source_root=str(ROOT),target_rmsnorm_row_limit=0)
    (tmp_path/'protocol.json').write_text(json.dumps(old))
    with pytest.raises(ValueError,match='Numerical source protocol mismatch'):
        numerical_source_check(tmp_path,{**old,'target_rmsnorm_row_limit':255})
    old['target_lm_head_row_block']=0
    (tmp_path/'protocol.json').write_text(json.dumps(old))
    with pytest.raises(ValueError,match='Numerical source protocol mismatch'):
        numerical_source_check(tmp_path,{**old,'target_lm_head_row_block':16})
