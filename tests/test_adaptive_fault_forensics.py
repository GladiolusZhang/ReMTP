import pytest
from scripts.analyze_adaptive_round5_faults import reconstruct,compare,event_detail


def sample():
    return [dict(generated_length=1,sequential_replay=dict(prefix_ids=[90,91,1]),
        output_ids=[2,3],nodes=[dict(trunk_accepted=True,token_id=2)],terminal='sampled-primary-full-cactus-bonus'),
        dict(generated_length=3,output_ids=[4,99,100],nodes=[dict(trunk_accepted=True,token_id=4),
        dict(trunk_accepted=True,token_id=99)],terminal='sampled-primary-full-cactus-bonus')]


def test_exact_reconstruction_clips_uncommitted_tail_to_api_budget():
    rows=sample();record=dict(output_tokens=4,output='1,2,3,4')
    trace=reconstruct(rows,record,lambda ids:','.join(map(str,ids)))
    assert trace['prompt_ids']==[90,91]
    assert trace['output_ids']==[1,2,3,4]
    assert trace['events'][-1]==(1,0)


@pytest.mark.parametrize('fault',['gap','checkpoint','decoded_text'])
def test_reconstruction_rejects_unproven_histories(fault):
    rows=sample();record=dict(output_tokens=4,output='1,2,3,4')
    if fault=='gap':rows[1]['generated_length']=4
    elif fault=='checkpoint':rows[1]['sequential_replay']=dict(prefix_ids=[90,91,1,2,88])
    else:record['output']='different'
    with pytest.raises(ValueError):reconstruct(rows,record,lambda ids:','.join(map(str,ids)))


def test_first_fork_preserves_exact_common_prefix_and_reports_missing_bonus_probability():
    rows=sample();decode=lambda ids:','.join(map(str,ids))
    left=reconstruct(rows,dict(output_tokens=4,output='1,2,3,4'),decode)
    other=sample();other[0]['output_ids']=[2,8]
    right=reconstruct(other,dict(output_tokens=4,output='1,2,8,4'),decode)
    result=compare(left,right,rows,other,decode)
    assert result['common_output_tokens']==2
    assert result['common_prefix_ids']==[90,91,1,2]
    assert 'not in scalar log' in result['right']['decision']['missing']
    assert result['right']['decision']['role']=='unmodified Target bonus'
