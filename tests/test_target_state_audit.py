import pytest
import torch

from remtp.target_diagnostics import crossed_checkpoints


def test_checkpoint_crossings_cover_skipped_lengths_without_repeating():
    checkpoints=(0,1,31,32,127,128,511,512,1023,1024,2040)
    assert crossed_checkpoints(0,-1,checkpoints)==(0,)
    assert crossed_checkpoints(34,30,checkpoints)==(31,32)
    assert crossed_checkpoints(132,126,checkpoints)==(127,128)
    assert crossed_checkpoints(136,132,checkpoints)==()
    assert crossed_checkpoints(1026,1020,checkpoints)==(1023,1024)

from remtp.target_state_audit import logical_slots,logical_kv_snapshot,row_snapshot
from scripts.run_target_state_audit import reference_state_path


def test_fused_norm_input_snapshot_survives_in_place_writes():
    hidden=torch.arange(12.).reshape(3,4)
    residual=hidden.clone()+10
    record=row_snapshot(((hidden,residual),{'position':torch.tensor([9,10,11])}),1,3)
    hidden.add_(100);residual.zero_()
    assert torch.equal(record[0][0],torch.tensor([4.,5.,6.,7.]))
    assert torch.equal(record[0][1],torch.tensor([14.,15.,16.,17.]))
    assert int(record[1]['position'])==10


def test_logical_kv_uses_permuted_blocks_and_exact_visible_prefix():
    table=torch.tensor([[3,1,4]])
    slots=logical_slots(table,5,2)
    assert slots.tolist()==[6,7,2,3,8]
    cache=torch.arange(20).reshape(2,10)
    assert cache[:,slots].tolist()==[[6,7,2,3,8],[16,17,12,13,18]]
    assert table.tolist()==[[3,1,4]]
    with pytest.raises(ValueError,match='exceeds'):logical_slots(table,7,2)


def test_tree_and_triton_physical_layouts_have_same_logical_kv():
    tree=torch.arange(40.).reshape(2,5,2,1,2)
    triton=tree.permute(1,0,2,3,4).contiguous()
    table=torch.tensor([[3,1,4]])
    a,slots=logical_kv_snapshot(tree,table,5,'TREE_ATTN')
    b,_=logical_kv_snapshot(triton,table,5,'TRITON_ATTN')
    assert torch.equal(a,b)
    assert a.shape==(2,5,1,2)
    assert slots.tolist()==[6,7,2,3,8]
    tree.zero_();triton.zero_()
    assert float(a.max())==37 and torch.equal(a,b)


def test_reference_pairs_prompt_logprobs_with_exact_call_not_sampling(tmp_path):
    path=tmp_path/'target_probe'/'req';path.mkdir(parents=True)
    (path/'state_00.pt').write_bytes(b'sampling')
    (path/'state_01.pt').write_bytes(b'prompt-logprobs')
    assert reference_state_path(tmp_path,dict(request='req',calls=2))==path/'state_01.pt'
    with pytest.raises(ValueError,match='exact reference'):reference_state_path(tmp_path,dict(request='req',calls=3))


def test_backend_contract_disambiguates_two_block_cache():
    tree=torch.arange(16.).reshape(2,2,2,1,2)
    triton=tree.permute(1,0,2,3,4).contiguous()
    table=torch.tensor([[1,0]])
    a,_=logical_kv_snapshot(tree,table,4,'TREE_ATTN')
    b,_=logical_kv_snapshot(triton,table,4,'TRITON_ATTN')
    assert torch.equal(a,b)
