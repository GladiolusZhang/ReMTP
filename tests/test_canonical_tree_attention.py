import torch

from remtp.canonical_tree_attention import canonical_tree_table


def test_canonical_paths_use_exact_prefix_and_ancestors_with_permuted_blocks():
    choices=((0,),(1,),(0,0),(1,0),(0,0,0))
    original=torch.tensor([[4,1,7]],dtype=torch.int32)
    table,lengths,starts,maximum=canonical_tree_table(original,5,choices,4)
    physical=lambda pos:int(original[0,pos//4])*4+pos%4
    expected_paths=((),(0,),(1,),(0,0),(1,0),(0,0,0))
    for row,path in enumerate(expected_paths):
        ancestor_rows=[0]+[expected_paths.index(path[:n]) for n in range(1,len(path)+1)]
        expected=[physical(p) for p in range(5)]+[physical(5+p) for p in ancestor_rows]
        assert table[row,:lengths[row]].tolist()==expected
    assert starts.tolist()==list(range(7))
    assert maximum==9 and table.shape==(6,32)
    assert original.tolist()==[[4,1,7]]


def test_siblings_and_rejected_cache_slots_cannot_change_canonical_visible_values():
    choices=((0,),(1,),(0,0))
    table,lengths,_,_=canonical_tree_table(torch.tensor([[0,1,2]]),5,choices,4)
    values=torch.arange(12,dtype=torch.float32)
    # Primary leaf reads root, primary and its own slot, excluding sibling.
    selected=table[3,:lengths[3]]
    before=values[selected].clone()
    values[7]=1000;values[9:]=2000
    assert torch.equal(values[selected],before)
    assert 7 not in selected and 9 not in selected


def test_table_padding_covers_unmasked_kernel_tile_reads_at_boundary():
    table,lengths,_,_=canonical_tree_table(torch.arange(12)[None,:],31,((0,),(0,0),(0,0,0)),4)
    assert table.shape==(4,64)
    assert lengths.tolist()==[32,33,34,35]
    assert table.max()<48 and table.min()>=0
