import json

import pytest

from scripts.investigate_target_consistency import manifest,load_explicit_manifest
from remtp.target_numerics import fixed_rows_linear
from scripts.validate_target_tree_consistency import comparison
from scripts.validate_target_tree_consistency import validate_reuse, ROOT
import torch


def source(tmp_path):
    path=tmp_path/'diagnostics/anchor4/prefix_diagnostic.jsonl'
    path.parent.mkdir(parents=True)
    rows=[]
    for length,generated in [(286,130),(109,1),(157,1),(236,128)]:
        rows.append(dict(request=f'req{length}',generated_length=generated,
                         sequential_replay=dict(prefix_ids=[length]*length,
                         top_ids=[1,2],top_probs=[.6,.4],eos_ids=[9],eos_probs=[.01])))
    path.write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
    return path,rows


def test_manifest_keeps_original_cases_despite_new_diagnostic_directory(tmp_path):
    _,rows=source(tmp_path)
    extra=tmp_path/'diagnostics/strict_delta0/prefix_diagnostic.jsonl'
    extra.parent.mkdir();extra.write_text(json.dumps(rows[0])+'\n')
    cases=manifest(tmp_path)
    assert [len(r['prefix_ids']) for r in cases]==[109,236,157,286]
    assert len({r['prefix_sha256'] for r in cases})==4
    assert all('anchor4' in r['source'] for r in cases)


def test_ambiguous_source_requires_explicit_resolution(tmp_path):
    path,rows=source(tmp_path)
    with path.open('a') as f:f.write(json.dumps(rows[0])+'\n')
    with pytest.raises(ValueError,match='Ambiguous'):manifest(tmp_path)


def test_new_failure_manifest_keeps_actual_prefix_and_rejects_token_changes(tmp_path):
    source(tmp_path);cases=manifest(tmp_path)
    path=tmp_path/'new_failure_manifest.json';path.write_text(json.dumps(cases))
    assert load_explicit_manifest(path)==cases
    cases[-1]['prefix_ids'][-1]+=1;path.write_text(json.dumps(cases))
    with pytest.raises(ValueError,match='prefix hash mismatch'):load_explicit_manifest(path)


def test_fixed_linear_preserves_rows_bias_and_isolates_unrelated_inputs():
    rng=torch.Generator().manual_seed(27)
    weight=torch.randn(11,8,generator=rng);bias=torch.randn(11,generator=rng)
    x=torch.randn(2,57,8,generator=rng)
    actual=fixed_rows_linear(x,weight,bias,16)
    assert actual.shape==(2,57,11)
    torch.testing.assert_close(actual,torch.nn.functional.linear(x,weight,bias))
    changed=x.clone();changed[1]=123
    assert torch.equal(actual[0],fixed_rows_linear(changed,weight,bias,16)[0])
    single=fixed_rows_linear(x.reshape(-1,8)[:1],weight,bias,16)
    assert torch.equal(single[0],actual[0,0])


def test_empty_input_keeps_shape():
    assert fixed_rows_linear(torch.zeros(0,8),torch.ones(11,8),block=16).shape==(0,11)


def test_probability_gate_ignores_common_offset_but_rejects_relative_error():
    x=torch.tensor([1.,2.,-1.,-10.])
    assert comparison(x,x+8,[1,0,2],3)['passed']
    changed=x.clone();changed[0]+=.2
    assert not comparison(x,changed,[1,0,2],3)['passed']
    changed=x.clone();changed[3]+=.2
    assert not comparison(x,changed,[1,0,2],3)['passed']


def test_root_reuse_requires_same_numerics_and_only_passed_cache_off(tmp_path):
    source=tmp_path/'tree';native=tmp_path/'native'
    protocol=dict(source_root=str(ROOT),target_bnb_row_block=64,target_reference_backend='TRITON_ATTN')
    cases=[dict(prefix_ids=[1,2,3])]
    for directory in (source,native):
        directory.mkdir()
        (directory/'protocol.json').write_text(json.dumps(protocol))
        (directory/'prefix_manifest.json').write_text(json.dumps(cases))
    checks=[dict(cache=0,passed=True) for _ in range(24)]+[dict(cache=1,passed=False)]
    (source/'root_gate.json').write_text(json.dumps(dict(checks=checks,passed=False)))
    validate_reuse(source,native,protocol,cases)
    with pytest.raises(ValueError,match='protocol mismatch'):
        validate_reuse(source,native,{**protocol,'target_bnb_row_block':16},cases)
    checks[0]['passed']=False
    (source/'root_gate.json').write_text(json.dumps(dict(checks=checks)))
    with pytest.raises(ValueError,match='24 cache-off root checks'):
        validate_reuse(source,native,protocol,cases)
