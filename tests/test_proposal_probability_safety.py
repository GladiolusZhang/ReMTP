from types import SimpleNamespace

import pytest
import torch

from remtp.dynamic_tree_vllm import _sample_q_token, _sample_branch_q_token


@pytest.mark.parametrize('q', [[float('nan'), 1.], [float('inf'), 1.], [-1., 2.], [0., 0.]])
@pytest.mark.parametrize('temperature', [0., .6])
def test_invalid_primary_and_branch_q_rejected_without_request_rng_change(q, temperature):
    meta = SimpleNamespace(temperature=torch.tensor([temperature]),
                           generators={0:torch.Generator().manual_seed(43)})
    before = meta.generators[0].get_state().clone()
    with pytest.raises(ValueError, match='Invalid MTP proposal Q'):
        _sample_q_token(torch.tensor(q), meta)
    with pytest.raises(ValueError, match='Invalid MTP proposal Q'):
        _sample_branch_q_token(torch.tensor(q), meta, parent_path=(0,), proposal_round=1)
    assert torch.equal(before, meta.generators[0].get_state())


def test_valid_sampling_preserves_original_exponential_race_and_rng():
    q = torch.tensor([.01, .7, .29])
    for seed in range(30):
        old = torch.Generator().manual_seed(seed)
        new = torch.Generator().manual_seed(seed)
        expected = (q / torch.empty_like(q).exponential_(generator=old)).argmax()
        meta = SimpleNamespace(temperature=torch.tensor([.6]), generators={0:new})
        assert int(_sample_q_token(q, meta)[0]) == int(expected)
        assert torch.equal(old.get_state(), new.get_state())
