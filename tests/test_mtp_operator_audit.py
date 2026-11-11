"""Probe inputs must survive in-place kernels; hooks must not leak."""
import pytest
import torch

from remtp.mtp_operator_audit import audit_pass, nonfinite


class Inplace(torch.nn.Module):
    def forward(self, x):
        x.fill_(float('nan'))
        return x


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_layernorm = Inplace()

    def forward(self, x):
        return self.hidden_layernorm(x)


@pytest.mark.parametrize('raise_inside', [False, True])
def test_snapshot_precedes_inplace_mutation_and_hooks_removed(raise_inside):
    model = Model()
    detail = {}
    rng = torch.get_rng_state().clone()
    try:
        with audit_pass(model, detail):
            model(torch.ones(1, 8))
            if raise_inside:
                raise RuntimeError('simulated failure')
    except RuntimeError:
        assert raise_inside
    saved = detail['operators'][0]
    assert saved['nonfinite_inputs'] == 0
    assert saved['nonfinite_output'] == 8
    assert torch.equal(saved['inputs'][0][0], torch.ones(1, 8))
    assert not model.hidden_layernorm._forward_pre_hooks
    assert not model.hidden_layernorm._forward_hooks
    assert torch.equal(rng, torch.get_rng_state())
    model(torch.ones(1, 8))
    assert len(detail['operators']) == 1


def test_nonfinite_tuple_ignores_optional_bias_and_integer_indices():
    assert nonfinite((torch.tensor([float('nan'), float('inf'), 1.]),
                      None, {'ids': torch.tensor([0, 1])})) == 2
