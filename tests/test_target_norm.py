import pytest
import torch

from remtp.target_norm import fixed_rows_norm


def test_fixed_norm_never_crosses_dispatch_and_preserves_fused_aliases():
    calls=[]
    def fused(x,residual,weight,eps):
        calls.append(x.shape[0]);residual.add_(x);x.copy_(residual*weight)
        return x,residual
    x=torch.arange(288*4,dtype=torch.float32).view(288,4)
    residual=torch.ones_like(x);before=x.clone();weight=torch.tensor([1.,2.,3.,4.])
    output,returned=fixed_rows_norm(fused,x,weight,1e-6,residual)
    assert calls==[255,33]
    assert output is x and returned is residual
    assert torch.equal(residual,before+1)
    assert torch.equal(output,(before+1)*weight)


def test_plain_norm_keeps_strided_inputs_and_original_small_call():
    calls=[]
    def norm(x,weight,eps):
        calls.append(x.shape[0]);return x*weight
    source=torch.arange(1024*8,dtype=torch.float32).view(1024,8)
    x=source[::2];before=source.clone();weight=torch.ones(8)*2
    assert torch.equal(fixed_rows_norm(norm,x,weight,1e-6),x*2)
    assert max(calls)<256 and sum(calls)==512
    assert torch.equal(source,before)
    calls.clear();fixed_rows_norm(norm,x[:13],weight,1e-6)
    assert calls==[13]
    with pytest.raises(ValueError,match='below the 256-row'):
        fixed_rows_norm(norm,x,weight,1e-6,limit=256)
