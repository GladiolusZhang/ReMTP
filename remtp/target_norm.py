"""Keep vLLM's original CUDA RMSNorm reduction below its 256-row switch.

The installed v0.18 kernels change their thread block at 256 tokens. Slicing
into at most 255 rows keeps the small-batch arithmetic and fused in-place
residual semantics, without replacing the norm formula or GPU extension.
"""
import functools


def fixed_rows_norm(original, x, weight, epsilon, residual=None, limit=255):
    if not 1 <= limit < 256:
        raise ValueError('RMSNorm row limit must be below the 256-row dispatch')
    flat=x.view(-1,x.shape[-1])  # view only: a fused call must mutate its input.
    other=None if residual is None else residual.view_as(flat)
    if flat.shape[0] <= limit:
        return original(x,weight,epsilon) if other is None else original(x,residual,weight,epsilon)
    if other is None:
        import torch
        output=torch.empty_like(flat)
        for start in range(0,flat.shape[0],limit):
            output[start:start+limit]=original(flat[start:start+limit],weight,epsilon)
        return output.view_as(x)
    for start in range(0,flat.shape[0],limit):
        original(flat[start:start+limit],other[start:start+limit],weight,epsilon)
    return x,residual


def install_fixed_rows_norm(limit):
    if limit != 255:
        raise ValueError('Validated candidate RMSNorm row limit is 255')
    from vllm.model_executor.layers import layernorm
    for name,fused in (('rms_norm',False),('fused_add_rms_norm',True)):
        original=getattr(layernorm,name)
        if getattr(original,'_remtp_fixed_norm',False):continue
        if fused:
            @functools.wraps(original)
            def wrapped(x,residual,weight,epsilon,_original=original):
                return fixed_rows_norm(_original,x,weight,epsilon,residual,limit)
        else:
            @functools.wraps(original)
            def wrapped(x,weight,epsilon,_original=original):
                return fixed_rows_norm(_original,x,weight,epsilon,limit=limit)
        wrapped._remtp_fixed_norm=True
        setattr(layernorm,name,wrapped)
    print(f'[ReMTP][TargetNumerics] original RMSNorm CUDA kernel, rows <= {limit} (all methods)',flush=True)
