"""Opt-in fixed16 GEMM for the unquantized BF16 output projection."""
import functools


def install_fixed_lm_head(block):
    if block!=16:raise ValueError('Validated LMHead row block is 16')
    import torch
    from vllm.model_executor.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod,ParallelLMHead
    from remtp.target_numerics import fixed_rows_linear
    original=UnquantizedEmbeddingMethod.apply
    if getattr(original,'_remtp_fixed_head',False):return

    @functools.wraps(original)
    def apply(self,layer,x,bias=None):
        if not isinstance(layer,ParallelLMHead):return original(self,layer,x,bias)
        if not x.is_cuda or x.dtype!=torch.bfloat16 or layer.weight.dtype!=torch.bfloat16:
            raise ValueError('Fixed LMHead candidate requires CUDA BF16 hidden/weight')
        if not getattr(layer,'_remtp_fixed_head_checked',False):
            print(f'[ReMTP][TargetNumerics] actual BF16 LMHead fixed16: weight={tuple(layer.weight.shape)}',flush=True)
            layer._remtp_fixed_head_checked=True
        return fixed_rows_linear(x,layer.weight,bias,block)

    apply._remtp_fixed_head=True
    UnquantizedEmbeddingMethod.apply=apply
    print('[ReMTP][TargetNumerics] unquantized LMHead fixed16 rows (all methods)',flush=True)
