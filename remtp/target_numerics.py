"""Optional consistent BNB FP4 compute route on Jetson sm87.

BNB 0.50 selects fused custom GEMM for tiny M but dequant+BF16 linear for
larger M on sm87. This switch retains FP4 storage and uses the latter route
for all M; only one layer's temporary weight is materialized at a time.
"""
import functools
import os


def install_target_numerics():
    head_block=int(os.getenv('REMTP_TARGET_LM_HEAD_ROW_BLOCK','0'))
    if head_block:
        from remtp.target_head import install_fixed_lm_head
        install_fixed_lm_head(head_block)
    norm_limit=int(os.getenv('REMTP_TARGET_RMSNORM_ROW_LIMIT','0'))
    if norm_limit:
        from remtp.target_norm import install_fixed_rows_norm
        install_fixed_rows_norm(norm_limit)
    if os.getenv('REMTP_CANONICAL_TREE_ATTENTION','0')=='1':
        from remtp.canonical_tree_attention import install_canonical_tree_attention
        install_canonical_tree_attention()
    if os.getenv('REMTP_TARGET_ATTENTION_FP32','0')=='1':
        from vllm.v1.attention.ops import triton_unified_attention
        from remtp.target_attention_fp32 import kernel_unified_attention_2d
        triton_unified_attention.kernel_unified_attention_2d=kernel_unified_attention_2d
        print('[ReMTP][TargetNumerics] unified 2D probability/value product FP32 (all methods)',flush=True)
    if os.getenv('REMTP_TARGET_TRITON_2D','0')=='1':
        from vllm.v1.attention.backends import triton_attn
        original_attention=triton_attn.unified_attention
        if not getattr(original_attention,'_remtp_2d',False):
            @functools.wraps(original_attention)
            def attention(*args,**kwargs):
                # Supported launch selector: no segmented 3D softmax.
                # Query lengths, positions, masks and tensors are untouched.
                kwargs['seq_threshold_3D']=None
                kwargs['num_par_softmax_segments']=None
                return original_attention(*args,**kwargs)
            attention._remtp_2d=True
            triton_attn.unified_attention=attention
        print('[ReMTP][TargetNumerics] TRITON_ATTN unified 2D path, matching TREE_ATTN',flush=True)
    if os.getenv('REMTP_TARGET_BF16_FULL_ACCUM','0')=='1':
        import torch
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
        print('[ReMTP][TargetNumerics] BF16 GEMM reduced precision reduction disabled (all methods)',flush=True)
    install_uniform_bnb_gemm()
    block=int(os.getenv('REMTP_TARGET_BNB_ROW_BLOCK','0'))
    if block:
        if block not in (16,64):raise ValueError('BNB row block must be 0,16,64')
        import torch
        from bitsandbytes.backends.cuda import ops
        original=ops._dequant_linear_fallback
        if getattr(original,'_remtp_rows',False):return

        @functools.wraps(original)
        def fixed(a,b,shape,scales,size,kind,bias=None,absmax_8bit=None,absmax_code=None,absmax_offset=None):
            if torch.cuda.get_device_capability(a.device.index)!=(8,7):
                return original(a,b,shape,scales,size,kind,bias,absmax_8bit,absmax_code,absmax_offset)
            if absmax_8bit is not None:
                dq=torch.empty_like(absmax_8bit,dtype=torch.float32)
                ops._dequantize_blockwise_impl(absmax_8bit,scales,absmax_code,256,torch.float32,out=dq)
                scales=dq+absmax_offset
            weight=torch.empty(shape,dtype=a.dtype,device=a.device)
            ops._dequantize_4bit_impl(b,scales,size,kind,a.dtype,out=weight)
            return fixed_rows_linear(a,weight,bias,block)

        fixed._remtp_rows=True
        ops._dequant_linear_fallback=fixed
        print(f'[ReMTP][TargetNumerics] fixed FP4 linear row block={block}',flush=True)


def fixed_rows_linear(x,weight,bias=None,block=64):
    """Fixed GEMM shape, independent of the caller's number of input rows."""
    import torch
    leading=x.shape[:-1];flat=x.reshape(-1,x.shape[-1])
    output=torch.empty(flat.shape[0],weight.shape[0],dtype=x.dtype,device=x.device)
    for start in range(0,flat.shape[0],block):
        count=min(block,flat.shape[0]-start)
        padded=torch.zeros(block,flat.shape[-1],dtype=x.dtype,device=x.device)
        padded[:count]=flat[start:start+count]
        output[start:start+count]=torch.nn.functional.linear(padded,weight,bias)[:count]
    return output.reshape(*leading,weight.shape[0])


def install_uniform_bnb_gemm():
    if os.getenv('REMTP_TARGET_BNB_UNIFORM_GEMM','0')!='1':return
    import torch
    from bitsandbytes.backends.cuda import ops
    original=ops._gemm_4bit_use_custom_fn
    if getattr(original,'_remtp_uniform',False):return

    @functools.wraps(original)
    def selected(device_index,dtype,m,n,k):
        if torch.cuda.get_device_capability(device_index)==(8,7):return False
        return original(device_index,dtype,m,n,k)

    selected._remtp_uniform=True
    ops._gemm_4bit_use_custom_fn=selected
    print('[ReMTP][TargetNumerics] sm87 FP4: uniform dequant + BF16 linear (all methods)',flush=True)
