"""Read tree ancestors in canonical token order without copying any KV data.

Each query is a virtual one-token sequence. A token-level block table maps its
committed prefix and ancestors to the original physical BFS cache slots. This
uses the existing unified attention kernel, with the same arithmetic as a
sequential Target request; the original cache and slot mapping are untouched.
"""
import functools


def canonical_tree_table(block_table, context, choices, block_size, tile=32):
    import torch
    if block_table.shape[0]!=1 or context<0:
        raise ValueError('Canonical tree attention requires one valid prefix')
    paths=[(),*map(tuple,choices)]
    if len(set(paths))!=len(paths):raise ValueError('Duplicate tree paths')
    indices={p:i for i,p in enumerate(paths)}
    depth=max(map(len,paths));width=((context+depth+1+tile-1)//tile)*tile
    ancestors=[]
    for path in paths:
        row=[0]+[indices[path[:n]] for n in range(1,len(path)+1)]
        ancestors.append(row+[0]*(depth+1-len(row)))
    mapping=torch.tensor(ancestors,device=block_table.device,dtype=torch.long)
    logical=torch.arange(width,device=block_table.device)
    tail=(logical-context).clamp(0,depth)
    physical=torch.where(logical[None,:]<context,logical[None,:],context+mapping[:,tail])
    blocks=block_table.expand(len(paths),-1).gather(1,physical//block_size)
    token_table=(blocks*block_size+physical%block_size).to(block_table.dtype)
    lengths=torch.tensor([context+len(p)+1 for p in paths],device=block_table.device,dtype=torch.int32)
    starts=torch.arange(len(paths)+1,device=block_table.device,dtype=torch.int32)
    return token_table,lengths,starts,context+depth+1


def install_canonical_tree_attention():
    from vllm.v1.attention.backends import tree_attn
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    if getattr(tree_attn.unified_attention,'_remtp_canonical',False):return
    original=tree_attn.unified_attention;prepare=GPUModelRunner._prepare_inputs
    layouts={}

    @functools.wraps(prepare)
    def prepared(self,*args,**kwargs):
        # Reuse layouts across layers only. A new model step/request may change
        # physical blocks in place, even when all tensor pointers stay the same.
        layouts.clear()
        return prepare(self,*args,**kwargs)

    @functools.wraps(original)
    def attention(*args,**kwargs):
        from remtp.fixed6_vllm import _RUNTIME
        q=kwargs.get('q');bias=kwargs.get('qq_bias')
        if (not _RUNTIME.target_tree_active or bias is None or q is None or
                q.shape[0]!=len(_RUNTIME.topology.nodes)+1):
            return original(*args,**kwargs)
        if args:raise ValueError('Canonical tree attention requires named arguments')
        k=kwargs['k'];v=kwargs['v'];table=kwargs['block_table'];count=q.shape[0]
        if table.shape[0]!=1:raise ValueError('Canonical tree attention supports one request')
        context=kwargs['max_seqlen_k']-count
        choices=_RUNTIME.topology.choices
        key=(table.data_ptr(),context,tuple(choices),k.shape[1])
        if key not in layouts:
            layouts[key]=canonical_tree_table(table,context,choices,k.shape[1])
        token_table,lengths,starts,maximum=layouts[key]
        # view() must share the original storage. No reshape fallback that
        # could accidentally materialize a full cache copy is permitted.
        mapped=dict(kwargs,k=k.view(-1,1,k.shape[2],k.shape[3]),
                    v=v.view(-1,1,v.shape[2],v.shape[3]),block_table=token_table,
                    seqused_k=lengths,cu_seqlens_q=starts,max_seqlen_q=1,
                    max_seqlen_k=maximum,qq_bias=None,
                    seq_threshold_3D=None,num_par_softmax_segments=None)
        return original(**mapped)

    attention._remtp_canonical=True
    tree_attn.unified_attention=attention
    GPUModelRunner._prepare_inputs=prepared
    print('[ReMTP][TargetNumerics] canonical ancestor KV reads, zero KV copies',flush=True)
