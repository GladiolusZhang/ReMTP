"""Bounded opt-in Target operation/KV observation; never writes model state."""
import functools
import json
import os
from pathlib import Path

import torch


def row_snapshot(value, row, count):
    """Copy before an in-place fused norm can overwrite its input."""
    if isinstance(value, torch.Tensor):
        selected=value[row] if value.ndim and value.shape[0]==count else value
        return selected.detach().cpu().clone()
    if isinstance(value,(tuple,list)):
        return tuple(row_snapshot(v,row,count) for v in value)
    if isinstance(value,dict):
        return {k:row_snapshot(v,row,count) for k,v in value.items()}
    return value


def logical_slots(table, length, block_size):
    if table.ndim!=2 or table.shape[0]!=1 or length<1:
        raise ValueError('State audit requires one nonempty logical sequence')
    positions=torch.arange(length,device=table.device)
    if (length-1)//block_size>=table.shape[1]:
        raise ValueError('State audit logical length exceeds block table')
    return (table[0,positions//block_size]*block_size+positions%block_size).long()


def logical_kv_snapshot(cache,table,length,backend):
    """Gather only visible tokens; never permute/flatten a full GPU cache."""
    if cache.ndim!=5:raise ValueError('Unexpected Target KV dimensionality')
    slots=logical_slots(table,length,cache.shape[2])
    blocks=slots//cache.shape[2];offsets=slots%cache.shape[2]
    if backend=='TREE_ATTN' and cache.shape[0]==2:
        selected=cache[:,blocks,offsets]
    elif backend=='TRITON_ATTN' and cache.shape[1]==2:
        selected=cache[blocks,:,offsets].permute(1,0,2,3)
    else:raise ValueError(f'Unsupported Target KV layout: {backend}/{tuple(cache.shape)}')
    return selected.detach().cpu().clone(),slots


def install_state_audit(current, directory, reference_echo=False):
    """Called after tree hooks are installed; observations scoped to MiMoModel."""
    from vllm.model_executor.models.mimo import MiMoModel
    from vllm.forward_context import get_forward_context
    from remtp import fixed6_vllm as fixed
    original=MiMoModel.forward
    if getattr(original,'_remtp_state_audit',False):return
    active={}

    def hooked(model):
        if getattr(model,'_remtp_audit_hooks',False):return
        seen=set()
        def register(module,kind,layer):
            if id(module) in seen:return
            seen.add(id(module))
            def before(_module,args,kwargs):
                if not active:return
                key=f"{active['layer'] if kind=='rope' else layer}:{kind}"
                active['ops'][key]={'inputs':row_snapshot((args,kwargs),active['row'],active['count'])}
            def after(_module,args,kwargs,output):
                if not active:return
                key=f"{active['layer'] if kind=='rope' else layer}:{kind}"
                active['ops'][key]['output']=row_snapshot(output,active['row'],active['count'])
            module.register_forward_pre_hook(before,with_kwargs=True)
            module.register_forward_hook(after,with_kwargs=True)
        for index,layer in enumerate(model.layers):
            def layer_before(_module,_args,index=index):
                if active:active['layer']=index
            layer.register_forward_pre_hook(layer_before)
            for name,module in (
                ('input_norm',layer.input_layernorm),('qkv',layer.self_attn.qkv_proj),
                ('rope',layer.self_attn.rotary_emb),('attention',layer.self_attn.attn),
                ('o_proj',layer.self_attn.o_proj),('post_norm',layer.post_attention_layernorm),
                ('gate_up',layer.mlp.gate_up_proj),('activation',layer.mlp.act_fn),
                ('down_proj',layer.mlp.down_proj),('layer',layer)):
                register(module,name,index)
        model._remtp_audit_hooks=True

    @functools.wraps(original)
    def forward(self,input_ids,positions,*args,**kwargs):
        if current and not getattr(self,'_remtp_audit_contract_checked',False):
            context=get_forward_context()
            for layer in self.layers:
                attn=layer.self_attn.attn
                metadata=context.attn_metadata[attn.layer_name]
                for name in ('block_table','slot_mapping','seq_lens','query_start_loc'):
                    if not isinstance(getattr(metadata,name),torch.Tensor):
                        raise ValueError(f'State audit metadata contract mismatch: {name}')
                cache=attn.kv_cache[context.virtual_engine]
                backend=attn.attn_backend.get_name()
                if (cache.ndim!=5 or backend not in ('TREE_ATTN','TRITON_ATTN') or
                    cache.shape[0 if backend=='TREE_ATTN' else 1]!=2):
                    raise ValueError('State audit requires ordinary Target KV tensors')
            self._remtp_audit_contract_checked=True
        if not current.get('capture_selected',False):
            return original(self,input_ids,positions,*args,**kwargs)
        hooked(self)
        tree=bool(fixed._RUNTIME.target_tree_active)
        requested=int(os.getenv('REMTP_TARGET_STATE_ROW_POSITION','-1'))
        if requested>=0:
            positions_list=positions.tolist()
            if requested not in positions_list:
                return original(self,input_ids,positions,*args,**kwargs)
            row=positions_list.index(requested);length=requested+1
        elif tree:
            length=len(current['prompt_ids'])+len(current['output_ids']);row=0
        else:
            row=len(positions)-2 if reference_echo else len(positions)-1
            length=int(positions[row])+1
        if row<0 or int(positions[row])!=length-1:
            raise ValueError('State audit root position is not aligned')
        active.update(row=row,count=len(positions),layer=0,ops={})
        record=dict(prefix_ids=current['prompt_ids']+current['output_ids'] if tree and requested<0 else
                    tuple(input_ids[:length].tolist()),logical_length=length,root_row=row,
                    input_ids=tuple(input_ids.tolist()),positions=tuple(positions.tolist()),
                    tree=tree,ops=active['ops'],kv={},metadata={})
        try:
            output=original(self,input_ids,positions,*args,**kwargs)
            record['root_model_output']=row_snapshot(output,row,len(positions))
            context=get_forward_context()
            for index,layer in enumerate(self.layers):
                attn=layer.self_attn.attn;cache=attn.kv_cache[context.virtual_engine]
                metadata=context.attn_metadata[attn.layer_name]
                record['kv'][index],slots=logical_kv_snapshot(cache,metadata.block_table,
                    length,attn.attn_backend.get_name())
                if index==0:
                    record['metadata']={name:getattr(metadata,name).detach().cpu().clone()
                        for name in ('block_table','slot_mapping','seq_lens','query_start_loc')}
                    record['metadata']['logical_slots']=slots.detach().cpu().clone()
            current['state_audit']=record
            return output
        finally:active.clear()

    forward._remtp_state_audit=True
    MiMoModel.forward=forward
    copy=fixed._copy_cache_slots
    if not getattr(copy,'_remtp_copy_audit',False):
        @functools.wraps(copy)
        def copied(module,sources,destinations):
            if not current.get('capture_selected',False) or not sources.numel():
                return copy(module,sources,destinations)
            cache=module.kv_cache[0].flatten(1,2)
            if not cache.numel():return copy(module,sources,destinations)
            expected=cache[:,sources.long()].detach().cpu().clone()
            result=copy(module,sources,destinations)
            actual=cache[:,destinations.long()].detach().cpu().clone()
            record=dict(request=current.get('request',''),generated_length=len(current.get('output_ids',())),
                layer=module.layer_name,sources=sources.tolist(),destinations=destinations.tolist(),
                selected_nodes=list(fixed._RUNTIME.selected_nodes),
                identical=bool(torch.equal(expected,actual)),max_error=float((expected.float()-actual.float()).abs().max()))
            path=Path(directory)/current['request'];path.mkdir(parents=True,exist_ok=True)
            with (path/'commit_audit.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            return result
        copied._remtp_copy_audit=True
        fixed._copy_cache_slots=copied
    print('[ReMTP][StateAudit] bounded Target operations, logical KV and commit copies',flush=True)
