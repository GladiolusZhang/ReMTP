"""Opt-in bounded Target hidden/logit probes; no sampling or cache mutation."""
import functools
import json
import os
from pathlib import Path

import torch


def crossed_checkpoints(generated,previous,checkpoints):
    return tuple(n for n in checkpoints if previous<n<=generated)


def install_target_diagnostics():
    directory=os.getenv('REMTP_TARGET_DIAGNOSTIC_DIR','')
    if not directory:return
    from vllm.model_executor.models.mimo import MiMoForCausalLM
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    if getattr(MiMoForCausalLM.compute_logits,'_target_probe',False):return
    compute=MiMoForCausalLM.compute_logits
    prepare=GPUModelRunner._prepare_inputs
    current={}
    minimum=int(os.getenv('REMTP_TARGET_PROBE_MIN_GENERATED','0'))
    min_prompt=int(os.getenv('REMTP_TARGET_PROBE_MIN_PROMPT_TOKENS','0'))
    maximum=int(os.getenv('REMTP_TARGET_PROBE_MAX_GENERATED','2147483647'))
    limit=int(os.getenv('REMTP_TARGET_PROBE_MAX_FRAMES','32'))
    checkpoints=tuple(int(n) for n in os.getenv('REMTP_TARGET_PROBE_CHECKPOINTS','').split(',') if n)
    manifest=os.getenv('REMTP_TARGET_PROBE_PREFIX_MANIFEST','')
    cases=json.loads(Path(manifest).read_text())['cases'] if manifest else None
    shape_rows=tuple(int(n) for n in os.getenv('REMTP_TARGET_PROBE_SHAPE_ROWS','1,10,13').split(',') if n)
    if any(n not in (1,10,13) for n in shape_rows):
        raise ValueError('Unsupported Target projection shape probe')
    if checkpoints and (min(checkpoints)<0 or tuple(sorted(set(checkpoints)))!=checkpoints):
        raise ValueError('Target probe checkpoints must be sorted unique nonnegative lengths')
    if minimum<0 or min_prompt<0 or maximum<minimum or not 1<=limit<=64:
        raise ValueError('Invalid bounded Target probe selection')
    if os.getenv('REMTP_TARGET_STATE_AUDIT','0')=='1':
        from remtp.target_state_audit import install_state_audit
        install_state_audit(current,directory,
            reference_echo=os.getenv('REMTP_TARGET_STATE_REFERENCE_ECHO','0')=='1')

    @functools.wraps(prepare)
    def prepared(self,*args,**kwargs):
        ids=self.input_batch.req_ids
        if len(ids)==1:
            state=self.requests[ids[0]]
            if current.get('request')!=ids[0]:current.update(calls=0,saved=0,last_checkpoint_generated=-1)
            current.update(request=ids[0],prompt_ids=tuple(state.prompt_token_ids or ()),
                           output_ids=tuple(state.output_token_ids))
            current.pop('state_audit',None)
            generated=len(current['output_ids'])
            current['selected_checkpoints']=crossed_checkpoints(generated,current['last_checkpoint_generated'],checkpoints)
            current['capture_selected']=(len(current['prompt_ids'])>=min_prompt and
                minimum<=generated<=maximum and current['saved']<limit and
                (not checkpoints or bool(current['selected_checkpoints'])))
            if cases is not None:
                matches=[c for c in cases if tuple(c['prompt_ids'])==current['prompt_ids']
                         and tuple(c['output_ids'])==current['output_ids']]
                current['capture_selected'] &= len(matches)==1
                current['case_label']=matches[0]['label'] if len(matches)==1 else None
        else:current.clear()
        result=prepare(self,*args,**kwargs)
        if current.get('capture_selected',False):
            count=int(self.query_start_loc.np[len(ids)])
            current['actual_positions']=tuple(self.positions.gpu[:count].tolist())
            # Chunked prefill can project a provisional row before the prompt
            # is complete. Exact sequential-prefix cases need the final row;
            # an earlier chunk must not consume the single-frame allowance.
            if (cases is not None and not current['output_ids'] and
                    (not current['actual_positions'] or
                     current['actual_positions'][-1]!=len(current['prompt_ids'])-1)):
                current['capture_selected']=False
            else:
                current['actual_input_ids']=tuple(self.input_ids.gpu[:count].tolist())
        return result

    @functools.wraps(compute)
    def computed(self,hidden):
        result=compute(self,hidden)
        call=current.get('calls',0)
        if result is None or not current:return result
        current['calls']=call+1
        if not current.get('capture_selected',False):return result
        current['saved']+=1
        current['last_checkpoint_generated']=len(current['output_ids'])
        record={**{k:v for k,v in current.items() if k!='state_audit'},'rows':hidden.shape[0],'dtype':str(result.dtype),
                'first_hidden':hidden[0].detach().cpu().clone(),
                'last_hidden':hidden[-1].detach().cpu().clone(),
                'first_logits':result[0].detach().cpu().clone(),
                'last_logits':result[-1].detach().cpu().clone(),'shape_probes':{}}
        from remtp import fixed6_vllm as fixed
        tree=bool(fixed._RUNTIME.target_tree_active)
        index=0 if tree else -1
        record['probe_row']=index
        if tree:
            record['topology']=fixed._RUNTIME.topology.choices
            record['draft_ids']=fixed._RUNTIME.draft_token_ids.detach().cpu().clone()
            record['logits_rows']=result.detach().cpu().clone()
            record['hidden_rows']=hidden.detach().cpu().clone()
        # Same actual root (tree) or last hidden (sequential); projection only.
        for rows in shape_rows:
            row=hidden[index].unsqueeze(0).expand(rows,-1).contiguous()
            logits=compute(self,row)
            record['shape_probes'][rows]=logits[-1].detach().cpu().clone()
        path=Path(directory)/current['request'];path.mkdir(parents=True,exist_ok=True)
        torch.save(record,path/f'logits_{call:02d}.pt')
        if 'state_audit' in current:
            torch.save(current['state_audit'],path/f'state_{call:02d}.pt')
        return result

    computed._target_probe=True
    GPUModelRunner._prepare_inputs=prepared
    MiMoForCausalLM.compute_logits=computed
    print(f'[ReMTP][TargetProbe] bounded actual hidden/logits: {directory}',flush=True)
