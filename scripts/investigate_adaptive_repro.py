#!/usr/bin/env python3
"""Finite short-prefix diagnostics; never counted as quality calibration."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import signal
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_adaptive_round2 import Round2,settings,write_json


def compare_states(left, right):
    """Compare captured distributions too: equal output can hide Q corruption."""
    import torch
    maximum_tv=0.;maximum_hidden=0.;matched=0
    for path in sorted(left.glob('proposal_*.pt')):
        other=right/path.name
        if not other.exists():raise ValueError('Missing reproduction snapshot')
        a=torch.load(path,map_location='cpu',weights_only=True)
        b=torch.load(other,map_location='cpu',weights_only=True)
        if a['history']!=b['history'] or a['topology']!=b['topology']:
            return dict(passed=False,matched_proposals=matched,reason='prefix/topology diverged')
        if not torch.equal(a['ids'],b['ids']):
            return dict(passed=False,matched_proposals=matched,reason='draft tokens diverged')
        maximum_tv=max(maximum_tv,float((a['q']-b['q']).abs().sum(-1).max()/2))
        maximum_hidden=max(maximum_hidden,float((a['root_hidden']-b['root_hidden']).abs().max()))
        matched+=1
    return dict(passed=matched==8 and maximum_tv<=1e-6,
                matched_proposals=matched,max_q_tv=maximum_tv,max_root_hidden_abs=maximum_hidden)


def request_states(directory):
    """The repeated 7311 prompt has the longest root of these two probes."""
    import torch
    paths=sorted((directory/'state_diagnostic').glob('*/proposal_00.pt'),key=lambda p:p.stat().st_mtime_ns)
    states=[(p.parent,int(torch.load(p,map_location='cpu',weights_only=True)['root_positions'][0])) for p in paths]
    longest=max(position for _,position in states)
    return [p for p,position in states if position==longest]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',required=True);parser.add_argument('--port',type=int,default=8000)
    parser.add_argument('--require-match',action='store_true')
    args=parser.parse_args();args.prefix_cache=1;args.deep_diagnostic=True
    controller=Round2(args)
    def interrupted(*args):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupted)
    results={}
    try:
        for cache in (1,0):
            # One server for three requests, including exact prompt reuse.
            # All branches use the same primary request seed per question.
            pairs=[(controller.by_id[7311],43),(controller.by_id[2629],42),(controller.by_id[7311],43)]
            records=controller.requests('reproduction',f'prefix_cache_{cache}',{**settings(head=1),'prefix_cache':cache},pairs,128,True)
            results[str(cache)]={'same_output':records[0]['output']==records[2]['output'],
                                 'tokens':[r['output_tokens'] for r in records],
                                 'note':'128-token prefixes, not a quality test'}
            states=request_states(controller.root/f'reproduction/prefix_cache_{cache}')
            results[str(cache)]['states']=compare_states(states[0],states[1])
            write_json(controller.root/'reproduction_checks.json',results)
        # Fresh server, same request, to separate same-process reuse from restart.
        repeat=controller.requests('reproduction','prefix_cache_0_restart',{**settings(head=1),'prefix_cache':0},
                                   [(controller.by_id[7311],43)],128,True)
        initial=json.loads((controller.root/'reproduction/prefix_cache_0/requests.jsonl').read_text().splitlines()[0])
        a=request_states(controller.root/'reproduction/prefix_cache_0')[0]
        b=request_states(controller.root/'reproduction/prefix_cache_0_restart')[0]
        results['restart_cache_off']={'same_output':initial['output']==repeat[0]['output'],
                                     'states':compare_states(a,b)}
        write_json(controller.root/'reproduction_checks.json',results)
        if args.require_match and not all(r['same_output'] and r['states']['passed'] for r in results.values()):
            controller.status('REPRODUCTION_FAILED')
            return 2
        controller.status('REPRODUCTION_COMPLETE')
        return 0
    finally:controller.stop_server()


if __name__=='__main__':raise SystemExit(main())
