#!/usr/bin/env python3
"""A stdlib-only dependency waiter; no Torch/model memory while root GPU runs."""
import argparse
from datetime import datetime, timezone, timedelta
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import traceback


def write(path, value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temp.replace(path)


def read(path):
    return json.loads(path.read_text())


def dependency_state(reference, unit):
    """Never take elapsed time as proof of successful dependency completion."""
    result=subprocess.run(['systemctl','--user','is-active',unit],
                          capture_output=True,text=True).stdout.strip()
    end_path=reference/'exit.json'
    if end_path.exists():
        end=read(end_path)
        if end.get('exit_code')!=0 or (reference/'failure.json').exists():
            raise RuntimeError('根因定位单元未通过，禁止启动新候选：'+str(end))
        status=read(reference/'status.json')
        if status.get('phase')!='COMPLETED_ROOT_CAUSE_UNIT':
            raise RuntimeError('根因定位阶段不是完整完成')
        if result in ('active','activating','deactivating'):
            return False
        return True
    if result not in ('active','activating','deactivating'):
        raise RuntimeError('根因定位已停止但没有成功退出证据：'+result)
    return False


def waiting_report(run, reference, current, deadline):
    text=('# Adaptive RAVEN 修复实验：等待根因诊断完成\n\n'
        '本进程只等待依赖，未加载Torch/模型，不占用第二份GPU服务。\n\n'
        '根因诊断目录：'+str(reference)+'\n\n'
        '当前检查时间：'+current+'\n\n'
        '总运行截止：'+deadline+'\n\n'
        '诊断完整通过后执行：48次CUDA兼容/完整H检查 → 6条修复自然轨迹 → '
        '12个实际Target前缀检查 → 80条同协议无探针五方法校准。\n\n'
        '单候选、无自动重试；正确率/正常结束判对、截断、吞吐及整题耗时分别评估。'
        '没有自动采用或展开留出/test。\n')
    temp=run/'report.md.tmp';temp.write_text(text);temp.replace(run/'report.md')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--reference-dir',type=Path,required=True)
    p.add_argument('--dependency-unit',required=True)
    a=p.parse_args();run=a.run_dir.resolve();ref=a.reference_dir.resolve()
    launch=read(run/'launch.json')
    deadline=datetime.now(timezone.utc)+timedelta(seconds=launch['runtime_max_seconds'])
    started=time.monotonic()
    def interrupted(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupted)
    write(run/'supervisor.json',dict(started_utc=datetime.now(timezone.utc).isoformat(),
        deadline_utc=deadline.isoformat(),waiting_pid=os.getpid(),torch_loaded=False,
        dependency_unit=a.dependency_unit))
    try:
        while True:
            if time.monotonic()-started>=launch['runtime_max_seconds']:
                raise TimeoutError('包含依赖等待的24小时上限已到')
            if dependency_state(ref,a.dependency_unit):
                with socket.socket() as sock:
                    if sock.connect_ex(('127.0.0.1',8000))==0:
                        raise RuntimeError('根因定位结束后端口仍被占用；不关闭未知进程')
                write(run/'status.json',dict(phase='DEPENDENCY_READY',controller_pid=os.getpid(),
                    updated_utc=datetime.now(timezone.utc).isoformat()))
                command=launch['controller_command']
                os.execv(command[0],command)
            current=datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=8))).strftime(
                '%Y-%m-%d %H:%M:%S UTC+8')
            write(run/'status.json',dict(phase='WAITING_ROOT_DIAGNOSTICS',
                current_work=dict(dependency=str(ref),unit=a.dependency_unit),
                updated_utc=datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC'),
                controller_pid=os.getpid(),torch_loaded=False))
            waiting_report(run,ref,current,deadline.astimezone(timezone(timedelta(hours=8))).strftime(
                '%Y-%m-%d %H:%M:%S UTC+8'))
            time.sleep(15)
    except BaseException as exc:
        kind='INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'DEPENDENCY_REJECTED'
        value=dict(kind=kind,type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc(),
                   updated_utc=datetime.now(timezone.utc).isoformat(),failed_stage='WAITING_ROOT_DIAGNOSTICS')
        write(run/'failure.json',value)
        write(run/'status.json',dict(phase=kind,current_work=dict(dependency=str(ref)),**value))
        write(run/'exit.json',dict(exit_code=130 if kind=='INTERRUPTED' else 3,
            phase=kind,server_cleaned=True,model_loaded=False,
            ended_utc=datetime.now(timezone.utc).isoformat()))
        temp=run/'report.md.tmp'
        temp.write_text('# Adaptive RAVEN 修复实验停止\n\n'+value['error']+
            '\n\n原始依赖/失败记录保留。没有加载新模型或自动重试。\n')
        temp.replace(run/'report.md')
        return 130 if kind=='INTERRUPTED' else 3


if __name__=='__main__':
    raise SystemExit(main())
