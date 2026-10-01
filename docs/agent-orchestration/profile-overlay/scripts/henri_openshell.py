"""Fail-closed OpenShell command gate for the approved local WSL pilot.

This gates only calls through this consumer. It does not sandbox host terminal
calls or establish remote Vast/GPU isolation. No probabilistic authorization.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import uuid
import yaml


def _home():
    from hermes_cli.config import get_hermes_home
    return Path(get_hermes_home())


def _wsl_path(path):
    p=Path(path).resolve()
    if not p.drive or len(p.drive)!=2 or p.drive[1]!=':':
        raise ValueError('pilot expects an absolute Windows artifact path')
    return '/mnt/'+p.drive[0].lower()+'/'+p.as_posix()[3:]


def _invoke(config,args,timeout=30):
    prefix=['wsl','-d',config['distro'],'--',config['cli']]
    return subprocess.run(prefix+args,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=timeout)


def guarded_execute(command,timeout=30):
    if not isinstance(command,list) or not command or not all(isinstance(x,str) and x and '\x00' not in x for x in command):
        raise ValueError('nonempty command argv required')
    if not 1<=timeout<=120: raise ValueError('timeout must be 1..120')
    home=_home(); config=json.loads((home/'henri-openshell.json').read_text(encoding='utf-8'))
    if config.get('enabled') is not True: raise RuntimeError('pilot consumer disabled')
    if command[0] not in config['allowed_executables']:
        raise PermissionError('executable outside approved pilot allowlist')
    boundary=Path(config['boundary']).resolve()
    if not boundary.is_file(): raise RuntimeError('operator boundary missing')
    if hashlib.sha256(boundary.read_bytes()).hexdigest()!=config['boundary_sha256']:
        raise PermissionError('operator boundary hash drift')
    run=home/'logs/henri-openshell'/uuid.uuid4().hex
    run.mkdir(parents=True,exist_ok=False)
    receipt={'scope':'local WSL sandbox consumer only; not host-wide/Vast/GPU verification',
        'command':command,'gateway':config['gateway'],'sandbox':config['sandbox'],'status':'BLOCKED','exit_code':2}
    try:
        result=_invoke(config,['-g',config['gateway'],'sandbox','get',config['sandbox'],'-o','json'])
        (run/'sandbox.json').write_text(result.stdout,encoding='utf-8')
        if result.returncode!=0: raise RuntimeError('sandbox read failed')
        sandbox=json.loads(result.stdout)
        if sandbox.get('phase')!='Ready': raise RuntimeError('sandbox not Ready')
        result=_invoke(config,['-g',config['gateway'],'policy','get',config['sandbox'],'--full','-o','json'])
        (run/'policy-response.json').write_text(result.stdout,encoding='utf-8')
        if result.returncode!=0: raise RuntimeError('effective policy read failed')
        effective=json.loads(result.stdout)
        if effective.get('status')!='effective' or not isinstance(effective.get('policy'),dict):
            raise RuntimeError('effective-policy provenance absent')
        candidate=run/'effective-policy.yaml'
        candidate.write_text(yaml.safe_dump(effective['policy'],sort_keys=True),encoding='utf-8')
        proof_args=['wsl','-d',config['distro'],'--',config['prover'],'check',_wsl_path(candidate),'--boundary',_wsl_path(boundary),'--output','json','--timeout','10s']
        proof=subprocess.run(proof_args,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=20)
        (run/'proof.json').write_text(proof.stdout,encoding='utf-8')
        (run/'proof-stderr.txt').write_text(proof.stderr,encoding='utf-8')
        verdict=json.loads(proof.stdout)
        if proof.returncode!=0 or verdict.get('result')!='within_boundary' or verdict.get('exit_code')!=0:
            raise PermissionError('policy containment failed: '+str(verdict.get('result')))
        coverage=set(verdict.get('coverage',{}).get('domains',[]))
        if not set(config['required_coverage'])<=coverage: raise PermissionError('proof coverage insufficient')
        receipt.update(proof_result=verdict['result'],coverage=sorted(coverage),sandbox_id=sandbox['id'],policy_hash=effective.get('hash'))
        start=time.perf_counter()
        result=_invoke(config,['-g',config['gateway'],'sandbox','exec','--name',config['sandbox'],'--timeout',str(timeout),'--no-tty','--no-login-shell','--',*command],timeout=timeout+20)
        (run/'stdout.txt').write_text(result.stdout,encoding='utf-8')
        (run/'stderr.txt').write_text(result.stderr,encoding='utf-8')
        receipt.update(status='EXECUTED' if result.returncode==0 else 'EXECUTION_FAILED',exit_code=result.returncode,
            output=result.stdout[-12000:],stderr=result.stderr[-2000:],latency_ms=(time.perf_counter()-start)*1000)
        # Read policy again to detect concurrent mutation. This does not make the check+exec atomic.
        after=_invoke(config,['-g',config['gateway'],'policy','get',config['sandbox'],'--full','-o','json'])
        if after.returncode!=0 or json.loads(after.stdout).get('hash')!=effective.get('hash'):
            receipt['status']='BLOCKED_POLICY_CHANGED'; receipt['exit_code']=2
    except Exception as e:
        receipt.update(status='BLOCKED',exit_code=2,error_type=type(e).__name__,error=str(e)[:700])
    receipt['limitations']=['check+exec is not atomic against concurrent operator policy changes','kernel boundary applies only to named sandbox','no automatic authorization or remote GPU claim']
    receipt['receipt']=str(run/'receipt.json')
    (run/'receipt.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False),encoding='utf-8')
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--timeout',type=int,default=30)
    p.add_argument('command',nargs=argparse.REMAINDER)
    a=p.parse_args(); cmd=a.command
    if cmd and cmd[0]=='--': cmd=cmd[1:]
    try: result=guarded_execute(cmd,a.timeout)
    except Exception as e: result={'status':'BLOCKED','exit_code':2,'error':str(e)[:500]}
    print(json.dumps(result,indent=2,ensure_ascii=False))
    return result['exit_code'] if isinstance(result.get('exit_code'),int) else 2


if __name__=='__main__':
    raise SystemExit(main())
