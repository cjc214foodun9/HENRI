"""Typed Jev judgments for the HENRI control plane. Never authorizes actions.

Real Decisions API, exact-request application memoization, explicit receipts.
This is a Hermes tool-facing caller; it does not change HENRI tensor execution.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import uuid
import urllib.error
import urllib.request

MODEL = 'typesafe/jev-1.13'
ENDPOINT = 'https://openrouter.ai/api/alpha/decisions'
RUBRIC_VERSION = 'henri-system1-v1'
RUBRICS = {
    'owner': ('Which HENRI role owns this task? This is advice, not permission.', {
        'RESEARCH':'Primary sources, literature, mathematical formulation.',
        'ARCHITECTURE':'Code, tensor/device/consumer audit, implementation or test design.',
        'INTEGRATION':'Approved infrastructure coordination and evidence delivery.',
        'ARBITRATION':'Ambiguous, conflicting, high-stakes or unsupported task.'}),
    'escalation': ('Which advisory computation route fits the observed task?', {
        'SOLO':'Routine collection, first-pass audit, or a bounded edit.',
        'DELEGATE':'Independent bounded questions suitable for parallel workers.',
        'MOA':'Load-bearing derivation/cross-file audit or two genuine failed repairs.',
        'HUMAN':'Approval, destructive scope, unresolved authority or security risk.'}),
    'failure': ('Classify the observed failure. Do not invent missing evidence.', {
        'INFRASTRUCTURE':'Transport, dependency, credentials, device or environment failure.',
        'HARNESS':'Evaluator, control, fixture, receipt or instrumentation defect.',
        'MECHANISM':'Valid experiment contradicts the proposed mechanism.',
        'UNKNOWN':'Not enough evidence to distinguish the cause.'}),
    'evidence': ('Which kind of evidence is present? Do not promote a claim.', {
        'RETRIEVAL':'Source fetch, citation, or content search.',
        'STATIC':'Schema, lint, file/hash, or source inspection.',
        'EXECUTION':'Actual command/run output with provenance and return code.',
        'EXTERNAL_OUTCOME':'Documented real task outcome with causal provenance.',
        'UNKNOWN':'Missing or ambiguous evidence.'}),
    'store': ('Which engineering store fits this artifact? Never move the artifact.', {
        'VAULT':'Readable research note or decision projection.',
        'ONTOLOGY':'Typed term, mapping, constraint or source evidence.',
        'AUDIT':'Governance decision or approval event.',
        'TELEMETRY':'Operational numeric measurements.',
        'ZONE_C':'Approved HENRI latent/reference artifact, not notes.',
        'UNKNOWN':'Unclear ownership or unsupported object.'}),
    'policy_risk': ('Classify risk from the supplied facts only. Never approves execution.', {
        'DENY':'Explicit requested access conflicts with operator restrictions.',
        'REVIEW':'Ambiguous, expanded, high-impact or missing security evidence.',
        'NO_OBVIOUS_CONFLICT':'No conflict is stated; deterministic prover and kernel checks still required.'}),
}


def stable(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',',':')).encode('utf-8')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def validate_response(data, kind):
    if not isinstance(data, dict) or not str(data.get('model','')).startswith(MODEL):
        raise ValueError('unexpected response model')
    if not data.get('id') or not data.get('provider'):
        raise ValueError('missing generation/provider provenance')
    answers=data.get('answers')
    if not isinstance(answers,dict) or set(answers)!={'decision'}:
        raise ValueError('missing or unexpected answer')
    answer=answers['decision']; choices=RUBRICS[kind][1]
    if answer.get('type')!='choice' or answer.get('choice') not in choices:
        raise ValueError('unknown answer type/choice')
    probabilities=answer.get('probabilities'); confidence=answer.get('confidence')
    if not isinstance(probabilities,dict) or set(probabilities)!=set(choices):
        raise ValueError('missing probability alternatives')
    for value in list(probabilities.values())+[confidence]:
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not 0<=value<=1:
            raise ValueError('invalid probability/confidence')
    if abs(sum(probabilities.values())-1)>0.02:
        raise ValueError('invalid probability mass')
    return answer


def decide(state,kind,*,home,session_id='henri-system1',max_age=300):
    start=time.perf_counter()
    if kind not in RUBRICS: raise ValueError('unsupported rubric')
    from agent.redact import redact_sensitive_text
    state_text=json.dumps(state,sort_keys=True,ensure_ascii=False)
    if redact_sensitive_text(state_text,force=True,redact_url_credentials=True)!=state_text:
        raise ValueError('sensitive state rejected before provider call')
    instructions,criteria=RUBRICS[kind]
    payload={'model':MODEL,'state':state,'questions':{'decision':{'type':'choice','instructions':instructions,'criteria':criteria}}}
    encoded=stable(payload)
    if len(encoded)>32768: raise ValueError('bounded request exceeds 32 KiB; reduce state')
    # Context byte bound is not a tokenizer-level context claim.
    request_hash=digest(encoded)
    cache_key=digest(stable({'rubric':RUBRIC_VERSION,'request':payload}))
    cache=home/'cache/henri-system1'/f'{cache_key}.json'
    cached=None
    if max_age>0 and cache.is_file():
        try:
            c=json.loads(cache.read_text(encoding='utf-8'))
            if c['request_sha256']==request_hash and 0<=time.time()-c['created_epoch']<=max_age:
                validate_response(c['response'],kind); cached=c
        except (ValueError,KeyError,TypeError,OSError):
            cached=None
    if cached:
        response=cached['response']; hit=True
    else:
        from hermes_cli.runtime_provider import resolve_runtime_provider
        rt=resolve_runtime_provider(requested='openrouter',target_model=MODEL)
        key=rt.get('api_key')
        if not key: raise RuntimeError('OpenRouter credential unavailable')
        wire=dict(payload,session_id=session_id)
        req=urllib.request.Request(ENDPOINT,data=stable(wire),method='POST',headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=30) as remote:
            response=json.loads(remote.read())
        validate_response(response,kind)
        cache.parent.mkdir(parents=True,exist_ok=True)
        tmp=cache.with_name(cache.name+'.'+uuid.uuid4().hex+'.tmp')
        tmp.write_text(json.dumps({'request_sha256':request_hash,'created_epoch':time.time(),'response':response},sort_keys=True),encoding='utf-8')
        os.replace(tmp,cache); hit=False
    answer=validate_response(response,kind)
    ambiguous=answer['choice'] in {'ARBITRATION','HUMAN','UNKNOWN','REVIEW','DENY'} or answer['confidence']<0.85
    return {'status':'ESCALATE' if ambiguous else 'ADVISORY','evidence_class':'INFERRED','kind':kind,'rubric_version':RUBRIC_VERSION,
        'requested_model':MODEL,'returned_model':response['model'],'provider':response['provider'],'generation_id':response['id'],
        'request_sha256':request_hash,'response_sha256':digest(stable(response)),
        'answer':answer,'latency_ms':(time.perf_counter()-start)*1000,'application_cache_hit':hit,
        'usage':response.get('usage'),'new_call_cost':0 if hit else response.get('usage',{}).get('cost'),
        'source_cost_is_historical':hit,'authorization':False,'execution_permitted':False,
        'limits':['pilot threshold not calibrated','typed correctness is not semantic correctness','no provider prompt-cache claim']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('decide'); p.add_argument('--state-file',type=Path,required=True)
    p.add_argument('--kind',choices=sorted(RUBRICS),required=True); p.add_argument('--out',type=Path,required=True)
    p.add_argument('--session-id',default='henri-system1'); p.add_argument('--max-age',type=int,default=300)
    args=parser.parse_args()
    try:
        if not 0<=args.max_age<=3600: raise ValueError('max-age must be 0..3600 seconds')
        if len(args.session_id)>256: raise ValueError('session-id too long')
        state=json.loads(args.state_file.read_text(encoding='utf-8'))
        from hermes_cli.config import get_hermes_home
        home=Path(get_hermes_home())
        result=decide(state,args.kind,home=home,session_id=args.session_id,max_age=args.max_age)
        rc=0
    except Exception as e:
        result={'status':'BLOCKED','evidence_class':'BLOCKED','error_type':type(e).__name__,
            'error':str(e)[:500],'authorization':False,'execution_permitted':False}; rc=2
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(result,indent=2,ensure_ascii=False))
    return rc


if __name__=='__main__':
    raise SystemExit(main())
