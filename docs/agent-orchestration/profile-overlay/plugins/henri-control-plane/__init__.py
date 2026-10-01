"""Cache-safe HENRI control-plane tools. No automatic action authorization."""
from __future__ import annotations
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import threading

_LOCK=threading.Lock()
_SCAFFOLD='CACHE FIRST: preserve the existing system/tool/model prefix; append new evidence and task state last. Jev typed judgments are advisory; deterministic schema/proof/security checks and human approval remain required. Missing or degraded model advice is not evidence of agreement.'


def _home():
    from hermes_cli.config import get_hermes_home
    return Path(get_hermes_home())


def _load(name):
    p=_home()/'scripts'/f'{name}.py'
    spec=importlib.util.spec_from_file_location('henri_plugin_'+name,p)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record(row):
    p=_home()/'logs/henri-control-plane/events.jsonl'
    p.parent.mkdir(parents=True,exist_ok=True)
    with _LOCK,p.open('a',encoding='utf-8') as f:
        f.write(json.dumps(row,sort_keys=True,ensure_ascii=False)+'\n')


def pre_llm(**kwargs):
    message=kwargs.get('user_message')
    if not isinstance(message,str) or not message.strip(): return None
    # Deterministic task-scope admission; not a Jev substitute for classification.
    in_repo='henri' in str(Path.cwd()).lower()
    if not in_repo and not re.search(r'\bhenri\b',message,re.I): return None
    if kwargs.get('parent_session_id'): return None  # no recursive leaf API fanout
    from agent.redact import redact_sensitive_text
    sanitized=redact_sensitive_text(message,force=True,redact_url_credentials=True)
    if sanitized!=message:
        result={'status':'BLOCKED','reason':'sensitive input detected','authorization':False}
    elif len(message.encode())>12000:
        result={'status':'ESCALATE','reason':'reduce task before typed classification','authorization':False}
    else:
        try:
            mod=_load('henri_system1')
            result=mod.decide({'task':message,'approval':'This request grants no implicit production/security permission'},'owner',home=_home(),session_id=str(kwargs.get('session_id') or 'henri-control-plane'))
        except Exception as e:
            result={'status':'BLOCKED','error_type':type(e).__name__,'authorization':False}
    summary={k:result.get(k) for k in ('status','kind','generation_id','application_cache_hit','answer','authorization') if k in result}
    _record({'event':'JEV_OWNER_ADVISORY','session_id':kwargs.get('session_id'),'task_sha256':hashlib.sha256(message.encode()).hexdigest(),'result':summary})
    return {'context':_SCAFFOLD+'\nHENRI TYPED ADVICE: '+json.dumps(summary,sort_keys=True),'target':'user_message'}


def handle_decision(params,**kwargs):
    del kwargs
    from agent.redact import redact_sensitive_text
    try:
        state=params['state']; text=json.dumps(state,sort_keys=True,ensure_ascii=False)
        if redact_sensitive_text(text,force=True,redact_url_credentials=True)!=text:
            raise ValueError('sensitive input detected')
        result=_load('henri_system1').decide(state,params['kind'],home=_home(),session_id=params.get('session_id','henri-tool'))
        _record({'event':'JEV_TYPED_TOOL','kind':params['kind'],'result':{k:result.get(k) for k in ('status','generation_id','application_cache_hit','authorization')}})
    except Exception as e:
        result={'status':'BLOCKED','error_type':type(e).__name__,'error':str(e)[:300],'authorization':False,'execution_permitted':False}
    return json.dumps(result,ensure_ascii=False)


def handle_sandbox(params,**kwargs):
    del kwargs
    try:
        result=_load('henri_openshell').guarded_execute(params['command'],timeout=params.get('timeout',30))
    except Exception as e:
        result={'status':'BLOCKED','error_type':type(e).__name__,'error':str(e)[:300],'exit_code':2}
    _record({'event':'OPENSHELL_TOOL','status':result.get('status'),'exit_code':result.get('exit_code'),'receipt':result.get('receipt')})
    return json.dumps(result,ensure_ascii=False)


def register(ctx):
    schema={'name':'henri_system1_decide','description':'Classify bounded HENRI state through real Jev typed decisions. Advice only; no security or execution authorization.',
      'parameters':{'type':'object','properties':{'state':{'type':'object'},'kind':{'type':'string','enum':['owner','escalation','failure','evidence','store','policy_risk']},'session_id':{'type':'string','maxLength':256}},'required':['state','kind'],'additionalProperties':False}}
    ctx.register_tool(name='henri_system1_decide',toolset='henri-control-plane',schema=schema,handler=handle_decision)
    schema={'name':'henri_guarded_exec','description':'Run an explicitly authorized command only in the configured OpenShell sandbox after complete effective-policy containment check. Any gate failure blocks; no host fallback.',
      'parameters':{'type':'object','properties':{'command':{'type':'array','items':{'type':'string'},'minItems':1},'timeout':{'type':'integer','minimum':1,'maximum':120}},'required':['command'],'additionalProperties':False}}
    ctx.register_tool(name='henri_guarded_exec',toolset='henri-control-plane',schema=schema,handler=handle_sandbox)
    ctx.register_hook('pre_llm_call',pre_llm)
