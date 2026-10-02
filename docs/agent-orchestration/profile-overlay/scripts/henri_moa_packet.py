"""Fixed language policy and exact task tail for HENRI advisory requests."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import importlib.util


def language_module():
    spec=importlib.util.spec_from_file_location('henri_packet_language',Path(__file__).with_name('henri_language.py'))
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod


def pack(task: str) -> dict:
    language=language_module()
    language.assess(task, track='formal')  # input bound, not a style transformation
    SCAFFOLD=language.SCAFFOLD
    return {'version': 'HENRI-STE-V1', 'static_prefix_sha256': hashlib.sha256(SCAFFOLD.encode()).hexdigest(),
            'messages': [{'role': 'user', 'content': SCAFFOLD},
                         {'role': 'user', 'content': 'TASK STATE:\n' + task}],
            'authorization': False, 'task_sha256': hashlib.sha256(task.encode()).hexdigest()}


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task-file',type=Path,required=True)
    args=parser.parse_args()
    try:
        if args.task_file.stat().st_size>65536: raise ValueError('task size bound exceeded')
        print(json.dumps(pack(args.task_file.read_text(encoding='utf-8')),sort_keys=True,indent=2))
        return 0
    except (OSError,ValueError,TypeError):
        print(json.dumps({'status':'BLOCKED','authorization':False,'version':'HENRI-STE-V1'}))
        return 2

if __name__=='__main__':raise SystemExit(main())
