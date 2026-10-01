"""Local infrastructure tests use actual HENRI Git objects, no model execution."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import os


def repo_root():
    if os.environ.get('HENRI_PROJECT_MEMORY_TEST_REPO'):
        return Path(os.environ['HENRI_PROJECT_MEMORY_TEST_REPO']).resolve()
    for parent in Path(__file__).resolve().parents:
        if (parent/'docs/agent-orchestration/profile-overlay/scripts/henri_project_memory.py').is_file():
            return parent
    raise RuntimeError('set HENRI_PROJECT_MEMORY_TEST_REPO for standalone tooling tests')


REPO = repo_root()
CLI = Path(os.environ.get('HENRI_PROJECT_MEMORY_TEST_CLI', str(REPO/'docs/agent-orchestration/profile-overlay/scripts/henri_project_memory.py')))
SOURCE = 'HENRI V2/agentic_graph/runtime.py'
SHA = 'c897d435cb89542f865c4c69e121a75ec2cb7c47'

class ProjectMemoryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Path(self.tmp.name)/'records'
    def tearDown(self):
        self.tmp.cleanup()
    def run_cli(self,*args):
        return subprocess.run([sys.executable,str(CLI),*args,'--repo',str(REPO),'--store',str(self.store)],capture_output=True,text=True)
    def add(self):
        return self.run_cli('add','--commit',SHA,'--title','GraphRuntime source audit','--summary','The engineering graph runtime is a control-plane source, not an external task outcome.','--source',SOURCE,'--evidence-class','OBSERVED','--reviewed-public')
    def test_cross_process_add_query_and_duplicate_are_deterministic(self):
        added=self.add()
        self.assertEqual(added.returncode,0,added.stderr+added.stdout)
        first=json.loads(added.stdout)
        duplicate=self.add()
        self.assertEqual(duplicate.returncode,0,duplicate.stderr+duplicate.stdout)
        self.assertEqual(first['record_id'],json.loads(duplicate.stdout)['record_id'])
        self.assertEqual(len(list(self.store.glob('*.json'))),1)
        queried=self.run_cli('query','--query','GraphRuntime','--commit',SHA)
        self.assertEqual(queried.returncode,0,queried.stderr+queried.stdout)
        payload=json.loads(queried.stdout)
        self.assertEqual(payload['count'],1)
        self.assertEqual(payload['records'][0]['source_commit'],SHA)
        verified=self.run_cli('verify')
        self.assertEqual(verified.returncode,0,verified.stderr+verified.stdout)

if __name__=='__main__':
    unittest.main(verbosity=2)
