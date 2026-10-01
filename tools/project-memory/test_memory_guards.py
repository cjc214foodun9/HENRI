import json
from pathlib import Path
import subprocess
import unittest
from test_project_memory import ProjectMemoryTest, SHA, SOURCE, REPO

class MemoryGuards(ProjectMemoryTest):
    def test_unpublished_record_remote_check_blocks(self):
        self.assertEqual(self.add().returncode,0)
        result=self.run_cli('remote-verify','--branch','feat/engineering-project-memory')
        self.assertEqual(result.returncode,2,result.stdout)
        self.assertIn('default repository record store',json.loads(result.stdout)['error'])
    def test_sensitive_summary_rejected(self):
        result=self.run_cli('add','--commit',SHA,'--source',SOURCE,'--title','Unsafe note','--summary','password="a-real-looking-secret"','--evidence-class','OBSERVED','--reviewed-public')
        self.assertEqual(result.returncode,2,result.stdout)
        self.assertIn('sensitive',json.loads(result.stdout)['error'])
    def test_wrong_sha_current_query_rejected_not_silent_empty(self):
        self.assertEqual(self.add().returncode,0)
        parent=subprocess.check_output(['git','-C',str(REPO),'rev-parse',SHA+'^'],text=True).strip()
        result=self.run_cli('query','--commit',parent,'--query','GraphRuntime')
        self.assertEqual(result.returncode,2,result.stdout)
        self.assertIn('SHA',json.loads(result.stdout)['error'])
    def test_traversal_rejected(self):
        result=self.run_cli('add','--commit',SHA,'--source','HENRI V2/agentic_graph/../runtime.py','--title','Traversal','--summary','Must refuse','--reviewed-public')
        self.assertEqual(result.returncode,2,result.stdout)
        self.assertIn('traversal',json.loads(result.stdout)['error'])
    def test_edited_record_rejected_before_recall(self):
        self.assertEqual(self.add().returncode,0)
        p=next(self.store.glob('*.json')); d=json.loads(p.read_text());d['summary']='Changed after hashing';p.write_text(json.dumps(d))
        result=self.run_cli('query','--commit',SHA,'--query','GraphRuntime')
        self.assertEqual(result.returncode,2,result.stdout)
        self.assertIn('hash drift',json.loads(result.stdout)['error'])
    def test_offline_honcho_refuses_before_import(self):
        result=self.run_cli('honcho-sync')
        self.assertEqual(result.returncode,2,result.stdout)
        self.assertEqual(json.loads(result.stdout)['error'],'Honcho projection is offline by operator policy')
    def test_missing_store_not_vacuous_pass(self):
        result=self.run_cli('verify')
        self.assertEqual(result.returncode,2,result.stdout)
        self.assertIn('missing',json.loads(result.stdout)['error'])

if __name__=='__main__':unittest.main(verbosity=2)
