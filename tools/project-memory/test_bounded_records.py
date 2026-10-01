import json
import unittest
from test_project_memory import ProjectMemoryTest

class BoundedRecords(ProjectMemoryTest):
    def test_oversized_record_refused_before_parse(self):
        self.store.mkdir();(self.store/'oversize.json').write_bytes(b' '*20000)
        result=self.run_cli('verify')
        self.assertEqual(result.returncode,2,result.stdout)
        self.assertIn('size bound',json.loads(result.stdout)['error'])
    def test_non_mapping_source_ref_refused(self):
        self.assertEqual(self.add().returncode,0)
        p=next(self.store.glob('*.json'));d=json.loads(p.read_text());d['sources']=[None];p.write_text(json.dumps(d))
        result=self.run_cli('verify')
        self.assertEqual(result.returncode,2,result.stdout)
    def test_git_option_commit_refused(self):
        result=self.run_cli('query','--commit=--all','--query','GraphRuntime')
        self.assertEqual(result.returncode,2,result.stdout)
        self.assertIn('SHA',json.loads(result.stdout)['error'])

if __name__=='__main__':unittest.main(verbosity=2)
