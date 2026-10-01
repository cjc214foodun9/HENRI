import hashlib
import json
from pathlib import Path
import unittest
from test_project_memory import ProjectMemoryTest, SHA, SOURCE

class AdditionalGuards(ProjectMemoryTest):
    def test_duplicate_json_key_refused(self):
        self.assertEqual(self.add().returncode,0)
        p=next(self.store.glob('*.json')); raw=p.read_text();p.write_text(raw.replace('"schema_version": 1','"schema_version": 1, "schema_version": 1'))
        result=self.run_cli('verify')
        self.assertEqual(result.returncode,2,result.stdout)
        self.assertIn('duplicate',json.loads(result.stdout)['error'])
    def test_ancestor_symlink_or_junction_not_followed(self):
        import os, subprocess
        root=Path(self.tmp.name); destination=root/'destination';destination.mkdir()
        link=root/'linked'
        if os.name=='nt':
            created=subprocess.run(['cmd','/c','mklink','/J',str(link),str(destination)],capture_output=True,text=True)
            self.assertEqual(created.returncode,0,created.stderr)
        else:
            link.symlink_to(destination,target_is_directory=True)
        self.store=link/'records'
        result=self.add()
        self.assertEqual(result.returncode,2,result.stdout)
        self.assertIn('link',json.loads(result.stdout)['error'])
        self.assertFalse((destination/'records').exists())
    def test_rehashed_wrong_source_ref_refused(self):
        self.assertEqual(self.add().returncode,0)
        p=next(self.store.glob('*.json')); d=json.loads(p.read_text());d['sources'][0]['sha256']='f'*64
        body={k:v for k,v in d.items() if k!='record_id'}
        d['record_id']='pm-'+hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
        p.unlink(); (self.store/(d['record_id']+'.json')).write_text(json.dumps(d))
        result=self.run_cli('verify')
        self.assertEqual(result.returncode,2,result.stdout)
        self.assertIn('source hash drift',json.loads(result.stdout)['error'])
    def test_excluded_benchmark_payload_refused(self):
        result=self.run_cli('add','--commit',SHA,'--source',SOURCE,'--title','Blocked payload','--summary','gold_answer=spoiler','--reviewed-public')
        self.assertEqual(result.returncode,2,result.stdout)
        self.assertIn('benchmark',json.loads(result.stdout)['error'])

if __name__=='__main__':unittest.main(verbosity=2)
