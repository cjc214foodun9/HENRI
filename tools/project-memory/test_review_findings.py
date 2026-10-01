"""Regressions from the independent review; only temporary local stores/remotes."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from test_project_memory import ProjectMemoryTest, SHA, SOURCE, CLI

class ReviewFindings(ProjectMemoryTest):
    def test_common_gold_and_secret_spelling_refused(self):
        for text in ('gold answer=42','answer_key: A','ground_truth=hidden','auth_token=some-real-looking-token','aws_key AKIAABCDEFGHIJKLMNOP'):
            with self.subTest(text=text):
                result=self.run_cli('add','--commit',SHA,'--source',SOURCE,'--title','Review refusal','--summary',text,'--reviewed-public')
                self.assertEqual(result.returncode,2,result.stdout+result.stderr)
                self.assertIn('sensitive',json.loads(result.stdout)['error'])
    def test_public_acknowledgement_required(self):
        result=self.run_cli('add','--commit',SHA,'--source',SOURCE,'--title','Unreviewed','--summary','Public source note')
        self.assertEqual(result.returncode,2,result.stdout+result.stderr)
        self.assertIn('public review acknowledgement',json.loads(result.stdout)['error'])

class RealLocalRemote(unittest.TestCase):
    def test_ref_sha_and_record_blob_real_git_path(self):
        spec=importlib.util.spec_from_file_location('pm_remote',CLI)
        module=importlib.util.module_from_spec(spec)
        old_bytecode=sys.dont_write_bytecode
        sys.dont_write_bytecode=True
        try:
            spec.loader.exec_module(module)
        finally:
            sys.dont_write_bytecode=old_bytecode
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);repo=root/'repo';remote=root/'remote.git'
            def git(*args):
                return subprocess.check_output(['git','-C',str(repo),*args],stderr=subprocess.PIPE)
            subprocess.run(['git','init',str(repo)],check=True,capture_output=True)
            subprocess.run(['git','init','--bare',str(remote)],check=True,capture_output=True)
            git('config','user.email','infra-test@example.invalid');git('config','user.name','Infrastructure test fixture')
            source=repo/SOURCE;source.parent.mkdir(parents=True);source.write_text('class FixtureRuntime: pass\n')
            git('add','--',SOURCE);git('commit','-m','fixture source')
            sha=git('rev-parse','HEAD').decode().strip()
            endpoint=remote.as_posix();module.REPO_URL=endpoint
            git('remote','add','origin',endpoint)
            body={'schema_version':1,'repo_url':endpoint,'source_commit':sha,'title':'Local Git test fixture','summary':'Tests ref/blob authority only; not a HENRI runtime.','evidence_class':'OBSERVED','sources':[module.source_ref(repo,sha,SOURCE)]}
            record=dict(body,record_id=module.record_id(body))
            store=repo/'docs/project-memory/records';store.mkdir(parents=True)
            path=store/(record['record_id']+'.json');path.write_text(json.dumps(record))
            git('add','--',str(path.relative_to(repo)));git('commit','-m','fixture record');git('push','origin','HEAD:refs/heads/fixture')
            result=module.remote_verify(repo,store,'fixture')
            self.assertEqual(result['status'],'REMOTE_VERIFIED');self.assertEqual(result['verified_records'],1)
            self.assertEqual(result['local_sha'],result['remote_sha'])
            git('commit','--allow-empty','-m','unpushed fixture')
            with self.assertRaisesRegex(ValueError,'SHA differs'):
                module.remote_verify(repo,store,'fixture')

if __name__=='__main__':unittest.main(verbosity=2)
