"""Infrastructure checks for language tracks. No HENRI model score."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

def script_path():
    explicit=os.environ.get('HENRI_LANGUAGE_TEST_CLI')
    if explicit:return Path(explicit)
    for parent in Path(__file__).resolve().parents:
        candidate=parent/'docs/agent-orchestration/profile-overlay/scripts/henri_language.py'
        if candidate.is_file():return candidate
    return Path(os.environ['HERMES_HOME'])/'scripts/henri_language.py'

CLI=script_path()

def module():
    spec=importlib.util.spec_from_file_location('henri_language_test',CLI)
    mod=importlib.util.module_from_spec(spec)
    old=sys.dont_write_bytecode;sys.dont_write_bytecode=True
    try: spec.loader.exec_module(mod)
    finally: sys.dont_write_bytecode=old
    return mod

class LanguageTracks(unittest.TestCase):
    def test_selected_patterns_are_advice_not_permission(self):
        self.assertTrue(CLI.is_file(),'language checker is missing')
        mod=module()
        text='The checkpoint is being deleted. The agent has written a log. The file must be deleted. Utilize the tool.'
        result=mod.assess(text,track='operational',mode='procedure')
        rules={f['rule'] for f in result['findings']}
        self.assertTrue({'progressive','perfect','passive_procedure','word_choice'}<=rules,result)
        self.assertEqual(result['status'],'ADVISORY')
        self.assertFalse(result['authorization'])
        self.assertEqual(result['input_sha256'],result['output_sha256'])
        self.assertFalse(result['full_ste_compliance'])

if __name__=='__main__':unittest.main(verbosity=2)
