import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from test_language import CLI

# Use the same profile overlay as the checker when tests run from the repo.
HOME=CLI.parent.parent

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec)
    old=sys.dont_write_bytecode;sys.dont_write_bytecode=True
    try:spec.loader.exec_module(mod)
    finally:sys.dont_write_bytecode=old
    return mod

class ConsumerPath(unittest.TestCase):
    def test_packet_fixed_prefix_exact_tail(self):
        p=HOME/'scripts/henri_moa_packet.py'
        self.assertTrue(p.is_file(),'packet caller missing')
        m=load(p,'packet_test')
        task='F(s,a)=E[ln q-ln p]. Use `within_boundary`. '
        a=m.pack(task);b=m.pack(task+'Different source.')
        self.assertEqual(a['static_prefix_sha256'],b['static_prefix_sha256'])
        self.assertTrue(a['messages'][0]['content'].startswith('HENRI-STE-V1'))
        self.assertEqual(a['messages'][-1]['content'],'TASK STATE:\n'+task)
    def test_guard_invalid_description_stops_before_dispatch(self):
        m=load(HOME/'scripts/henri_openshell.py','guard_test')
        with patch.object(m,'_invoke') as called:
            with self.assertRaisesRegex(ValueError,'bounded'):
                m.guarded_execute(['/usr/bin/id'],description='x'*65537)
            called.assert_not_called()
    def test_hook_uses_static_language_tail_without_new_models(self):
        p=load(HOME/'plugins/henri-control-plane/__init__.py','hook_test')
        class Decision:
            @staticmethod
            def decide(*args,**kwargs):return {'status':'ADVISORY','authorization':False}
        with patch.object(p,'_load',side_effect=lambda n:Decision if n=='henri_system1' else load(HOME/'scripts/henri_language.py','hook_language_test')),patch.object(p,'_record'):
            result=p.pre_llm(user_message='HENRI source audit.',session_id='test')
        self.assertEqual(result['target'],'user_message')
        self.assertIn('HENRI-STE-V1',result['context'])
        self.assertIn('Keep code, equations, identifiers',result['context'])

if __name__=='__main__':unittest.main(verbosity=2)
