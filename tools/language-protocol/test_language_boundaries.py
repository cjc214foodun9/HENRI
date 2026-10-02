import unittest
from test_language import module

class TrackBoundaries(unittest.TestCase):
    def test_soft_length_keeps_text_exact(self):
        text='Use '+' '.join('word' for _ in range(22))+'.'
        result=module().assess(text,track='operational',mode='procedure')
        self.assertEqual(result['status'],'ADVISORY')
        self.assertTrue(any(f['rule']=='length_target' for f in result['findings']))
        self.assertEqual(result['input_sha256'],result['output_sha256'])
    def test_formal_and_visual_skip_style(self):
        text='Utilize parameter = torch.nn.Parameter(x). The operator is running. F(s,a)=E[ln q-ln p].'
        for track in ('formal','visual'):
            result=module().assess(text,track=track)
            self.assertEqual(result['status'],'EXEMPT')
            self.assertEqual(result['findings'],[])
            self.assertFalse(result['rewritten'])
    def test_ing_nouns_and_numeric_values_do_not_trigger(self):
        result=module().assess('During the test, use the landing gear. The value is 0.35. Use approximately 10 units.',track='operational')
        self.assertEqual(result['findings'],[])
    def test_inline_identifier_and_quote_stay_exact(self):
        text='Use `ensure_exact_hash`.\n> Utilize a quoted term.'
        result=module().assess(text,track='operational',mode='procedure')
        self.assertEqual(result['findings'],[])
        self.assertEqual(result['protected_lines'],1)
    def test_explicit_hazard_uses_correct_marker_even_visual(self):
        m=module()
        result=m.assess('WARNING: The disk can lose data.',track='visual',hazard='damage')
        self.assertTrue(any(f['rule']=='safety_marker' for f in result['findings']))
        self.assertEqual(m.assess('CAUTION: The disk can lose data.',track='operational',hazard='damage')['findings'],[])
    def test_malformed_track_and_fence_refuse(self):
        m=module()
        with self.assertRaisesRegex(ValueError,'track'):m.assess('Use the tool.',track='guess')
        with self.assertRaisesRegex(ValueError,'unclosed'):m.assess('```python\nutilize()',track='operational')
    def test_compound_noun_requires_human_review_not_fake_pass(self):
        result=module().assess('Use the tensor buffer allocation control system.',track='operational',mode='procedure')
        self.assertFalse(result['full_ste_compliance'])
        self.assertIn('noun groups',result['coverage'])

if __name__=='__main__':unittest.main(verbosity=2)
