import unittest
from test_language import module

class TextCoverage(unittest.TestCase):
    def test_unclosed_identifier_is_advice_not_certification(self):
        result=module().assess('Record the finding as `within_boundary',track='operational')
        self.assertTrue(any(f['rule']=='unclosed_identifier' for f in result['findings']),result)
        self.assertFalse(result['full_ste_compliance'])
    def test_closed_identifier_stays_exact(self):
        result=module().assess('Record `within_boundary` only from the real prover.',track='operational')
        self.assertFalse(any(f['rule']=='unclosed_identifier' for f in result['findings']))
        self.assertEqual(result['input_sha256'],result['output_sha256'])

if __name__=='__main__':unittest.main(verbosity=2)
