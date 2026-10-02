import unittest
from test_language import module

class AdjectiveBoundaries(unittest.TestCase):
    def test_known_ing_adjectives_do_not_claim_progressive(self):
        result=module().assess('The file is missing. The value is remaining.',track='operational')
        self.assertFalse(any(f['rule']=='progressive' for f in result['findings']),result)
    def test_copula_participle_is_suspected_not_certain(self):
        result=module().assess('The valve is closed.',track='operational',mode='procedure')
        self.assertFalse(any(f['rule']=='passive_procedure' for f in result['findings']),result)
        self.assertTrue(any(f['rule']=='passive_candidate' for f in result['findings']),result)
    def test_direct_passive_command_still_fires(self):
        result=module().assess('The valve must be closed.',track='operational',mode='procedure')
        self.assertTrue(any(f['rule']=='passive_procedure' for f in result['findings']))

if __name__=='__main__':unittest.main(verbosity=2)
