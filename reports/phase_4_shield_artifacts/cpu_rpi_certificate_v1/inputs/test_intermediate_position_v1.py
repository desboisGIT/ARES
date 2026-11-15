import copy,unittest
from fractions import Fraction as F
import intermediate_position_v1 as p
class IntermediatePositionTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.d=p.rational_domain()
 def test_full_domain_real_margin_and_exact_euclidean_conversion(self):
  r=p.real_reserve(self.d)
  self.assertLess(r['position_reserve_per_axis'],F(1,200));self.assertLessEqual(r['omega_l2_upper'],3)
  self.assertEqual(r['velocity_stage_component_upper'],r['initial_velocity_component_upper']+r['H']*r['acceleration_stage_component_upper'])
  self.assertLess(3*r['chosen_reserve_per_axis']**2,r['chosen_Euclidean_reserve']**2)
 def test_all_original_arithmetic_stage_nodes_and_delay_branches(self):
  for branch in [0,1]:
   r=p.stage_arithmetic(self.d,branch)
   self.assertEqual(len(r['records']),25)
   self.assertEqual({s['stage'] for s in r['records']},{'k1_input','k2_input','k3_input','k4_input','endpoint_pre_normalization'})
   self.assertLess(r['maximum_position_arithmetic_error'],F(1,10000))
 def test_rate_motor_and_command_domain_violations_rejected(self):
  for mutation in [lambda d:d['raw13_real_ranges'][10].update(hi=1.1),lambda d:d['current_word_real_ranges'][1].update(hi=.6),lambda d:d['history_word_real_ranges'][1][3].update(hi=.6),lambda d:d['motor_real_range'].update(hi=6.6),lambda d:d['history_word_real_ranges'][0][0].update(hi=6.6)]:
   d=copy.deepcopy(self.d);mutation(d)
   with self.assertRaises(AssertionError):p.real_reserve(d)
 def test_nonpositive_and_scope_lost_domains_rejected(self):
  for mutation in [lambda d:d['physical_parameter_real_ranges']['m'].update(lo=0),lambda d:d['physical_parameter_real_ranges']['drag'].update(lo=-.1),lambda d:d['physical_parameter_real_ranges']['battery'].update(hi=1.1),lambda d:d['motor_real_range'].update(lo=-.1),lambda d:d['raw13_real_ranges'][0].update(hi=16.1)]:
   d=copy.deepcopy(self.d);mutation(d)
   with self.assertRaises(ValueError):p.real_reserve(d)
if __name__=='__main__':unittest.main()
