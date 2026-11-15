import copy,json,unittest
from pathlib import Path
import zero_rate_history_v1 as z
class ZeroRateHistoryTests(unittest.TestCase):
 @classmethod
 def setUpClass(c):
  c.data=json.loads((z.HERE/'public_trim_capture_v1.json').read_text());c.d=z.rational_domain();c.rate=z.rate_stage_lemma(c.d);c.motor=z.broad_motor_arithmetic()
 def case(self,h=None,hold=None):return z.history_case(self.data['measured_history'] if h is None else h,self.data['cases'][0]['public_hold_thrust_N'] if hold is None else hold,0,self.d,self.rate,self.motor)
 def test_all_public_cases_pass_without_oracle(self):
  for r in self.data['cases']:
   result=z.history_case(self.data['measured_history'],r['public_hold_thrust_N'],r['case'],self.d,self.rate,self.motor)
   self.assertTrue(result['conditional_history_inputs_and_stages_pass']);self.assertLessEqual(result['stage_up_axis_error_upper'],z.F(1,5000));self.assertLessEqual(result['stage_angular_rate_upper'],1)
 def test_discrete_rate_and_motor_margins_strict(self):
  self.assertLess(self.rate['macro_angular_endpoint_bound'],self.rate['W']);self.assertLess(self.motor['calculated_motor_error'],self.motor['standalone_settle_motor_error_per_H'])
 def test_noise_packet_outerbox_not_center_only(self):
  h=copy.deepcopy(self.data['measured_history']);h[0]['v'][0][2]=.08
  self.assertFalse(self.case(h)['conditional_history_inputs_and_stages_pass'])
 def test_bad_q_w_motor_thrust_or_timestamp_fail(self):
  for field,value in [('q',[0.,0.,0.,1.001]),('w',[.01,0.,0.])]:
   h=copy.deepcopy(self.data['measured_history']);h[0][field][0]=value;self.assertFalse(self.case(h)['conditional_history_inputs_and_stages_pass'])
  self.assertFalse(self.case(hold=6.4)['conditional_history_inputs_and_stages_pass'])
  h=copy.deepcopy(self.data['measured_history']);del h[3]
  with self.assertRaises(ValueError):self.case(h)
if __name__=='__main__':unittest.main()
