"""Fail-closed numerical-worker fixtures; not full motion certificates."""
import sys,json,unittest
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
from ares.phase2 import default_profile
from conditional_cpu_backend import ConditionalCPUBackend,GuardFailure

class ConditionalBackendTests(unittest.TestCase):
 def make(self):return ConditionalCPUBackend(default_profile(),delay_s=.0125)
 def command(self):return torch.tensor([[4.903325,0.,0.,0.]],dtype=torch.float32)
 def test_nominal_full_body_steps_keep_original_physics_and_no_certificate(self):
  b=self.make()
  for _ in range(20):
   result=b.step(self.command())
   self.assertTrue(result['conditional_numeric_budget_applicable'])
   self.assertFalse(result['flight_certificate'])
   self.assertFalse(result['platform_primitive_contract_audited'])
  self.assertTrue(torch.isfinite(b.plant.state).all())
  self.assertLess(float(b.plant.velocity.abs().max()),1e-4)
 def test_nonfinite_command_and_outside_domain_fail_before_mutation(self):
  for command in [torch.tensor([[float('nan'),0.,0.,0.]]),torch.tensor([[4.903325,1.,0.,0.]])]:
   b=self.make();before=b.plant.state.clone()
   with self.assertRaises(GuardFailure) as error:b.step(command)
   self.assertFalse(error.exception.state_advanced)
   self.assertTrue(torch.equal(before,b.plant.state))
   with self.assertRaises(GuardFailure):b.step(self.command())
 def test_quaternion_shell_position_and_precision_guards(self):
  b=self.make();b.plant.state[:,6]=1.001
  with self.assertRaisesRegex(GuardFailure,'quaternion'):b.step(self.command())
  b=self.make();b.plant.state[:,0]=16.1
  with self.assertRaisesRegex(GuardFailure,'state domain'):b.step(self.command())
  b=self.make();torch.set_float32_matmul_precision('high')
  try:
   with self.assertRaisesRegex(GuardFailure,'precision'):b.step(self.command())
  finally:torch.set_float32_matmul_precision('highest')
 def test_mkldnn_route_change_fails_closed(self):
  b=self.make();torch.backends.mkldnn.enabled=True
  try:
   with self.assertRaisesRegex(GuardFailure,'precision'):b.step(self.command())
  finally:torch.backends.mkldnn.enabled=False
 def test_post_step_domain_loss_latches_and_reports_advanced_state(self):
  b=self.make();b.plant.state[:,0]=16.;b.plant.state[:,7]=.05
  with self.assertRaises(GuardFailure) as error:b.step(self.command())
  self.assertTrue(error.exception.state_advanced)
  self.assertGreater(float(b.plant.position[0,0]),16.)
  with self.assertRaisesRegex(GuardFailure,'latched'):b.step(self.command())
  b.reset();self.assertFalse(b.step(self.command())['flight_certificate'])
 def test_static_program_parameters_and_battery_domains_fail_closed(self):
  b=self.make();b.plant.model.controller.K_angvel[0,0]=31.
  with self.assertRaisesRegex(GuardFailure,'static'):b.step(self.command())
  b=self.make()
  with self.assertRaisesRegex(GuardFailure,'battery'):b.step(self.command(),battery_authority=.94)
  with self.assertRaises(GuardFailure):ConditionalCPUBackend(default_profile(),delay_s=.021)

if __name__=='__main__':unittest.main()
