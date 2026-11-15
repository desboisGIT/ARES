from fractions import Fraction as F
import random,unittest,sys
from pathlib import Path
from unittest.mock import patch
import torch
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parents[3]))
from verified_sqrt import *
from backend import explicit_norm,NormWitness,ConditionalVerifiedNormCPUBackend,GuardFailure
from ares.phase2 import default_profile
class VerifiedNormTests(unittest.TestCase):
 def test_subnormal_normal_and_extreme_input_words(self):
  rng=random.Random(7234)
  for index in [0,1,2,0x007fffff,0x00800000,0x3f800000,0x7f7fffff]+[rng.randrange(0,0x7f800000) for _ in range(2000)]:
   value,w=sqrt_word(word(index));self.assertTrue(w['exact_squared_bin_pass'])
   if index:
    lo,hi=w['rounding_bin_lower'],w['rounding_bin_upper'];x=F.from_float(word(index))
    self.assertTrue(lo<0 or x>=lo*lo);self.assertLessEqual(x,hi*hi)
 def test_squared_bin_midpoint_ties_choose_even_including_zero_subnormals(self):
  for index in [0,1,2,3,0x007ffffe,0x007fffff,0x00800000,0x3f7fffff,0x3f800000,0x4f000001]:
   lo=F.from_float(word(index));hi=F.from_float(word(index+1));mid=(lo+hi)/2
   y,w=sqrt_fraction(mid*mid);self.assertEqual(bits(y),index if index%2==0 else index+1)
   self.assertTrue(w['upper_midpoint_tie'] or w['lower_midpoint_tie'])
   self.assertEqual(bits(sqrt_fraction(mid*mid*(1-F(1,2**100)))[0]),index)
   self.assertEqual(bits(sqrt_fraction(mid*mid*(1+F(1,2**100)))[0]),index+1)
  self.assertEqual(bits(sqrt_word(-0.)[0]),0x80000000)
 def test_invalid_sqrt_and_norm_inputs_fail(self):
  for x in [-1.,float('nan'),float('inf'),.1]:
   with self.assertRaises(ValueError):sqrt_word(x)
  with self.assertRaises(ValueError):explicit_norm(torch.ones(1,5),5,NormWitness())
  with self.assertRaises(ValueError):explicit_norm(torch.ones(1,3,dtype=torch.float64),3,NormWitness())
 def test_norm_exact_error_gamma_contract_on_all_associations(self):
  rng=random.Random(981)
  for width in [3,4]:
   for _ in range(100):
    v=torch.tensor([[rng.uniform(-1.,1.) for j in range(width)]],dtype=torch.float32);audit=NormWitness();y=explicit_norm(v,width,audit)
    exact=sum((F.from_float(x)**2 for x in v[0].tolist()),F(0))
    # Exact error in explicit squared sum satisfies gamma_(2n). Sqrt has its
    # own per-point bin witness instead of a library approximation assumption.
    total=v[0,0]*v[0,0]
    for j in range(1,width):total=total+v[0,j]*v[0,j]
    self.assertLessEqual(abs(F.from_float(total.item())-exact),F(2*width,2**24-2*width)*exact+F(4*width,2**126))
    self.assertEqual(y.item(),sqrt_word(total.item())[0]);self.assertEqual(audit.sqrt_points,1)
 def test_guarded_step_reset_use_no_opaque_norm_or_sqrt(self):
  with patch.object(torch.linalg,'vector_norm',side_effect=AssertionError('opaque norm')),patch.object(torch.Tensor,'norm',side_effect=AssertionError('opaque norm')),patch.object(torch,'sqrt',side_effect=AssertionError('opaque sqrt')):
   worker=ConditionalVerifiedNormCPUBackend(default_profile())
   worker.reset(quaternion=[.001,-.001,.0001,1.])
   for _ in range(4):r=worker.step(torch.tensor([[4.903325,0.,0.,0.]],dtype=torch.float32))
  self.assertTrue(r['norm_sqrt_runtime_bin_verified']);self.assertFalse(r['opaque_norm_or_sqrt_library_contract_required']);self.assertFalse(r['flight_certificate']);self.assertGreater(r['norm_witness']['sqrt_points'],100)
 def test_bad_candidate_cannot_bypass_exact_bin_and_worker_latches(self):
  worker=ConditionalVerifiedNormCPUBackend(default_profile())
  with patch('verified_sqrt.math.sqrt',return_value=0.):
   with self.assertRaises(GuardFailure) as error:worker.step(torch.tensor([[4.903325,0.,0.,0.]],dtype=torch.float32))
  self.assertTrue(error.exception.state_advanced)
  with self.assertRaisesRegex(GuardFailure,'latched'):worker.step(torch.tensor([[4.903325,0.,0.,0.]],dtype=torch.float32))
 def test_new_path_guard_rejects_changed_model_step(self):
  worker=ConditionalVerifiedNormCPUBackend(default_profile());worker.plant.model.step=type(worker.plant.model.step)(lambda self,*args:None,worker.plant.model)
  with self.assertRaises(GuardFailure):worker.step(torch.tensor([[4.903325,0.,0.,0.]],dtype=torch.float32))
if __name__=='__main__':unittest.main()
