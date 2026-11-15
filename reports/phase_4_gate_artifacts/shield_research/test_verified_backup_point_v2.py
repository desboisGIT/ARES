from fractions import Fraction as F
from pathlib import Path
import unittest,ast,random
import verified_backup_point_v1 as old
import verified_backup_point_v2 as new
HERE=Path(__file__).resolve().parent
class AdaptivePointTests(unittest.TestCase):
 def test_only_atan_AST_changes(self):
  a=ast.parse((HERE/'verified_backup_point_v1.py').read_text());b=ast.parse((HERE/'verified_backup_point_v2.py').read_text())
  for tree in [a,b]:
   cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='I');cls.body=[n for n in cls.body if not isinstance(n,ast.FunctionDef) or n.name!='atan']
  self.assertEqual(ast.dump(a,include_attributes=False),ast.dump(b,include_attributes=False))
 def test_old_narrow_interval_contained_in_new_and_width_bound(self):
  rng=random.Random(6132);xs=[F(0),F(1,2),-F(1,2),F(1,10**20),-F(1,10**20),F(1,256),-F(1,256),F(1,3)]
  xs +=[F(rng.randint(-100000,100000),200000) for _ in range(200)]
  for x in xs:
   a=old.I(x).atan();b=new.I(x).atan();self.assertLessEqual(b.lo,a.lo);self.assertLessEqual(a.hi,b.hi);self.assertLessEqual(b.hi-b.lo,F(1,10**12))
  for lo,hi in [(-F(1,2),F(1,2)),(-F(1,10**20),F(1,10**20)),(F(1,4),F(1,2))]:
   a=old.I(lo,hi).atan();b=new.I(lo,hi).atan();self.assertLessEqual(b.lo,a.lo);self.assertLessEqual(a.hi,b.hi)
 def test_zero_odd_symmetry_and_domain(self):
  self.assertEqual((new.I(F(0)).atan().lo,new.I(F(0)).atan().hi),(F(0),F(0)))
  for x in [F(1,2),F(1,100),F(1,10**15)]:
   p=new.I(x).atan();n=new.I(-x).atan();self.assertEqual((n.lo,n.hi),(-p.hi,-p.lo))
  for x in [F(1,2)+F(1,10**20),-F(1,2)-F(1,10**20)]:
   with self.assertRaises(ValueError):new.I(x).atan()
 def test_old_point_cases_and_exact_output_error_contract(self):
  for sign in [-1,1]:
   for k in range(10):
    args=([sign*.003*k,-.002*k,.001*k],[-.001*k,sign*.002*k,-.001*k],[.001*k,-.0007*k,.0004*k,1.],[0.,0.,0.],4.5+.1*k)
    a=old.backup_point(*args);b=new.backup_point(*args)
    self.assertTrue(b['command_roundoff_enclosed']);self.assertFalse(b['flight_certificate'])
    for aa,bb,ea,eb in zip(a['command'],b['command'],a['exact_error'],b['exact_error']):
     self.assertLessEqual(eb,F.from_float(1e-6));self.assertLessEqual(abs(F.from_float(aa)-F.from_float(bb)),ea+eb)
 def test_rounding_cancellation_boundaries_remain_exact(self):
  for num in [-3,-1,0,1,3]:
   x=F(num,2**50);a=new.I(x).atan();out,error=new.output32(a)
   self.assertLessEqual(max(abs(F.from_float(out)-a.lo),abs(F.from_float(out)-a.hi)),error)
   self.assertLessEqual(error,F.from_float(1e-6))
if __name__=='__main__':unittest.main()
