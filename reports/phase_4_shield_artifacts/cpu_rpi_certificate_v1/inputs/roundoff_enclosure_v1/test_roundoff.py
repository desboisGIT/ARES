"""Software-RNE and analytic-normalization fixtures, not a CPU binary audit."""
import sys,unittest,random,math,struct
from pathlib import Path
from fractions import Fraction as F
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from roundoff import *

def exact_rne32(q,flush=False):
 q=F(q);guess=f32(float(q))
 candidates=[guess,float(np.nextafter(np.float32(guess),np.float32(-np.inf))),float(np.nextafter(np.float32(guess),np.float32(np.inf)))]
 def key(v):
  bits=struct.unpack('>I',struct.pack('>f',v))[0]
  return abs(F.from_float(v)-q),bits&1
 best=min(candidates,key=key)
 # Exact nearest-bin comparison against adjacent representable values.
 before=float(np.nextafter(np.float32(best),np.float32(-np.inf)));after=float(np.nextafter(np.float32(best),np.float32(np.inf)))
 selfq=F.from_float(best);low=(F.from_float(before)+selfq)/2;high=(selfq+F.from_float(after))/2
 assert low<=q<=high
 return F(0) if flush and abs(selfq)<FTZ else selfq

def point(value):return ScalarRoundInterval.input(F(value),cast=True)
class RoundoffTests(unittest.TestCase):
 def bound(self,error,actual,ideal):self.assertLessEqual(abs(actual-ideal),F.from_float(error))
 def test_basic_operations_cover_exact_software_binary32(self):
  rng=random.Random(771)
  for _ in range(200):
   a=F.from_float(rng.uniform(-12,12));b=F.from_float(rng.uniform(.2,12))
   aa=point(a);bb=point(b);a32=exact_rne32(a);b32=exact_rne32(b)
   for op,exact in [(lambda a,b:a+b,lambda a,b:a+b),(lambda a,b:a*b,lambda a,b:a*b),(lambda a,b:a/b,lambda a,b:a/b)]:
    result=op(aa,bb)
    self.bound(result.error,exact_rne32(exact(a32,b32)),exact(a,b))
 def test_gamma6_dot_encloses_reassociation_and_fma(self):
  rng=random.Random(772)
  for _ in range(100):
   a=[F.from_float(rng.uniform(-100,100)) for _ in range(3)];b=[F.from_float(rng.uniform(-100,100)) for _ in range(3)]
   ideal=sum((x*y for x,y in zip(a,b)),F(0));aa=list(map(exact_rne32,a));bb=list(map(exact_rne32,b))
   interval=dot(list(map(point,a)),list(map(point,b)))
   products=[exact_rne32(x*y) for x,y in zip(aa,bb)]
   candidates=[exact_rne32(exact_rne32(products[0]+products[1])+products[2]),exact_rne32(products[0]+exact_rne32(products[1]+products[2]))]
   acc=F(0)
   for x,y in zip(aa,bb):acc=exact_rne32(acc+x*y)
   candidates.append(acc)
   for actual in candidates:self.bound(interval.error,actual,ideal)
 def test_motor_refinement_preserves_original_three_op_error(self):
  rng=random.Random(773)
  interval=motor_update(ScalarRoundInterval.input(F('4.1'),F('5.7'),cast=True),ScalarRoundInterval.input(F('3.9'),F('6.1'),cast=True),ScalarRoundInterval.input(F('.875'),F('.952'),cast=True))
  for _ in range(200):
   old=F.from_float(rng.uniform(4.1,5.7));target=F.from_float(rng.uniform(3.9,6.1));decay=F.from_float(rng.uniform(.875,.952))
   a,b,c=map(exact_rne32,(old,target,decay))
   actual=exact_rne32(b+exact_rne32(exact_rne32(a-b)*c));ideal=target+(old-target)*decay
   self.bound(interval.error,actual,ideal)
   self.assertLessEqual(frac(interval.real.lo),ideal);self.assertLessEqual(ideal,frac(interval.real.hi))
 def test_norm_shell_rational_derivation_has_margin_below_8u(self):
  for minimum in (.5,.9,1.,1.1):self.assertLessEqual(normalize_shell_bound(minimum),8*U)
  with self.assertRaises(EnclosureFailure):normalize_shell_bound(0.)
 def test_half_ulp_sqrt_and_casts_enclose_reference(self):
  for value in [F('0.3'),F('1.3'),F(4),F(17,8)]:
   x=point(value);result=x.sqrt();a32=exact_rne32(value);guess=f32(math.sqrt(float(a32)))
   before=float(np.nextafter(np.float32(guess),np.float32(-np.inf)));after=float(np.nextafter(np.float32(guess),np.float32(np.inf)))
   # Squared bin endpoints prove this positive sqrt candidate's RNE bin.
   lo=(F.from_float(before)+F.from_float(guess))/2;hi=(F.from_float(after)+F.from_float(guess))/2
   self.assertLessEqual(lo*lo,a32);self.assertLessEqual(a32,hi*hi)
   worst=max(abs(F.from_float(guess)-frac(result.real.lo)),abs(F.from_float(guess)-frac(result.real.hi)))
   self.assertLessEqual(worst,F.from_float(result.error))
  h=ScalarRoundInterval.constant(F(1,500));self.assertGreater(h.error,0)
  self.bound(h.error,F.from_float(f32(.002)),F(1,500))
 def test_invalid_numeric_domains_fail_closed(self):
  with self.assertRaises(EnclosureFailure):point(1)/ScalarRoundInterval(I(-1,1))
  with self.assertRaises(EnclosureFailure):ScalarRoundInterval(I(-1,2)).sqrt()
  with self.assertRaises(EnclosureFailure):ScalarRoundInterval(I(-1.,-.5)).exp()
  with self.assertRaises(EnclosureFailure):normalize([point(0),point(0),point(0),point(0)])

if __name__=='__main__':unittest.main()
