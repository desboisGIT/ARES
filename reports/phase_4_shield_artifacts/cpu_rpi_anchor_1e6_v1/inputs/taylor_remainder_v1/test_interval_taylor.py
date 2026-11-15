"""Focused research arithmetic/domain fixtures, not a flight certificate."""
import unittest
from fractions import Fraction as F
import math
import numpy as np
import interval_taylor as t

class IntervalTaylorTests(unittest.TestCase):
 def encloses(self,interval,q):
  self.assertLessEqual(t.frac(interval.lo),q);self.assertGreaterEqual(t.frac(interval.hi),q)
 def test_sqrt_endpoints_verified_by_exact_squares(self):
  for q in [F(1,3),F(1,10**30),F(2),F(123456789,1000000)]:
   x=t.I.rational(q);lo=t.sqrt_endpoint(float(x.lo),False);hi=t.sqrt_endpoint(float(x.hi),True)
   self.assertLessEqual(t.frac(lo)**2,q);self.assertGreaterEqual(t.frac(hi)**2,q)
 def test_motor_exp_rational_series_encloses_high_precision_reference(self):
  from decimal import Decimal,localcontext
  for x in [-.134,-.05,0.]:
   with localcontext() as ctx:
    ctx.prec=100;ref=F(Decimal.from_float(x).exp())
   self.assertLessEqual(t.frac(t.exp_endpoint(x,False)),ref)
   self.assertGreaterEqual(t.frac(t.exp_endpoint(x,True)),ref)
 def test_scalar_chain_derivatives_have_exact_rational_enclosures(self):
  x=t.D.variable(t.I.rational(2),0);r=x.reciprocal()
  self.encloses(r.g[0] if hasattr(r.g,'__getitem__') else t.I(r.g.lo[0],r.g.hi[0]),F(-1,4))
  self.encloses(t.I(r.h.lo[0,0],r.h.hi[0,0]),F(1,4))
  x=t.D.variable(t.I.rational(4),0);r=x.sqrt()
  self.encloses(t.I(r.g.lo[0],r.g.hi[0]),F(1,4));self.encloses(t.I(r.h.lo[0,0],r.h.hi[0,0]),F(-1,32))
  x=t.D.variable(t.I.rational(F(1,4)),0);r=x.atan()
  self.encloses(t.I(r.g.lo[0],r.g.hi[0]),F(16,17));self.encloses(t.I(r.h.lo[0,0],r.h.hi[0,0]),F(-128,289))
 def test_mixed_polynomial_hessian(self):
  x=t.D.variable(t.I.rational(F(-1,5),F(1,5)),0)
  y=t.D.variable(t.I.rational(F(-1,10),F(1,10)),1)
  z=x.square()*y
  self.assertLessEqual(z.h.lo[0,0],-.2);self.assertGreaterEqual(z.h.hi[0,0],.2)
  self.assertLessEqual(z.h.lo[0,1],-.4);self.assertGreaterEqual(z.h.hi[0,1],.4)
  self.assertLessEqual(z.h.lo[1,0],-.4);self.assertGreaterEqual(z.h.hi[1,0],.4)
 def test_structured_and_original_real_graph_point_enclosures_overlap(self):
  d=[.001]*21;point=[(.0001 if i%2 else -.0001) for i in range(21)]
  p={'m':.5,'g':9.80665,'drag':.1,'tau':.025,'j':[.002,.002,.003],
     'jn':[float(np.float32(v)) for v in [.002,.002,.003]],'battery':.975,'delay_fraction':.75}
  a,_,_=t.backup_map(d,.005,.01,0,point=point,noise_point=[.0001]*6,parameter_values=p)
  b,_,_=t.backup_map(d,.005,.01,0,point=point,noise_point=[.0001]*6,parameter_values=p,structured_controller=True)
  for x,y in zip(a,b):
   for u,v in [(x.v,y.v),(x.g,y.g),(x.h,y.h)]:
    self.assertTrue(np.all(u.lo<=v.hi) and np.all(v.lo<=u.hi))
 def test_increment_factor_and_original_physical_real_graphs_overlap(self):
  from interval_flow import flow_map as original
  from interval_flow_increment import flow_map as factored
  d=[.001]*21;u=[.01]*4;point=[(.0001 if i%2 else -.0001) for i in range(21)]
  p={'m':.5,'g':9.80665,'drag':.1,'tau':.025,'j':[.0018,.0022,.003],
     'jn':[float(np.float32(v)) for v in [.002,.002,.003]],'battery':.975,'delay_fraction':.75}
  kwargs=dict(point=point,command_point=[.001,-.001,.001,-.001],parameter_values=p)
  a,_,_=original(d,u,0,**kwargs);b,_,_=factored(d,u,0,**kwargs)
  for x,y in zip(a,b):
   for left,right in [(x.v,y.v),(x.g,y.g),(x.h,y.h)]:
    self.assertTrue(np.all(left.lo<=right.hi) and np.all(right.lo<=left.hi))
  for x in b[12:]:self.assertTrue(np.all(x.h.lo==0) and np.all(x.h.hi==0))
 def test_bad_chart_or_heading_saturation_domain_is_rejected(self):
  d=[0.]*21;d[6]=d[7]=d[8]=.8
  with self.assertRaises(t.DomainFailure):t.backup_map(d,0.,0.,0)
  d=[0.]*21;d[8]=.2
  with self.assertRaises(t.DomainFailure):t.backup_map(d,0.,0.,0)
 def test_force_near_zero_domain_is_rejected_before_derivatives(self):
  d=[0.]*21;d[2]=20.
  with self.assertRaises(t.DomainFailure):t.backup_map(d,0.,0.,0)

if __name__=='__main__':unittest.main()
