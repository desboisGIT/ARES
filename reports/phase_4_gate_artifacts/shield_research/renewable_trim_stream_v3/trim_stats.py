"""Exact sufficient statistics for the unchanged sampled trim lemma."""
from pathlib import Path
from fractions import Fraction as F
from functools import lru_cache
import sys
BASE=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(BASE));sys.path.insert(0,str(BASE/'renewable_trim_v1'))
from zero_rate_history_v1 import q,broad_motor_arithmetic,decay_upper,rational_domain
from energy import ceil_grid
import producerwide_v2 as wide
H=F(1,100);ALLOWED_HISTORY=(F(1,2),F(1),F(3,2),F(2))

@lru_cache(maxsize=1)
def domain_constants():
 d=rational_domain();rate=wide.small_rate_stage_lemma();omega=rate['omega_l2_upper'];h=F(1,500);a=h/4;A=a*a*omega*omega
 E2=A;E3=3*A+A*A;E4=4*A+4*a**3*omega**3+4*A*(1+E3)
 N=1/(1-h*omega);D=h*omega*N/2;drift=5*(2*D/(1-D))+D
 return d,rate,max(E2,E3,E4),drift,broad_motor_arithmetic()

class TrimStatisticsEvaluator:
 """Precompute all large static rationals; evaluate exact endpoint/max data."""
 def __init__(self,proof,target,history_seconds=F(1)):
  self.duration=q(history_seconds)
  if self.duration not in ALLOWED_HISTORY:raise ValueError('history_seconds must be explicitly .5,1,1.5 or2')
  self.intervals=int(self.duration/H);self.hold=q(target);self.proof_sha256=proof.source_sha256
  p=proof.certificate['physical_parameter_intervals_outward'];self.mass_lo,self.mass_hi=map(F,p['mass']);self.lag=F(p['motor_lag'][1]);drag=F(p['drag'][1]);db=proof.authority_drop
  h=F(1,500);ev=F(1,100);force=F(13,2);omega=F(1);up=F(1,5000);epsv=F(53,100000000);epsf=F(1,50000)
  self.drag_factor=1/(1-h*omega)**4;self.gamma=drag*self.drag_factor/self.mass_lo;self.battery_rate=db/H;self.numeric_rate=epsv/H
  if 2*self.gamma*H>=1:raise ValueError('sample gap cannot close trim estimate')
  motor_decay=decay_upper((F(7,10)-F(2000001,100000000)-H)/self.lag);macro_decay=decay_upper(H/self.lag)
  self.motor_error=motor_decay*(force+self.hold)+self.hold*self.battery_rate*(self.lag+H)+epsf/(1-macro_decay)
  self.fixed=(self.motor_error+(self.hold+self.motor_error)*up)/self.mass_lo
  self.drift=self.hold*self.battery_rate*(self.duration+H)/self.mass_lo;self.denominator=1-2*self.gamma*H
  self.endpoint_constant=2*ev/self.duration+self.numeric_rate;self.speed_constant=ev+self.numeric_rate*H
  a=1+self.gamma*H/self.denominator;b=self.gamma+2*self.gamma*self.gamma*H/self.denominator;c=1+2*self.gamma*H/self.denominator
  self.delta_coefficient=a/self.duration;self.max_coefficient=b;self.base=a*self.endpoint_constant+b*self.speed_constant+c*(self.fixed+self.drift)
  self.motor_dyadic=ceil_grid(self.motor_error)
  d,rate,nerr,coordinate_drift,broad=domain_constants();res=(force+self.hold)*decay_upper(F(66,100)/self.lag)+broad['standalone_settle_motor_error_per_H']/(1-broad['macro_decay_upper'])
  self.absolute_motor=(F(19,20)*self.hold-res,self.hold+res);mr=d['motor_real_range']
  self.target_domains_pass=q(mr['lo'])<=self.absolute_motor[0]<=self.absolute_motor[1]<=q(mr['hi']) and all(q(group[0]['lo'])<=self.hold<=q(group[0]['hi']) and all(q(v['lo'])<=-F(1,50)<=F(1,50)<=q(v['hi']) for v in group[1:]) for group in [d['current_word_real_ranges'],*d['history_word_real_ranges']])
 def evaluate(self,*,start_vz,end_vz,max_abs_vz):
  start,end,maximum=map(q,[start_vz,end_vz,max_abs_vz])
  if maximum<max(abs(start),abs(end)):raise ValueError('max statistic does not contain endpoints')
  mismatch=self.delta_coefficient*abs(end-start)+self.max_coefficient*maximum+self.base
  endpoint=abs(end-start)/self.duration+self.endpoint_constant;known=maximum+self.speed_constant
  accel=(endpoint+2*self.gamma*known+2*self.fixed+2*self.drift)/self.denominator;speed=known+accel*H
  return dict(conditional_trim_acceleration_bound=mismatch,conditional_stage_acceleration_bound=accel,conditional_stage_vertical_speed_bound=speed,motor_tracking_force_bound=self.motor_error,stage_drag_multiplier_upper=self.drag_factor,position_offset_coefficient_per_mass_upper=1/(F(19,20)*F(1,2)*F(3,2)),history_duration=self.duration,maximum_packet_gap=H,all_decisions_exact_rational=True,uses_continuous_ODE_solution=False,hypotheses_independently_established=False,flight_certificate=False)
