"""Public command-only motor/energy upper bounds; no sampled motor/E inputs."""
from fractions import Fraction as F
from pathlib import Path
import sys,math
BASE=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(BASE))
import verified_battery_v1 as primitive
from zero_rate_history_v1 import broad_motor_arithmetic,q

def ceil_grid(x,bits=40):
 x=q(x);scale=2**bits;return F(-((-x*scale).numerator//(-x*scale).denominator),scale)
def sqrt_upper(x):
 x=q(x);scale=2**60;lo=F(math.isqrt(x.numerator*scale*scale//x.denominator),scale);hi=lo if lo*lo==x else lo+F(1,scale)
 if not lo*lo<=x<=hi*hi:raise ArithmeticError('integer sqrt endpoint verification failed')
 return hi

def updater_bound(thrust_upper=F('5.474')):
 """Same verified power/updater primitives, tighter PUBLIC input upper T."""
 T=q(thrust_upper)
 if not 0<=T<=F(13,2):raise ValueError('power proof requires public motor upper in[0,6.5]')
 U,eta=primitive.U,primitive.ETA;H=primitive.frac(primitive.word(primitive.bits(.01)));c=primitive.frac(primitive.word(primitive.bits(.2)))
 power=T*sqrt_upper(T);power_word=power*(1+U)+eta
 product=power_word*H*(1+U)+eta;increment=product*8*(1+U)+eta
 jump=increment+U*(10000+increment)+eta
 def authority_error(E):
  ratio=E/F(43200);equot=U*ratio+eta;eprod=c*equot+U*c*(ratio+equot)+eta;return eprod+U+eta
 lower=1-c*F(10000,43200)-authority_error(10000)
 decrease=c*jump/F(43200)+2*authority_error(10000+jump)
 if not lower>F(19,20) or not decrease<F(64,10000000) or not 8000+1000*jump<10000:raise ArithmeticError('public finite-horizon energy bound failed')
 return dict(thrust_upper=T,energy_jump_upper=jump,authority_lower=lower,authority_decrease_upper=decrease,old_uniform_decline_bound_preserved=True,ten_second_energy_domain_from_8000=True,flight_certificate=False)

def bootstrap_bound(*,maximum_issued_thrust,steps,mass_gravity_upper):
 """Bounds from known commands plus full parameter SET, never sampled mg."""
 command=q(maximum_issued_thrust);mg=q(mass_gravity_upper)
 if type(steps) is not int or steps<0 or mg<=0 or not 0<command<=F(13,2):raise ValueError('positive public command/step bounds required')
 broad=broad_motor_arithmetic();motor=max(mg*(1+primitive.U)+primitive.ETA,command)+broad['standalone_settle_motor_error_per_H']/(1-broad['macro_decay_upper'])
 motor=ceil_grid(motor,20)
 energy=ceil_grid(steps*updater_bound(motor)['energy_jump_upper'])
 return dict(motor_upper=motor,energy_upper=energy,public_steps=steps,public_issued_thrust_upper=command,no_motor_or_energy_truth=True,flight_certificate=False)

class PublicEnergyLedger:
 """One acknowledged macro update at a time, bounded actual motor via words."""
 def __init__(self,energy_upper,motor_upper):
  self.energy=q(energy_upper);self.motor=q(motor_upper);self.ticks=0
  if not 0<=self.energy<=10000 or not 0<self.motor<=F(13,2):raise ValueError('valid public E/M upper bounds required')
 def advance(self,issued_command,previous_queue):
  if len(issued_command)!=4 or len(previous_queue)!=2 or any(len(w)!=4 for w in previous_queue):raise ValueError('SI command and two stored previous words required')
  targets=[q(w[0]) for w in [issued_command,*previous_queue]]
  if any(not 0<=t<=F(13,2) or primitive.frac(primitive.word(primitive.bits(t)))!=t for t in targets):raise ValueError('known force words outside power domain')
  target=max(targets);a_lo,a_hi=F(51,100),F(79,100)
  motor=max(a*self.motor+(1-a)*target for a in [a_lo,a_hi])+F(7,1000000)
  motor=ceil_grid(motor)
  if not 0<motor<=F(13,2):raise ValueError('public recurrence leaves verified motor domain')
  energy=ceil_grid(self.energy+updater_bound(motor)['energy_jump_upper'])
  if energy>10000:raise ValueError('public energy proof domain expired')
  self.motor,self.energy=motor,energy;self.ticks+=1
  return dict(motor_upper=self.motor,energy_upper=self.energy,renewal_energy_admissible=self.energy<=8000,flight_certificate=False)
