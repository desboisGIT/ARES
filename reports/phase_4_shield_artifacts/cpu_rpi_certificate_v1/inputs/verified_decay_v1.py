"""Research CPU decay primitive with an exact binary32 rounding witness.

This changes only the motor-exp implementation in a separate forward simulation
backend. The original differentiable training/GPU runtime is untouched. It does
not establish the rest of the plant's roundoff budget or issue certificates.
"""
from fractions import Fraction as F
import struct


def bits_to_float(bits):
    return struct.unpack('>f',struct.pack('>I',bits))[0]


def float_to_bits(value):
    return struct.unpack('>I',struct.pack('>f',value))[0]


def verified_exp_binary32(argument):
    """Return exp(x) with exact proof of its unique nearest binary32 bin.

    The represented input must be in [-.25,0]. The alternating exponential
    series gives rational lower/upper bounds. Floating evaluation only guesses
    a bit pattern; all acceptance inequalities are exact. Unisolated bins fail.
    """
    x=F.from_float(float(argument))
    if not -F(1,4)<=x<=0:
        raise ValueError('verified motor exponent domain is [-.25,0]')
    total=term=F(1)
    for n in range(1,101):
        term*=x/n
        total+=term
        if n<8 or n%2:
            continue
        upper=total
        lower=total+term*x/(n+1)
        candidate=float_to_bits(float((upper+lower)/2))
        for bits in (candidate,candidate-1,candidate+1):
            value=bits_to_float(bits)
            previous=F.from_float(bits_to_float(bits-1))
            following=F.from_float(bits_to_float(bits+1))
            bin_lower=(previous+F.from_float(value))/2
            bin_upper=(following+F.from_float(value))/2
            if bin_lower<lower<=upper<bin_upper:
                return value,{'bits':bits,'series_terms':n+1,
                    'exact_lower':lower,'exact_upper':upper,
                    'rounding_bin_lower':bin_lower,'rounding_bin_upper':bin_upper,
                    'primitive_rounding_proved':True,'flight_certificate':False}
    raise ArithmeticError('could not isolate a unique rounding bin')


def make_adapter_class():
    import torch
    from ares.diffaero_adapter import DiffAeroAdapter

    class VerifiedDecayCPUAdapter(DiffAeroAdapter):
        """Actual full-body adapter with verified exp only; no safety claim."""
        def __init__(self,*args,**kwargs):
            if kwargs.get('device','cpu')!='cpu' or kwargs.get('compile_dynamics',False):
                raise ValueError('verified decay prototype supports eager CPU only')
            super().__init__(*args,**kwargs)
            # Preserve the original represented quotient. Its arithmetic error
            # is a separate obligation; this proves exp of that binary input.
            represented=(-self.integration_dt/self.motor_lag).detach().tolist()
            values,witnesses=zip(*(verified_exp_binary32(z) for z in represented))
            self.verified_decay=torch.tensor(values,dtype=torch.float32)
            self.decay_witnesses=witnesses

        def step(self,action):
            action=self._tensor(action,4)
            thrust=torch.minimum(action[:,0].clamp_min(0),self.max_thrust)
            rates=action[:,1:].clamp(-self.max_body_rate,self.max_body_rate)
            self.last_action=torch.cat((thrust[:,None],rates),-1)
            for _ in range(self.substeps):
                self.applied_thrust=thrust+(self.applied_thrust-thrust)*self.verified_decay
                upstream=torch.cat(((self.applied_thrust/(self.mass*self.gravity))[:,None],rates),-1)
                self.state=self.model.step(self.state,upstream,self.integration_dt)
            return self.state
    return VerifiedDecayCPUAdapter


if __name__=='__main__':
    import json
    import math
    import torch
    from ares.phase2 import default_profile
    from ares.diffaero_adapter import DiffAeroAdapter
    torch.set_num_threads(1)
    for i in range(1001):
        x=-.25*i/1000
        value,witness=verified_exp_binary32(x)
        assert witness['rounding_bin_lower']<witness['exact_lower']
        assert witness['exact_upper']<witness['rounding_bin_upper']
        # A numerical sanity check is separate from the exact rounding witness.
        assert abs(value-math.exp(x))<8e-8
    Verified=make_adapter_class();verified=Verified(default_profile(),batch_size=4,dt=.01)
    original=DiffAeroAdapter(default_profile(),batch_size=4,dt=.01)
    with torch.no_grad():
        for i in range(100):
            action=torch.tensor([[5.+.2*math.sin(i/20),.1,-.05,.02]]).expand(4,-1)
            verified.step(action);original.step(action)
    difference=float((verified.state-original.state).abs().max())
    assert difference<1e-5
    result={'verified_primitive_rounding_cases':1001,
        'full_body_CPU_steps':100,'maximum_original_state_difference':difference,
        'all_physics_equations_preserved':True,'training_runtime_unchanged':True,
        'motor_exp_primitive_proved':True,'full_float32_budget_proved':False,
        'flight_certificate':False}
    from pathlib import Path
    Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
