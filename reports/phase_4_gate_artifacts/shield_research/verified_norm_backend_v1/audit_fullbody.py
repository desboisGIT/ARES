"""128 full13D finite comparison cases; never a universal implementation proof."""
from pathlib import Path
import sys,json,hashlib,importlib.util,time
import torch
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;sys.path.insert(0,str(HERE))
from backend import ConditionalVerifiedNormCPUBackend,load
from ares.phase2 import default_profile
from ares.flight_randomization import sample_flight_profile

def run():
 start=time.monotonic();generator=torch.Generator().manual_seed(549234)
 profiles=[sample_flight_profile(default_profile(),2200+i,'low' if i==0 else 'high' if i==1 else 'random') for i in range(128)]
 worker=ConditionalVerifiedNormCPUBackend(profiles,delay_s=.0125);Native=load('norm_native_decay',BASE/'verified_decay_v1.py').make_adapter_class();native=Native(profiles,dt=.01)
 def rand(width,limits):return (torch.rand((128,width),generator=generator)*2-1)*torch.tensor(limits)
 position=rand(3,[8.,8.,8.]);velocity=rand(3,[.08,.08,.06]);quat=torch.cat((rand(3,[.01,.01,.001]),torch.ones(128,1)),-1);omega=rand(3,[.2,.2,.04])
 worker.reset(position=position,velocity=velocity,quaternion=quat,angular_velocity=omega)
 native.state=worker.plant.state.clone();motor=worker.plant.mass*worker.plant.gravity
 worker.plant.applied_thrust=motor.clone();native.applied_thrust=motor.clone()
 command=torch.cat((torch.full((128,1),4.903325),rand(3,[.2,.2,.02])),-1)
 frames=worker.delay/.01;low=frames.floor().long();high=frames.ceil().long();fraction=(frames-low)[:,None];stack=torch.stack([command]+worker.history);index=torch.arange(128)
 applied=stack[low,index]*(1-fraction)+stack[high,index]*fraction
 with torch.no_grad():
  result=worker.step(command);native.step(applied)
 error=(worker.plant.state-native.state).abs().amax(0);motorerror=float((worker.plant.applied_thrust-native.applied_thrust).abs().max())
 assert float(error.max())<2e-6 and motorerror==0
 report=dict(schema_version='ares.explicit_norm_backend_finite_comparison.v1',cases=128,original_full13D_macro_steps=128,native_comparator='Original13D/RK4 with verified-decay only; native norm/sqrt implementation',maximum_state_component_difference=error.tolist(),maximum_motor_difference=motorerror,regression_tolerance=2e-6,regression_pass=True,norm_witness=result['norm_witness'],all_sqrt_points_runtime_bin_verified=True,finite_comparison_is_not_a_global_proof=True,flight_certificate=False,source51_untouched=True,elapsed_s=time.monotonic()-start)
 (HERE/'fullbody_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));return report
if __name__=='__main__':run()
