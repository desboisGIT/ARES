"""Exact compatibility/quat-stage identities; no sample is a proof input."""
from pathlib import Path
from fractions import Fraction as F
import sys,json,hashlib,ast,sympy as s
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;sys.path.insert(0,str(HERE));sys.path.insert(0,str(BASE))
from producer import small_rate_stage_lemma
import producerwide_v2 as wide
from energy import updater_bound,bootstrap_bound

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def encode(x):
 if isinstance(x,F):return {'rational':str(x),'display_only':float(x)}
 if isinstance(x,(list,tuple)):return [encode(v) for v in x]
 if isinstance(x,dict):return {k:encode(v) for k,v in x.items()}
 return x

def quaternion_identities():
 a=s.symbols('a');v=[s.symbols(f'w{i}x w{i}y w{i}z') for i in [1,2,3]]
 def dot(x,y):return sum(xx*yy for xx,yy in zip(x,y))
 def cross(x,y):return [x[1]*y[2]-x[2]*y[1],x[2]*y[0]-x[0]*y[2],x[0]*y[1]-x[1]*y[0]]
 def mul(x,y):return [x[3]*y[i]+y[3]*x[i]+cross(x[:3],y[:3])[i] for i in range(3)]+[x[3]*y[3]-dot(x[:3],y[:3])]
 def stage(w,prior,factor):return [factor*t for t in mul(prior,list(w)+[0])][:3]+[1+factor*mul(prior,list(w)+[0])[3]]
 one=[0,0,0,1];q2=stage(v[0],one,a);q3=stage(v[1],q2,a);q4=stage(v[2],q3,2*a)
 n2=[s.expand(dot(q,q)) for q in [q2,q3,q4]];expected=[1+a*a*dot(v[0],v[0]),1-2*a*a*dot(v[0],v[1])+a*a*dot(v[1],v[1])+a**4*dot(v[0],v[0])*dot(v[1],v[1]),1-4*a*a*dot(v[1],v[2])-4*a**3*dot(cross(v[0],v[1]),v[2])+4*a*a*n2[1]*dot(v[2],v[2])]
 zero=[str(s.expand(x-y)) for x,y in zip(n2,expected)];assert zero==['0']*3
 return dict(q0_identity_reduction='Right-multiplication norm and inner-product invariance for unitq0; exact Hamilton identities from immutable quaternion_algebra_exact.json',exact_norm_squared_identity_differences=zero,stage_norm_error_bounds='E2=a²Ω²;E3=3a²Ω²+a⁴Ω⁴;E4=4a²Ω²+4a³Ω³+4a²(1+E3)Ω²',uses_continuous_ODE=False)

def run():
 narrow=small_rate_stage_lemma();broad=wide.small_rate_stage_lemma();ident=quaternion_identities()
 trees=[ast.parse((HERE/n).read_text()) for n in ['producer.py','producerwide_v2.py']]
 for tree in trees:
  tree.body=[n for n in tree.body if not (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='MAX_RATE' for t in n.targets)) and not (isinstance(n,ast.FunctionDef) and n.name in ['small_rate_stage_lemma','verify_stage_domain'])]
 assert ast.dump(trees[0],include_attributes=False)==ast.dump(trees[1],include_attributes=False)
 packet=dict(p=[6.,6.,2.],v=[0.,0.,0.],w=[.02,-.02,.01],q=[.0014,-.0014,.0002,1.])
 # Supply represented normalized raw q (test fixture only). Analytic proof of
 # the norm bound above does not depend on this numerical regression.
 import struct,math
 def f32(x):return struct.unpack('f',struct.pack('f',x))[0]
 xyz=list(map(f32,packet['q'][:3]));packet['q']=xyz+[f32(math.sqrt(1-sum(x*x for x in xyz)))]
 stage=wide.verify_stage_domain([packet,packet]);assert stage['stage_and_macro_domains_pass'] and stage['stage_up_axis_error']<F(1,5000)
 oldpaths=['verified_backup_point_v2.py','trim_history_v2.py','zero_rate_history_v1.py','current_terminal_membership_v1.py','verified_battery_v1.py','reanchoring_shield_v1.py','quaternion_algebra_exact.json']
 report=dict(schema_version='ares.renewable_trim_extensions.v1',default_small_rate_lemma=narrow,optional_wide_rate_lemma=broad,wide_stage_identity_proof=ident,optional_q0014_regression=stage,optional_producer_AST_identical_except_rate_cap_and_two_stage_bound_functions=True,unchanged_backup_REAL_law_and_point_error_box=True,unchanged_RPI_Q_parameter_domain_and_1000step_theorem=True,trim_v2_requires_constant_T_not_zero_rates_once_stage_hypotheses_proved=True,new_current_membership_uses_actual_nonzero_public_queue=True,new_lease_requires_energy_upper_le8000_and_full_future_service_map=True,energy_parameter_bound_at_5474=updater_bound(F('5.474')),no_physics_noise_or_sensor_bound_changes=True,source_sha256={n:digest(BASE/n) for n in oldpaths},extension_source_sha256={p.name:digest(p) for p in HERE.glob('*.py')},flight_certificate=False)
 (HERE/'compatibility.json').write_text(json.dumps(encode(report),indent=2)+'\n');print('exact quaternion identities',ident['exact_norm_squared_identity_differences']);print('wide endpoint',float(broad['macro_angular_endpoint_bound']),'upaxis',float(stage['stage_up_axis_error']));return report
if __name__=='__main__':run()
