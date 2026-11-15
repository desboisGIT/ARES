"""Versioned adaptive-atan compatibility + public-point microbenchmark."""
from pathlib import Path
from fractions import Fraction as F
import ast,json,hashlib,random,time,statistics,gc,sys,struct,math
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE/'terminal_reservation_v1'))
import verified_backup_point_v1 as old
import verified_backup_point_v2 as new
from current_terminal_membership_v1 import current_terminal_membership

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def asttext(n):return ast.dump(n,include_attributes=False)
def compatibility():
 trees=[ast.parse((HERE/f'verified_backup_point_v{v}.py').read_text()) for v in [1,2]]
 methods=[]
 for tree in trees:
  cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='I');methods.append({n.name:asttext(n) for n in cls.body if isinstance(n,ast.FunctionDef)})
 assert methods[0].keys()==methods[1].keys()
 equal={n:methods[0][n]==methods[1][n] for n in methods[0] if n!='atan'};assert all(equal.values())
 functions={name:asttext(next(n for n in trees[0].body if isinstance(n,ast.FunctionDef) and n.name==name))==asttext(next(n for n in trees[1].body if isinstance(n,ast.FunctionDef) and n.name==name)) for name in ['backup_point','output32']};assert all(functions.values())
 for tree in trees:
  cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='I');cls.body=[n for n in cls.body if not isinstance(n,ast.FunctionDef) or n.name!='atan']
 assert asttext(trees[0])==asttext(trees[1])
 assert F(1,2)**67/67<F(1,10**12)
 report=dict(schema_version='ares.verified_point_kernel_v2_compatibility.v1',all_module_AST_except_I_atan_identical=True,I_methods_except_atan_identical=equal,backup_point_output32_AST_identical=functions,maximum_series_terms=33,exact_early_stop_width='1/1000000000000',universal_final_term_width_upper=str(F(1,2)**67/67),same_REAL_law_gains_domains_and_actuator_branches=True,same_exact_uniform_command_error_contract='output32 checks each final command against Fraction.from_float(1e-6) unchanged',existing_RPI_numeric_command_error_contract_unchanged=True,proof='For0<=x<=1/2 the alternating atan terms decrease to0; partial sum and next signed term bracket the exact atan. The interval width is exactly the next term. Stop only at exact width<=1e-12 or at33terms, whose worst-case next term is(1/2)^67/67<1e-12. Negative arguments reflect the enclosure. Atan is monotone, so endpoint brackets enclose each interval. The unchanged output32 then verifies the final1e-6 contract independently of any early stop.',flight_certificate=False,source_sha256={f'verified_backup_point_v{v}.py':digest(HERE/f'verified_backup_point_v{v}.py') for v in [1,2]})
 (HERE/'verified_backup_point_v2_compatibility.json').write_text(json.dumps(report,indent=2)+'\n');return report

def shapes():
 c=json.loads((ROOT/'reports/phase_4_shield_artifacts/cpu_rpi_certificate_v1/certificate.json').read_text());out={}
 for name,g in c['groups'].items():
  scale=list(map(F,g['state_scales_exact_rational']));out[name]=[[F(x)*scale[i]*scale[j] for j,x in enumerate(row)] for i,row in enumerate(g['Q_scaled_exact_rational'])]
 return out

def count_terms(x):
 x=abs(x);power=x
 for k in range(33):
  following=power*x*x
  if following/(2*k+3)<=F(1,10**12) or k==32:return k+1
  power=following

def quantile(values,p):
 a=sorted(values);return a[min(len(a)-1,int((len(a)-1)*p))]
def stats(values):return dict(calls=len(values),minimum_us=min(values)*1e6,median_us=statistics.median(values)*1e6,p95_us=quantile(values,.95)*1e6,p99_us=quantile(values,.99)*1e6,maximum_us=max(values)*1e6)
def benchmark():
 admission=json.loads((HERE/'initial_contract_v1.json').read_text())['cases'][0];packet=admission['public_final_packet'];anchor=admission['anchor'];hold=admission['hold_thrust']
 actual=(packet['p'],packet['v'],packet['q'],anchor,hold)
 rng=random.Random(74923);ss=shapes();cases=[];maxquad=F(0);minquad=F(1)
 def f32(x):return struct.unpack('f',struct.pack('f',x))[0]
 for _ in range(128):
  delta=[rng.uniform(-.005,.005) for j in range(3)];v=[rng.uniform(-.003,.003) for j in range(3)];xyz=[rng.uniform(-.0002,.0002),rng.uniform(-.0002,.0002),rng.uniform(-.0001,.0001)]
  scale=1.
  for attempt in range(40):
   p=[f32(float(anchor[j])+scale*delta[j]) for j in range(3)];vv=[f32(scale*x) for x in v];qx=[f32(scale*x) for x in xyz];quat=qx+[f32(math.sqrt(1-sum(x*x for x in qx)))]
   result=current_terminal_membership(p=p,v=vv,q=quat,w=[0.,0.,0.],anchor=anchor,hold_thrust=hold,issued_queue=[[hold,0.,0.,0.]]*2,trim_force_interval=(0,0),motor_residual_interval=(0,0),shapes=ss,ep=0,ev=0,ew=0,q_state_overapprox_error=0,battery_interval=(1,1))
   quadratic=max(r['maximum'] for r in result['channels'].values())
   if F(1,5)<=quadratic<=F(9,10):
    assert result['current_set_included'];break
   scale*=1.4 if quadratic<F(1,5) else .8
  else:raise AssertionError('could not construct exact admitted C point in prescribed annulus')
  maxquad=max(maxquad,quadratic);minquad=min(minquad,quadratic);cases.append((p,vv,quat,anchor,hold))
 largest=F(0);different=0;terms=[]
 for args in [actual]+cases:
  a=old.backup_point(*args);b=new.backup_point(*args)
  for x,y,ea,eb in zip(a['command'],b['command'],a['exact_error'],b['exact_error']):
   assert eb<=F.from_float(1e-6) and abs(F.from_float(x)-F.from_float(y))<=ea+eb
   largest=max(largest,eb);different+=x!=y
  qx,qy,qz,qw=map(F.from_float,args[2]);ratio=2*(qx*qy+qw*qz)/(qw*qw+qx*qx-qy*qy-qz*qz);terms.append(count_terms(ratio))
 for _ in range(12):old.backup_point(*actual);new.backup_point(*actual)
 times={1:[],2:[]};randomtimes={1:[],2:[]}
 enabled=gc.isenabled();gc.disable()
 try:
  for i in range(300):
   for version,fn in ([(1,old.backup_point),(2,new.backup_point)] if i%2==0 else [(2,new.backup_point),(1,old.backup_point)]):
    start=time.perf_counter();fn(*actual);times[version].append(time.perf_counter()-start)
  for rep in range(3):
   for i,args in enumerate(cases):
    for version,fn in ([(1,old.backup_point),(2,new.backup_point)] if (rep+i)%2==0 else [(2,new.backup_point),(1,old.backup_point)]):
     start=time.perf_counter();fn(*args);randomtimes[version].append(time.perf_counter()-start)
 finally:
  if enabled:gc.enable()
 report=dict(schema_version='ares.verified_point_kernel_v2_benchmark.v1',actual_public_admission_case=0,random_points=128,random_points_exact_C_membership_checked=True,maximum_random_point_C_quadratic=str(maxquad),minimum_random_point_C_quadratic=str(minquad),random_public_points_are_represented_binary32=True,all_output32_exact_error_checks_pass=True,maximum_certified_command_error=str(largest),number_different_command_words=different,adaptive_atan_term_counts=dict(minimum=min(terms),maximum=max(terms),median=statistics.median(terms),old=33),actual_case0={f'v{k}':stats(v) for k,v in times.items()},random_local_C={f'v{k}':stats(v) for k,v in randomtimes.items()},case0_median_speedup=statistics.median(times[1])/statistics.median(times[2]),random_median_speedup=statistics.median(randomtimes[1])/statistics.median(randomtimes[2]),finite_timing_only_no_deadline_guarantee=True,no_physics_domain_noise_changes=True,flight_certificate=False,source_sha256={n:digest(HERE/n) for n in ['verified_backup_point_v1.py','verified_backup_point_v2.py','initial_contract_v1.json','verified_backup_point_v2_audit.py']})
 (HERE/'verified_backup_point_v2_benchmark.json').write_text(json.dumps(report,indent=2)+'\n');return report
if __name__=='__main__':
 compatibility();print(json.dumps(benchmark(),indent=2))
