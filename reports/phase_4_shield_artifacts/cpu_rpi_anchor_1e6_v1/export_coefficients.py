"""Replayable exact-to-outward coefficient export for the conditional CPU RPI.

The proof checker must still verify every exported LMI; this exporter proves
rational coefficient containment and the stage-affine polynomial construction.
"""
from pathlib import Path
from fractions import Fraction as F
import hashlib,json,itertools,time
import numpy as np
HERE=Path(__file__).resolve().parent;INPUTS=HERE/'inputs';start=time.monotonic();conversion_checks=0
for name,expected in json.loads((INPUTS/'dependency_sha256.json').read_text()).items():
 assert hashlib.sha256((INPUTS/name).read_bytes()).hexdigest()==expected,name
text=(INPUTS/'backup_bernstein_embedding.py').read_text().split('# Exact rational physical coefficient intervals')[0]
text=text.replace("OUT=Path('reports/phase_4_learning_artifacts/diagnosis/shield_design')","OUT=RESEARCH_DIR")
ns={'RESEARCH_DIR':INPUTS};exec(text,ns);Poly=ns['Poly'];mb=ns['matrix_bern'];ctl=ns['controlled'];delays=ns['delayvertices']();meta=json.loads((INPUTS/'backup_common_P_feasibility.json').read_text());noise=json.loads((INPUTS/'backup_linear_noise_RPI_feasibility.json').read_text());phy=json.loads((INPUTS/'backup_Bernstein_embedding_feasibility.json').read_text())['outward_float32_physical_bounds'];ML,MH=map(F,phy['mass']);GL,GH=map(F,phy['gravity']);rs=json.loads((INPUTS/'backup_quantized_exact_oriented_remainder_audit.json').read_text());num=json.loads((INPUTS/'conditional_projective21_budget.json').read_text());ctrl=list(map(F,rs['controller_joint_remainder_rational']));physical=[max(F(row[i]) for row in rs['sector_flow_remainder_rational'])+F(num['component_error_upper_rational'][i]) for i in range(21)]
for i in range(3):physical[i]+=F(1,1000000)
for i,j in [(0,1),(3,4),(6,7),(9,10)]:physical[i]=physical[j]=max(physical[i],physical[j])
dB=F(1,156250);dT=MH*GH/(F(19,20)**2)*dB;drift=[F(0)]*21;drift[2]=F(4,3)*dT;drift[13]=drift[17]=-dT
candidate=json.loads((INPUTS/'backup_conditional_CPU_anchor_1_1000000_candidate.json').read_text());chosen={}
for rec in candidate['records']:
 if rec.get('Q_scaled') and rec['iterations'][-1].get('worst_float_residual_eig',-1)>=9e-9:chosen.setdefault(rec['group'],rec)
assert set(chosen)=={'planar_x','vertical','yaw'}

def frac(v):return F.from_float(float(v))
def bracket(v):
 global conversion_checks
 x=F(v);middle=float(x)
 if frac(middle)==x:lo=hi=middle
 else:lo=float(np.nextafter(middle,-np.inf));hi=float(np.nextafter(middle,np.inf))
 assert frac(lo)<=x<=frac(hi)
 conversion_checks+=1
 return lo,hi
def array_bracket(values):
 a=np.array(values,dtype=object);l=np.empty(a.shape);h=np.empty(a.shape)
 for idx in np.ndindex(a.shape):l[idx],h[idx]=bracket(a[idx])
 return l,h
def add(a,b):return np.nextafter(a[0]+b[0],-np.inf),np.nextafter(a[1]+b[1],np.inf)
def multiply_constant(a,c):
 c=float(c)
 if c>=0:return np.nextafter(a[0]*c,-np.inf),np.nextafter(a[1]*c,np.inf)
 return np.nextafter(a[1]*c,-np.inf),np.nextafter(a[0]*c,np.inf)
def scale_matrix(a,s):return [[v*s[j]/s[i] for j,v in enumerate(row)] for i,row in enumerate(a)]
def scale_vector(a,s):return [v/s[i] for i,v in enumerate(a)]
def inverse(a):
 n=len(a);x=[row.copy()+[F(int(i==j)) for j in range(n)] for i,row in enumerate(a)]
 for j in range(n):
  p=x[j][j];x[j]=[v/p for v in x[j]]
  for i in range(n):
   if i!=j:
    c=x[i][j];x[i]=[v-c*w for v,w in zip(x[i],x[j])]
 return [row[n:] for row in x]
def core_parse(rows):return [[Poly({(i,j):F(v) for i,j,v in p}) for p in row] for row in rows]
def generator_control(h,C,d):
 a=ctl(h,C,d);a[-2]=[F(0)]*6;a[-1]=[F(0)]*6;return a
def vec(h,d,queue=True):
 n=len(h)-1
 return [h[i][n]*d[0] for i in range(n)]+[F(int(queue)),F(0)]
def offsets_for(name,s):
 idx=meta['groups'][name]['indices'];eps=[physical[j]/s[i] for i,j in enumerate(idx)];dv=[drift[j]/s[i] for i,j in enumerate(idx)];active=[j for j,x in enumerate(eps) if x];offsets=[]
 for signs in itertools.product([-1,1],repeat=len(active)):
  e=eps.copy()
  for j,sign in zip(active,signs):e[j]*=sign
  for sign in ([-1,1] if any(dv) else [0]):offsets.append([x+sign*d for x,d in zip(e,dv)])
 keys={tuple(r) for r in offsets};assert all(tuple(-v for v in r) in keys for r in offsets)
 return offsets

generator=json.loads((INPUTS/'sector_exact_generator_polynomials.json').read_text());ordinary=json.loads((INPUTS/'backup_local_exact_power_coefficients.json').read_text());report={};DG=F(7,8)
for name in ['planar_x','vertical','yaw']:
 info=meta['groups'][name];s=list(map(frac,info['state_scaling']));n=len(s);Q=[[frac(v) for v in row] for row in chosen[name]['Q_scaled']];alpha=frac(chosen[name]['alpha']);qiv=array_bracket(Q)
 if name=='yaw':p=[[frac(v) for v in row] for row in info['scaled_P']];pi=inverse(p);outer=[[F(9,400)*v for v in row] for row in pi]
 else:baseline=next(r for r in noise['records'] if r['group']==name and r['tau']==.99);outer=[[F(9,4)*frac(v) for v in row] for row in baseline['Q_scaled']]
 oiv=array_bracket(outer);offsets=offsets_for(name,s);offiv=array_bracket(offsets)
 if name=='planar_x':
  cores=[core_parse(generator[k]) for k in ['H0']+[f'H_stage_{j}' for j in range(5)]];Bs=[mb(core,20,20) for core in cores];cs=[F(19,20)*F(1,2)/(MH*GH),F(1,2)/(ML*GL)];A0l=[];A0h=[];W0l=[];W0h=[];Ajl=[];Ajh=[];Wjl=[];Wjh=[]
  for i in range(21):
   for j in range(21):
    for c in cs:
     C=[-9*c,-15*c,F(-12),F(0)];nu=6*c*F(13,400)+max(ctrl[1:3])+F(1,1000000)
     for d in delays:
      raw=ctl(Bs[0][i][j],C,d);a=array_bracket(scale_matrix(raw,s));w=array_bracket(scale_vector([v*nu for v in vec(Bs[0][i][j],d)],s));A0l.append(a[0]);A0h.append(a[1]);W0l.append(w[0]);W0h.append(w[1]);al=[];ah=[];wl=[];wh=[]
      for B in Bs[1:]:
       a=array_bracket(scale_matrix(generator_control(B[i][j],C,d),s));w=array_bracket(scale_vector([v*nu for v in vec(B[i][j],d,False)],s));al.append(a[0]);ah.append(a[1]);wl.append(w[0]);wh.append(w[1])
      Ajl.append(al);Ajh.append(ah);Wjl.append(wl);Wjh.append(wh)
  baseA=(np.array(A0l),np.array(A0h));baseW=(np.array(W0l),np.array(W0h));genA=(np.array(Ajl),np.array(Ajh));genW=(np.array(Wjl),np.array(Wjh));a_parts=[];w_parts=[]
  for signs in itertools.product([-1,1],repeat=5):
   av=baseA;wv=baseW
   for j,sign in enumerate(signs):av=add(av,multiply_constant((genA[0][:,j],genA[1][:,j]),sign*DG));wv=add(wv,multiply_constant((genW[0][:,j],genW[1][:,j]),sign*DG))
   a_parts.append(av);w_parts.append(wv)
  # Base-major ordering matches polynomial parameter vertices then 32 gains.
  A=(np.stack([x[0] for x in a_parts],axis=1).reshape(-1,n,n),np.stack([x[1] for x in a_parts],axis=1).reshape(-1,n,n));W=(np.stack([x[0] for x in w_parts],axis=1).reshape(-1,n),np.stack([x[1] for x in w_parts],axis=1).reshape(-1,n))
 else:
  core=core_parse(ordinary['vertical_invm1' if name=='vertical' else 'yaw']);B=mb(core,20,5) if name=='vertical' else mb(core,0,20);als=[];ahs=[];wls=[];whs=[]
  for row in B:
   for core in row:
    variants=[(inv,b,d) for inv in [1/MH,1/ML] for b in [F(19,20),F(1)] for d in delays] if name=='vertical' else delays
    for var in variants:
     h=[r.copy() for r in core]
     if name=='vertical':
      inv,b,d=var
      for i in [0,1]:
       for j in [2,3]:h[i][j]*=inv
      for i in [0,1,2]:h[i][3]*=b
      C=[F(-3,4),F(-5,4),F(0)];nu=F(13,800)+ctrl[0]+F(1,1000000)
     else:d=var;C=[F(-12),F(0)];nu=ctrl[3]+F(1,1000000)
     a=array_bracket(scale_matrix(ctl(h,C,d),s));w=array_bracket(scale_vector([v*nu for v in vec(h,d)],s));als.append(a[0]);ahs.append(a[1]);wls.append(w[0]);whs.append(w[1])
  A=(np.array(als),np.array(ahs));W=(np.array(wls),np.array(whs))
 path=HERE/(name+'_interval_coefficients.npz');np.savez_compressed(path,A_lo=A[0],A_hi=A[1],command_w_lo=W[0],command_w_hi=W[1],additive_offsets_lo=offiv[0],additive_offsets_hi=offiv[1],Q_lo=qiv[0],Q_hi=qiv[1],outer_Q_lo=oiv[0],outer_Q_hi=oiv[1]);metadata={'schema_version':'ares.outward_RPI_LMI_coefficients.v1','group':name,'alpha_exact':str(alpha),'parameter_vertices':len(A[0]),'offset_sign_patterns':len(offsets),'expected_LMIs_one_command_sign':len(A[0])*len(offsets),'all_exact_rational_to_binary64_conversion_brackets_checked':True,'subsequent_sector_vertex_operations':'Each multiplication/addition uses binary64 RN then nextafter toward both infinities. Primitive IEEE contract required; matrices are not inferred from point spectra.','offsets_exact_rational':[list(map(str,row)) for row in offsets],'offsets_centrally_symmetric_verified_exact':True,'global_negative_command_sign_reduction':'For any w=−command+offset, −w=command−offset; centrally symmetric offsets include−offset and ww^T is unchanged. The checker may independently check both command signs.','physical_state_indices':info['indices'],'state_scales_exact_binary64':info['state_scaling'],'flight_certificate':False,'npz_sha256':hashlib.sha256(path.read_bytes()).hexdigest()};(HERE/(name+'_interval_coefficients.json')).write_text(json.dumps(metadata,indent=2)+'\n');report[name]=metadata;print('EXPORTED',name,len(A[0]),'offsets',len(offsets),'alpha',alpha,'elapsed',time.monotonic()-start,flush=True)
result={'schema_version':'ares.exact_outward_export.v1','exact_conversion_checks':conversion_checks,'groups':report,'replay_CWD_independent':True,'sources_verified_before_export':True,'flight_certificate':False,'elapsed_s':time.monotonic()-start};(HERE/'export_audit.json').write_text(json.dumps(result,indent=2)+'\n');print('DONE',conversion_checks,time.monotonic()-start,flush=True)
