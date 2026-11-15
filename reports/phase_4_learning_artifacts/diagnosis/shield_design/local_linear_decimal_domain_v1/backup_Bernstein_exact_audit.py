"""Exact rational local-model polynomial stability proof; NOT nonlinear RPI."""
from pathlib import Path
import json,time,numpy as np,itertools
ns={};exec(Path('/tmp/ares_backup_bernstein_embedding.py').read_text(),ns);F=ns['F'];OUT=ns['OUT'];meta=ns['meta'];start=time.monotonic()
# This direct expression crosscheck uses the actual unmodified full plant and
# controller autograd; it is independent numerical regression, not the proof.
plant={};exec(Path('/tmp/ares_backup_linear_feasibility.py').read_text().split('\nnom=')[0],plant);torch=plant['torch'];model_step=plant['model_step'];z=torch.zeros(21,requires_grad=True)
def expression(m,g,drag,tau,J,delay):
 delta=drag/m;t=(delta-1/11)/(4/9-1/11);kr=30*.002/J[1];u=(kr-300/11)/(100/3-300/11);h=[[F.from_float(p.evaluate(t,u)) for p in row] for row in ns['planar']];d=(F(1,2),F(1,2),F(0)) if delay==.005 else (F(0),F(1),F(0)) if delay==.01 else (F(0),F(0),F(1));ax=np.array(ns['planar_decorate'](h,(F.from_float(g),F.from_float(.5/(m*g)),d)),float)
 alo=float(ns['alpha_lo']);ahi=float(ns['alpha_hi']);u=(np.exp(-.002/tau)-alo)/(ahi-alo);h=[[F.from_float(p.evaluate(t,u)) for p in row] for row in ns['vertical']];av=np.array(ns['vertical_decorate'](h,(F.from_float(1/m),d)),float)
 u=(30*.003/J[2]-300/11)/(100/3-300/11);h=[[F.from_float(p.evaluate(0,u)) for p in row] for row in ns['yawcore']];ay=np.array(ns['controlled'](h,[F(-12),F(0)],d),float)
 full=np.zeros((21,21));ix=[0,3,7,10,15,19];iy=[1,4,6,9,14,18];iv=[2,5,12,13,17];iz=[8,11,16,20];sign=np.diag([1,1,-1,-1,-1,-1]);full[np.ix_(ix,ix)]=ax;full[np.ix_(iy,iy)]=sign@ax@sign;full[np.ix_(iv,iv)]=av;full[np.ix_(iz,iz)]=ay;return full
params=list(itertools.product([.45,.55],[9.78,9.83],[.05,.2],[.015,.04],[[.0018,.0018,.0027],[.0022,.0022,.0033]],[.005,.01,.02]));rng=np.random.default_rng(4)
for i in range(8):params.append((rng.uniform(.45,.55),rng.uniform(9.78,9.83),rng.uniform(.05,.2),rng.uniform(.015,.04),[float(x) for x in rng.uniform([.0018,.0018,.0027],[.0022,.0022,.0033])],float(rng.choice([.005,.01,.02]))))
errors=[]
for p in params:
 # expression() uses the planar y-inertia for both symmetric planar blocks;
 # mixed-inertia crosscheck each block against its own coefficient below.
 actual=torch.autograd.functional.jacobian(lambda x:model_step(x,*p),z,vectorize=True).numpy();pred=expression(*p)
 if p[4][0]!=p[4][1]:
  mod=list(p);mod[4]=[p[4][0],p[4][0],p[4][2]];other=expression(*mod);iy=[1,4,6,9,14,18];pred[np.ix_(iy,iy)]=other[np.ix_(iy,iy)]
 errors.append(float(np.max(np.abs(actual-pred))))
assert max(errors)<2e-12,max(errors);print('ACTUALJACOBIANcrosscheck',len(errors),max(errors),flush=True)

def ldl_positive(a):
 n=len(a);l=[[F(int(i==j)) for j in range(n)] for i in range(n)];d=[]
 for j in range(n):
  pivot=a[j][j]-sum(l[j][k]*l[j][k]*d[k] for k in range(j))
  if pivot<=0:return False,j,str(pivot)
  d.append(pivot)
  for i in range(j+1,n):l[i][j]=(a[i][j]-sum(l[i][k]*l[j][k]*d[k] for k in range(j)))/pivot
 return True,None,None
results={}
for name,bb,variants,decorate in [('planar_x',ns['bp'],ns['variants'],ns['planar_decorate']),('vertical',ns['bv'],[(r,d) for r in [F(20,11),F(20,9)] for d in ns['delayvertices']()],ns['vertical_decorate']),('yaw',ns['by'],ns['delayvertices'](),lambda h,d:ns['controlled'](h,[F(-12),F(0)],d))]:
 info=meta['groups'][name];p=[[F.from_float(float(v)) for v in row] for row in info['scaled_P']];s=list(map(lambda x:F.from_float(float(x)),info['state_scaling']));n=len(p);eps=F({'planar_x':'.003','vertical':'.001','yaw':'.008'}[name]);assert ldl_positive([[p[i][j]-(F('.0001') if i==j else 0) for j in range(n)] for i in range(n)])[0];count=0
 for i,row in enumerate(bb):
  for j,core in enumerate(row):
   for variant in variants:
    raw=decorate(core,variant);a=[[raw[ii][jj]*s[jj]/s[ii] for jj in range(n)] for ii in range(n)];pa=[[sum(p[ii][k]*a[k][jj] for k in range(n)) for jj in range(n)] for ii in range(n)];q=[[p[ii][jj]-sum(a[k][ii]*pa[k][jj] for k in range(n))-(eps if ii==jj else 0) for jj in range(n)] for ii in range(n)];okay,index,pivot=ldl_positive(q)
    if not okay:raise AssertionError((name,i,j,variant,index,pivot))
    count+=1
 m=max(sum(abs(x) for x in row) for row in p);results[name]={'exact_Bernstein_vertices_audited':count,'P_eigenvalue_lower_bound':'.0001','decrease_matrix_eigenvalue_lower_bound':str(eps),'P_norm_contraction_squared_upper_bound_rational':str(1-eps/m),'P_norm_contraction_upper_bound_display_only':float(1-eps/m)**.5};print('EXACTBERNSTEIN',name,count,'PASS',flush=True)
artifact={'status':'EXACT_RATIONAL_LOCAL_LINEAR_PARAMETER_INTERIOR_CERTIFICATE','precise_claim':'For the derived local-linear backup map over independently lifted coefficient intervals, exact Bernstein convex hull inclusion and exact rational common-P matrix inequalities prove strict contraction for the ENTIRE stated lifted box, including physical parameter interiors and all delay interpolation values. This does not establish nonlinear plant contraction or a robust invariant terminal region.','assumptions':['Real-valued local linearization of the specified 100Hz observer-free PD backup, five RK4 substeps, unit-equilibrium quaternion chart, no motor/command saturation at equilibrium.','Full stated mass/gravity/drag/inertia/lag ranges; isotropic drag; constant parameters; transport delays .005-.02s with linear fractional queue interpolation.','Battery multiplier exactly1 in this local model; bounded battery authority needs an additional explicit embedding before operational use.'],'not_proven':['Nonlinear state remainder','Bounded-measurement disturbance gain','Float32/GPU operation semantics or rounding','A nonzero robust invariant region','Infinite terminal hold','Mission safety or Phase4 gate'],'actual_Jacobian_numerical_crosscheck_cases':len(errors),'maximum_actual_Jacobian_entry_discrepancy':max(errors),'blocks':results,'planar_y':'Same exact certificate by sign similarity D=diag(1,1,-1,-1,-1,-1); each actual inertia axis uses its own ratio from the same full interval.','elapsed_s':time.monotonic()-start};(OUT/'backup_Bernstein_exact_audit.json').write_text(json.dumps(artifact,indent=2)+'\n');print('DONEexact',artifact['elapsed_s'],flush=True)
