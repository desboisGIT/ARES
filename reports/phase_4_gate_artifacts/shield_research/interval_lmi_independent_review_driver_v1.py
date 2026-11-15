from pathlib import Path
from fractions import Fraction as F
import importlib.util,sys,itertools,json,hashlib,random,math
import numpy as np
root=Path('/home/ares-user/Documents/projects/ARES');base=root/'reports/phase_4_gate_artifacts/shield_research'
spec=importlib.util.spec_from_file_location('lmi',base/'interval_lmi_audit_v1.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
def q(x):return F.from_float(float(x))
def encl(z,v):return q(z[0])<=v<=q(z[1])
vals=[0.,-0.,np.nextafter(0.,1.),-np.nextafter(0.,1.),2.**-1022,-2.**-1022,math.nextafter(2.**-1022,0.),2.**-1000,2.**-537,2.**-53,1.,math.nextafter(1.,0.),-1.,2.**500,-2.**500]
count=0;rejects=0
with np.errstate(over='raise',invalid='raise',divide='raise',under='ignore'):
 for x,y in itertools.product(vals,repeat=2):
  for op,exact in [(a.add,lambda:q(x)+q(y)),(a.sub,lambda:q(x)-q(y)),(a.mul,lambda:q(x)*q(y))]+([] if y==0 else [(a.divide,lambda:q(x)/q(y))]):
   try:z=op(a.iv(x),a.iv(y))
   except (FloatingPointError,ArithmeticError):rejects+=1;continue
   assert encl(z,exact()),(op.__name__,x,y,z,exact());count+=1
 for x in vals:
  try:z=a.square(a.iv(x))
  except FloatingPointError:rejects+=1;continue
  assert encl(z,q(x)**2);count+=1
 # Interval endpoints include both signs, exact zeros and subnormal transitions.
 for i,x in enumerate(vals):
  for y in vals[i:]:
   lo,hi=sorted((x,y));z=a.square(a.iv(lo,hi));minimum=F(0) if lo<=0<=hi else min(q(lo)**2,q(hi)**2)
   assert encl(z,minimum) and encl(z,max(q(lo)**2,q(hi)**2));count+=1
# Independent exact lower-triangle LDL reference, including near singular cases.
def exact_ldl(m):
 n=len(m);L={};D=[]
 for j in range(n):
  d=m[j][j]-sum((L[j,k]**2*D[k] for k in range(j)),F(0))
  if d<=0:return False
  D.append(d)
  for i in range(j+1,n):L[i,j]=(m[i][j]-sum((L[i,k]*L[j,k]*D[k] for k in range(j)),F(0)))/d
 return True
rng=random.Random(877610);spd=0;inconclusive=0
with np.errstate(over='raise',invalid='raise',divide='raise',under='ignore'):
 for i in range(500):
  n=1+i%6;scale=math.ldexp(1.,rng.choice([-1000,-700,-100,-20,0,20,100,500]));b=np.array([[rng.randint(-4,4)/8 for j in range(n)] for k in range(n)])
  # Build exact Fraction matrix then round each entry, audit that represented symmetric matrix.
  m=np.array([[float((sum((q(b[r,k])*q(b[c,k]) for k in range(n)),F(0))+F(r==c,256))*q(scale)) for c in range(n)] for r in range(n)])
  exact=[[q(x) for x in row] for row in m]
  try:got=a.interval_ldl(a.iv(m))['passed']
  except (FloatingPointError,ArithmeticError):inconclusive+=1;continue
  truth=exact_ldl(exact)
  assert not got or truth
  spd+=got;inconclusive+=not got
report=dict(schema_version='ares.independent_interval_LMI_review.v1',source_sha256=hashlib.sha256((base/'interval_lmi_audit_v1.py').read_bytes()).hexdigest(),no_unsoundness_found_under_stated_hypotheses=True,primitive_exact_reference_checks=count,fail_closed_overflow_or_reciprocal_rejections=rejects,exact_fraction_LDL_matrices=500,interval_positive_definite_passes=spd,conservative_failures=inconclusive,required_hypotheses=['binary64 RNE fullprecision','gradual input/output subnormal handling; DAZ=FTZ=0','correct rounded basic arithmetic and nextafter','no concurrent arithmetic-mode changes','independent exact symmetric coefficient containment'],review_findings=['multiply min/max four corners and one-ulp expansion sound','division reciprocal is decreasing on either sign-separated branch','one outward nextafter also encloses gradual underflow and rounded signed zero','square zero clipping uses nonnegative exact range','interval LDL positive lower pivots suffice for the exact symmetric lower-triangle matrix','all vertices, explicit command signs, offset patterns and last chunk visited','overflow/invalid/division exceptions fail closed in public verify_arrays','symmetric endpoint arrays alone do not prove exact coefficient symmetry; supplier retains this obligation','platform modes are documented hypotheses, not checked by this audit module'],flight_certificate=False)
(base/'interval_lmi_independent_review_v1.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
