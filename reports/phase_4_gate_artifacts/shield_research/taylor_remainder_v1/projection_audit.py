"""Exact rational audits of ellipsoid-to-component outer bounds."""
from fractions import Fraction as F
from pathlib import Path
import json,hashlib
OUT=Path(__file__).resolve().parent
BASE=OUT.parents[2]/'phase_4_learning_artifacts/diagnosis/shield_design'
def exact(a):return [[F.from_float(float(x)) for x in row] for row in a]
def inverse(a):
 n=len(a);m=[row.copy()+[F(int(i==j)) for j in range(n)] for i,row in enumerate(a)]
 for i in range(n):
  k=next(k for k in range(i,n) if m[k][i]);m[i],m[k]=m[k],m[i]
  d=m[i][i];m[i]=[x/d for x in m[i]]
  for j in range(n):
   if i!=j:
    d=m[j][i];m[j]=[x-d*y for x,y in zip(m[j],m[i])]
 return [row[n:] for row in m]
def positive_definite(a):
 n=len(a);L=[[F(0)]*n for _ in range(n)];d=[]
 assert all(a[i][j]==a[j][i] for i in range(n) for j in range(n))
 for i in range(n):
  di=a[i][i]-sum((L[i][k]**2*d[k] for k in range(i)),F(0))
  if di<=0:return False
  d.append(di);L[i][i]=F(1)
  for j in range(i+1,n):L[j][i]=(a[j][i]-sum((L[j][k]*L[i][k]*d[k] for k in range(i)),F(0)))/di
 return True

if __name__=='__main__':
 sources={}
 for name,target in [('backup_direct_contraction_audit.json','source_common_P_direct_audit.json'),('backup_linear_noise_RPI_feasibility.json','source_linear_noise_Q_candidate.json')]:
  destination=OUT/target
  if not destination.exists():destination.write_bytes((BASE/name).read_bytes())
  sources[name]=destination
 direct=json.loads(sources['backup_direct_contraction_audit.json'].read_text())
 chosen={'planar_x':['.111','.153','.0223','.58','.427','.595'],
         'vertical':['.339079','.396802','1.782115','1.438910','1.933647'],
         'yaw':['.011805','.414826','.363444','.510264']}
 records=[]
 for name,c in chosen.items():
  block=direct['blocks'][name];P=exact(block['scaled_P_exact_binary64']);Pinv=inverse(P)
  scales=list(map(lambda x:F.from_float(float(x)),block['state_scales_exact_binary64']))
  good=[F(cc)**2>=scale**2*Pinv[i][i] for i,(cc,scale) in enumerate(zip(c,scales))]
  assert positive_definite(P) and all(good)
  records.append({'group':name,'P_positive_definite_exact':True,'all_component_coefficients_proved':all(good),
                  'coefficients_rational':c,'physical_indices':block['physical_indices']})
 noise=json.loads(sources['backup_linear_noise_RPI_feasibility.json'].read_text())
 qbounds={'planar_x':['.077783','.070442','.010529','.197912','.288661','.326455'],
          'vertical':['.067302','.055462','.173839','.202997','.254497']}
 qrecords=[]
 for name,bs in qbounds.items():
  row=next(x for x in noise['records'] if x['group']==name and x['tau']==.99)
  Q=exact(row['Q_scaled']);scales=[F.from_float(float(x)) for x in direct['blocks'][name]['state_scales_exact_binary64']]
  good=[F(b)**2>=s*s*Q[i][i] for i,(b,s) in enumerate(zip(bs,scales))]
  assert positive_definite(Q) and all(good)
  qrecords.append({'group':name,'Q_positive_definite_exact':True,'component_bounds_rational':bs,'all_projection_bounds_proved':True})
 report={'scope':'Only exact ellipsoid-to-box projection and P/Q positive definiteness. Not local dynamics/noise RPI certification.',
         'P_component_coefficients':records,'noise_Q_box':qrecords,
         'source_sha256':{name:hashlib.sha256(sources[name].read_bytes()).hexdigest() for name in ['backup_direct_contraction_audit.json','backup_linear_noise_RPI_feasibility.json']}}
 (OUT/'projection_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
