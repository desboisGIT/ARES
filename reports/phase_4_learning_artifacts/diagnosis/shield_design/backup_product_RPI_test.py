"""Evaluate sufficient product-ellipsoid ISS inequalities; failure is informative."""
from pathlib import Path
from fractions import Fraction as F
import json,numpy as np,hashlib
root=Path.cwd();out=root/'reports/phase_4_learning_artifacts/diagnosis/shield_design';hpath=root/'reports/phase_4_gate_artifacts/shield_research/taylor_remainder_v1/taylor_bounds.json';h=json.loads(hpath.read_text());a=json.loads((out/'backup_direct_contraction_audit.json').read_text());meta=json.loads((out/'backup_common_P_feasibility.json').read_text());r={'planar_x':.1,'planar_y':.1,'vertical':.1,'yaw':.05};dB=float(F(1,5)*F(4851,5)*F(1,100)/F(43200));mgmax=float(np.nextafter(np.float32(.55),np.float32(np.inf)))*float(np.nextafter(np.float32(9.83),np.float32(np.inf)));drift=np.zeros(21);drift[2]=mgmax/(.5*1.5*.95**2)*dB;drift[13]=drift[17]=mgmax/.95**2*dB
records=[]
for run in h['runs']:
 if 'epsilon_state_nonlinear' not in run:continue
 blocks={}
 for k in r:
  info=meta['groups'][k];idx=info['indices'];P=np.abs(np.array(info['scaled_P']));sc=np.array(info['state_scaling']);cert=a['blocks']['planar_x' if k=='planar_y' else k];rho=float(F(cert['rho_upper_rational']));budget=(1-rho)*r[k]
  def bound(e):
   v=np.array(e)[idx]/sc;return float(np.sqrt(v@P@v))
  nl=bound(run['epsilon_state_nonlinear']);noise=bound(run['epsilon_measurement_noise']);dd=bound(drift);blocks[k]={'rho_upper':rho,'P_radius':r[k],'available_contraction_budget':budget,'nonlinear_P_norm_bound':nl,'measurement_P_norm_bound':noise,'authority_equilibrium_drift_P_norm_bound':dd,'rounding_P_norm_bound':None,'real_constant_authority_sufficient_inequality_passed':nl+noise<=budget,'real_with_authority_drift_sufficient_inequality_passed':nl+noise+dd<=budget,'constant_authority_noise_free_radius_scale_upper':budget/nl if nl else None,'maximum_constant_noise_P_budget_at_uniform_optimal_state_scale':budget**2/(4*nl) if nl else None}
  if noise:blocks[k]['maximum_measurement_bound_fraction_for_any_uniform_radius_scale']=budget**2/(4*nl)/noise if nl else None
 records.append({'label':run['label'],'branch':run['branch'],'blocks':blocks})
artifact={'status':'SUFFICIENT_ISS_INEQUALITIES_FAILED_NOT_A_COUNTEREXAMPLE_TO_PLANT_SAFETY','source_taylor_bounds':{'path':str(hpath.relative_to(root)),'sha256':hashlib.sha256(hpath.read_bytes()).hexdigest()},'method':'Each component epsilon vector is converted to a block P-norm upper bound via sqrt(e_scaled^T abs(P) e_scaled), and compared with (1-rho)*r. Display values are float64 diagnostics, not an enabled certificate.','battery_equilibrium_drift':{'authority_step_upper_bound':dB,'physical_component_bounds':drift.tolist()},'records':records,'conclusion':'The current independent Taylor-remainder + triangle ISS method does not establish the requested nonzero product invariant set, including the noise-free diagnostic. This does not prove the actual closed loop unstable or unsafe. Do not enable certification or fit smaller noise bounds to these results. Preserve all failed domains; next investigate correlated/structured remainder or S-procedure/finite-return-map disturbance geometry.','machine_rounding_budget_missing':True}
(out/'backup_product_RPI_feasibility.json').write_text(json.dumps(artifact,indent=2)+'\n')
for rec in records:
 print(rec['label'],rec['branch'],{k:{x:round(v[x],8) if isinstance(v[x],float) else v[x] for x in ['available_contraction_budget','nonlinear_P_norm_bound','measurement_P_norm_bound','authority_equilibrium_drift_P_norm_bound','real_constant_authority_sufficient_inequality_passed']} for k,v in rec['blocks'].items()},flush=True)
