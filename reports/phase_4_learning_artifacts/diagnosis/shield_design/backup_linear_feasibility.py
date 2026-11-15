"""Unverified local-linear feasibility probe, never a certificate."""
import torch,numpy as np,json,itertools
from scipy.linalg import solve_discrete_lyapunov,eigvalsh
from pathlib import Path
from ares.flight_control import acceleration_to_action,ControllerConfig
from ares.vendor.diffaero.quadrotor import QuadrotorModel
torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
DT=.01;KP=1.5;KV=2.5;MG0=.5*9.80665

def model_step(z,m,g,drag,tau,inertias,delay):
 mass=torch.tensor([m]);gravity=torch.tensor([g]);mg=m*g;eq=(9.80665-2*mg)/KP
 p=z[None,:3]+torch.tensor([[0.,0.,eq]]);v=z[None,3:6];xyz=z[None,6:9];q=torch.cat((xyz,(1-xyz.square().sum(-1,keepdim=True)).sqrt()),-1);w=z[None,9:12];motor=z[12]+mg
 obs={'p':p,'v':v,'q':q,'gaze':p+torch.tensor([[1.,0.,0.]])}
 command=acceleration_to_action(obs,-KP*p-KV*v,ControllerConfig(max_yaw_rate=1.))[0]
 old1=z[13:17]+torch.tensor([mg,0.,0.,0.]);old2=z[17:21]+torch.tensor([mg,0.,0.,0.]);history=[command,old1,old2];d=delay/DT;lo=int(np.floor(d+1e-10));hi=int(np.ceil(d-1e-10));f=d-lo;applied=history[lo]*(1-f)+history[hi]*f
 model=QuadrotorModel(mass,torch.tensor([inertias]),torch.full((1,3),drag),gravity,torch.full((1,3),30.),torch.tensor([[.002,.002,.003]]));state=torch.cat((p,q,v,w),-1)
 for i in range(5):
  motor=applied[0]+(motor-applied[0])*np.exp(-DT/5/tau);u=torch.cat(((motor/mg).reshape(1,1),applied[None,1:]),-1);state=model.step(state,u,DT/5)
 return torch.cat((state[0,:3]-torch.tensor([0.,0.,eq]),state[0,7:10],state[0,3:6],state[0,10:13],(motor-mg).reshape(1),command-torch.tensor([mg,0.,0.,0.]),z[13:17]))

nom=(.5,9.80665,.1,.025,[.002,.002,.003],.0125)
z=torch.zeros(21,requires_grad=True);A=torch.autograd.functional.jacobian(lambda x:model_step(x,*nom),z,vectorize=True).numpy();assert np.isfinite(A).all();P=solve_discrete_lyapunov(A.T,np.eye(21));L=np.linalg.cholesky(P);M=L.T;Mi=np.linalg.inv(M);rows=[]
for mass,gravity,drag,tau,inertia,delay in itertools.product([.45,.55],[9.78,9.83],[.05,.2],[.015,.04],[[.0018,.0018,.0027],[.0022,.0022,.0033]],[.005,.02]):
 params=(mass,gravity,drag,tau,inertia,delay);Ac=torch.autograd.functional.jacobian(lambda x:model_step(x,*params),z,vectorize=True).numpy();rho=float(max(abs(np.linalg.eigvals(Ac))));norms={str(n):float(np.linalg.norm(M@np.linalg.matrix_power(Ac,n)@Mi,2)) for n in [1,50,100,200]};rows.append({'parameters':params,'spectral_radius':rho,'return_map_P_norms':norms})
summary={'status':'Unverified floating-point local-linear feasibility only. No nonlinear, measurement, parameter-interior or rounding proof.','state':'21D: p-eq, v, q_xyz, body_w, motor_thrust-mg, two previous four-component commands relative to equilibrium.','backup':'Observed full-state PD netaccel=-1.5*p-2.5*v; nominalmass.5/gravity9.80665; existing geometric attitude converter maxyaw1, headingfixedworldx; allqueues andfull rigidbody retained.','equilibrium_z_formula':'(9.80665 - m*g/.5)/1.5','equilibrium_z_range_m':[(9.80665-.55*9.83/.5)/1.5,(9.80665-.45*9.78/.5)/1.5],'nominal_spectral_radius':float(max(abs(np.linalg.eigvals(A)))),'nominal_P_eigenvalues':[float(eigvalsh(P).min()),float(eigvalsh(P).max())],'finite_corner_cases':len(rows),'worst_spectral_radius':max(r['spectral_radius'] for r in rows),'worst_return_map_P_norms':{str(n):max(r['return_map_P_norms'][str(n)] for r in rows) for n in [1,50,100,200]},'rows':rows}
out=Path('reports/phase_4_learning_artifacts/diagnosis/shield_design');out.mkdir(parents=True,exist_ok=True);(out/'backup_linear_feasibility.json').write_text(json.dumps(summary,indent=2)+'\n');np.savez(out/'backup_local_linear_candidate.npz',A=A,P=P);print({k:v for k,v in summary.items() if k!='rows'},flush=True)
