"""Exact affine stage-gain generator construction for the planar sector family."""
from pathlib import Path
from fractions import Fraction as F
import json,numpy as np,time,itertools
start=time.monotonic();HERE=Path(__file__).resolve().parent;source=(HERE/'inputs/backup_bernstein_embedding.py').read_text().split('# Exact rational physical coefficient intervals')[0];source=source.replace("OUT=Path('reports/phase_4_learning_artifacts/diagnosis/shield_design')","OUT=RESEARCH_DIR");ns={'RESEARCH_DIR':HERE/'inputs'};exec(source,ns);Poly,zeros,eye,mm,plus,times,rk4,power,mb=[ns[k] for k in ['Poly','zeros','eye','mm','plus','times','rk4','power','matrix_bern']];out=ns['OUT']/'sector_flow_v1';bounds=json.loads((ns['OUT']/'backup_Bernstein_embedding_feasibility.json').read_text())['outward_float32_physical_bounds'];ML,MH=map(F,bounds['mass']);GL,GH=map(F,bounds['gravity']);DL,DH=map(F,bounds['drag_over_mass']);KL,KH=map(F,bounds['angular_rate_gain']);GC=(GL+GH)/2;DG=F(7,8);delta=Poly(DL)+Poly({(1,0):DH-DL});k=Poly(KL)+Poly({(0,1):KH-KL});a=zeros(5,5);a[0][1]=Poly(1);a[1][1]=-delta;a[2][3]=Poly(F(1,2));a[3][3]=-k;a[3][4]=k;a1=[[p for p in row] for row in a];a1[1][2]=Poly(2);f0=rk4(a);f1=plus(rk4(a1),times(f0,-1));fgc=plus(f0,times(f1,GC));h0=power(fgc,5);hj=[mm(mm(power(fgc,4-j),f1),power(fgc,j)) for j in range(5)]
# Exact algebra verifies that no products of distinct stage gain perturbations
# occur: F1 Fgc^k F1 =0 for all possible placements in a 5-step product.
for t in range(5):
 z=mm(mm(f1,power(fgc,t)),f1);assert all(not p.d for row in z for p in row)
# Constant gain specialization matches direct full held-input RK4 exactly.
for g in [GL,GH]:
 rebuilt=h0
 for h in hj:rebuilt=plus(rebuilt,times(h,g-GC))
 direct=power(plus(f0,times(f1,g)),5);assert all(x.d==y.d for r,s in zip(rebuilt,direct) for x,y in zip(r,s))
B0=mb(h0,20,20);BJ=[mb(h,20,20) for h in hj];As=[];Gs=[];Ajs=[];Gjs=[];cs=[F(19,20)*F(1,2)/(MH*GH),F(1,2)/(ML*GL)]
def ctl(h,C,d,queue=True):
 a=ns['controlled'](h,C,d)
 if not queue:
  a[-2]=[F(0)]*6;a[-1]=[F(0)]*6
 return a
def vec(h,d,queue=True):return [h[i][4]*d[0] for i in range(4)]+[F(1 if queue else 0),F(0)]
for i in range(21):
 for j in range(21):
  for c in cs:
   C=[-9*c,-15*c,F(-12),F(0)]
   for d in ns['delayvertices']():
    As.append(np.array(ctl(B0[i][j],C,d),float));Gs.append(np.array(vec(B0[i][j],d),float));Ajs.append([np.array(ctl(B[i][j],C,d,False),float) for B in BJ]);Gjs.append([np.array(vec(B[i][j],d,False),float) for B in BJ])
As,Gs,Ajs,Gjs=map(np.array,(As,Gs,Ajs,Gjs));signs=np.array(list(itertools.product([-1.,1.],repeat=5)));allA=As[:,None]+float(DG)*np.einsum('sj,ijab->isab',signs,Ajs);allG=Gs[:,None]+float(DG)*np.einsum('sj,ija->isa',signs,Gjs);meta=json.loads((ns['OUT']/'backup_common_P_feasibility.json').read_text());p=np.array(meta['groups']['planar_x']['scaled_P']);sc=np.array(meta['groups']['planar_x']['state_scaling']);L=np.linalg.cholesky(p).T;Li=np.linalg.inv(L);flat=allA.reshape(-1,6,6);trans=(L[None]@(flat*sc[None,None,:]/sc[None,:,None]))@Li;worst=float(np.linalg.svd(trans,compute_uv=False)[:,0].max())
# Stage force containment from physical terminal outer-domain bounds.
deltaF=max(F('1.5')*F('.173839'),F('1.5')*F('.202997'),F('1.5')*F('.254497'),F('.115898'));required=(GH-GL)/2+deltaF/ML;assert required<DG
np.savez_compressed(out/'sector_planar_generator_candidates.npz',A0=As,G0=Gs,A_generators=Ajs,G_generators=Gjs,gain_corner_signs=signs,all_gain_vertex_A=allA,all_gain_vertex_G=allG)
polys={'H0':h0,**{f'H_stage_{j}':h for j,h in enumerate(hj)}};(out/'sector_exact_generator_polynomials.json').write_text(json.dumps({name:[[[[i,j,str(v)] for (i,j),v in sorted(p.d.items())] for p in row] for row in core] for name,core in polys.items()})+'\n');r={'status':'EXACT_STAGE_AFFINITY_AND_RANGE_CONTAINMENT_PLUS_FLOAT_MATRIX_CANDIDATES','g_center_rational':str(GC),'g_radius_rational':str(DG),'required_gain_radius_bound_rational':str(required),'required_gain_radius_display':float(required),'exact_cross_stage_products_zero':True,'constant_gravity_specialization_exact':True,'coefficient_base_vertices':len(As),'independent_stage_gain_vertices':len(signs),'total_local_planar_vertices':len(flat),'old_common_P_worst_norm_float':worst,'nonlinear_RPI_proven':False,'arithmetic_execution_proven':False,'elapsed_s':time.monotonic()-start};(out/'sector_generator_construction.json').write_text(json.dumps(r,indent=2)+'\n');print(r,flush=True)
