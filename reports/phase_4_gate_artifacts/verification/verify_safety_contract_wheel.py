import hashlib,json,os,subprocess,sys,tempfile,zipfile
from pathlib import Path
repo=Path('/home/ares-user/Documents/projects/ARES')
base=repo/'reports/phase_4_gate_artifacts/verification'
wheel=repo/'dist/ares-0.6.0-py3-none-any.whl'
manifest=json.loads((base/'safety_contract_source_manifest.json').read_text())
with zipfile.ZipFile(wheel) as archive:
 for name,digest in manifest.items():assert hashlib.sha256(archive.read('ares/'+name)).hexdigest()==digest,name
work=Path(tempfile.mkdtemp(prefix='ares-safety-contract-wheel-'));target=work/'installed'
subprocess.run([sys.executable,'-m','pip','install','--no-deps','--target',str(target),str(wheel)],check=True,cwd=work,stdout=subprocess.DEVNULL)
guard=work/'guard';guard.mkdir()
(guard/'sitecustomize.py').write_text("import sys\nclass NoCuPy:\n def find_spec(self,fullname,path=None,target=None):\n  if fullname=='cupy' or fullname.startswith('cupy.'):\n   raise ImportError('CuPy prohibited in CPU wheel verification')\nsys.meta_path.insert(0,NoCuPy())\n")
env=dict(os.environ,PYTHONPATH=os.pathsep.join((str(guard),str(target))))
checkpoint=json.loads((repo/'reports/phase_4_gate.json').read_text())['checkpoints']['bptt']
assert checkpoint['sha256']=='e535060b0b1f6586cb6deeaafd59b87016028c3d412257934cec2ce9ef41063f'
assert hashlib.sha256(Path(checkpoint['path']).read_bytes()).hexdigest()==checkpoint['sha256']
code='''import hashlib,json,sys
from pathlib import Path
from dataclasses import replace
import ares,ares.safety_contract
from ares.safety_contract import MeasurementContract,robust_free_witness
from ares.flight_training import load_checkpoint
assert Path(ares.__file__).is_relative_to(Path(sys.argv[1]))
assert Path(ares.safety_contract.__file__).is_relative_to(Path(sys.argv[1]))
c=MeasurementContract(position_error_m=.005,velocity_error_mps=.01,world_ray_error=.003,depth_error_m=.02,capture_time_s=1.,expires_s=1.1,supplier='external bounded capture rig',provenance='fixture deterministic set guarantee; not measured RMS',source_kind='hard_external',bound_basis='deterministic_guarantee',voxel_model='homogeneous_static_until_expiry',depth_model='first_hit_range')
args=dict(position=[.5,.5,.5],world_direction=[1.,.2,-.1],measured_depth=2.,voxel_lower=[1.4,.6,.3],voxel_upper=[1.6,.8,.5],optical_range=1.,time_s=1.05)
positive=robust_free_witness(contract=c,**args)
rejected=robust_free_witness(contract=replace(c,source_kind='Gaussian'),**args)
assert positive.conditional_input_valid and positive.free_witness and not positive.flight_certificate
assert not rejected.conditional_input_valid and not rejected.free_witness and not rejected.flight_certificate
model,payload=load_checkpoint(sys.argv[2],device='cpu')
root=Path(ares.__file__).parent
sources=payload['metadata']['runtime_source_sha256'];assert len(sources)==51
for name,digest in sources.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
assert (root/'safety_contract.py').is_file() and 'safety_contract.py' not in sources
assert not any(n=='cupy' or n.startswith('cupy.') for n in sys.modules)
print(json.dumps({'outside_checkout':True,'cupy_imported':False,'conditional_voxel_witness':True,'gaussian_rejected':True,'flight_certificate':False,'frozen_checkpoint_source_count':len(sources),'additional_unused_module':'safety_contract.py','no_retraining':True}))
'''
subprocess.run([sys.executable,'-c',code,str(target),checkpoint['path']],check=True,cwd=work,env=env)
subprocess.run([sys.executable,'-m','ares','evaluate-flight',checkpoint['path'],'--device','cpu','--episodes','1','--batch-size','1','--max-steps','1','--output',str(base/'safety_contract_frozen_bptt_evaluation.json')],check=True,cwd=work,env=env)
assert hashlib.sha256(Path(checkpoint['path']).read_bytes()).hexdigest()==checkpoint['sha256']
print(json.dumps({'wheel_sha256':hashlib.sha256(wheel.read_bytes()).hexdigest(),'wheel_bytes':wheel.stat().st_size,'installed_python_sources':len(manifest),'checkpoint_sha256':checkpoint['sha256'],'evaluation_sourceguard_passed':True,'verification_directory':str(work)}))
