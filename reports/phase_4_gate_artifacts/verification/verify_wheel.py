import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

repo=Path('/home/ares-user/Documents/projects/ARES')
work=Path(tempfile.mkdtemp(prefix='ares-phase4-gate-wheel-'))
target=work/'installed'
wheel=repo/'dist/ares-0.6.0-py3-none-any.whl'
subprocess.run([sys.executable,'-m','pip','install','--no-deps','--target',str(target),str(wheel)],
               check=True,cwd=work,stdout=subprocess.DEVNULL)
guard=work/'guard';guard.mkdir()
(guard/'sitecustomize.py').write_text('''import sys
class NoCuPy:
 def find_spec(self,fullname,path=None,target=None):
  if fullname=='cupy' or fullname.startswith('cupy.'):
   raise ImportError('CuPy prohibited in CPU wheel verification')
sys.meta_path.insert(0,NoCuPy())
''')
env=dict(os.environ,PYTHONPATH=os.pathsep.join((str(guard),str(target))))
report=work/'phase4.json'
subprocess.run([sys.executable,'-m','ares','phase4','--device','cpu','--train-updates','2',
    '--batch-size','2','--rollout-steps','4','--episodes','2','--max-steps','10',
    '--shield-episodes','0','--residual-acceleration','.5','--residual-penalty','1','--guidance-speed-fraction','.95','--smoke','--output',str(report)],check=True,cwd=work,env=env,
    stdout=subprocess.DEVNULL)
data=json.loads(report.read_text())
assert data['software_preparation_passed'] and not data['phase_passed']
assert data['adapter_version']=='ares.diffaero.v2'
assert set(data['checkpoints']) == {'bptt','ppo'}
assert all(c['residual_acceleration']==.5 and c['guidance_speed_fraction']==.95 for c in data['policy_configuration'].values())
import math,hashlib
for algorithm, training in data['training'].items():
 assert training['config']['residual_penalty']==1
 assert any(row['residual_l2_weighted']>0 for row in training['training_curve'])
 assert all(math.isfinite(row['residual_l2_weighted']) for row in training['training_curve'])
 assert hashlib.sha256(Path(data['checkpoints'][algorithm]['path']).read_bytes()).hexdigest()==data['checkpoints'][algorithm]['sha256']
checkpoint=data['checkpoints']['bptt']['path']
for algorithm, item in data['checkpoints'].items():
 subprocess.run([sys.executable,'-m','ares','evaluate-flight',item['path'],'--device','cpu','--episodes','1','--batch-size','1','--max-steps','1','--output',str(work/f'evaluated_{algorithm}.json')],check=True,cwd=work,env=env,stdout=subprocess.DEVNULL)
subprocess.run([sys.executable,'-m','ares','control-check','--device','cpu','--count','8','--duration','4','--settle','2','--output',str(work/'control.json')],check=True,cwd=work,env=env,stdout=subprocess.DEVNULL)
assert json.loads((work/'control.json').read_text())['development_regression_passed']
subprocess.run([sys.executable,'-m','ares','evaluate-flight',checkpoint,'--episodes','1',
    '--batch-size','1','--max-steps','1','--output',str(work/'evaluated.json')],
    check=True,cwd=work,env=env,stdout=subprocess.DEVNULL)
code='''from pathlib import Path
from importlib.resources import files
import sys,json,ares,ares.phase4,ares.flight_policy,ares.flight_control,ares.flight_guidance,ares.control_validation
from ares.flight_training import load_checkpoint
for module in (ares,ares.phase4,ares.flight_policy,ares.flight_control,ares.flight_guidance,ares.control_validation):
 assert Path(module.__file__).resolve().is_relative_to(Path(sys.argv[1]).resolve())
assert ares.__version__=='0.6.0'
license=files('ares').joinpath('vendor/diffaero/LICENSE').read_text()
assert 'Redistribution' in license
assert files('ares').joinpath('vendor/diffaero/UPSTREAM.md').is_file()
model,payload=load_checkpoint(sys.argv[2])
assert payload['metadata']['adapter_version']=='ares.diffaero.v2'
assert model.config.residual_acceleration==.5
assert payload['policy_config']['residual_acceleration']==.5
assert model.config.guidance_speed_fraction==.95
assert payload['metadata']['training_config']['residual_penalty']==1
from ares.flight_env import FlightEnv
from ares.worldgen import generate_scene
env_check=FlightEnv(generate_scene(100,'apartment','train'),payload['metadata']['profile'],batch_size=1,device='cpu',horizon=1,guidance_speed_fraction=model.config.guidance_speed_fraction)
assert env_check.guidance_config.speed_fraction==.95
assert env_check.speed_limit==1.5
from ares.phase4 import require_runtime_unchanged
require_runtime_unchanged(payload['metadata']['runtime_source_sha256'])
assert not any(k=='cupy' or k.startswith('cupy.') for k in sys.modules)
print(json.dumps({'version':ares.__version__,'outside_checkout':True,'vendor_license':True,'cpu_train_both_learners_and_frozen_evaluation':True,'control_check_passed':True,'cupy_imported':False}))
'''
for item in data['checkpoints'].values():
 subprocess.run([sys.executable,'-c',code,str(target),item['path']],check=True,cwd=work,env=env)
from shutil import copy2
verification=repo/'reports/phase_4_gate_artifacts/verification'
copy2(report,verification/'cpu_phase4.json')
copy2(work/'control.json',verification/'cpu_control.json')
for algorithm in data['checkpoints']:
 copy2(work/f'evaluated_{algorithm}.json',verification/f'evaluated_{algorithm}.json')
print(json.dumps({'wheel':str(wheel),'bytes':wheel.stat().st_size,'verification_directory':str(work)}))
