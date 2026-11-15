"""Run eight independent guarded single-thread CPU SIL workers; preserve failures."""
import hashlib
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]


def run(run_id='retry_002'):
    if not run_id or '/' in run_id or run_id in ('.','..'):raise ValueError('simple new run identifier required')
    output_dir=HERE/run_id
    output_dir.mkdir(exist_ok=False)
    topology=[];selected=[];seen=set()
    for cpu in sorted(os.sched_getaffinity(0)):
        path=Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        package=int((path/'physical_package_id').read_text());core=int((path/'core_id').read_text())
        topology.append({'cpu':cpu,'package':package,'core':core})
        if (package,core) not in seen:seen.add((package,core));selected.append(cpu)
    if len(selected)<8:raise RuntimeError('eight distinct physical cores required for this declared sweep')
    jobs=[];started=time.perf_counter();stopped=False
    for case,cpu in enumerate(selected[:8]):
        log=output_dir/f'case_{case}.stdout.log';handle=log.open('w')
        command=['taskset','-c',str(cpu),sys.executable,str(HERE/'run_coupled.py'),
                 '--start-case',str(case),'--cases','1','--steps','1000','--output',f'{run_id}/case_{case}.json']
        env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        process=subprocess.Popen(command,cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,env=env)
        jobs.append({'case':case,'cpu':cpu,'process':process,'log':log,'handle':handle})
    delivered={}
    while any(job['process'].poll() is None for job in jobs):
        for job in jobs:
            text=job['log'].read_text();last=text.strip().splitlines()[-1] if text.strip() else ''
            if last and delivered.get(job['case'])!=last:
                delivered[job['case']]=last
                print(json.dumps({'case':job['case'],'latest':last}),flush=True)
            failed=('FUNCTIONAL_FAILURE ' in text or job['process'].poll() not in (None,0))
            if failed and not stopped:
                stopped=True;print('STOPPING_SWEEP_FUNCTIONAL_FAILURE case='+str(job['case']),flush=True)
                for other in jobs:
                    if other['process'].poll() is None:other['process'].send_signal(signal.SIGTERM)
        time.sleep(.2)
    cases=[];workers=[]
    for job in jobs:
        job['handle'].close();path=output_dir/f'case_{job["case"]}.json'
        if path.exists():
            result=json.loads(path.read_text());cases.extend(result['cases'])
            workers.append({'case':job['case'],'cpu':job['cpu'],'returncode':job['process'].returncode,
                'report_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'source_hashes_stable':result['source_hashes_stable'],'source_sha256':result['source_sha256'],
                'process_cpu_affinity':result['process_cpu_affinity'],'run_valid':result['run_valid']})
        else:
            cases.append({'case':job['case'],'passed':False,'functional_behavior_passed':False,
                'completed_steps':0,'error':{'type':'WorkerFailedBeforeReport','returncode':job['process'].returncode},'trace':[]})
            workers.append({'case':job['case'],'cpu':job['cpu'],'returncode':job['process'].returncode,'run_valid':False})
    report={'version':'coupled_reanchoring_v1.sweep','run_id':run_id,'conditional_software_SIL_only':True,'flight_certificate':False,
            'hardware_or_Gaussian_runtime_certified':False,'all_cases_retained':len(cases)==8,
            'case_count':len(cases),'requested_steps_per_case':1000,'physical_core_topology':topology,
            'affinity_is_not_exclusive_core_reservation':True,'workers':workers,'cases':cases,
            'all_source_hashes_stable':all(w.get('source_hashes_stable',False) for w in workers),
            'functional_failure_triggered_stop':stopped,'elapsed_s':time.perf_counter()-started,
            'all_functional_checks_passed':all(c['functional_behavior_passed'] for c in cases),
            'all_timing_gates_passed':all(c.get('timing_gate_passed',False) for c in cases),
            'all_cases_passed':all(c['passed'] for c in cases),
            'orchestrator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (output_dir/'full_sweep.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('cases','workers','physical_core_topology')}),flush=True)
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',default='retry_002')
    args=parser.parse_args();run(args.run_id)
