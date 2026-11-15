"""Three fresh 1000-step physical scenarios; stop all on a functional failure."""
import json,os,signal,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]

def run():
    cases=('unknown_plane','occupied_plane','occupied_corner');jobs=[];stopped=False;start=time.perf_counter()
    for scenario,cpu in zip(cases,(4,5,6)):
        if cpu not in os.sched_getaffinity(0):raise RuntimeError('declared physical-core CPU unavailable')
        log=HERE/f'{scenario}.stdout.log';handle=log.open('w')
        cmd=['taskset','-c',str(cpu),sys.executable,str(HERE/'run_scenario.py'),'--scenario',scenario,'--steps','1000']
        process=subprocess.Popen(cmd,cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
        jobs.append((scenario,cpu,process,handle,log))
    while any(job[2].poll() is None for job in jobs):
        for name,cpu,process,handle,log in jobs:
            if not stopped and ('FUNCTIONAL_FAILURE ' in log.read_text() or process.poll() not in (None,0)):
                stopped=True;print('STOPPING_FUNCTIONAL_FAILURE '+name,flush=True)
                for other in jobs:
                    if other[2].poll() is None:other[2].send_signal(signal.SIGTERM)
        time.sleep(.2)
    records=[]
    for name,cpu,process,handle,log in jobs:
        handle.close();path=HERE/f'{name}.json'
        if path.exists():records.append(json.loads(path.read_text()))
        else:records.append({'scenario':name,'passed':False,'error':{'type':'WorkerFailedBeforeReport','returncode':process.returncode},'trace':[]})
    report={'version':'adversarial_reanchoring_v1','all_scenarios_retained':True,'conditional_software_SIL_only':True,
        'flight_certificate':False,'all_scenarios_passed':all(r['passed'] for r in records),
        'functional_failure_triggered_stop':stopped,'elapsed_s':time.perf_counter()-start,
        'worker_affinity':dict(zip(cases,(4,5,6))),'exclusive_core_reservation':False,'scenarios':records}
    (HERE/'all_scenarios.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='scenarios'},indent=2))
    return report
if __name__=='__main__':run()
