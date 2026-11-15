"""One declared Docker diagnostic, unchanged frozen component inside bounded GC."""
import argparse,gc,hashlib,json,os,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;RESEARCH=HERE.parent;ROOT=RESEARCH.parents[2]

def worker(steps,output):
    # Import only in the isolated child. The parent never imports model modules.
    sys.path.insert(0,str(ROOT));sys.path.insert(0,str(RESEARCH));sys.path.insert(0,str(RESEARCH/'coupled_reanchoring_v1'))
    from control_worker_gc_v1 import ControlWorkerGC
    import run_coupled as component
    def hashes():
        result=component.hashes()
        for p in (RESEARCH/'control_worker_gc_v1.py',RESEARCH/'test_control_worker_gc_v1.py',Path(__file__)):
            result[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
        return result
    before=hashes();started=time.perf_counter();raw=str(Path(output).with_name('component_raw.json'))
    with ControlWorkerGC(max_steps=steps) as control:
        # The sealed runner's validated loop is exactly range(steps). This mode
        # leaves its source/call path unchanged; per-tick mission users instead
        # call control_step() and physical_step_completed() explicitly.
        control.reserve_declared_steps(steps)
        result=component.run(cases=1,start_case=0,steps=steps,output=raw)
        case=result['cases'][0]
        if case['completed_steps']==steps and case['error'] is None:
            control.report_declared_result(attempted_steps=steps,completed_words=steps)
    after=hashes();report={'version':'gc_worker_diagnostic_v1','declared_scenario':'same frozen coupled component case0',
        'declared_steps':steps,'runtime_uid':os.getuid(),'runtime_pid':os.getpid(),'gc_context':control.report,
        'component':result,'source_hashes_stable':before==after,'source_sha256':before,
        'component_functional_passed':case['functional_behavior_passed'],'component_timing_passed':case['timing_gate_passed'],
        'gc_discipline_passed':control.report['prior_enabled_state_restored'] and not control.report['collections_occurred_inside_control'],
        'actual_attempts_known_from_successful_frozen_loop':control.declared_result_reported,
        'wall_elapsed_including_setup_and_preemption_s':time.perf_counter()-started,
        'flight_certificate':False,'no_hardware_or_worst_case_timing_claim':True,'old_reports_unchanged':True}
    Path(output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('component','source_sha256')},indent=2),flush=True)
    return report

def launch(steps):
    initial={'enabled':gc.isenabled(),'thresholds':gc.get_threshold()};started=time.perf_counter()
    image='sha256:404a8ed9de35680a82f627899d018e1cc4a8acb90845c88080bc795eaae0a281'
    name='ares-gc-worker-v1-'+str(os.getpid());output='/tmp/gc_component_diagnostic.json'
    relative=str(Path(__file__).relative_to(ROOT))
    command=['docker','create','--name',name,'--network','none','--cpuset-cpus','7',
        '--mount',f'type=bind,src={ROOT},dst=/fixture,readonly','--workdir','/tmp','--entrypoint','python',image,
        '/fixture/'+relative,'--worker','--steps',str(steps),'--output',output]
    created=subprocess.run(command,text=True,capture_output=True,check=True).stdout.strip()
    with (HERE/'docker_gc_diagnostic.log').open('w') as log:
        start_result=subprocess.run(['docker','start','-a',created],stdout=log,stderr=subprocess.STDOUT)
    state=json.loads(subprocess.check_output(['docker','inspect',created],text=True))[0]['State']
    for remote,local in ((output,'docker_gc_diagnostic.json'),('/tmp/component_raw.json','docker_component_raw.json')):
        subprocess.run(['docker','cp',created+':'+remote,str(HERE/local)],check=True)
    subprocess.run(['docker','rm',created],text=True,capture_output=True,check=True)
    current={'enabled':gc.isenabled(),'thresholds':gc.get_threshold()}
    result=json.loads((HERE/'docker_gc_diagnostic.json').read_text())
    old=json.loads((ROOT/'reports/phase_4_verified_safety_artifacts/docker_runtime02.json').read_text())
    oldcase=old['cases'][0];newcase=result['component']['cases'][0]
    changed=[]
    for a,b in zip(oldcase['trace'],newcase['trace']):
        for key,value in a.items():
            if key!='guard_timing' and b.get(key)!=value:changed.append({'tick':a['tick'],'field':key})
    envelope={'version':'gc_worker_diagnostic_v1.docker_launch','image_id':image,'container_id':created,
        'runtime_exit_code':state['ExitCode'],'docker_start_exit_code':start_result.returncode,
        'configured_steps':steps,'worker_cpu_affinity':[7],'container_network':'none','repository_mount':'read-only',
        'parent_gc_before':initial,'parent_gc_after':current,'parent_gc_settings_unchanged':initial==current,
        'same_prior_scenario_first10_non_timing_fields_changed':changed,
        'old_timing_failure_report_preserved':True,'runtime_uid':result['runtime_uid'],
        'component_functional_passed':result['component_functional_passed'],'component_timing_passed':result['component_timing_passed'],
        'gc_discipline_passed':result['gc_discipline_passed'],'source_hashes_stable':result['source_hashes_stable'],
        'elapsed_wall_s':time.perf_counter()-started,'flight_certificate':False,'no_worst_case_timing_claim':True}
    (HERE/'docker_launch.json').write_text(json.dumps(envelope,indent=2)+'\n')
    print(json.dumps(envelope,indent=2))
    return envelope

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worker',action='store_true');parser.add_argument('--steps',type=int,default=1000)
    parser.add_argument('--output',default='/tmp/gc_component_diagnostic.json');args=parser.parse_args()
    if not 1<=args.steps<=1000:parser.error('frozen component diagnostic must have1..1000 steps')
    worker(args.steps,args.output) if args.worker else launch(args.steps)
