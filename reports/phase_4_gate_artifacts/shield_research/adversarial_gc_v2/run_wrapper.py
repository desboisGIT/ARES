"""Prepared bounded-GC wrapper; default invocation starts NO physical runs.

Use --run only after the separately coordinated timing window. Frozen scenario
code is imported unchanged; child-local HERE relocation isolates output files.
"""
import argparse,ast,gc,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;RESEARCH=HERE.parent;ROOT=RESEARCH.parents[2]
ORIGINAL=RESEARCH/'adversarial_reanchoring_v1'
SCENARIOS=('unknown_plane','occupied_plane','occupied_corner')
GC_SHA='8516b07975569dd82eb9183ff2585493b58bdea00c4c51461eb4bf4422c01636'
LAUNCHER_MODULE='ares.verified_safety._launcher'
COMPATIBILITY=HERE/'compatibility_manifest_v1.json'

def file_hash(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def old_manifest():
    return {str(path.relative_to(ROOT)):file_hash(path) for path in sorted(ORIGINAL.rglob('*'))
            if path.is_file() and '__pycache__' not in path.parts}

def scan_shared_execution(source_inventory):
    # Read-only static check of checksum-bound shared research modules and
    # actually imported ARES modules; wrapper metadata checks are excluded.
    imported={Path(getattr(module,'__file__','')).resolve() for module in tuple(sys.modules.values())
              if getattr(module,'__file__',None) and str(getattr(module,'__file__','')).endswith('.py')}
    checked=[];references=[]
    for relative in source_inventory:
        path=ROOT/relative
        if path.suffix!='.py' or path==Path(__file__).resolve():continue
        if relative.startswith('ares/') and path.resolve() not in imported:continue
        tree=ast.parse(path.read_text());checked.append(relative)
        for node in ast.walk(tree):
            target=None
            if isinstance(node,ast.Import):
                for alias in node.names:
                    if 'verified_safety' in alias.name or '_launcher' in alias.name:target=alias.name
            elif isinstance(node,ast.ImportFrom) and node.module and ('verified_safety' in node.module or '_launcher' in node.module):target=node.module
            elif isinstance(node,ast.Constant) and isinstance(node.value,str) and ('ares.verified_safety' in node.value or node.value=='_launcher'):target=node.value
            if target is not None:references.append({'file':relative,'line':node.lineno,'reference':target})
    return {'checked_source_files':checked,'import_or_lookup_references':references}

def run_child(scenario,output_dir):
    output_dir=Path(output_dir).resolve();output_dir.mkdir(exist_ok=False)
    sys.dont_write_bytecode=True
    sys.path.insert(0,str(ROOT));sys.path.insert(0,str(RESEARCH));sys.path.insert(0,str(ORIGINAL))
    import run_scenario as frozen
    from control_worker_gc_v1 import ControlWorkerGC
    def sources():
        result=frozen.hashes() # Called only with the ORIGINAL HERE binding.
        for path in (Path(__file__),RESEARCH/'control_worker_gc_v1.py',COMPATIBILITY):
            result[str(path.relative_to(ROOT))]=file_hash(path)
        return result
    before=sources();evidence_before=old_manifest();reference=json.loads((ORIGINAL/f'{scenario}.json').read_text())
    source_drift={key:{'reference':value,'current':before.get(key)} for key,value in reference['source_sha256'].items() if before.get(key)!=value}
    compatibility=json.loads(COMPATIBILITY.read_text());allowances=compatibility['allowances'];accepted_drift={};unapproved_drift={}
    if set(allowances)!={'ares/verified_safety/_launcher.py'}:raise RuntimeError('only the explicitly authorized packaging launcher may differ')
    expected_launcher_binding=(reference['source_sha256'].get('ares/verified_safety/_launcher.py')==allowances['ares/verified_safety/_launcher.py']['reference_sha256']
        and before.get('ares/verified_safety/_launcher.py')==allowances['ares/verified_safety/_launcher.py']['current_sha256'])
    for key,change in source_drift.items():
        allowed=allowances.get(key)
        if allowed and change['reference']==allowed['reference_sha256'] and change['current']==allowed['current_sha256']:
            accepted_drift[key]=change
        else:unapproved_drift[key]=change
    launcher_absent_before=LAUNCHER_MODULE not in sys.modules
    shared_scan_before=scan_shared_execution(before);import_attempts=[]
    def observe_import(event,args):
        if event=='import' and args and isinstance(args[0],str) and args[0].startswith('ares.verified_safety'):
            import_attempts.append(args[0])
    sys.addaudithook(observe_import) # Child-only observer, no command/control mutation.
    roots_before={name:str(getattr(frozen,name)) for name in ('HERE','RESEARCH','ROOT')}
    original_here=frozen.HERE;original_code=frozen.run.__code__;result=None;control=None;error=None
    started=time.perf_counter()
    try:
        if file_hash(RESEARCH/'control_worker_gc_v1.py')!=GC_SHA:raise RuntimeError('frozen GC helper hash mismatch')
        if not expected_launcher_binding:raise RuntimeError('authorized old/current launcher binding differs')
        if unapproved_drift:raise RuntimeError('non-allowlisted protocol source bytes differ from reference')
        if not launcher_absent_before or shared_scan_before['import_or_lookup_references']:
            raise RuntimeError('unused-launcher execution precondition failed')
        # The parent authorized output isolation only. RESEARCH/ROOT/imports,
        # law, measurements, proof, intent and scenario arguments stay unchanged.
        frozen.HERE=output_dir
        with ControlWorkerGC(max_steps=1000) as control:
            control.reserve_declared_steps(1000)
            result=frozen.run(scenario,steps=1000,output=str(output_dir/'inner_scenario.json'))
            if result['completed_steps']==1000 and result['error'] is None:
                control.report_declared_result(attempted_steps=1000,completed_words=1000)
    except BaseException as exc:
        error={'type':type(exc).__name__,'message':str(exc)}
    finally:
        frozen.HERE=original_here
    after=sources();evidence_after=old_manifest()
    roots_after={name:str(getattr(frozen,name)) for name in ('HERE','RESEARCH','ROOT')}
    launcher_absent_after=LAUNCHER_MODULE not in sys.modules;shared_scan_after=scan_shared_execution(after)
    differences=[]
    if result is not None:
        if len(result['trace'])!=len(reference['trace']):differences.append({'field':'trace_length','old':len(reference['trace']),'new':len(result['trace'])})
        for old,new in zip(reference['trace'],result['trace']):
            for key,value in old.items():
                if key!='guard_timing' and new.get(key)!=value:differences.append({'tick':old['tick'],'field':key})
    else:differences.append({'field':'no_scenario_report'})
    report={'version':'adversarial_gc_v2','scenario':scenario,'configured_steps':1000,'isolated_child_pid':os.getpid(),
        'parent_pid':os.getppid(),'process_cpu_affinity':sorted(os.sched_getaffinity(0)),
        'output_rebinding':{'field':'run_scenario.HERE','original':str(original_here),'during_child_run':str(output_dir),
            'authorized_scope':'output paths and inner source inventory only','restored_in_finally':frozen.HERE==original_here,
            'ROOT_and_RESEARCH_unchanged':all(roots_before[n]==roots_after[n] for n in ('ROOT','RESEARCH')),
            'run_function_bytecode_identity_unchanged':frozen.run.__code__ is original_code},
        'outer_complete_source_sha256_before':before,'outer_complete_source_sha256_after':after,
        'outer_source_hashes_stable':before==after,'original_reference_source_drift':source_drift,
        'compatibility_manifest_sha256':file_hash(COMPATIBILITY),'accepted_unused_launcher_drift':accepted_drift,
        'unapproved_source_drift':unapproved_drift,'launcher_absent_before':launcher_absent_before,
        'launcher_absent_after':launcher_absent_after,'launcher_import_attempts':import_attempts,
        'shared_execution_scan_before':shared_scan_before,'shared_execution_scan_after':shared_scan_after,
        'unused_launcher_allowance_checks_passed':expected_launcher_binding and not unapproved_drift and launcher_absent_before and launcher_absent_after
            and not import_attempts and not shared_scan_before['import_or_lookup_references'] and not shared_scan_after['import_or_lookup_references']
            and before.get('ares/verified_safety/_launcher.py')==after.get('ares/verified_safety/_launcher.py'),
        'inner_source_inventory_relocation_note':'Inner hash stability is not complete provenance; outer inventory explicitly binds original scenario scripts and shared roots.',
        'original_evidence_manifest_before':evidence_before,'original_evidence_manifest_after':evidence_after,
        'original_evidence_unchanged':evidence_before==evidence_after,
        'reference_report_sha256':file_hash(ORIGINAL/f'{scenario}.json'),'non_timing_differences':differences,
        'same_functional_and_non_timing_trace':not differences,
        'gc_context':control.report if control is not None else None,'wrapper_error':error,
        'scenario_report':result,'elapsed_wall_s':time.perf_counter()-started,
        'functional_passed':bool(result and result['functional_behavior_passed']),
        'strict_actual_10ms_timing_passed':bool(result and result['timing_gate_passed']),
        'gc_discipline_passed':bool(control and control.report and control.report['prior_enabled_state_restored'] and not control.report['collections_occurred_inside_control']),
        'conditional_software_SIL_only':True,'flight_certificate':False,'no_hardware_or_WCET_claim':True,
        'all_old_failed_timing_reports_retained':evidence_before==evidence_after}
    report['passed']=all((report['functional_passed'],report['strict_actual_10ms_timing_passed'],report['gc_discipline_passed'],
        report['same_functional_and_non_timing_trace'],report['outer_source_hashes_stable'],report['original_evidence_unchanged'],
        report['unused_launcher_allowance_checks_passed'])) and error is None
    (output_dir/'wrapper_report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    if not report['functional_passed'] or error:print('FUNCTIONAL_FAILURE '+json.dumps(error or {'scenario':scenario}),flush=True)
    print(json.dumps({key:report[key] for key in ('scenario','functional_passed','strict_actual_10ms_timing_passed',
          'gc_discipline_passed','same_functional_and_non_timing_trace','outer_source_hashes_stable','original_evidence_unchanged','passed')}),flush=True)
    return report

def launch(run_id):
    if not run_id or '/' in run_id or run_id in ('.','..'):raise ValueError('simple new run id required')
    destination=HERE/run_id;destination.mkdir(exist_ok=False)
    parent_before={'enabled':gc.isenabled(),'thresholds':gc.get_threshold()};old_before=old_manifest();jobs=[];stopped=False
    for scenario,cpu in zip(SCENARIOS,(4,5,6)):
        if cpu not in os.sched_getaffinity(0):raise RuntimeError('declared CPU affinity unavailable')
        log=destination/f'{scenario}.log';handle=log.open('w')
        args=['taskset','-c',str(cpu),sys.executable,str(Path(__file__)), '--child','--scenario',scenario,'--output-dir',str(destination/scenario)]
        env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
        process=subprocess.Popen(args,cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,env=env)
        jobs.append((scenario,process,handle,log))
    while any(job[1].poll() is None for job in jobs):
        for scenario,process,handle,log in jobs:
            if not stopped and ('FUNCTIONAL_FAILURE ' in log.read_text() or process.poll() not in (None,0)):
                stopped=True
                for other in jobs:
                    if other[1].poll() is None:other[1].send_signal(signal.SIGTERM)
        time.sleep(.2)
    reports=[]
    for scenario,process,handle,log in jobs:
        handle.close();path=destination/scenario/'wrapper_report.json'
        reports.append(json.loads(path.read_text()) if path.exists() else {'scenario':scenario,'passed':False,'wrapper_error':{'type':'WorkerFailedBeforeReport','exit':process.returncode}})
    parent_after={'enabled':gc.isenabled(),'thresholds':gc.get_threshold()}
    report={'version':'adversarial_gc_v2.sweep','run_id':run_id,'all_scenarios_retained':True,'scenarios':reports,
        'parent_gc_before':parent_before,'parent_gc_after':parent_after,'parent_gc_settings_unchanged':parent_before==parent_after,
        'all_original_evidence_unchanged':old_before==old_manifest(),'functional_failure_triggered_stop':stopped,
        'all_scenarios_passed':all(r['passed'] for r in reports),'flight_certificate':False,'conditional_software_SIL_only':True}
    (destination/'all_scenarios.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report

def prepared():
    result={'version':'adversarial_gc_v2.prepared','status':'prepared_not_run','scenarios':SCENARIOS,'steps_each':1000,
        'helper_sha256':file_hash(RESEARCH/'control_worker_gc_v1.py'),'original_scenario_source_sha256':file_hash(ORIGINAL/'run_scenario.py'),
        'compatibility_manifest_sha256':file_hash(COMPATIBILITY),'wrapper_sha256':file_hash(Path(__file__)),'original_evidence_manifest':old_manifest(),
        'output_relocation_required':True,'outer_complete_source_and_evidence_checks':True,
        'new_scenario_or_proof_or_criterion':False,'timing_runs_started':False}
    (HERE/'prepared.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='original_evidence_manifest'},indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true')
    parser.add_argument('--run-id',default='attempt_001');parser.add_argument('--child',action='store_true')
    parser.add_argument('--scenario',choices=SCENARIOS);parser.add_argument('--output-dir');args=parser.parse_args()
    if args.child:
        if not args.scenario or not args.output_dir:parser.error('child requires scenario and output-dir')
        run_child(args.scenario,args.output_dir)
    elif args.run:launch(args.run_id)
    else:prepared()
