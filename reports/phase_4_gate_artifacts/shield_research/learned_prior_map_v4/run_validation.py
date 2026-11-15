"""Strict immutable eight-case validation; every attempted row stays counted."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,time
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent.parents[2]
sys.path.insert(0,str(HERE))
from validation_common import verdict,verify_protocol

def run(output,checkpoint,manifest=None,cpu=None):
    manifest=Path(manifest or HERE/'validation_cohort_v1.json').resolve();raw=manifest.read_bytes();protocol=json.loads(raw);sha=hashlib.sha256(raw).hexdigest()
    verify_protocol(ROOT,protocol)
    target=Path(output).resolve();workers=target.with_suffix('.workers')
    if target.exists() or workers.exists():raise FileExistsError('validation outputs already exist; refusing overwrite or retry')
    target.parent.mkdir(parents=True,exist_ok=True);workers.mkdir()
    rows=[];started=time.perf_counter()
    for plan in protocol['cases']:
        case=plan['case'];result=workers/f'case_{case}.json';log=workers/f'case_{case}.log';s=protocol['settings']
        cmd=[sys.executable,str(HERE/'run_mission.py'),'--case',str(case),'--steps',str(s['max_steps']),'--task-seed',str(plan['task_seed']),
            '--history-seconds',str(s['history_seconds']),'--calibration-remaining',str(s['calibration_remaining']),
            '--protocol',str(manifest),'--checkpoint',str(checkpoint),'--output',str(result)]
        if cpu is not None:cmd=['taskset','-c',str(cpu),*cmd]
        print('VALIDATION_START',case,plan['task_seed'],flush=True);launch_error=None;code=None;t=time.perf_counter()
        with log.open('x') as stream:
            try:code=subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT,timeout=600,check=False).returncode
            except Exception as exc:launch_error={'type':type(exc).__name__,'message':str(exc)}
        if result.exists():
            try:
                report=json.loads(result.read_text());status=verdict(report)
                bound=(report.get('case')==case and report.get('task_seed')==plan['task_seed'] and report.get('requested_steps')==s['max_steps']
                    and report.get('profile_mode')==plan['profile_mode'] and report.get('physical_profile_seed')==plan['profile_seed']
                    and report.get('protocol_sha256')==sha and report.get('protocol_unchanged') is True)
            except Exception as exc:
                report={'error':{'type':'WorkerReportParseError','message':str(exc)},'completed_steps':None};status=verdict(report);bound=False
        else:
            report={'error':launch_error or {'type':'MissingWorkerReport','message':'see retained worker log'},'completed_steps':None};status=verdict(report);bound=False
        okay=code==0 and launch_error is None and bound and status['requested_checks_passed']
        row={**plan,**status,'passed':bool(okay),'worker_returncode':code,'launch_error':launch_error,'worker_binding_valid':bound,
            'completed_steps':report.get('completed_steps'),'mission_success':report.get('mission_success',False),
            'duration_sim_s':report.get('completed_steps',0)/100 if report.get('completed_steps') is not None else None,
            'all_in_shield_control_max_s':report.get('all_in_shield_control_max_s'),'all_in_shield_control_above10ms':report.get('all_in_shield_control_above10ms'),
            'safety_counts':{k:report.get(k) for k in ['collisions','unknown_entries','actual_C_misses','queue_mismatches']},
            'source_hashes_stable':report.get('sources_stable',False),'operational_sources_stable':report.get('operational_sources_stable',False),
            'error':report.get('error'),'result':str(result),'result_sha256':hashlib.sha256(result.read_bytes()).hexdigest() if result.exists() else None,
            'log':str(log),'worker_wall_s':time.perf_counter()-t}
        rows.append(row);print('VALIDATION_CASE',json.dumps(row),flush=True)
    summary={'schema_version':'ares.prior_map_eight_case_validation.v1','protocol_sha256':sha,'protocol':str(manifest),
        'cases':rows,'planned_denominator':8,'all_declared_cases_attempted':len(rows)==8,'all_attempts_retained':True,
        'source_hashes_stable':all(r['source_hashes_stable'] for r in rows),'run_valid':all(r['worker_binding_valid'] for r in rows),
        'functional_checks_passed':all(r['functional_checks_passed'] for r in rows),'timing_checks_passed':all(r['timing_checks_passed'] for r in rows),
        'mission_checks_passed':all(r['mission_checks_passed'] for r in rows),'all_cases_passed':len(rows)==8 and all(r['passed'] for r in rows),
        'no_tuning_or_same_code_retries_within_cohort':True,'conditional_SIL_only':True,'phase4_gate_passed':False,'wall_elapsed_s':time.perf_counter()-started}
    with target.open('x') as stream:stream.write(json.dumps(summary,indent=2)+'\n')
    print('VALIDATION_SUMMARY',target,summary['all_cases_passed'],flush=True);return summary
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--manifest',type=Path);p.add_argument('--cpu',type=int);a=p.parse_args();result=run(a.output,a.checkpoint,a.manifest,a.cpu)
    raise SystemExit(0 if result['all_cases_passed'] else 1)
