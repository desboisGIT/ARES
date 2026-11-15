"""Versioned isolated GC wrapper; frozen component protocol remains unchanged."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
BASE=HERE.parent


def load_file(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dependencies():
    runner=load_file('portable_frozen_component',BASE/'coupled_reanchoring_v1/run_coupled.py')
    gc_module=load_file('portable_control_gc',BASE/'control_worker_gc_v1.py')
    return runner,gc_module


def snapshot(runner):
    closure=dict(runner.hashes())
    root=BASE.parents[2]
    for path in [BASE/'control_worker_gc_v1.py',*HERE.glob('*.py')]:
        closure[str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
    return closure


def write_new(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False)
        stream.write('\n')


def execute_wrapped(runner,gc_class,*,case,steps,raw_output):
    """One frozen declared loop; counts are reported, not per-tick instrumentation."""
    if type(case) is not int or not 0<=case<8 or type(steps) is not int or not 1<=steps<=1000:
        raise ValueError('case0..7 and steps1..1000 required')
    if os.path.lexists(raw_output):
        raise FileExistsError('raw output already exists')
    with gc_class(max_steps=steps) as scope:
        scope.reserve_declared_steps(steps)
        raw=runner.run(cases=1,start_case=case,steps=steps,output=str(raw_output))
        rows=raw.get('cases',[])
        if len(rows)!=1 or rows[0].get('case')!=case:
            raise ValueError('frozen runner returned wrong case count or ID')
        row=rows[0]
        attempted=len(row.get('trace',[]))
        if row.get('error'):
            error_tick=row['error'].get('tick')
            if type(error_tick) is not int or not 0<=error_tick<steps:
                raise ValueError('invalid error tick')
            attempted=max(attempted,error_tick+1)
        completed=row.get('guard_total_latency_s',{}).get('count',row.get('completed_steps'))
        if type(completed) is not int or not 0<=completed<=steps:
            raise ValueError('invalid reported physical step count')
        # Acknowledgement timing count records commands whose physical step
        # returned and was acknowledged. Failed partial physical steps are not
        # claimed complete; raw error/trace evidence remains unchanged.
        attempted=max(attempted,completed)
        if attempted>steps:raise ValueError('reported attempt exceeds horizon')
        scope.report_declared_result(attempted_steps=attempted,completed_words=completed)
    return raw,scope.report


def status(cases,*,mode,expected_cases,steps,stable):
    if mode not in ('smoke','validate'):
        raise ValueError('unknown mode')
    complete=(len(cases)==expected_cases and {row.get('case') for row in cases}==set(range(expected_cases))
              and all(type(row.get('case')) is int and row.get('requested_steps')==steps
                      and row.get('completed_steps')==steps for row in cases))
    functional=bool(complete and stable and all(row.get('functional_behavior_passed') is True for row in cases))
    timing=bool(complete and stable and all(row.get('timing_gate_passed') is True for row in cases))
    conditional=functional and timing
    return {'mode':mode,'functional_checks_passed':functional,'timing_checks_passed':timing,
            'conditional_SIL_passed':conditional,
            'requested_check_passed':functional if mode=='smoke' else conditional}


def child(case,steps,raw_output,metadata_output):
    if any(os.path.lexists(path) for path in (raw_output,metadata_output)):
        raise FileExistsError('worker outputs must be fresh')
    runner,gc_module=dependencies()
    before=snapshot(runner)
    raw,gc_report=execute_wrapped(runner,gc_module.ControlWorkerGC,case=case,steps=steps,raw_output=raw_output)
    after=snapshot(runner)
    stable=(before==after and raw.get('source_hashes_stable') is True and raw.get('run_valid') is True)
    write_new(metadata_output,{'version':'portable_component_worker.v2','case':case,'requested_steps':steps,
              'source_sha256_before':before,'source_sha256_after':after,'source_hashes_stable':stable,
              'source_proof_input_hashes_stable':before==after,'gc':gc_report,
              'raw_report':str(raw_output),'raw_report_sha256':hashlib.sha256(raw_output.read_bytes()).hexdigest(),
              'flight_certificate':False,'hardware_deployment_claim':False})
    return 0 if stable else 1


def parent(*,mode,cases,steps,output):
    if mode=='validate' and (cases!=8 or steps!=1000):
        raise ValueError('strict validate requires eight cases and1000 steps')
    if not 1<=cases<=8 or not 1<=steps<=1000:
        raise ValueError('cases1..8,steps1..1000 required')
    directory=output.with_name(output.name+'.workers')
    if os.path.lexists(output) or os.path.lexists(directory):
        raise FileExistsError('output and worker directory must be fresh')
    output.parent.mkdir(parents=True,exist_ok=True)
    directory.mkdir()
    runner,_=dependencies()
    before=snapshot(runner)
    records=[];workers=[]
    for case in range(cases):
        raw_path=directory/f'case_{case}.json';meta_path=directory/f'case_{case}.metadata.json'
        log_path=directory/f'case_{case}.log'
        command=[sys.executable,str(Path(__file__).resolve()),'--worker','--case',str(case),
                 '--steps',str(steps),'--output',str(raw_path),'--metadata-output',str(meta_path)]
        with log_path.open('x') as log:
            completed=subprocess.run(command,cwd=BASE.parents[2],env=dict(os.environ),stdout=log,stderr=subprocess.STDOUT)
        metadata=json.loads(meta_path.read_text()) if meta_path.is_file() else {}
        raw=json.loads(raw_path.read_text()) if raw_path.is_file() else {}
        rows=raw.get('cases',[])
        worker_ok=(completed.returncode==0 and metadata.get('source_hashes_stable') is True
                   and metadata.get('source_sha256_before')==before and metadata.get('source_sha256_after')==before
                   and metadata.get('gc',{}).get('prior_enabled_state_restored') is True
                   and metadata.get('gc',{}).get('collections_occurred_inside_control') is False)
        workers.append({'case':case,'returncode':completed.returncode,'valid':worker_ok,
                        'metadata_path':str(meta_path),'report_path':str(raw_path),'log':str(log_path),
                        'gc':metadata.get('gc')})
        if len(rows)==1 and rows[0].get('case')==case:
            records.extend(rows)
    after=snapshot(runner)
    stable=before==after and all(worker['valid'] for worker in workers)
    result={'version':'portable_component_aggregate.v2',**status(records,mode=mode,expected_cases=cases,steps=steps,stable=stable),
            'cases':records,'workers':workers,'source_sha256':before,'source_hashes_stable':stable,
            'source_proof_input_hashes_stable':before==after,'run_valid':stable,
            'all_requested_cases_passed':False,'all_cases_retained':True,
            'frozen_protocol_unchanged':True,'one_isolated_child_per_case':True,
            'gc_counts_reported_not_per_tick_instrumented':True,
            'flight_certificate':False,'hardware_deployment_claim':False,'worst_case_latency_proved':False}
    result['all_requested_cases_passed']=result['conditional_SIL_passed']
    write_new(output,result)
    print(json.dumps({key:result[key] for key in ('mode','functional_checks_passed','timing_checks_passed',
                                                 'conditional_SIL_passed','requested_check_passed')}|{'output':str(output),'flight_certificate':False}))
    return 0 if result['requested_check_passed'] else 1


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=('smoke','validate'),nargs='?',default='smoke')
    parser.add_argument('--cases',type=int)
    parser.add_argument('--steps',type=int)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--case',type=int)
    parser.add_argument('--metadata-output',type=Path)
    args=parser.parse_args(argv)
    output=args.output.absolute()
    if args.worker:
        if args.case is None or args.steps is None or args.metadata_output is None:parser.error('worker requires case,steps,metadata-output')
        return child(args.case,args.steps,output,args.metadata_output.absolute())
    cases=args.cases if args.cases is not None else (1 if args.mode=='smoke' else 8)
    steps=args.steps if args.steps is not None else (10 if args.mode=='smoke' else 1000)
    return parent(mode=args.mode,cases=cases,steps=steps,output=output)


if __name__=='__main__':raise SystemExit(main())
