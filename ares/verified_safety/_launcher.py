"""Checksum-verified, isolated launch of byte-preserved conditional SIL code."""
import argparse
import hashlib
from importlib import resources
import io
import json
import os
from pathlib import Path,PurePosixPath
import stat
import subprocess
import sys
import tempfile
import zipfile

BUNDLE_SHA256 = '44c5e574a6668c49f30965f36fb5ae97065c1a60408e6812218c6220fa740091'
MANIFEST_SHA256 = 'b26f47087ba3c55e4772a5d27df505a0ba7916328718b9ea74c088958b3e1a4f'
RUNNER = 'reports/phase_4_gate_artifacts/shield_research/coupled_reanchoring_v1/run_coupled.py'


def checked_members(data,manifest,expected_sha256):
    """Validate the complete archive before creating any extraction files."""
    if hashlib.sha256(data).hexdigest()!=expected_sha256:
        raise ValueError('bundle checksum mismatch')
    checked={}
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for member in archive.infolist():
            name=member.filename
            path=PurePosixPath(name)
            if (not name or path.is_absolute() or '\\' in name or ':' in name
                    or any(part in ('','..','.') for part in name.split('/'))
                    or not name.startswith('reports/') or member.is_dir()):
                raise ValueError('unsafe archive member: '+name)
            if stat.S_ISLNK(member.external_attr>>16):
                raise ValueError('symlink archive member: '+name)
            if name in checked:
                raise ValueError('duplicate archive member: '+name)
            expected=manifest.get(name)
            if (not isinstance(expected,dict) or member.file_size!=expected.get('bytes')):
                raise ValueError('unexpected archive member or size: '+name)
            payload=archive.read(member)
            if hashlib.sha256(payload).hexdigest()!=expected.get('sha256'):
                raise ValueError('member checksum mismatch: '+name)
            checked[name]=payload
    if set(checked)!=set(manifest):
        raise ValueError('bundle member set mismatch')
    return checked


def load_members():
    package=resources.files(__package__)
    encoded=package.joinpath('bundle_members_v1.json').read_bytes()
    if hashlib.sha256(encoded).hexdigest()!=MANIFEST_SHA256:
        raise ValueError('member manifest checksum mismatch')
    return checked_members(package.joinpath('verified_safety_bundle_v1.zip').read_bytes(),
                           json.loads(encoded),BUNDLE_SHA256)


def report_status(report, *, mode, expected_cases, expected_steps, returncode=0):
    """Separate integration success from strict conditional timing validity."""
    if mode not in ('smoke','validate'):
        raise ValueError('unknown check mode')
    cases = report.get('cases', [])
    complete = (isinstance(cases,list) and len(cases)==expected_cases
                and all(isinstance(case,dict) for case in cases))
    if complete:
        identifiers=[case.get('case') for case in cases]
        complete=(all(type(identifier) is int for identifier in identifiers)
                  and set(identifiers)==set(range(expected_cases))
                  and all(case.get('requested_steps')==expected_steps
                          and case.get('completed_steps')==expected_steps for case in cases))
    stable = (returncode==0 and report.get('source_hashes_stable') is True
              and report.get('run_valid') is True)
    functional = bool(complete and stable and
                      all(case.get('functional_behavior_passed') is True for case in cases))
    timing = bool(complete and stable and
                  all(case.get('timing_gate_passed') is True for case in cases))
    conditional = functional and timing
    return {'mode':mode,'functional_checks_passed':functional,
            'timing_checks_passed':timing,'conditional_SIL_passed':conditional,
            'requested_check_passed':functional if mode=='smoke' else conditional}


def status_exit_code(status):
    return 0 if status['requested_check_passed'] else 1


def main(argv=None):
    parser=argparse.ArgumentParser(description='Conditional CPU SIL toolkit; no physical flight certificate')
    parser.add_argument('command',choices=('smoke','validate'))
    parser.add_argument('--device',choices=('cpu',),default='cpu')
    parser.add_argument('--cases',type=int)
    parser.add_argument('--steps',type=int)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args(argv)
    cases=args.cases if args.cases is not None else (1 if args.command=='smoke' else 8)
    steps=args.steps if args.steps is not None else (10 if args.command=='smoke' else 1000)
    if not 1<=cases<=8 or not 1<=steps<=1000:
        parser.error('cases must be 1..8 and steps 1..1000')
    if args.command=='validate' and (cases!=8 or steps!=1000):
        parser.error('validate uses eight cases and 1000 steps; use smoke for smaller checks')
    if args.output is None:
        output=Path(tempfile.mkdtemp(prefix='ares-verified-safety-results-'))/'result.json'
    else:
        output=args.output.expanduser().absolute()
    if os.path.lexists(output):
        parser.error('output already exists; choose a fresh path')
    log=output.with_name(output.name+'.log')
    if os.path.lexists(log):
        parser.error('output log already exists; choose a fresh path')
    output.parent.mkdir(parents=True,exist_ok=True)
    import ares
    installed=Path(ares.__file__).resolve().parent
    try:
        members=load_members()
        with tempfile.TemporaryDirectory(prefix='ares-verified-safety-runtime-') as directory:
            root=Path(directory)
            for name,payload in members.items():
                destination=root.joinpath(*PurePosixPath(name).parts)
                destination.parent.mkdir(parents=True,exist_ok=True)
                destination.write_bytes(payload)
            (root/'ares').symlink_to(installed,target_is_directory=True)
            env=dict(os.environ,PYTHONPATH=str(installed.parent))
            command=[sys.executable,str(root/RUNNER),'--cases',str(cases),'--steps',str(steps),'--output',str(output)]
            with log.open('x') as stream:
                completed=subprocess.run(command,cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT)
            report=json.loads(output.read_text()) if output.is_file() else {}
            status=report_status(report,mode=args.command,expected_cases=cases,
                                 expected_steps=steps,returncode=completed.returncode)
        print(json.dumps({**status,'output':str(output),'log':str(log),
                          'flight_certificate':False,'hardware_deployment_claim':False}))
        return status_exit_code(status)
    except (OSError,ValueError,zipfile.BadZipFile) as error:
        print(json.dumps({'mode':args.command,'functional_checks_passed':False,'timing_checks_passed':False,
                          'conditional_SIL_passed':False,'requested_check_passed':False,'error':str(error),'output':str(output),
                          'log':str(log),'flight_certificate':False}),file=sys.stderr)
        return 1
