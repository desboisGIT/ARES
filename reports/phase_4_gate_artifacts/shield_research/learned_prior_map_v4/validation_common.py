"""Operational source binding; unused delivery code is recorded but not flight."""
from pathlib import Path
import hashlib,sys,json,math
DELIVERY_FILES=frozenset('ares/verified_safety/'+name for name in ['__init__.py','__main__.py','_launcher.py'])
def operational_hashes(all_hashes):return {k:v for k,v in all_hashes.items() if k not in DELIVERY_FILES}
def imported_delivery_modules():return sorted(n for n in sys.modules if n=='ares.verified_safety' or n.startswith('ares.verified_safety.'))
def verify_protocol(root,protocol):
    rows=protocol['cases']
    if len(rows)!=8 or {r['case'] for r in rows}!=set(range(8)) or len({r['task_seed'] for r in rows})!=8:raise ValueError('exact eight-case predeclared cohort required')
    if protocol['settings']!={'max_steps':4500,'history_seconds':1.0,'calibration_remaining':750,'service_lease_ticks':5500,'policy_rate_hz':60,'shield_rate_hz':100,'reach_thresholds_m':[.1,.35]}:raise ValueError('unknown frozen mission settings')
    for name,expected in protocol['operational_source_sha256'].items():
        p=Path(root)/name
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=expected:raise ValueError('frozen operational dependency differs: '+name)
    return True

def verdict(report,*,require_mission=True):
    completed=report.get('completed_steps');positive=report.get('positive_certificates')
    complete=type(completed) is int and completed>0 and type(positive) is int and positive==completed
    safety=complete and all(report.get(k)==0 for k in ['actual_C_misses','collisions','unknown_entries','queue_mismatches'])
    stable=report.get('sources_stable') is True and report.get('operational_sources_stable') is True and report.get('delivery_modules_imported')==[]
    gc=report.get('control_worker_gc',{});gc=gc if isinstance(gc,dict) else {};discipline=gc.get('prior_enabled_state_restored') is True and gc.get('collections_occurred_inside_control') is False
    functional=bool(safety and stable and discipline and report.get('error') is None and report.get('speed_limit_exceeding_steps',0)==0)
    maximum=report.get('all_in_shield_control_max_s')
    timing=bool(functional and report.get('all_in_shield_control_above10ms')==0 and type(maximum) in (int,float) and math.isfinite(maximum) and 0<maximum<=.01)
    mission=bool(functional and report.get('mission_success') is True)
    return {'functional_checks_passed':functional,'timing_checks_passed':timing,'mission_checks_passed':mission,
            'requested_checks_passed':bool(functional and timing and (mission or not require_mission))}
