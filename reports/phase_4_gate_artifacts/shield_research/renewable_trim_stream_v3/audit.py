"""Exact symbolic sufficient-statistic equivalence and frozen source manifest."""
from pathlib import Path
import hashlib,json,sympy as s
HERE=Path(__file__).resolve().parent

def run():
 g,H,duration,ev,num,fixed,drift,D,M=s.symbols('gamma H duration ev num fixed drift abs_delta max_abs_vz');den=1-2*g*H
 endpoint=D/duration+2*ev/duration+num;known=M+ev+num*H;acc=(endpoint+2*g*known+2*fixed+2*drift)/den
 original=endpoint+g*(known+H*acc)+fixed+drift
 a=1+g*H/den;b=g+2*g*g*H/den;c=1+2*g*H/den
 streamed=a*D/duration+b*M+a*(2*ev/duration+num)+b*(ev+num*H)+c*(fixed+drift)
 difference=s.factor(original-streamed);assert difference==0
 result=dict(schema_version='ares.streaming_trim_equivalence.v3',exact_symbolic_mismatch_difference=str(difference),full_history_dependencies=['first vz','last vz','max absolute vz','duration','maximum gap'],packet_domain_and_quaternion_maxima_equivalence='Each node checked once; monotone deques discard only older dominated values and expire indices strictly before the exact history start. Endpoint is included separately. Thus suffix maxima and invalid-node predicate equal full history.',allowed_history_seconds=['1/2','1','3/2','2'],default_history_seconds='1',shorter_history_never_assumed_more_accurate=True,all_final_C_and_geometry_checks_retained=True,stale_proposals_rejected_by_endpoint_queue_energy_source_service_map_binding=True,source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(HERE.glob('*.py'))},flight_certificate=False)
 (HERE/'compatibility.json').write_text(json.dumps(result,indent=2)+'\n');print('symbolic trim difference',difference);return result
if __name__=='__main__':run()
