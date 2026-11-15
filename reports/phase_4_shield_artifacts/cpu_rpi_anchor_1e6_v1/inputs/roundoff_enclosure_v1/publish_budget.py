"""Exact rational ceilings of the analytic stored-word/projective budget rows."""
from pathlib import Path
from fractions import Fraction as F
import json,hashlib
OUT=Path(__file__).resolve().parent
r=json.loads((OUT/'conditional_budgets.json').read_text())
rows=[x for x in r['rows'] if x['status']=='conditional_forward_budget_available' and x['verified_decay'] and x['error_shell_refinement'] and x['range_unit_refinement'] and x['current_command_is_stored_binary32_word'] and x['world_position_abs_bound']==16]
assert len(rows)==2
chosen=['.000005']*3+['.000000065','.000000065','.00000053']+['.00000012','.00000012','.000000017']+['.000001','.000001','.0000002']+['.000007']+['0']*8
for row in rows:assert all(F(s)>=F.from_float(v) for s,v in zip(chosen,row['same_center_projective_state21_error']))
report={'schema_version':'ares.conditional_projective21_numeric_budget.v1','flight_certificate':False,'platform_contract_audited':False,
 'conditional_per_H_s':'1/100','state_order':'p3,v3,pi(rawq)_xyz3,w3,Fmotor,queuecurrent4,queueold4; same analytical physical-parameter centers',
 'component_error_upper_rational':[str(F(s)) for s in chosen],'component_error_upper_decimal':chosen,
 'bound_is_analytic_not_sample_fit':True,'world_initial_position_component_bound_m':'16',
 'new_command_deviation_box_SI':['.116','.224','.224','.0243'],
 'required_backend':'original eagerCPU13Dvendor/adapter with explicitly verified-decay primitive, fullprecisionRNE and statednorm/sqrt/dot contracts',
 'source_parameter_primitive_assumptions':r['primitive_assumptions'],'initial_raw_q_norm_shell':r['initial_quaternion_norm_shell'],
 'reference_quaternion':'pi(rawq)=rawq/||rawq|| exactpositivehemisphere; outputbudgetprojectsrawFloat32q again',
 'new_command_quantization':'Storedbinary32word is mathematicalflowinput. Verifiedcontrollerpointlaw error<=1e-6 mustbeaddedonceviaGamma separately; queuecopyaritherrorzero.',
 'exclusions':['controller arithmetic except separately verifiedpointlaw','batteryenergy/power updater; b providedinsidehardrange','GPU/compiledkernels','installedplatform/SSE/vector_norm implementation audit notdonehere','externalstate/map hardbound validity'],
 'source_sha256':{'conditional_budgets.json':hashlib.sha256((OUT/'conditional_budgets.json').read_bytes()).hexdigest(),'publish_budget.py':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
 'all_computed_rows_below_rational_ceiling':True}
temporary=OUT/'conditional_projective21_budget.tmp'
temporary.write_text(json.dumps(report,indent=2)+'\n');temporary.replace(OUT/'conditional_projective21_budget.json')
print('PUBLISHED',report['component_error_upper_rational'])
