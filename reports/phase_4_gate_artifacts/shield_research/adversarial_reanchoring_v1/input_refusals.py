"""Isolated preflight negative tests; no claim of executed collision-free flight."""
from copy import copy
from dataclasses import replace
from fractions import Fraction as F
import json,math,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import run_scenario as scenario
common=scenario.common
import torch
from bounded_service import WitnessMap
from reanchoring_shield_v1 import ProofBundle,InitialAdmission
from current_terminal_membership_v1 import current_terminal_membership


def run():
    torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');torch.set_flush_denormal(False)
    before=scenario.hashes();proof=ProofBundle(scenario.ROOT/'reports/phase_4_shield_artifacts/cpu_rpi_certificate_v1')
    admission=InitialAdmission.load(proof,scenario.RESEARCH/'public_trim_history_v1.json',0)
    known=WitnessMap.from_dict(json.loads((scenario.RESEARCH/'bounded_service_v1/known_free_map.json').read_text()),time_s=0.)
    packet=common.public_initial_packet(admission.row);anchor=admission.row['anchor'];records=[]
    def refuse(name,action,*,prior_valid_certificate=False):
        row={'input':name,'outcome':'unexpected_acceptance','bad_input_certified':None,'applied_commands':0,
             'executed_flight':False,'prior_valid_command_certificate':prior_valid_certificate}
        try:
            result=action()
            if isinstance(result,dict) and result.get('command') is None and result.get('conditional_motion_certificate') is False:
                row.update(outcome='preflight_refused',bad_input_certified=False,reason=result.get('reason'))
            else:row.update(reason='call returned without explicit refusal')
        except Exception as error:
            row.update(outcome='preflight_refused',bad_input_certified=False,reason=str(error),exception_type=type(error).__name__)
        records.append(row)
    def decide(changed):return scenario.make_guard(proof,known,admission).decide(changed,tick=0,proposed_anchor=anchor)
    for name,changes in (
        ('stale_packet',{'capture_time_s':-.02,'expires_s':-.01}),
        ('future_packet',{'capture_time_s':.01}),('Gaussian_source',{'source_kind':'gaussian'}),
        ('oversized_position_bound',{'position_error_m':.005}),
        ('oversized_velocity_bound',{'velocity_error_mps':.01})):
        refuse(name,lambda changes=changes:decide(replace(packet,contract=replace(packet.contract,**changes))))
    refuse('oversized_body_rate_bound',lambda:decide(replace(packet,angular_velocity_error_radps=.001)))
    refuse('negative_qw_packet_history_binding',lambda:decide(replace(packet,q=tuple(-v for v in packet.q))))
    def negative_chart():
        return current_terminal_membership(p=packet.p,v=packet.v,q=tuple(-v for v in packet.q),w=packet.w,
            anchor=anchor,hold_thrust=admission.row['hold_thrust'],issued_queue=admission.row['issued_queue'],
            trim_force_interval=admission.row['trim_force_interval'],motor_residual_interval=admission.row['motor_residual_interval'],shapes=proof.shapes)
    refuse('negative_qw_membership_chart_component',negative_chart)
    refuse('insufficient_known_free_reserve',lambda:scenario.make_guard(proof,WitnessMap(),admission).decide(packet,tick=0,proposed_anchor=anchor))
    refuse('service_lease_shorter_than_backup',lambda:scenario.make_guard(proof,known,admission,service_lease_expires_tick=999))
    broken=copy(proof);broken.shapes={key:[list(row) for row in matrix] for key,matrix in proof.shapes.items()}
    broken.shapes['planar_x'][0][0]=F(-1)
    refuse('malformed_nonSPD_Q',lambda:scenario.make_guard(broken,known,admission).decide(packet,tick=0,proposed_anchor=anchor))
    wrong_queue=[list(word) for word in admission.row['issued_queue']];wrong_queue[0][0]=math.nextafter(wrong_queue[0][0],math.inf)
    refuse('initial_transport_words_mismatch_admission',lambda:scenario.make_guard(proof,known,admission,issued_queue=wrong_queue))
    transport=scenario.make_guard(proof,known,admission)
    valid=transport.decide(packet,tick=0,proposed_anchor=anchor)
    wrong=list(valid['command']);wrong[0]=math.nextafter(wrong[0],math.inf)
    refuse('acknowledged_transport_word_mismatch',lambda:transport.acknowledge_issued(wrong),prior_valid_certificate=True)
    records[-1]['transport_latched_blocked']=transport.blocked
    records[-1]['note']='A prior valid word was planned but never applied; the mismatched word receives no certificate. No flight execution is claimed.'
    att=dict(common.ATTESTATION);att['guarded_IEEE_basic_arithmetic']=False
    refuse('missing_arithmetic_mode_attestation',lambda:scenario.make_guard(proof,known,admission,arithmetic_backend_attestation=att))
    # Actual backend-mode loss, before any command can advance the physical state.
    capture=json.loads((scenario.RESEARCH/'public_trim_capture_v1.json').read_text())
    profile=common.sample_flight_profile(common.load_json(scenario.ROOT/'ares/data/configs/drone_sim_v2.json'),9100,'low')
    private=common.PlantAndOracle(profile,capture['simulation_truth_for_replay_only'],0,scenario.AnalyticRenderer(),known)
    state_before=private.backend.plant.state.clone()
    def lost_mode():
        torch.set_num_threads(2)
        try:return private.backend.step(torch.tensor([valid['command']],dtype=torch.float32))
        finally:torch.set_num_threads(1)
    refuse('enforced_backend_thread_mode_lost',lost_mode)
    records[-1]['physical_state_unchanged']=bool(torch.equal(state_before,private.backend.plant.state))
    records[-1]['backend_latched_blocked']=private.backend._blocked
    after=scenario.hashes()
    result={'version':'adversarial_reanchoring_v1.input_refusals','conditional_software_SIL_only':True,
        'flight_certificate':False,'no_executed_flight_in_this_test':True,'all_inputs_retained':True,
        'records':records,'all_bad_inputs_refused':all(r['outcome']=='preflight_refused' for r in records),
        'applied_command_count':0,'source_hashes_stable':before==after,'source_sha256':before,
        'scope':'Preflight validation/transaction refusal only; no survival of permanent sensor or mode loss is claimed.'}
    (HERE/'input_refusals.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'},indent=2))
    return result
if __name__=='__main__':run()
