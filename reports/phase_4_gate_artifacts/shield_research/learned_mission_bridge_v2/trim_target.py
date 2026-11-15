"""Untrusted trim-word proposal from public feedback; not an admission proof."""
from fractions import Fraction as F
from pathlib import Path
import sys,math
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from verified_backup_point_v2 import output32,I

def choose_trim_word(records, *, nominal_mass=.5, nominal_capacity_J=54000.,history_steps=100):
    if len(records)<20:raise ValueError('public command/velocity history required')
    rows=records[-history_steps:];t=[r['global_tick']*.01 for r in rows]
    v=[float(r['public_packet']['v'][2]) for r in rows]
    mid=sum(t)/len(t);vm=sum(v)/len(v);den=sum((x-mid)**2 for x in t)
    if den<=0:raise ValueError('distinct ordered public feedback epochs required')
    acceleration=sum((x-mid)*(y-vm) for x,y in zip(t,v))/den
    mean_thrust=sum(r['issued_command'][0] for r in rows)/len(rows)
    current_velocity=sum(v[-20:])/len(v[-20:])
    balance=mean_thrust-nominal_mass*acceleration
    # Center the constant-command window about zero velocity. Anticipate its
    # midpoint battery loss with a NOMINAL public power model, never true E/m.
    velocity_correction=0.  # Hold balance; a constant deceleration would overshoot during2.7s.
    midpoint_battery_correction=1.35*mean_thrust*(.2*8*max(0.,mean_thrust)**1.5/nominal_capacity_J)/.98
    estimate=balance+velocity_correction+midpoint_battery_correction
    estimate=max(4.,min(6.5,estimate));word,error=output32(I(F.from_float(estimate)))
    return {'target_word':word,'nominal_mass':nominal_mass,'nominal_capacity_J':nominal_capacity_J,
        'public_sample_count':len(rows),'mean_issued_thrust_N':mean_thrust,'public_vz_slope_mps2':acceleration,
        'mean_recent_public_vz_mps':current_velocity,'zero_acceleration_word_estimate_N':balance,
        'velocity_correction_N':velocity_correction,'midpoint_battery_correction_N':midpoint_battery_correction,
        'word_error_exact':str(error),'proposal_only_not_a_trim_proof':True,'sampled_mass_motor_energy_used':False}
