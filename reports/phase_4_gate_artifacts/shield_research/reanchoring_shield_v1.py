"""Conditional full-state backup reservations for the bounded-service CPU SIL.

This consumer accepts public packets, map witnesses and command history only.
It cannot read a plant, sampled profile, actual motor state or true energy.
Certificates refer to the finite-horizon simulator model and stated enforced
service/backend contracts, never physical hardware or the Gaussian runtime.
"""
from fractions import Fraction as F
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE/'terminal_reservation_v1'))

from ares.safety_contract import validate_contract
from current_terminal_membership_v1 import current_terminal_membership, propagate_public_residuals, sqrt_bounds
from reservation import free_grid_reserve, rational
from verified_backup_point_v2 import backup_point


class ContractFailure(ValueError):
    pass


_ADMISSION_CACHE = {}


class ProofBundle:
    def __init__(self, path):
        self.path = Path(path).resolve()
        manifest = json.loads((self.path/'bundle_sha256.json').read_text())
        for name, expected in manifest.items():
            actual = hashlib.sha256((self.path/name).read_bytes()).hexdigest()
            if actual != expected:
                raise ContractFailure('proof dependency changed: '+name)
        self.certificate = json.loads((self.path/'certificate.json').read_text())
        c = self.certificate
        if (c.get('schema_version') != 'ares.conditional_cpu_backup_proof_bundle.v1'
                or c.get('conditional_endpoint_RPI_lemmas_closed') is not True
                or c.get('fixed_anchor') is not True):
            raise ContractFailure('required fixed-anchor theorem unavailable')
        for name, obligation in c['proof_obligations'].items():
            if not any(obligation.get(key) is True for key in (
                    'proved', 'proved_under_primitive_contract', 'proved_under_verified_backend')):
                raise ContractFailure('open mathematical obligation: '+name)
        for name, expected in c['frozen_physical_source51_sha256'].items():
            if hashlib.sha256((ROOT/'ares'/name).read_bytes()).hexdigest() != expected:
                raise ContractFailure('frozen model source changed: '+name)
        for name, expected in c['vendor_sources'].items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != expected:
                raise ContractFailure('frozen vendor source changed: '+name)
        for name in ('verified_backup_point_v1.py', 'verified_decay_v1.py', 'verified_battery_v1.py'):
            archived = self.path/'inputs'/name
            if (not archived.exists() or hashlib.sha256((HERE/name).read_bytes()).digest()
                    != hashlib.sha256(archived.read_bytes()).digest()):
                raise ContractFailure('executing verified primitive differs from proof: '+name)
        point_extension = json.loads((HERE/'verified_backup_point_v2_compatibility.json').read_text())
        if (not point_extension['all_module_AST_except_I_atan_identical']
                or not point_extension['existing_RPI_numeric_command_error_contract_unchanged']):
            raise ContractFailure('versioned point-kernel compatibility proof unavailable')
        for name, expected in point_extension['source_sha256'].items():
            if hashlib.sha256((HERE/name).read_bytes()).hexdigest() != expected:
                raise ContractFailure('versioned point-kernel source changed: '+name)
        norm_path = HERE/'verified_norm_backend_v1'
        norm_manifest = json.loads((norm_path/'manifest.json').read_text())
        if (not norm_manifest['norm3_norm4_error_budget_compatibility_proved']
                or norm_manifest['opaque_norm_sqrt_primitive_contract_required']):
            raise ContractFailure('explicit norm/sqrt error contract unavailable')
        for name, expected in norm_manifest['files_sha256'].items():
            if hashlib.sha256((norm_path/name).read_bytes()).hexdigest() != expected:
                raise ContractFailure('executing explicit norm proof changed: '+name)
        self.shapes, self.radii = {}, {}
        for name, group in c['groups'].items():
            scales = list(map(F, group['state_scales_exact_rational']))
            shape = [[F(x)*scales[i]*scales[j] for j, x in enumerate(row)]
                     for i, row in enumerate(group['Q_scaled_exact_rational'])]
            self.shapes[name] = shape
            self.radii[name] = [sqrt_bounds(shape[i][i])[1] for i in range(len(shape))]
            audit = json.loads((self.path/group['audit_path']).read_text())
            if not audit['passed'] or not audit['every_supplied_vertex_and_sign_checked']:
                raise ContractFailure('matrix audit unavailable: '+name)
        self.mass_upper = F(c['physical_parameter_intervals_outward']['mass'][1])
        self.mg_upper = self.mass_upper*F(c['physical_parameter_intervals_outward']['gravity'][1])
        self.authority_drop = F(c['proof_obligations']['verified_battery_decline']['per_period_decline_upper_rational'])
        self.period = F(c['period_s_rational'])
        self.horizon = c['horizon_periods']
        self.source_sha256 = hashlib.sha256((self.path/'certificate.json').read_bytes()).hexdigest()


class InitialAdmission:
    """Recompute the initial full-state set from a public-only history file."""
    @classmethod
    def load(cls, proof, public_history_path, case):
        path = Path(public_history_path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        key = proof.source_sha256, digest
        if key not in _ADMISSION_CACHE:
            from assemble_initial_contract_v1 import assemble
            _ADMISSION_CACHE[key] = assemble(proof, json.loads(path.read_text()))
        result = _ADMISSION_CACHE[key]
        row = next((r for r in result['cases'] if r['case'] == case), None)
        if row is None or not row['conditional_initial_admission_proved']:
            raise ContractFailure('public-history initial admission failed')
        admission = cls()
        admission.row, admission.public_history_sha256 = row, digest
        admission.mathematical_proof_sha256 = proof.source_sha256
        return admission


class ReanchoringShield:
    """Issue the verified fixed-anchor backup; admit a new anchor only by sets.

    A rejected proposal preserves the previous reserve. Fresh feedback and the
    enforced backend/service leases remain required for cached backup execution.
    The initial admission is an external, SHA-bound public-history proof.
    """
    def __init__(self, proof, known_free_map, *, hold_thrust, trim_force_interval,
                 motor_residual_interval, issued_queue, deadline_tick,
                 service_lease_expires_tick, initial_energy_upper_J,
                 arithmetic_backend_attestation, initial_admission):
        if not isinstance(proof, ProofBundle):
            raise TypeError('validated ProofBundle required')
        if (type(deadline_tick) is not int or not 1 <= deadline_tick <= proof.horizon
                or type(service_lease_expires_tick) is not int
                or service_lease_expires_tick < deadline_tick):
            raise ContractFailure('guaranteed fresh service must cover the complete backup horizon')
        if not 0 <= rational(initial_energy_upper_J) <= 8000:
            raise ContractFailure('public energy budget outside verified horizon')
        required = ('explicit_norm_sqrt_runtime_verified', 'verified_motor_decay',
                    'verified_power_and_battery', 'guarded_IEEE_basic_arithmetic',
                    'same_full_rigid_body_RK4_model', 'no_GPU_or_compiled_path',
                    'exact_current_orientation_feedback')
        if any(arithmetic_backend_attestation.get(name) is not True for name in required):
            raise ContractFailure('enforced verified arithmetic backend required')
        self.proof, self.map = proof, known_free_map
        self.hold = rational(hold_thrust)
        self.trim = tuple(map(rational, trim_force_interval))
        self.motor = tuple(map(rational, motor_residual_interval))
        self.queue = [tuple(map(rational, word)) for word in issued_queue]
        if (not isinstance(initial_admission, InitialAdmission)
                or initial_admission.mathematical_proof_sha256 != proof.source_sha256):
            raise ContractFailure('recomputed public-history admission required')
        a = initial_admission.row
        if (self.hold != rational(a['hold_thrust'])
                or self.trim != tuple(map(rational, a['trim_force_interval']))
                or self.motor != tuple(map(rational, a['motor_residual_interval']))
                or self.queue != [tuple(map(rational, word)) for word in a['issued_queue']]
                or rational(initial_energy_upper_J) < rational(a['public_initial_energy_upper_J'])):
            raise ContractFailure('initial supplied family differs from its public-history proof')
        self.initial_packet = a['public_final_packet']
        self.initial_admission = initial_admission
        if (len(self.trim) != 2 or self.trim[0] > self.trim[1]
                or len(self.motor) != 2 or self.motor[0] > self.motor[1]
                or len(self.queue) != 2 or any(len(word) != 4 for word in self.queue)):
            raise ContractFailure('ordered initial residuals and two public commands required')
        def stored32(value):
            try:
                represented = struct.unpack('>f', struct.pack('>f', float(value)))[0]
                return math.isfinite(represented) and rational(represented) == value
            except (OverflowError, ValueError):
                return False
        if not stored32(self.hold) or any(not stored32(value) for word in self.queue for value in word):
            raise ContractFailure('hold and issued queue must be exact stored binary32 words')
        self.deadline_tick = deadline_tick
        self.service_expiry = service_lease_expires_tick
        self.tick = 0
        self.anchor = None
        self.reserve = None
        self.pending = None
        self.blocked = False
        self.attestation = dict(arithmetic_backend_attestation)
        exact_deadline = deadline_tick*proof.period
        end_time = float(exact_deadline)
        if rational(end_time) < exact_deadline:
            end_time = math.nextafter(end_time, math.inf)
        # Retain the individual static witnesses through the backup deadline.
        # A planner forgetting a voxel does not erase its existing reservation.
        self._free_through_deadline = tuple(cell for cell in known_free_map.free
            if known_free_map.known_free(cell, time_s=0.)
            and known_free_map.known_free(cell, time_s=end_time))

    def _packet(self, packet, tick):
        if self.blocked or tick != self.tick or self.pending is not None:
            raise ContractFailure('blocked, duplicate or out-of-order command transaction')
        if not tick < self.deadline_tick <= self.service_expiry:
            raise ContractFailure('backup/service deadline reached')
        now = float(tick*self.proof.period)
        status = validate_contract(packet.contract, time_s=now)
        if (not status.conditional_input_valid or packet.contract.capture_time_s != now
                or rational(packet.contract.expires_s) < self.deadline_tick*self.proof.period):
            raise ContractFailure('fresh hard packet/complete backup lease unavailable')
        if (rational(packet.contract.position_error_m) > F(1, 200)
                or rational(packet.contract.velocity_error_mps) > F(1, 100)
                or rational(packet.angular_velocity_error_radps) > F(1, 1000)):
            raise ContractFailure('packet error exceeds the proved feedback domain')
        if getattr(packet, 'orientation_contract', None) != 'exact_current_raw_quaternion':
            raise ContractFailure('exact same-epoch orientation contract required')
        if tick == 0 and any(tuple(getattr(packet, name)) != tuple(self.initial_packet[name])
                             for name in ('p', 'v', 'q', 'w')):
            raise ContractFailure('initial packet must continue the admitted public history at the rebased epoch')
        return now

    def _membership(self, packet, anchor):
        return current_terminal_membership(
            p=packet.p, v=packet.v, q=packet.q, w=packet.w, anchor=anchor,
            hold_thrust=self.hold, issued_queue=self.queue,
            trim_force_interval=self.trim, motor_residual_interval=self.motor,
            shapes=self.proof.shapes)

    def _geometry(self, anchor):
        # d = Fhold-mg/b decreases as the battery depletes. Reserve the whole
        # remaining d family and C, including 5mm intermediate movement and
        # a conservative 171mm body radius covering the Float32 .17m endpoint.
        remaining = self.deadline_tick-self.tick
        drift = self.proof.mg_upper*self.proof.authority_drop*remaining/F(19, 20)**2
        d_lower, d_upper = self.trim[0]-drift, self.trim[1]
        rx = self.proof.radii['planar_x'][0]
        rz = self.proof.radii['vertical'][0]
        lower = [anchor[0]-rx, anchor[1]-rx, anchor[2]+d_lower/F(3, 4)-rz]
        upper = [anchor[0]+rx, anchor[1]+rx, anchor[2]+d_upper/F(3, 4)+rz]
        if any(lo-F(1, 200) < -16 or hi+F(1, 200) > 16 for lo, hi in zip(lower, upper)):
            return {'reserved': False, 'reason': 'numerical_world_position_domain'}
        result = free_grid_reserve(lower, upper, origin=self.map.origin,
            shape=self.map.shape, resolution=self.map.resolution, free_cells=self._free_through_deadline,
            body_radius=F(171, 1000), interstep_reserve=F(1, 200))
        return result

    def decide(self, packet, *, tick, proposed_anchor):
        now = self._packet(packet, tick)
        candidate = tuple(map(rational, proposed_anchor))
        if len(candidate) != 3:
            raise ContractFailure('three public anchor coordinates required')
        membership = self._membership(packet, candidate)
        geometry = self._geometry(candidate) if membership['current_set_included'] else None
        accepted = membership['current_set_included'] and geometry['reserved']
        reason = ('accepted_new_anchor' if accepted else
                  'proposal_outside_terminal_set' if geometry is None else 'proposal_reserve_not_known_free')
        if accepted:
            selected, selected_reserve = candidate, geometry
        elif self.anchor is not None:
            selected, selected_reserve = self.anchor, self.reserve
        else:
            return {'conditional_motion_certificate': False, 'command': None,
                    'reason': reason, 'proposal_accepted': False, 'tick': tick}
        # The fixed backup is valid under the complete initial C family; a
        # later, looser posterior need not fit C again to retain that proof.
        point = backup_point(packet.p, packet.v, packet.q, list(map(float, selected)), float(self.hold))
        # Preserve represented controller constants. If converting a rational
        # anchor changes it, use the represented value for membership/geometry.
        represented = tuple(rational(float(x)) for x in selected)
        if represented != selected:
            raise ContractFailure('anchor must be an exactly represented binary64 command constant')
        if accepted:
            self.anchor, self.reserve = selected, selected_reserve
        self.pending = tuple(map(rational, point['command']))
        return {'conditional_motion_certificate': True,
                'certificate_scope': 'bounded-service verified-CPU finite-horizon simulator',
                'command': point['command'], 'reason': reason,
                'proposal_accepted': bool(accepted), 'cached_backup_used': not accepted,
                'anchor': tuple(map(float, selected)), 'tick': tick,
                'backup_deadline_s': float(self.deadline_tick*self.proof.period),
                'hard_service_lease_expires_s': float(self.service_expiry*self.proof.period),
                'membership_maxima': {name: float(row['maximum']) for name, row in membership['channels'].items()},
                'reserved_cell_count': len(selected_reserve['intersecting_cells']),
                'proof_sha256': self.proof.source_sha256,
                'command_point_error_verified': point['command_roundoff_enclosed'],
                'default_Gaussian_runtime_certified': False,
                'physical_hardware_certified': False}

    def acknowledge_issued(self, command):
        """Advance the public queue only after the exact stored word was issued."""
        word = tuple(map(rational, command))
        if self.pending is None or word != self.pending:
            self.blocked = True
            raise ContractFailure('transport word mismatch or missing command transaction')
        propagated = propagate_public_residuals(
            motor_residual_interval=self.motor, trim_force_interval=self.trim,
            decay_interval=(F(51, 100), F(79, 100)), battery_interval=(F(19, 20), F(1)),
            hold_thrust=self.hold, issued_commands=[word, *self.queue],
            authority_drop_upper=self.proof.authority_drop, numerical_error=F(7, 1000000),
            mass_gravity_upper=self.proof.mg_upper)
        # Outward dyadic endpoints prevent unbounded rational denominator
        # growth in a long command history; this enlarges the belief, never
        # the physical/numerical disturbance allowed by the RPI theorem.
        scale = 2**48
        lower, upper = propagated['motor_residual_interval']
        self.motor = (F((lower*scale).numerator//(lower*scale).denominator, scale),
                      F(-((-upper*scale).numerator//(-upper*scale).denominator), scale))
        self.trim = propagated['trim_force_interval']
        self.queue = [word, self.queue[0]]
        self.pending = None
        self.tick += 1


__all__ = ['ContractFailure', 'ProofBundle', 'InitialAdmission', 'ReanchoringShield']
