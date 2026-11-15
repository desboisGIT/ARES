"""Contract/reservation regression tests; coupled dynamics are tested separately."""
from dataclasses import replace
from fractions import Fraction as F
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/'bounded_service_v1'))
from ares.safety_contract import MeasurementContract
from bounded_service import StatePacket, WitnessMap
from reanchoring_shield_v1 import ContractFailure, InitialAdmission, ProofBundle, ReanchoringShield


class ReanchoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.proof = ProofBundle(HERE.parents[1]/'phase_4_shield_artifacts/cpu_rpi_certificate_v1')
        cls.admission = InitialAdmission.load(cls.proof, HERE/'public_trim_history_v1.json', 0)
        cls.map = WitnessMap.from_dict(json.loads((HERE/'bounded_service_v1/known_free_map.json').read_text()), time_s=0.)

    def packet(self, tick=0, **contract_changes):
        row = self.admission.row
        c = dict(position_error_m=math.nextafter(.005, 0), velocity_error_mps=math.nextafter(.01, 0),
            world_ray_error=.001, depth_error_m=.01, capture_time_s=float(F(tick, 100)), expires_s=10.,
            supplier='bounded_SIL_unit_fixture', provenance='public_history_same_epoch',
            source_kind='hard_external', bound_basis='deterministic_guarantee')
        c.update(contract_changes)
        p = row['public_final_packet']
        return StatePacket(tuple(p['p']), tuple(p['v']), tuple(p['q']), MeasurementContract(**c), tuple(p['w']))

    def guard(self, **changes):
        row = self.admission.row
        fields = dict(hold_thrust=row['hold_thrust'], trim_force_interval=row['trim_force_interval'],
            motor_residual_interval=row['motor_residual_interval'], issued_queue=row['issued_queue'],
            deadline_tick=1000, service_lease_expires_tick=1000,
            initial_energy_upper_J=row['public_initial_energy_upper_J'], initial_admission=self.admission,
            arithmetic_backend_attestation={key: True for key in (
                'explicit_norm_sqrt_runtime_verified', 'verified_motor_decay', 'verified_power_and_battery',
                'guarded_IEEE_basic_arithmetic', 'same_full_rigid_body_RK4_model', 'no_GPU_or_compiled_path',
                'exact_current_orientation_feedback')})
        fields.update(changes)
        return ReanchoringShield(self.proof, self.map, **fields)

    def test_initial_admission_is_recomputed_and_nonzero(self):
        row = self.admission.row
        self.assertTrue(row['conditional_initial_admission_proved'])
        self.assertTrue(all(v['maximum'] < 1 for v in row['membership']['channels'].values()))
        self.assertGreater(row['motor_residual_interval'][1], 0)
        self.assertGreater(row['trim_force_interval'][1], 0)
        self.assertGreater(row['public_initial_energy_upper_J'], 4000)
        with self.assertRaises(ContractFailure):
            self.guard(trim_force_interval=(0, 0))
        with self.assertRaises(ContractFailure):
            self.guard(initial_admission=None)

    def test_positive_certificate_and_exact_transport_transaction(self):
        guard = self.guard()
        result = guard.decide(self.packet(), tick=0, proposed_anchor=self.admission.row['anchor'])
        self.assertTrue(result['conditional_motion_certificate'])
        self.assertTrue(result['proposal_accepted'])
        self.assertGreater(result['reserved_cell_count'], 0)
        self.assertFalse(result['default_Gaussian_runtime_certified'])
        with self.assertRaises(ContractFailure):
            guard.decide(self.packet(), tick=0, proposed_anchor=self.admission.row['anchor'])
        guard.acknowledge_issued(result['command'])
        self.assertEqual(guard.tick, 1)
        with self.assertRaises(ContractFailure):
            guard.acknowledge_issued(result['command'])
        self.assertTrue(guard.blocked)

    def test_failed_proposal_preserves_cached_backup(self):
        guard = self.guard()
        anchor = self.admission.row['anchor']
        first = guard.decide(self.packet(), tick=0, proposed_anchor=anchor)
        guard.acknowledge_issued(first['command'])
        cached = guard.anchor, guard.reserve
        proposal = [anchor[0]+3, anchor[1], anchor[2]]
        result = guard.decide(self.packet(1), tick=1, proposed_anchor=proposal)
        self.assertTrue(result['conditional_motion_certificate'])
        self.assertTrue(result['cached_backup_used'])
        self.assertFalse(result['proposal_accepted'])
        self.assertEqual((guard.anchor, guard.reserve), cached)

    def test_stale_unbounded_or_oversized_packet_rejects(self):
        for changes in ({'capture_time_s':-.01}, {'expires_s':9.},
                        {'source_kind':'gaussian'}, {'bound_basis':'confidence_interval'},
                        {'position_error_m':.005}, {'velocity_error_mps':.01}):
            with self.subTest(changes=changes), self.assertRaises(ContractFailure):
                self.guard().decide(self.packet(**changes), tick=0,
                                    proposed_anchor=self.admission.row['anchor'])
        packet = self.packet()
        missing = SimpleNamespace(**{name: getattr(packet, name) for name in ('p','v','q','w','contract','angular_velocity_error_radps')})
        with self.assertRaises(ContractFailure):
            self.guard().decide(missing, tick=0, proposed_anchor=self.admission.row['anchor'])

    def test_lease_and_backend_obligations_are_required(self):
        for changes in ({'service_lease_expires_tick':999}, {'initial_energy_upper_J':8001},
                        {'arithmetic_backend_attestation':{}}, {'deadline_tick':1001}):
            with self.subTest(changes=changes), self.assertRaises(ContractFailure):
                self.guard(**changes)
        row = self.admission.row
        with self.assertRaises(ContractFailure):
            self.guard(hold_thrust=F.from_float(row['hold_thrust'])+F(1, 2**70))

    def test_world_domain_and_future_map_do_not_certify(self):
        guard = self.guard()
        self.assertFalse(guard._geometry((F(159,10),F(6),F(2)))['reserved'])
        # A map witness from a future capture cannot bootstrap a current lease.
        # This interface stand-in represents a noncausal map packet.
        future = SimpleNamespace(free=self.map.free, shape=self.map.shape, origin=self.map.origin,
            resolution=self.map.resolution,
            known_free=lambda cell, time_s: time_s >= 1.)
        old = self.map
        try:
            self.map = future
            result = self.guard().decide(self.packet(), tick=0, proposed_anchor=self.admission.row['anchor'])
            self.assertFalse(result['conditional_motion_certificate'])
        finally:
            self.map = old

    def test_nonrepresentable_deadline_rounds_outward(self):
        exact = F(3, 100)
        rounded = float(exact)
        self.assertLess(F.from_float(rounded), exact)
        expiring = SimpleNamespace(free=self.map.free, shape=self.map.shape, origin=self.map.origin,
            resolution=self.map.resolution, known_free=lambda cell, time_s: time_s <= rounded)
        old = self.map
        try:
            self.map = expiring
            result = self.guard(deadline_tick=3).decide(self.packet(), tick=0,
                proposed_anchor=self.admission.row['anchor'])
            self.assertFalse(result['conditional_motion_certificate'])
        finally:
            self.map = old


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ReanchoringTests))
    Path(__file__).with_suffix('.json').write_text(json.dumps({'tests_run':result.testsRun,
        'passed':result.wasSuccessful(),'scope':'consumer contract and transaction logic; no dynamics/timing claim'},indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
