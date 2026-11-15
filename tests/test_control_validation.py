"""Seeded full rigid body regression, independent of learned network weights."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from ares.contracts import load_json, validate_scene
from ares.control_validation import development_room, make_cases, run_control_validation, _terminal_metrics

PROFILE = Path(__file__).resolve().parents[1]/'ares/data/configs/drone_sim_v2.json'


class ControlValidationTests(unittest.TestCase):
    def test_cases_are_predeclared_replayable_and_cover_endpoints(self):
        profile = load_json(PROFILE)
        cases = make_cases(profile,64,51)
        self.assertEqual(cases,make_cases(profile,64,51))
        self.assertEqual({c['family'] for c in cases[:8]},
                         {'hover','low_speed_hold','vertical_up','vertical_down',
                          'horizontal_brake','three_dimensional_brake','occupied_wall','unknown_slab'})
        for key in ('mass_kg','gravity_mps2','motor_lag_s','inertia_kg_m2'):
            lo = cases[0]['profile']['parameters'][key]['nominal']
            hi = cases[8]['profile']['parameters'][key]['nominal']
            bounds = profile['parameters'][key]['range']
            self.assertEqual(lo,[b[0] for b in bounds] if isinstance(lo,list) else bounds[0])
            self.assertEqual(hi,[b[1] for b in bounds] if isinstance(hi,list) else bounds[1])
        self.assertEqual({c['declared_invalid'] for c in cases if c['declared_invalid']},
                         {'invalid_initial_clearance','initial_overspeed','braking_out_of_space'})

    def test_tilt_stream_preserves_existing_zero_cases_and_other_inputs(self):
        profile = load_json(PROFILE)
        zero = make_cases(profile,32,51)
        self.assertEqual(zero,make_cases(profile,32,51,initial_tilt_deg=0.))
        tilted = make_cases(profile,32,51,initial_tilt_deg=10.)
        self.assertEqual(tilted,make_cases(profile,32,51,initial_tilt_deg=10.))
        for original, changed in zip(zero,tilted):
            for key in ('profile','position','velocity','seed','declared_invalid','family'):
                self.assertEqual(original[key],changed[key])
            self.assertGreater(changed['initial_tilt_deg'],9.9)
            self.assertLessEqual(changed['initial_tilt_deg'],10.)
            self.assertAlmostEqual(sum(x*x for x in changed['imposed_quaternion_xyzw']),1.)
        for invalid in (-1.,20.01,float('nan'),True):
            with self.assertRaises(ValueError):
                make_cases(profile,initial_tilt_deg=invalid)

    def test_tilted_cpu_coupled_dynamics_remain_finite(self):
        report = run_control_validation(PROFILE,count=8,initial_tilt_deg=10.,duration_s=.05,
                                        settle_s=.02,include_nominal=False)
        self.assertEqual(len(report['cases']),8)
        self.assertTrue(all(r['initial_tilt_deg'] > 0 for r in report['cases']))
        self.assertTrue(report['cases'][7]['coupled_shield'])
        self.assertIn('vendor/diffaero/math.py',report['source_hashes'])
        self.assertTrue(all(torch.isfinite(torch.tensor(v)).all() for v in report['timeseries'].values()))

    def test_scene_is_development_only_and_unknown_is_not_physical_wall(self):
        room, observed = development_room(unknown_slab=True)
        validate_scene(room)
        self.assertTrue((observed[45,:,:] == 0).all())
        self.assertNotIn([45,30,30],room['grid']['occupied'])
        self.assertNotIn('60000',room['scene_id'])

    def test_low_mass_hover_regression_and_json_cpu_lists(self):
        with tempfile.TemporaryDirectory() as folder:
            report = run_control_validation(PROFILE,count=1,seed=7,duration_s=4.,settle_s=2.,output_dir=folder)
            repaired, nominal = report['cases']
            self.assertLess(repaired['metrics']['final_speed_mps'],.05)
            self.assertGreater(nominal['metrics']['final_speed_mps'],.2)
            self.assertLess(repaired['metrics']['drift_m'],nominal['metrics']['drift_m']*.2)
            self.assertTrue(report['timeseries']['repaired_0_acquisition'])
            self.assertIsInstance(report['timeseries']['repaired_0'][0],list)
            saved = load_json(Path(folder)/'control_validation.json')
            self.assertEqual(saved['seed'],7)
            self.assertFalse(saved['cases'][0]['shield_certified'])
            self.assertTrue(saved['source_hashes_stable'])
            self.assertEqual(saved['summary']['repaired']['valid_count'],1)
            self.assertTrue(saved['summary']['repaired']['valid_gate_passed'])
            self.assertTrue(saved['empirical_bounds']['finite_empirical'])
            self.assertFalse(saved['empirical_bounds']['universal_certification'])
            self.assertEqual(saved['empirical_bounds']['max_terminal_tail_radius_m']['case_count'],1)
            self.assertIn('flight_control.py',saved['source_hashes'])

    def test_terminal_tail_keeps_corrective_motion_and_fixed_cutoffs(self):
        series = torch.zeros(60,21)
        series[:,3] = .8
        series[:40,9] = -1.
        series[40:,3] = .3
        series[40:,0] = torch.linspace(0,.23,20)
        metrics = _terminal_metrics(series)
        self.assertAlmostEqual(metrics['terminal_tail_radius_m'],.23,places=6)
        self.assertAlmostEqual(metrics['terminal_entry_time_s'],.41,places=6)
        self.assertEqual(metrics['terminal_tail_sample_count'],20)
        self.assertEqual(metrics['fixed_latency_active_braking_sample_count'],1)
        self.assertEqual(metrics['fixed_latency_min_active_braking_mps2'],1.)

    def test_source_mutation_refuses_report(self):
        with patch('ares.control_validation._hashes',side_effect=[{'source':'before'},{'source':'after'}]):
            with self.assertRaisesRegex(RuntimeError,'changed during run'):
                run_control_validation(PROFILE,count=1,duration_s=.01,settle_s=0.,include_nominal=False)

    def test_invalid_arguments_rejected(self):
        for options in ({'count':0},{'batch_size':0},{'duration_s':0},{'settle_s':-1},
                        {'noise_bounds':{'velocity_mps':-1}},{'noise_bounds':{'position_m':.03}},{'command_delay_s':float('nan')}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                run_control_validation(PROFILE,**options)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_cuda_small_batch_real_plant(self):
        report = run_control_validation(PROFILE,device='cuda',count=8,duration_s=.05,settle_s=.02,
                                        include_nominal=False)
        self.assertEqual(len(report['cases']),8)
        self.assertTrue(all(torch.isfinite(torch.tensor(v)).all() for v in report['timeseries'].values()))


if __name__ == '__main__': unittest.main()
