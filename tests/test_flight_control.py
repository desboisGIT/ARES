"""Observed-state trim, deterministic conversion, and actual rigid-body stops."""
from dataclasses import fields, replace
import json
import math
from pathlib import Path
import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'optional torch dependency unavailable')
class FlightControlTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def observation(self, batch=2, device='cpu'):
        return {'p': torch.zeros(batch, 3, device=device), 'v': torch.zeros(batch, 3, device=device),
                'q': torch.tensor([[0., 0., 0., 1.]], device=device).expand(batch, -1),
                'gaze': torch.tensor([[2., 0., 0.]], device=device).expand(batch, -1)}

    def test_conversion_is_pure_bounded_and_has_finite_gradients(self):
        from ares.flight_control import acceleration_to_action, initialize_memory
        obs = self.observation()
        memory = initialize_memory(obs)
        obs.update(memory.observation_state())
        obs['p'] = obs['p'].requires_grad_()
        obs['q'] = obs['q'].clone().requires_grad_()
        obs['gaze'] = torch.tensor([[0., 0., 2.], [0., 0., -2.]])
        a = torch.tensor([[1., 0., 0.], [0., 1., 0.]], requires_grad=True)
        first = acceleration_to_action(obs, a)
        torch.testing.assert_close(first, acceleration_to_action(obs, a))
        first.square().sum().backward()
        for value in (a, obs['p'], obs['q']):
            self.assertTrue(bool(torch.isfinite(value.grad).all()))
        self.assertGreater(float(first[0, 2].detach()), 0)
        self.assertLess(float(first[1, 1].detach()), 0)
        huge = acceleration_to_action(obs, a.detach() * 1e6)
        self.assertTrue(bool((huge[:, 0] <= 24.5).all()))
        self.assertTrue(bool((huge[:, 0] >= 0).all()))
        self.assertTrue(bool((huge[:, 1:].abs() <= 6).all()))
        torch.testing.assert_close(memory.bias, torch.zeros_like(memory.bias))

    def test_world_heading_cap_preserves_tilt_feedback_and_finite_gradients(self):
        from ares.flight_control import ControllerConfig, acceleration_to_action
        from ares.flight_policy import quaternion_rotate
        obs = self.observation()
        obs['p'] = obs['p'].requires_grad_()
        obs['q'] = torch.tensor([[0., math.sin(.1), 0., math.cos(.1)],
                                 [math.sin(.1), 0., 0., math.cos(.1)]], requires_grad=True)
        obs['gaze'] = torch.tensor([[-2., 0., 0.], [-2., 0., 0.]])
        desired = torch.tensor([[0., 1.5, 0.], [1.5, 0., 0.]], requires_grad=True)
        config = ControllerConfig()
        action = acceleration_to_action(obs, desired, config)
        tilt_obs = dict(obs); tilt_obs['gaze'] = obs['p']
        tilt = acceleration_to_action(tilt_obs, desired, config)
        heading_world = quaternion_rotate(obs['q'], action[:, 1:]-tilt[:, 1:])
        torch.testing.assert_close(heading_world, torch.tensor([[0., 0., 1.], [0., 0., 1.]]),
                                   atol=1e-6, rtol=1e-6)
        self.assertTrue(bool((tilt[:, 1:].norm(dim=-1) > .1).all()))
        action.square().sum().backward()
        for tensor in (obs['p'], obs['q'], desired):
            self.assertTrue(bool(torch.isfinite(tensor.grad).all()))
        for invalid in (0., -.1, math.inf, math.nan):
            with self.assertRaises(ValueError):
                ControllerConfig(max_yaw_rate=invalid)

    def test_observer_snapshot_and_masked_reset_are_isolated(self):
        from ares.flight_control import initialize_memory, update_memory
        obs = self.observation()
        memory = initialize_memory(obs)
        obs['v'] = torch.ones(2, 3)
        action = torch.tensor([[24.5, 0., 0., 0.], [0., 0., 0., 0.]])
        for _ in range(200):
            memory = update_memory(memory, obs, action, .02, stop_requested=True)
        self.assertTrue(bool((memory.bias.norm(dim=-1) <= 4.00001).all()))
        snapshot = memory.observation_state()
        snapshot['controller_bias'].zero_()
        self.assertGreater(float(memory.bias.abs().sum()), 0)
        fresh = initialize_memory(obs)
        reset = update_memory(memory, obs, action, 0, reset_mask=torch.tensor([True, False]))
        for field in fields(memory):
            torch.testing.assert_close(getattr(reset, field.name)[0], getattr(fresh, field.name)[0])
            torch.testing.assert_close(getattr(reset, field.name)[1], getattr(memory, field.name)[1])
        self.assertFalse(reset.bias.requires_grad)

    def test_stop_has_low_speed_capture_hysteresis_and_no_unit_brake(self):
        from ares.flight_control import initialize_memory, update_memory, stop_acceleration
        obs = self.observation()
        memory = initialize_memory(obs)
        action = torch.tensor([[4.903325, 0., 0., 0.]]).expand(2, -1)
        memory = update_memory(memory, obs, action, .02, stop_requested=True)
        self.assertTrue(bool(memory.holding.all()))
        obs.update(memory.observation_state())
        obs['controller_velocity'] = torch.tensor([[.001, 0., 0.], [-.001, 0., 0.]])
        a = stop_acceleration(obs, 1.)
        self.assertLess(float(a.norm(dim=-1).max()), .003)
        self.assertTrue(bool((a[:, 0] * obs['controller_velocity'][:, 0] < 0).all()))
        obs['controller_velocity'] = torch.tensor([[3., 0., 0.], [0., 0., -3.]])
        a = stop_acceleration(obs, 1.)
        torch.testing.assert_close(a.norm(dim=-1), torch.ones(2))
        released = update_memory(memory, obs, action, .02, stop_requested=False)
        self.assertFalse(bool(released.holding.any()))

    def test_corner_hover_vertical_tracking_and_coupled_braking_in_diffaero_v2(self):
        from ares.flight_control import (acceleration_to_action, initialize_memory,
                                         update_memory, stop_acceleration)
        from ares.diffaero_adapter import DiffAeroAdapter
        from ares.flight_randomization import sample_flight_profile
        profile = json.loads((Path(__file__).parents[1] / 'ares/data/configs/drone_sim_v2.json').read_text())
        profiles = [sample_flight_profile(profile, 0, mode) for mode in ('low', 'high')]
        # Four coupled corners include heavy/low-drag/slow actuators and the
        # light/high-drag/fast opposite, independently of controller inputs.
        for mass, drag, lag in ((.55, .05, .04), (.45, .2, .015)):
            item = sample_flight_profile(profile, 0, 'nominal')
            for name, value in (('mass_kg', mass), ('linear_drag', drag), ('motor_lag_s', lag)):
                item['parameters'][name]['nominal'] = value
            profiles.append(item)
        adapter = DiffAeroAdapter(profiles, dt=.02)
        def observe():
            return {'p': adapter.position.detach().clone(), 'v': adapter.velocity.detach().clone(),
                    'q': adapter.quaternion.detach().clone(),
                    'gaze': adapter.position.detach() + torch.tensor([2., 0., 0.])}
        obs = observe(); memory = initialize_memory(obs)
        for _ in range(200):
            obs.update(memory.observation_state())
            action = acceleration_to_action(obs, stop_acceleration(obs))
            adapter.step(action)
            obs = observe()
            memory = update_memory(memory, obs, action, .02, stop_requested=True)
        self.assertLess(float(adapter.velocity.norm(dim=-1).max()), .06)
        self.assertLess(float(adapter.position.norm(dim=-1).max()), .25)
        # A -1 vertical command after observed trim must really brake even at
        # the light/low-gravity corner where nominal thrust nearly equals weight.
        start_velocity = adapter.velocity.detach().clone()
        for _ in range(25):
            obs.update(memory.observation_state())
            action = acceleration_to_action(obs, torch.tensor([[0., 0., -1.]]).expand(4, -1))
            adapter.step(action); obs = observe()
            memory = update_memory(memory, obs, action, .02, stop_requested=False)
        measured = (adapter.velocity[:, 2] - start_velocity[:, 2]) / .5
        self.assertTrue(bool((measured < -.75).all()), measured)
        # Inject an observable coupled velocity, retain only learned public trim.
        adapter.reset(position=[0., 0., 0.], velocity=[1.1, -.7, .8])
        obs = observe()
        memory = replace(memory, velocity=obs['v'].clone(), holding=torch.zeros(4, dtype=torch.bool),
                         hold_anchor=obs['p'].clone(), previous_quaternion=obs['q'].clone())
        history = []
        for _ in range(300):
            obs.update(memory.observation_state())
            action = acceleration_to_action(obs, stop_acceleration(obs, 2.))
            adapter.step(action); obs = observe()
            memory = update_memory(memory, obs, action, .02, stop_requested=True)
            history.append(adapter.velocity.detach().norm(dim=-1))
        self.assertLess(float(torch.stack(history)[-50:].max()), .06)
        self.assertTrue(bool(memory.holding.all()))
        self.assertLess(float((adapter.position - memory.hold_anchor).norm(dim=-1).max()), .10)

    def test_development_bandwidth_reduces_cold_sag_with_bounded_noise_and_delay(self):
        from ares.flight_control import (ControllerConfig, acceleration_to_action,
                                         initialize_memory, update_memory, stop_acceleration)
        from ares.diffaero_adapter import DiffAeroAdapter
        profile = json.loads((Path(__file__).parents[1] / 'ares/data/configs/drone_sim_v2.json').read_text())
        profiles = []
        for mass in (.45, .55):
            for gravity in (9.78, 9.83):
                item = {name: parameter['nominal'] for name, parameter in profile['parameters'].items()}
                item.update(mass_kg=mass, gravity_mps2=gravity, motor_lag_s=.04,
                            linear_drag=.05, inertia_kg_m2=[.0022, .0022, .0033])
                profiles.append(item)
        def acquisition(config, dt):
            adapter = DiffAeroAdapter(profiles, dt=dt)
            generator = torch.Generator().manual_seed(1711)
            def noise(bound):
                vector = torch.randn(4, 3, generator=generator)
                vector /= vector.norm(dim=-1, keepdim=True).clamp_min(1e-9)
                return vector * torch.rand(4, 1, generator=generator).pow(1/3) * bound
            def observe():
                return {'p': adapter.position.clone()-adapter.velocity*(.02+1/60)+noise(.01),
                        'v': adapter.velocity.clone()+noise(.025), 'q': adapter.quaternion.clone(),
                        'gaze': adapter.position.clone()+torch.tensor([2., 0., 0.])}
            obs = observe()
            memory = initialize_memory(obs, config)
            # Public nominal initial command; never initialize from sampled weight.
            commands = [torch.tensor([4.903325, 0., 0., 0.]).expand(4, -1) for _ in range(5)]
            sag = 0.
            final_speed = []
            for step in range(round(4/dt)):
                obs.update(memory.observation_state())
                action = acceleration_to_action(obs, stop_acceleration(obs, 2., config), config)
                commands.insert(0, action); del commands[5:]
                delay = .02/dt
                low, high = math.floor(delay), math.ceil(delay)
                fraction = delay-low
                adapter.step(commands[low]*(1-fraction)+commands[high]*fraction)
                obs = observe()
                memory = update_memory(memory, obs, action, dt, config, stop_requested=True)
                sag = max(sag, float((-adapter.position[:, 2]).max()))
                if step*dt >= 3:
                    final_speed.append(adapter.velocity.norm(dim=-1))
            return sag, float(torch.stack(final_speed).max())
        with torch.no_grad():
            for dt in (.01, 1/60):
                with self.subTest(dt=dt):
                    candidate = ControllerConfig()
                    baseline = replace(candidate, observer_bandwidth=4.)
                    candidate_sag, final_speed = acquisition(candidate, dt)
                    baseline_sag, _ = acquisition(baseline, dt)
                    self.assertLess(candidate_sag, .12)
                    self.assertGreater(baseline_sag, .15)
                    self.assertLess(candidate_sag, .65*baseline_sag)
                    self.assertLess(final_speed, .06)

    def test_yaw_cap_reduces_velocity_feedback_transients_in_actual_rigid_body(self):
        from ares.flight_control import (ControllerConfig, acceleration_to_action,
                                         initialize_memory, update_memory)
        from ares.diffaero_adapter import DiffAeroAdapter
        from ares.flight_policy import quaternion_rotate
        profile = json.loads((Path(__file__).parents[1] / 'ares/data/configs/drone_sim_v2.json').read_text())
        profiles = []
        for mass in (.45, .55):
            for inertia in ([.0018, .0018, .0027], [.0022, .0022, .0033]):
                item = {key: value['nominal'] for key, value in profile['parameters'].items()}
                item.update(mass_kg=mass, inertia_kg_m2=inertia, motor_lag_s=.04,
                            gravity_mps2=9.83, linear_drag=.05)
                profiles.append(item)
        @torch.no_grad()
        def trajectory(config):
            plant = DiffAeroAdapter(profiles, dt=1/60)
            plant.reset(quaternion=[0., math.sin(.15/2), 0., math.cos(.15/2)])
            def observe():
                return {'p': plant.position.clone(), 'v': plant.velocity.clone(),
                        'q': plant.quaternion.clone(),
                        'gaze': plant.position.clone()+torch.tensor([-2., 0., 0.])}
            obs = observe(); memory = initialize_memory(obs, config)
            queue = [torch.tensor([4.903325, 0., 0., 0.]).expand(4, -1) for _ in range(3)]
            velocities, errors = [], []
            for _ in range(240):
                obs.update(memory.observation_state())
                desired = 2.5*(torch.tensor([[0., 1., 0.]]).expand(4, -1)-obs['controller_velocity'])
                action = acceleration_to_action(obs, desired, config)
                self.assertTrue(bool(torch.isfinite(action).all()))
                self.assertTrue(bool(((action[:, 0] >= 0) & (action[:, 0] <= config.max_thrust)).all()))
                self.assertTrue(bool((action[:, 1:].abs() <= config.max_body_rate).all()))
                queue.insert(0, action); del queue[3:]
                plant.step(.8*queue[1]+.2*queue[2])  # Public test transport delay .02 s.
                obs = observe()
                memory = update_memory(memory, obs, action, 1/60, config)
                body_z = quaternion_rotate(plant.quaternion, torch.tensor([[0., 0., 1.]]).expand(4, -1))
                target = desired+torch.tensor([0., 0., config.gravity])-memory.bias
                target = target/target.norm(dim=-1, keepdim=True)
                errors.append(torch.linalg.cross(body_z, target, dim=-1).norm(dim=-1))
                velocities.append(plant.velocity.clone())
            series = torch.stack(velocities)
            return series[:, :, 0].abs().max(), series[:, :, 2].abs().max(), torch.stack(errors)
        capped = ControllerConfig()
        # A large finite cap reproduces the previous adaptive heading law for
        # this nominal gain; all other public controller constants are identical.
        old = replace(capped, max_yaw_rate=100.)
        x_speed, z_speed, error = trajectory(capped)
        old_x, old_z, old_error = trajectory(old)
        self.assertLess(float(x_speed), .25)
        self.assertLess(float(z_speed), .25)
        self.assertLess(float(error.max()), .35)
        self.assertLess(float(error[-60:].max()), .01)
        self.assertLess(float(x_speed), .65*float(old_x))
        self.assertLess(float(z_speed), .65*float(old_z))
        self.assertLess(float(error.max()), .75*float(old_error.max()))

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_conversion_observer_and_stop(self):
        from ares.flight_control import (initialize_memory, update_memory,
                                         acceleration_to_action, stop_acceleration)
        obs = self.observation(64, 'cuda'); memory = initialize_memory(obs)
        obs.update(memory.observation_state())
        action = acceleration_to_action(obs, stop_acceleration(obs))
        memory = update_memory(memory, obs, action, 1 / 60, stop_requested=True)
        self.assertEqual(memory.bias.device.type, 'cuda')
        self.assertTrue(bool(torch.isfinite(action).all()))


if __name__ == '__main__':
    unittest.main()
