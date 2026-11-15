"""Independent observation, randomization, task and rigid-body environment checks."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch

try:
    import numpy as np
    import torch
    import scipy
except ImportError:
    raise unittest.SkipTest('optional Phase 4 numpy/scipy/torch unavailable')

from ares.flight_env import FlightEnv, waypoint_tasks
from ares.flight_randomization import randomization_manifest
from ares.flight_guidance import guidance_velocity
from ares.worldgen import generate_scene
from tests.test_depth import room_scene


class FlightEnvTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.scene = room_scene()
        cls.profile = json.loads((Path(__file__).parents[1] / 'ares/data/configs/drone_sim_v2.json').read_text())

    def environment(self, **kwargs):
        return FlightEnv(self.scene, self.profile, batch_size=2, seed=38, **kwargs)

    def test_clearance_box_faces_corners_and_world_bounds_have_physical_gradients(self):
        env = self.environment(randomization='nominal')
        env.parameters['body_radius_m'] = torch.tensor([.15,.15])
        env._box_centers = torch.tensor([[1.3,1.3,1.3]])
        p = torch.tensor([[1.,1.3,1.3],[1.,1.,1.3]],requires_grad=True)
        clearance = env.clearance(p)
        torch.testing.assert_close(clearance,torch.tensor([.05,2**.5*.2-.15]),atol=1e-6,rtol=0)
        clearance.sum().backward()
        torch.testing.assert_close(p.grad,torch.tensor([[-1.,0.,0.],[-2**-.5,-2**-.5,0.]]),atol=1e-6,rtol=0)
        p = (env.origin+torch.tensor([[.20,1.,1.],[.20,1.,1.]])).requires_grad_()
        clearance = env.clearance(p)
        torch.testing.assert_close(clearance,torch.full((2,),.05),atol=1e-6,rtol=0)
        clearance.sum().backward()
        torch.testing.assert_close(p.grad,torch.tensor([[1.,0.,0.],[1.,0.,0.]]))

    def test_narrow_valid_route_has_feasible_loss_and_rewards_motion_near_handoff(self):
        # This development task has <.03m true clearance, despite being valid.
        env = FlightEnv(generate_scene(102,'apartment','train'),self.profile,
                        batch_size=1,seed=1741+2*10007+8)
        self.assertGreater(float(env.task_route_clearance_m[0]),0)
        self.assertLess(float(env.task_route_clearance_m[0]),.03)
        torch.testing.assert_close(env.obstacle_margin_m,.5*env.task_route_clearance_m)
        points = torch.tensor(env.tasks[0]['positions'])
        samples = torch.cat([a+(b-a)*torch.linspace(0,1,61)[:,None]
                             for a,b in zip(points[:-1],points[1:])])
        clearance = env.clearance(samples)
        self.assertGreaterEqual(float(clearance.min()),float(env.task_route_clearance_m[0])-1e-6)
        self.assertTrue(bool((clearance>env.obstacle_margin_m).all()))
        incoming = (points[1]-points[0]); incoming /= incoming.norm()
        position = (points[1]-.11*incoming)[None]
        velocity = guidance_velocity(position,env.targets[:,0],env.targets[:,1],env.speed_limit)
        state = env.adapter.state.detach().clone()
        state[:,:3] = position; state[:,7:10] = velocity
        env.adapter.state = state
        action = torch.zeros((1,4))
        moving = env.differentiable_loss(action,action)
        env.adapter.state = torch.cat((state[:,:7],torch.zeros((1,3)),state[:,10:]),-1)
        hovering = env.differentiable_loss(action,action)
        self.assertEqual(float(moving['obstacle']),0.)
        self.assertEqual(float(hovering['obstacle']),0.)
        self.assertLess(float(moving['total']),float(hovering['total']))

    def test_selected_reset_preserves_survivor_clearance_budget(self):
        env = self.environment()
        margin = env.obstacle_margin_m.clone()
        route = env.task_route_clearance_m.clone()
        env.reset_idx([0])
        torch.testing.assert_close(env.obstacle_margin_m[1],margin[1],atol=0,rtol=0)
        torch.testing.assert_close(env.task_route_clearance_m[1],route[1],atol=0,rtol=0)

    def test_observation_has_only_sensor_state_and_mission_inputs(self):
        env = self.environment(randomization='nominal')
        obs = env.observation()
        self.assertEqual(set(obs), {'depth', 'features', 'p', 'v', 'q', 'wp1', 'wp2', 'gaze', 'speedlimit',
                                         'previous_command', 'controller_bias', 'controller_velocity',
                                         'controller_hold_anchor', 'controller_holding', 'controller_stopping'})
        self.assertEqual(obs['depth'].shape, (2, 2, 24, 32))
        self.assertEqual(obs['features'].shape, (2, 43))
        self.assertTrue(all(bool(torch.isfinite(tensor).all()) for tensor in obs.values()))
        torch.testing.assert_close(obs['features'][:, 23:39], env._ring(obs['p'], obs['q']))
        # Initial delivery latency leaves depth unavailable rather than inventing returns.
        self.assertTrue(bool((obs['depth'][:, 1] == 0).all()))
        for forbidden in ('material_ids', 'occupied', 'distance_field', 'mass_kg', 'ground_truth'):
            self.assertNotIn(forbidden, obs)

    def test_delayed_map_capture_uses_public_estimate_while_renderer_uses_physical_pose(self):
        env = self.environment(randomization='nominal', dt=.02)
        env._depth_history.clear()
        env.parameters['compute_latency_s'] = torch.full((2,), .02)
        env.parameters['odometry_drift_per_m'] = torch.full((2,), .2)
        env.parameters['range_noise_std_m'] = torch.full((2,), .1)
        env.travel = torch.full((2,), 2.)
        env.drift_direction = torch.tensor([[1., 0., 0.]]).expand(2, -1)
        physical = env.adapter.position.detach().clone()

        def unit_noise(shape, mask=None):
            return torch.ones(shape, device=env.device)

        with patch.object(env, '_noise', side_effect=unit_noise), \
                patch.object(env.sensors, 'observe', wraps=env.sensors.observe) as render, \
                patch.object(env, '_integrate_map', wraps=env._integrate_map) as integrate:
            captured = env._observe()
            self.assertFalse(integrate.called)  # capture is still in transport
            torch.testing.assert_close(render.call_args.args[0], physical, atol=0, rtol=0)
            torch.testing.assert_close(env._depth_history[-1][2], captured['p'], atol=0, rtol=0)
            self.assertGreater(float((captured['p']-physical).abs().sum()), .5)
            # Delivery must retain capture-time estimated pose, even after the
            # current physical position and public drift have changed.
            env.adapter.state = torch.cat((env.adapter.position+torch.tensor([.3, 0., 0.]),
                                           env.adapter.state[:, 3:]), -1)
            env.travel = torch.full((2,), 4.)
            env.episode_time_s = torch.full((2,), .04)
            delivered = env._observe()
            self.assertEqual(integrate.call_count, 1)
            mapped_position = integrate.call_args.args[1]
            torch.testing.assert_close(mapped_position, captured['p'], atol=0, rtol=0)
            self.assertFalse(torch.equal(mapped_position, physical))
            self.assertFalse(torch.equal(mapped_position, delivered['p']))

    def test_ring_treats_unknown_as_blocked_and_uses_map_memory(self):
        env = self.environment(randomization='nominal')
        position = torch.tensor([[1.1, 1.1, 1.1]] * 2)
        quaternion = torch.tensor([[0., 0., 0., 1.]] * 2)
        env._known.zero_()
        torch.testing.assert_close(env._ring(position, quaternion), torch.zeros(2, 16))
        # Measured free cells establish range; hidden geometry has no direct role.
        env._known.fill_(1)
        measured = env._ring(position, quaternion)
        self.assertTrue(bool((measured > 0).all()))
        hidden = env.occupied_np.copy()
        env.occupied_np[:] = True
        torch.testing.assert_close(env._ring(position, quaternion), measured)
        env.occupied_np[:] = hidden
        # An observed obstacle ends the forward range even in otherwise known space.
        env._known[:, 7, 5, 5] = 2
        self.assertLess(float(env._ring(position, quaternion)[0, 0]), float(measured[0, 0]))

    def test_all_29_parameters_sampled_and_endpoint_audited(self):
        low = self.environment(randomization='low')
        high = self.environment(randomization='high')
        audit = randomization_manifest(low.sampled_profiles + high.sampled_profiles)
        self.assertEqual(len(audit['parameters']), 29)
        for record in audit['parameters'].values():
            for component in record['components']:
                self.assertTrue(component['within_bounds'])
                self.assertTrue(component['low_endpoint_seen'])
                self.assertTrue(component['high_endpoint_seen'])
        random_env = self.environment(randomization='random')
        self.assertNotEqual(random_env.sampled_profiles[0]['parameters']['mass_kg']['nominal'],
                            random_env.sampled_profiles[1]['parameters']['mass_kg']['nominal'])
        old = random_env.parameters['mass_kg'].clone()
        random_env.reset()
        self.assertFalse(torch.equal(old, random_env.parameters['mass_kg']))

    def test_two_leg_tasks_are_nontrivial_repeatable_and_clear(self):
        tasks = waypoint_tasks(self.scene, .25, 12, 47)
        self.assertEqual(tasks, waypoint_tasks(self.scene, .25, 12, 47))
        self.assertNotEqual(tasks, waypoint_tasks(self.scene, .25, 12, 48))
        occupied = set(map(tuple, self.scene['grid']['occupied']))
        origin = np.asarray(self.scene['grid']['origin'])
        resolution = self.scene['grid']['resolution']
        for task in tasks:
            self.assertTrue(all(distance >= .6 - 1e-6 for distance in task['leg_lengths_m']))
            for start, end in zip(task['positions'][:-1], task['positions'][1:]):
                # Dense independent center-line samples must never enter a solid cell.
                for point in np.linspace(start, end, 50):
                    self.assertNotIn(tuple(np.floor((point - origin) / resolution).astype(int)), occupied)

    def test_time_limit_and_terminal_state_freeze(self):
        env = self.environment(randomization='nominal', dt=.03, horizon=1)
        action = env.adapter.last_action.clone()
        _, _, done, info = env.step(action)
        self.assertTrue(bool(done.all()))
        self.assertAlmostEqual(env.time_s, .03)
        self.assertTrue(bool((info['terminated'] | info['truncated']).all()))
        terminal = env.adapter.state.clone()
        env.step(action + torch.tensor([[2., 1., 1., 1.]]))
        torch.testing.assert_close(env.adapter.state, terminal, atol=0, rtol=0)

    def test_action_queue_is_causal_beyond_one_period_and_keeps_gradients(self):
        env = self.environment(randomization='nominal', dt=.01)
        # 1.5 control periods: no part of the newly issued action arrives now.
        env.parameters['compute_latency_s'] = torch.full((2,), .015)
        first = (env.adapter.last_action + torch.tensor([[1., .4, .3, 0.]])).requires_grad_()
        env.step(first)
        initial_gradient = torch.autograd.grad(env.adapter.velocity.sum(), first,
                                               retain_graph=True)[0]
        torch.testing.assert_close(initial_gradient, torch.zeros_like(first))
        second = env.adapter.last_action.detach().clone().requires_grad_()
        env.step(second)
        first_gradient, second_gradient = torch.autograd.grad(
            env.adapter.velocity.sum(), (first, second))
        self.assertGreater(float(first_gradient.abs().sum()), 0)
        torch.testing.assert_close(second_gradient, torch.zeros_like(second))

    def test_velocity_loss_uses_shared_turn_and_endpoint_guidance(self):
        env = self.environment(randomization='nominal')
        env.adapter.reset(position=torch.tensor([[1., 1., 1.], [1.5, 1., 1.]]),
                          velocity=torch.tensor([[.3, .1, 0.], [.2, 0., 0.]]))
        env.targets = torch.tensor([[[2., 1., 1.], [2., 2., 1.]],
                                    [[1., 1., 1.], [2., 1., 1.]]])
        env.waypoint_index = torch.tensor([0, 1])
        target = torch.stack((env.targets[0, 0], env.targets[1, 1]))
        desired = guidance_velocity(env.adapter.position, target, env.targets[:, 1],
                                    env.speed_limit, env.guidance_config)
        action = torch.tensor([[4.903325, 0., 0., 0.]]).expand(2, -1)
        loss = env.differentiable_loss(action, action)['target_velocity']
        torch.testing.assert_close(loss, (env.adapter.velocity-desired).square().sum(-1))
        # Turn braking must reach the dynamics loss, rather than retaining the
        # old full-speed label while the policy is asked to slow down.
        self.assertLess(float(desired[0].norm()), env.speed_limit)

    def test_first_waypoint_uses_tight_handoff_and_final_keeps_existing_tolerance(self):
        env = self.environment(randomization='nominal', dt=.01)
        incoming = env.targets[:, 0]-env.starts
        incoming = incoming/incoming.norm(dim=-1, keepdim=True)
        q = env.adapter.quaternion.detach().clone()
        action = torch.tensor([[4.903325, 0., 0., 0.]]).expand(2, -1)
        env.adapter.reset(position=env.targets[:, 0]-.20*incoming, quaternion=q)
        env.step(action)
        self.assertEqual(env.waypoint_index.tolist(), [0, 0])
        env.adapter.reset(position=env.targets[:, 0]-.05*incoming, quaternion=q)
        env.step(action)
        self.assertEqual(env.waypoint_index.tolist(), [1, 1])
        final_direction = env.targets[:, 1]-env.targets[:, 0]
        final_direction = final_direction/final_direction.norm(dim=-1, keepdim=True)
        env.adapter.reset(position=env.targets[:, 1]-.20*final_direction, quaternion=q)
        _, _, done, info = env.step(action)
        self.assertTrue(bool(done.all()))
        self.assertTrue(bool(info['success'].all()))

    def test_full_environment_loss_backpropagates_through_transitions(self):
        env = self.environment(randomization='nominal')
        action = env.adapter.last_action.clone().requires_grad_()
        env.step(action)
        env.differentiable_loss(action, torch.zeros_like(action))['target_velocity'].mean().backward()
        self.assertIsNotNone(action.grad)
        self.assertTrue(bool(torch.isfinite(action.grad).all()))
        self.assertGreater(float(action.grad.abs().sum()), 0)
        env.detach()
        self.assertFalse(env.adapter.state.requires_grad)
        self.assertTrue(all(not tensor.requires_grad for tensor in env.observation().values()))


if __name__ == '__main__':
    unittest.main()
