"""Independent episode boundaries, sensor identities and learner rollout graphs."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

try:
    import torch
    import scipy
except ImportError:
    raise unittest.SkipTest('optional Phase 4 numpy/scipy/torch unavailable')

from ares.flight_env import FlightEnv
from ares.flight_randomization import sample_flight_profile
from ares.flight_training import TrainingConfig, train_bptt, train_ppo, generalized_advantage_estimate
from ares.flight_policy import FlightPolicy
from tests.test_depth import room_scene


class IndependentFlightResetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.scene = room_scene()
        cls.profile = json.loads((Path(__file__).parents[1] / 'ares/data/configs/drone_sim_v2.json').read_text())

    def env(self, **kwargs):
        return FlightEnv(self.scene, self.profile, batch_size=2, seed=38, **kwargs)

    def test_selected_reset_preserves_survivor_trajectory_observer_and_sensor_stream(self):
        control, reset = self.env(), self.env()
        command = torch.tensor([[5., .1, -.1, .03]]).expand(2, -1)
        for _ in range(5):
            control.step(command)
            reset.step(command)
        profile = reset.sampled_profiles[1]
        before = {key: value[1].clone() for key, value in reset.observation().items()}
        known = reset._known[1].clone()
        reset.reset_idx(torch.tensor([True, False]))
        self.assertIs(reset.sampled_profiles[1], profile)
        for key, value in before.items():
            torch.testing.assert_close(reset.observation()[key][1], value, atol=0, rtol=0)
        torch.testing.assert_close(reset._known[1], known, atol=0, rtol=0)
        self.assertEqual(reset.episode_counts, [1, 0])
        self.assertEqual(reset.episode_steps.tolist(), [0, 5])
        for _ in range(8):
            control.step(command)
            reset.step(command)
            torch.testing.assert_close(control.adapter.state[1], reset.adapter.state[1], atol=0, rtol=0)
            for key, value in control.observation().items():
                torch.testing.assert_close(value[1], reset.observation()[key][1], atol=0, rtol=0)
            torch.testing.assert_close(control._known[1], reset._known[1], atol=0, rtol=0)

    def test_selected_profile_resamples_every_parameter_reproducibly(self):
        env = self.env()
        previous = copy.deepcopy(env.sampled_profiles)
        old_parameters = env.parameters
        env.reset_idx([0])
        expected = sample_flight_profile(self.profile, 38+1000003, 'random')
        self.assertEqual(env.sampled_profiles[0], expected)
        self.assertEqual(env.sampled_profiles[1], previous[1])
        self.assertEqual(len(env.parameters), 29)
        for name in env.parameters:
            self.assertIsNot(env.parameters[name], old_parameters[name])
            torch.testing.assert_close(env.parameters[name][1], old_parameters[name][1], atol=0, rtol=0)
            self.assertEqual(env.sampled_profiles[0]['parameters'][name], expected['parameters'][name])
        env.reset_idx([1])
        expected = sample_flight_profile(self.profile, 39+1000003, 'random')
        self.assertEqual(env.sampled_profiles[1], expected)
        self.assertEqual(env.episode_counts, [1, 1])

    def test_action_delay_gradient_survives_other_episode_reset(self):
        env = self.env(randomization='nominal', dt=.01)
        # Preserve a public profile setting across adapter reconstruction.
        env.profile = copy.deepcopy(env.profile)
        env.profile['parameters']['compute_latency_s'] = {'nominal': .015, 'range': [.015, .015]}
        env.reset()
        first = torch.tensor([[6., .4, .3, 0.], [6., .4, .3, 0.]], requires_grad=True)
        env.step(first)
        before_loss = env.adapter.velocity.square().sum()
        env.reset_idx([0])
        env.step(torch.tensor([[5., 0., 0., 0.], [5., 0., 0., 0.]]))
        gradient = torch.autograd.grad(env.adapter.velocity.sum(), first, retain_graph=True)[0]
        torch.testing.assert_close(gradient[0], torch.zeros(4), atol=0, rtol=0)
        self.assertGreater(float(gradient[1].abs().sum()), 0)
        # Old losses still own valid tensors after parameters and masks are replaced.
        (before_loss+env.adapter.velocity.square().sum()).backward()
        self.assertTrue(bool(torch.isfinite(first.grad).all()))

    def test_failed_and_successful_lanes_reset_without_truncating_survivor(self):
        for success in (False, True):
            env = self.env(randomization='nominal', horizon=3)
            env.episode_steps = torch.tensor([2, 1])
            env.done = torch.tensor([True, False])
            env.success = torch.tensor([success, False])
            env.collided = torch.tensor([not success, False])
            survivor = env.adapter.state[1].clone()
            env.reset_idx(env.done)
            self.assertEqual(env.done.tolist(), [False, False])
            self.assertEqual(env.success.tolist(), [False, False])
            self.assertEqual(env.collided.tolist(), [False, False])
            torch.testing.assert_close(env.adapter.state[1], survivor, atol=0, rtol=0)
            _, _, done, _ = env.step(torch.tensor([[4.903325, 0., 0., 0.]]).expand(2, -1))
            self.assertFalse(bool(done.any()))
            self.assertEqual(env.episode_steps.tolist(), [1, 2])
            _, _, done, info = env.step(torch.tensor([[4.903325, 0., 0., 0.]]).expand(2, -1))
            self.assertEqual(done.tolist(), [False, True])
            self.assertEqual(info['truncated'].tolist(), [False, True])

    def test_dead_transitions_are_zero_and_gae_masks_are_independent(self):
        env = self.env(randomization='nominal')
        env.done = torch.tensor([True, False])
        action = torch.tensor([[4.9, 0., 0., 0.]]).expand(2, -1)
        _, reward, _, info = env.step(action)
        self.assertEqual(info['active'].tolist(), [False, True])
        self.assertEqual(float(reward[0]), 0.)
        self.assertEqual(float(env.differentiable_loss(action, action)['total'][0]), 0.)
        rewards = torch.tensor([[1., 1.], [10., 10.]])
        terminal = torch.tensor([[True, False], [True, True]])
        boundary = torch.tensor([[True, False], [True, True]])
        advantage, _ = generalized_advantage_estimate(rewards, torch.zeros_like(rewards),
            torch.zeros_like(rewards), terminal, boundary, gamma=.5, gae_lambda=1.)
        torch.testing.assert_close(advantage, torch.tensor([[1., 6.], [10., 10.]]))

    def test_learners_exclude_already_dead_lanes_from_transition_counts(self):
        class InitiallyDeadEnv(FlightEnv):
            def reset(self):
                obs = super().reset()
                self.done = torch.tensor([True, False], device=self.device)
                return obs

        for trainer in (train_bptt, train_ppo):
            env = InitiallyDeadEnv(self.scene, self.profile, batch_size=2, seed=38,
                                  randomization='nominal', horizon=10)
            result = trainer(env, FlightPolicy(), TrainingConfig(steps=1, rollout_steps=4,
                epochs=1, minibatch_size=8))
            self.assertEqual(result['training_steps'], 7)
            self.assertEqual(result['training_curve'][0]['episode_resets'], 1)
            self.assertTrue(torch.isfinite(torch.tensor(result['training_curve'][0]['loss'])))

    def test_curriculum_stage_hook_runs_between_windows_and_preserves_optimizer(self):
        class StagedEnv(FlightEnv):
            def __init__(self, *args, **kwargs):
                self.full_resets, self.hooks, self.partial_masks = 0, [], []
                self.trainer_previous = []
                self.inside_step = False
                super().__init__(*args, **kwargs)

            def reset(self):
                self.full_resets += 1
                obs = super().reset()
                self.episode_steps = torch.tensor([0, 1], device=self.device)
                return obs

            def reset_idx(self, mask):
                mask = torch.as_tensor(mask)
                if mask.dtype == torch.bool and not bool(mask.all()):
                    self.partial_masks.append(mask.clone())
                return super().reset_idx(mask)

            def start_update(self, update):
                self.hooks.append((update, self.steps))
                return self.reset() if update == 1 else None

            def step(self, action):
                self.inside_step = True
                result = super().step(action)
                self.inside_step = False
                return result

            def differentiable_loss(self, action, previous_action):
                if not self.inside_step:
                    self.trainer_previous.append(previous_action.detach().clone())
                return super().differentiable_loss(action, previous_action)

        original_adam = torch.optim.Adam
        for trainer in (train_bptt, train_ppo):
            optimizers = []

            def make_adam(*args, **kwargs):
                optimizer = original_adam(*args, **kwargs)
                optimizers.append(optimizer)
                return optimizer

            env = StagedEnv(self.scene, self.profile, batch_size=2, seed=38,
                            randomization='nominal', horizon=3)
            policy = FlightPolicy()
            with patch('ares.flight_training.torch.optim.Adam', side_effect=make_adam):
                result = trainer(env, policy, TrainingConfig(steps=3, rollout_steps=2,
                    epochs=1, minibatch_size=4))
            self.assertEqual(env.hooks, [(0, 0), (1, 2), (2, 4)])
            self.assertEqual(env.full_resets, 3)  # construction, trainer start, declared stage
            self.assertTrue(env.partial_masks)
            self.assertTrue(all(int(mask.sum()) == 1 for mask in env.partial_masks))
            self.assertEqual(len(optimizers), 1)
            self.assertEqual(int(optimizers[0].state[policy.actor.weight]['step']), 3)
            self.assertEqual(result['training_steps'], 12)
            self.assertTrue(all(torch.isfinite(torch.tensor(row['loss']))
                                for row in result['training_curve']))
            if trainer is train_bptt:
                torch.testing.assert_close(env.trainer_previous[2], torch.zeros(2, 4))
                self.assertGreater(float(env.trainer_previous[4].abs().sum()), 0.)

    def test_both_learners_train_across_mixed_episode_boundaries(self):
        class StaggeredEnv(FlightEnv):
            def reset(self):
                obs = super().reset()
                self.episode_steps = torch.tensor([0, 2], device=self.device)
                return obs

        for trainer in (train_bptt, train_ppo):
            env = StaggeredEnv(self.scene, self.profile, batch_size=2, seed=38,
                               randomization='nominal', horizon=4)
            result = trainer(env, FlightPolicy(), TrainingConfig(steps=2, rollout_steps=7,
                epochs=1, minibatch_size=8))
            self.assertEqual(result['training_steps'], 28)
            self.assertTrue(all(row['gradient_norm'] > 0 for row in result['training_curve']))
            self.assertTrue(all(row['episode_resets'] > 0 for row in result['training_curve']))
            self.assertEqual(env.episode_counts[0]+1, env.episode_counts[1])
            for row in result['training_curve']:
                self.assertTrue(torch.isfinite(torch.tensor(row['loss'])))


if __name__ == '__main__':
    unittest.main()
