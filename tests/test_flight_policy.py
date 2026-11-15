"""Policy bounds, real dynamics training, PPO probability and checkpoint checks."""
import json
from pathlib import Path
import tempfile
import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'Phase 4 optional torch dependency unavailable')
class FlightPolicyTests(unittest.TestCase):
    def setUp(self):
        from ares.flight_policy import FlightPolicy
        self.policy = FlightPolicy()
        torch.set_num_threads(2)

    def observation(self, batch=2):
        return {'depth': torch.rand(batch, 2, 24, 32), 'features': torch.rand(batch, 43),
                'p': torch.zeros(batch, 3), 'v': torch.zeros(batch, 3),
                'q': torch.tensor([[0., 0., 0., 1.]]).expand(batch, -1),
                'wp1': torch.tensor([[2., 0., 0.]]).expand(batch, -1),
                'wp2': torch.tensor([[4., 0., 0.]]).expand(batch, -1),
                'gaze': torch.tensor([[4., 0., 0.]]).expand(batch, -1),
                'speedlimit': torch.ones(batch, 1)}

    def test_nominal_initialization_physical_actions_and_sensor_gradient(self):
        obs = self.observation()
        torch.testing.assert_close(self.policy.desired_acceleration(obs), self.policy.reference_acceleration(obs))
        action = self.policy(obs)
        self.assertEqual(action.shape, (2, 4))
        self.assertTrue(bool(torch.isfinite(action).all()))
        self.assertTrue(bool(((action[:, 0] >= 0) & (action[:, 0] <= 24.5)).all()))
        self.assertTrue(bool((action[:, 1:].abs() <= 6).all()))
        # A nonzero trained actor transmits gradients to the actual depth CNN.
        with torch.no_grad():
            self.policy.actor.weight.fill_(.01)
        self.policy(obs).square().sum().backward()
        self.assertGreater(float(self.policy.depth_encoder[0].weight.grad.abs().sum()), 0)

    def test_hover_and_roll_pitch_sign(self):
        obs = self.observation()
        zero = torch.zeros(2, 3)
        hover = self.policy.action_from_acceleration(obs, zero)
        torch.testing.assert_close(hover[:, 0], torch.full((2,), 4.903325))
        torch.testing.assert_close(hover[:, 1:], torch.zeros(2, 3))
        command = self.policy.action_from_acceleration(obs, torch.tensor([[1., 0., 0.], [0., 1., 0.]]))
        self.assertGreater(float(command[0, 2]), 0)
        self.assertLess(float(command[1, 1]), 0)

    def test_vertical_gaze_has_finite_position_and_attitude_gradients(self):
        obs = self.observation()
        obs['p'] = obs['p'].clone().requires_grad_()
        obs['q'] = obs['q'].clone().requires_grad_()
        obs['gaze'] = torch.tensor([[0., 0., 2.], [0., 0., -2.]])
        acceleration = torch.zeros(2, 3, requires_grad=True)
        action = self.policy.action_from_acceleration(obs, acceleration)
        action.square().sum().backward()
        for tensor in (obs['p'], obs['q'], acceleration):
            self.assertTrue(bool(torch.isfinite(tensor.grad).all()))
        torch.testing.assert_close(action[:, 3], torch.zeros(2))

    def test_squashed_gaussian_log_probability(self):
        obs = self.observation()
        _, latent, old, _ = self.policy.sample(obs)
        new, entropy, _ = self.policy.evaluate_latent(obs, latent)
        torch.testing.assert_close(new, old)
        self.assertTrue(bool(torch.isfinite(entropy).all()))
        distribution, _ = self.policy.distribution(obs)
        expected = (distribution.log_prob(latent) - torch.log(
            self.policy.config.residual_acceleration * (1 - latent.tanh().square()))).sum(-1)
        torch.testing.assert_close(new, expected, atol=1e-5, rtol=1e-5)

    def test_gae_bootstraps_time_limit_and_stops_reset_trace(self):
        from ares.flight_training import generalized_advantage_estimate
        rewards = torch.tensor([[1., 1.], [10., 10.]])
        values = torch.zeros_like(rewards)
        next_values = torch.tensor([[4., 4.], [0., 0.]])
        terminated = torch.tensor([[True, False], [True, True]])
        boundaries = torch.ones(2, 2, dtype=torch.bool)
        advantages, returns = generalized_advantage_estimate(rewards, values, next_values,
                                                            terminated, boundaries, gamma=.5)
        torch.testing.assert_close(advantages, torch.tensor([[1., 3.], [10., 10.]]))
        torch.testing.assert_close(returns, advantages)

    def test_both_trainers_update_weights_through_real_dynamics(self):
        from ares.flight_training import TrainingConfig, train_bptt, train_ppo
        from ares.flight_policy import FlightPolicy
        from ares.diffaero_adapter import DiffAeroAdapter
        profile = json.loads((Path(__file__).parents[1] / 'ares/data/configs/drone_sim_v2.json').read_text())
        make_observation = self.observation

        class Environment:
            device = 'cpu'

            def __init__(self):
                self.adapter = DiffAeroAdapter(profile, batch_size=2)

            def reset(self):
                self.adapter.reset()
                return self.observation()

            def observation(self):
                obs = make_observation()
                obs.update(p=self.adapter.position, v=self.adapter.velocity, q=self.adapter.quaternion)
                return obs

            def step(self, action):
                self.adapter.step(action)
                reward = -self.adapter.position.sub(torch.tensor([1., 0., 0.])).square().sum(-1)
                done = torch.zeros(2, dtype=torch.bool)
                return self.observation(), reward, done, {'terminated': done, 'truncated': done}

            def differentiable_loss(self, action, previous_action):
                # Depends on the post-step rigid-body state, not direct policy output.
                loss = (self.adapter.velocity - torch.tensor([1., 0., 0.])).square().sum(-1)
                return {'total': loss, 'targetvelocity': loss}

            def detach(self):
                self.adapter.detach()

        for trainer in (train_bptt, train_ppo):
            policy = FlightPolicy()
            initial = policy.actor.weight.detach().clone()
            progress = []
            result = trainer(Environment(), policy, TrainingConfig(steps=2, rollout_steps=4,
                             epochs=2, minibatch_size=4, learning_rate=.001), progress=progress.append)
            self.assertEqual(result['training_steps'], 16)
            self.assertEqual([row['update'] for row in progress], [1, 2])
            self.assertEqual(progress[-1]['transitions'], 16)
            self.assertFalse(torch.equal(initial, policy.actor.weight))
            self.assertGreater(result['training_curve'][-1]['cnn_gradient_norm'], 0)
            self.assertTrue(all(row['gradient_norm'] > 0 for row in result['training_curve']))

    def test_weights_only_checkpoint_roundtrip_and_exclusive_publication(self):
        from ares.flight_training import load_checkpoint, save_checkpoint
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'policy.pt'
            kwargs = dict(algorithm='bptt', seed=7, training_steps=12, training_curve=[{'loss': 1.}],
                          metadata={'heldout_evaluated': False})
            save_checkpoint(path, self.policy, **kwargs)
            restored, payload = load_checkpoint(path)
            obs = self.observation()
            torch.testing.assert_close(self.policy(obs), restored(obs))
            self.assertEqual(payload['seed'], 7)
            self.assertFalse(payload['metadata']['heldout_evaluated'])
            with self.assertRaises(FileExistsError):
                save_checkpoint(path, self.policy, **kwargs)


if __name__ == '__main__':
    unittest.main()
