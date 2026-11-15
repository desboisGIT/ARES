"""Physics and autograd checks for the attributed upstream full rigid body."""
import json
from pathlib import Path
import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Phase 4 optional torch dependency unavailable")
class DiffAeroAdapterTests(unittest.TestCase):
    def setUp(self):
        from ares.diffaero_adapter import DiffAeroAdapter
        self.adapter = DiffAeroAdapter
        self.profile = json.loads((Path(__file__).parents[1] / "ares/data/configs/drone_sim_v2.json").read_text())

    def test_hover_and_body_rate_frame(self):
        model = self.adapter(self.profile, batch_size=2)
        # Different yaw must not rotate the body-rate feedback into another axis.
        model.reset(quaternion=[[0, 0, 0, 1], [0, 0, .70710678, .70710678]])
        command = torch.tensor([[4.903325, 1, 0, 0], [4.903325, 1, 0, 0]])
        for _ in range(20):
            model.step(command)
        torch.testing.assert_close(model.angular_velocity[0], model.angular_velocity[1], atol=1e-6, rtol=1e-6)
        self.assertGreater(float(model.angular_velocity[0, 0]), .99)
        torch.testing.assert_close(model.quaternion.norm(dim=-1), torch.ones(2))
        hover = self.adapter(self.profile)
        for _ in range(20):
            hover.step([[4.903325, 0, 0, 0]])
        torch.testing.assert_close(hover.position, torch.zeros(1, 3), atol=1e-6, rtol=0)

    def test_rollout_gradient_matches_finite_difference(self):
        def rollout(command):
            model = self.adapter(self.profile)
            for _ in range(5):
                model.step(command)
            return model.position[0, 2] + model.velocity[0, 1] + model.angular_velocity[0, 2]
        command = torch.tensor([[6., .3, -.2, .4]], requires_grad=True)
        rollout(command).backward()
        analytic = command.grad.clone()
        for component in range(4):
            plus, minus = command.detach().clone(), command.detach().clone()
            plus[0, component] += .002
            minus[0, component] -= .002
            numeric = (rollout(plus) - rollout(minus)) / .004
            torch.testing.assert_close(analytic[0, component], numeric, atol=8e-5, rtol=.003)

    def test_per_episode_profiles_and_actuator_lag(self):
        params = {key: value['nominal'] for key, value in self.profile['parameters'].items()}
        second = dict(params, mass_kg=.55, motor_lag_s=.04, inertia_kg_m2=[.0018, .0022, .003])
        model = self.adapter([params, second])
        initial = model.applied_thrust.clone()
        model.step([[10, 0, 0, 0], [10, 0, 0, 0]])
        tau = torch.tensor([.025, .04])
        expected = 10 + (initial - 10) * torch.exp(-.02 / tau)
        torch.testing.assert_close(model.applied_thrust, expected)
        self.assertFalse(torch.equal(model.velocity[0], model.velocity[1]))

    def test_sampled_inertia_changes_response_with_fixed_nominal_controller(self):
        params = {key: value['nominal'] for key, value in self.profile['parameters'].items()}
        inertia = self.profile['parameters']['inertia_kg_m2']['range']
        profiles = [dict(params, inertia_kg_m2=[axis[endpoint] for axis in inertia])
                    for endpoint in (0, 1)]
        for device in ('cpu', 'cuda') if torch.cuda.is_available() else ('cpu',):
            with self.subTest(device=device):
                model = self.adapter(profiles, device=device)
                command = torch.tensor([[4.903325, 1., 0., 0.]]*2, device=device, requires_grad=True)
                model.step(command)
                rates = model.angular_velocity[:, 0]
                expected = torch.tensor([.4865829, .4204217], device=device)
                torch.testing.assert_close(rates, expected, atol=2e-6, rtol=0)
                self.assertGreater(float((rates[0]-rates[1]).detach()), .06)
                nominal = torch.diag_embed(torch.tensor([[.002,.002,.003]]*2, device=device))
                torch.testing.assert_close(model.model.controller.inertia, nominal)
                self.assertFalse(torch.equal(model.model._J, model.model.controller.inertia))
                rates.sum().backward()
                self.assertTrue(bool(torch.isfinite(command.grad).all()))
                self.assertGreater(float(command.grad[:,1].min()), 0.)
                self.assertEqual(model.metadata['nominal_controller_inertia_kg_m2'], list(model.controller_inertia))

    def test_controller_inertia_is_configurable_and_validated(self):
        for inertia in ([0.,.002,.003], [-.002,.002,.003], [.002,.003], [float('nan'),.002,.003]):
            with self.assertRaisesRegex(ValueError, 'controller_inertia'):
                self.adapter(self.profile, controller_inertia=inertia)
        model = self.adapter(self.profile, controller_inertia=[.001,.001,.002])
        model.step([[4.903325,1,0,0]])
        self.assertLess(float(model.angular_velocity[0,0]), .3)

    def test_compile_requires_cuda_and_boolean_option(self):
        with self.assertRaisesRegex(ValueError, 'CUDA'):
            self.adapter(self.profile, compile_dynamics=True)
        with self.assertRaisesRegex(ValueError, 'boolean'):
            self.adapter(self.profile, compile_dynamics='yes')

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "CUDA unavailable")
    def test_compiled_cuda_state_and_gradient_parity(self):
        params = {key: value['nominal'] for key, value in self.profile['parameters'].items()}
        profiles = [dict(params, inertia_kg_m2=[.0018+(i%8)*.00005,
                        .0022-(i%8)*.00005,.0027+(i%8)*.00007]) for i in range(64)]
        eager = self.adapter(profiles, device='cuda')
        compiled = self.adapter(profiles, device='cuda', compile_dynamics=True)
        actions = torch.tensor([[6.,.3,-.2,.1]],device='cuda').repeat(64,1)
        results, gradients = [], []
        for model in (eager, compiled):
            action = actions.clone().requires_grad_()
            for _ in range(2):
                model.step(action)
            model.state.square().sum().backward()
            results.append(model.state.detach())
            gradients.append(action.grad.detach())
        torch.testing.assert_close(results[0],results[1],atol=3e-6,rtol=3e-5)
        torch.testing.assert_close(gradients[0],gradients[1],atol=3e-6,rtol=3e-5)
        self.assertTrue(compiled.metadata['torch_compile_dynamics'])
        self.assertFalse(eager.metadata['torch_compile_dynamics'])

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_differentiable_batch(self):
        model = self.adapter(self.profile, batch_size=8, device='cuda')
        command = torch.tensor([[6., .3, .2, .1]], device='cuda').expand(8, -1).clone().requires_grad_()
        model.step(command).square().sum().backward()
        self.assertTrue(bool(torch.isfinite(command.grad).all()))
        self.assertGreater(float(command.grad.abs().sum()), 0)


if __name__ == '__main__':
    unittest.main()
