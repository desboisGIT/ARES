"""Versioned controller inputs, deterministic PPO replay and old weight meaning."""
from pathlib import Path
import math
import tempfile
import unittest

try:
    import torch
except ImportError:
    raise unittest.SkipTest('flight extra unavailable')

from ares.flight_policy import FlightPolicy
from ares.flight_training import save_checkpoint, load_checkpoint


class ControllerPolicyTests(unittest.TestCase):
    def observation(self):
        return {'depth':torch.zeros(2,2,24,32),'features':torch.zeros(2,43),
            'p':torch.zeros(2,3),'v':torch.zeros(2,3),'q':torch.tensor([[0.,0.,0.,1.]]).repeat(2,1),
            'wp1':torch.zeros(2,3),'wp2':torch.zeros(2,3),'gaze':torch.ones(2,3),
            'speedlimit':torch.ones(2,1),'controller_bias':torch.tensor([[0.,0.,1.],[0.,0.,-1.]]),
            'controller_velocity':torch.zeros(2,3),'controller_hold_anchor':torch.zeros(2,3),
            'controller_holding':torch.zeros(2,dtype=torch.bool),'controller_stopping':torch.zeros(2,dtype=torch.bool)}

    def test_adaptive_compensation_and_legacy_conversion_are_distinct(self):
        observation=self.observation();acceleration=torch.zeros(2,3)
        old=FlightPolicy({'controller_version':'nominal_v1'})
        new=FlightPolicy()
        torch.testing.assert_close(old.action_from_acceleration(observation,acceleration)[:,0],torch.full((2,),4.903325))
        torch.testing.assert_close(new.action_from_acceleration(observation,acceleration)[:,0],torch.tensor([4.403325,5.403325]))
        self.assertEqual(old.state_encoder[0].in_features,43)
        self.assertEqual(new.state_encoder[0].in_features,54)

    def test_repeated_latent_evaluation_never_advances_observer_state(self):
        policy=FlightPolicy();observation=self.observation()
        original={k:v.clone() for k,v in observation.items()}
        latent=torch.full((2,3),.3)
        first=policy.evaluate_latent(observation,latent)[0]
        second=policy.evaluate_latent(observation,latent)[0]
        torch.testing.assert_close(first,second)
        for key,value in observation.items():torch.testing.assert_close(value,original[key])

    def test_pre_version_checkpoint_loads_nominal_without_reinterpreting_weights(self):
        policy=FlightPolicy({'controller_version':'nominal_v1','guidance_version':'legacy_v1'})
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'legacy.pt'
            save_checkpoint(path,policy,algorithm='bptt',seed=0,training_steps=1,training_curve=[])
            payload=torch.load(path,weights_only=True)
            payload['policy_config'].pop('controller_version')
            payload['policy_config'].pop('guidance_version')
            payload['policy_config'].pop('max_yaw_rate')
            torch.save(payload,path)
            restored,_=load_checkpoint(path)
            self.assertEqual(restored.config.controller_version,'nominal_v1')
            self.assertEqual(restored.config.guidance_version,'legacy_v1')
            self.assertAlmostEqual(restored.config.max_yaw_rate,6.*math.pi)
            torch.testing.assert_close(policy(self.observation()),restored(self.observation()))

    def test_pre_yaw_limit_adaptive_checkpoint_keeps_original_conversion(self):
        policy=FlightPolicy({'max_yaw_rate':6.*math.pi})
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'adaptive.pt'
            save_checkpoint(path,policy,algorithm='bptt',seed=0,training_steps=1,training_curve=[])
            payload=torch.load(path,weights_only=True)
            payload['policy_config'].pop('max_yaw_rate')
            torch.save(payload,path)
            restored,_=load_checkpoint(path)
            torch.testing.assert_close(policy(self.observation()),restored(self.observation()))

    def test_braking_guidance_slows_before_right_angle_handoff(self):
        observation=self.observation()
        observation['wp1'][:]=torch.tensor([.35,0.,0.])
        observation['wp2'][:]=torch.tensor([.35,1.,0.])
        old=FlightPolicy({'guidance_version':'legacy_v1'})
        new=FlightPolicy()
        requested=new.reference_acceleration(observation)/new.config.velocity_gain
        self.assertLessEqual(float(requested.norm(dim=-1).max()),.201)
        self.assertTrue(torch.all(requested[:,1]==0))
        self.assertGreater(float(old.reference_acceleration(observation).norm(dim=-1).min()),1.)

    def test_configured_residual_bound_rotates_without_limiting_nominal_braking(self):
        from ares.flight_policy import quaternion_inverse_rotate
        policy=FlightPolicy({'residual_acceleration':.5})
        observation=self.observation()
        observation['q'][:]=torch.tensor([0.,0.,math.sin(math.pi/4),math.cos(math.pi/4)])
        latent=torch.tensor([[4.,-3.,2.],[-4.,3.,-2.]],requires_grad=True)
        reference=policy.reference_acceleration(observation)
        acceleration=policy.acceleration_from_latent(observation,latent)
        residual=quaternion_inverse_rotate(observation['q'],acceleration-reference)
        torch.testing.assert_close(residual,.5*latent.tanh())
        self.assertTrue(bool((residual.abs()<=.5).all()))
        acceleration.square().sum().backward()
        self.assertTrue(bool(torch.isfinite(latent.grad).all()))
        # The bound applies to learned corrections; explicit controller braking
        # and the zero-residual initialization retain their original authority.
        brake=torch.tensor([[0.,0.,-2.],[0.,0.,2.]])
        large=FlightPolicy({'residual_acceleration':2.})
        torch.testing.assert_close(policy.action_from_acceleration(observation,brake),
                                   large.action_from_acceleration(observation,brake))
        torch.testing.assert_close(policy.desired_acceleration(observation),
                                   large.desired_acceleration(observation))

    def test_configured_residual_scale_has_exact_ppo_probability(self):
        policy=FlightPolicy({'residual_acceleration':.5})
        observation=self.observation()
        distribution,_=policy.distribution(observation)
        latent=torch.full((2,3),.2)
        expected=(distribution.log_prob(latent)-torch.log(.5*(1-latent.tanh().square()))).sum(-1)
        actual,entropy,_=policy.evaluate_latent(observation,latent)
        torch.testing.assert_close(actual,expected)
        self.assertTrue(bool(torch.isfinite(entropy).all()))

    def test_nonfinite_or_nonpositive_residual_bound_is_rejected(self):
        for value in (0.,-.5,float('nan'),float('inf')):
            with self.subTest(bound=value),self.assertRaisesRegex(ValueError,'policy configuration'):
                FlightPolicy({'residual_acceleration':value})

    def test_guidance_speed_fraction_is_explicit_and_does_not_weaken_braking(self):
        obs=self.observation()
        obs['wp1'][:]=torch.tensor([4.,0.,0.]);obs['wp2'][:]=torch.tensor([6.,0.,0.])
        full=FlightPolicy();reserved=FlightPolicy({'guidance_speed_fraction':.95})
        torch.testing.assert_close(reserved.reference_acceleration(obs),
                                   full.reference_acceleration(obs)*.95)
        brake=torch.tensor([[-2.,0.,0.],[-2.,0.,0.]])
        torch.testing.assert_close(reserved.action_from_acceleration(obs,brake),
                                   full.action_from_acceleration(obs,brake))
