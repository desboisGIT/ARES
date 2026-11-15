"""Original frozen FlightEnv observation path mirrored onto the verified plant.

Only this sensor boundary sees physical state. NN receives exactly FlightEnv's
Gaussian observations, low-resolution noisy depth, delayed map ring and memory.
The bounded monitor is separate and never replaces any network feature.
"""
from collections import deque
import torch
from ares.flight_env import FlightEnv
from ares.flight_sensors import FlightSensors
from ares.flight_control import initialize_memory,update_memory
class SensorSnapshot:
    def __init__(self,plant):self.parameters=plant.parameters;self.state=plant.state.detach().clone()
    @property
    def position(self):return self.state[:,:3]
    @property
    def quaternion(self):return self.state[:,3:7]
    @property
    def velocity(self):return self.state[:,7:10]
    @property
    def angular_velocity(self):return self.state[:,10:13]

class GaussianObserver:
    def __init__(self,scene,profile,sampled,plant,waypoints,seed,policy):
        env=FlightEnv(scene,profile,batch_size=1,device='cpu',seed=seed,dt=1/60,horizon=600,
            randomization='random',speed_limit=1.5,guidance_speed_fraction=policy.config.guidance_speed_fraction)
        self.plant=plant;env.adapter=SensorSnapshot(plant);env.sampled_profiles=[sampled];env.parameters=plant.parameters
        env.sensors=FlightSensors(scene,[sampled],device='cpu',seed=seed,environment_ids=[seed])
        env.targets=torch.tensor([waypoints],dtype=torch.float32)
        env.waypoint_index.zero_();env.travel.zero_();env._known.zero_();env._depth_history.clear()
        env.previous_action=torch.tensor([[.5*9.80665,0,0,0]],dtype=torch.float32)
        env.previous_command=env.previous_action.clone()
        # Same initial measured-map body-clear seed as original reset. It is not
        # the shield's bounded map and conveys no sampled occupancy to the NN.
        from ares.fast_world import inflation_structure
        import numpy as np
        cell=np.floor((plant.position[0].numpy()-env.origin_np)/env.resolution).astype(int)
        structure=inflation_structure(sampled['parameters']['body_radius_m']['nominal'],env.resolution)
        for delta in np.argwhere(structure)-np.asarray(structure.shape)//2:
            xyz=tuple(cell+delta)
            if all(0<=xyz[k]<env.shape[k] for k in range(3)):env._known[(0,*xyz)]=1
        env.steps=0;env.time_s=0.;env.episode_time_s.zero_()
        self.env=env;self.last=None;self.policy=policy;self.initialized=False
    @torch.no_grad()
    def observe(self,frame_index,time_s,sensor_state):
        env=self.env;env.adapter.state=sensor_state.detach().clone();env.steps=frame_index;env.time_s=time_s
        env.episode_time_s=torch.tensor([time_s],dtype=torch.float32)
        observation=env._observe()
        if not self.initialized:
            env.controller_memory=initialize_memory(observation);self.initialized=True
        observation.update(env.controller_memory.observation_state())
        self.last=observation
        return observation
    @torch.no_grad()
    def after_physical_step(self,issued,before_position):
        env=self.env
        env.travel+=(self.plant.position-before_position).norm(dim=-1)
        env.previous_action=torch.tensor([issued],dtype=torch.float32);env.previous_command=env.previous_action
        if self.last is not None:
            env.controller_memory=update_memory(env.controller_memory,self.last,env.previous_command,.01)
        first=(self.plant.position-env.targets[:,0]).norm(dim=-1)<env.guidance_config.waypoint_radius
        env.waypoint_index=torch.where(first,torch.ones_like(env.waypoint_index),env.waypoint_index)
    def reached(self):
        env=self.env
        return bool((env.waypoint_index==1).all() and
            (self.plant.position-env.targets[:,1]).norm(dim=-1).item()<.35 and
            self.plant.velocity.norm(dim=-1).item()<.8)
