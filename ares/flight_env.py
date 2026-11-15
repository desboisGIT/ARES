"""ARES waypoint tasks around DiffAero, with observed-only policy inputs.

Geometry and differentiable clearance losses belong to the training evaluator.
Sensor images and the ring of ranges come from measurements and map memory.
The state estimator here is a randomized approximation, not Phase 2 SLAM.
"""
from collections import deque
import math

import numpy as np
from scipy import ndimage
import torch
import torch.nn.functional as F

from .contracts import validate_scene, validate_profile
from .diffaero_adapter import DiffAeroAdapter
from .fast_world import inflation_structure
from .flight_randomization import sample_flight_profile, randomization_manifest
from .flight_sensors import FlightSensors
from .flight_collision import FlightCollisionChecker
from .flight_control import initialize_memory, update_memory
from .flight_guidance import GuidanceConfig, guidance_velocity
from .vendor.diffaero.math import quat_rotate, quat_inv

FLIGHT_ENV_VERSION = 'ares.flight_env.v2'


def waypoint_tasks(scene, radius, count, seed):
    """Random two-leg configuration-space routes; never policy observations.

    Axis-aligned segments have exact sphere clearance. The second leg may turn
    or reverse. Tasks include clutter, but are not an exhaustive narrow-gap set.
    """
    grid = scene['grid']
    shape, res = tuple(grid['shape']), grid['resolution']
    occupied = np.zeros(shape, bool)
    occupied[tuple(np.asarray(grid['occupied']).T)] = True
    safe = ndimage.binary_erosion(~occupied, structure=inflation_structure(radius, res), border_value=0)
    origin = np.asarray(grid['origin'])
    cells = np.argwhere(safe)
    cells = cells[(origin[2]+(cells[:, 2]+.5)*res >= .7) &
                  (origin[2]+(cells[:, 2]+.5)*res <= 2.)]
    if not len(cells):
        raise ValueError('scene has no body-clear flight task launch centers')
    rng = np.random.default_rng(seed)
    tasks = []
    directions = np.concatenate((np.eye(3,dtype=int),-np.eye(3,dtype=int)))
    for _ in range(count):
        for attempt in range(100):
            start = cells[rng.integers(len(cells))].copy()
            current = start.copy()
            points, lengths = [], []
            previous = None
            for leg in range(2):
                options = directions[rng.permutation(6)]
                if previous is not None:
                    options = sorted(options, key=lambda d:int(np.array_equal(d,previous)))
                best = None
                for direction in options:
                    run, endpoint = 0, current.copy()
                    for distance in range(1,int(rng.integers(5,14))):
                        cell = current+direction*distance
                        if np.any(cell<0) or np.any(cell>=shape) or not safe[tuple(cell)]:
                            break
                        run, endpoint = distance,cell
                    if run>=3:
                        best = endpoint, direction, run
                        break
                if best is None:
                    break
                current, previous, run = best
                points.append(current.copy()); lengths.append(run*res)
            if len(points)==2:
                centers = origin+(np.stack((start,*points))+.5)*res
                tasks.append({'positions':centers.tolist(),'leg_lengths_m':lengths})
                break
        else:
            raise ValueError('could not sample a nontrivial two-waypoint route')
    return tasks


class FlightEnv:
    def __init__(self, scene, profile, *, batch_size=16, device='cpu', seed=0,
                 dt=1/60, horizon=600, randomization='random', speed_limit=1.5,
                 guidance_speed_fraction=1.):
        validate_scene(scene); validate_profile(profile)
        if (type(batch_size) is not int or batch_size<1 or type(seed) is not int or
                type(horizon) is not int or horizon<1 or not math.isfinite(dt) or dt<=0 or
                not math.isfinite(speed_limit) or speed_limit<=0):
            raise ValueError('positive batch/horizon/interval/speed and integer seed required')
        self.scene,self.profile = scene,profile
        self.batch_size,self.device,self.seed = batch_size,torch.device(device),seed
        self.dt,self.horizon,self.randomization = dt,horizon,randomization
        self.speed_limit = speed_limit
        self.guidance_config = GuidanceConfig(speed_fraction=guidance_speed_fraction)
        self.episode = -1
        self.episode_counts = [-1] * batch_size
        self._generators = [torch.Generator(device=self.device) for _ in range(batch_size)]
        self._noise_buffers = [None] * batch_size
        self._noise_offsets = [0] * batch_size
        self.shape = tuple(scene['grid']['shape'])
        self.resolution = float(scene['grid']['resolution'])
        self.origin_np = np.asarray(scene['grid']['origin'],dtype=float)
        self.origin = torch.tensor(self.origin_np,device=self.device,dtype=torch.float32)
        self.occupied_np = np.zeros(self.shape,bool)
        self.occupied_np[tuple(np.asarray(scene['grid']['occupied']).T)] = True
        occupied = np.argwhere(self.occupied_np)
        self._box_lower_np = self.origin_np + occupied*self.resolution
        self._box_upper_np = self._box_lower_np + self.resolution
        self._box_centers = torch.tensor((self._box_lower_np+self._box_upper_np)/2,
                                         device=self.device,dtype=torch.float32)
        self.collision_checker=FlightCollisionChecker(scene,device=str(self.device))
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(seed)
        self.reset()

    def reset(self):
        """Start fresh episodes for the whole batch."""
        return self.reset_idx(torch.ones(self.batch_size, device=self.device, dtype=torch.bool))

    def reset_idx(self, indices):
        """Reset selected episodes and return observations for the entire batch.

        Tensor replacements preserve surviving rollout graphs. Episode randomness
        belongs to each environment, so another member's reset never changes it.
        """
        indices = torch.as_tensor(indices, device=self.device)
        if indices.numel() == 0 and indices.dtype != torch.bool:
            return self._observation
        if indices.dtype == torch.bool:
            if indices.shape != (self.batch_size,):
                raise ValueError('reset mask must have one entry per environment')
            mask = indices
        else:
            if indices.ndim > 1 or indices.dtype.is_floating_point:
                raise ValueError('reset indices must be integer indexes or a boolean mask')
            indices = indices.long().reshape(-1)
            if bool(((indices < 0) | (indices >= self.batch_size)).any()):
                raise ValueError('reset index outside batch')
            mask = torch.zeros(self.batch_size, device=self.device, dtype=torch.bool)
            mask[indices] = True
        selected = mask.nonzero(as_tuple=False).flatten().tolist()
        if not selected:
            return self._observation
        initial = not hasattr(self, 'adapter')
        old_adapter = getattr(self, 'adapter', None)
        profiles = list(getattr(self, 'sampled_profiles', [None] * self.batch_size))
        tasks = list(getattr(self, 'tasks', [None] * self.batch_size))
        noise_ids = list(getattr(self, '_sensor_noise_ids', [0] * self.batch_size))
        for index in selected:
            self.episode_counts[index] += 1
            episode_seed = self.seed + index + self.episode_counts[index] * 1000003
            profiles[index] = sample_flight_profile(self.profile, episode_seed, self.randomization)
            radius = profiles[index]['parameters']['body_radius_m']['range'][1]
            tasks[index] = waypoint_tasks(self.scene, radius, 1, episode_seed)[0]
            self._generators[index].manual_seed(episode_seed)
            self._noise_buffers[index] = torch.randn(8192, device=self.device,
                                                      generator=self._generators[index])
            self._noise_offsets[index] = 0
            # Shared sensor seed with a stable, episode-specific stream identity.
            noise_ids[index] = (index + self.episode_counts[index] * self.batch_size) % 2**32
        self.episode += 1
        self.sampled_profiles, self.tasks, self._sensor_noise_ids = profiles, tasks, noise_ids
        adapter = DiffAeroAdapter(profiles, device=self.device, dt=self.dt,
                                 compile_dynamics=self.device.type == 'cuda')
        targets = torch.tensor([task['positions'] for task in tasks], device=self.device, dtype=torch.float32)
        self.starts, self.targets = targets[:, 0], targets[:, 1:]
        # The soft loss must not demand more clearance than the assigned route
        # actually offers. This evaluator-only budget is fixed for each episode.
        route_clearance = torch.zeros(self.batch_size,device=self.device)
        for index in selected:
            points = np.asarray(tasks[index]['positions'])
            best = float('inf')
            for start,end in zip(points[:-1],points[1:]):
                lo,hi = np.minimum(start,end),np.maximum(start,end)
                if np.count_nonzero(np.abs(end-start)>1e-8) != 1:
                    raise ValueError('clearance budget requires axis-aligned task legs')
                separation = np.maximum(np.maximum(self._box_lower_np-hi,lo-self._box_upper_np),0)
                box_distance = np.linalg.norm(separation,axis=-1).min() if len(separation) else float('inf')
                boundary_distance = min((lo-self.origin_np).min(),
                    (self.origin_np+np.asarray(self.shape)*self.resolution-hi).min())
                best = min(best,box_distance,boundary_distance)
            clearance = best-profiles[index]['parameters']['body_radius_m']['nominal']
            if not math.isfinite(clearance) or clearance <= 0:
                raise ValueError(f'task route is not body-clear: lane {index}, clearance {clearance}')
            route_clearance[index] = clearance
        old_clearance = getattr(self,'task_route_clearance_m',route_clearance)
        self.task_route_clearance_m = torch.where(mask,route_clearance,old_clearance)
        self.obstacle_margin_m = torch.minimum(torch.full_like(self.task_route_clearance_m,.10),
                                               .5*self.task_route_clearance_m)
        heading = torch.atan2((targets[:, 1]-targets[:, 0])[:, 1], (targets[:, 1]-targets[:, 0])[:, 0])
        q = torch.stack((heading*0, heading*0, torch.sin(heading/2), torch.cos(heading/2)), -1)
        adapter.reset(self.starts, quaternion=q)
        if not initial:
            for name in ('state', 'applied_thrust', 'last_action'):
                new, old = getattr(adapter, name), getattr(old_adapter, name)
                setattr(adapter, name, torch.where(mask.reshape((-1,)+(1,)*(new.ndim-1)), new, old))
            adapter.model._acc = torch.where(mask[:, None], adapter.model._acc, old_adapter.model._acc)
        self.adapter, self.parameters = adapter, adapter.parameters
        self.sensors = FlightSensors(self.scene, profiles, device=str(self.device), seed=self.seed,
                                     environment_ids=noise_ids)

        def replace(name, value):
            old = getattr(self, name, value)
            setattr(self, name, torch.where(mask.reshape((-1,)+(1,)*(value.ndim-1)), value, old))

        zero = torch.zeros(self.batch_size, device=self.device)
        replace('_transition_active', torch.ones_like(zero, dtype=torch.bool))
        for name in ('done', 'success', 'collided'):
            replace(name, zero.bool())
        for name in ('waypoint_index', 'episode_steps', 'speed_limit_steps'):
            replace(name, zero.long())
        for name in ('travel', 'energy_used', 'maximum_speed', 'episode_time_s'):
            replace(name, zero)
        replace('minimum_clearance', torch.full_like(zero, float('inf')))
        replace('minimum_clearance_is_lower_bound', torch.ones_like(zero, dtype=torch.bool))
        drift = self._noise((self.batch_size, 3), mask)
        drift = drift / drift.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        replace('drift_direction', drift)
        nominal_command=torch.zeros((self.batch_size,4),device=self.device)
        nominal_command[:,0]=.5*9.80665
        replace('previous_action', nominal_command)
        replace('previous_command', nominal_command)
        replace('_previous_position', adapter.position.detach().clone())
        # Size queues for the profile's full latency range, not the current draw.
        history_length = math.ceil(self.profile['parameters']['compute_latency_s']['range'][1]/self.dt)+3
        if initial:
            self.steps, self.time_s = 0, 0.
            self._depth_history = deque(maxlen=history_length)
            self._action_history = deque([self.previous_action.clone() for _ in range(history_length)], maxlen=history_length)
        else:
            self._action_history = deque((torch.where(mask[:, None], self.previous_action, action)
                                          for action in self._action_history), maxlen=history_length)
            self._depth_history = deque((
                (torch.where(mask, torch.full_like(ready, float('inf')), ready), record, p, q)
                for ready, record, p, q in self._depth_history), maxlen=history_length)
        replace('_known', torch.zeros((self.batch_size, *self.shape), device=self.device, dtype=torch.uint8))
        for env in selected:
            r = profiles[env]['parameters']['body_radius_m']['nominal']
            cell = np.floor((self.starts[env].cpu().numpy()-self.origin_np)/self.resolution).astype(int)
            structure = inflation_structure(r, self.resolution)
            for delta in np.argwhere(structure)-np.asarray(structure.shape)//2:
                xyz = tuple(cell+delta)
                if all(0 <= xyz[k] < self.shape[k] for k in range(3)):
                    self._known[(env, *xyz)] = 1
        fresh = self._observe(mask, replace_latest=not initial)
        if initial:
            self._observation = fresh
        else:
            self._observation = {name: torch.where(mask.reshape((-1,)+(1,)*(value.ndim-1)),
                                                  value, self._observation[name])
                                 for name, value in fresh.items()}
        if initial:
            self.controller_memory=initialize_memory(self._observation)
        else:
            self.controller_memory=update_memory(self.controller_memory,self._observation,
                                                 self.previous_command,0.,reset_mask=mask)
        self._observation.update(self.controller_memory.observation_state())
        return self._observation

    def _noise(self, shape, mask=None):
        # Buffered independent streams avoid one CUDA RNG launch per lane per
        # measurement while keeping other episodes' reset schedules irrelevant.
        selected = [True] * self.batch_size if mask is None else mask.tolist()
        count = math.prod(shape[1:])
        rows = []
        for index, live in enumerate(selected):
            if not live:
                rows.append(torch.zeros(shape[1:], device=self.device))
                continue
            offset = self._noise_offsets[index]
            if offset + count > self._noise_buffers[index].numel():
                self._noise_buffers[index] = torch.randn(max(8192, count), device=self.device,
                                                         generator=self._generators[index])
                offset = 0
            rows.append(self._noise_buffers[index][offset:offset+count].reshape(shape[1:]))
            self._noise_offsets[index] = offset + count
        return torch.stack(rows)

    def clearance(self,position):
        """Exact free-space sphere/box gap with a piecewise analytic gradient.

        Nearest-box selection is evaluator geometry, outside autograd; only the
        selected box's distance retains a graph. Inside occupancy the minimum
        signed box distance supplies a penetration signal, not an exact union
        penetration depth. Continuous collision evidence remains independent.
        """
        if len(self._box_centers):
            with torch.no_grad():
                best = torch.full((len(position),),float('inf'),device=position.device)
                nearest = torch.zeros(len(position),dtype=torch.long,device=position.device)
                for start in range(0,len(self._box_centers),4096):
                    q = (position[:,None]-self._box_centers[None,start:start+4096]).abs()-self.resolution/2
                    distance = q.clamp_min(0).norm(dim=-1)+q.amax(-1).clamp_max(0)
                    value,index = distance.min(-1)
                    nearer = value<best
                    nearest = torch.where(nearer,index+start,nearest)
                    best = torch.minimum(best,value)
            q = (position-self._box_centers[nearest]).abs()-self.resolution/2
            distance = q.clamp_min(0).norm(dim=-1)+q.amax(-1).clamp_max(0)
        else:
            distance = torch.full_like(position[:,0],float('inf'))
        lower = (position-self.origin).amin(-1)
        upper = (self.origin+torch.tensor(self.shape,device=self.device)*self.resolution-position).amin(-1)
        return torch.minimum(distance,torch.minimum(lower,upper)) - self.parameters['body_radius_m']

    @torch.no_grad()
    def _integrate_map(self,raw_depth,position,quaternion):
        # Sparse measured-ray integration. Invalid returns certify no free space.
        height,width = raw_depth.shape[1:]
        rows,cols = torch.meshgrid(torch.arange(0,height,3,device=self.device),
                                   torch.arange(0,width,3,device=self.device),indexing='ij')
        rows,cols=rows.flatten(),cols.flatten()
        depth = raw_depth[:,rows,cols]
        fx = width/(2*torch.tan(self.parameters['depth_horizontal_fov_rad']/2))
        fy = height/(2*torch.tan(self.parameters['depth_vertical_fov_rad']/2))
        u=(cols[None]-(width-1)/2)/fx[:,None]
        v=(rows[None]-(height-1)/2)/fy[:,None]
        body=torch.stack((torch.ones_like(u),-u,-v),-1)
        directions=quat_rotate(quaternion[:,None].expand(-1,len(cols),-1),body)
        sigma=self.parameters['depth_noise_std_m'][:,None]+self.parameters['depth_noise_quadratic_per_m2'][:,None]*depth.square()
        free_distance=(depth-3*sigma-self.resolution).clamp_min(0)
        fractions=torch.linspace(0,1,24,device=self.device)
        points=position[:,None,None]+directions[:,:,None]*free_distance[:,:,None,None]*fractions[None,None,:,None]
        cells=torch.floor((points-self.origin)/self.resolution).long()
        valid=torch.isfinite(depth)[:,:,None].expand(-1,-1,24)&torch.isfinite(points).all(-1)
        valid &= ((cells>=0)&(cells<torch.tensor(self.shape,device=self.device))).all(-1)
        env=torch.arange(self.batch_size,device=self.device)[:,None,None].expand_as(valid)
        selected=cells[valid]
        flat=((env[valid]*self.shape[0]+selected[:,0])*self.shape[1]+selected[:,1])*self.shape[2]+selected[:,2]
        self._known.flatten().scatter_reduce_(0,flat,torch.ones_like(flat,dtype=torch.uint8),reduce='amax',include_self=True)
        endpoint=position[:,None]+directions*depth[:,:,None]
        hit=torch.floor((endpoint-self.origin)/self.resolution).long()
        valid=torch.isfinite(depth)&((hit>=0)&(hit<torch.tensor(self.shape,device=self.device))).all(-1)
        env=torch.arange(self.batch_size,device=self.device)[:,None].expand_as(valid)
        selected=hit[valid]
        flat=((env[valid]*self.shape[0]+selected[:,0])*self.shape[1]+selected[:,1])*self.shape[2]+selected[:,2]
        self._known.flatten().scatter_reduce_(0,flat,torch.full_like(flat,2,dtype=torch.uint8),reduce='amax',include_self=True)

    @torch.no_grad()
    def _ring(self,position,quaternion):
        angles=torch.arange(16,device=self.device)*2*math.pi/16
        directions=torch.stack((angles.cos(),angles.sin(),angles*0),-1)[None].expand(self.batch_size,-1,-1)
        directions=quat_rotate(quaternion[:,None].expand(-1,16,-1),directions)
        distances=torch.arange(.1,4.01,self.resolution/2,device=self.device)
        points=position[:,None,None]+directions[:,:,None]*distances[None,None,:,None]
        cells=torch.floor((points-self.origin)/self.resolution).long()
        inside=((cells>=0)&(cells<torch.tensor(self.shape,device=self.device))).all(-1)
        cells=torch.maximum(torch.zeros_like(cells),torch.minimum(cells,torch.tensor(self.shape,device=self.device)-1))
        env=torch.arange(self.batch_size,device=self.device)[:,None,None].expand_as(inside)
        free=(self._known[env,cells[...,0],cells[...,1],cells[...,2]]==1)&inside
        # Unknown stops a range just as an observed obstacle does.
        clear=free.cumprod(-1).bool()
        return torch.where(clear,distances[None,None],0).amax(-1)/4

    def _observe(self, mask=None, *, replace_latest=False):
        if mask is None:
            mask = torch.ones(self.batch_size, device=self.device, dtype=torch.bool)
        p,q,v,w=self.adapter.position,self.adapter.quaternion,self.adapter.velocity,self.adapter.angular_velocity
        with torch.no_grad():
            # Effective measurement age includes camera cadence; stochastic
            # errors scale with sampled IMU/flow rates. No ground truth map is
            # given to the network or to the ring extraction.
            age=1/self.parameters['depth_fps']+self.parameters['compute_latency_s']
            flow_sigma=self.parameters['optical_flow_noise_std_mps']*torch.sqrt(100/self.parameters['optical_flow_rate_hz'])
            accel_sigma=self.parameters['imu_accel_noise_std']/self.parameters['imu_rate_hz'].sqrt()
            velocity_error=self._noise((self.batch_size,3),mask)*(flow_sigma+accel_sigma)[:,None]
            position_error=self.drift_direction*(self.travel*self.parameters['odometry_drift_per_m'])[:,None]
            position_error[:,2]+=self._noise((self.batch_size,),mask)*self.parameters['range_noise_std_m']
            gyro_error=self._noise((self.batch_size,3),mask)*self.parameters['imu_gyro_noise_std'][:,None]
        estimated_p=p-v*age[:,None]+position_error
        estimated_v=v+velocity_error
        estimated_w=w+gyro_error
        with torch.no_grad():
            # Sensor queues model capture cadence and delivery delay. This
            # simplified rig does not run the Phase 2 estimator or fuse flow.
            sensor=self.sensors.observe(p.detach(),q.detach(),self.steps)
            available=self.episode_time_s+self.parameters['compute_latency_s']
            available=torch.where(mask, available, torch.full_like(available, float('inf')))
            # Map rays use the same delayed/drifting public pose as the
            # policy; only physical image rendering receives the true pose.
            capture=(available,sensor,estimated_p.detach(),q.detach())
            if replace_latest and self._depth_history:
                old_ready,old_sensor,old_p,old_q=self._depth_history[-1]
                merged={name:torch.where(mask.reshape((-1,)+(1,)*(value.ndim-1)),value,old_sensor[name])
                        if torch.is_tensor(value) else value for name,value in sensor.items()}
                self._depth_history[-1]=(torch.where(mask,available,old_ready),merged,
                    torch.where(mask[:,None],estimated_p.detach(),old_p),torch.where(mask[:,None],q.detach(),old_q))
            else:
                self._depth_history.append(capture)
            raw=torch.full_like(sensor['raw_depth'],float('nan'))
            image=torch.zeros_like(sensor['depth'])
            for ready,record,capture_p,capture_q in self._depth_history:
                delivered=(ready<=self.episode_time_s+1e-8)&mask
                raw=torch.where(delivered[:,None,None],record['raw_depth'],raw)
                image=torch.where(delivered[:,None,None,None],record['depth'],image)
                if delivered.any():
                    selected=torch.where(delivered[:,None,None],record['raw_depth'],float('nan'))
                    self._integrate_map(selected,capture_p,capture_q)
        inv=quat_inv(q)
        waypoint1=torch.where((self.waypoint_index==0)[:,None],self.targets[:,0],self.targets[:,1])
        waypoint2=self.targets[:,1]
        gaze=waypoint1
        gravity=torch.zeros_like(v); gravity[:,2]=-1
        speed=torch.full((self.batch_size,1),self.speed_limit,device=self.device)
        ring=self._ring(estimated_p.detach(),q.detach())
        previous=torch.cat((self.previous_action[:,:1]/24.5,self.previous_action[:,1:]/6),-1)
        features=torch.cat((quat_rotate(inv,estimated_v),quat_rotate(inv,gravity),estimated_w,
            quat_rotate(inv,waypoint1-estimated_p),quat_rotate(inv,waypoint2-estimated_p),
            quat_rotate(inv,gaze-estimated_p),q,speed,ring,previous),-1)
        return {'depth':image.detach(),'features':features,'p':estimated_p,'v':estimated_v,'q':q,
                'wp1':waypoint1,'wp2':waypoint2,'gaze':gaze,'speedlimit':speed,
                'previous_command':self.previous_command}

    def observation(self):
        return self._observation

    def differentiable_loss(self,action,previous_action):
        p,v=self.adapter.position,self.adapter.velocity
        target=torch.where((self.waypoint_index==0)[:,None],self.targets[:,0],self.targets[:,1])
        delta=target-p
        distance=delta.norm(dim=-1,keepdim=True)
        target_v=guidance_velocity(p,target,self.targets[:,1],self.speed_limit,self.guidance_config)
        velocity=(v-target_v).square().sum(-1)
        obstacle=F.relu(self.obstacle_margin_m-self.clearance(p)).square()
        smooth=((action-previous_action)/torch.tensor([24.5,6,6,6],device=self.device)).square().sum(-1)
        direction=delta/(distance+1e-6)
        bodyforward=quat_rotate(self.adapter.quaternion,torch.tensor([1.,0.,0.],device=self.device).expand_as(p))
        gaze=1-(bodyforward*direction).sum(-1)
        speed=F.relu(v.norm(dim=-1)-self.speed_limit).square()
        active=getattr(self,'_transition_active',~self.done).float()
        total=(velocity+15*obstacle+.1*smooth+.1*gaze+3*speed)*active
        return {'total':total,'target_velocity':velocity*active,'obstacle':obstacle*active,
                'smoothness':smooth*active,'gaze':gaze*active,'speed':speed*active}

    def step(self,action,*,stop_requested=None):
        if not isinstance(action,torch.Tensor) or action.shape!=(self.batch_size,4) or not torch.isfinite(action).all():
            raise ValueError('finite tensor actions Bx4 required')
        active=(~self.done).clone()
        self._transition_active=active
        old_state=self.adapter.state
        old_position=self.adapter.position
        old_thrust=self.adapter.applied_thrust
        old_last_action=self.adapter.last_action
        # Causal fractional action queue, including latencies longer than one
        # policy period. Graphs are truncated explicitly between BPTT windows.
        self._action_history.appendleft(action)
        delay=self.parameters['compute_latency_s']/self.dt
        low=delay.floor().long(); high=delay.ceil().long()
        history=torch.stack(tuple(self._action_history))
        env=torch.arange(self.batch_size,device=self.device)
        fraction=(delay-low)[:,None]
        applied=history[low,env]*(1-fraction)+history[high,env]*fraction
        # Energy-dependent authority; power model is an explicit approximation.
        battery=(1-.2*self.energy_used/self.parameters['battery_energy_j']).clamp(.5,1)
        applied=torch.cat((applied[:,:1]*battery[:,None],applied[:,1:]),-1)
        self.adapter.step(applied)
        self.adapter.state=torch.where(self.done.clone()[:,None],old_state,self.adapter.state)
        self.adapter.applied_thrust=torch.where(self.done.clone(),old_thrust,self.adapter.applied_thrust)
        self.adapter.last_action=torch.where(active[:,None],self.adapter.last_action,old_last_action)
        self.steps+=1; self.time_s=self.steps*self.dt
        self.episode_steps=self.episode_steps+active.long()
        self.episode_time_s=self.episode_steps*self.dt
        with torch.no_grad():
            displacement=(self.adapter.position-old_position).norm(dim=-1)
            self.travel+=displacement
            self.energy_used+=self.adapter.applied_thrust.clamp_min(0).pow(1.5)*self.dt*8*(~self.done)
            sweep=self.collision_checker.check(old_position,self.adapter.position,self.parameters['body_radius_m'])
            clearance=sweep['clearance']
            closer=clearance<self.minimum_clearance
            self.minimum_clearance_is_lower_bound=torch.where(closer,sweep['clearance_is_lower_bound'],self.minimum_clearance_is_lower_bound)
            self.minimum_clearance=torch.minimum(self.minimum_clearance,clearance)
            collision=(sweep['collision']|sweep['left_bounds'])&(~self.done)
            self.collided|=collision
            self.maximum_speed=torch.maximum(self.maximum_speed,self.adapter.velocity.norm(dim=-1))
            self.speed_limit_steps+=((self.adapter.velocity.norm(dim=-1)>self.speed_limit+1e-5)&(~self.done)).long()
            first=(self.adapter.position-self.targets[:,0]).norm(dim=-1)<self.guidance_config.waypoint_radius
            self.waypoint_index=torch.where(first&(~self.done),torch.ones_like(self.waypoint_index),self.waypoint_index)
            reached=((self.adapter.position-self.targets[:,1]).norm(dim=-1)<.35)&(self.adapter.velocity.norm(dim=-1)<.8)&(self.waypoint_index==1)&(~self.done)
            self.success|=reached&(~self.collided)
            terminated=collision|reached
            truncated=(self.episode_steps>=self.horizon)&(~terminated)&(~self.done)
            self.done|=terminated|truncated
        previous=self.previous_action
        self.previous_action=torch.where(active[:,None],action.detach(),self.previous_action)
        self.previous_command=self.previous_action
        losses=self.differentiable_loss(action,previous)
        reward=-losses['target_velocity'].detach().expand(self.batch_size)*self.dt
        # Per-environment PPO reward, distinct from the batch training loss.
        target=torch.where((self.waypoint_index==0)[:,None],self.targets[:,0],self.targets[:,1])
        progress=(old_position-target).norm(dim=-1)-(self.adapter.position-target).norm(dim=-1)
        reward=progress.detach()*3-.01-10*collision.float()+10*reached.float()
        reward=torch.where(active,reward,torch.zeros_like(reward))
        self._observation=self._observe()
        self.controller_memory=update_memory(self.controller_memory,self._observation,action.detach(),
                                             self.dt,stop_requested=stop_requested)
        self._observation.update(self.controller_memory.observation_state())
        return self._observation,reward,self.done.clone(),{'terminated':terminated,'truncated':truncated,
            'success':self.success.clone(),'collision':self.collided.clone(),'active':active}

    def detach(self):
        self.adapter.detach()
        self.previous_action=self.previous_action.detach()
        self._action_history=deque((action.detach() for action in self._action_history),maxlen=self._action_history.maxlen)
        self._observation={key:value.detach() for key,value in self._observation.items()}

    def known_grid(self,index):
        return self._known[index].detach().cpu().numpy().copy()

    def randomization_audit(self):
        return randomization_manifest(self.sampled_profiles)
