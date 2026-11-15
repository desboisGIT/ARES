"""Declared all-direction bounded FIRST-HIT preflight prior; no truth labels."""
from itertools import product
from fractions import Fraction as F
import math,random
from bounded_service import WitnessMap
from reservation import free_grid_reserve

def observe_prior(supplier,state,*,time_s=0.,half_width_cells=6):
    result=WitnessMap();anchor=tuple(math.floor((p-o)/result.resolution) for p,o in zip(state.p,result.origin))
    candidates=set(product(*(range(max(0,a-half_width_cells),min(n,a+half_width_cells+1)) for a,n in zip(anchor,result.shape))))
    candidates.update(((0,24,8),(47,24,8),(36,24,8),(40,24,8)))
    for ray_id,cell in enumerate(sorted(candidates)):
        # The old artificial x28 skip is removed; no free label is invented.
        center=tuple(o+(i+.5)*result.resolution for o,i in zip(result.origin,cell))
        direction=tuple(c-p for c,p in zip(center,state.p))
        if sum(d*d for d in direction)<1e-20:continue
        ray=supplier.ray(state,direction,ray_id=ray_id)
        result.observe(ray,cell,optical_range=1.,time_s=time_s)
    return result

def route_in_prior(points,snapshot,padding=(F(3,10),F(3,10),F(1,2))):
    for start,end in zip(points[:-1],points[1:]):
        a=[F.from_float(float(x)) for x in start];b=[F.from_float(float(x)) for x in end]
        count=max(1,math.ceil(math.dist(start,end)/.025))
        for j in range(count):
            left=[x+F(j,count)*(y-x) for x,y in zip(a,b)];right=[x+F(j+1,count)*(y-x) for x,y in zip(a,b)]
            lower=[min(x,y)-r for x,y,r in zip(left,right,padding)];upper=[max(x,y)+r for x,y,r in zip(left,right,padding)]
            if not free_grid_reserve(lower,upper,origin=snapshot.origin,shape=snapshot.shape,resolution=snapshot.resolution,free_cells=snapshot.free)['reserved']:return False
    return True

def generate_pair(anchor,seed,snapshot):
    rng=random.Random(f'ares.phase4.prior_pairs.v1:{seed}');rejected=0
    for draw in range(256):
        first=[anchor[0]+rng.uniform(.50,.65),anchor[1]+rng.uniform(.15,.35),anchor[2]+rng.uniform(-.025,.025)]
        second=[first[0]+rng.uniform(.30,.55),first[1]+rng.uniform(.85,1.),first[2]+rng.uniform(-.025,.075)]
        if math.dist(anchor,first)<.51 or math.dist(first,second)<.90:rejected+=1;continue
        if route_in_prior([anchor,first,second],snapshot):
            return [first,second],{'seed':seed,'draw':draw,'preflight_geometry_rejections':rejected,'padding_xyz_m':[.3,.3,.5],
              'uses_public_observed_free_cells_only':True,'mission_outcomes_not_used':True,'leg_lengths_m':[math.dist(anchor,first),math.dist(first,second)]}
        rejected+=1
    raise ValueError('no publicly witnessed nontrivial task pair in declared prior volume')
