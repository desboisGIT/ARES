"""Immutable hard-witness lease snapshot, expensive validation at preflight."""
from pathlib import Path
from dataclasses import dataclass,asdict
from fractions import Fraction as F
import sys,json,hashlib,math,struct
BASE=Path(__file__).resolve().parent.parent
def rounded(x,upper):
 y=float(x)
 while (F.from_float(y)<x if upper else F.from_float(y)>x):y=math.nextafter(y,math.inf if upper else -math.inf)
 return y

def canonical(x):
 if isinstance(x,F):return str(x)
 if isinstance(x,dict):return {str(k):canonical(v) for k,v in x.items()}
 if isinstance(x,(tuple,list)):return [canonical(v) for v in x]
 return x
def snapshot_digest(shape,resolution,origin,start_tick,through_tick,manifest):
 header=dict(schema='ares.immutable.map.lease.v3',shape=shape,resolution=resolution,origin=origin,start_tick=start_tick,through_tick=through_tick)
 state=hashlib.sha256(json.dumps(canonical(header),sort_keys=True,separators=(',',':')).encode())
 for cell,sha in manifest:state.update(struct.pack('>qqq',*cell)+bytes.fromhex(sha))
 return state.hexdigest()

@dataclass(frozen=True)
class MapLeaseSnapshot:
 shape:tuple
 resolution:float
 origin:tuple
 free:frozenset
 start_tick:int
 through_tick:int
 digest:str
 witness_manifest:tuple
 witness_records:tuple=()
 immutable:bool=True
 def __post_init__(self):
  grid=(tuple(self.shape),float(self.resolution),tuple(self.origin))
  if type(self.free) is not frozenset or type(self.witness_manifest) is not tuple or type(self.witness_records) is not tuple or self.immutable is not True:raise ValueError('immutable snapshot containers required')
  if self.free!=frozenset(c for c,_ in self.witness_manifest) or self.free!=frozenset(r[0] for r in self.witness_records) or any(r[1]!=grid for r in self.witness_records):raise ValueError('snapshot witness/grid binding differs')
  payload=dict(shape=grid[0],resolution=grid[1],origin=grid[2],start_tick=self.start_tick,through_tick=self.through_tick,witness_manifest=self.witness_manifest)
  expected=snapshot_digest(grid[0],grid[1],grid[2],self.start_tick,self.through_tick,self.witness_manifest)
  if expected!=self.digest:raise ValueError('snapshot metadata digest differs')
 @classmethod
 def capture(cls,known_map,*,start_tick,through_tick):
  if type(start_tick) is not int or type(through_tick) is not int or not 0<=start_tick<=through_tick:raise ValueError('ordered global lease ticks required')
  # Both endpoints outward: reject rather than enlarge any source lease.
  start_lo=rounded(F(start_tick,100),False);start_hi=rounded(F(start_tick,100),True);end_lo=rounded(F(through_tick,100),False);end_hi=rounded(F(through_tick,100),True)
  grid=(tuple(known_map.shape),float(known_map.resolution),tuple(known_map.origin));cells=[];manifest=[];retained=[]
  for cell in sorted(known_map.free):
   if all(known_map.known_free(cell,time_s=t) for t in [start_lo,start_hi,end_lo,end_hi]):
    ray,witness,optical=known_map.free[cell]
    raw=dict(cell=cell,grid=grid,ray=asdict(ray),witness=asdict(witness),optical_range=optical)
    entry=hashlib.sha256(json.dumps(canonical(raw),sort_keys=True,separators=(',',':')).encode()).hexdigest();cells.append(tuple(cell));manifest.append((tuple(cell),entry));retained.append((tuple(cell),grid,ray,witness,optical))
  payload=dict(shape=tuple(known_map.shape),resolution=known_map.resolution,origin=tuple(known_map.origin),start_tick=start_tick,through_tick=through_tick,witness_manifest=manifest)
  digest=snapshot_digest(tuple(known_map.shape),float(known_map.resolution),tuple(known_map.origin),start_tick,through_tick,manifest)
  return cls(tuple(known_map.shape),float(known_map.resolution),tuple(known_map.origin),frozenset(cells),start_tick,through_tick,digest,tuple(manifest),tuple(retained))
 def covers(self,start_tick,through_tick):return self.start_tick<=start_tick<=through_tick<=self.through_tick
 def known_free(self,cell,*,time_s):
  if not math.isfinite(time_s):return False
  return rounded(F(self.start_tick,100),False)<=time_s<=rounded(F(self.through_tick,100),True) and tuple(cell) in self.free
 def epoch_view(self,epoch):return EpochSnapshotView(self,epoch)

@dataclass(frozen=True)
class EpochSnapshotView:
 snapshot:MapLeaseSnapshot
 epoch:int
 def __post_init__(self):
  if not self.snapshot.start_tick<=self.epoch<=self.snapshot.through_tick:raise ValueError('epoch outside immutable lease')
  object.__setattr__(self,'_relative_start',rounded(F(self.snapshot.start_tick-self.epoch,100),True));object.__setattr__(self,'_relative_end',rounded(F(self.snapshot.through_tick-self.epoch,100),False))
 @property
 def free(self):return self.snapshot.free
 @property
 def shape(self):return self.snapshot.shape
 @property
 def resolution(self):return self.snapshot.resolution
 @property
 def origin(self):return self.snapshot.origin
 def known_free(self,cell,*,time_s):
  # Cached conservative binary64 lease bounds; no ray/contract/Fraction work.
  return self._relative_start<=time_s<=self._relative_end and tuple(cell) in self.snapshot.free

class MapLeaseCache:
 """Validate immutable ray entries once; make cheap immutable deadline views."""
 def __init__(self,known_map):
  self.original=known_map;self._grid=(tuple(known_map.shape),float(known_map.resolution),tuple(known_map.origin));self._seen={};self._valid={};self.entries_validated=0;self._version=0;self._snapshots={};self.refresh()
 def _check_grid(self):
  if (tuple(self.original.shape),float(self.original.resolution),tuple(self.original.origin))!=self._grid:raise ValueError('map grid metadata changed after witness cache creation')
 def refresh(self):
  self._check_grid()
  for cell,entry in self.original.free.items():
   if self._seen.get(cell) is entry:continue
   self._seen[cell]=entry;ray,witness,optical=entry;c=ray.state.contract
   previous=self._valid.get(tuple(cell))
   if previous and previous[0]<=c.capture_time_s and c.expires_s<=previous[1]:continue  # Retain the older stronger immutable lease.
   # Same source-kind/finiteness/geometry proof as trusted WitnessMap entry.
   # Lease checking is done ONCE; later snapshots only compare cached bounds.
   if not self.original.known_free(cell,time_s=c.capture_time_s) or not witness.free_witness:continue
   raw=dict(cell=cell,grid=self._grid,ray=asdict(ray),witness=asdict(witness),optical_range=optical)
   sha=hashlib.sha256(json.dumps(canonical(raw),sort_keys=True,separators=(',',':')).encode()).hexdigest()
   self._valid[tuple(cell)]=(float(c.capture_time_s),float(c.expires_s),sha,entry)
   self.entries_validated+=1;self._version+=1
  return self
 def snapshot(self,*,start_tick,through_tick):
  self._check_grid()
  if type(start_tick) is not int or type(through_tick) is not int or not 0<=start_tick<=through_tick:raise ValueError('ordered global ticks required')
  cached=self._snapshots.get(through_tick)
  if cached and cached[1]==self._version and cached[0].covers(start_tick,through_tick) and (cached[2] is None or start_tick<cached[2]):return cached[0]
  start=rounded(F(start_tick,100),False);end=rounded(F(through_tick,100),True)
  manifest=tuple((cell,v[2]) for cell,v in sorted(self._valid.items()) if v[0]<=start and end<=v[1]);free=frozenset(cell for cell,_ in manifest)
  future=[F.from_float(v[0])*100 for v in self._valid.values() if start<v[0] and end<=v[1]]
  next_tick=None if not future else min(-((-v.numerator)//v.denominator) for v in future)
  if cached and cached[0].witness_manifest==manifest and cached[0].covers(start_tick,through_tick):
   self._snapshots[through_tick]=(cached[0],self._version,next_tick);return cached[0]
  sha=snapshot_digest(self._grid[0],self._grid[1],self._grid[2],start_tick,through_tick,manifest)
  result=MapLeaseSnapshot(self._grid[0],self._grid[1],self._grid[2],free,start_tick,through_tick,sha,manifest,tuple((cell,self._grid,*self._valid[cell][3]) for cell,_ in manifest))
  self._snapshots[through_tick]=(result,self._version,next_tick);return result
