"""Fresh aligned voxel furniture fixture, separate from sealed NN cohorts."""
from itertools import product
from fractions import Fraction as F
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'bounded_service_v1'))
from sensor_renderer import AnalyticRenderer
from ares.geometry import voxel_mesh
from ares.contracts import validate_scene
class ClutterRenderer(AnalyticRenderer):
    def _boxes(self):
        return (((0,0,0),(1,12,4)),((11,0,0),(12,12,4)),
                ((0,0,0),(12,1,4)),((0,11,0),(12,12,4)),
                ((0,0,0),(12,12,1)),((0,0,3),(12,12,4)),
                ((7,5,1),(8,6,3)),((5,7,1),(6,8,2)),
                ((8,6,1),(9,9,3)))
    def occupied(self,cell):
        if any(not 0<=x<n for x,n in zip(cell,self.shape)):return True
        p=tuple((F(x)+F(1,2))*F(1,4) for x in cell)
        return any(all(F(a)<=x<F(b) for x,a,b in zip(p,lo,hi)) for lo,hi in self._boxes())

def make_scene(seed=90100):
    renderer=ClutterRenderer();shape=(60,60,20)
    occupied=[]
    for cell in product(*(range(n) for n in shape)):
        p=tuple((F(x)+F(1,2))*F(1,5) for x in cell)
        if any(all(F(a)<=x<F(b) for x,a,b in zip(p,lo,hi)) for lo,hi in renderer._boxes()):occupied.append(list(cell))
    scene={'schema_version':'ares.scene.v1','scene_id':f'learned-shield-prospective-shelves-v2:{seed}',
      'seed':seed,'units':'m','frame':'right_handed_z_up','license':{'spdx':'CC0-1.0','commercial_ok':True},
      'grid':{'resolution':.2,'origin':[0.,0.,0.],'shape':list(shape),'occupied':occupied},
      'collision_mesh':voxel_mesh(occupied),'rooms':[{'room_id':'room_1','label':'furniture_test','polygon':[[1,1],[11,1],[11,11],[1,11]]}],
      'doors':[],'surfaces':[{'material':'normal','voxel_indices':occupied}],
      'lighting':[{'room_id':'room_1','state':'lit'}],'spawn':[6.1,6.1,1.9]}
    validate_scene(scene);return scene,renderer
