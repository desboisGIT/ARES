import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parent))
from adapter import command_to_anchor,feasible_fraction_from_public_set,word_intervention
from verified_backup_point_v2 import backup_point
class AdapterTests(unittest.TestCase):
    def test_command_inverse_tilt_and_heading(self):
        for yaw in (0.,.01,-.02):
            p=SimpleNamespace(p=(6.,6.,2.),v=(.01,-.02,.005),q=(0.,0.,math.sin(yaw/2),math.cos(yaw/2)))
            target=(6.05,5.96,2.03)
            command=backup_point(p.p,p.v,p.q,target,5.)['command']
            anchor=command_to_anchor(command,p,5.)['anchor']
            self.assertLess(max(abs(x-y) for x,y in zip(anchor,target)),2e-6)
    def test_candidate_screen_not_acceptance(self):
        row={'state_bounds':[(-.01,.01),(-.01,.01)]}
        result=feasible_fraction_from_public_set({'channels':{'planar_x':row}}, {'planar_x':np.eye(2)},[2.,0.,0.])
        self.assertGreater(result,.45);self.assertLess(result,.5)
    def test_sensor_snapshot_does_not_follow_future_plant_state(self):
        import torch
        from gaussian_observer import SensorSnapshot
        plant=SimpleNamespace(parameters={},state=torch.zeros((1,13)))
        plant.state[0,6]=1
        snapshot=SensorSnapshot(plant)
        plant.state[0,0]=9
        self.assertEqual(float(snapshot.position[0,0]),0.)
        self.assertEqual(float(snapshot.quaternion[0,3]),1.)
    def test_invalid_and_exact_word_interventions(self):
        p=SimpleNamespace(p=(0,0,0),v=(0,0,0),q=(0,0,0,1))
        with self.assertRaises(ValueError):command_to_anchor([float('nan'),0,0,0],p,5)
        self.assertFalse(word_intervention([5.,0,0,0],[5.+1e-9,0,0,0]))
        self.assertTrue(word_intervention([5.,0,0,0],[5.001,0,0,0]))
if __name__=='__main__':unittest.main(verbosity=2)
