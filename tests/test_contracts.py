import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

from ares.contracts import ContractError, load_json, validate_profile, validate_scene


ROOT = Path(__file__).resolve().parents[1]


class ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene = load_json(ROOT / "ares/data/scenes/three_room.json")
        cls.profile = load_json(ROOT / "ares/data/configs/drone_nominal.json")

    def bad_scene(self, change, reason):
        scene = copy.deepcopy(self.scene)
        change(scene)
        with self.assertRaisesRegex(ContractError, reason):
            validate_scene(scene)

    def bad_profile(self, name, value, reason):
        profile = copy.deepcopy(self.profile)
        profile["parameters"][name] = value
        with self.assertRaisesRegex(ContractError, reason):
            validate_profile(profile)

    def test_fixtures_validate(self):
        self.assertIsNone(validate_scene(self.scene))
        self.assertIsNone(validate_profile(self.profile))

    def test_enclosed_shell_and_open_doorways(self):
        grid = self.scene["grid"]
        occupied = {tuple(c) for c in grid["occupied"]}
        nx, ny, nz = grid["shape"]
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    if x in (0,nx-1) or y in (0,ny-1) or z in (0,nz-1):
                        self.assertIn((x,y,z), occupied)
        radius = self.profile["parameters"]["body_radius_m"]["nominal"]
        for door in self.scene["doors"]:
            self.assertEqual(door["state"], "open")
            self.assertTrue(all(tuple(c) not in occupied for c in door["voxel_indices"]))
            ys = [c[1] for c in door["voxel_indices"]]
            self.assertGreater((max(ys)-min(ys)+1)*grid["resolution"], 2*radius)

    def test_mesh_is_exact_exposed_voxel_faces(self):
        occupied = {tuple(c) for c in self.scene["grid"]["occupied"]}
        expected = set()
        for cell in occupied:
            for axis in range(3):
                for sign in (-1,1):
                    neighbor = list(cell)
                    neighbor[axis] += sign
                    if tuple(neighbor) not in occupied:
                        expected.add((axis, cell[axis] + (sign == 1),
                                      tuple(cell[j] for j in range(3) if j != axis)))
        vertices = self.scene["collision_mesh"]["vertices"]
        actual = {}
        for triangle in self.scene["collision_mesh"]["triangles"]:
            points = [tuple(round(v/0.2) for v in vertices[i]) for i in triangle]
            axes = [j for j in range(3) if len({p[j] for p in points}) == 1]
            self.assertEqual(len(axes), 1)
            axis = axes[0]
            key = (axis, points[0][axis], tuple(min(p[j] for p in points) for j in range(3) if j != axis))
            actual[key] = actual.get(key,0) + 1
        self.assertEqual(set(actual), expected)
        self.assertTrue(all(count == 2 for count in actual.values()))

    def test_missing_and_unknown_keys(self):
        self.bad_scene(lambda s: s.pop("seed"), "scene: missing keys: seed")
        self.bad_scene(lambda s: s.update(extra=1), "scene: unknown keys: extra")

    def test_numbers_are_finite_and_not_booleans(self):
        self.bad_scene(lambda s: s.update(seed=True), "seed: expected integer")
        self.bad_scene(lambda s: s["spawn"].__setitem__(0,float("nan")), "spawn.*expected finite number")
        self.bad_scene(lambda s: s["grid"].update(resolution=True), "grid.resolution: expected finite number")

    def test_grid_indices(self):
        self.bad_scene(lambda s: s["grid"].update(resolution=0.1), "expected 0.2 metres")
        self.bad_scene(lambda s: s["grid"]["occupied"].append([32,0,0]), "voxel index out of bounds")
        self.bad_scene(lambda s: s["grid"]["occupied"].append(s["grid"]["occupied"][0]), "duplicate voxel index")
        self.bad_scene(lambda s: s["grid"]["shape"].__setitem__(0,0), "dimensions must be positive")

    def test_mesh_indices_and_geometry(self):
        self.bad_scene(lambda s: s["collision_mesh"]["triangles"][0].__setitem__(0,999999), "vertex index out of bounds")
        self.bad_scene(lambda s: s["collision_mesh"]["triangles"].__setitem__(0,[0,0,0]), "duplicate or degenerate triangle")
        self.bad_scene(lambda s: s["collision_mesh"]["vertices"][0].__setitem__(0,-1), "vertex outside grid bounds")

    def test_rooms(self):
        self.bad_scene(lambda s: s["rooms"][0].update(label=""), "expected nonempty string")
        self.bad_scene(lambda s: s["rooms"][1].update(room_id="room_1"), "duplicate room_id")
        self.bad_scene(lambda s: s["rooms"][0].update(polygon=[[0.2,0.2],[2,2.2],[0.2,2.2],[2,0.2]]), "self-intersecting polygon")
        self.bad_scene(lambda s: s["rooms"][0]["polygon"].reverse(), "positive counterclockwise area")

    def test_doors(self):
        self.bad_scene(lambda s: s["doors"][0].update(state="ajar"), "expected open or closed")
        self.bad_scene(lambda s: s["doors"][0].update(room_ids=["room_1","missing"]), "two distinct existing rooms")
        self.bad_scene(lambda s: s["doors"][0].update(voxel_indices=[[0,0,0]]), "open door voxels must be free")
        self.bad_scene(lambda s: s["doors"][0].update(state="closed"), "closed door voxels must be occupied")
        self.bad_scene(lambda s: s["doors"][0].update(voxel_indices=[[5,5,5]]), "door must border both referenced rooms")

    def test_surfaces_and_lighting(self):
        self.bad_scene(lambda s: s["surfaces"][0].update(material="silver"), "unknown depth material")
        self.bad_scene(lambda s: s["surfaces"][0]["voxel_indices"].pop(), "every occupied voxel requires a material")
        self.bad_scene(lambda s: s["lighting"].pop(), "every room requires lighting state")

    def test_spawn(self):
        self.bad_scene(lambda s: s.update(spawn=[-1,1,1]), "outside grid bounds")
        self.bad_scene(lambda s: s.update(spawn=[0.1,0.1,0.1]), "must lie in free space")

    def test_profile_bounds(self):
        self.bad_profile("mass_kg", {"nominal":0.5,"range":[0,1]}, "values must be positive")
        self.bad_profile("mass_kg", {"nominal":0.5,"range":[0.6,1]}, "outside randomization range")
        self.bad_profile("depth_width", {"nominal":32.5,"range":[32,33]}, "expected integer")
        self.bad_profile("depth_horizontal_fov_rad", {"nominal":1.5,"range":[1.0,math.pi]}, "below pi radians")
        self.bad_profile("depth_vertical_fov_rad", {"nominal":1.0,"range":[0.0,1.5]}, "values must be positive")
        self.bad_profile("depth_noise_std_m", {"nominal":0,"range":[-1,1]}, "values must be nonnegative")
        self.bad_profile("mass_kg", {"nominal":True,"range":[0.1,1]}, "expected finite number")

    def test_inertia(self):
        self.bad_profile("inertia_kg_m2", {"nominal":[1,1,3],"range":[[1,1],[1,1],[3,3]]}, "inertia violates triangle inequalities")
        self.bad_profile("inertia_kg_m2", {"nominal":[1,1,1],"range":[[0.1,2],[0.1,2],[0.1,2]]}, "randomized inertia can violate triangle inequalities")

    def test_json_reader(self):
        for text, reason in (("[]", "expected object"), ('{"x":1,"x":2}', "duplicate key: x"),
                             ('{"x":NaN}', "nonfinite constant: NaN"), ('{"x":', "json:")):
            with self.subTest(text=text), tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/"input.json"
                path.write_text(text)
                with self.assertRaisesRegex(ContractError, reason):
                    load_json(path)


if __name__ == "__main__":
    unittest.main()
