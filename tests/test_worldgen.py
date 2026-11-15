"""Independent geometry and scene invariants for the Phase 1 generator."""

import hashlib
import itertools
import json
import math
import random
import unittest

from ares.contracts import validate_scene
from ares.geometry import voxel_mesh
from ares.worldgen import generate_scene


def triangle_cross(vertices, triangle):
    a, b, c = (vertices[index] for index in triangle)
    u = [b[i] - a[i] for i in range(3)]
    v = [c[i] - a[i] for i in range(3)]
    return [u[1]*v[2] - u[2]*v[1], u[2]*v[0] - u[0]*v[2],
            u[0]*v[1] - u[1]*v[0]]


class VoxelMeshTests(unittest.TestCase):
    def test_analytic_solids_area_bounds_and_outward_winding(self):
        # Solid rectangular cuboids have area 2(ab + ac + bc), regardless
        # of the internal faces between occupied voxels.
        for dimensions, resolution, origin in (
            ((1, 1, 1), 0.2, (0, 0, 0)),
            ((2, 1, 1), 0.5, (-2, -1, -3)),
            ((2, 3, 2), 0.2, (-1.4, 2.6, -0.8)),
        ):
            with self.subTest(dimensions=dimensions, origin=origin):
                cells = list(itertools.product(*(range(n) for n in dimensions)))
                mesh = voxel_mesh(cells, resolution=resolution, origin=origin)
                vertices = mesh['vertices']
                triangles = mesh['triangles']
                self.assertEqual(len(vertices), len(set(map(tuple, vertices))))
                expected_faces = 2 * sum(dimensions[i] * dimensions[j]
                                         for i, j in ((0, 1), (0, 2), (1, 2)))
                self.assertAlmostEqual(sum(math.sqrt(sum(c*c for c in triangle_cross(vertices, t))) / 2
                                           for t in triangles), expected_faces * resolution**2)
                for axis in range(3):
                    self.assertAlmostEqual(min(v[axis] for v in vertices), origin[axis])
                    self.assertAlmostEqual(max(v[axis] for v in vertices),
                                           origin[axis] + dimensions[axis] * resolution)
                for triangle in triangles:
                    self.assertEqual(len(set(triangle)), 3)
                    normal = triangle_cross(vertices, triangle)
                    axis = max(range(3), key=lambda i: abs(normal[i]))
                    coordinate = vertices[triangle[0]][axis]
                    if math.isclose(coordinate, origin[axis], abs_tol=1e-9):
                        self.assertLess(normal[axis], 0)
                    else:
                        self.assertAlmostEqual(coordinate, origin[axis] + dimensions[axis] * resolution)
                        self.assertGreater(normal[axis], 0)

    def test_disconnected_and_concave_solids_have_no_internal_faces(self):
        for cells, area in (([(0, 0, 0), (2, 0, 0)], 12),
                            ([(0, 0, 0), (1, 0, 0), (0, 1, 0)], 14)):
            mesh = voxel_mesh(cells, resolution=1)
            self.assertAlmostEqual(sum(math.sqrt(sum(c*c for c in triangle_cross(mesh['vertices'], t))) / 2
                                       for t in mesh['triangles']), area)
        self.assertEqual(voxel_mesh([]), {'vertices': [], 'triangles': []})

    def test_small_representable_voxels_keep_distinct_corners(self):
        resolution = 1e-14
        mesh = voxel_mesh([(0, 0, 0)], resolution=resolution)
        self.assertEqual(len(set(map(tuple, mesh['vertices']))), 8)
        for vertex in mesh['vertices']:
            for coordinate in vertex:
                self.assertIn(coordinate / resolution, (0, 1))
        area_in_voxels = sum(math.sqrt(sum((c / resolution**2)**2
                                           for c in triangle_cross(mesh['vertices'], t))) / 2
                             for t in mesh['triangles'])
        self.assertAlmostEqual(area_in_voxels, 6)

    def test_unrepresentable_coordinates_raise(self):
        for resolution, origin in ((0.2, (1e20, 0, 0)),
                                   (1e308, (1e308, 0, 0))):
            with self.subTest(resolution=resolution, origin=origin):
                with self.assertRaises((ValueError, TypeError)):
                    voxel_mesh([(0, 0, 0)], resolution=resolution, origin=origin)

    def test_mesh_rejects_malformed_geometry_arguments(self):
        for cells in ([(True, 0, 0)], [(0.5, 0, 0)], [(0, 0)], [(0, 0, 0, 0)]):
            with self.subTest(cells=cells), self.assertRaises((ValueError, TypeError)):
                voxel_mesh(cells)
        for resolution in (True, 0, -1, float('nan'), float('inf')):
            with self.subTest(resolution=resolution), self.assertRaises((ValueError, TypeError)):
                voxel_mesh([(0, 0, 0)], resolution=resolution)
        for origin in ((0, 0), (False, 0, 0), (0, 0, float('nan'))):
            with self.subTest(origin=origin), self.assertRaises((ValueError, TypeError)):
                voxel_mesh([(0, 0, 0)], origin=origin)


class WorldGeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = [(kind, split, seed, generate_scene(seed, kind=kind, split=split))
                      for kind in ('single_room', 'apartment', 'floor')
                      for split in ('train', 'validation', 'test')
                      for seed in (0, 17)]

    def test_corpus_validates_and_exercises_domain_variation(self):
        materials, lights, doors = set(), set(), set()
        for kind, split, seed, scene in self.corpus:
            with self.subTest(kind=kind, split=split, seed=seed):
                self.assertIsNone(validate_scene(scene))
                self.assertEqual(scene['seed'], seed)
                materials.update(s['material'] for s in scene['surfaces'] if s['voxel_indices'])
                lights.update(light['state'] for light in scene['lighting'])
                doors.update(door['state'] for door in scene['doors'])
                if kind == 'single_room':
                    self.assertEqual(len(scene['rooms']), 1)
                elif kind == 'apartment':
                    self.assertGreaterEqual(len(scene['rooms']), 3)
                    self.assertGreaterEqual(len({room['label'] for room in scene['rooms']}), 3)
                else:
                    self.assertGreaterEqual(len(scene['rooms']), 10)
                    self.assertTrue(any('corridor' in room['label'].lower() for room in scene['rooms']))
                # Closed leaves may block flight, but opening all doors must
                # expose one connected building rather than isolated boxes.
                reachable = {scene['rooms'][0]['room_id']}
                for _ in scene['rooms']:
                    for door in scene['doors']:
                        if reachable.intersection(door['room_ids']):
                            reachable.update(door['room_ids'])
                self.assertEqual(reachable, {r['room_id'] for r in scene['rooms']})
        self.assertEqual(materials, {'normal', 'shiny', 'black', 'glass', 'mirror'})
        self.assertEqual(lights, {'lit', 'dark'})
        self.assertEqual(doors, {'open', 'closed'})

    def test_determinism_split_separation_and_no_global_random_side_effect(self):
        state = random.getstate()
        ids = set()
        geometries = {}
        for kind, split, seed, scene in self.corpus:
            self.assertEqual(scene, generate_scene(seed, kind=kind, split=split))
            self.assertNotIn(scene['scene_id'], ids)
            ids.add(scene['scene_id'])
            geometries.setdefault(kind, set()).add(json.dumps(scene['grid'], sort_keys=True))
            # Partition only physical geometry, so changing optical finishes
            # or lighting cannot move an identical building into a held-out set.
            canonical_grid = json.dumps(scene['grid'], sort_keys=True,
                                        separators=(',', ':'), allow_nan=False).encode()
            bucket = int(hashlib.sha256(canonical_grid).hexdigest()[:8], 16) % 10
            expected_split = 'test' if bucket == 0 else 'validation' if bucket == 1 else 'train'
            self.assertEqual(split, expected_split, (kind, split, seed, bucket))
        self.assertEqual(random.getstate(), state)
        for kind, values in geometries.items():
            self.assertGreater(len(values), 3, kind)
        for kind in ('single_room', 'apartment', 'floor'):
            for seed in (0, 17):
                split_grids = {json.dumps(scene['grid'], sort_keys=True)
                               for scene_kind, _, scene_seed, scene in self.corpus
                               if scene_kind == kind and scene_seed == seed}
                self.assertEqual(len(split_grids), 3, (kind, seed))

    def test_shell_is_enclosed_and_spawn_has_body_clearance(self):
        for kind, split, seed, scene in self.corpus:
            with self.subTest(kind=kind, split=split, seed=seed):
                grid = scene['grid']
                occupied = set(map(tuple, grid['occupied']))
                nx, ny, nz = grid['shape']
                for axis, extent in enumerate((nx, ny, nz)):
                    other = [i for i in range(3) if i != axis]
                    for side in (0, extent - 1):
                        for a, b in itertools.product(range(grid['shape'][other[0]]),
                                                      range(grid['shape'][other[1]])):
                            cell = [0, 0, 0]
                            cell[axis], cell[other[0]], cell[other[1]] = side, a, b
                            self.assertIn(tuple(cell), occupied)
                resolution, origin, spawn = grid['resolution'], grid['origin'], scene['spawn']
                radius = 0.15
                indices = [range(math.floor((spawn[i]-radius-origin[i])/resolution),
                                 math.floor((spawn[i]+radius-origin[i])/resolution)+1)
                           for i in range(3)]
                for cell in itertools.product(*indices):
                    if cell not in occupied:
                        continue
                    distance_squared = sum(max(origin[i]+cell[i]*resolution-spawn[i],
                                               0, spawn[i]-(origin[i]+(cell[i]+1)*resolution))**2
                                           for i in range(3))
                    self.assertGreaterEqual(distance_squared, radius**2 - 1e-10)

    def test_realistic_rectangular_rooms_and_furniture(self):
        for kind, split, seed, scene in self.corpus:
            with self.subTest(kind=kind, split=split, seed=seed):
                grid = scene['grid']
                has_furniture = False
                for room in scene['rooms']:
                    polygon = room['polygon']
                    self.assertEqual(len(polygon), 4)
                    widths = [max(p[i] for p in polygon)-min(p[i] for p in polygon) for i in range(2)]
                    self.assertGreaterEqual(min(widths), 1.0)
                    self.assertLessEqual(max(widths), 60)
                    # An occupied cell well within a room above the floor
                    # indicates actual volumetric furnishing rather than its shell.
                    inset = [[min(p[0] for p in polygon)+0.4, min(p[1] for p in polygon)+0.4],
                             [max(p[0] for p in polygon)-0.4, max(p[1] for p in polygon)-0.4]]
                    for x, y, z in grid['occupied']:
                        point = [grid['origin'][i]+(c+0.5)*grid['resolution'] for i, c in enumerate((x,y,z))]
                        if (0.4 < point[2]-grid['origin'][2] < 1.4
                                and all(inset[0][i] < point[i] < inset[1][i] for i in range(2))):
                            has_furniture = True
                            break
                self.assertTrue(has_furniture)

    def test_doorway_dimensions_and_approaches(self):
        for kind, split, seed, scene in self.corpus:
            grid = scene['grid']
            occupied = set(map(tuple, grid['occupied']))
            for door in scene['doors']:
                with self.subTest(kind=kind, split=split, seed=seed, door=door['door_id']):
                    cells = set(map(tuple, door['voxel_indices']))
                    spans = [sorted({c[i] for c in cells}) for i in range(3)]
                    thickness_axis = 0 if len(spans[0]) == 1 else 1
                    width_axis = 1 - thickness_axis
                    self.assertEqual(len(spans[thickness_axis]), 1)
                    self.assertGreaterEqual(len(spans[width_axis])*grid['resolution'], 1.0 - 1e-9)
                    self.assertGreaterEqual(len(spans[2])*grid['resolution'], 1.8 - 1e-9)
                    self.assertEqual(len(cells), math.prod(len(span) for span in spans))
                    self.assertEqual(bool(cells & occupied), door['state'] == 'closed')
                    # Full-width clearance immediately on both sides, at a
                    # normal flight height, permits approach to a closed leaf.
                    z = spans[2][min(4, len(spans[2])-1)]
                    for side in (-1, 1):
                        for depth in (1, 2):
                            for width in spans[width_axis]:
                                point = [0, 0, z]
                                point[thickness_axis] = spans[thickness_axis][0] + side*depth
                                point[width_axis] = width
                                self.assertNotIn(tuple(point), occupied)

    def test_floor_units_have_private_interiors_and_corridor_entries(self):
        for kind, split, seed, scene in self.corpus:
            if kind != 'floor':
                continue
            with self.subTest(split=split, seed=seed):
                labels = {r['room_id']: r['label'] for r in scene['rooms']}
                units = {label.split('/')[0] for label in labels.values() if label != 'corridor'}
                entrances = set()
                for door in scene['doors']:
                    first, second = [labels[rid] for rid in door['room_ids']]
                    if 'corridor' in (first, second):
                        other = second if first == 'corridor' else first
                        entrances.add(other.split('/')[0])
                    else:
                        self.assertEqual(first.split('/')[0], second.split('/')[0],
                                         'private units must not have interconnecting doors')
                self.assertEqual(entrances, units)

    def test_invalid_generator_arguments(self):
        for seed in (True, False, 1.5, '1', None):
            with self.subTest(seed=seed), self.assertRaises((ValueError, TypeError)):
                generate_scene(seed)
        for kind in ('castle', '', None, True):
            with self.subTest(kind=kind), self.assertRaises((ValueError, TypeError)):
                generate_scene(0, kind=kind)
        for split in ('training', '', None, True):
            with self.subTest(split=split), self.assertRaises((ValueError, TypeError)):
                generate_scene(0, split=split)


if __name__ == '__main__':
    unittest.main()
