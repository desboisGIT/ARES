import copy
from pathlib import Path
import tempfile
import unittest

from ares.contracts import ContractError, load_json, validate_scene
from ares.importer import (import_recorded_map, import_recorded_map_with_provenance,
                           load_recorded_map)


FIXTURE = Path(__file__).resolve().parents[1] / 'ares/data/recorded_maps/synthetic_enclosed_room.json'


def two_rooms(state):
    recording = load_json(FIXTURE)
    recording['grid'] = {'resolution': 0.2, 'origin': [0, 0, 0], 'shape': [7, 5, 5],
                         'cells': [[x, y, z, 2 if x in (0, 3, 6) or y in (0, 4) or z in (0, 4) else 1]
                                   for x in range(7) for y in range(5) for z in range(5)]}
    recording['rooms'] = [
        {'room_id': 'left', 'label': 'Synthetic left', 'polygon': [[.2,.2],[.6,.2],[.6,.8],[.2,.8]]},
        {'room_id': 'right', 'label': 'Synthetic right', 'polygon': [[.8,.2],[1.2,.2],[1.2,.8],[.8,.8]]}]
    recording['doors'] = [{'door_id': 'door', 'room_ids': ['left', 'right'],
                           'state': state, 'voxel_indices': [[3,2,1]]}]
    if state == 'open':
        next(row for row in recording['grid']['cells'] if row[:3] == [3,2,1])[3] = 1
    recording['lighting'] = [{'room_id': rid, 'state': 'lit'} for rid in ('left', 'right')]
    recording['spawn'] = [.3,.5,.3]
    return recording


class ImporterTests(unittest.TestCase):
    def setUp(self):
        self.recording = load_json(FIXTURE)

    def test_translated_enclosed_fixture_and_immutability(self):
        original = copy.deepcopy(self.recording)
        scene = import_recorded_map(self.recording)
        validate_scene(scene)
        self.assertEqual(scene, load_recorded_map(FIXTURE))
        self.assertEqual(scene, import_recorded_map(self.recording))
        self.assertEqual(self.recording, original)
        self.assertEqual(scene['grid']['origin'], [12.4, -8.2, 1.6])
        occupied = {tuple(c) for c in scene['grid']['occupied']}
        self.assertEqual(len(occupied), 98)
        self.assertTrue(all((x,y,z) in occupied for x in range(5) for y in range(5)
                            for z in range(5) if 0 in (x,y,z) or 4 in (x,y,z)))
        self.assertGreater(min(v[0] for v in scene['collision_mesh']['vertices']), 12)
        scene['rooms'][0]['label'] = 'changed'
        self.assertEqual(self.recording, original)

    def test_unknown_and_missing_are_occupied(self):
        rows = self.recording['grid']['cells']
        rows.remove(next(r for r in rows if r[:3] == [1,1,1]))
        next(r for r in rows if r[:3] == [1,1,2])[3] = 0
        scene, provenance = import_recorded_map_with_provenance(self.recording)
        self.assertIn([1,1,1], scene['grid']['occupied'])
        self.assertIn([1,1,2], scene['grid']['occupied'])
        self.assertEqual(provenance['unknown_cell_count'], 2)
        self.assertEqual(provenance['explicit_unknown_count'], 1)
        self.assertEqual(provenance['missing_cell_count'], 1)
        self.assertEqual(provenance['source_cell_count'], 124)
        self.assertEqual(provenance['scene_occupied_count'], 100)
        self.assertNotIn('provenance', scene)

    def test_closed_and_open_door_metadata_and_collision(self):
        for state in ('open', 'closed'):
            with self.subTest(state=state):
                recording = two_rooms(state)
                scene = import_recorded_map(recording)
                self.assertEqual(scene['doors'], recording['doors'])
                self.assertEqual([3,2,1] in scene['grid']['occupied'], state == 'closed')

    def test_open_door_cannot_clear_unknown(self):
        recording = two_rooms('open')
        recording['grid']['cells'].remove(next(r for r in recording['grid']['cells'] if r[:3] == [3,2,1]))
        with self.assertRaisesRegex(ContractError, 'open door voxels must be free'):
            import_recorded_map(recording)

    def test_closed_door_cannot_override_recorded_free(self):
        recording = two_rooms('open')
        recording['doors'][0]['state'] = 'closed'
        with self.assertRaisesRegex(ContractError, 'closed door voxels must be occupied'):
            import_recorded_map(recording)

    def test_malformed_exports(self):
        changes = [
            (lambda r: r['grid']['cells'].append(r['grid']['cells'][0]), 'more entries'),
            (lambda r: r['grid']['cells'].__setitem__(1, r['grid']['cells'][0]), 'duplicate voxel'),
            (lambda r: r['grid']['cells'][0].__setitem__(3, 3), 'expected 0'),
            (lambda r: r['grid']['cells'][0].__setitem__(0, True), 'expected integer'),
            (lambda r: r['grid']['cells'][0].__setitem__(0, -1), 'out of bounds'),
            (lambda r: r['grid'].__setitem__('resolution', .1), 'resampling'),
            (lambda r: r['grid'].__setitem__('shape', [1001,1001,1]), 'resource limit'),
            (lambda r: r['grid']['origin'].__setitem__(0, float('inf')), 'finite'),
            (lambda r: r.__setitem__('spawn', [1e308,0,0]), 'outside grid'),
            (lambda r: r.__setitem__('frame', 'camera'), 'right_handed_z_up'),
            (lambda r: r.pop('rooms'), 'missing keys'),
            (lambda r: r.__setitem__('rooms', []), 'must be nonempty'),
            (lambda r: r.pop('doors'), 'missing keys'),
            (lambda r: r.__setitem__('default_material', 'guess'), 'unknown depth material'),
            (lambda r: r.__setitem__('surprise', 1), 'unknown keys'),
        ]
        for change, reason in changes:
            with self.subTest(reason=reason):
                recording = copy.deepcopy(self.recording)
                change(recording)
                with self.assertRaisesRegex(ContractError, reason):
                    import_recorded_map(recording)

    def test_json_duplicate_keys_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'map.json'
            path.write_text('{"schema_version":1,"schema_version":2}')
            with self.assertRaisesRegex(ContractError, 'duplicate key'):
                load_recorded_map(path)


if __name__ == '__main__':
    unittest.main()
