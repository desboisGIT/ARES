"""End-to-end evidence checks without substituting for the full corpus gate."""

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from ares.cli import main
from ares.contracts import load_json, validate_scene
from ares.phase1 import run_phase1
from ares.worldgen import generate_scene


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / 'ares/data/recorded_maps/synthetic_enclosed_room.json'


def digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


class Phase1PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        cls.output = Path(cls.directory.name)
        cls.first = run_phase1(count=6, seed=1741, workers=1, output_dir=cls.output / 'first')
        cls.second = run_phase1(count=6, seed=1741, workers=1, output_dir=cls.output / 'second')
        cls.manifest = load_json(cls.output / 'first/corpus_manifest.json')

    def test_small_corpus_is_evidence_without_full_or_human_pass(self):
        result = self.first
        self.assertEqual(result['requested_count'], 6)
        self.assertEqual(result['valid_count'], 6)
        self.assertTrue(result['checks']['all_requested_scenes_valid'])
        self.assertFalse(result['checks']['at_least_10000_valid_scenes'])
        self.assertFalse(result['software_passed'])
        self.assertFalse(result['phase_passed'])
        self.assertEqual(result['human_review'], {'passed': False, 'status': 'pending_human_review'})
        self.assertEqual(result['counts_by_kind'], {'single_room': 2, 'apartment': 2, 'floor': 2})
        self.assertEqual(result['counts_by_split'], {'train': 4, 'validation': 1, 'test': 1})
        self.assertEqual(result['failures'], [])
        self.assertEqual(result['importer']['evidence'], 'synthetic_adapter_fixture_only')
        self.assertEqual(self.manifest['count'], 6)
        self.assertEqual(len(self.manifest['cases']), 6)
        self.assertEqual(self.manifest['corpus_id'], result['corpus_id'])
        self.assertEqual(result['gallery']['count'], 4)
        self.assertFalse((self.output / 'first/review/floor_plan_review.json').exists())

    def test_corpus_and_gallery_identity_exclude_run_timing(self):
        other = load_json(self.output / 'second/corpus_manifest.json')
        self.assertEqual(self.manifest, other)
        self.assertEqual(self.first['corpus_id'], self.second['corpus_id'])
        self.assertEqual(self.first['gallery']['gallery_id'], self.second['gallery']['gallery_id'])
        identity = dict(self.manifest)
        corpus_id = identity.pop('corpus_id')
        self.assertEqual(digest(identity), corpus_id)
        self.assertTrue(all('seconds' not in case for case in identity['cases']))

    def test_manifest_hashes_bind_actual_scene_and_identity_independent_content(self):
        for case in self.manifest['cases']:
            with self.subTest(scene_id=case['scene_id']):
                scene = generate_scene(case['seed'], case['kind'], case['split'])
                validate_scene(scene)
                self.assertEqual(case['sha256'], digest(scene))
                content = {key: value for key, value in scene.items() if key not in ('scene_id', 'seed')}
                self.assertEqual(case['content_sha256'], digest(content))
                self.assertNotEqual(case['sha256'], case['content_sha256'])
        gallery = self.first['gallery']
        by_id = {case['scene_id']: case for case in self.manifest['cases']}
        self.assertTrue(set(gallery['scene_ids']) <= set(by_id))
        for scene_id, scene_hash in gallery['source_scene_sha256'].items():
            self.assertEqual(scene_hash, by_id[scene_id]['sha256'])
            self.assertEqual(by_id[scene_id]['split'], 'train')

    def test_canonical_generator_bytes_survive_python_hash_randomization(self):
        code = ('import json; from ares.worldgen import generate_scene; '
                'print(json.dumps(generate_scene(1741,"apartment","validation"),'
                'sort_keys=True,separators=(",",":"),allow_nan=False))')
        outputs = []
        for hash_seed in ('1', '982451653'):
            environment = dict(os.environ, PYTHONHASHSEED=hash_seed)
            completed = subprocess.run([sys.executable, '-c', code], cwd=ROOT, env=environment,
                                       capture_output=True, check=True, timeout=20)
            self.assertEqual(completed.stderr, b'')
            outputs.append(completed.stdout)
        self.assertEqual(outputs[0], outputs[1])
        validate_scene(json.loads(outputs[0]))

    def invoke_cli(self, arguments):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = main(arguments)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_cli_generate_and_import_produce_valid_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            generated = directory / 'generated.json'
            code, stdout, stderr = self.invoke_cli([
                'generate', '--seed', '1741', '--kind', 'apartment', '--split', 'validation',
                '--output', str(generated)])
            self.assertEqual((code, stderr), (0, ''))
            self.assertTrue(json.loads(stdout)['valid'])
            validate_scene(load_json(generated))
            imported = directory / 'imported.json'
            code, stdout, stderr = self.invoke_cli(['import-map', str(FIXTURE), '--output', str(imported)])
            self.assertEqual((code, stderr), (0, ''))
            self.assertTrue(json.loads(stdout)['valid'])
            scene = load_json(imported)
            validate_scene(scene)
            provenance = load_json(directory / 'imported_provenance.json')
            self.assertEqual(provenance['recording_id'], scene['scene_id'])
            self.assertEqual(provenance['source_cell_count'], 125)
            self.assertEqual(provenance['unknown_cell_count'], 0)
            self.assertEqual(provenance['scene_occupied_count'], len(scene['grid']['occupied']))
            self.assertEqual(provenance['default_material_assumption'], 'normal')
            self.assertEqual(provenance['unknown_policy'], 'occupied')
            self.assertNotIn('provenance', scene)

    def test_cli_negative_count_is_input_error_without_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'result.json'
            code, stdout, stderr = self.invoke_cli(['phase1', '--count', '-1', '--output', str(output)])
            self.assertEqual(code, 2)
            self.assertEqual(stdout, '')
            self.assertIn('positive integers', stderr)
            self.assertFalse(output.exists())
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == '__main__':
    unittest.main()
