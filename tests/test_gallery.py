import copy
import json
from pathlib import Path
import tempfile
import unittest

from ares.contracts import ContractError
from ares.gallery import build_gallery, validate_review


class GalleryTests(unittest.TestCase):
    def setUp(self):
        self.scene = json.loads((Path(__file__).parents[1] / 'ares/data/scenes/three_room.json').read_text())
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)

    def gallery(self, count=50):
        scenes = []
        for i in range(count):
            s = copy.deepcopy(self.scene)
            s['scene_id'] = f'plan-{i}'
            scenes.append(s)
        return build_gallery(scenes, self.path, seed=12)

    def review(self, manifest):
        return dict(schema_version='ares.floor_plan_review.v1', gallery_id=manifest['gallery_id'], reviewer='Human reviewer', reviewed_at='2026-11-12', source_scene_sha256=manifest['source_scene_sha256'], decisions=[dict(scene_id=sid, plausible=True, note='') for sid in manifest['scene_ids']])

    def test_escaping_and_no_initial_choices(self):
        self.scene['scene_id'] = '<script>alert("bad")</script>'
        self.scene['rooms'][0]['label'] = '<img src=x onerror=alert(1)>'
        build_gallery([self.scene], self.path, seed=1)
        page = (self.path / 'index.html').read_text()
        self.assertNotIn('<img src=x', page)
        self.assertNotIn('<script>alert', page)
        self.assertIn('&lt;img', page)
        self.assertNotIn(' checked', page)
        self.assertEqual(page.count('<svg '), 1)
        self.assertIn('Spawn location', page)
        self.assertIn('aria-expanded="false"', page)
        self.assertIn('article.expanded svg{height:80vh}', page)

    def test_review_strictness(self):
        manifest = self.gallery()
        good = self.review(manifest)
        self.assertTrue(validate_review(good, manifest)['passed'])
        for field, value in [('gallery_id', 'wrong'), ('reviewer', ''), ('source_scene_sha256', {})]:
            bad = copy.deepcopy(good)
            bad[field] = value
            with self.assertRaises(ContractError):
                validate_review(bad, manifest)
        for decisions in [good['decisions'][:-1], good['decisions'] + [good['decisions'][0]]]:
            bad = copy.deepcopy(good)
            bad['decisions'] = decisions
            with self.assertRaises(ContractError):
                validate_review(bad, manifest)
        bad = copy.deepcopy(good)
        bad['decisions'][0]['plausible'] = False
        self.assertFalse(validate_review(bad, manifest)['passed'])
        bad['decisions'][0]['plausible'] = 1
        with self.assertRaises(ContractError):
            validate_review(bad, manifest)
        bad = dict(good, extra='no')
        with self.assertRaises(ContractError):
            validate_review(bad, manifest)

    def test_under_fifty_cannot_pass(self):
        manifest = self.gallery(1)
        self.assertFalse(validate_review(self.review(manifest), manifest)['passed'])

    def test_generated_scene_type(self):
        self.scene['scene_id'] = 'world-v1:held_out:warehouse:42'
        build_gallery([self.scene], self.path, seed=42)
        self.assertIn('Type: warehouse', (self.path / 'index.html').read_text())
