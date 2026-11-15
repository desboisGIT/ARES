import http.client
import json
import threading
import unittest

from ares.viewer import build_server


class FakeSession:
    def __init__(self):
        self.action = None
        self.resets = 0
        self.index = 0
        self.started = False

    def start(self):
        self.started = True

    def state(self):
        return dict(mode='replay', actor='ui_test', frame={}, frame_count=2,
                    frame_index=self.index, done=False, result=None,
                    controls=dict(max_speed=1, max_yaw_rate=1))

    def set_input(self, action):
        self.action = action

    def reset(self):
        self.resets += 1

    def seek(self, index):
        if index >= 2:
            raise ValueError('out of range')
        self.index = index


class ViewerTests(unittest.TestCase):
    def setUp(self):
        self.session = FakeSession()
        self.server = build_server(self.session, port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.cleanup)

    def cleanup(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(2)

    def request(self, method, path, payload=None, raw=None, headers=None):
        connection = http.client.HTTPConnection(*self.server.server_address, timeout=2)
        body = raw if raw is not None else json.dumps(payload) if payload is not None else None
        connection.request(method, path, body, headers or {})
        response = connection.getresponse()
        result = response.status, dict(response.getheaders()), response.read()
        connection.close()
        return result

    def test_state_and_packaged_page(self):
        status, headers, body = self.request('GET', '/api/state')
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)['actor'], 'ui_test')
        self.assertEqual(headers['Cache-Control'], 'no-store')
        status, headers, body = self.request('GET', '/')
        self.assertEqual(status, 200)
        self.assertIn(b'first-person simulation', body)
        self.assertIn("connect-src 'self'", headers['Content-Security-Policy'])
        self.assertNotIn('Access-Control-Allow-Origin', headers)

    def test_action_reset_seek(self):
        self.assertEqual(self.request('POST', '/api/start', {})[0], 200)
        self.assertTrue(self.session.started)
        action = dict(velocity_body=[1, -1, 0], yaw_rate=.5, land=False)
        self.assertEqual(self.request('POST', '/api/action', action)[0], 200)
        self.assertEqual(self.session.action, action)
        self.assertEqual(self.request('POST', '/api/reset', {})[0], 200)
        self.assertEqual(self.session.resets, 1)
        self.assertEqual(self.request('POST', '/api/seek', {'frame_index': 1})[0], 200)
        self.assertEqual(self.session.index, 1)
        self.assertEqual(self.request('POST', '/api/seek', {'frame_index': 2})[0], 400)

    def test_invalid_actions_do_not_reach_session(self):
        base = dict(velocity_body=[0, 0, 0], yaw_rate=0, land=False)
        cases = [dict(base, extra=0), dict(base, velocity_body=[True, 0, 0]),
                 dict(base, yaw_rate=True), dict(base, land=1),
                 dict(base, velocity_body=[0, 0]), dict(base, yaw_rate='1'),
                 dict(base, yaw_rate=float('nan')), dict(base, yaw_rate=float('inf'))]
        for payload in cases:
            with self.subTest(payload=payload):
                self.assertEqual(self.request('POST', '/api/action', payload)[0], 400)
        self.assertIsNone(self.session.action)
        self.assertEqual(self.request('POST', '/api/reset', {'extra': 1})[0], 400)
        self.assertEqual(self.request('POST', '/api/seek', {'frame_index': True})[0], 400)
        self.assertEqual(self.request('POST', '/api/reset', raw='{')[0], 400)
        self.assertEqual(self.request('POST', '/api/reset', raw='x'*16385)[0], 413)

    def test_routes_and_origin(self):
        for path in ['/etc/passwd', '/../pyproject.toml', '/api/state?file=x', '/data/viewer/index.html']:
            self.assertEqual(self.request('GET', path)[0], 404)
        self.assertEqual(self.request('POST', '/unknown', {})[0], 404)
        self.assertEqual(self.request('GET', '/', headers={'Host':'evil.example'})[0], 403)
        self.assertEqual(self.request('POST', '/api/reset', {}, headers={'Origin':'https://evil.example'})[0], 403)
        origin = 'http://%s:%s' % self.server.server_address
        self.assertEqual(self.request('POST', '/api/reset', {}, headers={'Origin':origin})[0], 200)

    def test_loopback_binding_only(self):
        for host in ['0.0.0.0', '192.168.1.1', 'example.com']:
            with self.assertRaises(ValueError):
                build_server(self.session, host=host, port=0)


if __name__ == '__main__':
    unittest.main()
