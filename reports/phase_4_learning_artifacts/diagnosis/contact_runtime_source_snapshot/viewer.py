"""Dependency-free, loopback-only browser UI for normalized flight sessions."""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
import ipaddress
import json
import math
from urllib.parse import urlsplit

_MAX_BODY = 16384


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def build_server(session, *, host='127.0.0.1', port=8766):
    """Build a server; caller owns serve_forever/shutdown and session lifetime.

    Session methods must be thread-safe. Physics advances independently of HTTP.
    """
    if host != 'localhost':
        try:
            if not ipaddress.ip_address(host).is_loopback:
                raise ValueError('viewer host must be loopback')
        except ValueError as exc:
            raise ValueError('viewer host must be loopback') from exc
    if ':' in host:
        import socket
        class Server(ThreadingHTTPServer):
            address_family = socket.AF_INET6
    else:
        Server = ThreadingHTTPServer

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def reply(self, status, value, content_type='application/json'):
            body = value if isinstance(value, bytes) else json.dumps(value, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; img-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
            self.end_headers()
            self.wfile.write(body)

        def valid_host(self):
            try:
                hostname = urlsplit('http://' + self.headers.get('Host', '')).hostname
                return hostname == 'localhost' or ipaddress.ip_address(hostname).is_loopback
            except (ValueError, TypeError):
                return False

        def do_GET(self):
            if not self.valid_host():
                self.reply(403, {'error': 'host rejected'})
                return
            if self.path in ('/', '/index.html'):
                self.reply(200, files('ares').joinpath('data/viewer/index.html').read_bytes(), 'text/html; charset=utf-8')
            elif self.path == '/api/state':
                try:
                    self.reply(200, session.state())
                except (ValueError, TypeError):
                    self.reply(500, {'error': 'invalid session state'})
            else:
                self.reply(404, {'error': 'unknown route'})

        def do_POST(self):
            if not self.valid_host():
                self.reply(403, {'error': 'host rejected'})
                return
            origin = self.headers.get('Origin')
            if origin:
                parsed = urlsplit(origin)
                # Exact origin including port: other local sites have no authority.
                if parsed.scheme != 'http' or parsed.netloc != self.headers.get('Host') or parsed.path or parsed.query or parsed.fragment:
                    self.reply(403, {'error': 'origin rejected'})
                    return
            if self.path not in ('/api/action', '/api/reset', '/api/seek', '/api/start'):
                self.reply(404, {'error': 'unknown route'})
                return
            try:
                if self.headers.get('Transfer-Encoding'):
                    raise ValueError()
                length = int(self.headers.get('Content-Length', '-1'))
                if length > _MAX_BODY:
                    self.reply(413, {'error': 'payload too large'})
                    return
                if length < 0:
                    raise ValueError()
                payload = json.loads(self.rfile.read(length), parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
                if type(payload) is not dict:
                    raise ValueError()
                if self.path == '/api/action':
                    if set(payload) != {'velocity_body', 'yaw_rate', 'land'}:
                        raise ValueError()
                    velocity = payload['velocity_body']
                    if type(velocity) is not list or len(velocity) != 3 or not all(_number(v) for v in velocity) or not _number(payload['yaw_rate']) or type(payload['land']) is not bool:
                        raise ValueError()
                    session.set_input(payload)
                elif self.path in ('/api/reset', '/api/start'):
                    if payload:
                        raise ValueError()
                    if self.path == '/api/start':
                        if not callable(getattr(session, 'start', None)):
                            self.reply(409, {'error': 'session cannot start'})
                            return
                        session.start()
                    else:
                        session.reset()
                else:
                    if set(payload) != {'frame_index'} or type(payload['frame_index']) is not int or payload['frame_index'] < 0:
                        raise ValueError()
                    if not callable(getattr(session, 'seek', None)):
                        self.reply(409, {'error': 'session is not seekable'})
                        return
                    session.seek(payload['frame_index'])
            except (ValueError, TypeError, KeyError, IndexError, OverflowError):
                self.reply(400, {'error': 'invalid payload'})
                return
            self.reply(200, {'ok': True})

    return Server((host, port), Handler)


def serve_viewer(session, *, host='127.0.0.1', port=8766):
    """Serve until interrupted, then release the socket and session."""
    server = build_server(session, host=host, port=port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        session.close()
