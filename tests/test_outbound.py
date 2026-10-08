"""
Unit tests for dw/outbound.py - the transport every outbound request of a
workflow goes through: per-hop validation, the dialed address pinned to the
checked one, a byte cap and a total deadline. The host policy it enforces is
pinned in tests/test_locations.py.
"""

import http.server
import os
import socketserver
import threading
import time

import pytest

from dw.security import InvalidInputError
from dw.trust import TRUST_WORKFLOWS_ENV_VAR


@pytest.fixture
def untrusted(monkeypatch):
    """The posture a server runs on: workflow files are untrusted input."""
    monkeypatch.setenv(TRUST_WORKFLOWS_ENV_VAR, "0")


@pytest.fixture
def trusted(monkeypatch):
    monkeypatch.setenv(TRUST_WORKFLOWS_ENV_VAR, "1")


class _EchoHandler(http.server.BaseHTTPRequestHandler):
    """/ok answers 2 bytes; /host echoes the Host header; /big answers 1000
    bytes; /nolength streams 1000 bytes with no Content-Length; /trickle
    sends 50 declared bytes one at a time; /method echoes the method and the
    body length it received. /hop redirects to /ok; /loop to itself;
    /internal to the metadata address; /elsewhere 307s to /auth (which
    echoes Authorization) under another name for the same server;
    /to-port/N 307s to /auth on this host at port N; /see-other, /found and /temporary answer 303, 302 and 307 to /method.
    /slow-headers starts a header and adds a byte every 0.2 s, never
    finishing it. Every path asked for is recorded in `seen`, and a trickle
    whose client went away sets `dropped`."""

    seen = []
    dropped = threading.Event()

    def do_GET(self):
        self.received = 0
        self._answer()

    def do_POST(self):
        self.received = len(
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
        )
        self._answer()

    def _answer(self):
        path = self.path
        self.seen.append(path)
        here = f"http://{self.headers['Host']}"
        redirects = {
            "/hop": (302, f"{here}/ok"),
            "/loop": (302, f"{here}/loop"),
            "/internal": (302, "http://169.254.169.254/latest/meta-data/"),
            "/see-other": (303, f"{here}/method"),
            "/found": (302, f"{here}/method"),
            "/temporary": (307, f"{here}/method"),
        }
        # A media suffix is ignored, for loaders that insist on one
        # (/internal.wav redirects as /internal does)
        redirect = redirects.get(os.path.splitext(path)[0])
        if redirect:
            return self._redirect(redirect[1], status=redirect[0])
        if path.startswith("/to-port/"):
            port = path.split("/")[2]
            return self._redirect(f"http://127.0.0.1:{port}/auth", status=307)
        if path.startswith("/elsewhere"):
            port = self.headers["Host"].rsplit(":", 1)[1]
            return self._redirect(f"http://localhost:{port}/auth", status=307)
        if path.startswith("/trickle"):
            return self._trickle()
        if path.startswith("/slow-headers"):
            return self._slow_headers()
        if path.startswith("/host"):
            body = self.headers["Host"].encode()
        elif path.startswith("/auth"):
            body = (self.headers.get("Authorization") or "none").encode()
        elif path.startswith("/method"):
            body = f"{self.command} {self.received}".encode()
        elif path.startswith("/big") or path.startswith("/nolength"):
            body = b"x" * 1000
        else:
            body = b"ok"
        self.send_response(200)
        if not path.startswith("/nolength"):
            self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _trickle(self):
        self.send_response(200)
        self.send_header("Content-Length", "50")
        self.end_headers()
        try:
            for _ in range(50):
                self.wfile.write(b"x")
                self.wfile.flush()
                time.sleep(0.05)
        except OSError:
            self.dropped.set()  # the client gave up, which is the point

    def _slow_headers(self):
        try:
            self.wfile.write(b"HTTP/1.1 200 OK\r\nX-Slow: ")
            self.wfile.flush()
            for _ in range(300):
                time.sleep(0.2)
                self.wfile.write(b"a")
                self.wfile.flush()
        except OSError:
            self.dropped.set()

    def _redirect(self, target, status=302):
        self.send_response(status)
        self.send_header("Location", target)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *args):
        pass


class _Server(socketserver.ThreadingMixIn, socketserver.TCPServer):
    daemon_threads = True


@pytest.fixture
def local_server():
    _EchoHandler.seen = []
    _EchoHandler.dropped = threading.Event()
    server = _Server(("127.0.0.1", 0), _EchoHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


@pytest.fixture
def server_is_public(monkeypatch):
    """The local server's 127.0.0.1 passes the policy as if it were public,
    so a name the scripted resolver answers with it is pinned and dialed
    for real. Every other internal address stays internal."""
    from dw import locations

    real = locations._is_internal
    monkeypatch.setattr(
        locations,
        "_is_internal",
        lambda address: str(address) != "127.0.0.1" and real(address),
    )


def _scripted(monkeypatch, answers):
    """A getaddrinfo that answers names from `answers` - a name maps to a
    list of answers, one per lookup, the last repeated; an empty answer
    fails the way DNS does - and records every name looked up. An IP
    literal goes to the real resolver: urllib3 calls the same getaddrinfo
    to dial the pin, and that is not a lookup of the name. The answer
    carries the port asked about, so urllib3's own lookup (a trusted run)
    can dial it."""
    import ipaddress
    import socket

    real = socket.getaddrinfo
    lookups = []

    def _resolve(host, port=None, *a, **k):
        try:
            ipaddress.ip_address(host)
            return real(host, port, *a, **k)
        except ValueError:
            pass
        lookups.append(host)
        script = answers.get(host, [[]])
        answer = script[min(lookups.count(host), len(script)) - 1]
        if not answer:
            raise socket.gaierror(socket.EAI_NONAME, "Name or service not known")
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, port or 0))
            for address in answer
        ]

    monkeypatch.setattr("dw.locations.socket.getaddrinfo", _resolve)
    return lookups


class TestSafeRequest:
    """Review 2026-10-07 #3/#4: one path for every outbound request - the
    address dialed is the one the policy checked, the body is capped and
    timed, and POST goes the same way as GET."""

    def test_one_lookup_per_hop_and_the_dial_is_the_checked_address(
        self, untrusted, local_server, server_is_public, monkeypatch
    ):
        """A rebinding name: public to the first lookup, internal to any
        second one. The policy's lookup is the only one, and the server
        that answers is the address it checked."""
        from dw.outbound import safe_get

        port = local_server.split(":")[1]
        lookups = _scripted(
            monkeypatch, {"media.example": [["127.0.0.1"], ["10.0.0.1"]]}
        )
        response = safe_get(f"http://media.example:{port}/host", timeout=5)
        assert response.content == f"media.example:{port}".encode()
        assert lookups == ["media.example"]

    def test_a_redirect_hop_is_resolved_once_and_pinned_again(
        self, untrusted, local_server, server_is_public, monkeypatch
    ):
        from dw.outbound import safe_get

        port = local_server.split(":")[1]
        lookups = _scripted(monkeypatch, {"media.example": [["127.0.0.1"]]})
        assert safe_get(f"http://media.example:{port}/hop", timeout=5).content == b"ok"
        assert lookups == ["media.example", "media.example"]
        assert _EchoHandler.seen == ["/hop", "/ok"]

    def test_a_name_that_does_not_resolve_is_refused(
        self, untrusted, local_server, monkeypatch
    ):
        """A SERVFAIL to the policy must not become the client's own lookup."""
        from dw.outbound import safe_get

        port = local_server.split(":")[1]
        _scripted(monkeypatch, {})
        with pytest.raises(InvalidInputError, match="did not resolve"):
            safe_get(f"http://media.example:{port}/ok", timeout=5)
        assert _EchoHandler.seen == []

    def test_a_percent_encoded_host_is_refused_before_any_dial(
        self, untrusted, local_server, server_is_public
    ):
        """Even with 127.0.0.1 passing the policy, the spelling is refused:
        the rule is about the escape, not the address it hides."""
        from dw.outbound import safe_get

        port = local_server.split(":")[1]
        with pytest.raises(InvalidInputError, match="percent-encoded"):
            safe_get(f"http://127%2e0%2e0%2e1:{port}/ok", timeout=5)
        assert _EchoHandler.seen == []

    @pytest.mark.parametrize(
        "path, seen",
        [
            ("/host/a b", "/host/a%20b"),
            ("/x/../host", "/host"),
            ("/host?q=<x>", "/host?q=%3Cx%3E"),
        ],
    )
    def test_a_url_requests_re_encodes_still_goes_through_the_pin(
        self, untrusted, local_server, server_is_public, monkeypatch, path, seen
    ):
        """requests matches adapters on the prepared URL; a pin mounted on
        the raw one missed these and let the stock adapter resolve
        media.example for itself."""
        from dw.outbound import safe_get

        port = local_server.split(":")[1]
        lookups = _scripted(monkeypatch, {"media.example": [["127.0.0.1"]]})
        response = safe_get(f"http://media.example:{port}{path}", timeout=5)
        assert response.content == f"media.example:{port}".encode()
        assert _EchoHandler.seen == [seen]
        assert lookups == ["media.example"]

    def test_an_idna_host_goes_through_the_pin(
        self, untrusted, local_server, server_is_public, monkeypatch
    ):
        from dw.outbound import safe_get

        port = local_server.split(":")[1]
        lookups = _scripted(monkeypatch, {"xn--bcher-kva.example": [["127.0.0.1"]]})
        response = safe_get(f"http://bücher.example:{port}/host", timeout=5)
        assert response.content == f"xn--bcher-kva.example:{port}".encode()
        assert lookups == ["xn--bcher-kva.example"]

    def test_a_proxy_in_the_environment_does_not_bypass_the_pin(
        self, untrusted, local_server, server_is_public, monkeypatch
    ):
        """A proxy resolves the name for itself; an untrusted fetch ignores
        the environment's (and ~/.netrc's) settings."""
        from dw.outbound import safe_get

        port = local_server.split(":")[1]
        _scripted(monkeypatch, {"media.example": [["127.0.0.1"]]})
        for name in ("HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
            monkeypatch.setenv(name, "http://127.0.0.1:9")
        for name in ("NO_PROXY", "no_proxy"):
            monkeypatch.delenv(name, raising=False)
        assert safe_get(f"http://media.example:{port}/ok", timeout=5).content == b"ok"

    def test_the_pinned_session_refuses_another_host(self, local_server):
        """Mounted for every http(s) URL, so nothing reaches a stock adapter -
        and what is not its host is refused, not dialed at its address."""
        from dw.outbound import _pinned_session

        port = local_server.split(":")[1]
        session = _pinned_session(f"http://media.example:{port}/", "127.0.0.1")
        with pytest.raises(InvalidInputError, match="pinned to media.example"):
            session.get(f"http://other.example:{port}/ok", timeout=5)
        assert _EchoHandler.seen == []

    def test_a_trickling_body_is_refused_at_the_total_timeout(
        self, untrusted, local_server, server_is_public, monkeypatch
    ):
        """Each byte arrives well inside the per-operation timeout; only the
        total bound stops it."""
        from dw.outbound import safe_get

        port = local_server.split(":")[1]
        _scripted(monkeypatch, {"media.example": [["127.0.0.1"]]})
        started = time.monotonic()
        with pytest.raises(
            InvalidInputError, match="took longer than the .* allowed for one fetch"
        ):
            safe_get(
                f"http://media.example:{port}/trickle", timeout=5, total_timeout=0.5
            )
        assert time.monotonic() - started < 2

    @pytest.mark.parametrize("posture", ["untrusted", "trusted"])
    def test_trickling_headers_are_refused_at_the_total_timeout(
        self, local_server, server_is_public, monkeypatch, posture
    ):
        """No single read waits long enough for the per-operation timeout,
        and the request has not returned, so nothing between chunks can
        look at the clock. The connection itself is aborted at the
        deadline - pinned or not - and the server sees it go."""
        from dw.outbound import safe_get
        from dw.trust import TRUST_WORKFLOWS_ENV_VAR

        monkeypatch.setenv(
            TRUST_WORKFLOWS_ENV_VAR, "1" if posture == "trusted" else "0"
        )
        port = local_server.split(":")[1]
        _scripted(monkeypatch, {"media.example": [["127.0.0.1"]]})
        started = time.monotonic()
        with pytest.raises(
            InvalidInputError, match="took longer than the .* allowed for one fetch"
        ):
            safe_get(
                f"http://media.example:{port}/slow-headers",
                timeout=5,
                total_timeout=2,
            )
        assert time.monotonic() - started < 2.5
        assert _EchoHandler.dropped.wait(2), "the socket was left open"

    def test_a_trusted_fetch_resolves_for_itself(
        self, trusted, local_server, monkeypatch
    ):
        """The deadline adapter bounds a trusted run without pinning it:
        the name is looked up by urllib3, at the dial, and not by the
        policy."""
        from dw import outbound

        port = local_server.split(":")[1]
        lookups = _scripted(monkeypatch, {"media.example": [["127.0.0.1"]]})

        def _never_pinned(url, address):
            raise AssertionError("a trusted run must not be pinned")

        monkeypatch.setattr(outbound, "_pinned_session", _never_pinned)
        response = outbound.safe_get(f"http://media.example:{port}/host", timeout=5)
        assert response.content == f"media.example:{port}".encode()
        assert lookups == ["media.example"]

    def test_an_ssl_error_mid_body_becomes_a_requests_error(self):
        """An HTTPS socket closed at the deadline mid-body raises urllib3's
        SSLError; translated as iter_content does, it reaches _follow's
        handler and becomes the total-timeout refusal."""
        import io
        from types import SimpleNamespace

        import requests
        from urllib3.exceptions import SSLError
        from urllib3.response import HTTPResponse

        from dw.outbound import _chunks

        raw = HTTPResponse(body=io.BytesIO(b""), preload_content=False)

        def _read1(*a, **k):
            raise SSLError("closed")

        raw.read1 = _read1
        with pytest.raises(requests.exceptions.SSLError):
            list(_chunks(SimpleNamespace(raw=raw)))

    def test_the_deadline_is_cleared_after_the_call(self, trusted, local_server):
        from dw.outbound import _REQUEST_DEADLINE, _abort_socket, safe_get

        safe_get(f"http://{local_server}/ok", timeout=5)
        assert _REQUEST_DEADLINE.get() is None
        # urllib3 closes pooled connections only when they are collected,
        # so the call itself must cancel the abort timers it started - or
        # every fetch leaves a thread waiting out its full deadline
        time.sleep(0.1)
        # Only the adapter's own abort timers: another test's timer in the
        # same process is not this call's leak
        assert not [
            t
            for t in threading.enumerate()
            if isinstance(t, threading.Timer) and t.function is _abort_socket
        ]

    def test_a_redirect_to_an_internal_address_is_refused(
        self, untrusted, local_server, server_is_public, monkeypatch
    ):
        """The first hop is a public name; its 302 names the metadata
        address, which the policy refuses before anything is dialed."""
        from dw.outbound import safe_get

        port = local_server.split(":")[1]
        _scripted(monkeypatch, {"media.example": [["127.0.0.1"]]})
        with pytest.raises(InvalidInputError, match="inside this deployment"):
            safe_get(f"http://media.example:{port}/internal", timeout=5)
        assert _EchoHandler.seen == ["/internal"]

    @pytest.mark.parametrize(
        "path, answer",
        [
            ("/see-other", b"GET 0"),
            ("/found", b"GET 0"),
            ("/temporary", b"POST 8"),
        ],
    )
    def test_a_redirected_post_changes_method_as_requests_did(
        self, untrusted, local_server, server_is_public, monkeypatch, path, answer
    ):
        """303 (and 301/302 for a POST) is followed with a GET and no body;
        307 replays the POST."""
        from dw.outbound import safe_post

        port = local_server.split(":")[1]
        _scripted(monkeypatch, {"media.example": [["127.0.0.1"]]})
        response = safe_post(
            f"http://media.example:{port}{path}",
            "a test endpoint",
            timeout=5,
            json={"a": 1},
        )
        assert response.content == answer

    def test_trust_lifts_pinning_but_keeps_the_cap(self, trusted, local_server):
        from dw.outbound import safe_get

        assert safe_get(f"http://{local_server}/ok", timeout=5).content == b"ok"
        with pytest.raises(InvalidInputError, match="larger than"):
            safe_get(f"http://{local_server}/big", timeout=5, max_bytes=100)
        with pytest.raises(InvalidInputError, match="larger than"):
            safe_get(f"http://{local_server}/nolength", timeout=5, max_bytes=100)

    def test_a_body_exactly_at_the_cap_is_kept(self, trusted, local_server):
        from dw.outbound import safe_get

        response = safe_get(
            f"http://{local_server}/nolength", timeout=5, max_bytes=1000
        )
        assert len(response.content) == 1000
        with pytest.raises(InvalidInputError, match="larger than"):
            safe_get(f"http://{local_server}/nolength", timeout=5, max_bytes=999)

    def test_a_declared_length_over_the_cap_is_refused_before_the_body(
        self, trusted, local_server, monkeypatch
    ):
        from dw import outbound

        read = []
        monkeypatch.setattr(
            outbound, "_read_capped", lambda *a, **k: read.append(a) or b""
        )
        with pytest.raises(InvalidInputError, match="larger than"):
            outbound.safe_get(f"http://{local_server}/big", timeout=5, max_bytes=100)
        assert read == []

    def test_a_redirect_loop_is_refused(self, trusted, local_server):
        from dw.outbound import MAX_MEDIA_REDIRECTS, safe_get

        with pytest.raises(InvalidInputError, match=f"{MAX_MEDIA_REDIRECTS} times"):
            safe_get(f"http://{local_server}/loop", timeout=5)

    def test_a_redirect_is_followed_to_a_good_target(self, trusted, local_server):
        from dw.outbound import safe_get

        assert safe_get(f"http://{local_server}/hop", timeout=5).content == b"ok"

    def test_post_goes_through_the_same_path(self, trusted, local_server):
        from dw.outbound import safe_post

        response = safe_post(
            f"http://{local_server}/ok", "a test endpoint", timeout=5, json={"a": 1}
        )
        assert response.content == b"ok"

    def test_a_redirect_to_another_host_drops_the_credential(
        self, trusted, local_server
    ):
        """requests.post dropped Authorization when a redirect changed host;
        hops followed by hand must too, or the token sent to one endpoint
        goes wherever that endpoint redirects."""
        from dw.outbound import safe_post

        response = safe_post(
            f"http://{local_server}/elsewhere",
            "a test endpoint",
            timeout=5,
            json={"a": 1},
            headers={"Authorization": "Bearer secret"},
        )
        assert response.content == b"none"

    def test_a_redirect_to_another_port_drops_the_credential(
        self, trusted, local_server
    ):
        """Same host, different port is a different service: requests'
        should_strip_auth drops the token there, and so must hops followed
        by hand."""
        from dw.outbound import safe_post

        other = _Server(("127.0.0.1", 0), _EchoHandler)
        threading.Thread(target=other.serve_forever, daemon=True).start()
        try:
            response = safe_post(
                f"http://{local_server}/to-port/{other.server_address[1]}",
                "a test endpoint",
                timeout=5,
                headers={"Authorization": "Bearer secret"},
            )
        finally:
            other.shutdown()
            other.server_close()
        assert response.content == b"none"

    def test_the_credential_reaches_the_host_it_was_meant_for(
        self, trusted, local_server
    ):
        from dw.outbound import safe_post

        response = safe_post(
            f"http://{local_server}/auth",
            "a test endpoint",
            timeout=5,
            headers={"Authorization": "Bearer secret"},
        )
        assert response.content == b"Bearer secret"
