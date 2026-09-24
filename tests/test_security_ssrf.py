"""SSRF beyond loopback, and where the HuggingFace token may travel.

tests/test_locations.py pins the host policy for the spellings a workflow
author would write by accident - 127.0.0.1, localhost, 169.254.169.254, one
RFC1918 address each. This file covers the spellings an attacker writes on
purpose: the rest of the private and link-local space, IPv6 and IPv4-mapped
IPv6, the decimal/octal/hex spellings glibc's resolver accepts for an IPv4
address, DNS names that answer with any of those, and redirects. The second
half pins where `remote_text_encoder` sends this machine's HuggingFace token
(#112) against lookalike hosts, URL-parser tricks and a cross-host redirect.

Nothing here touches the network. Name resolution goes through
`fake_resolver`, which answers a numeric spelling the way glibc's
`getaddrinfo` does (via `inet_aton`, which is pure computation) and a name
only from its own table; HTTP goes through `_Transport`, which stands in for
requests' `HTTPAdapter.send` - the one place every requests call in the
engine, and diffusers' own `load_image`, reaches the wire.
"""

import io
import socket
from unittest.mock import patch
from urllib.parse import urlparse

import pytest
import requests
from PIL import Image

from dw.locations import token_host_allowed, validate_media_url
from dw.security import InvalidInputError, TRUST_WORKFLOWS_ENV_VAR


@pytest.fixture
def untrusted(monkeypatch):
    """The posture a deployed server runs on (conftest defaults to trusted)."""
    monkeypatch.setenv(TRUST_WORKFLOWS_ENV_VAR, "0")


def fake_resolver(names=None):
    """A getaddrinfo that never leaves the process.

    A numeric host resolves the way glibc resolves it - `inet_aton` accepts
    '2130706433', '0x7f000001', '0177.0.0.1' and '127.1', exactly the
    spellings a string-matching SSRF filter misses. Any other name answers
    only from `names`, and an unknown one fails the way DNS does.
    """
    names = names or {}

    def getaddrinfo(host, port=None, *args, **kwargs):
        addresses = names.get(host)
        if addresses is None:
            try:
                addresses = [socket.inet_ntoa(socket.inet_aton(host))]
            except OSError:
                raise socket.gaierror(socket.EAI_NONAME, "Name or service not known")
        answers = []
        for address in addresses:
            family = socket.AF_INET6 if ":" in address else socket.AF_INET
            sockaddr = (address, port or 0, 0, 0) if ":" in address else (address, 0)
            answers.append((family, socket.SOCK_STREAM, 6, "", sockaddr))
        return answers

    return getaddrinfo


@pytest.fixture
def no_real_sockets(monkeypatch):
    """Belt and braces: a test that forgot a mock fails instead of dialing."""

    def refuse(*args, **kwargs):
        raise AssertionError("a test in this file tried to open a real connection")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


def _refused(url, names=None):
    with patch("dw.locations.socket.getaddrinfo", fake_resolver(names)):
        with pytest.raises(InvalidInputError) as refusal:
            validate_media_url(url, "an image argument")
    return str(refusal.value)


class TestInternalAddressSpellings:
    """Every spelling of an address inside the deployment is refused."""

    @pytest.mark.parametrize(
        "url",
        [
            # link-local, including the cloud metadata address on other ports
            "http://169.254.169.254/latest/meta-data/iam/",
            "http://169.254.169.254:8080/",
            "http://169.254.0.1/",
            # the whole of RFC1918, both ends of each block
            "http://10.0.0.1/",
            "http://10.255.255.254/",
            "http://172.16.0.1/",
            "http://172.31.255.254/",
            "http://192.168.0.1/",
            "http://192.168.255.254/",
            # the rest of loopback, and 'this host'
            "http://127.0.0.2/",
            "http://127.255.255.254/",
            "http://0.0.0.0/",
        ],
    )
    def test_ipv4_literals(self, untrusted, no_real_sockets, url):
        assert "inside this deployment" in _refused(url)

    @pytest.mark.parametrize(
        "url",
        [
            "http://[::1]/",
            "http://[0:0:0:0:0:0:0:1]/",
            "http://[::]/",
            # unique local, including AWS's IPv6 metadata endpoint
            "http://[fd00::1]/",
            "http://[fd00:ec2::254]/latest/meta-data/",
            "http://[fc00::1]/",
            # link-local, with and without a zone id
            "http://[fe80::1]/",
            "http://[fe80::1%25eth0]/",
        ],
    )
    def test_ipv6_literals(self, untrusted, no_real_sockets, url):
        assert "inside this deployment" in _refused(url)

    @pytest.mark.parametrize(
        "url",
        [
            "http://[::ffff:127.0.0.1]/",
            "http://[::ffff:7f00:1]/",
            "http://[::ffff:169.254.169.254]/",
            "http://[::ffff:a9fe:a9fe]/",
            "http://[::ffff:10.0.0.1]/",
            "http://[::ffff:192.168.1.1]/",
        ],
    )
    def test_ipv4_mapped_ipv6(self, untrusted, no_real_sockets, url):
        assert "inside this deployment" in _refused(url)

    @pytest.mark.parametrize(
        "url",
        [
            # 127.0.0.1 as one decimal, as hex, as octal, and shortened
            "http://2130706433/",
            "http://0x7f000001/",
            "http://0x7f.0x0.0x0.0x1/",
            "http://0177.0.0.1/",
            "http://017700000001/",
            "http://127.1/",
            "http://0/",
            # 169.254.169.254 the same ways
            "http://2852039166/",
            "http://0xa9fea9fe/",
            "http://0251.0376.0251.0376/",
            # 10.0.0.1 and 192.168.0.1
            "http://167772161/",
            "http://0xc0a80001/",
        ],
    )
    def test_numeric_spellings_the_resolver_accepts(
        self, untrusted, no_real_sockets, url
    ):
        """Refused because the check runs on what the name resolves to, not
        on the string - these never look like an IP address to a regex."""
        assert "inside this deployment" in _refused(url)

    @pytest.mark.parametrize(
        "address",
        [
            "127.0.0.1",
            "169.254.169.254",
            "10.1.2.3",
            "172.20.0.5",
            "192.168.1.10",
            "::1",
            "fd12:3456::1",
            "fe80::1",
            "::ffff:127.0.0.1",
        ],
    )
    def test_a_name_that_resolves_inside(self, untrusted, no_real_sockets, address):
        assert "inside this deployment" in _refused(
            "http://innocent.example.com/x.png",
            {"innocent.example.com": [address]},
        )

    def test_one_internal_answer_among_public_ones_is_enough(
        self, untrusted, no_real_sockets
    ):
        """The fetch may connect to any of the answers, so every one counts."""
        assert "inside this deployment" in _refused(
            "http://round-robin.example.com/x.png",
            {"round-robin.example.com": ["93.184.216.34", "10.0.0.7"]},
        )

    def test_userinfo_does_not_hide_the_real_host(self, untrusted, no_real_sockets):
        """'user@host' - the part after the '@' is where the request goes."""
        assert "inside this deployment" in _refused(
            "http://example.com@169.254.169.254/latest/meta-data/"
        )

    def test_the_metadata_hostname_is_refused_by_what_it_answers(
        self, untrusted, no_real_sockets
    ):
        assert "inside this deployment" in _refused(
            "http://metadata.google.internal/computeMetadata/v1/",
            {"metadata.google.internal": ["169.254.169.254"]},
        )

    @pytest.mark.xfail(
        strict=True,
        reason="100.64.0.0/10 (CGNAT, Alibaba's 100.100.100.200 metadata) is "
        "not is_private in ipaddress, and _is_internal never asks is_global",
    )
    def test_shared_address_space_metadata_is_refused(self, untrusted, no_real_sockets):
        assert "inside this deployment" in _refused(
            "http://100.100.100.200/latest/meta-data/"
        )

    @pytest.mark.parametrize(
        "url, names",
        [
            ("http://93.184.216.34/x.png", None),
            ("https://example.com/x.png", {"example.com": ["93.184.216.34"]}),
            ("https://example.com/x.png", {"example.com": ["2606:2800:21f::1"]}),
        ],
    )
    def test_a_public_address_is_still_allowed(
        self, untrusted, no_real_sockets, url, names
    ):
        """The policy is not 'refuse everything': a public host passes."""
        with patch("dw.locations.socket.getaddrinfo", fake_resolver(names)):
            assert validate_media_url(url, "an image argument") == url


# ---------------------------------------------------------------- transport


def _png_bytes():
    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), "red").save(buffer, format="PNG")
    return buffer.getvalue()


class _Transport:
    """Stands in for requests' HTTPAdapter.send: answers from a table of
    {url: (status, headers, body)} and records every request that would
    have gone on the wire, headers included."""

    def __init__(self, routes):
        self.routes = routes
        self.sent = []

    def send(self, adapter, request, **kwargs):
        self.sent.append(request)
        status, headers, body = self.routes.get(request.url, (404, {}, b"not found"))
        response = requests.Response()
        response.status_code = status
        response.headers.update(headers)
        response.url = request.url
        response.request = request
        response.raw = io.BytesIO(body)
        response.reason = "scripted"
        return response

    def install(self, monkeypatch):
        transport = self

        def send(adapter, request, **kwargs):
            return transport.send(adapter, request, **kwargs)

        monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", send)
        return self

    def hosts(self):
        return [urlparse(request.url).hostname for request in self.sent]


PUBLIC = {"cdn.example.com": ["93.184.216.34"], "evil.example": ["45.33.32.156"]}


class TestRedirects:
    """A URL is checked once, before the fetch - but the fetch follows
    redirects, and the redirect target was never in the document to check."""

    @pytest.mark.xfail(
        strict=True,
        reason="fetch_image validates the first URL only; requests follows "
        "a 302 from a public host to 169.254.169.254 unchecked",
    )
    def test_fetch_image_does_not_follow_a_redirect_inside(
        self, untrusted, no_real_sockets, monkeypatch, tmp_path
    ):
        from dw.arguments import fetch_image

        transport = _Transport(
            {
                "https://cdn.example.com/x.png": (
                    302,
                    {"Location": "http://169.254.169.254/latest/meta-data/"},
                    b"",
                ),
                "http://169.254.169.254/latest/meta-data/": (
                    200,
                    {"Content-Type": "image/png"},
                    _png_bytes(),
                ),
            }
        ).install(monkeypatch)

        with patch("dw.locations.socket.getaddrinfo", fake_resolver(PUBLIC)):
            try:
                fetch_image("https://cdn.example.com/x.png", str(tmp_path))
            except Exception:
                pass

        assert "169.254.169.254" not in transport.hosts()

    @pytest.mark.xfail(
        strict=True,
        reason="audio fetch validates the first URL only; requests follows "
        "a redirect to loopback unchecked",
    )
    def test_audio_fetch_does_not_follow_a_redirect_inside(
        self, untrusted, no_real_sockets, monkeypatch, tmp_path
    ):
        from dw.tasks.audio_utils import load_audio

        transport = _Transport(
            {
                "https://cdn.example.com/a.wav": (
                    307,
                    {"Location": "http://127.0.0.1:8765/api/server"},
                    b"",
                ),
            }
        ).install(monkeypatch)

        with patch("dw.locations.socket.getaddrinfo", fake_resolver(PUBLIC)):
            try:
                load_audio("https://cdn.example.com/a.wav", str(tmp_path))
            except Exception:
                pass

        assert "127.0.0.1" not in transport.hosts()

    @pytest.mark.xfail(
        strict=True,
        reason="validate_media_url checks urlparse's host (example.com, after "
        "the backslash) while requests dials urllib3's (169.254.169.254)",
    )
    def test_a_backslash_does_not_split_the_checked_host_from_the_dialed_one(
        self, untrusted, no_real_sockets, monkeypatch, tmp_path
    ):
        """Parser differential: the host policy and the HTTP client must
        agree on which host a URL names, or the policy checks one and the
        request goes to the other."""
        from dw.arguments import fetch_image

        transport = _Transport({}).install(monkeypatch)
        with patch(
            "dw.locations.socket.getaddrinfo",
            fake_resolver({"example.com": ["93.184.216.34"]}),
        ):
            try:
                fetch_image(
                    "http://169.254.169.254\\@example.com/latest/meta-data/",
                    str(tmp_path),
                )
            except Exception:
                pass
        assert "169.254.169.254" not in transport.hosts()

    def test_a_refused_first_url_sends_nothing(
        self, untrusted, no_real_sockets, monkeypatch, tmp_path
    ):
        """The refusal comes before the request, not after the response."""
        from dw.arguments import fetch_image

        transport = _Transport({}).install(monkeypatch)
        with patch("dw.locations.socket.getaddrinfo", fake_resolver()):
            with pytest.raises(InvalidInputError):
                fetch_image("http://2852039166/latest/meta-data/", str(tmp_path))
        assert transport.sent == []


# ------------------------------------------------------------- the HF token

HF_TOKEN = "hf_test_token_never_real"

HF_HOSTS = {
    "api-inference.huggingface.co": ["18.0.0.1"],
    "huggingface.co": ["18.0.0.2"],
    "abc.us-east-1.aws.endpoints.huggingface.cloud": ["18.0.0.3"],
    "someone-encoder.hf.space": ["18.0.0.4"],
    "huggingface.co.evil.example": ["45.33.32.156"],
    "evilhuggingface.co": ["45.33.32.156"],
    "huggingface.co-evil.example": ["45.33.32.156"],
    "evil.example": ["45.33.32.156"],
    "hf.space.evil.example": ["45.33.32.156"],
    "notreallyhf.space": ["45.33.32.156"],
    "huggingface.cloud.evil.example": ["45.33.32.156"],
}


class _Embeds:
    def to(self, device):
        return self


def _encode(url, monkeypatch, routes=None):
    """Run remote_text_encoder against a scripted transport and return what
    went on the wire. The token is a fixed fake; torch.load never sees a
    real body."""
    from dw.pipeline_processors import remote

    transport = _Transport(
        routes
        or {
            url: (200, {"Content-Type": "application/octet-stream"}, b"embeds"),
        }
    ).install(monkeypatch)
    monkeypatch.setattr(remote, "get_token", lambda: HF_TOKEN)
    monkeypatch.setattr(remote.torch, "load", lambda *a, **k: _Embeds())
    with patch("dw.locations.socket.getaddrinfo", fake_resolver(HF_HOSTS)):
        try:
            remote.remote_text_encoder(["a prompt"], url, "cpu")
        except RuntimeError:
            # a scripted non-200 at the end of a redirect chain
            pass
    return transport


def _carried_token(request):
    return HF_TOKEN in (request.headers.get("Authorization") or "")


class TestHuggingFaceTokenScope:
    @pytest.mark.parametrize(
        "host",
        [
            "huggingface.co",
            "api-inference.huggingface.co",
            "abc.us-east-1.aws.endpoints.huggingface.cloud",
            "huggingface.cloud",
            "hf.space",
            "someone-encoder.hf.space",
            "HuggingFace.CO",
        ],
    )
    def test_the_token_hosts(self, host):
        assert token_host_allowed(host)

    @pytest.mark.parametrize(
        "host",
        [
            "huggingface.co.evil.example",
            "evilhuggingface.co",
            "huggingface.co-evil.example",
            "huggingface-co.evil.example",
            "hf.space.evil.example",
            "notreallyhf.space",
            "huggingface.cloud.evil.example",
            "xhuggingface.cloud",
            "huggingface.co.",
            "huggingface",
            "co",
            "",
            None,
        ],
    )
    def test_lookalikes_do_not_get_it(self, host):
        assert not token_host_allowed(host)

    @pytest.mark.parametrize(
        "url",
        [
            "https://api-inference.huggingface.co/models/x",
            "https://abc.us-east-1.aws.endpoints.huggingface.cloud/",
            "https://someone-encoder.hf.space/encode",
        ],
    )
    def test_a_huggingface_endpoint_gets_the_token(
        self, untrusted, no_real_sockets, monkeypatch, url
    ):
        transport = _encode(url, monkeypatch)
        assert len(transport.sent) == 1
        assert _carried_token(transport.sent[0])

    @pytest.mark.parametrize(
        "url",
        [
            "https://huggingface.co.evil.example/encode",
            "https://evilhuggingface.co/encode",
            "https://huggingface.co-evil.example/encode",
            "https://evil.example/huggingface.co/encode",
            "https://evil.example/?host=huggingface.co",
            # userinfo: everything before '@' is credentials, not the host
            "https://huggingface.co@evil.example/encode",
            "https://huggingface.co:443@evil.example/encode",
            "https://api-inference.huggingface.co%2f@evil.example/encode",
            # a fragment or a backslash that a browser would read differently
            "https://evil.example#@huggingface.co/encode",
            "https://evil.example/\\@huggingface.co/encode",
        ],
    )
    def test_a_lookalike_url_never_carries_it(
        self, untrusted, no_real_sockets, monkeypatch, url
    ):
        transport = _encode(url, monkeypatch)
        for request in transport.sent:
            assert not _carried_token(request), request.url

    @pytest.mark.parametrize(
        "url",
        [
            "https://huggingface.co@evil.example/encode",
            pytest.param(
                "https://evil.example\\@huggingface.co/encode",
                marks=pytest.mark.xfail(
                    strict=True,
                    reason="remote.py decides on urlparse's host (huggingface.co, "
                    "after the backslash) but requests dials urllib3's "
                    "(evil.example) - the token leaves with the request",
                ),
            ),
            "https://evil.example%5c@huggingface.co/encode",
            "https://evil.example%40huggingface.co/encode",
        ],
    )
    def test_the_token_goes_only_where_the_connection_goes(
        self, untrusted, no_real_sockets, monkeypatch, url
    ):
        """Parser differential: the decision reads urllib.parse's idea of the
        host, the connection uses requests'/urllib3's. Whatever either one
        makes of a URL, a request that carries the token must be addressed
        to a HuggingFace host by the parser that actually dials."""
        from urllib3.util import parse_url

        transport = _encode(url, monkeypatch)
        for request in transport.sent:
            if _carried_token(request):
                assert token_host_allowed(parse_url(request.url).host), request.url

    def test_a_redirect_to_another_host_drops_the_token(
        self, untrusted, no_real_sockets, monkeypatch
    ):
        """A HuggingFace endpoint answering 307 elsewhere must not hand the
        credential on with the replayed POST."""
        start = "https://api-inference.huggingface.co/models/x"
        transport = _encode(
            start,
            monkeypatch,
            routes={
                start: (307, {"Location": "https://evil.example/collect"}, b""),
                "https://evil.example/collect": (
                    200,
                    {"Content-Type": "application/octet-stream"},
                    b"embeds",
                ),
            },
        )
        assert transport.hosts() == ["api-inference.huggingface.co", "evil.example"]
        assert _carried_token(transport.sent[0])
        assert not _carried_token(transport.sent[1])

    def test_a_redirect_to_http_on_the_same_host_drops_the_token(
        self, untrusted, no_real_sockets, monkeypatch
    ):
        """A downgrade to cleartext is a different origin too."""
        start = "https://api-inference.huggingface.co/models/x"
        downgrade = "http://api-inference.huggingface.co/models/x"
        transport = _encode(
            start,
            monkeypatch,
            routes={
                start: (307, {"Location": downgrade}, b""),
                downgrade: (
                    200,
                    {"Content-Type": "application/octet-stream"},
                    b"embeds",
                ),
            },
        )
        cleartext = [r for r in transport.sent if r.url.startswith("http://")]
        assert cleartext, "the scripted redirect was not followed"
        assert not any(_carried_token(r) for r in cleartext)

    def test_trust_lifts_the_scope(self, no_real_sockets, monkeypatch):
        """--trust-workflows lifts the scope: the documented escape hatch,
        pinned so it cannot widen silently into the untrusted default."""
        monkeypatch.setenv(TRUST_WORKFLOWS_ENV_VAR, "1")
        transport = _encode("https://evil.example/encode", monkeypatch)
        assert _carried_token(transport.sent[0])
        monkeypatch.setenv(TRUST_WORKFLOWS_ENV_VAR, "0")
        transport = _encode("https://evil.example/encode", monkeypatch)
        assert not _carried_token(transport.sent[0])
