"""Outbound HTTP for a workflow: the one place a request is made.

`dw/locations.py` decides *which* hosts a workflow may name (the policy);
this module carries the request out, and the rule that holds across it is
that the policy is enforced on the wire, not just on the string:

- every hop - the first request and each redirect - is validated again, so a
  public URL cannot bounce the fetch to an internal one;
- the name is resolved once, and the connection dials the address that was
  checked rather than resolving a second time (no DNS-rebinding window);
- the body is read under a byte cap and the whole fetch, redirects included,
  under a total deadline, so a server answering a byte at a time cannot hold
  the card's worker;
- a credential header does not follow a redirect to another origin.

`safe_get` / `safe_post` are the entry points. No other module under `dw/`
builds a `requests` session, adapter or urllib3 pool.
"""

import contextvars
import functools
import logging
import socket
import threading
import time
from urllib.parse import urljoin, urlsplit

from .locations import (
    _dial_host,
    validate_media_url,
)
from .security import InvalidInputError
from .trust import workflows_are_trusted

logger = logging.getLogger("dw")


# How many redirects a media fetch follows before giving up. requests' own
# default is 30; a CDN needs one or two
MAX_MEDIA_REDIRECTS = 5

# The most a fetched body may hold - the worker reads it whole into host RAM
# before the media loader sees it, and a workflow names the URL
MAX_MEDIA_BYTES = 1024**3

_READ_CHUNK = 1 << 16

# The longest a media fetch may take end to end, redirects included. A
# per-operation timeout alone lets a server answering a byte at a time hold
# the card's worker for as long as it keeps answering
MEDIA_TOTAL_TIMEOUT = 600


class _Deadline:
    """The time.monotonic() deadline of one _safe_request, and the abort
    timers its connections started. The call cancels the timers itself
    when it ends: urllib3 2's PoolManager.clear() closes no connections
    (they close when collected), so a timer left to the connection's own
    close() would sit out the full deadline after every fetch."""

    def __init__(self, at):
        self.at = at
        self.timers = []

    def cancel(self):
        for timer in self.timers:
            timer.cancel()


# The _Deadline of the _safe_request in progress on this thread (or
# context), for the connections it opens to enforce
_REQUEST_DEADLINE = contextvars.ContextVar("_REQUEST_DEADLINE", default=None)


def _abort_socket(sock):
    """Shut a socket down from another thread, waking whatever read or
    write is blocked on it."""
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    try:
        sock.close()
    except OSError:
        pass


@functools.lru_cache(maxsize=None)
def _deadline_adapter_class():
    """An HTTPAdapter whose connections are aborted at the deadline of the
    _safe_request that opened them.

    A per-operation timeout bounds each socket read, not their sum: a
    server that sends a response header a byte every fraction of a second
    never lets one read time out, and http.client takes header lines up to
    64 KiB, a hundred of them. Checking the clock between body chunks
    cannot reach that phase - the request has not returned yet - so the
    bound lives on the connection: once connected, a timer closes its
    socket when the deadline comes, whatever is waiting on it.
    """
    from requests.adapters import HTTPAdapter
    from urllib3.connection import HTTPConnection, HTTPSConnection
    from urllib3.connectionpool import HTTPConnectionPool, HTTPSConnectionPool

    class _DeadlineMixin:
        _deadline_timer = None

        def connect(self):
            super().connect()
            deadline = _REQUEST_DEADLINE.get()
            if deadline is None or self.sock is None:
                return
            remaining = max(deadline.at - time.monotonic(), 0)
            current = self.sock.gettimeout()
            self.sock.settimeout(
                remaining if current is None else min(current, remaining)
            )
            timer = threading.Timer(remaining, _abort_socket, args=(self.sock,))
            timer.daemon = True
            deadline.timers.append(timer)
            timer.start()
            self._deadline_timer = timer

        def close(self):
            if self._deadline_timer is not None:
                self._deadline_timer.cancel()
                self._deadline_timer = None
            super().close()

    class _DeadlineHTTPConnection(_DeadlineMixin, HTTPConnection):
        pass

    class _DeadlineHTTPSConnection(_DeadlineMixin, HTTPSConnection):
        pass

    class _DeadlineHTTPPool(HTTPConnectionPool):
        ConnectionCls = _DeadlineHTTPConnection

    class _DeadlineHTTPSPool(HTTPSConnectionPool):
        ConnectionCls = _DeadlineHTTPSConnection

    pools = {"http": _DeadlineHTTPPool, "https": _DeadlineHTTPSPool}

    class _DeadlineAdapter(HTTPAdapter):
        def init_poolmanager(self, *args, **kwargs):
            super().init_poolmanager(*args, **kwargs)
            self.poolmanager.pool_classes_by_scheme = dict(pools)

        def proxy_manager_for(self, proxy, **proxy_kwargs):
            # An http(s) proxy - a trusted run's, from the environment - gets
            # the same connections. A SOCKS proxy brings its own pool
            # classes, and keeps them
            manager = super().proxy_manager_for(proxy, **proxy_kwargs)
            if not proxy.lower().startswith("socks"):
                manager.pool_classes_by_scheme = dict(pools)
            return manager

    return _DeadlineAdapter


@functools.lru_cache(maxsize=None)
def _pinned_adapter_class():
    class _PinnedAdapter(_deadline_adapter_class()):
        """Dials `address` for every request to `host`, with TLS still
        negotiated and checked against `host`. The address is one the host
        policy resolved and passed, so a name that answers a public address
        to the lookup and an internal one to the connect (DNS rebinding)
        reaches only the address that was checked."""

        def __init__(self, host, address, **kwargs):
            super().__init__(**kwargs)
            self.host = host
            self.address = address

        def _host_header(self, url):
            # The name as the URL wrote it, with its port: a server on a
            # non-default port routes and builds redirects by both
            netloc = urlsplit(url).netloc.rsplit("@", 1)[-1]
            return netloc or self.host

        def build_connection_pool_key_attributes(self, request, verify, cert=None):
            # The pool - and so the socket - is keyed on the address; the
            # request itself keeps the name, which is what redirects resolve
            # against and what anything inspecting the request sees
            host_params, pool_kwargs = super().build_connection_pool_key_attributes(
                request, verify, cert
            )
            host_params["host"] = self.address
            if host_params.get("scheme") == "https":
                pool_kwargs["server_hostname"] = self.host
                pool_kwargs["assert_hostname"] = self.host
            return host_params, pool_kwargs

        def send(self, request, **kwargs):
            # Mounted for every http(s) URL, so a request this session was
            # not built for reaches here rather than a stock adapter that
            # resolves for itself - and is refused, since `address` was
            # checked for one host only
            if _dial_host(request.url) != self.host:
                raise InvalidInputError(
                    f"Refusing to send to '{request.url}': this connection is "
                    f"pinned to {self.host}"
                )
            # The connection is to an address, so http.client would name
            # the address in Host unless the request already names the host
            request.headers.setdefault("Host", self._host_header(request.url))
            return super().send(request, **kwargs)

    return _PinnedAdapter


def _pinned_session(url, address):
    """A requests Session that dials `address` for `url`'s host, and nothing
    else. trust_env is off: a proxy from the environment would resolve the
    name for itself, and ~/.netrc would attach credentials the workflow
    never named."""
    import requests

    session = requests.Session()
    session.trust_env = False
    adapter = _pinned_adapter_class()(_dial_host(url), address)
    # On both scheme prefixes, not on `url`: requests matches adapters
    # against the prepared URL, which re-encodes a path, a query or an IDNA
    # host - a prefix of the raw URL can miss it and fall to the default
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def _plain_session():
    """A requests Session that resolves for itself - the one a trusted run
    is fetched with, its environment's proxy settings included. It is still
    bounded by the deadline. A seam the tests replace."""
    import requests

    session = requests.Session()
    adapter = _deadline_adapter_class()()
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def _session_for(url, addresses):
    """The session a validated URL is fetched with: pinned to an address
    the policy checked, or plain when the policy is lifted
    (--trust-workflows)."""
    if workflows_are_trusted():
        return _plain_session()
    if not addresses:
        # The policy refuses a name that resolves to nothing; a `validate`
        # that reported no addresses would otherwise fail open here
        raise InvalidInputError(
            f"Refusing to fetch '{url}': no checked address to dial"
        )
    return _pinned_session(url, str(addresses[0]))


_DEFAULT_PORTS = {"http": 80, "https": 443}


def _origin(parts):
    """The (scheme, host, port) a split URL names, the scheme's default
    port filled in."""
    scheme = parts.scheme.lower()
    return scheme, parts.hostname, parts.port or _DEFAULT_PORTS.get(scheme)


def _without_credential(kwargs, current, target):
    """`kwargs` for the hop to `target`, Authorization dropped unless the
    redirect stays on `current`'s scheme, host and port (default ports
    count as equal) - at least as strict as requests' own redirect handling,
    so a token sent to one endpoint does not follow its redirect somewhere
    else."""
    headers = kwargs.get("headers")
    if not headers:
        return kwargs
    before, after = urlsplit(current), urlsplit(target)
    if _origin(before) == _origin(after):
        return kwargs
    kept = {k: v for k, v in headers.items() if k.lower() != "authorization"}
    return {**kwargs, "headers": kept}


# What a redirect does to the request, as requests' own handling (and every
# browser) does it: 303 always becomes a GET, 301 and 302 turn a POST into
# one, and the body goes with the method. 307 and 308 replay as they are
_BODY_ARGUMENTS = ("json", "data", "files")
_BODY_HEADERS = ("content-length", "content-type", "transfer-encoding")


def _redirected(method, status, kwargs):
    """The method and `kwargs` the hop after a `status` redirect is sent with."""
    if method == "HEAD" or not (
        status == 303 or (status in (301, 302) and method == "POST")
    ):
        return method, kwargs
    kept = {k: v for k, v in kwargs.items() if k not in _BODY_ARGUMENTS}
    if kept.get("headers"):
        kept["headers"] = {
            k: v for k, v in kept["headers"].items() if k.lower() not in _BODY_HEADERS
        }
    return "GET", kept


def _chunks(response):
    """The body as it arrives. urllib3's read1 returns after one socket
    read, so a caller checking a deadline between chunks gets to check it;
    iter_content's read blocks until a whole chunk is in, which a server
    sending a byte a minute stretches without limit. A body not read off
    urllib3 (another transport, a test double) goes through iter_content."""
    import requests
    from urllib3.exceptions import (
        DecodeError,
        ProtocolError,
        ReadTimeoutError,
        SSLError,
    )
    from urllib3.response import BaseHTTPResponse

    raw = getattr(response, "raw", None)
    if not isinstance(raw, BaseHTTPResponse):
        yield from response.iter_content(_READ_CHUNK)
        return
    # The same translation iter_content makes, so callers see requests'
    # exceptions either way
    try:
        while chunk := raw.read1(_READ_CHUNK, decode_content=True):
            yield chunk
    except ReadTimeoutError as e:
        raise requests.ConnectionError(e) from e
    except SSLError as e:
        # An HTTPS socket closed at the deadline mid-body surfaces here
        raise requests.exceptions.SSLError(e) from e
    except ProtocolError as e:
        raise requests.exceptions.ChunkedEncodingError(e) from e
    except DecodeError as e:
        raise requests.exceptions.ContentDecodingError(e) from e


def _read_capped(response, max_bytes, what, url, deadline=None, total_timeout=None):
    """The body, refused past `max_bytes` or past `deadline` (a
    time.monotonic() value). A bytearray, grown in place: a joined list of
    chunks holds the body twice at the end, and the cap is a gigabyte."""
    body = bytearray()
    for chunk in _chunks(response):
        body += chunk
        if len(body) > max_bytes:
            response.close()
            raise InvalidInputError(
                f"Refusing {what} from '{url}': larger than {max_bytes} bytes"
            )
        if deadline is not None and time.monotonic() > deadline:
            response.close()
            raise _too_slow(what, url, total_timeout)
    return body


def _too_slow(what, url, total_timeout=None):
    """The refusal for a fetch past its total_timeout - however the deadline
    showed itself (the clock between chunks, or the connection aborted)."""
    allowed = (
        f"the {total_timeout:g} s allowed"
        if total_timeout is not None
        else "the time allowed"
    )
    return InvalidInputError(
        f"Refusing {what} from '{url}': it took longer than {allowed} for one fetch"
    )


def _safe_request(
    method,
    url,
    what,
    timeout,
    validate=validate_media_url,
    max_bytes=MAX_MEDIA_BYTES,
    total_timeout=MEDIA_TOTAL_TIMEOUT,
    **kwargs,
):
    """One outbound request on the workflow's behalf: `validate` on the URL
    and on every redirect target, each hop dialed at an address the policy
    checked, the body capped at `max_bytes`, and the whole exchange bounded
    by `total_timeout` seconds as well as `timeout` per socket operation.

    A fetch that follows redirects on its own goes wherever the first host
    tells it to - a public URL answering 302 to 169.254.169.254 was fetched
    unchecked (#407) - so hops are followed here, one at a time. A host
    checked at one lookup and resolved again at the dial can answer
    differently the second time, so each hop is resolved once, by the
    policy, and dialed at what it checked. And a per-operation timeout
    alone lets a server trickle a byte at a time for as long as it likes,
    with the worker waiting on it.

    `validate` is validate_media_url or validate_remote_encoder_url - both
    take (url, what, *, addresses).

    Returns:
        The final requests.Response with its `.content` read, status
        already checked

    Raises:
        InvalidInputError: For a refused hop, a redirect chain past
            MAX_MEDIA_REDIRECTS, a body over the cap, or the total timeout
        requests.HTTPError: For an error status
    """
    deadline = _Deadline(time.monotonic() + total_timeout)
    # The connections this call opens read it, and abort at it
    token = _REQUEST_DEADLINE.set(deadline)
    try:
        return _follow(
            method,
            url,
            what,
            timeout,
            validate,
            max_bytes,
            total_timeout,
            deadline.at,
            kwargs,
        )
    finally:
        _REQUEST_DEADLINE.reset(token)
        deadline.cancel()


def _follow(
    method, url, what, timeout, validate, max_bytes, total_timeout, deadline, kwargs
):
    """`_safe_request`'s hops, with its deadline already in force."""
    import requests

    checked = []
    current = validate(url, what, addresses=checked)
    for _ in range(MAX_MEDIA_REDIRECTS + 1):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise _too_slow(what, url, total_timeout)
        session = _session_for(current, checked)
        response = None
        try:
            response = session.request(
                method,
                current,
                timeout=min(timeout, remaining),
                allow_redirects=False,
                stream=True,
                **kwargs,
            )
            # An aborted socket need not raise: http.client reads EOF in the
            # middle of a header as the end of the headers, and hands back a
            # response with nothing after them. Past the deadline, whatever
            # arrived is a truncation, not an answer
            if time.monotonic() >= deadline:
                response.close()
                raise _too_slow(what, url, total_timeout)
            if response.is_redirect:
                target = urljoin(current, response.headers["Location"])
                response.close()
                logger.debug(f"{current} redirects to {target}")
                checked = []
                target = validate(
                    target,
                    f"{what} (redirected from '{url}')",
                    addresses=checked,
                )
                method, kwargs = _redirected(method, response.status_code, kwargs)
                kwargs = _without_credential(kwargs, current, target)
                current = target
                continue
            response.raise_for_status()
            declared = response.headers.get("Content-Length")
            if declared and declared.isdigit() and int(declared) > max_bytes:
                response.close()
                raise InvalidInputError(
                    f"Refusing {what} from '{url}': larger than {max_bytes} bytes "
                    f"({declared} declared)"
                )
            body = _read_capped(response, max_bytes, what, url, deadline, total_timeout)
            if time.monotonic() >= deadline:
                # The same for a body cut short by the abort
                raise _too_slow(what, url, total_timeout)
            response._content = body
            response._content_consumed = True
            return response
        except (requests.RequestException, OSError) as e:
            # The deadline aborts the socket, which surfaces as a dropped
            # connection or a short body; say what actually happened
            if time.monotonic() >= deadline:
                raise _too_slow(what, url, total_timeout) from e
            raise
        finally:
            # The response holds its connection (and that connection's
            # deadline timer) until it is closed; the session closes only
            # the idle ones. A read body stays on the response, and an
            # error status keeps its status and headers
            if response is not None:
                response.close()
            session.close()
    raise InvalidInputError(
        f"Refusing to fetch {what} from '{url}': it redirects more than "
        f"{MAX_MEDIA_REDIRECTS} times"
    )


def safe_get(
    url,
    what="a media argument",
    timeout=60,
    max_bytes=MAX_MEDIA_BYTES,
    total_timeout=MEDIA_TOTAL_TIMEOUT,
):
    """GET a media URL for a workflow (`_safe_request`)."""
    return _safe_request(
        "GET", url, what, timeout, max_bytes=max_bytes, total_timeout=total_timeout
    )


def safe_post(
    url,
    what,
    timeout=120,
    max_bytes=MAX_MEDIA_BYTES,
    validate=validate_media_url,
    total_timeout=MEDIA_TOTAL_TIMEOUT,
    **kwargs,
):
    """POST on a workflow's behalf, the way safe_get fetches: `kwargs` are
    requests' own (`json=`, `headers=`)."""
    return _safe_request(
        "POST",
        url,
        what,
        timeout,
        validate=validate,
        max_bytes=max_bytes,
        total_timeout=total_timeout,
        **kwargs,
    )
