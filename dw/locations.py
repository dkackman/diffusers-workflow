"""The one policy for a location a workflow's arguments supply.

A workflow JSON is untrusted input under the default posture (see
docs/SECURITY.md's Trust model section). Its media arguments name *where* to
read from, and until this module existed each loader answered that question
for itself: `fetch_image` accepted any absolute path the JSON wrote,
`gather_images` handed its `glob` straight to the filesystem, and any
`http(s)` URL was fetched whatever host it named. That is arbitrary file read
and SSRF from a document the server treats as data (#114, #115, #116, #112).

Two rules, applied wherever a caller-supplied location is resolved:

- A path must land inside one of the roots this installation already works
  in - the workflow's own directory, the asset libraries on the search path,
  the output root. `..` never appears in a legitimate one and `validate_path`
  already refuses it, so in practice this closes the *absolute* path that
  pointed somewhere else entirely. The remedy is an `asset:` reference, which
  is what the roots exist for.
- An `http(s)` URL must not name a host inside the deployment - loopback,
  link-local (the cloud metadata address), a private range, or anything else
  that is not globally routable (100.64.0.0/10, Tailscale's range). The check
  runs on the resolved address, not on the literal string, so a hostname that
  answers 127.0.0.1 is caught too, and `safe_get` / `safe_post` run it again
  on every redirect before following it, then dial the address it checked
  rather than resolving the name a second time.

Both yield to `--trust-workflows`, exactly as the import and remote-code
gates do: an operator who has vouched for a workflow's source may point it at
a scratch directory or an internal endpoint. Neither yields to anything else -
there is no per-argument opt-out, because the argument is the untrusted part.

Enforcement is in two places on purpose. `location_errors` runs at validation
time, so `validate_workflow` refuses the workflow before a model load is
spent on it; the loaders call the same functions at run time, because a
location that arrives through a variable or a previous result was never in
the document to check.
"""

import contextvars
import functools
import ipaddress
import logging
import os
import socket
import threading
import time
from urllib.parse import urljoin, urlparse, urlsplit

from . import references
from .security import (
    InvalidInputError,
    PathTraversalError,
    validate_path,
    validate_url,
)
from .trust import workflows_are_trusted

logger = logging.getLogger("dw")

# Media argument names follow the same conventions realize_args dispatches on
# (dw/arguments.py): a key named like its media, or an explicit
# {"media_type", "location"} reference, or an object's "from_file"
MEDIA_KEY_SUFFIXES = ("_image", "_video", "_audio")
MEDIA_KEY_NAMES = ("image", "video", "audio", "location", "from_file")

# Task arguments that name a file to read but not by the conventions above -
# a generic name the key match would miss, so it is listed per command. Each
# gets the same validate-time refusal as a media key, rather than only the
# loader's one at run time (#630). The finishing commands' `media` and
# apply_lut's `lut` likewise (#635, SE-F044)
TASK_MEDIA_ARGUMENTS = {
    "join_windows": ("source",),
    "grade": ("media",),
    "sharpen": ("media",),
    "film_grain": ("media",),
    "apply_lut": ("media", "lut"),
}

# Of those, the arguments read from a local file only: no loader fetches
# them, so an http(s) URL is refused rather than passed on as a location
LOCAL_ONLY_TASK_ARGUMENTS = {"apply_lut": ("lut",)}

# The tasks whose arguments name a filesystem pattern rather than one file
GLOB_ARGUMENT = "glob"


def is_http_url(value):
    """Whether a value is a string the loaders would fetch over HTTP."""
    return isinstance(value, str) and value.startswith(("http://", "https://"))


def _refuse_other_url(location, what):
    """Refuse a location written as a URL whose scheme is not http(s).

    No loader opens a `file://` (or any other scheme) URL, but as a relative
    path it joined onto the workflow directory, landed inside a root and so
    passed containment - a refusal at run time only because no such directory
    existed, and none at validation (#618). The rule `validate_model_name`
    applies to a model_name (#117), applied to every media location.

    "://" is what makes a location a URL; whether its scheme is allowed is
    `validate_url`'s decision, so the two never disagree (`HTTPS://` is a
    URL it accepts, not a refusal here).
    """
    text = str(location)
    if "://" not in text:
        return
    try:
        validate_url(text)
    except InvalidInputError as e:
        scheme = urlparse(text).scheme or text.split("://", 1)[0]
        raise InvalidInputError(
            f"Refusing to read {what} at '{location}': it is a '{scheme}' URL, "
            f"and only http(s) URLs are fetched ({e}). Name a local file by "
            f"its path or with an 'asset:' reference."
        ) from e


def media_roots(base_dir=None):
    """Every directory a workflow's own locations may point inside, resolved.

    The workflow's directory, each asset library on the search path, and the
    output root - the three places this installation keeps the media a
    workflow works with. A root that cannot be resolved (no workspace, no
    active run) is dropped rather than failing the check open.

    Args:
        base_dir: The workflow file's directory, when one anchors the search
    """
    candidates = []
    if base_dir:
        candidates.append(base_dir)

    from .assets import asset_library

    try:
        candidates.extend(
            root.root for root in asset_library(base_dir=base_dir).roots()
        )
    except Exception:
        logger.debug("Could not resolve the asset search path", exc_info=True)

    from .runs import output_root

    try:
        candidates.append(output_root())
    except Exception:
        logger.debug("Could not resolve the output root", exc_info=True)

    roots = []
    for candidate in candidates:
        if not candidate:
            continue
        try:
            resolved = os.path.normpath(
                os.path.realpath(os.path.abspath(os.path.expanduser(str(candidate))))
            )
        except (OSError, ValueError):
            continue
        if resolved not in roots:
            roots.append(resolved)
    return roots


def _within(path, root):
    return path == root or path.startswith(root + os.sep)


def validate_media_path(
    location, base_dir=None, what="a media argument", require_exists=True
):
    """The validated absolute path a media location names, confined.

    Args:
        location: The path the workflow supplied, relative or absolute
        base_dir: Directory a relative path is resolved against - the
            workflow file's directory
        what: Short phrase naming the argument, for the error message
        require_exists: Whether a contained path that does not exist is an
            error. False for the validation-time pass, which is about policy
            rather than about what happens to be on disk right now

    Returns:
        The absolute, resolved, contained path

    Raises:
        PathTraversalError: If the path resolves outside every root
        InvalidInputError, PathTraversalError: Whatever validate_path raises
    """
    _refuse_other_url(location, what)
    # base_dir is the first root, so a relative path keeps resolving against
    # the workflow file exactly as it did before this check existed.
    # allow_create here, with the existence check moved below the containment
    # one: refusing an out-of-root path only once it turned out to exist made
    # the refusal itself a file-existence oracle for the whole filesystem
    # (#114)
    resolved = validate_path(
        location if os.path.isabs(str(location)) else _joined(location, base_dir),
        allow_create=True,
    )
    if not workflows_are_trusted():
        roots = media_roots(base_dir)
        if not any(_within(resolved, root) for root in roots):
            raise PathTraversalError(
                f"Refusing to read {what} at '{location}': it resolves "
                f"outside every directory this workflow may read "
                f"(its own directory, the asset libraries and the output "
                f"root). This includes "
                f"another workspace's own directories - each workspace is "
                f"isolated by design, not just a generic path-traversal "
                f"refusal, so a bare path into one is refused the same way "
                f"a path outside the installation entirely would be. Put "
                f"the file in the asset library and name it with an "
                f"'asset:' reference, use keep_output(shared=True) to copy "
                f"a generated file into the library every workspace shares "
                f"if it needs to cross that boundary on purpose, or pass "
                f"--trust-workflows if you trust this workflow's source."
            )

    if require_exists and not os.path.exists(resolved):
        raise InvalidInputError(f"Path does not exist: {resolved}")
    return resolved


def _joined(location, base_dir):
    return os.path.join(base_dir, str(location)) if base_dir else str(location)


def validate_media_glob(pattern, base_dir=None, what="a glob argument"):
    """The validated glob pattern, confined to one of the media roots.

    A glob is a location with wildcards in it, and it is checked the same
    way - but on the pattern's fixed leading directory, since the pattern
    itself does not exist as a path. Each *match* is checked individually
    too by the loader that opens it, which is what catches a wildcard
    escaping through a symlink.

    Returns:
        The pattern, absolute, for glob to expand
    """
    absolute = pattern if os.path.isabs(str(pattern)) else _joined(pattern, base_dir)
    if workflows_are_trusted():
        return absolute

    # The part of the pattern before the first wildcard: the directory the
    # expansion starts from, which is the thing containment is about
    fixed = str(absolute)
    for wildcard in ("*", "?", "["):
        cut = fixed.find(wildcard)
        if cut >= 0:
            fixed = fixed[:cut]
    fixed = os.path.dirname(fixed) if not fixed.endswith(os.sep) else fixed
    if ".." in fixed.replace("\\", "/").split("/"):
        raise PathTraversalError(
            f"Refusing {what} '{pattern}': it contains a '..' path segment."
        )
    try:
        resolved = os.path.normpath(os.path.realpath(os.path.abspath(fixed or ".")))
    except (OSError, ValueError) as e:
        raise InvalidInputError(f"Invalid glob pattern {pattern!r}: {e}")

    roots = media_roots(base_dir)
    if not any(_within(resolved, root) for root in roots):
        raise PathTraversalError(
            f"Refusing {what} '{pattern}': it expands outside every "
            f"directory this workflow may read (its own directory, the asset "
            f"libraries and the output root). Glob inside the "
            f"asset library, or pass --trust-workflows if you trust this "
            f"workflow's source."
        )
    return absolute


def contained_matches(paths, base_dir=None, what="a glob argument"):
    """The matches of an allowed glob that are themselves inside a root.

    A pattern can be contained and still match outside its own tree through
    a symlink, so every match is re-checked on its real path. A match that
    escapes is dropped with a warning rather than failing the run: the
    pattern was legitimate, one entry under it was not.
    """
    if workflows_are_trusted():
        return list(paths)

    roots = media_roots(base_dir)
    kept = []
    for path in paths:
        try:
            resolved = os.path.normpath(os.path.realpath(os.path.abspath(path)))
        except (OSError, ValueError):
            continue
        if any(_within(resolved, root) for root in roots):
            kept.append(path)
        else:
            logger.warning(
                f"Skipping {what} match '{path}': it resolves to {resolved}, "
                f"outside every directory this workflow may read"
            )
    return kept


# Hosts a workflow may not send the server to: its own loopback, the
# link-local range cloud metadata services answer on, and the private ranges
# that make up whatever network the box sits in. This is the SSRF boundary -
# an internal address is exactly the thing a caller cannot otherwise reach,
# which is why naming one is the attack rather than a mistake. `is_global`
# closes what the named ranges leave open: 100.64.0.0/10 is neither private
# nor reserved to `ipaddress`, and it is both Tailscale's tailnet and the
# range Alibaba's metadata service answers on (#407)
def _is_internal(address):
    return (
        not address.is_global
        or address.is_loopback
        or address.is_link_local
        or address.is_private
        or address.is_reserved
        or address.is_multicast
        or address.is_unspecified
    )


def _resolved_addresses(host):
    """Every IP a hostname answers on, as ip_address objects.

    A literal is returned as itself without a lookup. A name that does not
    resolve yields nothing, and the policy decides what that means.
    """
    try:
        return [ipaddress.ip_address(host)]
    except ValueError:
        pass
    try:
        infos = socket.getaddrinfo(host, None)
    except (socket.gaierror, UnicodeError, OSError):
        logger.debug(f"Could not resolve {host!r} for the host policy")
        return []
    addresses = []
    for info in infos:
        try:
            addresses.append(ipaddress.ip_address(info[4][0]))
        except ValueError:
            continue
    return addresses


def _dial_host(url):
    """The host urllib3 will connect to for `url`, as it will spell it.

    urllib.parse leaves `127%2e0%2e0%2e1` encoded, so no lookup answers it
    and the policy once passed it; urllib3 percent-decodes and IDNA-encodes
    the same host and dials 127.0.0.1. The policy reads the dialing parser's
    answer, so the name it checks is the name that is dialed.
    """
    from urllib3.exceptions import LocationParseError
    from urllib3.util import parse_url

    try:
        host = parse_url(str(url)).host or ""
    except LocationParseError as e:
        raise InvalidInputError(f"Invalid URL '{url}': {e}") from e
    return host.strip("[]")


def validate_media_url(url, what="a media argument", *, addresses=None):
    """The validated URL, refused if it names a host inside the deployment.

    Args:
        url: The http(s) URL the workflow supplied
        what: Short phrase naming the argument, for the error message
        addresses: A list the checked addresses are appended to, so a fetch
            dials one of them instead of resolving the name a second time
            (which may answer differently). Left empty under
            --trust-workflows, where nothing is checked

    Returns:
        The URL, unchanged

    Raises:
        InvalidInputError: If the scheme is not http(s), the host is
            percent-encoded or does not resolve, or it is internal to the
            deployment
    """
    validated = validate_url(url)
    if workflows_are_trusted():
        return validated

    host = _dial_host(validated)
    resolved = _resolved_addresses(host)
    internal = [address for address in resolved if _is_internal(address)]
    if internal:
        raise InvalidInputError(
            f"Refusing to fetch {what} from '{url}': {host} resolves to "
            f"{internal[0]}, an address inside this deployment (loopback, "
            f"link-local, private or otherwise not global). A workflow may not use the server to "
            f"reach its own network. Pass --trust-workflows if you trust "
            f"this workflow's source."
        )
    # After the internal check, so an IPv6 zone id (`fe80::1%25eth0`) is
    # refused for the address it names; any other '%' is a spelling the
    # policy and the client may not read alike, and no real host needs it
    if "%" in (urlsplit(validated).hostname or "") or "%" in host:
        raise InvalidInputError(
            f"Refusing to fetch {what} from '{url}': its host is "
            f"percent-encoded, which the HTTP client decodes into a different "
            f"host than the one written. Write the host plainly, or pass "
            f"--trust-workflows if you trust this workflow's source."
        )
    if not resolved:
        # Nothing to check is not the same as nothing wrong: a resolver that
        # fails here (SERVFAIL, a timeout an attacker's DNS can arrange) can
        # answer an internal address to the client a moment later
        raise InvalidInputError(
            f"Refusing to fetch {what} from '{url}': {host} did not resolve, "
            f"so there is no address to check it against. Check the host "
            f"name, or pass --trust-workflows if you trust this workflow's "
            f"source."
        )
    if addresses is not None:
        addresses.extend(resolved)
    return validated


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
        f"the {total_timeout:g} s allowed" if total_timeout else "the time allowed"
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


# Hosts this machine's HuggingFace token belongs to. The token is the
# credential the box holds for the Hub; a workflow chooses
# `remote_text_encoder.url`, so attaching the token to whatever it named
# would let an untrusted document exfiltrate it with one POST (#112). An
# endpoint outside these is still reachable - it just does not get the
# credential, and an endpoint that needs one is by definition a HuggingFace
# endpoint
HF_TOKEN_HOST_SUFFIXES = (
    "huggingface.co",
    "huggingface.cloud",
    "hf.space",
)


def token_host_allowed(host):
    """Whether the HuggingFace token may be attached to a request to `host`."""
    host = (host or "").lower()
    return any(
        host == suffix or host.endswith("." + suffix)
        for suffix in HF_TOKEN_HOST_SUFFIXES
    )


def validate_remote_encoder_url(
    url, what="the remote text encoder url", *, addresses=None
):
    """The validated remote text-encoder URL, or a refusal saying why.

    https only, and no address inside this deployment: the workflow file is
    untrusted input, and this field sends a request - with a credential - to
    an address it chooses. `--trust-workflows` lifts both, for an operator
    running their own endpoint on the box or over plain http on a LAN.
    Takes the same arguments as validate_media_url, so either one is a
    `validate` for `_safe_request`.

    Raises:
        InvalidInputError: On a non-https scheme or an internal host
    """
    if not workflows_are_trusted():
        scheme = urlparse(str(url)).scheme
        if scheme != "https":
            raise InvalidInputError(
                f"Refusing the remote text encoder at '{url}': its scheme is "
                f"'{scheme or 'none'}', and an untrusted workflow may only "
                f"reach an https endpoint - the request carries this "
                f"machine's HuggingFace token. Pass --trust-workflows if you "
                f"trust this workflow's source."
            )
    return validate_media_url(url, what, addresses=addresses)


def validate_model_name(name, base_dir=None):
    """A model identifier: a Hub repo id, or a path inside the media roots.

    `download_model` has always refused a path-shaped `repo_id`; the same
    name reached `from_pretrained_arguments.model_name` unchecked, so a
    workflow could name an absolute path and let diffusers decide (#117).
    A local model directory stays supported - inside a root, like any other
    location.

    Raises:
        PathTraversalError, InvalidInputError: If it is neither
    """
    from huggingface_hub.utils import HFValidationError, validate_repo_id

    try:
        validate_repo_id(str(name))
        return str(name)
    except HFValidationError as e:
        # A URL is neither a repo id nor a path, but joined onto the workflow
        # directory it resolves inside a root and so passed the containment
        # check below - the one shape of the four `download_model` refuses
        # that got through here (#117). `from_pretrained` would refuse it
        # downstream; the point of this check is not to rely on that
        if "://" in str(name):
            raise InvalidInputError(
                f"Refusing a model_name of '{name}': it is a URL, not a Hub "
                f"repo id or a local model directory ({e})."
            )
        # Not a URL either, so it is checked as a path next. Every refusal
        # below is nested onto this same HF message (#529) - a caller who
        # only reads the leaf ("Repo id must be in the form...") sees the one
        # rule `download_model` states for a repo id, whichever shape of it
        # tripped; the outer sentence still says *why this particular value*
        # was refused (a '..' segment, or a directory outside every root).
        try:
            return validate_media_path(
                str(name), base_dir, "a model_name", require_exists=False
            )
        except PathTraversalError as path_error:
            raise PathTraversalError(f"{path_error} ({e}).")
        except InvalidInputError as path_error:
            raise InvalidInputError(f"{path_error} ({e}).")


# The key a Hub file inside a model repo is named by - a lora's, an IP
# adapter's, a task's weights. It names a file *within* a repo, so it is a
# relative path and nothing else
WEIGHT_NAME_KEY = "weight_name"

# Tasks that read their weights with safetensors and nothing else, from a file
# named bare (no subfolder: the task knows its default repo's layout). A
# pickle format is refused by name rather than left to fail in the loader:
# the refusal is the documented contract, not an accident of which loader runs
SAFETENSORS_ONLY_COMMANDS = ("upscale_h3_latents",)
SAFETENSORS_SUFFIX = ".safetensors"


def validate_weight_name(name, suffixes=None, what="weight_name", subfolders=True):
    """A file name inside a Hub repo: relative, and never climbing out of it.

    `hf_hub_download(filename=...)` joins the name onto the local cache
    directory, so a name shaped like a path is a path on this machine. A
    subfolder is legitimate where `subfolders` allows one (an IP adapter's
    lives under one); an absolute path, a backslash, a drive, an empty segment or a '.'/'..'
    segment is not. The refusal names only what the caller wrote, never a
    directory on the server.

    Args:
        name: The file name the workflow supplied
        suffixes: The file endings allowed, or None for any
        what: Short phrase naming the argument, for the error message
        subfolders: Whether the name may hold a '/' at all

    Raises:
        PathTraversalError: On a name that is not a plain relative path
        InvalidInputError: On an empty name, or one with a refused ending
    """
    if not isinstance(name, str) or not name.strip():
        raise InvalidInputError(f"Refusing an empty {what}.")
    if "\x00" in name:
        raise InvalidInputError(f"Refusing a {what} containing a null byte.")
    if name.startswith(("/", "~")) or "\\" in name or ":" in name:
        raise PathTraversalError(
            f"Refusing a {what} of '{name}': it names a file inside the model "
            f"repo, so it must be a relative path - no leading '/' or '~', no "
            f"backslash, no drive."
        )
    if any(segment in ("", ".", "..") for segment in name.split("/")):
        raise PathTraversalError(
            f"Refusing a {what} of '{name}': it has an empty, '.' or '..' "
            f"path segment, so it does not name a file inside the model repo."
        )
    if not subfolders and "/" in name:
        raise PathTraversalError(
            f"Refusing a {what} of '{name}': it is a file name, not a path - no '/'."
        )
    if suffixes and not name.lower().endswith(tuple(suffixes)):
        raise InvalidInputError(
            f"Refusing a {what} of '{name}': only "
            f"{', '.join(repr(suffix) for suffix in suffixes)} weights are "
            f"read here."
        )
    return name


# ------------------------------------------------------------- validation


def _is_media_key(key):
    return isinstance(key, str) and (
        key in MEDIA_KEY_NAMES or key.endswith(MEDIA_KEY_SUFFIXES)
    )


def _deferred(value):
    """Whether a location is resolved later rather than being one now."""
    return references.is_ref(references.DEFERRED, value)


def _check(value, base_dir, what):
    """The policy message for one literal location, or None if it is fine."""
    if not isinstance(value, str) or not value or _deferred(value):
        return None
    try:
        if is_http_url(value):
            validate_media_url(value, what)
        elif "://" in value:
            _refuse_other_url(value, what)
            validate_media_url(value, what)
        elif os.path.isabs(value):
            validate_media_path(value, base_dir, what, require_exists=False)
        elif ".." in value.replace("\\", "/").split("/"):
            # A relative path is under base_dir by construction unless it
            # climbs out, and validate_path refuses '..' - but only when the
            # loader reaches it, which is a queued job and three seconds in
            # rather than a validation answer. Checked on the segments here
            # the way validate_media_glob checks a pattern's, so the two
            # spellings of "read outside the roots" are refused at the same
            # moment (#124)
            raise PathTraversalError(
                f"Refusing to read {what} at '{value}': it contains a '..' "
                f"path segment, so it does not resolve inside any directory "
                f"this workflow may read. Put the file in the asset library "
                f"and name it with an 'asset:' reference."
            )
    except (PathTraversalError, InvalidInputError) as e:
        return str(e)
    except Exception:
        # A path that simply does not exist is not this check's business -
        # the loader will say so, with the run's own error
        logger.debug(f"Location check skipped for {value!r}", exc_info=True)
    return None


def location_errors(definition, source_indices=None, base_dir=None):
    """Every media location in the definition that policy refuses.

    Reported as [{path, message}] like the other validation passes, so a
    caller learns before `run_workflow` that the workflow will not be
    allowed to read what it names - rather than after a pipeline load.

    Args:
        definition: The expanded, substituted workflow definition
        source_indices: Step index -> index in the file the author wrote
        base_dir: The workflow file's directory
    """
    errors = []
    steps = definition.get("steps") or []
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        source = references.author_index(source_indices, index)
        _walk(step, f"steps[{source}]", base_dir, errors, _weight_rules(step))
        _task_media_errors(step, f"steps[{source}]", base_dir, errors)
    return errors


def refuse_url_for_local_file(value, what):
    """Refuse an http(s) URL where only a file on the server is read.

    Containment would refuse it too, but as a path "outside every directory",
    which misnames the problem; this names it.
    """
    if is_http_url(value):
        raise InvalidInputError(
            f"Refusing to read {what} at '{value}': it is read from a file on "
            f"the server, never fetched from a URL. Upload it with "
            f"upload_asset and name it with an 'asset:' reference."
        )


def _task_media_errors(step, path, base_dir, errors):
    """The TASK_MEDIA_ARGUMENTS of a step's task, checked like a media key.
    A `{"location": ...}` value is already the walk's, through its key."""
    task = step.get("task")
    if not isinstance(task, dict):
        return
    arguments = task.get("arguments")
    if not isinstance(arguments, dict):
        return
    command = task.get("command")
    local_only = LOCAL_ONLY_TASK_ARGUMENTS.get(command, ())
    for key in TASK_MEDIA_ARGUMENTS.get(command, ()):
        here = f"{path}.task.arguments.{key}"
        for sub_path, item in _each(arguments.get(key), here):
            message = None
            if key in local_only:
                message = _refusal(refuse_url_for_local_file, item, f"'{key}'")
            message = message or _check(item, base_dir, f"'{key}'")
            if message:
                errors.append({"path": sub_path, "message": message})


def _weight_rules(step):
    """validate_weight_name's (suffixes, subfolders) for a step's task, and
    whether its model_name must be a Hub repo id (a task that downloads its
    weights reads no local directory, so one is refused here, not at run)."""
    task = step.get("task")
    command = task.get("command") if isinstance(task, dict) else None
    if command in SAFETENSORS_ONLY_COMMANDS:
        return (SAFETENSORS_SUFFIX,), False, True
    return None, True, False


def _walk(node, path, base_dir, errors, weight_rules=(None, True, False)):
    if isinstance(node, dict):
        for key, value in node.items():
            here = f"{path}.{key}"
            if key == "model_name" and isinstance(value, str):
                message = _model_name_message(value, base_dir, weight_rules[2])
                if message:
                    errors.append({"path": here, "message": message})
                continue
            if key == WEIGHT_NAME_KEY and isinstance(value, str):
                if not _deferred(value):
                    suffixes, subfolders, _ = weight_rules
                    message = _refusal(
                        validate_weight_name,
                        value,
                        suffixes,
                        WEIGHT_NAME_KEY,
                        subfolders,
                    )
                    if message:
                        errors.append({"path": here, "message": message})
                continue
            if key == "remote_text_encoder" and isinstance(value, dict):
                url = value.get("url")
                if isinstance(url, str) and not _deferred(url):
                    message = _refusal(validate_remote_encoder_url, url)
                    if message:
                        errors.append({"path": f"{here}.url", "message": message})
                continue
            if key == GLOB_ARGUMENT and isinstance(value, str):
                message = _glob_message(value, base_dir)
                if message:
                    errors.append({"path": here, "message": message})
                continue
            if _is_media_key(key):
                for sub_path, item in _each(value, here):
                    message = _check(item, base_dir, f"'{key}'")
                    if message:
                        errors.append({"path": sub_path, "message": message})
            if key == "voices" and isinstance(value, dict):
                # attribute_voices maps a voice's name to its reference, and
                # a string reference is a clip it reads - the key is the
                # voice's name, not a media key, so it is checked here or
                # only when the run reaches it (#494)
                for name, item in value.items():
                    message = _check(item, base_dir, f"voice '{name}'")
                    if message:
                        errors.append({"path": f"{here}.{name}", "message": message})
            if key == "urls" and isinstance(value, list):
                for sub_path, item in _each(value, here):
                    message = _check(item, base_dir, f"'{key}'")
                    if message:
                        errors.append({"path": sub_path, "message": message})
            _walk(value, here, base_dir, errors, weight_rules)
    elif isinstance(node, list):
        for index, item in enumerate(node):
            _walk(item, f"{path}[{index}]", base_dir, errors, weight_rules)


def _each(value, path):
    if isinstance(value, list):
        return [(f"{path}[{i}]", item) for i, item in enumerate(value)]
    return [(path, value)]


def _glob_message(pattern, base_dir):
    if _deferred(pattern):
        return None
    return _refusal(validate_media_glob, pattern, base_dir)


def _model_name_message(name, base_dir, hub_only=False):
    if _deferred(name):
        return None
    if hub_only:
        return _refusal(validate_hub_repo_id, name)
    return _refusal(validate_model_name, name, base_dir)


def validate_hub_repo_id(name, what="model_name"):
    """A Hugging Face repo id and nothing else - for a task that downloads.

    Raises:
        InvalidInputError: On a path, a URL, or any other non-repo-id
    """
    from huggingface_hub.utils import HFValidationError, validate_repo_id

    try:
        validate_repo_id(str(name))
    except HFValidationError:
        raise InvalidInputError(
            f"Refusing a {what} of '{name}': this task downloads its weights, "
            f"so it takes a Hugging Face repo id ('owner/name'), not a path."
        ) from None
    return str(name)


def _refusal(check, *args):
    """The message `check` refused its argument with, or None if it allowed it."""
    try:
        check(*args)
    except (PathTraversalError, InvalidInputError) as e:
        return str(e)
    except Exception:
        logger.debug(f"Location check skipped for {args[0]!r}", exc_info=True)
    return None
