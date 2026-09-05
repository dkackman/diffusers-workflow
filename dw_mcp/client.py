"""HTTP access to a running dw.serve, and the single place an API failure
becomes a message a non-developer can act on."""

import os
import tempfile
from urllib.parse import quote

import httpx

DEFAULT_BASE_URL = "http://127.0.0.1:8765"

# Twin of dw.server.app.LOOPBACK_HOSTS. Duplicated rather than imported:
# importing anything under dw/ runs dw/__init__.py and pulls in torch, which
# this pure HTTP client must not do (tests/test_mcp_server.py guards that).
LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def is_loopback_url(url):
    """True when `url` names this machine's loopback interface - the case
    where an unauthenticated dw.serve is only reachable by this user."""
    from urllib.parse import urlparse

    return (urlparse(url).hostname or "").lower() in LOOPBACK_HOSTS


def path_segment(name):
    """Percent-encode a name for interpolation into a request path,
    including its '/' characters.

    httpx normalizes dot-segments (`..`) out of a request URL client-side,
    before the request ever reaches the server - so an unquoted name like
    `../escape` is silently rewritten into a different, valid-looking path
    and the server's own path-traversal check never runs on it. Quoting
    keeps the literal bytes intact on the wire, so it is the server's own
    validation - not this client - that decides what a name is allowed to
    contain.
    """
    return quote(name, safe="")


def api_path(*segments):
    """Build a request path from literal and caller-supplied segments,
    percent-encoding each one - including its '/' characters.

    httpx normalizes dot-segments (`..`) out of a request URL client-side,
    before the request ever reaches the server - so an unquoted segment like
    `../escape` is silently rewritten into a different, valid-looking path
    and the server's own path-traversal check never runs on it. Quoting
    keeps the literal bytes intact on the wire, so it is the server's own
    validation - not this client - that decides what a segment is allowed to
    contain.

    This is the only sanctioned way to interpolate a value into a request
    path: a handler that builds one with an f-string instead re-opens the
    hole this closes, one call site at a time.
    """
    return "/" + "/".join(path_segment(str(segment)) for segment in segments)


class DwApiError(Exception):
    """A request to dw.serve failed. The message is meant to be read by the
    person driving the MCP client, not by a developer with a stack trace."""


def resolve_token(explicit=None):
    """The bearer token dw.serve was started with, if any: the explicit
    value, else DW_API_TOKEN - the same variable dw.serve itself reads, so
    one export configures both ends. Empty means unauthenticated."""
    return explicit or os.environ.get("DW_API_TOKEN") or ""


# The workspace name every request is scoped to when the session has chosen
# one. Not DW_WORKSPACE: that names a *directory* to the engine, where this
# names one of a server's workspaces - two different things that would be a
# confusing single variable
WORKSPACE_ENV_VAR = "DW_MCP_WORKSPACE"

DEFAULT_WORKSPACE = "default"


def resolve_workspace(explicit=None):
    """Which of the server's workspaces this session works in: the explicit
    value, else DW_MCP_WORKSPACE, else the server's default."""
    return explicit or os.environ.get(WORKSPACE_ENV_VAR) or DEFAULT_WORKSPACE


def resolve_base_url(explicit=None):
    """Where dw.serve is: the explicit value, else DW_MCP_URL, else the
    default port."""
    url = explicit or os.environ.get("DW_MCP_URL") or DEFAULT_BASE_URL
    return url.rstrip("/")


class DwClient:
    """One method per kind of REST call. Knows nothing about MCP - the tool
    handlers are plain functions over this."""

    def __init__(
        self,
        base_url=None,
        timeout=30.0,
        transport=None,
        token=None,
        workspace=None,
    ):
        self.base_url = resolve_base_url(base_url)
        # Mutable: use_workspace switches it for the rest of the session,
        # which is what makes a switch one visible call rather than a
        # parameter on every tool
        self.workspace = resolve_workspace(workspace)
        self.timeout = timeout
        token = resolve_token(token)
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        self._http = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            transport=transport,
            headers=headers,
        )

    def close(self):
        self._http.close()

    # ------------------------------------------------------------- requests

    def get_json(self, path, params=None):
        return self._json(self._request("GET", path, params=params), path)

    def post_json(self, path, payload=None):
        return self._json(self._request("POST", path, json=payload or {}), path)

    def put_json(self, path, payload):
        return self._json(self._request("PUT", path, json=payload), path)

    def delete_json(self, path, params=None):
        return self._json(self._request("DELETE", path, params=params), path)

    def post_bytes(self, path, data, params=None):
        """Send a file's bytes as the request body - the shape
        POST /api/uploads takes, so a single file needs no multipart
        parser at either end."""
        return self._json(
            self._request("POST", path, content=data, params=params), path
        )

    def get_bytes(self, path):
        """Raw body plus content type - for the output media served from the
        /outputs static mount rather than an /api route."""
        response = self._request("GET", path)
        self._raise_for_status(response, path)
        return response.content, response.headers.get("content-type", "")

    def get_bytes_if(self, path, accept_content_type):
        """Like `get_bytes`, but the body is only downloaded when
        `accept_content_type(content_type)` is true.

        Headers arrive before the body over HTTP, so a rejection closes the
        connection having read nothing past them - useful for `/outputs`,
        where a rejected file (a video, say) can be arbitrarily large.
        Returns `(None, content_type)` on rejection, `(body, content_type)`
        on acceptance. An error status is still raised either way, since the
        body has to be read to report it.
        """
        response = self._stream_request("GET", path)
        try:
            content_type = response.headers.get("content-type", "")
            if response.status_code < 400 and not accept_content_type(content_type):
                return None, content_type
            self._call_httpx(response.read, path)
            self._raise_for_status(response, path)
            return response.content, content_type
        finally:
            response.close()

    def stream_to_file(self, path, destination):
        """Stream `path`'s body straight to `destination` on disk, in
        chunks, rather than buffering it whole - for a body too large to
        hold in memory (the videos `get_bytes` can't return). Returns
        `(content_type, bytes_written)`.

        Written atomically: chunks land in a temp file next to
        `destination` first, moved into place with `os.replace` only after
        the whole body has arrived. A connection drop or server error
        mid-stream removes the temp file and re-raises rather than leaving
        a torn partial file at `destination` - which would otherwise
        "exist" for a later `overwrite=False` caller and mask the failure.
        """
        response = self._stream_request("GET", path)
        try:
            if response.status_code >= 400:
                self._call_httpx(response.read, path)
                self._raise_for_status(response, path)
            content_type = response.headers.get("content-type", "")
            parent = os.path.dirname(destination) or "."
            fd, temp_path = tempfile.mkstemp(dir=parent)

            def write_chunks():
                written = 0
                with os.fdopen(fd, "wb") as file:
                    for chunk in response.iter_bytes():
                        file.write(chunk)
                        written += len(chunk)
                return written

            try:
                written = self._call_httpx(write_chunks, path)
                os.replace(temp_path, destination)
            except BaseException:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
                raise
            return content_type, written
        finally:
            response.close()

    # ------------------------------------------------------------ internals

    def _request(self, method, path, **kwargs):
        return self._call_httpx(
            lambda: self._http.request(method, path, **self._scoped(kwargs)), path
        )

    def _scoped(self, kwargs):
        """Add the session's workspace to a request's query string.

        One place rather than a parameter on every handler: routes that are
        not workspace-scoped (prompts, models, system) ignore an unknown
        query parameter, and the server treats a missing selector as its
        default - so the default workspace sends nothing and every request
        looks exactly as it did before workspaces existed.
        """
        if self.workspace == DEFAULT_WORKSPACE:
            return kwargs
        params = dict(kwargs.get("params") or {})
        params.setdefault("workspace", self.workspace)
        return {**kwargs, "params": params}

    def _stream_request(self, method, path, **kwargs):
        scoped = self._scoped(kwargs)
        return self._call_httpx(
            lambda: self._http.send(
                self._http.build_request(method, path, **scoped), stream=True
            ),
            path,
        )

    def _call_httpx(self, send, path):
        try:
            return send()
        # ConnectTimeout is a subclass of TimeoutException, so it must be
        # caught here, before TimeoutException below.
        except (httpx.ConnectTimeout, httpx.ConnectError):
            raise DwApiError(
                f"Cannot reach diffusers-workflow at {self.base_url}. "
                "Start the server with `dw-serve` (or `python -m dw.serve`) "
                "and try again."
            )
        except httpx.TimeoutException:
            raise DwApiError(
                f"Request to {path} timed out after {self.timeout}s. The "
                "server may be busy loading a model."
            )
        except httpx.HTTPError as e:
            raise DwApiError(f"Request to {path} failed: {e}")

    def _json(self, response, path):
        self._raise_for_status(response, path)
        try:
            return response.json()
        except ValueError:
            raise DwApiError(f"{path} returned a non-JSON body: {response.text[:200]}")

    def _raise_for_status(self, response, path):
        if response.status_code < 400:
            return
        # 5xx is always a server-side failure, even when the body happens to
        # carry a `detail` (dw/server/app.py raises 500s with one) - the
        # status has to survive so it reads as distinct from a validation
        # message.
        if response.status_code < 500:
            detail = None
            try:
                body = response.json()
                if isinstance(body, dict):
                    detail = body.get("detail")
            except ValueError:
                detail = None
            if detail:
                # The API writes these for humans already - 400s carry
                # validation messages, 404s and 409s carry the reason
                formatted_detail = self._format_detail(detail)
                raise DwApiError(formatted_detail)
        raise DwApiError(
            f"{path} failed with HTTP {response.status_code}: "
            f"{response.text[:200] or 'no body'}"
        )

    def _format_detail(self, detail):
        """Format a detail from an API error response into a human-readable
        message. FastAPI validation errors (422) have detail as a list of dicts
        with 'loc' and 'msg' keys; string details are returned verbatim."""
        if isinstance(detail, list):
            messages = []
            for entry in detail:
                if isinstance(entry, dict):
                    msg = entry.get("msg")
                    loc = entry.get("loc")
                    if msg is None:
                        messages.append(str(entry))
                    elif loc and isinstance(loc, list):
                        field_name = loc[-1]
                        messages.append(f"{field_name}: {msg}")
                    else:
                        messages.append(msg)
                else:
                    messages.append(str(entry))
            return ". ".join(messages) if messages else str(detail)
        return str(detail)
