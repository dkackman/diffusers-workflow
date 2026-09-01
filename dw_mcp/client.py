"""HTTP access to a running dw.serve, and the single place an API failure
becomes a message a non-developer can act on."""

import os
from urllib.parse import quote

import httpx

DEFAULT_BASE_URL = "http://127.0.0.1:8765"


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


class DwApiError(Exception):
    """A request to dw.serve failed. The message is meant to be read by the
    person driving the MCP client, not by a developer with a stack trace."""


def resolve_base_url(explicit=None):
    """Where dw.serve is: the explicit value, else DW_MCP_URL, else the
    default port."""
    url = explicit or os.environ.get("DW_MCP_URL") or DEFAULT_BASE_URL
    return url.rstrip("/")


class DwClient:
    """One method per kind of REST call. Knows nothing about MCP - the tool
    handlers are plain functions over this."""

    def __init__(self, base_url=None, timeout=30.0, transport=None):
        self.base_url = resolve_base_url(base_url)
        self.timeout = timeout
        self._http = httpx.Client(
            base_url=self.base_url, timeout=timeout, transport=transport
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
            response.read()
            self._raise_for_status(response, path)
            return response.content, content_type
        finally:
            response.close()

    # ------------------------------------------------------------ internals

    def _request(self, method, path, **kwargs):
        return self._call_httpx(
            lambda: self._http.request(method, path, **kwargs), path
        )

    def _stream_request(self, method, path, **kwargs):
        return self._call_httpx(
            lambda: self._http.send(
                self._http.build_request(method, path, **kwargs), stream=True
            ),
            path,
        )

    def _call_httpx(self, send, path):
        try:
            return send()
        except httpx.ConnectTimeout:
            raise DwApiError(
                f"Cannot reach diffusers-workflow at {self.base_url}. "
                "Start the server with `dw-serve` (or `python -m dw.serve`) "
                "and try again."
            )
        except httpx.ConnectError:
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
            # FastAPI validation error format
            messages = []
            for entry in detail:
                if isinstance(entry, dict):
                    msg = entry.get("msg", "Unknown error")
                    loc = entry.get("loc")
                    if loc and isinstance(loc, list):
                        # Extract the field name from the location
                        # loc is typically ["body", "field_name"] or similar
                        field_name = loc[-1] if loc else "field"
                        messages.append(f"{field_name}: {msg}")
                    else:
                        messages.append(msg)
                else:
                    # Fallback for unexpected entry format
                    messages.append(str(entry))
            return ". ".join(messages) if messages else str(detail)
        return str(detail)
