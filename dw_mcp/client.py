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

    def delete_json(self, path):
        return self._json(self._request("DELETE", path), path)

    def get_bytes(self, path):
        """Raw body plus content type - for the output media served from the
        /outputs static mount rather than an /api route."""
        response = self._request("GET", path)
        self._raise_for_status(response, path)
        return response.content, response.headers.get("content-type", "")

    # ------------------------------------------------------------ internals

    def _request(self, method, path, **kwargs):
        try:
            return self._http.request(method, path, **kwargs)
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
                raise DwApiError(str(detail))
        raise DwApiError(
            f"{path} failed with HTTP {response.status_code}: "
            f"{response.text[:200] or 'no body'}"
        )
