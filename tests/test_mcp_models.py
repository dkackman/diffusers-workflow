"""Model cache and diffusers-version tools.

Three of these commit the machine to something slow or unrecoverable - a
multi-gigabyte download, deleting cached weights, replacing the installed
diffusers - so most of what is worth pinning here is the gate: that each
refuses before it reaches the network, and that the refusal names the flag.
"""

import json

import httpx
import pytest

from dw_mcp import models
from dw_mcp.client import DwApiError, DwClient


def recording_client(body=None, status=200):
    """A client whose transport records the request it was given."""
    seen = {}

    def handler(request):
        seen["method"] = request.method
        seen["path"] = request.url.path
        # raw_path keeps the percent-escapes: url.path decodes them, which
        # would hide exactly the quoting this needs to prove
        seen["raw_path"] = request.url.raw_path.decode()
        seen["params"] = dict(request.url.params)
        seen["body"] = json.loads(request.content) if request.content else None
        return httpx.Response(status, json=body if body is not None else {})

    return DwClient(transport=httpx.MockTransport(handler)), seen


def refusing_client():
    """A client that fails the test if it is used at all - the gated tools
    must refuse before any request goes out."""

    def handler(request):
        raise AssertionError(f"gate let a {request.method} {request.url.path} through")

    return DwClient(transport=httpx.MockTransport(handler))


class TestTheGate:
    @pytest.mark.parametrize(
        "call",
        [
            lambda c: models.download_model(c, "org/model"),
            lambda c: models.delete_model(c, "org/model"),
            lambda c: models.update_diffusers(c),
        ],
        ids=["download_model", "delete_model", "update_diffusers"],
    )
    def test_it_refuses_without_an_acknowledgement(self, call):
        with pytest.raises(DwApiError) as excinfo:
            call(refusing_client())

        assert "acknowledged_cost=true" in str(excinfo.value)

    @pytest.mark.parametrize(
        "call, method, path",
        [
            (
                lambda c: models.download_model(c, "org/model", acknowledged_cost=True),
                "POST",
                "/api/models/download",
            ),
            (
                lambda c: models.delete_model(c, "org/model", acknowledged_cost=True),
                "DELETE",
                "/api/models",
            ),
            (
                lambda c: models.update_diffusers(c, acknowledged_cost=True),
                "POST",
                "/api/system/diffusers/update",
            ),
        ],
        ids=["download_model", "delete_model", "update_diffusers"],
    )
    def test_an_acknowledgement_lets_it_through(self, call, method, path):
        client, seen = recording_client()

        call(client)

        assert seen["method"] == method
        assert seen["path"] == path

    def test_each_refusal_says_what_that_particular_tool_costs(self):
        # One shared message would tell the user "this occupies the GPU" for
        # a download, which is wrong and trains them to wave the gate through
        messages = []
        for call in (
            lambda c: models.download_model(c, "org/model"),
            lambda c: models.delete_model(c, "org/model"),
            lambda c: models.update_diffusers(c),
        ):
            with pytest.raises(DwApiError) as excinfo:
                call(refusing_client())
            messages.append(str(excinfo.value))

        assert len(set(messages)) == 3


class TestDownloads:
    def test_download_model_sends_the_repo_id(self):
        client, seen = recording_client()

        models.download_model(client, "black-forest-labs/FLUX.1-dev", True)

        assert seen["body"] == {"repo_id": "black-forest-labs/FLUX.1-dev"}

    def test_download_model_reports_that_it_did_not_wait(self):
        client, _ = recording_client({"id": "d1", "status": "starting"})

        result = models.download_model(client, "org/model", acknowledged_cost=True)

        assert result["id"] == "d1"
        assert "list_downloads" in result["next"]

    def test_list_downloads_is_a_plain_read(self):
        client, seen = recording_client({"downloads": []})

        assert models.list_downloads(client) == {"downloads": []}
        assert (seen["method"], seen["path"]) == ("GET", "/api/models/downloads")

    def test_cancel_download_needs_no_acknowledgement(self):
        # Cancelling stops a cost rather than starting one - gating it would
        # make the safe direction the harder one
        client, seen = recording_client()

        models.cancel_download(client, "d1")

        assert seen["method"] == "POST"
        assert seen["path"] == "/api/models/downloads/d1/cancel"

    def test_a_download_id_is_quoted_into_the_path(self):
        # httpx collapses dot-segments client-side, so an unquoted id would
        # be rewritten into a different valid-looking path and the server's
        # own traversal check would never see it
        client, seen = recording_client()

        models.cancel_download(client, "../escape")

        assert seen["raw_path"] == "/api/models/downloads/..%2Fescape/cancel"

    def test_an_unknown_download_reports_the_servers_message(self):
        client, _ = recording_client({"detail": "Unknown download"}, status=404)

        with pytest.raises(DwApiError, match="Unknown download"):
            models.cancel_download(client, "nope")


class TestDeletion:
    def test_delete_model_sends_the_repo_as_a_query_parameter(self):
        client, seen = recording_client()

        models.delete_model(client, "org/model", acknowledged_cost=True)

        assert seen["params"] == {"repo": "org/model"}

    def test_a_busy_server_refusal_reaches_the_caller(self):
        # The server refuses a delete while a job is queued or running; that
        # reason is the actionable part, so it must not be flattened
        client, _ = recording_client(
            {"detail": "A job is running or queued - deleting model files ..."},
            status=409,
        )

        with pytest.raises(DwApiError, match="A job is running or queued"):
            models.delete_model(client, "org/model", acknowledged_cost=True)


class TestDiffusersVersion:
    def test_get_diffusers_state_is_a_plain_read(self):
        client, seen = recording_client({"version": "0.31.0"})

        assert models.get_diffusers_state(client) == {"version": "0.31.0"}
        assert (seen["method"], seen["path"]) == ("GET", "/api/system/diffusers")

    def test_update_diffusers_warns_that_it_can_break_the_install(self):
        with pytest.raises(DwApiError) as excinfo:
            models.update_diffusers(refusing_client())

        message = str(excinfo.value).lower()
        assert "github" in message and "break" in message
