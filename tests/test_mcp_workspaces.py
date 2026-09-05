"""Which workspace an MCP session works in: chosen once, and carried by
every request it makes afterwards."""

import httpx
import pytest

from dw_mcp.client import DEFAULT_WORKSPACE, DwApiError, DwClient
from dw_mcp.workspaces import (
    create_workspace,
    delete_workspace,
    list_workspaces,
    use_workspace,
)


def listing(*names):
    return {
        "workspace_root": "/studio",
        "default": DEFAULT_WORKSPACE,
        "workspaces": [{"name": name, "default": name == "default"} for name in names],
    }


def recording(response=None, status=200):
    """A client that answers everything the same way, recording the requests
    it was asked to make."""
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(status, json=response if response is not None else {})

    return DwClient(transport=httpx.MockTransport(handler)), seen


class TestSelection:
    def test_the_default_sends_no_selector(self):
        """A session that has not chosen looks exactly like one from before
        workspaces existed."""
        client, seen = recording(listing("default"))
        list_workspaces(client)
        # the path is /api/workspaces, so check the query, not the string
        assert seen[0].url.params.get("workspace") is None

    def test_choosing_one_scopes_every_later_request(self):
        client, seen = recording(listing("default", "shots"))
        use_workspace(client, "shots")
        client.get_json("/api/workflows")
        assert "workspace=shots" in str(seen[-1].url)

    def test_the_environment_names_one(self, monkeypatch):
        monkeypatch.setenv("DW_MCP_WORKSPACE", "studio")
        client, seen = recording()
        client.get_json("/api/gallery")
        assert "workspace=studio" in str(seen[-1].url)

    def test_an_explicit_choice_beats_the_environment(self, monkeypatch):
        monkeypatch.setenv("DW_MCP_WORKSPACE", "studio")
        client = DwClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json={})),
            workspace="shots",
        )
        assert client.workspace == "shots"

    def test_an_unknown_name_is_refused_before_it_scopes_anything(self):
        client, _seen = recording(listing("default", "shots"))
        with pytest.raises(DwApiError, match="No workspace named"):
            use_workspace(client, "ghost")
        assert client.workspace == DEFAULT_WORKSPACE

    def test_the_listing_says_which_one_is_current(self):
        client, _seen = recording(listing("default", "shots"))
        use_workspace(client, "shots")
        assert list_workspaces(client)["current"] == "shots"

    def test_a_file_fetch_carries_the_selector_too(self):
        """get_output_image and friends read the /outputs route, which is
        workspace-scoped like everything else."""
        client, seen = recording(listing("default", "shots"))
        use_workspace(client, "shots")
        client.get_bytes("/outputs/still.png")
        assert "workspace=shots" in str(seen[-1].url)


class TestLifecycle:
    def test_creating_one_does_not_switch_to_it(self):
        client, seen = recording({"name": "shots"})
        create_workspace(client, "shots")
        assert seen[-1].method == "POST"
        assert client.workspace == DEFAULT_WORKSPACE

    def test_deleting_refuses_until_the_cost_is_acknowledged(self):
        client, seen = recording({"detail": "would remove 12 files"}, status=409)
        with pytest.raises(DwApiError) as refusal:
            delete_workspace(client, "shots")
        # the refusal carries what the server said it would remove, so the
        # caller can decide rather than just being told no
        assert "12 files" in str(refusal.value)
        assert "acknowledged_cost=True" in str(refusal.value)
        assert seen[-1].url.params.get("acknowledged") is None

    def test_deleting_the_current_one_falls_back_to_the_default(self):
        client, seen = recording(listing("default", "shots"))
        use_workspace(client, "shots")
        result = delete_workspace(client, "shots", acknowledged_cost=True)
        assert seen[-1].method == "DELETE"
        assert "acknowledged=true" in str(seen[-1].url)
        assert client.workspace == DEFAULT_WORKSPACE
        assert result["current"] == DEFAULT_WORKSPACE
