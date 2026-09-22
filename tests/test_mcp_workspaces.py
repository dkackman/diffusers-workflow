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
    server_info,
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

    def test_creating_one_says_it_did_not_switch(self):
        """A create-then-run sequence landed a five-shot job in the wrong
        workspace; the result now says where the session still is."""
        client, _ = recording({"name": "shots"})
        result = create_workspace(client, "shots")
        assert result["current"] == DEFAULT_WORKSPACE
        assert "use_workspace" in result["next"]

    def test_creating_with_use_switches_to_it(self):
        client, _seen = recording(listing("default", "shots"))
        result = create_workspace(client, "shots", use=True)
        assert client.workspace == "shots"
        assert result["current"] == "shots"

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


class TestServerInfo:
    def test_default_workspace_returns_server_directories_unchanged(self):
        """When in the default workspace, directories come straight from
        /api/server and the workspace field is added."""

        def handler(request):
            if request.url.path == "/api/server":
                return httpx.Response(
                    200,
                    json={
                        "device": "cuda",
                        "version": "0.1.0",
                        "directories": {
                            "workflows": "/home/user/workflows",
                            "assets": "/home/user/assets",
                            "outputs": "/home/user/outputs",
                            "prompts": "/home/user/prompts",
                        },
                    },
                )
            return httpx.Response(200, json={})

        client = DwClient(transport=httpx.MockTransport(handler))
        result = server_info(client)

        assert result["workspace"] == DEFAULT_WORKSPACE
        assert result["directories"]["workflows"] == "/home/user/workflows"
        assert result["directories"]["assets"] == "/home/user/assets"
        assert result["directories"]["outputs"] == "/home/user/outputs"
        assert result["directories"]["prompts"] == "/home/user/prompts"
        assert result["device"] == "cuda"
        assert result["version"] == "0.1.0"

    def test_named_workspace_swaps_directories(self):
        """When in a named workspace, directories are replaced with the
        workspace-specific ones from /api/workspaces."""

        def handler(request):
            if request.url.path == "/api/server":
                return httpx.Response(
                    200,
                    json={
                        "device": "cuda",
                        "version": "0.1.0",
                        "directories": {
                            "workflows": "/home/user/workflows",
                            "assets": "/home/user/assets",
                            "outputs": "/home/user/outputs",
                            "prompts": "/home/user/prompts",
                        },
                    },
                )
            elif request.url.path == "/api/workspaces":
                return httpx.Response(
                    200,
                    json={
                        "workspace_root": "/studio",
                        "default": DEFAULT_WORKSPACE,
                        "workspaces": [
                            {
                                "name": DEFAULT_WORKSPACE,
                                "default": True,
                                "workflows": "/home/user/workflows",
                                "assets": "/home/user/assets",
                                "outputs": "/home/user/outputs",
                                "prompts": "/home/user/prompts",
                            },
                            {
                                "name": "shots",
                                "default": False,
                                "workflows": "/studio/shots/workflows",
                                "assets": "/studio/shots/assets",
                                "outputs": "/studio/shots/outputs",
                                "prompts": "/home/user/prompts",
                            },
                        ],
                    },
                )
            return httpx.Response(200, json={})

        client = DwClient(transport=httpx.MockTransport(handler))
        client.workspace = "shots"
        result = server_info(client)

        assert result["workspace"] == "shots"
        assert result["directories"]["workflows"] == "/studio/shots/workflows"
        assert result["directories"]["assets"] == "/studio/shots/assets"
        assert result["directories"]["outputs"] == "/studio/shots/outputs"
        assert result["directories"]["prompts"] == "/home/user/prompts"
        assert result["device"] == "cuda"
        assert result["version"] == "0.1.0"


class TestPerCallPin:
    """`workspace=` on an output-side tool: for this one call, without
    switching the session (#99). A job pinned there with
    `run_workflow(workspace=...)` is otherwise unreachable - its files
    resolve against the session's workspace and answer "does not exist"."""

    def pinned(self, call):
        client, seen = recording()
        call(client)
        return seen[-1].url

    def test_every_output_side_tool_carries_the_pin(self):
        from dw_mcp import assets, catalog, media

        calls = (
            lambda client: catalog.list_gallery(client, workspace="qa"),
            lambda client: catalog.get_gallery_metadata(
                client, "run/out.mp4", workspace="qa"
            ),
            lambda client: media.delete_output(client, "run/out.mp4", workspace="qa"),
            lambda client: assets.keep_output(
                client, "run/out.mp4", asset_name="shot.mp4", workspace="qa"
            ),
        )
        for call in calls:
            assert self.pinned(call).params.get("workspace") == "qa"

    def test_the_pin_reaches_the_outputs_route_too(self):
        """/outputs is a route, not an /api one, and streams rather than
        returning JSON - the two places a query parameter is easiest to
        drop."""
        from dw_mcp import media

        client, seen = recording()

        def handler(request):
            seen.append(request)
            return httpx.Response(
                200, content=b"hello", headers={"content-type": "text/plain"}
            )

        client = DwClient(transport=httpx.MockTransport(handler))
        media.get_output_text(client, "run/out.txt", workspace="qa")
        assert seen[-1].url.params.get("workspace") == "qa"

    def test_the_pin_does_not_switch_the_session(self):
        from dw_mcp import catalog

        client, seen = recording()
        catalog.list_gallery(client, workspace="qa")
        assert client.workspace == DEFAULT_WORKSPACE
        catalog.list_gallery(client)
        assert seen[-1].url.params.get("workspace") is None

    def test_the_pin_wins_over_the_session_s_own(self):
        from dw_mcp import catalog

        client, seen = recording()
        client.workspace = "shots"
        catalog.list_gallery(client, workspace="qa")
        assert seen[-1].url.params.get("workspace") == "qa"

    def test_naming_the_default_reaches_it_from_a_named_session(self):
        """The one spelling that cannot be a missing selector: a session in
        'shots' asking for the default sends none, which is what the server
        reads as its default."""
        from dw_mcp import catalog

        client, seen = recording()
        client.workspace = "shots"
        catalog.list_gallery(client, workspace=DEFAULT_WORKSPACE)
        assert seen[-1].url.params.get("workspace") is None


class TestMountedPinWarning:
    """dw.serve --mcp shares one DwClient across every connected agent
    (#298) - there is no per-session pin there. use_workspace and
    create_workspace(use=True) cannot stop another client's call from
    landing between this session's calls, but they can say so when a pin
    that was not the default is about to move."""

    def test_switching_away_from_a_named_pin_warns_when_mounted(self):
        client, _seen = recording(listing("default", "shots", "qa"))
        client.mounted = True
        use_workspace(client, "shots")
        result = use_workspace(client, "qa")
        assert "warning" in result
        assert "shots" in result["warning"]
        assert "qa" in result["warning"]

    def test_switching_from_the_default_does_not_warn(self):
        client, _seen = recording(listing("default", "shots"))
        client.mounted = True
        result = use_workspace(client, "shots")
        assert "warning" not in result

    def test_unmounted_clients_never_warn(self):
        """The local stdio server is one process per session - the pin is
        already session-scoped there, so the warning would be noise."""
        client, _seen = recording(listing("default", "shots", "qa"))
        use_workspace(client, "shots")
        result = use_workspace(client, "qa")
        assert "warning" not in result

    def test_switching_to_the_same_workspace_does_not_warn(self):
        client, _seen = recording(listing("default", "shots"))
        client.mounted = True
        use_workspace(client, "shots")
        result = use_workspace(client, "shots")
        assert "warning" not in result

    def test_create_with_use_warns_the_same_way(self):
        client, _seen = recording({"name": "qa"})
        client.mounted = True
        client.workspace = "shots"
        result = create_workspace(client, "qa", use=True)
        assert client.workspace == "qa"
        assert result["current"] == "qa"
        assert "warning" in result
        assert "shots" in result["warning"]

    def test_create_without_use_never_warns(self):
        client, _seen = recording({"name": "qa"})
        client.mounted = True
        client.workspace = "shots"
        result = create_workspace(client, "qa")
        assert "warning" not in result
