"""The read-only tools: each is a thin pass-through, so the tests pin the
route, the parameters, and that nothing is reshaped on the way back."""

import httpx
import pytest

from dw_mcp import catalog
from dw_mcp.client import DwClient


def recording_client(body=None, status=200):
    """A client whose transport records the request it was given."""
    seen = {}

    def handler(request):
        seen["method"] = request.method
        seen["path"] = request.url.path
        seen["params"] = dict(request.url.params)
        return httpx.Response(status, json=body if body is not None else {})

    return DwClient(transport=httpx.MockTransport(handler)), seen


def scripted(routes):
    """A client whose transport answers a fixed map of (method, path) ->
    (status, body), for a route the recording_client above can't."""

    def handler(request):
        key = (request.method, request.url.path)
        if key not in routes:
            return httpx.Response(404, json={"detail": f"unrouted {key}"})
        status, body = routes[key]
        return httpx.Response(status, json=body)

    return DwClient(transport=httpx.MockTransport(handler)), None


@pytest.mark.parametrize(
    "call, path",
    [
        (lambda c: catalog.list_workflows(c), "/api/workflows"),
        (lambda c: catalog.get_workflow(c, "folder/w"), "/api/workflows/folder/w"),
        (
            lambda c: catalog.get_workflow(c, "folder/w", variables_only=True),
            "/api/workflows/folder/w/variables",
        ),
        (lambda c: catalog.get_schema(c), "/api/schema"),
        (lambda c: catalog.list_pipelines(c), "/api/pipelines"),
        (
            lambda c: catalog.get_pipeline_signature(c, "FluxPipeline"),
            "/api/pipelines/FluxPipeline",
        ),
        (lambda c: catalog.list_classes(c, "schedulers"), "/api/classes"),
        (
            lambda c: catalog.get_class(c, "diffusers.AutoencoderKL"),
            "/api/classes/diffusers.AutoencoderKL",
        ),
        (lambda c: catalog.list_tasks(c), "/api/tasks"),
        (lambda c: catalog.get_task(c, "upscale"), "/api/tasks/upscale"),
        (lambda c: catalog.list_models(c), "/api/models"),
        (lambda c: catalog.get_memory(c), "/api/memory"),
        (lambda c: catalog.get_health(c), "/api/health"),
        (lambda c: catalog.get_server_info(c), "/api/server"),
        (lambda c: catalog.list_jobs(c), "/api/jobs"),
        (lambda c: catalog.list_gallery(c), "/api/gallery"),
        (
            lambda c: catalog.get_gallery_metadata(c, "a.png"),
            "/api/gallery/a.png/metadata",
        ),
    ],
)
def test_each_catalog_tool_calls_its_route(call, path):
    client, seen = recording_client()

    call(client)

    assert seen["method"] == "GET"
    assert seen["path"] == path


def test_list_classes_sends_the_required_kind():
    client, seen = recording_client()

    catalog.list_classes(client, "quantization")

    assert seen["params"]["kind"] == "quantization"


def test_get_class_sends_the_target():
    client, seen = recording_client()

    catalog.get_class(client, "diffusers.FluxPipeline", target="call")

    assert seen["params"]["target"] == "call"


def test_list_gallery_sends_its_limit():
    client, seen = recording_client()

    catalog.list_gallery(client, limit=7)

    assert seen["params"]["limit"] == "7"


def test_list_gallery_sends_a_subfolder_only_when_given():
    client, seen = recording_client()
    catalog.list_gallery(client, limit=7)
    assert "subfolder" not in seen["params"]

    client, seen = recording_client()
    catalog.list_gallery(client, subfolder="final")
    assert seen["params"]["subfolder"] == "final"

    # '' is a real filter - files at a run's root - not "no filter"
    client, seen = recording_client()
    catalog.list_gallery(client, subfolder="")
    assert seen["params"]["subfolder"] == ""


def test_list_gallery_sends_folder_and_version_only_when_given():
    client, seen = recording_client()
    catalog.list_gallery(client, limit=7)
    assert "folder" not in seen["params"]
    assert "version" not in seen["params"]

    # together they name one run - what a person calls "v4"
    client, seen = recording_client()
    catalog.list_gallery(client, folder="acorn/cut", version=4)
    assert seen["params"]["folder"] == "acorn/cut"
    assert seen["params"]["version"] == "4"


def test_list_gallery_sends_only_orphans_only_when_true():
    client, seen = recording_client()
    catalog.list_gallery(client, limit=7)
    assert "only_orphans" not in seen["params"]

    client, seen = recording_client()
    catalog.list_gallery(client, only_orphans=True)
    assert seen["params"]["only_orphans"] == "true"


def test_list_gallery_sends_media_only_when_true():
    client, seen = recording_client()
    catalog.list_gallery(client, limit=7)
    assert "media" not in seen["params"]

    client, seen = recording_client()
    catalog.list_gallery(client, media=True)
    assert seen["params"]["media"] == "true"


FULL_ENTRY = {
    "summary": "a cut sequence",
    "shape": "sequence",
    "traits": ["has-audio"],
    "cost": [{"device": "cuda", "minutes": 42}],
    "kinds": ["video/mp4"],
    "variable_names": ["shots", "seed"],
    "lists": {"shots": ["name", "prompt"]},
}


def test_an_unfiltered_listing_is_summarised():
    """The whole catalog in full detail is ~6.8k tokens for a question
    that is really 'which shape do I want' (#101)."""
    client, _seen = recording_client(
        {"workflows": ["a"], "details": {"a": dict(FULL_ENTRY)}}
    )

    answer = catalog.list_workflows(client)

    assert answer["details"]["a"] == {"summary": "a cut sequence", "shape": "sequence"}
    assert answer["view"] == "summary"
    assert "shape" in answer["note"]
    assert answer["workflows"] == ["a"]


def test_a_listing_asked_for_by_shape_comes_back_whole():
    client, _seen = recording_client(
        {"workflows": ["a"], "details": {"a": dict(FULL_ENTRY)}}
    )

    answer = catalog.list_workflows(client, shape="sequence")

    assert answer["details"]["a"] == FULL_ENTRY
    assert "note" not in answer


def test_list_workflows_always_asks_for_the_compact_view():
    client, seen = recording_client({"workflows": [], "details": {}})

    catalog.list_workflows(client)

    assert seen["path"] == "/api/workflows"
    assert seen["params"] == {"view": "compact"}


def test_list_workflows_passes_its_filters_through():
    client, seen = recording_client({"workflows": [], "details": {}})

    catalog.list_workflows(
        client,
        shape="sequence",
        traits=["has-audio", "chained"],
        configures="templates/x",
        include_models=True,
    )

    assert seen["params"] == {
        "view": "compact",
        "shape": "sequence",
        "traits": "has-audio,chained",
        "configures": "templates/x",
        "include_models": "true",
    }


def test_get_workflow_sends_the_name_percent_encoded_on_the_wire():
    """httpx.URL.path decodes escapes back for display, so an unquoted and
    a quoted request can look identical on `.path` - only the wire bytes
    (`.raw_path`) tell them apart. A name with '..' has to survive intact
    onto the wire so the server's own traversal check is what refuses it,
    rather than httpx silently normalizing the dot-segment away first."""
    seen = {}

    def handler(request):
        seen["raw_path"] = request.url.raw_path
        return httpx.Response(200, json={})

    client = DwClient(transport=httpx.MockTransport(handler))

    catalog.get_workflow(client, "../escape")

    assert seen["raw_path"] == b"/api/workflows/..%2Fescape"


def test_gallery_metadata_passes_the_media_block_through_and_says_how_to_read_it():
    body = {
        "name": "score.mp3",
        "metadata": None,
        "job": {"id": "job-1", "status": "succeeded"},
        "media": {"kind": "audio", "duration_seconds": 45.05, "peak_dbfs": -1.0},
    }
    client, _ = scripted({("GET", "/api/gallery/score.mp3/metadata"): (200, body)})

    result = catalog.get_gallery_metadata(client, "score.mp3")

    assert result["media"]["duration_seconds"] == 45.05
    assert "audio_duration" in result["next"]


def test_gallery_metadata_reads_an_asset_and_says_the_numbers_are_inputs():
    """#127: the same tool answers for an input asset, and the hint it
    carries is the one that matters before a run rather than after."""
    body = {
        "name": "asset:uploads/room-bed.wav",
        "source": "asset",
        "metadata": None,
        "job": None,
        "media": {"kind": "audio", "duration_seconds": 3.3, "sample_rate": 32000},
    }
    client, _ = scripted(
        {
            (
                "GET",
                "/api/gallery/asset:uploads/room-bed.wav/metadata",
            ): (200, body)
        }
    )

    result = catalog.get_gallery_metadata(client, "asset:uploads/room-bed.wav")

    assert result["media"]["duration_seconds"] == 3.3
    assert "loop_audio" in result["next"]
    assert "audio_duration" not in result["next"]


def test_gallery_metadata_points_at_get_job_workflow_when_embedded_is_null():
    """#384: a video output's 'metadata' is always null (only an image
    embeds it), and the job that made it is known - the hint has to name
    the actual recovery route rather than leave the caller with null."""
    body = {
        "name": "templates/minimax/dialogue-short/20260923-185419-650e12db/final/x.mp4",
        "source": "output",
        "metadata": None,
        "job": {"id": "c68bc29607ec", "status": "succeeded"},
        "media": {"kind": "video", "duration_seconds": 6.0},
    }
    client, _ = scripted(
        {("GET", "/api/gallery/" + body["name"] + "/metadata"): (200, body)}
    )

    result = catalog.get_gallery_metadata(client, body["name"])

    assert "get_job_workflow" in result["next"]
    assert "c68bc29607ec" in result["next"]


def test_gallery_metadata_says_a_kept_asset_has_no_provenance():
    """#384: a kept asset has 'job: null' by construction - nothing traces
    it back to the run that made it, and the hint has to say that rather
    than staying silent about the null 'metadata'."""
    body = {
        "name": "asset:qa-cast/ep37-shot2-alibi.mp4",
        "source": "asset",
        "metadata": None,
        "job": None,
        "media": {"kind": "video", "duration_seconds": 4.2},
    }
    client, _ = scripted(
        {
            ("GET", "/api/gallery/asset:qa-cast/ep37-shot2-alibi.mp4/metadata"): (
                200,
                body,
            )
        }
    )

    result = catalog.get_gallery_metadata(client, "asset:qa-cast/ep37-shot2-alibi.mp4")

    assert "no provenance" in result["next"]
    assert "get_job_workflow" not in result["next"]


def test_gallery_metadata_points_a_cut_at_assess_output():
    """#388: whole-file numbers cannot see inside a join, so a file carrying
    shots is pointed at the tool that measures each seam."""
    name = "cut/20260923-120000-abcdef01/final/cut.mp4"
    body = {
        "name": name,
        "source": "output",
        "metadata": None,
        "job": None,
        "media": {
            "kind": "video",
            "shots": [{"name": "a"}, {"name": "b"}, {"name": "c"}],
        },
    }
    client, _ = scripted({("GET", f"/api/gallery/{name}/metadata"): (200, body)})

    result = catalog.get_gallery_metadata(client, name)

    assert "assess_output" in result["next"]
    assert name in result["next"]
    assert "3 shots" in result["next"]


def test_gallery_metadata_does_not_point_an_uncut_file_at_assess_output():
    body = {
        "name": "shot.mp4",
        "source": "output",
        "metadata": None,
        "job": None,
        "media": {"kind": "video", "shots": None},
    }
    client, _ = scripted({("GET", "/api/gallery/shot.mp4/metadata"): (200, body)})

    result = catalog.get_gallery_metadata(client, "shot.mp4")

    assert "assess_output" not in (result.get("next") or "")
