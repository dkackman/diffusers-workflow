"""GET, PUT and DELETE /api/preferences (dw/server/routes/system.py):
the overlay the server writes over the shipped ladders, its etags, and the
refusals a PUT gets."""

import json
import os

import pytest

from dw import preferences

SHIPPED_KEY = "text_to_image"


def _workflow(root, name, devices_note=""):
    path = os.path.join(root, name + ".json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(
            {
                "id": os.path.basename(name),
                "variables": {"prompt": "a cat"},
                "steps": [
                    {
                        "name": "noop",
                        "pipeline": {"configuration": {}},
                    }
                ],
            },
            handle,
        )


@pytest.fixture
def client(gallery_client, tmp_path, monkeypatch):
    """The gallery client, on a cpu host, with qa/a and qa/b in its
    workspace's catalog."""
    workflows = str(tmp_path / "workflows")
    _workflow(workflows, "qa/a")
    _workflow(workflows, "qa/b")
    monkeypatch.setattr(
        preferences, "default_host", lambda: preferences.Host(device_type="cpu")
    )
    return gallery_client


def _overlay():
    path = preferences.overlay_path()
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        return json.load(handle)


def test_get_lists_the_shipped_ladder_with_its_source(client):
    answer = client.get("/api/preferences").json()
    assert answer["default_tier"] == "standard"
    assert answer["etag"].startswith('"')
    entry = answer["preferences"][SHIPPED_KEY]
    assert entry["source"] == "shipped"
    assert set(entry["ladder"]) <= set(preferences.TIERS)
    assert "resolutions" not in entry


def test_get_with_resolve_answers_a_pick_or_a_refusal_per_tier(client):
    answer = client.get(
        "/api/preferences", params={"key": SHIPPED_KEY, "resolve": "true"}
    )
    assert answer.status_code == 200
    assert answer.headers["etag"] == answer.json()["preferences"][SHIPPED_KEY]["etag"]
    resolutions = answer.json()["preferences"][SHIPPED_KEY]["resolutions"]
    assert resolutions
    for found in resolutions.values():
        assert "target" in found or "error" in found


def test_an_unknown_key_is_a_404(client):
    assert client.get("/api/preferences", params={"key": "nope"}).status_code == 404
    assert client.delete("/api/preferences/nope").status_code == 404


def test_put_stores_a_server_ladder_and_resolves_past_a_device_skip(client):
    ladder = {"best": [{"workflow": "qa/a", "devices": ["mps"]}, "qa/b"]}
    answer = client.put("/api/preferences/qa_ladder", json=ladder)
    assert answer.status_code == 200, answer.text
    body = answer.json()
    assert body["key"] == "qa_ladder"
    assert body["source"] == "server"
    assert body["ladder"] == ladder
    best = body["resolutions"]["best"]
    assert best["target"] == "qa/b"
    assert "mps-only" in json.dumps(best["skipped"])
    assert _overlay()["preferences"]["qa_ladder"] == ladder
    listed = client.get("/api/preferences").json()["preferences"]
    assert listed["qa_ladder"]["source"] == "server"
    assert listed[SHIPPED_KEY]["source"] == "shipped"


def test_a_ladder_nothing_here_fits_is_stored_and_shows_the_refusal(client):
    ladder = {"best": [{"workflow": "qa/a", "devices": ["mps"]}]}
    body = client.put("/api/preferences/qa_mps", json=ladder).json()
    error = body["resolutions"]["best"]["error"]
    assert "mps-only" in error


@pytest.mark.parametrize(
    "ladder, words",
    [
        ({"best": ["qa/missing"]}, "not in the catalog"),
        ({"best": ["qa/missing", "qa/gone"]}, "'qa/gone'"),
        ({"ultra": ["qa/a"]}, "ultra"),
        ({"best": []}, "non-empty"),
        ({}, "ladder"),
        ({"best": [{"workflow": "qa/a", "colour": "red"}]}, "colour"),
        ({"best": [{"workflow": "qa/a", "arguments": {"nope": 1}}]}, "nope"),
        # a target's arguments meet the reference-name rule a step's do
        # (#1723, SE-F085): a traversal name is refused, not stored
        (
            {"best": [{"workflow": "qa/a", "arguments": {"prompt": "asset:../../x"}}]},
            "Invalid asset name",
        ),
        (
            {
                "best": [
                    {"workflow": "qa/a", "arguments": {"prompt": "output:../../x.png"}}
                ]
            },
            "Invalid output name",
        ),
        (
            {"best": [{"workflow": "qa/a", "arguments": {"prompt": "prompt:../x"}}]},
            "Invalid prompt name",
        ),
    ],
)
def test_a_put_the_store_cannot_hold_is_a_422_and_writes_nothing(client, ladder, words):
    answer = client.put("/api/preferences/qa_bad", json=ladder)
    assert answer.status_code == 422, answer.text
    assert words in answer.text
    assert _overlay() is None


def test_a_bad_key_name_is_a_422(client):
    assert (
        client.put("/api/preferences/has%20space", json={"best": ["qa/a"]}).status_code
        == 422
    )


def test_a_stale_if_match_is_a_412_and_writes_nothing(client):
    first = client.put("/api/preferences/qa_tag", json={"best": ["qa/a"]}).json()
    stale = first["etag"]
    second = client.put(
        "/api/preferences/qa_tag",
        json={"best": ["qa/b"]},
        headers={"If-Match": stale},
    )
    assert second.status_code == 200
    refused = client.put(
        "/api/preferences/qa_tag",
        json={"best": ["qa/a"]},
        headers={"If-Match": stale},
    )
    assert refused.status_code == 412
    detail = refused.json()["detail"]
    assert detail["code"] == "stale_write"
    assert detail["etag"] == second.json()["etag"]
    assert _overlay()["preferences"]["qa_tag"] == {"best": ["qa/b"]}
    assert (
        client.delete(
            "/api/preferences/qa_tag", headers={"If-Match": stale}
        ).status_code
        == 412
    )


def test_delete_resets_a_shipped_key_to_shipped(client):
    shipped = client.get("/api/preferences").json()["preferences"][SHIPPED_KEY]
    client.put(f"/api/preferences/{SHIPPED_KEY}", json={"best": ["qa/a"]})
    overridden = client.get("/api/preferences").json()["preferences"][SHIPPED_KEY]
    assert overridden["source"] == "server"
    answer = client.delete(f"/api/preferences/{SHIPPED_KEY}")
    assert answer.status_code == 200
    body = answer.json()
    assert body["removed"] is False
    assert body["source"] == "shipped"
    assert body["ladder"] == shipped["ladder"]
    assert body["etag"] == shipped["etag"]
    # The overlay held nothing else, so it is gone
    assert _overlay() is None


def test_delete_drops_a_key_only_the_server_had(client):
    client.put("/api/preferences/qa_own", json={"best": ["qa/a"]})
    body = client.delete("/api/preferences/qa_own").json()
    assert body == {"key": "qa_own", "removed": True}
    assert "qa_own" not in client.get("/api/preferences").json()["preferences"]


def test_default_tier_is_set_with_its_own_etag(client):
    read = client.get("/api/preferences").json()
    answer = client.put(
        "/api/preferences",
        json={"default_tier": "best"},
        headers={"If-Match": read["etag"]},
    )
    assert answer.status_code == 200
    assert answer.json()["default_tier"] == "best"
    assert client.get("/api/preferences").json()["default_tier"] == "best"
    assert preferences.load_preferences()["default_tier"] == "best"
    stale = client.put(
        "/api/preferences",
        json={"default_tier": "draft"},
        headers={"If-Match": read["etag"]},
    )
    assert stale.status_code == 412
    assert (
        client.put("/api/preferences", json={"default_tier": "ultra"}).status_code
        == 422
    )


def test_default_tier_changes_the_untiered_pick(client):
    client.put(
        "/api/preferences/qa_tiers",
        json={"standard": ["qa/a"], "best": ["qa/b"]},
    )
    store = preferences.load_preferences()
    assert store["default_tier"] == "standard"
    client.put("/api/preferences", json={"default_tier": "best"})
    assert preferences.load_preferences()["default_tier"] == "best"


def test_writes_keep_the_overlay_whole_json(client):
    client.put("/api/preferences/qa_one", json={"best": ["qa/a"]})
    client.put("/api/preferences", json={"default_tier": "draft"})
    client.put("/api/preferences/qa_two", json={"draft": ["qa/b"]})
    document = _overlay()
    assert document == {
        "default_tier": "draft",
        "preferences": {"qa_one": {"best": ["qa/a"]}, "qa_two": {"draft": ["qa/b"]}},
    }


@pytest.mark.parametrize("target", ["../../etc/passwd", "/etc/passwd", "qa/nope"])
def test_a_refused_target_names_no_server_path(client, tmp_path, target):
    """SE-F085: a target that climbs out, sits outside every source, or is
    absent is refused naming the target as written and the sources by
    label - never the server's folders."""
    answer = client.put("/api/preferences/qa_sec", json={"best": [target]})
    assert answer.status_code == 422, answer.text
    for folder in {str(tmp_path), os.path.realpath(str(tmp_path))}:
        assert folder not in answer.text
    assert _overlay() is None


def test_a_refused_builtin_target_names_no_install_path(client):
    import dw

    answer = client.put("/api/preferences/qa_sec", json={"best": ["builtin:a/b.json"]})
    assert answer.status_code == 422, answer.text
    assert "packaged workflows" in answer.text
    assert os.path.dirname(dw.__file__) not in answer.text


# --- the full read: resolve as admission does, with used_by and contract


def _composer(root, name, key):
    path = os.path.join(root, name + ".json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(
            {
                "id": os.path.basename(name),
                "steps": [{"name": "s", "workflow": {"path": f"preference:{key}"}}],
            },
            handle,
        )


def _no_probe(*_args, **_kwargs):
    raise AssertionError("the plain resolve read must make no Hub calls")


def test_full_read_adds_used_by_and_contract(client, tmp_path, monkeypatch):
    monkeypatch.setattr(
        preferences, "downloads_gate_check", lambda: lambda *a: (None, None)
    )
    _composer(str(tmp_path / "workflows"), "qa/uses", SHIPPED_KEY)
    answer = client.get("/api/preferences", params={"resolve": "true", "full": "true"})
    assert answer.status_code == 200, answer.text
    entry = answer.json()["preferences"][SHIPPED_KEY]
    assert isinstance(entry["used_by"], list)
    assert "qa/uses" in entry["used_by"]
    assert entry["used_by"] == sorted(entry["used_by"])
    assert entry["contract"]["required"] == ["prompt", "seed", "width", "height"]
    assert "num_inference_steps" in entry["contract"]["optional"]


def test_plain_resolve_is_unchanged(client, monkeypatch):
    monkeypatch.setattr(preferences, "downloads_gate_check", _no_probe)
    answer = client.get("/api/preferences", params={"resolve": "true"})
    assert answer.status_code == 200, answer.text
    for entry in answer.json()["preferences"].values():
        assert "used_by" not in entry
        assert "contract" not in entry
        assert "resolutions" in entry
    # full without resolve is ignored
    plain = client.get("/api/preferences", params={"full": "true"}).json()
    for entry in plain["preferences"].values():
        assert not {"used_by", "contract", "resolutions"} & set(entry)


def test_full_read_runs_the_gate_check(client, monkeypatch):
    def gate():
        def check(name, definition, arguments, child):
            if name == "qa/a":
                return "gated, no access to org/x", None
            return None, None

        return check

    monkeypatch.setattr(preferences, "downloads_gate_check", gate)
    client.put("/api/preferences/qa_gated", json={"best": ["qa/a", "qa/b"]})
    answer = client.get(
        "/api/preferences",
        params={"key": "qa_gated", "resolve": "true", "full": "true"},
    )
    best = answer.json()["preferences"]["qa_gated"]["resolutions"]["best"]
    assert best["target"] == "qa/b"
    assert best["skipped"] == [
        {"workflow": "qa/a", "reason": "gated, no access to org/x"}
    ]


def test_full_read_survives_hub_failure(client, monkeypatch):
    def gate():
        def check(*_args):
            raise OSError("the Hub is unreachable")

        return check

    monkeypatch.setattr(preferences, "downloads_gate_check", gate)
    client.put("/api/preferences/qa_offline", json={"best": ["qa/a"]})
    answer = client.get(
        "/api/preferences",
        params={"key": "qa_offline", "resolve": "true", "full": "true"},
    )
    assert answer.status_code == 200, answer.text
    best = answer.json()["preferences"]["qa_offline"]["resolutions"]["best"]
    assert best["target"] == "qa/a"
    assert best["notes"] == []
    assert best["skipped"] == []


# --- the dry-run check: POST /api/preferences/{key}/check


@pytest.mark.parametrize(
    "text, words", [("{not json", "does not parse"), ("[1, 2]", "not a workflow")]
)
def test_a_target_file_that_is_no_workflow_is_a_422_not_a_500(
    client, tmp_path, text, words
):
    (tmp_path / "workflows" / "qa" / "broken.json").write_text(text)
    put = client.put("/api/preferences/qa_ladder", json={"best": ["qa/broken"]})
    assert put.status_code == 422, put.text
    assert words in put.text
    assert str(tmp_path) not in put.text
    assert _overlay() is None
    check = client.post(
        f"/api/preferences/{SHIPPED_KEY}/check",
        json={"ladder": {"best": ["qa/broken"]}},
    )
    assert check.status_code == 200, check.text
    assert words in check.json()["tiers"]["best"]["target_errors"]["qa/broken"]


def test_check_writes_nothing(client):
    answer = client.post(
        f"/api/preferences/{SHIPPED_KEY}/check",
        json={"ladder": {"standard": ["qa/a", "qa/b"]}},
    )
    assert answer.status_code == 200, answer.text
    standard = answer.json()["tiers"]["standard"]
    assert standard["target"] == "qa/a"
    assert standard["after"] == ["qa/b"]
    assert _overlay() is None


def test_check_accepts_a_new_key(client):
    answer = client.post(
        "/api/preferences/brand_new/check", json={"ladder": {"best": ["qa/a"]}}
    )
    assert answer.status_code == 200, answer.text
    assert answer.json()["tiers"]["best"]["target"] == "qa/a"


def test_check_refuses_a_bad_shape(client):
    answer = client.post(
        "/api/preferences/qa_x/check", json={"ladder": {"huge": ["qa/a"]}}
    )
    assert answer.status_code == 422
    assert "huge" in answer.text


@pytest.mark.parametrize("key", ["..x", "has%20space"])
def test_check_validates_the_key(client, key):
    answer = client.post(
        f"/api/preferences/{key}/check", json={"ladder": {"best": ["qa/a"]}}
    )
    assert answer.status_code == 422
    assert "Invalid preference key" in answer.text


def test_check_a_key_with_a_slash_never_reaches_a_check(client):
    """`../x` is two path segments: no route takes it, as no PUT does."""
    answer = client.post(
        "/api/preferences/..%2Fx/check", json={"ladder": {"best": ["qa/a"]}}
    )
    assert answer.status_code in (404, 405)


def test_check_warns_on_the_shipped_contract(client):
    answer = client.post(
        f"/api/preferences/{SHIPPED_KEY}/check", json={"ladder": {"best": ["qa/a"]}}
    )
    warnings = answer.json()["tiers"]["best"]["warnings"]["qa/a"]
    assert "does not take width" in warnings


def _hub_workflow(root, name, repo):
    path = os.path.join(root, name + ".json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(
            {
                "id": os.path.basename(name),
                "variables": {"prompt": "a cat"},
                "steps": [
                    {
                        "name": "s",
                        "pipeline": {
                            "configuration": {
                                "component_type": "StableDiffusionPipeline"
                            },
                            "from_pretrained_arguments": {"model_name": repo},
                        },
                    }
                ],
            },
            handle,
        )


def test_check_fast_makes_no_hub_call(client, tmp_path, monkeypatch):
    import huggingface_hub
    import requests

    from dw import plan_downloads

    calls = []

    def hub(*args, **_kwargs):
        calls.append(args)
        raise AssertionError("the fast check must make no Hub calls")

    for name in ("model_info", "get_hf_file_metadata", "_fetch_index"):
        monkeypatch.setattr(plan_downloads, name, hub)
    for name in ("model_info", "auth_check", "get_paths_info"):
        monkeypatch.setattr(huggingface_hub.HfApi, name, hub)
    for name in ("head", "get"):
        monkeypatch.setattr(requests, name, hub)
    _hub_workflow(str(tmp_path / "workflows"), "qa/hub", "qa-org/not-cached")
    answer = client.post(
        "/api/preferences/qa_hub/check",
        json={"ladder": {"best": ["qa/hub", "qa/a"]}, "known_blocked": ["qa/b"]},
    )
    assert answer.status_code == 200, answer.text
    best = answer.json()["tiers"]["best"]
    assert best["target"] == "qa/hub"
    assert best["notes"] == ["will download its weights"]
    assert calls == []


def test_check_skips_known_blocked_on_the_fast_check(client):
    answer = client.post(
        "/api/preferences/qa_k/check",
        json={"ladder": {"best": ["qa/a", "qa/b"]}, "known_blocked": ["qa/a"]},
    )
    best = answer.json()["tiers"]["best"]
    assert best["target"] == "qa/b"
    assert best["skipped"] == [
        {"workflow": "qa/a", "reason": "gated, no access (last full check)"}
    ]


def test_check_full_runs_the_gate_check(client, monkeypatch):
    def gate():
        def check(name, definition, arguments, child):
            if name == "qa/a":
                return "gated, no access to org/x", None
            return None, None

        return check

    monkeypatch.setattr(preferences, "downloads_gate_check", gate)
    answer = client.post(
        "/api/preferences/qa_full/check",
        json={"ladder": {"best": ["qa/a", "qa/b"]}, "full": True},
    )
    best = answer.json()["tiers"]["best"]
    assert best["target"] == "qa/b"
    assert best["skipped"][0]["reason"] == "gated, no access to org/x"
