"""Model preferences: `preference:<key>` sub-workflow paths (dw/preferences.py, #1626).

Pins the resolver's fit rules and refusals, the tier walk, the overlay, the
pin replay, and that every reader of `workflow.path` follows the resolved
child. No GPU, no model download, no network.
"""

import json
import os

import pytest

from dw import preferences
from dw.library import SubWorkflowNotFound
from dw.preferences import Host, PreferenceError
from dw.server.job_record import RERUN_SPEC_KEYS
from dw.worker_protocol import Execute
from dw.workflow import workflow_from_definition

from tests.test_examples import REPO_ROOT

WORKFLOWS = os.path.join(REPO_ROOT, "workflows")
PATH = "preference:text_to_image"
CUDA = Host(device_type="cuda", smallest_card_gb=48, host_ram_gb=128)


def definition(name, cost=None, **extra):
    return {"id": name, "cost": cost or [], "variables": {"prompt": "x"}, **extra}


def store(ladder, sources=None, default_tier="standard"):
    return {
        "default_tier": default_tier,
        "preferences": {"k": ladder},
        "sources": sources or {},
    }


def opener(catalog):
    """A fake open_target over {name: definition}."""

    def open_target(name):
        if name not in catalog:
            raise SubWorkflowNotFound(name, [])
        return catalog[name], object()

    return open_target


def resolve(ladder, catalog, host=CUDA, tier="best", path="preference:k", **kw):
    return preferences.resolve(
        path, tier, host, opener(catalog), store(ladder, **kw.pop("s", {})), **kw
    )


# --- 1. the resolver ------------------------------------------------------


def test_first_target_that_fits_runs():
    found = resolve({"best": ["a", "b"]}, {"a": definition("a"), "b": definition("b")})
    assert found.target == "a"
    assert found.skipped == []
    assert found.source == "shipped"


def test_devices_skip_names_cuda_only():
    ladder = {"best": [{"workflow": "a", "devices": ["cuda"]}, "b"]}
    catalog = {"a": definition("a"), "b": definition("b")}
    found = resolve(ladder, catalog, host=Host(device_type="mps"))
    assert found.target == "b"
    assert found.skipped == [{"workflow": "a", "reason": "cuda-only"}]


def test_host_ram_skip():
    catalog = {"a": definition("a", host_ram_gb=64), "b": definition("b")}
    found = resolve(
        {"best": ["a", "b"]}, catalog, host=Host("cuda", 48, host_ram_gb=16)
    )
    assert found.target == "b"
    assert "declares about 64 GB of host RAM" in found.skipped[0]["reason"]


def test_host_ram_fit_keeps_the_gates_margin():
    """The resolver's verdict is admission's (`host_too_small`): a target
    declaring exactly the host's RAM is refused there, so it is skipped
    here, never resolved into a refusal."""
    from dw.host_ram_declared import host_too_small

    catalog = {"a": definition("a", host_ram_gb=64), "b": definition("b")}
    host = Host("cuda", 48, host_ram_gb=64)
    found = resolve({"best": ["a", "b"]}, catalog, host=host)
    assert found.target == "b"
    refusal = host_too_small(catalog["a"], {}, 64 * 1024)
    assert refusal and refusal in found.skipped[0]["reason"]
    # With room for the margin it fits
    assert resolve({"best": ["a"]}, catalog, host=Host("cuda", 48, 72)).target == "a"


def test_host_ram_from_a_tier_variable_is_read_with_the_arguments():
    tiered = definition(
        "a",
        host_ram_gb="variable:ram",
        variables={"prompt": "x", "ram": 8},
    )
    ladder = {"best": [{"workflow": "a", "arguments": {"ram": 200}}, "b"]}
    found = resolve(ladder, {"a": tiered, "b": definition("b")})
    assert found.target == "b"
    assert "200 GB" in found.skipped[0]["reason"]


def test_vram_skip_against_smallest_card():
    big = definition("a", cost=[{"device": "cuda", "vram_gb": 40, "minutes": 1}])
    catalog = {"a": big, "b": definition("b")}
    found = resolve({"best": ["a", "b"]}, catalog, host=Host("cuda", 24, 128))
    assert found.target == "b"
    assert "40 GB VRAM on cuda" in found.skipped[0]["reason"]


def test_vram_cost_on_a_numbered_card_counts():
    """`cuda:0` is a cuda entry: the dispatcher's figure splits the ordinal."""
    big = definition("a", cost=[{"device": "cuda:0", "vram_gb": 40, "minutes": 1}])
    catalog = {"a": big, "b": definition("b")}
    found = resolve({"best": ["a", "b"]}, catalog, host=Host("cuda", 24, 128))
    assert found.target == "b"
    assert "40 GB VRAM on cuda" in found.skipped[0]["reason"]


def test_declared_vram_estimate_is_the_figure():
    """A `vram_estimate` (768x512x121 voxels at 200 B + 20 GB = ~28.9 GB)
    is what dispatch holds a card to, above a smaller measured cost."""
    estimated = definition(
        "a",
        cost=[{"device": "cuda", "vram_gb": 20, "minutes": 1}],
        variables={"prompt": "x", "width": 768, "height": 512, "num_frames": 121},
        vram_estimate={
            "voxel_variables": ["width", "height", "num_frames"],
            "base_gb": 20,
            "bytes_per_voxel": 200,
        },
    )
    catalog = {"a": estimated, "b": definition("b")}
    found = resolve({"best": ["a", "b"]}, catalog, host=Host("cuda", 24, 128))
    assert found.target == "b"
    assert "28.86" in found.skipped[0]["reason"]
    assert resolve({"best": ["a"]}, catalog, host=Host("cuda", 32, 128)).target == "a"


def test_vram_cost_for_another_device_is_not_a_skip():
    other = definition("a", cost=[{"device": "mps", "vram_gb": 99, "minutes": 1}])
    assert resolve({"best": ["a"]}, {"a": other}, host=Host("cuda", 24, 128))


def test_target_with_no_cost_entry_is_not_skipped():
    found = resolve({"best": ["a"]}, {"a": definition("a")}, host=Host("cuda", 1, 1))
    assert found.target == "a"


def test_gate_skip():
    def gate(name, _definition, _arguments, _child):
        return ("gated, no access to org/x", None) if name == "a" else (None, None)

    host = Host("cuda", 48, 128, gate_check=gate)
    catalog = {"a": definition("a"), "b": definition("b")}
    found = resolve({"best": ["a", "b"]}, catalog, host=host)
    assert found.target == "b"
    assert found.skipped[0]["reason"] == "gated, no access to org/x"


def test_not_downloaded_is_a_note_not_a_skip():
    host = Host("cuda", 48, 128, gate_check=lambda *a: (None, "will download 3.0 GB"))
    found = resolve({"best": ["a"]}, {"a": definition("a")}, host=host)
    assert found.target == "a"
    assert found.skipped == []
    assert found.notes == ["will download 3.0 GB"]
    assert found.line().endswith("; will download 3.0 GB")


def test_failing_gate_probe_does_not_skip():
    def gate(*args):
        raise RuntimeError("hub down")

    host = Host("cuda", 48, 128, gate_check=gate)
    assert resolve({"best": ["a"]}, {"a": definition("a")}, host=host).target == "a"


def test_stale_targets_are_all_named():
    with pytest.raises(PreferenceError) as raised:
        resolve({"best": ["gone1", "a", "gone2"]}, {"a": definition("a")})
    message = str(raised.value)
    assert "'gone1'" in message and "'gone2'" in message
    assert "not in the catalog" in message


def test_no_fit_names_every_skip():
    ladder = {"best": [{"workflow": "a", "devices": ["cuda"]}, "b"]}
    heavy = definition("b", cost=[{"device": "mps", "vram_gb": 90, "minutes": 1}])
    catalog = {"a": definition("a"), "b": heavy}
    with pytest.raises(PreferenceError) as raised:
        resolve(ladder, catalog, host=Host("mps", 18, 64))
    message = str(raised.value)
    assert "no target fits" in message
    assert "a: cuda-only" in message
    assert "b: needs 90 GB VRAM on mps" in message


def test_unknown_key_names_known_keys():
    with pytest.raises(PreferenceError) as raised:
        resolve({"best": ["a"]}, {"a": definition("a")}, path="preference:nope")
    assert "Unknown preference 'nope'" in str(raised.value)
    assert "known keys: k" in str(raised.value)


@pytest.mark.parametrize("path", ["preference:../x", "preference:bad key"])
def test_bad_key_refused(path):
    with pytest.raises(PreferenceError, match="Invalid preference key"):
        resolve({"best": ["a"]}, {"a": definition("a")}, path=path)


def test_bad_tier_refused():
    with pytest.raises(PreferenceError, match="Unknown tier 'ultra'"):
        resolve({"best": ["a"]}, {"a": definition("a")}, tier="ultra")


def test_bad_target_argument_refused():
    ladder = {"best": [{"workflow": "a", "arguments": {"nope": 1}}]}
    with pytest.raises(PreferenceError, match="arguments are refused"):
        resolve(ladder, {"a": definition("a")})


def test_tier_without_targets_refused():
    with pytest.raises(PreferenceError, match="no targets for tier draft"):
        resolve({"best": ["a"]}, {"a": definition("a")}, tier="draft")


def test_plan_line_format():
    ladder = {"best": [{"workflow": "a", "devices": ["cuda"]}, "b"]}
    catalog = {"a": definition("a"), "b": definition("b")}
    found = resolve(ladder, catalog, host=Host("mps"), path="preference:k")
    assert (
        found.line()
        == "preference:k (tier best) -> b; skipped a: cuda-only; source: shipped"
    )


def test_pin_replays_target_instead_of_resolving():
    pin = {
        "target": "b",
        "arguments": {"prompt": "pinned"},
        "skipped": [{"workflow": "a", "reason": "cuda-only"}],
        "source": "server",
    }
    catalog = {"a": definition("a"), "b": definition("b")}
    found = preferences.resolve(
        "preference:k", "best", CUDA, opener(catalog), None, pinned=pin
    )
    assert (found.target, found.arguments, found.source) == (
        "b",
        {"prompt": "pinned"},
        "server",
    )
    assert found.pinned and found.line().endswith("source: server (pinned)")
    assert found.pin() == pin


def test_pin_with_missing_target_refused():
    pin = {"target": "gone", "arguments": {}, "skipped": [], "source": "shipped"}
    with pytest.raises(PreferenceError, match="pinned target 'gone' is no longer"):
        preferences.resolve("preference:k", "best", CUDA, opener({}), None, pinned=pin)


def test_pin_key_is_path_at_tier():
    assert preferences.pin_key(PATH, "best") == "preference:text_to_image@best"


# --- 3. the overlay -------------------------------------------------------


def write(path, document):
    path.write_text(json.dumps(document))
    return str(path)


def test_overlay_key_replaces_whole_shipped_key(tmp_path):
    shipped = write(
        tmp_path / "s.json",
        {
            "default_tier": "standard",
            "preferences": {
                "k": {"draft": ["a"], "best": ["b"]},
                "other": {"draft": ["c"]},
            },
        },
    )
    overlay = write(
        tmp_path / "o.json",
        {"default_tier": "best", "preferences": {"k": {"best": ["z"]}}},
    )
    loaded = preferences.load_preferences(shipped=shipped, overlay=overlay)
    assert loaded["preferences"]["k"] == {"best": ["z"]}  # draft not merged in
    assert loaded["preferences"]["other"] == {"draft": ["c"]}
    assert loaded["sources"] == {"k": "server", "other": "shipped"}
    assert loaded["default_tier"] == "best"


def test_overlay_absent_or_broken_keeps_shipped(tmp_path):
    shipped = write(tmp_path / "s.json", {"preferences": {"k": {"draft": ["a"]}}})
    broken = tmp_path / "o.json"
    broken.write_text("{nope")
    for overlay in (str(tmp_path / "missing.json"), str(broken)):
        loaded = preferences.load_preferences(shipped=shipped, overlay=overlay)
        assert loaded["sources"] == {"k": "shipped"}
        assert loaded["default_tier"] == "standard"


# --- 2. the tier walk through a real Workflow -----------------------------


@pytest.fixture
def cuda_session(tmp_path):
    session = preferences.Session(
        host=Host(device_type="cuda", smallest_card_gb=48, host_ram_gb=128)
    )
    token = preferences.activate(session)
    yield session
    preferences.deactivate(token)


def parent(tmp_path, variables=None, path=PATH, arguments=None):
    step = {"name": "gen", "workflow": {"path": path}}
    if arguments is not None:
        step["workflow"]["arguments"] = arguments
    spec = {"id": "parent", "variables": variables or {}, "steps": [step]}
    return workflow_from_definition(spec, str(tmp_path), WORKFLOWS, None)


def target_of(workflow, arguments=None):
    return workflow.expanded_definition(arguments)["steps"][0]["workflow"]["path"]


def test_tier_defaults_to_the_stores_default(tmp_path, cuda_session):
    assert target_of(parent(tmp_path)) == "models/z-image"  # standard


def test_folded_tier_variable_beats_default(tmp_path, cuda_session):
    assert target_of(parent(tmp_path, {"tier": "draft"})) == "templates/text-to-image"


def test_caller_argument_folds_into_tier(tmp_path, cuda_session):
    workflow = parent(tmp_path, {"tier": "draft"})
    assert target_of(workflow, {"tier": "best"}) == "models/minimax-h3-image"


def test_step_tier_argument_beats_variable(tmp_path, cuda_session):
    workflow = parent(tmp_path, {"tier": "draft"}, arguments={"tier": "best"})
    assert target_of(workflow) == "models/minimax-h3-image"
    # the step's tier picks the rung and is not handed to a target without one
    arguments = workflow.expanded_definition()["steps"][0]["workflow"]["arguments"]
    assert "tier" not in arguments


def test_tier_walks_up_to_composing_parent(tmp_path, cuda_session):
    outer = parent(tmp_path, {"tier": "draft"}, path="templates/compose-workflows")
    outer.expanded_definition()  # sets the tier a child inherits
    assert outer._effective_tier == "draft"
    child = workflow_from_definition(
        {
            "id": "child",
            "steps": [{"name": "g", "workflow": {"path": PATH}}],
        },
        str(tmp_path),
        WORKFLOWS,
        None,
    )
    child._inherited_tier = outer._effective_tier
    assert target_of(child) == "templates/text-to-image"


def test_bad_tier_refused_at_step_path(tmp_path, cuda_session):
    workflow = parent(tmp_path, arguments={"tier": "ultra"})
    errors = workflow.validation_errors()
    assert errors[0]["path"] == "steps[0].workflow.path"
    assert "Unknown tier 'ultra'" in errors[0]["message"]


def test_unknown_key_refused_at_step_path(tmp_path, cuda_session):
    errors = parent(tmp_path, path="preference:nope").validation_errors()
    assert errors[0]["path"] == "steps[0].workflow.path"
    assert "Unknown preference 'nope'" in errors[0]["message"]
    assert "text_to_image" in errors[0]["message"]


def test_variable_holding_a_preference_resolves(tmp_path, cuda_session):
    workflow = parent(
        tmp_path, {"model": PATH}, path="variable:model", arguments={"tier": "best"}
    )
    assert target_of(workflow) == "models/minimax-h3-image"
    assert len(workflow.preference_lines()) == 1


def test_overriding_that_variable_with_a_catalog_path_has_no_preference(
    tmp_path, cuda_session
):
    workflow = parent(tmp_path, {"model": PATH}, path="variable:model")
    assert target_of(workflow, {"model": "models/z-image"}) == "models/z-image"
    assert workflow.preference_lines() == []
    assert workflow.preference_pins() == {}


def test_step_arguments_merge_over_target_over_defaults(tmp_path, cuda_session):
    cuda_session.store = {
        "default_tier": "standard",
        "preferences": {
            "text_to_image": {
                "standard": [
                    {
                        "workflow": "models/z-image",
                        "arguments": {"width": 512, "height": 512},
                    }
                ]
            }
        },
        "sources": {},
    }
    workflow = parent(tmp_path, arguments={"height": 768})
    arguments = workflow.expanded_definition()["steps"][0]["workflow"]["arguments"]
    assert arguments == {"width": 512, "height": 768}


def test_rung_skips_cuda_only_target_on_mps(tmp_path):
    session = preferences.Session(host=Host("mps", 18, 64))
    token = preferences.activate(session)
    try:
        workflow = parent(tmp_path, arguments={"tier": "best"})
        assert target_of(workflow) == "models/qwen-image-2.1"
        assert workflow.preference_lines() == [
            "preference:text_to_image (tier best) -> models/qwen-image-2.1; "
            "skipped models/minimax-h3-image: cuda-only; source: shipped"
        ]
    finally:
        preferences.deactivate(token)


# --- 4. readers follow the resolved child ---------------------------------


@pytest.fixture
def standard(tmp_path, cuda_session):
    return parent(tmp_path, arguments={"prompt": "a cat"})


def test_validation_accepts_the_resolved_child(standard):
    assert standard.validation_errors() == []
    standard.validate()


def test_downloads_required_sees_the_target(standard, monkeypatch):
    import dw.plan_downloads

    monkeypatch.setattr(
        dw.plan_downloads, "scan_models", lambda cache_dir=None: {"repos": []}
    )

    def offline(*a, **k):
        raise RuntimeError("offline")

    monkeypatch.setattr(dw.plan_downloads, "model_info", offline)
    required = dw.plan_downloads.downloads_required(
        standard.expanded_definition(), WORKFLOWS, None, None, False
    )
    assert any("z-image" in (entry.get("repo") or "").lower() for entry in required)


def test_pipeline_ownership_sees_the_target(standard):
    from dw.pipeline_ownership import _loaded_keys

    keys = _loaded_keys(standard, standard.expanded_definition())
    assert keys  # the child's pipeline keys, not nothing for a bad path


def test_argument_warnings_follow_the_target(tmp_path, cuda_session):
    from dw.argument_warnings import sub_workflow_argument_warnings

    workflow = parent(tmp_path, arguments={"not_a_variable": 1})
    warnings = sub_workflow_argument_warnings(workflow, workflow.expanded_definition())
    assert len(warnings) == 1
    assert "'models/z-image' declares no variable 'not_a_variable'" in warnings[0]


def test_plan_carries_preference_lines_and_prices_the_target(
    standard, monkeypatch, tmp_path
):
    import dw.plan_downloads
    from dw.plan import build_plan

    monkeypatch.setattr(
        dw.plan_downloads, "scan_models", lambda cache_dir=None: {"repos": []}
    )
    monkeypatch.setattr(
        dw.plan_downloads,
        "model_info",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")),
    )
    plan = build_plan(
        standard,
        {},
        device="cuda",
        prompt_dir=str(tmp_path),
        lookup_sizes=False,
    )
    assert plan["preferences"] == [
        "preference:text_to_image (tier standard) -> models/z-image; source: shipped"
    ]
    assert plan["downloads_required"]
    # priced from the target's own cost entry, not unpriceable
    assert plan["estimate"]["minutes"] is not None


# --- 5. the pin ------------------------------------------------------------


def test_recorded_prefers_the_sessions_resolutions(standard, cuda_session):
    standard.expanded_definition()
    assert preferences.recorded(standard) == cuda_session.resolved
    assert list(cuda_session.resolved) == ["preference:text_to_image@standard"]


def test_recorded_falls_back_to_the_workflows_pins(standard):
    standard.expanded_definition()  # inside cuda_session, then drop it
    token = preferences.activate(preferences.Session())
    try:
        assert preferences.recorded(standard) == standard.preference_pins()
        assert (
            standard.preference_pins()["preference:text_to_image@standard"]["target"]
            == "models/z-image"
        )
    finally:
        preferences.deactivate(token)


def test_recorded_without_a_session_or_pins():
    class Bare:
        pass

    assert preferences.recorded(Bare()) == {}


def test_session_pin_replays_and_other_tier_resolves_fresh(tmp_path):
    pin = {
        "preference:text_to_image@standard": {
            "target": "models/z-image-sdnq",
            "arguments": {},
            "skipped": [],
            "source": "server",
        }
    }
    session = preferences.Session(host=CUDA, pinned=pin)
    token = preferences.activate(session)
    try:
        assert target_of(parent(tmp_path)) == "models/z-image-sdnq"
        assert (
            session.resolved["preference:text_to_image@standard"]["source"] == "server"
        )
        fresh = parent(tmp_path, arguments={"tier": "draft"})
        assert target_of(fresh) == "templates/text-to-image"
    finally:
        preferences.deactivate(token)


def test_pinned_target_gone_is_refused_at_step_path(tmp_path):
    pin = {
        "preference:text_to_image@standard": {
            "target": "models/not-in-catalog",
            "arguments": {},
            "skipped": [],
            "source": "shipped",
        }
    }
    token = preferences.activate(preferences.Session(host=CUDA, pinned=pin))
    try:
        errors = parent(tmp_path).validation_errors()
        assert errors[0]["path"] == "steps[0].workflow.path"
        assert "no longer in the catalog" in errors[0]["message"]
    finally:
        preferences.deactivate(token)


def test_rerun_spec_keys_carry_preferences():
    assert "preferences" in RERUN_SPEC_KEYS


def test_execute_round_trips_preferences():
    pins = {"preference:k@best": {"target": "a", "arguments": {}, "skipped": []}}
    message = Execute(
        definition={"id": "x", "steps": []},
        file_spec="/w/x.json",
        source="path",
        workflow_dir="/w",
        output_dir="/out",
        arguments={},
        log_level="INFO",
        preferences=pins,
    )
    wire = message.to_wire()
    assert wire["preferences"] == pins
    assert Execute.from_wire(wire).preferences == pins


def test_execute_without_preferences_leaves_the_key_off_the_wire():
    message = Execute(
        definition={"id": "x", "steps": []},
        file_spec="/w/x.json",
        source="path",
        workflow_dir="/w",
        output_dir="/out",
        arguments={},
        log_level="INFO",
    )
    assert "preferences" not in message.to_wire()


# --- 6. the shipped file -----------------------------------------------------


def shipped_targets():
    with open(preferences.SHIPPED_PATH, encoding="utf-8") as f:
        document = json.load(f)
    for key, ladder in document["preferences"].items():
        for tier, entries in ladder.items():
            for entry in entries:
                yield key, tier, entry if isinstance(entry, str) else entry["workflow"]


def test_every_shipped_target_is_a_catalog_workflow():
    from dw.library import resolve_sub_workflow_reference

    targets = list(shipped_targets())
    assert targets
    for key, tier, name in targets:
        resolved, _ = resolve_sub_workflow_reference(name, WORKFLOWS, None)
        assert os.path.isfile(resolved), f"{key}/{tier}: {name}"


def test_shipped_tiers_are_valid():
    store = preferences.load_preferences(overlay=os.devnull)
    assert store["default_tier"] in preferences.TIERS
    for ladder in store["preferences"].values():
        assert set(ladder) <= set(preferences.TIERS)


def test_shipped_file_is_the_packaged_defaults():
    """The ladders live in the package (#1625's file), not in the examples
    tree a library lists as workflows."""
    from dw.library import workflow_names

    assert preferences.SHIPPED_PATH.name == "preference_defaults.json"
    assert preferences.SHIPPED_PATH.parent.name == "dw"
    assert os.path.isfile(preferences.SHIPPED_PATH)
    assert not os.path.exists(os.path.join(WORKFLOWS, "preferences.json"))
    assert "preferences" not in workflow_names(WORKFLOWS)
