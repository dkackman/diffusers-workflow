"""The plan a validate call answers with: what a run will execute for the
arguments given, fingerprinted so an acknowledgement can be bound to it."""

import copy
import json

import pytest

from dw.plan import build_plan
from dw.runs import new_run_id
from dw.workflow import workflow_from_definition


def definition():
    """A list-driven workflow with a stored prompt and an output reference,
    so every mutable input the fingerprint must ignore or honour is here."""
    return {
        "id": "plan_test",
        "description": "docs only",
        "summary": "docs only",
        "seed": "variable:seed",
        "cost": [{"device": "cuda", "name": "card", "vram_gb": 8, "minutes": 10}],
        "variables": {
            "seed": 1,
            "prompt": "prompt:scenic/dusk",
            "frames": 25,
            "shots": [{"name": "a", "prompt": "one"}, {"name": "b", "prompt": "two"}],
        },
        "steps": [
            {
                "name": "still",
                "pipeline": {
                    "configuration": {"component_type": "{Fake}"},
                    "from_pretrained_arguments": {"model_name": "org/still-model"},
                    "arguments": {
                        "prompt": "variable:prompt",
                        "image": "output:ltx2/Gyre/latest/still.png",
                        "num_frames": "variable:frames",
                    },
                },
            },
            {
                "name": "shot",
                "for_each": "variable:shots",
                "task": {"command": "x", "arguments": {"prompt": "item:prompt"}},
            },
        ],
    }


@pytest.fixture
def prompt_library(tmp_path):
    library = tmp_path / "prompts"
    (library / "scenic").mkdir(parents=True)
    (library / "scenic" / "dusk.json").write_text(
        json.dumps({"text": "a harbour at dusk"})
    )
    return library


@pytest.fixture
def output_root(tmp_path):
    root = tmp_path / "outputs"
    run = root / "ltx2" / "Gyre" / new_run_id({"a": 1})
    run.mkdir(parents=True)
    (run / "still.png").write_bytes(b"png")
    return root


@pytest.fixture
def plan(tmp_path, prompt_library, output_root, monkeypatch):
    """build_plan over the fixture, with the hub cache empty and the hub
    unreachable, so downloads never touch the network in these tests."""
    import dw.plan

    def offline(*a, **k):
        raise RuntimeError("offline")

    monkeypatch.setattr(dw.plan, "scan_models", lambda cache_dir=None: {"repos": []})
    monkeypatch.setattr(dw.plan, "model_info", offline)

    def make(spec=None, arguments=None, **overrides):
        spec = definition() if spec is None else spec
        candidate = workflow_from_definition(
            copy.deepcopy(spec), str(output_root), str(tmp_path), str(tmp_path)
        )
        kwargs = dict(device="cuda", prompt_dir=str(prompt_library), lookup_sizes=False)
        kwargs.update(overrides)
        return build_plan(candidate, arguments or {}, **kwargs)

    return make


PLAN_KEYS = {
    "fingerprint",
    "steps",
    "list_entries",
    "cached_steps",
    "downloads_required",
    "estimate",
}


class TestShape:
    def test_the_documented_keys(self, plan):
        answer = plan()
        assert set(answer) == PLAN_KEYS
        assert answer["fingerprint"].startswith("sha256:")
        assert len(answer["fingerprint"]) == len("sha256:") + 64
        assert answer["cached_steps"] is None

    def test_steps_counts_the_expanded_members(self, plan):
        assert plan()["steps"] == 3  # still, shot@a, shot@b

    def test_list_entries_is_the_callers_list_length(self, plan):
        shots = [{"name": n, "prompt": n} for n in "abcde"]
        answer = plan(arguments={"shots": shots})
        assert answer["list_entries"] == {"shots": 5}
        assert answer["steps"] == 6

    def test_a_literal_for_each_list_is_not_an_entry(self, plan):
        spec = definition()
        spec["steps"][1]["for_each"] = [
            {"name": "x", "prompt": "x"},
            {"name": "y", "prompt": "y"},
        ]
        del spec["variables"]["shots"]
        assert plan(spec)["list_entries"] == {}


class TestFingerprintIsStableAcross:
    def test_a_different_top_level_seed(self, plan):
        assert plan()["fingerprint"] == plan(arguments={"seed": 99})["fingerprint"]

    def test_a_different_step_seed(self, plan):
        spec = definition()
        spec["steps"][0]["seed"] = 5
        spec["steps"][0]["pipeline"]["seed"] = 5
        other = copy.deepcopy(spec)
        other["steps"][0]["seed"] = 6
        other["steps"][0]["pipeline"]["seed"] = 6
        assert plan(spec)["fingerprint"] == plan(other)["fingerprint"]

    def test_key_order(self, plan):
        spec = definition()
        reordered = {k: spec[k] for k in reversed(list(spec))}
        assert plan(spec)["fingerprint"] == plan(reordered)["fingerprint"]

    def test_description_summary_and_cost_edits(self, plan):
        spec = definition()
        spec["description"] = "rewritten"
        spec["summary"] = "rewritten"
        spec["cost"][0]["minutes"] = 99
        spec["configures"] = "templates/x"
        assert plan(spec)["fingerprint"] == plan()["fingerprint"]

    def test_a_new_run_landing_under_latest(self, plan, output_root):
        before = plan()["fingerprint"]
        newer = output_root / "ltx2" / "Gyre" / new_run_id({"b": 2})
        newer.mkdir(parents=True)
        (newer / "still.png").write_bytes(b"png2")
        assert plan()["fingerprint"] == before

    def test_argument_order(self, plan):
        a = plan(arguments={"frames": 9, "seed": 3})["fingerprint"]
        b = plan(arguments={"seed": 3, "frames": 9})["fingerprint"]
        assert a == b


class TestFingerprintChangesWith:
    def test_a_longer_list(self, plan):
        longer = [{"name": n, "prompt": n} for n in "abc"]
        assert plan()["fingerprint"] != plan(arguments={"shots": longer})["fingerprint"]

    def test_a_changed_stored_prompt(self, plan, prompt_library):
        before = plan()["fingerprint"]
        (prompt_library / "scenic" / "dusk.json").write_text(
            json.dumps({"text": "a harbour at dawn"})
        )
        assert plan()["fingerprint"] != before

    def test_a_changed_numeric_argument(self, plan):
        assert plan()["fingerprint"] != plan(arguments={"frames": 121})["fingerprint"]

    def test_a_different_asset_name(self, plan):
        spec = definition()
        spec["steps"][0]["pipeline"]["arguments"]["image"] = "asset:a.png"
        other = copy.deepcopy(spec)
        other["steps"][0]["pipeline"]["arguments"]["image"] = "asset:b.png"
        assert plan(spec)["fingerprint"] != plan(other)["fingerprint"]

    def test_a_step_added_removed_or_renamed(self, plan):
        base = plan()["fingerprint"]
        renamed = definition()
        renamed["steps"][0]["name"] = "frame"
        removed = definition()
        del removed["steps"][1]
        added = definition()
        added["steps"].append(
            {"name": "extra", "task": {"command": "x", "arguments": {}}}
        )
        prints = {
            base,
            plan(renamed)["fingerprint"],
            plan(removed)["fingerprint"],
            plan(added)["fingerprint"],
        }
        assert len(prints) == 4


def cost(device="cuda", minutes=10, per_entry=None, name="card"):
    entry = {"device": device, "name": name, "vram_gb": 8, "minutes": minutes}
    if per_entry:
        entry["per_entry"] = per_entry
    return entry


class TestEstimate:
    def test_no_cost_block_is_unknown(self, plan):
        spec = definition()
        del spec["cost"]
        assert plan(spec)["estimate"] == {
            "minutes": None,
            "basis": "unknown",
            "device": "cuda",
            "measured_on": None,
            "partial": False,
        }

    def test_an_empty_cost_list_is_unknown(self, plan):
        spec = definition()
        spec["cost"] = []
        assert plan(spec)["estimate"]["basis"] == "unknown"

    def test_the_serving_devices_entry_is_the_catalog_figure(self, plan):
        spec = definition()
        spec["cost"] = [cost("mps", 40, name="M2"), cost("cuda", 10, name="4090")]
        assert plan(spec)["estimate"] == {
            "minutes": 10.0,
            "basis": "catalog",
            "device": "cuda",
            "measured_on": "4090",
            "partial": False,
        }

    def test_another_devices_entry_is_reported_as_such(self, plan):
        spec = definition()
        spec["cost"] = [cost("mps", 40, name="M2")]
        assert plan(spec)["estimate"] == {
            "minutes": 40.0,
            "basis": "other_device",
            "device": "cuda",
            "measured_on": "M2",
            "partial": False,
        }

    def test_per_entry_scales_by_the_callers_list(self, plan):
        spec = definition()
        # 10 minutes for the 2-entry default, of which 3 per entry: 4 fixed
        spec["cost"] = [
            cost("cuda", 10, {"variable": "shots", "minutes": 3, "entries": 2})
        ]
        shots = [{"name": n, "prompt": n} for n in "abcde"]
        answer = plan(spec, arguments={"shots": shots})["estimate"]
        assert answer["minutes"] == 4 + 3 * 5
        assert answer["basis"] == "per_entry"

    def test_per_entry_floors_at_zero(self, plan):
        spec = definition()
        spec["cost"] = [
            cost("cuda", 1, {"variable": "shots", "minutes": 3, "entries": 2})
        ]
        # 1 - 3 * 2 + 3 * 1 is negative: a shorter list than was measured
        one = [{"name": "a", "prompt": "a"}]
        assert plan(spec, arguments={"shots": one})["estimate"]["minutes"] == 0.0

    def test_per_entry_naming_no_list_falls_back_to_catalog(self, plan):
        spec = definition()
        spec["cost"] = [
            cost("cuda", 10, {"variable": "other", "minutes": 3, "entries": 2})
        ]
        answer = plan(spec)["estimate"]
        assert (answer["minutes"], answer["basis"]) == (10.0, "catalog")

    def test_other_device_beats_per_entry(self, plan):
        spec = definition()
        spec["cost"] = [
            cost("mps", 10, {"variable": "shots", "minutes": 3, "entries": 2})
        ]
        answer = plan(spec)["estimate"]
        assert (answer["minutes"], answer["basis"]) == (10.0, "other_device")

    def test_minutes_is_rounded_to_one_decimal(self, plan):
        spec = definition()
        spec["cost"] = [cost("cuda", 10.04)]
        assert plan(spec)["estimate"]["minutes"] == 10.0


def composing(child_path):
    return {
        "id": "parent",
        "cost": [cost("cuda", 2)],
        "steps": [
            {"name": "own", "task": {"command": "x", "arguments": {}}},
            {"name": "child", "workflow": {"path": child_path, "arguments": {}}},
        ],
    }


class TestSubWorkflowEstimate:
    def test_a_childs_catalog_cost_is_added(self, plan, tmp_path):
        child = {"id": "child", "cost": [cost("cuda", 5)], "steps": []}
        (tmp_path / "child.json").write_text(json.dumps(child))
        answer = plan(composing("child.json"))["estimate"]
        assert (answer["minutes"], answer["partial"]) == (7.0, False)

    def test_a_child_without_a_cost_makes_the_estimate_partial(self, plan, tmp_path):
        (tmp_path / "child.json").write_text(json.dumps({"id": "child", "steps": []}))
        answer = plan(composing("child.json"))["estimate"]
        assert (answer["minutes"], answer["partial"]) == (2.0, True)

    def test_an_unreadable_child_makes_the_estimate_partial(self, plan):
        answer = plan(composing("missing.json"))["estimate"]
        assert (answer["minutes"], answer["partial"]) == (2.0, True)

    def test_a_builtin_adds_nothing_and_is_not_partial(self, plan):
        answer = plan(composing("builtin:text-to-image.json"))["estimate"]
        assert (answer["minutes"], answer["partial"]) == (2.0, False)

    def test_a_child_measured_on_another_device_is_still_added(self, plan, tmp_path):
        child = {"id": "child", "cost": [cost("mps", 5)], "steps": []}
        (tmp_path / "child.json").write_text(json.dumps(child))
        answer = plan(composing("child.json"))["estimate"]
        assert answer["minutes"] == 7.0
        # the parent's own basis is what is reported
        assert answer["basis"] == "catalog"
