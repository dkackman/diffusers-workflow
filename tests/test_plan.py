"""The plan a validate call answers with: what a run will execute for the
arguments given, fingerprinted so an acknowledgement can be bound to it."""

import copy
import json

import pytest

from dw.plan import build_plan, gate_warnings, unseeded_cache_warnings
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
                # Both steps declare a result: a step that saves nothing and
                # which nothing reads does not run at all now (#122), and
                # these fixtures are about the plan rather than about elision
                "result": {"content_type": "image/png"},
            },
            {
                "name": "shot",
                "for_each": "variable:shots",
                "task": {"command": "x", "arguments": {"prompt": "item:prompt"}},
                "result": {"content_type": "video/mp4"},
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
    "elided_steps",
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
    def test_an_edited_composed_child(self, plan, tmp_path):
        child = {"id": "child", "steps": []}
        (tmp_path / "child.json").write_text(json.dumps(child))
        spec = composing("child.json")
        before = plan(spec)["fingerprint"]
        child["steps"].append({"name": "x", "task": {"command": "x", "arguments": {}}})
        (tmp_path / "child.json").write_text(json.dumps(child))
        assert plan(spec)["fingerprint"] != before

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
            "unpriced": [],
            "runs": None,
            "cached_minutes": None,
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
            "unpriced": [],
            "runs": None,
            "cached_minutes": None,
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
            "unpriced": [],
            "runs": None,
            "cached_minutes": None,
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

    def test_a_longer_list_re_prices_the_catalog_figure(self, plan):
        """The failure #85 was re-opened for: 10 minutes measured on the
        2-shot default was quoted verbatim for a 10-shot run. Linear over
        the entry count, and labelled `derived` so it is not read as a
        measurement."""
        spec = definition()
        shots = [{"name": f"s{n}", "prompt": "x"} for n in range(10)]
        answer = plan(spec, arguments={"shots": shots})["estimate"]
        assert (answer["minutes"], answer["basis"]) == (50.0, "derived")

    def test_a_shorter_list_re_prices_downward(self, plan):
        spec = definition()
        one = [{"name": "a", "prompt": "a"}]
        answer = plan(spec, arguments={"shots": one})["estimate"]
        assert (answer["minutes"], answer["basis"]) == (5.0, "derived")

    def test_the_stored_list_is_still_the_catalog_figure(self, plan):
        """The figure was measured with these defaults, so nothing is
        extrapolated when the caller does not change them."""
        answer = plan(definition())["estimate"]
        assert (answer["minutes"], answer["basis"]) == (10.0, "catalog")

    def test_a_measured_per_entry_rate_beats_the_extrapolation(self, plan):
        spec = definition()
        spec["cost"] = [
            cost("cuda", 10, {"variable": "shots", "minutes": 3, "entries": 2})
        ]
        shots = [{"name": f"s{n}", "prompt": "x"} for n in range(10)]
        answer = plan(spec, arguments={"shots": shots})["estimate"]
        assert (answer["minutes"], answer["basis"]) == (34.0, "per_entry")

    def test_two_changed_lists_withhold_the_figure(self, plan):
        """Nothing honest to extrapolate along, so the number is withheld
        rather than quoted for one of the two lists."""
        spec = definition()
        spec["variables"]["angles"] = [{"name": "wide"}]
        spec["steps"].append(
            {
                "name": "angle",
                "for_each": "variable:angles",
                "task": {"command": "x", "arguments": {"name": "item:name"}},
            }
        )
        arguments = {
            "shots": [{"name": f"s{n}", "prompt": "x"} for n in range(4)],
            "angles": [{"name": "wide"}, {"name": "tight"}],
        }
        answer = plan(spec, arguments=arguments)["estimate"]
        assert (answer["minutes"], answer["basis"]) == (None, "unknown")

    def test_another_devices_figure_is_not_extrapolated(self, plan):
        """`other_device` already says the figure is not this machine's -
        re-pricing it would dress a guess as arithmetic."""
        spec = definition()
        spec["cost"] = [cost("mps", 40, name="M2")]
        shots = [{"name": f"s{n}", "prompt": "x"} for n in range(10)]
        answer = plan(spec, arguments={"shots": shots})["estimate"]
        assert (answer["minutes"], answer["basis"]) == (40.0, "other_device")

    def test_minutes_is_rounded_to_one_decimal(self, plan):
        spec = definition()
        spec["cost"] = [cost("cuda", 10.04)]
        assert plan(spec)["estimate"]["minutes"] == 10.0

    def test_a_shifted_scalar_cost_driver_falls_back_to_unknown(self, plan):
        """The catalog figure was measured at frames=25; a caller who
        overrides a declared cost_driver away from that value is not
        describing the run the figure was measured for, and `_repriced`
        only re-prices a for_each list's length, not a bare variable (#267)."""
        spec = definition()
        spec["cost_drivers"] = ["frames"]
        answer = plan(spec, arguments={"frames": 9})["estimate"]
        assert (answer["minutes"], answer["basis"]) == (None, "unknown")

    def test_an_unshifted_scalar_cost_driver_still_quotes_the_catalog_figure(
        self, plan
    ):
        spec = definition()
        spec["cost_drivers"] = ["frames"]
        answer = plan(spec, arguments={"frames": 25})["estimate"]
        assert (answer["minutes"], answer["basis"]) == (10.0, "catalog")

    def test_a_list_driver_still_reprices_instead_of_going_unknown(self, plan):
        """A list cost_driver's length change is `_repriced`'s job already -
        declaring it a driver must not route it through the new scalar
        fallback and withhold the figure instead."""
        spec = definition()
        spec["cost_drivers"] = ["shots"]
        shots = [{"name": f"s{n}", "prompt": "x"} for n in range(10)]
        answer = plan(spec, arguments={"shots": shots})["estimate"]
        assert (answer["minutes"], answer["basis"]) == (50.0, "derived")

    def test_an_undeclared_variable_shift_is_not_a_driver_shift(self, plan):
        """Only a declared cost_driver triggers the fallback - any other
        variable overridden away from its default is none of this rule's
        business."""
        spec = definition()
        spec["cost_drivers"] = ["shots"]
        answer = plan(spec, arguments={"frames": 9})["estimate"]
        assert (answer["minutes"], answer["basis"]) == (10.0, "catalog")


def observed(minutes=8.04, runs=11, device="cuda", name="RTX 3090", warm=False):
    """This box's history for the workflow, as `observed_for` reports it."""
    block = {"device": device, "name": name, "runs": runs}
    prefix = "warm" if warm else "cold"
    block[f"{prefix}_minutes"] = minutes
    block[f"{prefix}_runs"] = runs
    return block


class TestObservedEstimate:
    """#154: a figure this box measured beats one a maintainer curated, and
    `basis` says which was used - `unknown` has to mean nobody has a number,
    not nobody wrote one down."""

    def test_history_is_quoted_when_there_is_no_cost_block(self, plan):
        spec = definition()
        del spec["cost"]
        answer = plan(spec, observed=observed())["estimate"]
        assert answer == {
            "minutes": 8.0,
            "basis": "observed",
            "device": "cuda",
            "measured_on": "RTX 3090",
            "partial": False,
            "unpriced": [],
            "runs": 11,
            "cached_minutes": None,
        }

    def test_history_beats_a_curated_figure(self, plan):
        """The catalog says 10; this box has run it eleven times at 8."""
        answer = plan(definition(), observed=observed())["estimate"]
        assert (answer["minutes"], answer["basis"]) == (8.0, "observed")

    def test_a_callable_is_asked_with_the_run_s_arguments(self, plan):
        asked = []

        def history(arguments):
            asked.append(arguments)
            return observed()

        shots = [{"name": "a", "prompt": "a"}]
        plan(definition(), arguments={"shots": shots}, observed=history)
        assert asked == [{"shots": shots}]

    def test_a_callable_that_raises_falls_back_to_the_cost_block(self, plan):
        def boom(arguments):
            raise RuntimeError("no history")

        answer = plan(definition(), observed=boom)["estimate"]
        assert (answer["minutes"], answer["basis"]) == (10.0, "catalog")

    def test_no_history_falls_back_to_the_cost_block(self, plan):
        answer = plan(definition(), observed=lambda arguments: None)["estimate"]
        assert (answer["minutes"], answer["basis"]) == (10.0, "catalog")

    def test_only_warm_runs_are_not_quoted(self, plan):
        """A warm run had the weights resident; quoting it as the cost of a
        run that has to load them under-quotes by the load."""
        answer = plan(definition(), observed=observed(warm=True))["estimate"]
        assert (answer["minutes"], answer["basis"]) == (10.0, "catalog")

    def test_another_backends_history_is_not_quoted(self, plan):
        answer = plan(definition(), observed=observed(device="mps"))["estimate"]
        assert (answer["minutes"], answer["basis"]) == (10.0, "catalog")

    def test_a_composed_child_is_not_added_to_an_observed_figure(self, plan, tmp_path):
        """An observed run already ran the child, so adding the child's
        curated cost would count it twice."""
        (tmp_path / "child.json").write_text(
            json.dumps({"id": "child", "cost": [cost("cuda", 5)], "steps": []})
        )
        answer = plan(composing("child.json"), observed=observed(minutes=6))["estimate"]
        assert answer["minutes"] == 6.0
        assert answer["partial"] is False
        assert answer["unpriced"] == []


class TestLowConfidenceObservedEstimate:
    """#301: a single run is not the same statistical basis as a dozen - an
    observed figure below SMALL_N_THRESHOLD runs is blended toward the
    curated cost when one exists, and flagged `low_confidence` when there
    is nothing curated to blend toward, rather than being quoted with the
    same authority as a figure with runs to spare."""

    def test_a_single_run_blends_toward_the_curated_figure(self, plan):
        """Catalog says 10; one observed run says 6 - the answer should
        sit between them rather than repeat the thin point figure."""
        answer = plan(definition(), observed=observed(minutes=6, runs=1))["estimate"]
        assert answer["basis"] == "observed"
        assert answer["runs"] == 1
        assert 6.0 < answer["minutes"] < 10.0
        assert "low_confidence" not in answer

    def test_a_blended_estimate_says_so(self, plan):
        """#319: a blend is still `basis: observed`, so it needs its own
        marker to be distinguishable from a raw, full-authority figure -
        and it needs to name the two numbers it sat between, so a caller
        can reconcile it against `list_workflows`' own `observed_minutes`."""
        answer = plan(definition(), observed=observed(minutes=6, runs=1))["estimate"]
        assert answer["tempered"] is True
        assert answer["observed_minutes"] == 6.0
        assert answer["curated_minutes"] == 10.0

    def test_two_runs_blend_less_than_one(self, plan):
        one = plan(definition(), observed=observed(minutes=6, runs=1))["estimate"]
        two = plan(definition(), observed=observed(minutes=6, runs=2))["estimate"]
        assert two["minutes"] < one["minutes"]

    def test_three_runs_is_no_longer_low_n(self, plan):
        answer = plan(definition(), observed=observed(minutes=6, runs=3))["estimate"]
        assert answer["minutes"] == 6.0
        assert "low_confidence" not in answer
        assert "tempered" not in answer

    def test_a_single_run_with_no_curated_figure_is_flagged_instead(self, plan):
        """No cost block to blend toward - the point figure is quoted as-is
        but flagged, rather than invented a range for."""
        spec = definition()
        del spec["cost"]
        answer = plan(spec, observed=observed(minutes=6, runs=1))["estimate"]
        assert answer["minutes"] == 6.0
        assert answer["low_confidence"] is True

    def test_a_thin_rolled_up_child_is_flagged(self, plan, tmp_path):
        """The #268 rollup has no parent cost block to blend toward by
        construction, so a thin roll-up gets the flag."""
        (tmp_path / "child.json").write_text(json.dumps({"id": "child", "steps": []}))
        parent = {
            "id": "parent",
            "steps": [
                {"name": "child", "workflow": {"path": "child.json", "arguments": {}}},
            ],
        }

        def observed_for_child(path, child_definition, arguments=None):
            return observed(minutes=6, runs=1)

        answer = plan(parent, observed_for_child=observed_for_child)["estimate"]
        assert answer["basis"] == "observed"
        assert answer["minutes"] == 6.0
        assert answer["low_confidence"] is True


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
        assert answer["unpriced"] == []

    def test_a_child_without_a_cost_makes_the_estimate_partial(self, plan, tmp_path):
        (tmp_path / "child.json").write_text(json.dumps({"id": "child", "steps": []}))
        answer = plan(composing("child.json"))["estimate"]
        assert (answer["minutes"], answer["partial"]) == (2.0, True)
        assert answer["unpriced"] == ["child.json"]

    def test_an_unreadable_child_makes_the_estimate_partial(self, plan):
        answer = plan(composing("missing.json"))["estimate"]
        assert (answer["minutes"], answer["partial"]) == (2.0, True)
        assert answer["unpriced"] == ["missing.json"]

    def test_a_builtin_adds_nothing_and_is_not_partial(self, plan):
        answer = plan(composing("builtin:text-to-image.json"))["estimate"]
        assert (answer["minutes"], answer["partial"]) == (2.0, False)
        assert answer["unpriced"] == []

    def test_an_unpriced_parent_with_a_priced_child_is_partial(self, plan, tmp_path):
        """A for_each step with no cost block contributes nothing to the
        total; a composed child's own figure should not be reported as
        though it were the whole run's (#242)."""
        child = {"id": "child", "cost": [cost("cuda", 5)], "steps": []}
        (tmp_path / "child.json").write_text(json.dumps(child))
        parent = {
            "id": "parent",
            "steps": [
                {"name": "own", "task": {"command": "x", "arguments": {}}},
                {
                    "name": "child",
                    "workflow": {"path": "child.json", "arguments": {}},
                },
            ],
        }
        answer = plan(parent)["estimate"]
        assert (answer["minutes"], answer["partial"]) == (5.0, True)
        assert answer["unpriced"] == ["parent"]

    def test_a_fully_unpriced_workflow_is_unknown_not_partial(self, plan):
        """No cost block anywhere - 'unknown' already says nobody has a
        number; 'partial' would wrongly imply some of it is known."""
        answer = plan({"id": "parent", "steps": []})["estimate"]
        assert (answer["minutes"], answer["partial"]) == (None, False)
        assert answer["unpriced"] == []

    def test_a_child_measured_on_another_device_is_still_added(self, plan, tmp_path):
        child = {"id": "child", "cost": [cost("mps", 5)], "steps": []}
        (tmp_path / "child.json").write_text(json.dumps(child))
        answer = plan(composing("child.json"))["estimate"]
        assert answer["minutes"] == 7.0
        # the parent's own basis is what is reported
        assert answer["basis"] == "catalog"

    def test_a_composed_childs_shifted_scalar_driver_is_unpriced(self, plan, tmp_path):
        """The composing step's own `arguments` are what the child actually
        runs with, not its declared defaults - a scalar cost_driver moved
        away from the value the child's catalog cost was measured against
        is the same #267 failure one level down (#341)."""
        child = {
            "id": "child",
            "cost": [cost("cuda", 5)],
            "cost_drivers": ["num_frames"],
            "variables": {"num_frames": 124},
            "steps": [],
        }
        (tmp_path / "child.json").write_text(json.dumps(child))
        parent = composing("child.json")
        parent["steps"][1]["workflow"]["arguments"] = {"num_frames": 345}
        answer = plan(parent)["estimate"]
        assert (answer["minutes"], answer["partial"]) == (2.0, True)
        assert answer["unpriced"] == ["child.json"]

    def test_a_composed_child_at_its_default_is_still_priced(self, plan, tmp_path):
        """The companion case: a composing step that passes the child's own
        default for a declared scalar driver is priced normally (#341)."""
        child = {
            "id": "child",
            "cost": [cost("cuda", 5)],
            "cost_drivers": ["num_frames"],
            "variables": {"num_frames": 124},
            "steps": [],
        }
        (tmp_path / "child.json").write_text(json.dumps(child))
        parent = composing("child.json")
        parent["steps"][1]["workflow"]["arguments"] = {"num_frames": 124}
        answer = plan(parent)["estimate"]
        assert (answer["minutes"], answer["partial"]) == (7.0, False)
        assert answer["unpriced"] == []

    def test_a_childs_observed_history_rolls_up_when_the_parent_has_none(
        self, plan, tmp_path
    ):
        """A parent with no cost block of its own, composing a child this
        box has actually run: the child's own observed minutes should be
        quoted as the parent's, with `basis: observed`, rather than falling
        back to unknown just because the parent itself carries no figure
        (#268)."""
        (tmp_path / "child.json").write_text(json.dumps({"id": "child", "steps": []}))
        parent = {
            "id": "parent",
            "steps": [
                {"name": "child", "workflow": {"path": "child.json", "arguments": {}}},
            ],
        }

        def observed_for_child(path, child_definition, arguments=None):
            return observed(minutes=6)

        answer = plan(parent, observed_for_child=observed_for_child)["estimate"]
        assert answer["minutes"] == 6.0
        assert answer["basis"] == "observed"
        assert answer["partial"] is False
        assert answer["unpriced"] == []

    def test_the_rolled_up_estimate_carries_the_childs_runs_and_measured_on(
        self, plan, tmp_path
    ):
        """The #268 rollup quotes the child's minutes under `basis:
        observed` - `runs`/`measured_on` have to come along with it, since
        an "observed" estimate with `runs: null` says it was measured but
        not how many times (#275)."""
        (tmp_path / "child.json").write_text(json.dumps({"id": "child", "steps": []}))
        parent = {
            "id": "parent",
            "steps": [
                {"name": "child", "workflow": {"path": "child.json", "arguments": {}}},
            ],
        }

        def observed_for_child(path, child_definition, arguments=None):
            return observed(minutes=6, runs=3, name="RTX 3090")

        answer = plan(parent, observed_for_child=observed_for_child)["estimate"]
        assert answer["basis"] == "observed"
        assert answer["runs"] == 3
        assert answer["measured_on"] == "RTX 3090"

    def test_the_rolled_up_runs_is_the_weakest_childs(self, plan, tmp_path):
        """Multiple observed children on the same device: `runs` is the min
        across them, the weakest history."""
        (tmp_path / "a.json").write_text(json.dumps({"id": "a", "steps": []}))
        (tmp_path / "b.json").write_text(json.dumps({"id": "b", "steps": []}))
        parent = {
            "id": "parent",
            "steps": [
                {"name": "a", "workflow": {"path": "a.json", "arguments": {}}},
                {"name": "b", "workflow": {"path": "b.json", "arguments": {}}},
            ],
        }

        def observed_for_child(path, child_definition, arguments=None):
            runs = 3 if path == "a.json" else 9
            return observed(minutes=6, runs=runs, name="RTX 3090")

        answer = plan(parent, observed_for_child=observed_for_child)["estimate"]
        assert answer["basis"] == "observed"
        assert answer["runs"] == 3
        assert answer["measured_on"] == "RTX 3090"

    def test_the_rolled_up_measured_on_is_null_when_children_disagree(
        self, plan, tmp_path
    ):
        """Children observed on different cards: nothing honest to name as
        `measured_on`, so it is withheld rather than picking one."""
        (tmp_path / "a.json").write_text(json.dumps({"id": "a", "steps": []}))
        (tmp_path / "b.json").write_text(json.dumps({"id": "b", "steps": []}))
        parent = {
            "id": "parent",
            "steps": [
                {"name": "a", "workflow": {"path": "a.json", "arguments": {}}},
                {"name": "b", "workflow": {"path": "b.json", "arguments": {}}},
            ],
        }

        def observed_for_child(path, child_definition, arguments=None):
            name = "RTX 3090" if path == "a.json" else "RTX 4090"
            return observed(minutes=6, runs=3, name=name)

        answer = plan(parent, observed_for_child=observed_for_child)["estimate"]
        assert answer["basis"] == "observed"
        assert answer["runs"] == 3
        assert answer["measured_on"] is None

    def test_an_unpriced_childs_history_leaves_the_estimate_partial(
        self, plan, tmp_path
    ):
        """`observed_for_child` answering nothing for one child, and that
        child having no catalog cost either, is still a gap - the rollup
        must not paper over it with `observed` just because a sibling did
        have history."""
        (tmp_path / "child.json").write_text(json.dumps({"id": "child", "steps": []}))
        parent = {
            "id": "parent",
            "steps": [
                {"name": "child", "workflow": {"path": "child.json", "arguments": {}}},
            ],
        }
        answer = plan(
            parent, observed_for_child=lambda path, defn, arguments=None: None
        )["estimate"]
        assert answer["minutes"] is None
        assert answer["basis"] == "unknown"
        assert answer["partial"] is False

    def test_a_composed_childs_shifted_scalar_driver_stays_unpriced_despite_default_bucket_history(
        self, plan, tmp_path
    ):
        """The observed-history rollup must not paper over a shifted scalar
        driver by quoting the *default* bucket's figure just because it
        exists - `observed_for_child` is called with the composing step's
        own arguments, and answering None for the shifted bucket falls
        through to the catalog path's `_scalar_driver_shifted` unpriced
        check rather than silently reusing the default bucket's history
        (#341)."""
        child = {
            "id": "child",
            "cost": [cost("cuda", 5)],
            "cost_drivers": ["num_frames"],
            "variables": {"num_frames": 124},
            "steps": [],
        }
        (tmp_path / "child.json").write_text(json.dumps(child))
        parent = {
            "id": "parent",
            "steps": [
                {
                    "name": "child",
                    "workflow": {
                        "path": "child.json",
                        "arguments": {"num_frames": 345},
                    },
                },
            ],
        }

        calls = []

        def observed_for_child(path, child_definition, arguments=None):
            calls.append(arguments)
            if (arguments or {}).get("num_frames") == 124:
                return observed(minutes=6, runs=3)
            return None

        answer = plan(parent, observed_for_child=observed_for_child)["estimate"]
        assert calls == [{"num_frames": 345}]
        assert answer["basis"] == "unknown"
        assert (answer["minutes"], answer["partial"]) == (None, False)

    def test_a_composed_childs_observed_history_is_bucketed_by_the_composing_steps_arguments(
        self, plan, tmp_path
    ):
        """The companion case: when the composing step's arguments do match
        a bucket this box has history for, that bucket's real figure is
        quoted rather than always the child's stored-default bucket
        (#341)."""
        (tmp_path / "child.json").write_text(json.dumps({"id": "child", "steps": []}))
        parent = {
            "id": "parent",
            "steps": [
                {
                    "name": "child",
                    "workflow": {
                        "path": "child.json",
                        "arguments": {"num_frames": 345},
                    },
                },
            ],
        }

        def observed_for_child(path, child_definition, arguments=None):
            if (arguments or {}).get("num_frames") == 345:
                return observed(minutes=9, runs=2)
            return observed(minutes=6, runs=5)

        answer = plan(parent, observed_for_child=observed_for_child)["estimate"]
        assert answer["basis"] == "observed"
        assert answer["minutes"] == 9.0


class TestDownloadsRequired:
    def test_a_repo_not_in_the_cache_is_required(self, plan):
        assert plan()["downloads_required"] == [
            {
                "repo": "org/still-model",
                "gb": None,
                "gated": None,
                "access_blocked": None,
            }
        ]

    def test_a_cached_repo_is_not(self, plan, monkeypatch):
        import dw.plan

        monkeypatch.setattr(
            dw.plan,
            "scan_models",
            lambda cache_dir=None: {"repos": [{"repo_id": "org/still-model"}]},
        )
        monkeypatch.setattr(
            dw.plan, "repo_download_incomplete", lambda *a, **k: False
        )
        assert plan()["downloads_required"] == []

    def test_a_present_but_incomplete_repo_is_still_required(self, plan, monkeypatch):
        """#382: a repo scan_models lists (an interrupted pull left the
        revision folder in place) but whose snapshot is missing files the
        load needs stays in downloads_required rather than reading as
        cached."""
        import dw.plan

        monkeypatch.setattr(
            dw.plan,
            "scan_models",
            lambda cache_dir=None: {"repos": [{"repo_id": "org/still-model"}]},
        )
        seen = []

        def fake_incomplete(name, cache_dir, variant=None):
            seen.append((name, variant))
            return True

        monkeypatch.setattr(dw.plan, "repo_download_incomplete", fake_incomplete)
        assert plan()["downloads_required"] == [
            {
                "repo": "org/still-model",
                "gb": None,
                "gated": None,
                "access_blocked": None,
            }
        ]
        assert seen == [("org/still-model", None)]

    def test_cache_dir_reaches_scan_models(self, plan, monkeypatch):
        import dw.plan

        seen = []

        def spy(cache_dir=None):
            seen.append(cache_dir)
            return {"repos": []}

        monkeypatch.setattr(dw.plan, "scan_models", spy)
        plan(cache_dir="/somewhere")
        assert seen == ["/somewhere"]

    def test_a_local_path_is_not_a_download_and_is_not_probed(
        self, plan, tmp_path, monkeypatch
    ):
        """A model_name that is not shaped like a hub id is a local checkout
        and never a download - decided by shape, never by touching the disk,
        since the free pre-flight takes the name from the request body and
        must not become a directory-existence oracle."""
        import os

        probed = []
        real_isdir = os.path.isdir
        monkeypatch.setattr(
            os.path, "isdir", lambda path: probed.append(path) or real_isdir(path)
        )
        names = (str(tmp_path / "weights"), "/Users/someone/.ssh", "./weights")
        for local in names:
            spec = definition()
            spec["steps"][0]["pipeline"]["from_pretrained_arguments"]["model_name"] = (
                local
            )
            assert plan(spec)["downloads_required"] == [], local
        assert not set(names) & set(probed)

    def test_a_single_file_url_is_listed_without_a_size(self, plan):
        spec = definition()
        spec["steps"][0]["pipeline"]["from_pretrained_arguments"] = {
            "from_single_file": "https://example.test/x.safetensors"
        }
        assert plan(spec)["downloads_required"] == [
            {
                "repo": None,
                "url": "https://example.test/x.safetensors",
                "gb": None,
                "gated": None,
                "access_blocked": None,
            }
        ]

    def test_a_single_file_local_path_is_not_listed(self, plan):
        spec = definition()
        spec["steps"][0]["pipeline"]["from_pretrained_arguments"] = {
            "from_single_file": "checkpoints/x.safetensors"
        }
        assert plan(spec)["downloads_required"] == []

    def test_components_and_children_are_scanned_and_deduplicated(self, plan, tmp_path):
        spec = definition()
        spec["steps"][0]["pipeline"]["components"] = {
            "vae": {"from_pretrained_arguments": {"model_name": "org/vae"}}
        }
        child = {
            "id": "child",
            "steps": [
                {
                    "name": "c",
                    "pipeline": {
                        "configuration": {"component_type": "{Fake}"},
                        "from_pretrained_arguments": {"model_name": "org/still-model"},
                        "arguments": {},
                    },
                }
            ],
        }
        (tmp_path / "child.json").write_text(json.dumps(child))
        spec["steps"].append(
            {"name": "sub", "workflow": {"path": "child.json", "arguments": {}}}
        )
        assert [d["repo"] for d in plan(spec)["downloads_required"]] == [
            "org/still-model",
            "org/vae",
        ]

    def test_sizes_come_from_the_hub_in_gib(self, plan, monkeypatch):
        import dw.plan

        class Sibling:
            def __init__(self, size):
                self.size = size

        class Info:
            siblings = [Sibling(2 * 1024**3), Sibling(None), Sibling(512 * 1024**2)]
            gated = False

        calls = []

        def fake_model_info(name, **kwargs):
            calls.append((name, kwargs))
            return Info()

        monkeypatch.setattr(dw.plan, "model_info", fake_model_info)
        answer = plan(lookup_sizes=True)["downloads_required"]
        assert answer == [
            {
                "repo": "org/still-model",
                "gb": 2.5,
                "gated": False,
                "access_blocked": False,
            }
        ]
        assert calls[0][0] == "org/still-model"
        assert calls[0][1]["files_metadata"] is True
        assert calls[0][1]["timeout"] == 5.0

    def test_a_gated_repo_this_token_may_access_is_not_blocked(self, plan, monkeypatch):
        import dw.plan

        class Sibling:
            rfilename = "config.json"

        class Info:
            siblings = [Sibling()]
            gated = "manual"

        monkeypatch.setattr(dw.plan, "model_info", lambda name, **k: Info())
        monkeypatch.setattr(dw.plan, "get_hf_file_metadata", lambda url, **k: object())
        assert plan(lookup_sizes=True)["downloads_required"] == [
            {
                "repo": "org/still-model",
                "gb": None,
                "gated": "manual",
                "access_blocked": False,
            }
        ]

    def test_a_gated_repo_this_token_lacks_access_to_is_blocked(
        self, plan, monkeypatch
    ):
        import httpx
        import dw.plan
        from huggingface_hub.utils import GatedRepoError

        response = httpx.Response(403, request=httpx.Request("GET", "https://hf.co/x"))

        class Sibling:
            rfilename = "config.json"

        class Info:
            siblings = [Sibling()]
            gated = "manual"

        def boom(*a, **k):
            raise GatedRepoError("no access", response=response)

        monkeypatch.setattr(dw.plan, "model_info", lambda name, **k: Info())
        monkeypatch.setattr(dw.plan, "get_hf_file_metadata", boom)
        assert plan(lookup_sizes=True)["downloads_required"] == [
            {
                "repo": "org/still-model",
                "gb": None,
                "gated": "manual",
                "access_blocked": True,
            }
        ]

    def test_a_gated_repo_model_info_itself_refuses_is_blocked(self, plan, monkeypatch):
        import httpx
        import dw.plan
        from huggingface_hub.utils import GatedRepoError

        response = httpx.Response(403, request=httpx.Request("GET", "https://hf.co/x"))

        def boom(*a, **k):
            raise GatedRepoError("no access", response=response)

        monkeypatch.setattr(dw.plan, "model_info", boom)
        assert plan(lookup_sizes=True)["downloads_required"] == [
            {
                "repo": "org/still-model",
                "gb": None,
                "gated": True,
                "access_blocked": True,
            }
        ]

    def test_a_gated_repo_with_no_probeable_file_is_unknown(self, plan, monkeypatch):
        import dw.plan

        class Info:
            siblings = []
            gated = "auto"

        monkeypatch.setattr(dw.plan, "model_info", lambda name, **k: Info())
        assert plan(lookup_sizes=True)["downloads_required"] == [
            {
                "repo": "org/still-model",
                "gb": None,
                "gated": "auto",
                "access_blocked": None,
            }
        ]

    def test_a_gate_probe_failing_for_an_unrelated_reason_is_unknown(
        self, plan, monkeypatch
    ):
        import dw.plan

        class Sibling:
            rfilename = "config.json"

        class Info:
            siblings = [Sibling()]
            gated = "auto"

        def boom(*a, **k):
            raise RuntimeError("offline")

        monkeypatch.setattr(dw.plan, "model_info", lambda name, **k: Info())
        monkeypatch.setattr(dw.plan, "get_hf_file_metadata", boom)
        assert plan(lookup_sizes=True)["downloads_required"] == [
            {
                "repo": "org/still-model",
                "gb": None,
                "gated": "auto",
                "access_blocked": None,
            }
        ]

    def test_a_hub_failure_is_a_null_size(self, plan, monkeypatch):
        import dw.plan

        def boom(*a, **k):
            raise RuntimeError("offline")

        monkeypatch.setattr(dw.plan, "model_info", boom)
        assert plan(lookup_sizes=True)["downloads_required"] == [
            {
                "repo": "org/still-model",
                "gb": None,
                "gated": None,
                "access_blocked": None,
            }
        ]

    def test_lookup_sizes_false_never_calls_the_hub(self, plan, monkeypatch):
        import dw.plan

        def boom(*a, **k):
            raise AssertionError("must not be called")

        monkeypatch.setattr(dw.plan, "model_info", boom)
        plan(lookup_sizes=False)


class TestGateWarnings:
    def test_a_blocked_repo_gets_a_warning(self):
        required = [
            {
                "repo": "org/free-model",
                "gb": 1.0,
                "gated": False,
                "access_blocked": False,
            },
            {
                "repo": "org/gated-model",
                "gb": None,
                "gated": True,
                "access_blocked": True,
            },
        ]
        warnings = gate_warnings(required)
        assert len(warnings) == 1
        assert "org/gated-model" in warnings[0]
        assert "huggingface.co/org/gated-model" in warnings[0]

    def test_an_accepted_gate_gets_no_warning(self):
        required = [
            {
                "repo": "org/manual-gate",
                "gb": 1.0,
                "gated": "manual",
                "access_blocked": False,
            }
        ]
        assert gate_warnings(required) == []

    def test_unknown_gate_status_gets_no_warning(self):
        required = [
            {"repo": "org/offline", "gb": None, "gated": None, "access_blocked": None}
        ]
        assert gate_warnings(required) == []


class TestCachedSteps:
    def test_no_probe_is_unknown(self, plan):
        assert plan()["cached_steps"] is None

    def test_the_probe_is_asked_with_the_arguments_and_counted(self, plan):
        seen = []

        def probe(arguments):
            seen.append(arguments)
            return ["still", "shot@a"]

        answer = plan(arguments={"frames": 9}, cache_probe=probe)
        assert answer["cached_steps"] == 2
        assert seen == [{"frames": 9}]

    def test_a_probe_that_cannot_answer_is_unknown(self, plan):
        assert plan(cache_probe=lambda arguments: None)["cached_steps"] is None

    def test_an_unseeded_workflow_is_zero_without_asking(self, plan):
        def probe(arguments):
            raise AssertionError("must not be asked")

        spec = definition()
        del spec["seed"]
        del spec["variables"]["seed"]
        assert plan(spec, cache_probe=probe)["cached_steps"] == 0

    def test_a_seed_variable_left_null_is_unseeded(self, plan):
        def probe(arguments):
            raise AssertionError("must not be asked")

        spec = definition()
        spec["variables"]["seed"] = None
        assert plan(spec, cache_probe=probe)["cached_steps"] == 0


class TestCachedMinutesEstimate:
    """#255: `plan.estimate.minutes` used to ignore `cached_steps` entirely,
    so a fully-cached rerun of a seeded workflow quoted the same minutes as
    a cold one. `cached_minutes` is the share of `minutes` a caller would
    actually wait for, once the steps the step cache would answer are
    subtracted - `minutes` itself is left alone."""

    def test_with_no_probe_cached_minutes_is_unknown(self, plan):
        answer = plan()["estimate"]
        assert answer["minutes"] == 10.0
        assert answer["cached_minutes"] is None

    def test_nothing_cached_leaves_cached_minutes_equal_to_minutes(self, plan):
        answer = plan(cache_probe=lambda arguments: [])["estimate"]
        assert answer["minutes"] == 10.0
        assert answer["cached_minutes"] == 10.0

    def test_a_partial_hit_reduces_cached_minutes(self, plan):
        # 3 expanded steps (still, shot@a, shot@b); 1 of them cached
        answer = plan(cache_probe=lambda arguments: ["still"])["estimate"]
        assert answer["minutes"] == 10.0
        assert answer["cached_minutes"] == 6.7

    def test_a_full_hit_is_zero(self, plan):
        answer = plan(cache_probe=lambda arguments: ["still", "shot@a", "shot@b"])[
            "estimate"
        ]
        assert answer["minutes"] == 10.0
        assert answer["cached_minutes"] == 0.0


class TestUnseededCacheWarning:
    """`cached_steps: 0` reads as 'probed, nothing hit' from outside; the
    warning is what says the cache is off entirely (#107)."""

    def test_a_seeded_workflow_says_nothing(self):
        assert unseeded_cache_warnings(definition()) == []

    def test_an_unseeded_workflow_explains_the_zero(self):
        spec = definition()
        del spec["seed"]
        del spec["variables"]["seed"]
        (warning,) = unseeded_cache_warnings(spec)
        assert "seed" in warning and "cached_steps" in warning

    def test_a_seed_passed_as_an_argument_counts(self):
        spec = definition()
        spec["variables"]["seed"] = None
        assert unseeded_cache_warnings(spec, {"seed": 7}) == []


class TestAnAdapterIsADownloadToo:
    """A `loras` entry carries its repo under `model_name` directly rather
    than inside a `from_pretrained_arguments` block, so the source walk
    missed it: a box with every base weight and not the IC-LoRA answered
    `downloads_required: []` and then pulled it mid-run (found verifying
    #151)."""

    def definition(self):
        return {
            "id": "adapted",
            "steps": [
                {
                    "name": "shot",
                    "pipeline": {
                        "configuration": {"component_type": "LTX2InContextPipeline"},
                        "from_pretrained_arguments": {
                            "model_name": "Lightricks/LTX-2.5-Diffusers"
                        },
                        "loras": [
                            {
                                "model_name": "Lightricks/LTX-2.5-22b-IC-LoRA-Ingredients",
                                "weight_name": "ic-lora.safetensors",
                            }
                        ],
                    },
                    "result": {"content_type": "video/mp4"},
                }
            ],
        }

    def repos(self, present, monkeypatch):
        import dw.plan

        monkeypatch.setattr(
            dw.plan,
            "scan_models",
            lambda cache_dir=None: {"repos": [{"repo_id": name} for name in present]},
        )
        monkeypatch.setattr(dw.plan, "repo_download_incomplete", lambda *a, **k: False)
        return [
            entry["repo"]
            for entry in dw.plan.downloads_required(
                self.definition(), None, None, None, False
            )
        ]

    def test_the_adapter_is_named_when_it_is_absent(self, monkeypatch):
        assert self.repos(["Lightricks/LTX-2.5-Diffusers"], monkeypatch) == [
            "Lightricks/LTX-2.5-22b-IC-LoRA-Ingredients"
        ]

    def test_nothing_is_named_when_both_are_present(self, monkeypatch):
        assert (
            self.repos(
                [
                    "Lightricks/LTX-2.5-Diffusers",
                    "Lightricks/LTX-2.5-22b-IC-LoRA-Ingredients",
                ],
                monkeypatch,
            )
            == []
        )

    def test_both_are_named_on_an_empty_cache(self, monkeypatch):
        assert self.repos([], monkeypatch) == [
            "Lightricks/LTX-2.5-22b-IC-LoRA-Ingredients",
            "Lightricks/LTX-2.5-Diffusers",
        ]
