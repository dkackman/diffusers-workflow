"""The plan a validate call answers with: what a run will execute for the
arguments given, fingerprinted so an acknowledgement can be bound to it."""

import copy
import json

import pytest

from dw.plan import (
    CONSTRAINT_DOCUMENTATION_KEYS,
    DOCUMENTATION_KEYS,
    PLACEMENT_RESULT_KEYS,
    build_plan,
    gate_warnings,
)
from dw.cost_scaling import scale_curated
from dw.validation import unseeded_cache_warnings
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


# Every top-level schema key and every step `result` key, sorted into what
# the fingerprint leaves out and what it keeps (#905). A key added to the
# schema fails TestFingerprintKeys until it is sorted here, so a metadata key
# can neither bust acknowledgements nor slip into the work unreviewed
KEPT_TOP_LEVEL_KEYS = {
    "id",  # names every file the run saves, and the step cache keys on it
    "variables",  # the folded values are the work
    "variable_constraints",  # a rule rounds or refuses a value
    "argument_template",  # read as arguments when not run as a child
    "seed",  # removed separately: a seed is not the work, but it is the output
    "steps",
}
KEPT_RESULT_KEYS = {
    "content_type",
    "save",
    "fps",
    "sample_rate",
    "audio_sample_rate",
    "subtype",
    "compression_level",
    "bitrate_mode",
    "embed_metadata",
    "format",
}


def schema():
    from importlib import resources

    return json.loads(
        resources.files("dw").joinpath("workflow_schema.json").read_text()
    )


class TestFingerprintKeys:
    def test_the_exclusion_lists_are_pinned(self):
        assert DOCUMENTATION_KEYS == (
            "configures",
            "cost",
            "cost_drivers",
            "description",
            "shape",
            "summary",
            "traits",
            "vram_estimate",
        )
        assert CONSTRAINT_DOCUMENTATION_KEYS == ("reason",)
        assert PLACEMENT_RESULT_KEYS == ("file_base_name", "role", "subfolder")

    def test_every_top_level_schema_key_is_sorted(self):
        keys = set(schema()["properties"])
        assert set(DOCUMENTATION_KEYS) | KEPT_TOP_LEVEL_KEYS == keys
        assert not set(DOCUMENTATION_KEYS) & KEPT_TOP_LEVEL_KEYS

    def test_every_result_schema_key_is_sorted(self):
        keys = set(schema()["$defs"]["result"]["properties"])
        assert set(PLACEMENT_RESULT_KEYS) | KEPT_RESULT_KEYS == keys
        assert not set(PLACEMENT_RESULT_KEYS) & KEPT_RESULT_KEYS

    def test_every_constraint_schema_key_but_the_reason_is_a_rule(self):
        keys = set(schema()["$defs"]["variable_constraint"]["properties"])
        assert set(CONSTRAINT_DOCUMENTATION_KEYS) <= keys


def with_pipeline_settings(spec):
    """The fixture's pipeline step with an attention backend, a cache
    config, a dtype and a LoRA set, so a test can move each one."""
    spec = copy.deepcopy(spec)
    pipeline = spec["steps"][0]["pipeline"]
    pipeline["configuration"]["attention_backend"] = "flash"
    pipeline["configuration"]["cache"] = {"type": "mag", "threshold": 0.1}
    pipeline["from_pretrained_arguments"]["torch_dtype"] = "torch.bfloat16"
    pipeline["loras"] = [{"model_name": "org/lora-a", "scale": 1.0}]
    return spec


class TestFingerprintIgnoresWhatCannotChangeTheOutput:
    """#905: the fingerprint is keyed only by what can change the output."""

    def test_cost_drivers(self, plan):
        spec = definition()
        spec["cost_drivers"] = ["frames"]
        other = copy.deepcopy(spec)
        other["cost_drivers"] = ["frames", "shots"]
        assert plan(spec)["fingerprint"] == plan(other)["fingerprint"]
        assert plan(spec)["fingerprint"] == plan()["fingerprint"]

    def test_vram_estimate(self, plan):
        spec = definition()
        spec["vram_estimate"] = {
            "base_gb": 4,
            "bytes_per_voxel": 1.0,
            "voxel_variables": ["frames"],
            "reason": "measured",
        }
        other = copy.deepcopy(spec)
        other["vram_estimate"]["base_gb"] = 9
        other["vram_estimate"]["reason"] = "re-measured"
        assert plan(spec)["fingerprint"] == plan(other)["fingerprint"]
        assert plan(spec)["fingerprint"] == plan()["fingerprint"]

    def test_an_integral_float_spelled_as_an_int(self, plan):
        # A client's JSON round trip can turn 12.0 into 12 (#905)
        spec = definition()
        spec["variables"]["frames"] = 25.0
        assert plan(spec)["fingerprint"] == plan()["fingerprint"]

    def test_shape_and_traits(self, plan):
        spec = definition()
        spec["shape"] = "shot"
        spec["traits"] = ["has-audio"]
        assert plan(spec)["fingerprint"] == plan()["fingerprint"]

    def test_a_constraints_reason(self, plan):
        spec = definition()
        spec["variable_constraints"] = {
            "frames": {"modulus": 8, "remainder": 1, "reason": "the VAE packs 8"}
        }
        other = copy.deepcopy(spec)
        other["variable_constraints"]["frames"]["reason"] = "reworded"
        assert plan(spec)["fingerprint"] == plan(other)["fingerprint"]

    def test_a_results_role_subfolder_and_file_name(self, plan):
        spec = definition()
        result = spec["steps"][0]["result"]
        result.update(role="final", subfolder="final", file_base_name="still")
        assert plan(spec)["fingerprint"] == plan()["fingerprint"]

    def test_a_composed_childs_cost_and_documentation(self, plan, tmp_path):
        step = {"name": "x", "task": {"command": "x", "arguments": {}}}
        child = {"id": "child", "variables": {"n": 1}, "steps": [step]}
        (tmp_path / "child.json").write_text(json.dumps(child))
        spec = composing("child.json")
        before = plan(spec)["fingerprint"]
        child.update(
            cost=[cost("cuda", 5)],
            cost_drivers=["n"],
            description="documented",
            vram_estimate={"base_gb": 1, "voxel_variables": ["n"]},
        )
        (tmp_path / "child.json").write_text(json.dumps(child, indent=2))
        assert plan(spec)["fingerprint"] == before


class TestFingerprintStillChangesWithWhatChangesTheOutput:
    @pytest.mark.parametrize(
        "move",
        [
            lambda p: p["configuration"].update(attention_backend="native"),
            lambda p: p["configuration"].update(cache={"type": "first_block"}),
            lambda p: p["configuration"]["cache"].update(threshold=0.2),
            lambda p: p["from_pretrained_arguments"].update(
                torch_dtype="torch.float16"
            ),
            lambda p: p["loras"][0].update(model_name="org/lora-b"),
            lambda p: p["loras"][0].update(scale=0.5),
            lambda p: p.pop("loras"),
        ],
        ids=[
            "attention_backend",
            "cache_type",
            "cache_threshold",
            "dtype",
            "lora",
            "lora_scale",
            "no_lora",
        ],
    )
    def test_a_pipeline_setting(self, plan, move):
        spec = with_pipeline_settings(definition())
        other = copy.deepcopy(spec)
        move(other["steps"][0]["pipeline"])
        assert plan(spec)["fingerprint"] != plan(other)["fingerprint"]

    def test_a_results_format(self, plan):
        spec = definition()
        spec["steps"][0]["result"]["format"] = "webp"
        assert plan(spec)["fingerprint"] != plan()["fingerprint"]


CATALOG_H3 = "workflows/templates/minimax/reference-to-video.json"
CATALOG_MUSIC = "workflows/templates/minimax/music.json"
# The knobs stage B of #866 made cost drivers of every H3 template
H3_UNSCALABLE = ("weights_dtype", "lora_model_name", "attention_backend", "cache")


def _repo_json(path):
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    return json.loads((root / path).read_text())


def _minimax_with_h3_drivers():
    import glob
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    found = []
    for path in sorted(glob.glob(str(root / "workflows/templates/minimax/*.json"))):
        drivers = json.loads(pathlib.Path(path).read_text()).get("cost_drivers") or []
        if "weights_dtype" in drivers:
            found.append(str(pathlib.Path(path).relative_to(root)))
    return found


@pytest.mark.parametrize("path", _minimax_with_h3_drivers())
def test_h3_unscalable_drivers_leave_every_fingerprint(plan, path):
    """Stage 0's pin, held for stage B (#866): declaring H3's unscalable
    knobs as cost drivers changes no template's plan fingerprint."""
    template = _repo_json(path)
    before = copy.deepcopy(template)
    before["cost_drivers"] = [
        name for name in template["cost_drivers"] if name not in H3_UNSCALABLE
    ]
    assert plan(template)["fingerprint"] == plan(before)["fingerprint"]


def test_every_h3_template_with_drivers_declares_its_unscalable_knobs():
    """#866 stage B: a minimax H3 template that buckets on its size also
    buckets on the knobs that move its wall clock with no rule to scale by."""
    import glob
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    missing = []
    for path in sorted(glob.glob(str(root / "workflows/templates/minimax/*.json"))):
        definition = json.loads(pathlib.Path(path).read_text())
        variables = definition.get("variables") or {}
        if not definition.get("cost_drivers") or "weights_dtype" not in variables:
            continue
        absent = [n for n in H3_UNSCALABLE if n not in definition["cost_drivers"]]
        if absent:
            missing.append((pathlib.Path(path).name, absent))
    assert _minimax_with_h3_drivers()
    assert missing == []


class TestTheCatalogH3Template:
    """The stage's repro on the real template: a copy differing only in
    cost_drivers plans the same work, one differing in attention backend
    or weights dtype does not."""

    @pytest.fixture
    def template(self):
        import pathlib

        root = pathlib.Path(__file__).resolve().parent.parent
        return json.loads((root / CATALOG_H3).read_text())

    def test_cost_drivers_leave_the_fingerprint(self, plan, template):
        other = copy.deepcopy(template)
        other["cost_drivers"] = [
            name for name in other["cost_drivers"] if name not in H3_UNSCALABLE
        ]
        del other["vram_estimate"]
        assert plan(template)["fingerprint"] == plan(other)["fingerprint"]

    def test_a_client_round_trip_keeps_it(self, plan, template):
        # The inline get_workflow JSON comes back with video_shift 12.0 as
        # 12 (and its kin); the by-name form keeps the float (#905)
        def respell(value):
            if isinstance(value, float) and value.is_integer():
                return int(value)
            if isinstance(value, dict):
                return {key: respell(item) for key, item in value.items()}
            if isinstance(value, list):
                return [respell(item) for item in value]
            return value

        inline = respell(template)
        # The template does hold one, or this checks nothing
        assert json.dumps(inline) != json.dumps(template)
        assert plan(template)["fingerprint"] == plan(inline)["fingerprint"]

    def test_the_seed_stays_out(self, plan, template):
        # The #85 contract: an acknowledgement buys the cost, not the seed
        assert (
            plan(template, arguments={"seed": 42})["fingerprint"]
            == plan(template, arguments={"seed": 43})["fingerprint"]
        )

    @pytest.mark.parametrize(
        "arguments",
        [{"attention_backend": "native"}, {"weights_dtype": "{int8}"}],
        ids=["attention_backend", "weights_dtype"],
    )
    def test_an_output_setting_moves_it(self, plan, template, arguments):
        assert (
            plan(template)["fingerprint"]
            != plan(template, arguments=arguments)["fingerprint"]
        )


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

    def test_another_devices_figure_is_re_priced_for_the_list(self, plan):
        """#589: a 2-shot figure from another backend quoted verbatim for a
        10-shot run was the 1/100 gate. It stays `other_device` - not a
        measurement here - but the entry count multiplies."""
        spec = definition()
        spec["cost"] = [cost("mps", 40, name="M2")]
        shots = [{"name": f"s{n}", "prompt": "x"} for n in range(10)]
        answer = plan(spec, arguments={"shots": shots})["estimate"]
        assert (answer["minutes"], answer["basis"]) == (200.0, "other_device")

    def test_another_devices_per_entry_rate_prices_the_list(self, plan):
        spec = definition()
        spec["cost"] = [
            cost("mps", 10, {"variable": "shots", "minutes": 3, "entries": 2})
        ]
        shots = [{"name": f"s{n}", "prompt": "x"} for n in range(5)]
        answer = plan(spec, arguments={"shots": shots})["estimate"]
        assert (answer["minutes"], answer["basis"]) == (19.0, "other_device")

    def test_another_devices_figure_with_two_changed_lists_is_unknown(self, plan):
        spec = definition()
        spec["cost"] = [cost("mps", 40, name="M2")]
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

    def test_minutes_is_rounded_to_one_decimal(self, plan):
        spec = definition()
        spec["cost"] = [cost("cuda", 10.04)]
        assert plan(spec)["estimate"]["minutes"] == 10.0

    def test_a_shifted_unscalable_cost_driver_is_flagged(self, plan):
        """The catalog figure was measured at frames=25; a caller who
        overrides a declared cost_driver away from that value is not
        describing the run the figure was measured for (#267). `frames` is
        not one of the drivers a figure scales along, so the figure is kept
        and flagged rather than withheld (#866, Q4)."""
        spec = definition()
        spec["cost_drivers"] = ["frames"]
        answer = plan(spec, arguments={"frames": 9})["estimate"]
        assert (answer["minutes"], answer["basis"]) == (10.0, "catalog")
        assert answer["low_confidence"] is True
        assert answer["differs"] == ["frames"]

    def test_a_shifted_unscalable_driver_is_flagged_on_another_device_too(self, plan):
        """#589: the other_device branch skipped the #267 check, so a 241-frame
        run was quoted the 121-frame clip's figure with nothing to say so."""
        spec = definition()
        spec["cost"] = [cost("mps", 40, name="M2")]
        spec["cost_drivers"] = ["frames"]
        answer = plan(spec, arguments={"frames": 9})["estimate"]
        assert (answer["minutes"], answer["basis"]) == (40.0, "other_device")
        assert answer["low_confidence"] is True
        assert answer["differs"] == ["frames"]

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

    def test_a_shifted_per_entry_field_in_a_list_driver_is_scaled(self, plan):
        """#593: a per-shot num_frames outside the values the default entries
        were measured with has no matching bucket, same as a scalar (#267) -
        so the figure is scaled by the frames summed over the list (#866)."""
        spec = definition()
        spec["cost"] = [cost("mps", 40, name="M2")]
        spec["cost_drivers"] = ["shots"]
        spec["variables"]["shots"] = [
            {"name": "a", "prompt": "p", "num_frames": 124},
            {"name": "b", "prompt": "p", "num_frames": 141},
        ]
        same = [
            {"name": "x", "prompt": "p", "num_frames": 124},
            {"name": "y", "prompt": "p", "num_frames": 124},
        ]
        long = [
            {"name": "x", "prompt": "p", "num_frames": 243},
            {"name": "y", "prompt": "p", "num_frames": 243},
        ]
        # Another device's figure stays other_device, scaled (#589)
        kept = plan(spec, arguments={"shots": same})["estimate"]
        assert (kept["minutes"], kept["basis"]) == (37.4, "other_device")
        assert kept["scaled"].endswith("frames 248 vs 265 over shots")
        moved = plan(spec, arguments={"shots": long})["estimate"]
        assert (moved["minutes"], moved["basis"]) == (73.4, "other_device")
        as_text = [dict(entry, num_frames=str(entry["num_frames"])) for entry in long]
        moved = plan(spec, arguments={"shots": as_text})["estimate"]
        assert (moved["minutes"], moved["basis"]) == (73.4, "other_device")
        same_text = [dict(entry, num_frames="124") for entry in same]
        kept = plan(spec, arguments={"shots": same_text})["estimate"]
        assert (kept["minutes"], kept["basis"]) == (37.4, "other_device")
        default = plan(spec)["estimate"]
        assert (default["minutes"], default["basis"]) == (40.0, "other_device")
        assert "scaled" not in default

    def test_a_lengthened_list_with_a_shifted_field_scales_both(self, plan):
        """#796: the derived basis scaled by entry count and skipped the #593
        field check, so 6 shots with one at 345 frames was quoted confidently.
        The count and the mean frames per entry now both scale (#866)."""
        spec = definition()
        spec["cost_drivers"] = ["shots"]
        spec["variables"]["shots"] = [
            {"name": f"s{n}", "prompt": "p", "num_frames": 124} for n in range(5)
        ]
        same = [{"name": f"s{n}", "prompt": "p", "num_frames": 124} for n in range(10)]
        kept = plan(spec, arguments={"shots": same})["estimate"]
        assert (kept["minutes"], kept["basis"]) == (20.0, "derived")
        assert "scaled" not in kept
        long = same[:9] + [{"name": "z", "prompt": "p", "num_frames": 345}]
        moved = plan(spec, arguments={"shots": long})["estimate"]
        # 4 min x 10/5 shots x (1461/10)/(620/5) frames per shot
        assert (moved["minutes"], moved["basis"]) == (23.6, "derived")
        assert moved["scaled"] == (
            "scaled from the curated 20 min: frames 1461 vs 620 over shots"
        )

    def test_position_fields_do_not_block_a_derived_figure(self, plan):
        """#796: start_frame/cut_frames grow with the count; only num_frames
        is a cost field, so an in-range lengthened list stays derived."""
        spec = definition()
        spec["cost_drivers"] = ["shots"]
        make = lambda n: [  # noqa: E731
            {"name": f"s{i}", "prompt": "p", "num_frames": 124, "start_frame": 124 * i}
            for i in range(n)
        ]
        spec["variables"]["shots"] = make(5)
        answer = plan(spec, arguments={"shots": make(10)})["estimate"]
        assert (answer["minutes"], answer["basis"]) == (20.0, "derived")

    def test_a_shifted_field_withholds_an_observed_figure(self, plan):
        """#796: observed history matched on list length must not price a
        list with a shot longer than any the default entries had."""
        spec = definition()
        spec["cost_drivers"] = ["shots"]
        spec["variables"]["shots"] = [
            {"name": f"s{n}", "prompt": "p", "num_frames": 124} for n in range(4)
        ]
        long = [dict(e) for e in spec["variables"]["shots"]]
        long[3]["num_frames"] = 345
        answer = plan(spec, arguments={"shots": long}, observed=observed())["estimate"]
        # The observed 8.04 is withheld; the curated 10 is scaled instead
        assert (answer["minutes"], answer["basis"]) == (14.5, "derived")

    def test_a_per_entry_list_driver_prices_past_its_measured_count(self, plan):
        """#772: a windows list whose entries carry an `index` label priced
        3 windows but went unknown at 4, because the new index looked like a
        shifted cost field. A list the cost prices per entry extrapolates
        both ways."""
        spec = definition()
        spec["cost"] = [
            cost("mps", 12, {"variable": "shots", "minutes": 2, "entries": 3})
        ]
        spec["cost_drivers"] = ["shots"]
        spec["variables"]["shots"] = [
            {"name": f"w{n}", "index": n, "prompt": "p"} for n in range(3)
        ]
        for count, minutes in ((2, 10.0), (4, 14.0), (5, 16.0)):
            shots = [{"name": f"w{n}", "index": n, "prompt": "p"} for n in range(count)]
            answer = plan(spec, arguments={"shots": shots})["estimate"]
            assert (answer["minutes"], answer["basis"]) == (minutes, "other_device")

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


def scalable(minutes=10):
    """The fixture with steps and a canvas as declared cost drivers, its
    curated figure measured at their defaults (#866)."""
    spec = definition()
    spec["cost"] = [cost("cuda", minutes)]
    spec["variables"].update({"num_inference_steps": 9, "width": 960, "height": 544})
    spec["cost_drivers"] = ["num_inference_steps", "width", "height", "shots"]
    return spec


class TestDriverScaling:
    """#866: a curated figure is measured at the workflow's variable
    defaults; a run that moves steps, the canvas or the frames is quoted
    scaled along them rather than as no figure at all."""

    def test_the_defaults_quote_the_curated_figure(self, plan):
        answer = plan(scalable())["estimate"]
        assert (answer["minutes"], answer["basis"]) == (10.0, "catalog")
        assert "scaled" not in answer

    def test_steps_scale_in_proportion(self, plan):
        answer = plan(scalable(), arguments={"num_inference_steps": 30})["estimate"]
        assert (answer["minutes"], answer["basis"]) == (33.3, "derived")
        assert answer["scaled"] == "scaled from the curated 10 min: steps 30 vs 9"

    def test_the_canvas_scales_by_its_area(self, plan):
        arguments = {"width": 1344, "height": 768}
        answer = plan(scalable(), arguments=arguments)["estimate"]
        # 10 x 1344.768 / 960.544
        assert (answer["minutes"], answer["basis"]) == (19.8, "derived")
        assert answer["scaled"] == (
            "scaled from the curated 10 min: canvas 1344x768 vs 960x544"
        )

    def test_one_side_of_the_canvas_scales_too(self, plan):
        answer = plan(scalable(), arguments={"width": 1920})["estimate"]
        assert (answer["minutes"], answer["basis"]) == (20.0, "derived")
        assert "canvas 1920x544 vs 960x544" in answer["scaled"]

    def test_per_entry_frames_scale_by_their_sum(self, plan):
        spec = scalable()
        spec["variables"]["shots"] = [
            {"name": "a", "prompt": "one", "num_frames": 124},
            {"name": "b", "prompt": "two", "num_frames": 141},
        ]
        shots = [
            {"name": "a", "prompt": "one", "num_frames": 124},
            {"name": "b", "prompt": "two", "num_frames": 345},
        ]
        answer = plan(spec, arguments={"shots": shots})["estimate"]
        # 10 x 469/265: the same two shots, one of them longer
        assert (answer["minutes"], answer["basis"]) == (17.7, "derived")
        assert answer["scaled"].endswith("frames 469 vs 265 over shots")

    def test_a_list_not_priced_by_length_scales_by_the_sum_alone(self, plan):
        """No for_each runs over `takes`, so its length prices nothing - the
        summed frames carry the whole ratio, length and all."""
        spec = scalable()
        spec["variables"]["takes"] = [{"num_frames": 100}]
        spec["cost_drivers"] = spec["cost_drivers"] + ["takes"]
        takes = [{"num_frames": 100}, {"num_frames": 150}]
        answer = plan(spec, arguments={"takes": takes})["estimate"]
        assert (answer["minutes"], answer["basis"]) == (25.0, "derived")
        assert answer["scaled"].endswith("frames 250 vs 100 over takes")

    def test_steps_canvas_and_frames_multiply(self, plan):
        spec = scalable()
        spec["variables"]["frames"] = 124
        spec["variables"]["num_frames"] = 124
        spec["steps"][0]["pipeline"]["arguments"]["num_frames"] = "variable:num_frames"
        spec["cost_drivers"] = spec["cost_drivers"] + ["num_frames"]
        arguments = {
            "num_inference_steps": 30,
            "width": 1344,
            "height": 768,
            "num_frames": 248,
        }
        answer = plan(spec, arguments=arguments)["estimate"]
        # 10 x 30/9 x 1344.768/960.544 x 248/124
        assert (answer["minutes"], answer["basis"]) == (131.8, "derived")
        assert answer["scaled"] == (
            "scaled from the curated 10 min: steps 30 vs 9, "
            "canvas 1344x768 vs 960x544, frames 248 vs 124"
        )

    def test_it_multiplies_with_the_list_length(self, plan):
        shots = [{"name": f"s{n}", "prompt": "x"} for n in range(4)]
        arguments = {"shots": shots, "num_inference_steps": 18}
        answer = plan(scalable(), arguments=arguments)["estimate"]
        # 10 x 4/2 shots x 18/9 steps
        assert (answer["minutes"], answer["basis"]) == (40.0, "derived")
        assert answer["scaled"] == "scaled from the curated 20 min: steps 18 vs 9"

    def test_a_non_numeric_value_stays_unknown(self, plan):
        # A text default keeps a text argument as text; an int one would
        # have cast "wide" away to the default before the plan saw it
        spec = scalable()
        spec["variables"]["width"] = "960"
        answer = plan(spec, arguments={"width": "wide"})["estimate"]
        assert (answer["minutes"], answer["basis"]) == (None, "unknown")
        assert "scaled" not in answer

    def test_a_number_spelled_as_text_scales(self, plan):
        answer = plan(scalable(), arguments={"num_inference_steps": "18"})["estimate"]
        assert (answer["minutes"], answer["basis"]) == (20.0, "derived")

    def test_another_devices_figure_stays_other_device(self, plan):
        spec = scalable()
        spec["cost"] = [cost("mps", 40)]
        answer = plan(spec, arguments={"num_inference_steps": 18})["estimate"]
        assert (answer["minutes"], answer["basis"]) == (80.0, "other_device")
        assert answer["scaled"] == "scaled from the curated 40 min: steps 18 vs 9"

    def test_a_thin_observed_figure_tempers_toward_the_scaled_one(self, plan):
        arguments = {"num_inference_steps": 18}
        thin = observed(minutes=24, runs=1)
        answer = plan(scalable(), arguments=arguments, observed=thin)["estimate"]
        assert answer["basis"] == "observed"
        assert answer["tempered"] is True
        assert answer["curated_minutes"] == 20.0
        assert 20.0 < answer["minutes"] < 24.0
        assert answer["scaled"] == "scaled from the curated 10 min: steps 18 vs 9"

    def test_a_settled_observed_figure_is_not_scaled(self, plan):
        arguments = {"num_inference_steps": 18}
        answer = plan(scalable(), arguments=arguments, observed=observed(minutes=24))[
            "estimate"
        ]
        assert (answer["minutes"], answer["basis"]) == (24.0, "observed")
        assert "scaled" not in answer


class TestScalingAccuracy:
    """#866's bar: each measured pair on the RTX 3090 (#851's B and L runs)
    is predicted within 30% by scaling the other."""

    @pytest.mark.parametrize(
        "measured, defaults, effective, actual",
        [
            # B4 -> B1: steps 12 -> 30 at 960x544
            (
                6.6,
                {"num_inference_steps": 12, "width": 960, "height": 544},
                {"num_inference_steps": 30, "width": 960, "height": 544},
                15.1,
            ),
            # B4 -> B5: canvas 960x544 -> 1344x768 at 12 steps
            (
                6.6,
                {"num_inference_steps": 12, "width": 960, "height": 544},
                {"num_inference_steps": 12, "width": 1344, "height": 768},
                11.8,
            ),
            # B3 -> L5: steps 12 -> 30 at 1344x768
            (
                12.9,
                {"num_inference_steps": 12, "width": 1344, "height": 768},
                {"num_inference_steps": 30, "width": 1344, "height": 768},
                29.5,
            ),
        ],
    )
    def test_within_thirty_percent(self, measured, defaults, effective, actual):
        minutes, axes = scale_curated(measured, defaults, effective)
        assert axes
        assert abs(minutes - actual) / actual < 0.30

    def test_nothing_moved_is_the_figure_itself(self):
        defaults = {"num_inference_steps": 9, "width": 960, "height": 544}
        assert scale_curated(7.88, defaults, dict(defaults)) == (7.88, [])

    def test_a_moved_non_number_is_none(self):
        defaults = {"num_inference_steps": 9}
        assert scale_curated(7.88, defaults, {"num_inference_steps": None}) is None


CATALOG_DIALOGUE = "workflows/templates/minimax/dialogue-short.json"


class TestTheCatalogTemplatesScale:
    """#866 on the real templates: H3's reference-to-video and the
    dialogue-short sequence, measured at their defaults."""

    @staticmethod
    def load(path):
        import pathlib

        root = pathlib.Path(__file__).resolve().parent.parent
        return json.loads((root / path).read_text())

    def test_reference_to_video_at_its_defaults(self, plan):
        answer = plan(self.load(CATALOG_H3))["estimate"]
        assert (answer["minutes"], answer["basis"]) == (7.9, "catalog")
        assert "scaled" not in answer

    def test_reference_to_video_at_thirty_steps(self, plan):
        template = self.load(CATALOG_H3)
        answer = plan(template, arguments={"num_inference_steps": 30})["estimate"]
        # 7.88 x 30/9
        assert (answer["minutes"], answer["basis"]) == (26.3, "derived")
        assert "steps 30 vs 9" in answer["scaled"]

    def test_reference_to_video_at_the_larger_canvas(self, plan):
        template = self.load(CATALOG_H3)
        arguments = {"width": 1344, "height": 768}
        answer = plan(template, arguments=arguments)["estimate"]
        assert (answer["minutes"], answer["basis"]) == (15.6, "derived")
        assert "canvas 1344x768 vs 960x544" in answer["scaled"]

    def test_dialogue_short_at_its_defaults(self, plan):
        answer = plan(self.load(CATALOG_DIALOGUE))["estimate"]
        assert answer["basis"] == "catalog"
        assert answer["minutes"] == pytest.approx(21, abs=0.5)

    def test_dialogue_short_with_one_shot_longer(self, plan):
        template = self.load(CATALOG_DIALOGUE)
        shots = copy.deepcopy(template["variables"]["shots"])
        assert [shot["num_frames"] for shot in shots] == [124, 124, 124, 124, 141]
        shots[0]["num_frames"] = 192
        answer = plan(template, arguments={"shots": shots})["estimate"]
        # 21.2 x 705/637 frames
        assert (answer["minutes"], answer["basis"]) == (23.5, "derived")
        assert answer["scaled"] == (
            "scaled from the curated 21.2 min: frames 705 vs 637 over shots"
        )


class TestUnscalableDrivers:
    """#866 stage B: a declared scalar driver with no rule to scale by keeps
    the (scaled) figure, sets `low_confidence` and names itself in
    `differs`, for every template (Q4)."""

    def test_reference_to_video_in_int8(self, plan):
        template = _repo_json(CATALOG_H3)
        answer = plan(template, arguments={"weights_dtype": "{int8}"})["estimate"]
        assert (answer["minutes"], answer["basis"]) == (7.9, "catalog")
        assert answer["low_confidence"] is True
        assert answer["differs"] == ["weights_dtype"]

    def test_reference_to_video_without_its_lora_at_thirty_steps(self, plan):
        template = _repo_json(CATALOG_H3)
        arguments = {"lora_model_name": None, "num_inference_steps": 30}
        answer = plan(template, arguments=arguments)["estimate"]
        # 7.88 x 30/9, scaled and flagged
        assert (answer["minutes"], answer["basis"]) == (26.3, "derived")
        assert "steps 30 vs 9" in answer["scaled"]
        assert answer["low_confidence"] is True
        assert answer["differs"] == ["lora_model_name"]

    def test_every_moved_knob_is_named(self, plan):
        template = _repo_json(CATALOG_H3)
        arguments = {
            "weights_dtype": "{int8}",
            "attention_backend": "native",
            "cache": {"type": "first_block"},
        }
        answer = plan(template, arguments=arguments)["estimate"]
        assert answer["minutes"] == 7.9
        assert sorted(answer["differs"]) == [
            "attention_backend",
            "cache",
            "weights_dtype",
        ]

    def test_reference_to_video_at_its_defaults_is_not_flagged(self, plan):
        template = _repo_json(CATALOG_H3)
        answer = plan(template)["estimate"]
        assert (answer["minutes"], answer["basis"]) == (7.9, "catalog")
        assert "low_confidence" not in answer
        assert "differs" not in answer

    def test_the_defaults_passed_explicitly_are_not_flagged(self, plan):
        template = _repo_json(CATALOG_H3)
        variables = template["variables"]
        arguments = {name: variables[name] for name in H3_UNSCALABLE}
        answer = plan(template, arguments=arguments)["estimate"]
        assert "low_confidence" not in answer

    def test_music_at_another_duration(self, plan):
        """#440's case, now a flagged figure rather than `unknown`."""
        template = _repo_json(CATALOG_MUSIC)
        answer = plan(template, arguments={"audio_duration": 240})["estimate"]
        assert (answer["minutes"], answer["basis"]) == (3.6, "catalog")
        assert answer["low_confidence"] is True
        assert answer["differs"] == ["audio_duration"]

    def test_a_thin_bucket_blends_toward_a_flagged_figure(self, plan):
        """One int8 run, blended toward the int4 catalog figure: the blend
        leans on a figure measured at another setting, so it says so."""
        spec = definition()
        spec["cost_drivers"] = ["frames"]
        answer = plan(
            spec, arguments={"frames": 9}, observed=observed(minutes=6, runs=1)
        )["estimate"]
        assert answer["basis"] == "observed"
        assert answer["tempered"] is True
        assert answer["low_confidence"] is True
        assert answer["differs"] == ["frames"]

    def test_a_full_bucket_is_not_flagged(self, plan):
        """Three runs at the moved setting are a measurement of it."""
        spec = definition()
        spec["cost_drivers"] = ["frames"]
        answer = plan(
            spec, arguments={"frames": 9}, observed=observed(minutes=6, runs=3)
        )["estimate"]
        assert answer["minutes"] == 6.0
        assert "low_confidence" not in answer
        assert "differs" not in answer

    def test_a_composed_child_at_another_setting_flags_the_total(self, plan, tmp_path):
        child = {
            "id": "child",
            "cost": [cost("cuda", 5)],
            "variables": {"weights_dtype": "{int4}"},
            "cost_drivers": ["weights_dtype"],
            "steps": [],
        }
        (tmp_path / "child.json").write_text(json.dumps(child))
        parent = {
            "id": "parent",
            "cost": [cost("cuda", 2)],
            "steps": [
                {"name": "own", "task": {"command": "x", "arguments": {}}},
                {
                    "name": "child",
                    "workflow": {
                        "path": "child.json",
                        "arguments": {"weights_dtype": "{int8}"},
                    },
                },
            ],
        }
        answer = plan(parent)["estimate"]
        assert answer["minutes"] == 7.0
        assert answer["low_confidence"] is True
        assert answer["differs"] == ["weights_dtype"]


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

    def test_a_number_never_carries_the_unknown_basis(self, plan, tmp_path):
        """#593: an uncosted parent that only composes priced children sums
        to a figure, which takes the children's basis rather than `unknown`."""
        child = {"id": "child", "cost": [cost("cuda", 5)], "steps": []}
        (tmp_path / "child.json").write_text(json.dumps(child))
        parent = composing("child.json")
        del parent["cost"]
        parent["steps"] = parent["steps"][1:]
        answer = plan(parent)["estimate"]
        assert (answer["minutes"], answer["basis"]) == (5.0, "catalog")

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

    def test_a_composed_childs_shifted_scalar_driver_is_scaled(self, plan, tmp_path):
        """The composing step's own `arguments` are what the child actually
        runs with, not its declared defaults - a scalar cost_driver moved
        away from the value the child's catalog cost was measured against
        is the same #267 case one level down (#341), scaled (#866)."""
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
        # 2 of its own + 5 x 345/124
        assert (answer["minutes"], answer["partial"]) == (15.9, False)
        assert (answer["basis"], answer["unpriced"]) == ("derived", [])
        assert answer["scaled"] == (
            "child.json: scaled from the curated 5 min: frames 345 vs 124"
        )

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

    def test_a_composed_childs_shifted_scalar_driver_is_scaled_despite_default_bucket_history(
        self, plan, tmp_path
    ):
        """The observed-history rollup must not paper over a shifted scalar
        driver by quoting the *default* bucket's figure just because it
        exists - `observed_for_child` is called with the composing step's
        own arguments, and answering None for the shifted bucket falls
        through to the catalog path's driver check rather than silently
        reusing the default bucket's history (#341), which now scales the
        child's figure (#866)."""
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
        assert answer["basis"] == "derived"
        assert (answer["minutes"], answer["partial"]) == (13.9, False)

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
        monkeypatch.setattr(dw.plan, "repo_download_incomplete", lambda *a, **k: False)
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

    def test_task_steps_list_the_repos_their_commands_declare(self, plan):
        """#975: an image processor (remove_background) and a registered
        command (segment) both name their weights, after the pipeline's."""
        spec = definition()
        spec["steps"] += [
            {
                "name": "cutout",
                "task": {"command": "remove_background", "arguments": {}},
                "result": {"content_type": "image/png"},
            },
            {
                "name": "mask",
                "task": {"command": "segment", "arguments": {"prompt": "a cup"}},
                "result": {"content_type": "image/png"},
            },
        ]
        assert [d["repo"] for d in plan(spec)["downloads_required"]] == [
            "org/still-model",
            "briaai/RMBG-2.0",
            "IDEA-Research/grounding-dino-base",
            "facebook/sam2-hiera-large",
        ]

    def test_a_task_override_replaces_the_default_repo(self, plan):
        spec = definition()
        spec["steps"].append(
            {
                "name": "mask",
                "task": {
                    "command": "segment",
                    "arguments": {"model_name": "org/dino", "sam_model_name": None},
                },
            }
        )
        assert [d["repo"] for d in plan(spec)["downloads_required"]] == [
            "org/still-model",
            "org/dino",
            "facebook/sam2-hiera-large",
        ]

    def test_a_repo_a_pipeline_and_a_task_both_load_is_listed_once(self, plan):
        spec = definition()
        spec["steps"][0]["pipeline"]["from_pretrained_arguments"]["model_name"] = (
            "openai/whisper-base"
        )
        spec["steps"].append(
            {
                "name": "words",
                "task": {"command": "transcribe_audio", "arguments": {}},
            }
        )
        assert [d["repo"] for d in plan(spec)["downloads_required"]] == [
            "openai/whisper-base"
        ]

    def test_two_task_steps_naming_one_repo_list_it_once(self, plan):
        spec = definition()
        spec["steps"] += [
            {"name": "a", "task": {"command": "transcribe_audio", "arguments": {}}},
            {"name": "b", "task": {"command": "check_script", "arguments": {}}},
        ]
        repos = [d["repo"] for d in plan(spec)["downloads_required"]]
        assert repos == ["org/still-model", "openai/whisper-base"]

    def test_a_task_argument_that_is_a_reference_or_a_path_is_not_listed(self, plan):
        for local in ("previous_result:detector", "/models/dino", "./models/dino"):
            spec = definition()
            spec["steps"].append(
                {
                    "name": "mask",
                    "task": {
                        "command": "segment",
                        "arguments": {"model_name": local, "sam_model_name": local},
                    },
                }
            )
            assert [d["repo"] for d in plan(spec)["downloads_required"]] == [
                "org/still-model"
            ], local

    def test_a_cached_task_repo_is_not_required(self, plan, monkeypatch):
        import dw.plan

        monkeypatch.setattr(
            dw.plan,
            "scan_models",
            lambda cache_dir=None: {
                "repos": [
                    {"repo_id": "org/still-model"},
                    {"repo_id": "briaai/RMBG-2.0"},
                ]
            },
        )
        monkeypatch.setattr(dw.plan, "repo_download_incomplete", lambda *a, **k: False)
        spec = definition()
        spec["steps"].append(
            {"name": "cutout", "task": {"command": "remove_background"}}
        )
        assert plan(spec)["downloads_required"] == []

    def test_a_hook_that_raises_costs_its_entries_and_not_the_plan(
        self, plan, monkeypatch
    ):
        import dw.tasks.registry

        def exploding(command):
            def hook(arguments):
                raise RuntimeError("hook broke")

            return hook

        monkeypatch.setattr(dw.tasks.registry, "required_weights_hook", exploding)
        spec = definition()
        spec["steps"].append(
            {"name": "mask", "task": {"command": "segment", "arguments": {}}}
        )
        assert [d["repo"] for d in plan(spec)["downloads_required"]] == [
            "org/still-model"
        ]

    def test_a_composed_childs_task_step_contributes_its_repo(self, plan, tmp_path):
        spec = definition()
        child = {
            "id": "child",
            "steps": [
                {
                    "name": "c",
                    "task": {
                        "command": "segment",
                        "arguments": {"model_name": "org/child-dino"},
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
            "org/child-dino",
            "facebook/sam2-hiera-large",
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


class TestUnpricedNamesTheChildren:
    def test_an_unpriced_pure_composition_names_its_child(self, plan, tmp_path):
        """#655: no number and no reason gave a caller nothing to act on."""
        (tmp_path / "child.json").write_text(json.dumps({"id": "child", "steps": []}))
        definition = {
            "id": "parent",
            "steps": [
                {"name": "shot", "workflow": {"path": "child.json", "arguments": {}}}
            ],
        }
        answer = plan(definition)["estimate"]
        assert (answer["minutes"], answer["basis"]) == (None, "unknown")
        assert answer["partial"] is False
        assert [name for name in answer["unpriced"]] == ["child.json"]
