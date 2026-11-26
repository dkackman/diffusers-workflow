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
    missing_repo_errors,
)
from dw.cost_scaling import scale_curated
from dw.step_cache import unseeded_cache_warnings
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
    import dw.plan_downloads

    def offline(*a, **k):
        raise RuntimeError("offline")

    monkeypatch.setattr(
        dw.plan_downloads, "scan_models", lambda cache_dir=None: {"repos": []}
    )
    monkeypatch.setattr(dw.plan_downloads, "model_info", offline)

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
    "presets",  # a bundle sets the variables the caller did not pass
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
            "host_ram_gb",
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
        list with a shot longer than any the default entries had. A block
        that does not say what frames its runs rendered keeps the discard
        (#1696)."""
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

    def _shifted(self, fixed=None):
        spec = definition()
        spec["cost_drivers"] = ["shots"]
        spec["variables"]["shots"] = [
            {"name": f"s{n}", "prompt": "p", "num_frames": 124} for n in range(4)
        ]
        if fixed is not None:
            spec["cost"][0]["fixed_minutes"] = fixed
        long = [dict(e) for e in spec["variables"]["shots"]]
        long[3]["num_frames"] = 345
        return spec, long

    def test_a_shifted_field_rescales_history_that_names_its_frames(self, plan):
        """#1696: a bucket that says what frames its runs rendered is
        rescaled to this run's frames instead of discarded (#796)."""
        spec, long = self._shifted()
        history = dict(observed(minutes=8.0, runs=11), entry_frames=496)
        answer = plan(spec, arguments={"shots": long}, observed=history)["estimate"]
        # 8 min x 717/496 frames
        assert (answer["minutes"], answer["basis"]) == (11.6, "derived")
        assert answer["scaled"] == (
            "scaled from the observed 8 min at 496 frames: num_frames 717 vs 496"
        )
        assert not answer.get("low_confidence")

    def test_a_frames_rescale_holds_the_fixed_share(self, plan):
        """#1696: the cost row's fixed_minutes share of the observed figure
        does not scale with the frames."""
        spec, long = self._shifted(fixed=4)
        history = dict(observed(minutes=8.0, runs=11), entry_frames=496)
        answer = plan(spec, arguments={"shots": long}, observed=history)["estimate"]
        # 8 x 0.4 = 3.2 fixed, + 4.8 x 717/496
        assert (answer["minutes"], answer["basis"]) == (10.1, "derived")
        assert answer["scaled"] == (
            "scaled from the observed 8 min at 496 frames (3.2 min of it fixed): "
            "num_frames 717 vs 496"
        )

    def test_a_frames_rescale_scales_the_warm_figure_too(self):
        """The block's warm figure scales as its cold one does, fixed share
        held; the estimate quotes cold, but the block stays consistent."""
        from dw.plan import _frames_rescaled

        spec, long = self._shifted(fixed=4)
        history = dict(observed(minutes=8.0, runs=11), entry_frames=496)
        history.update(warm_minutes=4.0, warm_runs=11)
        block = _frames_rescaled(history, spec, {"variables": {"shots": long}}, "cuda")
        assert block["warm_minutes"] == pytest.approx(1.6 + 2.4 * 717 / 496)
        assert block["cold_minutes"] == pytest.approx(3.2 + 4.8 * 717 / 496)

    @pytest.mark.parametrize("was", [300, 1500])
    def test_a_frames_ratio_outside_half_to_double_is_low_confidence(self, plan, was):
        spec, long = self._shifted()
        history = dict(observed(minutes=8.0, runs=11), entry_frames=was)
        answer = plan(spec, arguments={"shots": long}, observed=history)["estimate"]
        assert answer["basis"] == "derived"
        assert answer["low_confidence"] is True

    def test_a_thin_bucket_tempers_toward_the_curated_figure_at_the_same_frames(
        self, plan
    ):
        """#1696: under SMALL_N_THRESHOLD runs the rescaled figure blends
        toward the curated one scaled to the same frames (14.5 min), not
        toward the unscaled 10."""
        spec, long = self._shifted()
        thick = dict(observed(minutes=8.0, runs=11), entry_frames=496)
        thin = dict(observed(minutes=8.0, runs=1), entry_frames=496)
        rescaled = plan(spec, arguments={"shots": long}, observed=thick)["estimate"]
        tempered = plan(spec, arguments={"shots": long}, observed=thin)["estimate"]
        assert tempered["basis"] == "derived"
        assert rescaled["minutes"] < tempered["minutes"] < 14.5

    def test_history_at_this_run_s_frames_stays_observed(self, plan):
        """#1696: a bucket whose runs rendered the frames this run asks for
        needs no rescale, so its measurement stands as observed."""
        spec, long = self._shifted()
        history = dict(observed(minutes=8.0, runs=11), entry_frames=717)
        answer = plan(spec, arguments={"shots": long}, observed=history)["estimate"]
        assert (answer["minutes"], answer["basis"]) == (8.0, "observed")
        assert "scaled" not in answer

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

    def test_an_unrun_tier_scales_the_default_tiers_history(self, plan):
        """#1236: history exists for the default tier only. A draft tier is
        that figure scaled by the steps ratio, not the curated row (which
        would quote 10 x 4/8 = 5 min)."""
        spec = definition()
        spec["variables"]["tier"] = "standard"
        spec["variables"]["num_inference_steps"] = 8
        spec["cost_drivers"] = ["tier", "num_inference_steps"]
        spec["presets"] = {
            "tier": {
                "standard": {"num_inference_steps": 8},
                "draft": {"num_inference_steps": 4},
            }
        }
        spec["steps"][0]["pipeline"]["arguments"]["num_inference_steps"] = (
            "variable:num_inference_steps"
        )

        def history(arguments):
            return observed(minutes=3.0) if arguments.get("tier") is None else None

        answer = plan(spec, arguments={"tier": "draft"}, observed=history)["estimate"]
        assert answer["basis"] == "derived"
        assert answer["minutes"] == 1.5
        assert "observed 3.0 min" in answer["scaled"]
        assert answer["low_confidence"] is True

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

    def test_an_undeclared_scalar_driver_is_flagged_not_scaled(self, plan):
        """#440's case before #1199: a driver no rule scales stays flagged."""
        template = _repo_json(CATALOG_MUSIC)
        for entry in template["cost"]:
            entry.pop("scales_with", None)
        answer = plan(template, arguments={"audio_duration": 240})["estimate"]
        assert (answer["minutes"], answer["basis"]) == (3.6, "catalog")
        assert answer["low_confidence"] is True
        assert answer["differs"] == ["audio_duration"]


CATALOG_MUSIC_LONG = "workflows/templates/minimax/music-long.json"


def proportional(spec, *names):
    for entry in spec["cost"]:
        entry["scales_with"] = list(names)
    return spec


class TestDeclaredProportionalDrivers:
    """#1199 drill: `templates/minimax/music` quoted 3.6 min whatever its
    `audio_duration`, and `music-long` had no figure at all. A cost entry's
    `scales_with` names the drivers its figure is proportional to - a scalar
    variable, or a field summed over a list driver's entries - so the
    workflow, not the engine, says what a music ceiling costs."""

    def test_a_declared_scalar_scales_in_proportion(self, plan):
        spec = proportional(definition(), "audio_duration")
        spec["variables"]["audio_duration"] = 120
        spec["cost_drivers"] = ["audio_duration"]
        answer = plan(spec, arguments={"audio_duration": 240})["estimate"]
        assert (answer["minutes"], answer["basis"]) == (20.0, "derived")
        assert answer["scaled"] == (
            "scaled from the curated 10 min: audio_duration 240 vs 120"
        )
        assert "differs" not in answer

    def test_a_declared_scalar_at_its_default_is_the_figure(self, plan):
        spec = proportional(definition(), "audio_duration")
        spec["variables"]["audio_duration"] = 120
        spec["cost_drivers"] = ["audio_duration"]
        answer = plan(spec)["estimate"]
        assert (answer["minutes"], answer["basis"]) == (10.0, "catalog")

    def test_a_declared_field_is_summed_over_a_list(self, plan):
        spec = proportional(definition(), "audio_duration")
        spec["variables"]["shots"] = [
            {"name": "a", "prompt": "one", "audio_duration": 45},
            {"name": "b", "prompt": "two", "audio_duration": 45},
        ]
        spec["cost_drivers"] = ["shots"]
        spec["variables"]["takes"] = [{"audio_duration": 45}]
        spec["cost_drivers"].append("takes")
        takes = [{"audio_duration": 90}, {"audio_duration": 45}]
        # `takes` is priced by no for_each, so its sum carries the ratio
        answer = plan(spec, arguments={"takes": takes})["estimate"]
        assert (answer["minutes"], answer["basis"]) == (30.0, "derived")
        assert answer["scaled"].endswith("audio_duration 135 vs 45 over takes")

    def test_every_catalog_scales_with_names_a_driver_or_its_field(self):
        import pathlib

        root = pathlib.Path(__file__).resolve().parent.parent
        found = 0
        for path in sorted((root / "workflows").rglob("*.json")):
            spec = json.loads(path.read_text())
            if not isinstance(spec, dict):
                continue
            variables = spec.get("variables") or {}
            drivers = [n for n in spec.get("cost_drivers") or [] if n in variables]
            reachable = set(drivers)
            for name in drivers:
                if isinstance(variables[name], list):
                    for entry in variables[name]:
                        if isinstance(entry, dict):
                            reachable.update(entry)
            for entry in spec.get("cost") or []:
                for name in entry.get("scales_with") or []:
                    found += 1
                    assert name in reachable, f"{path.name}: scales_with {name}"
        assert found

    def test_music_scales_with_its_ceiling(self, plan):
        template = _repo_json(CATALOG_MUSIC)
        answer = plan(template, arguments={"audio_duration": 240})["estimate"]
        assert (answer["minutes"], answer["basis"]) == (7.2, "derived")
        assert "audio_duration 240 vs 120" in answer["scaled"]
        assert "differs" not in answer

    def test_music_long_quotes_its_defaults(self, plan):
        template = _repo_json(CATALOG_MUSIC_LONG)
        answer = plan(template)["estimate"]
        assert answer["basis"] == "catalog"
        assert answer["minutes"] is not None

    def test_music_long_scales_with_its_sections_and_their_ceilings(self, plan):
        template = _repo_json(CATALOG_MUSIC_LONG)
        curated = next(e for e in template["cost"] if e["device"] == "cuda")["minutes"]
        defaults = template["variables"]["sections"]
        was = sum(entry["audio_duration"] for entry in defaults)
        sections = [
            dict(
                defaults[n % len(defaults)],
                name=f"part_{n}",
                seconds=85,
                audio_duration=128,
            )
            for n in range(4)
        ]
        answer = plan(template, arguments={"sections": sections})["estimate"]
        assert answer["basis"] == "derived"
        assert answer["minutes"] == pytest.approx(curated * 512 / was, abs=0.1)

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

    def test_narration_long_is_priced_through_its_child(self, plan, tmp_path):
        """The shipped template's own step is a split task; its cost block
        covers that, so the estimate is the child's and not partial (#1222)."""
        import shutil
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent / "workflows" / "templates"
        for name in ("narration-long.json", "narration-from-takes.json"):
            shutil.copy(root / name, tmp_path / name)
        spec = json.loads((tmp_path / "narration-long.json").read_text())
        answer = plan(spec)["estimate"]
        assert answer["minutes"] is not None
        assert (answer["partial"], answer["unpriced"]) == (False, [])
        assert any("unknown until the run" in note for note in answer["differs"])

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
        import dw.plan_downloads

        monkeypatch.setattr(
            dw.plan_downloads,
            "scan_models",
            lambda cache_dir=None: {"repos": [{"repo_id": "org/still-model"}]},
        )
        monkeypatch.setattr(
            dw.plan_downloads, "repo_download_incomplete", lambda *a, **k: False
        )
        assert plan()["downloads_required"] == []

    def test_a_present_but_incomplete_repo_is_still_required(self, plan, monkeypatch):
        """#382: a repo scan_models lists (an interrupted pull left the
        revision folder in place) but whose snapshot is missing files the
        load needs stays in downloads_required rather than reading as
        cached."""
        import dw.plan_downloads

        monkeypatch.setattr(
            dw.plan_downloads,
            "scan_models",
            lambda cache_dir=None: {"repos": [{"repo_id": "org/still-model"}]},
        )
        seen = []

        def fake_incomplete(name, cache_dir, variant=None):
            seen.append((name, variant))
            return True

        monkeypatch.setattr(
            dw.plan_downloads, "repo_download_incomplete", fake_incomplete
        )
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
        import dw.plan_downloads

        seen = []

        def spy(cache_dir=None):
            seen.append(cache_dir)
            return {"repos": []}

        monkeypatch.setattr(dw.plan_downloads, "scan_models", spy)
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

    @pytest.mark.parametrize(
        "override, expected",
        [
            ("org/other", ["org/still-model", "org/other"]),
            (
                {"transformer_ref": "org/other", "vae": "org/vae"},
                ["org/still-model", "org/other", "org/vae"],
            ),
            ({"default": "org/other"}, ["org/still-model", "org/other"]),
            ({"default": "org/still-model"}, ["org/still-model"]),
            (None, ["org/still-model"]),
        ],
    )
    def test_a_load_components_repo_override_is_listed(self, plan, override, expected):
        """#981: the string, per-component and default forms, deduplicated
        against model_name, and unchanged when the key is absent."""
        spec = definition()
        if override is not None:
            spec["steps"][0]["pipeline"]["load_components"] = {
                "pretrained_model_name_or_path": override
            }
        assert [d["repo"] for d in plan(spec)["downloads_required"]] == expected

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

    def test_a_sam3_track_people_step_lists_only_the_sam3_repo(self, plan):
        spec = definition()
        spec["steps"].append(
            {
                "name": "people",
                "task": {"command": "track_people", "arguments": {"backend": "sam3"}},
            }
        )
        repos = [d["repo"] for d in plan(spec)["downloads_required"]]
        assert repos == ["org/still-model", "facebook/sam3"]
        assert "IDEA-Research/grounding-dino-base" not in repos
        assert "facebook/sam2-hiera-large" not in repos

    def test_a_sam3_repo_cached_as_its_config_alone_is_still_listed(
        self, plan, tmp_path, monkeypatch
    ):
        # #1404: mini-ai's cache held facebook/sam3's config.json and nothing
        # else, and the plan read the repo as present - so no quote, no gate
        import dw.plan_downloads

        cache = tmp_path / "hub"
        repo = cache / "models--facebook--sam3"
        snapshot = repo / "snapshots" / "c0ffee"
        snapshot.mkdir(parents=True)
        (repo / "refs").mkdir()
        (repo / "refs" / "main").write_text("c0ffee")
        (snapshot / "config.json").write_text(
            json.dumps({"architectures": ["Sam3VideoModel"]})
        )
        monkeypatch.setattr(
            dw.plan_downloads,
            "scan_models",
            lambda cache_dir=None: {"repos": [{"repo_id": "facebook/sam3"}]},
        )
        spec = {
            "id": "people",
            "steps": [
                {
                    "name": "people",
                    "task": {
                        "command": "track_people",
                        "arguments": {"video": "v.mp4", "backend": "sam3"},
                    },
                }
            ],
        }
        repos = [
            d["repo"] for d in plan(spec, cache_dir=str(cache))["downloads_required"]
        ]
        assert repos == ["facebook/sam3"]
        # Its weights arriving makes it cached, and the quote goes away
        (snapshot / "model.safetensors").write_bytes(b"x")
        assert plan(spec, cache_dir=str(cache))["downloads_required"] == []

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
        import dw.plan_downloads

        monkeypatch.setattr(
            dw.plan_downloads,
            "scan_models",
            lambda cache_dir=None: {
                "repos": [
                    {"repo_id": "org/still-model"},
                    {"repo_id": "briaai/RMBG-2.0"},
                ]
            },
        )
        monkeypatch.setattr(
            dw.plan_downloads, "repo_download_incomplete", lambda *a, **k: False
        )
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
        import dw.plan_downloads

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

        monkeypatch.setattr(dw.plan_downloads, "model_info", fake_model_info)
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
        import dw.plan_downloads

        class Sibling:
            rfilename = "config.json"

        class Info:
            siblings = [Sibling()]
            gated = "manual"

        monkeypatch.setattr(dw.plan_downloads, "model_info", lambda name, **k: Info())
        monkeypatch.setattr(
            dw.plan_downloads, "get_hf_file_metadata", lambda url, **k: object()
        )
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
        import dw.plan_downloads
        from huggingface_hub.utils import GatedRepoError

        response = httpx.Response(403, request=httpx.Request("GET", "https://hf.co/x"))

        class Sibling:
            rfilename = "config.json"

        class Info:
            siblings = [Sibling()]
            gated = "manual"

        def boom(*a, **k):
            raise GatedRepoError("no access", response=response)

        monkeypatch.setattr(dw.plan_downloads, "model_info", lambda name, **k: Info())
        monkeypatch.setattr(dw.plan_downloads, "get_hf_file_metadata", boom)
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
        import dw.plan_downloads
        from huggingface_hub.utils import GatedRepoError

        response = httpx.Response(403, request=httpx.Request("GET", "https://hf.co/x"))

        def boom(*a, **k):
            raise GatedRepoError("no access", response=response)

        monkeypatch.setattr(dw.plan_downloads, "model_info", boom)
        assert plan(lookup_sizes=True)["downloads_required"] == [
            {
                "repo": "org/still-model",
                "gb": None,
                "gated": True,
                "access_blocked": True,
            }
        ]

    def test_a_repo_the_hub_does_not_have_is_not_found(self, plan, monkeypatch):
        """A misspelled model_name answers 401 RepositoryNotFound: the entry
        says so, where an unanswered lookup stays all-null."""
        import httpx
        import dw.plan_downloads
        from huggingface_hub.utils import RepositoryNotFoundError

        response = httpx.Response(401, request=httpx.Request("GET", "https://hf.co/x"))

        def boom(*a, **k):
            raise RepositoryNotFoundError("no such repo", response=response)

        monkeypatch.setattr(dw.plan_downloads, "model_info", boom)
        assert plan(lookup_sizes=True)["downloads_required"] == [
            {
                "repo": "org/still-model",
                "gb": None,
                "gated": None,
                "access_blocked": None,
                "not_found": True,
            }
        ]

    def test_an_unanswered_lookup_is_not_not_found(self, plan, monkeypatch):
        """Offline is a state, not an error - only the hub's own answer
        marks a repo missing."""
        import dw.plan_downloads

        def offline(*a, **k):
            raise OSError("network unreachable")

        monkeypatch.setattr(dw.plan_downloads, "model_info", offline)
        [entry] = plan(lookup_sizes=True)["downloads_required"]
        assert "not_found" not in entry

    def test_a_gated_repo_with_no_probeable_file_is_unknown(self, plan, monkeypatch):
        import dw.plan_downloads

        class Info:
            siblings = []
            gated = "auto"

        monkeypatch.setattr(dw.plan_downloads, "model_info", lambda name, **k: Info())
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
        import dw.plan_downloads

        class Sibling:
            rfilename = "config.json"

        class Info:
            siblings = [Sibling()]
            gated = "auto"

        def boom(*a, **k):
            raise RuntimeError("offline")

        monkeypatch.setattr(dw.plan_downloads, "model_info", lambda name, **k: Info())
        monkeypatch.setattr(dw.plan_downloads, "get_hf_file_metadata", boom)
        assert plan(lookup_sizes=True)["downloads_required"] == [
            {
                "repo": "org/still-model",
                "gb": None,
                "gated": "auto",
                "access_blocked": None,
            }
        ]

    def test_a_hub_failure_is_a_null_size(self, plan, monkeypatch):
        import dw.plan_downloads

        def boom(*a, **k):
            raise RuntimeError("offline")

        monkeypatch.setattr(dw.plan_downloads, "model_info", boom)
        assert plan(lookup_sizes=True)["downloads_required"] == [
            {
                "repo": "org/still-model",
                "gb": None,
                "gated": None,
                "access_blocked": None,
            }
        ]

    def test_lookup_sizes_false_never_calls_the_hub(self, plan, monkeypatch):
        import dw.plan_downloads

        def boom(*a, **k):
            raise AssertionError("must not be called")

        monkeypatch.setattr(dw.plan_downloads, "model_info", boom)
        plan(lookup_sizes=False)


BASE = "org/flex"
OTHER = "org/base"
BASE_REV = "a" * 40
OTHER_REV = "b" * 40
WEIGHTS = ("transformer", "text_encoder", "vae")
OWN = ("scheduler", "tokenizer")


def _modular_index():
    """A modular_model_index.json: three weight components and two small ones,
    every spec on BASE, and one key that is no component."""
    entry = lambda name: [  # noqa: E731
        "diffusers",
        "Thing",
        {"pretrained_model_name_or_path": BASE, "subfolder": name, "revision": None},
    ]
    index = {"_class_name": "FlexModularPipeline"}
    index.update({name: entry(name) for name in WEIGHTS + OWN})
    return index


def _flex_step(load_components=None):
    """A modular step on BASE whose load_components sends the weights to OTHER
    - under configuration, as the catalog's Pika templates write it."""
    load_components = load_components or {
        "pretrained_model_name_or_path": {name: OTHER for name in WEIGHTS},
        "revision": {**{name: OTHER_REV for name in WEIGHTS}, "default": BASE_REV},
    }
    return {
        "name": "flex",
        "pipeline": {
            "configuration": {
                "component_type": "ModularPipeline",
                "load_components": load_components,
            },
            "from_pretrained_arguments": {"model_name": BASE, "workflow": "ref2va"},
            "loras": [
                {"model_name": BASE, "weight_name": "ref2va/adapter.safetensors"}
            ],
            "arguments": {},
        },
        "result": {"content_type": "video/mp4"},
    }


def _flex_spec(*steps):
    spec = definition()
    spec["steps"] = list(steps) or [_flex_step()]
    return spec


class _Sibling:
    def __init__(self, rfilename, size):
        self.rfilename = rfilename
        self.size = size


class _Info:
    gated = False

    def __init__(self, siblings):
        self.siblings = siblings


GIB = 1024**3


class TestAComponentOverrideIsNotTheWholeBaseRepo:
    """#1343: a modular step whose load_components loads its weights from
    another repo fetches only what is left on its own - the base repo's
    components that no override moved, by its modular_model_index.json - and
    its adapter file, not the base repo whole."""

    @pytest.fixture
    def hub(self, monkeypatch):
        import dw.plan_downloads

        calls = []
        siblings = [_Sibling(f"{name}/weights.bin", 50 * GIB) for name in WEIGHTS]
        siblings += [_Sibling(f"{name}/config.json", GIB // 2) for name in OWN]
        siblings += [_Sibling("ref2va/adapter.safetensors", 2 * GIB)]

        def fake_model_info(name, revision=None, **k):
            calls.append((name, revision))
            return _Info(siblings if name == BASE else [_Sibling("x", 10 * GIB)])

        class Metadata:
            size = 2 * GIB

        monkeypatch.setattr(dw.plan_downloads, "model_info", fake_model_info)
        monkeypatch.setattr(
            dw.plan_downloads, "get_hf_file_metadata", lambda url, **k: Metadata()
        )
        monkeypatch.setattr(
            dw.plan_downloads,
            "_fetch_modular_index",
            lambda repo, rev: _modular_index(),
        )
        return calls

    def test_only_the_components_left_on_the_base_repo_and_the_adapter(self, plan, hub):
        assert plan(_flex_spec(), lookup_sizes=True)["downloads_required"] == [
            {
                "repo": BASE,
                "components": ["scheduler", "tokenizer"],
                "revision": BASE_REV,
                "gb": 1.0,
                "gated": False,
                "access_blocked": False,
            },
            {"repo": OTHER, "gb": 10.0, "gated": False, "access_blocked": False},
            {
                "repo": BASE,
                "filename": "ref2va/adapter.safetensors",
                "revision": None,
                "gb": 2.0,
                "gated": None,
                "access_blocked": False,
            },
        ]
        assert (BASE, BASE_REV) in hub

    def test_a_component_already_cached_is_not_listed(self, plan, hub, monkeypatch):
        import dw.plan_downloads

        monkeypatch.setattr(
            dw.plan_downloads,
            "_cached_file",
            lambda repo, filename, revision, cache_dir=None: (
                "/cache/x" if repo == BASE and filename.startswith(OWN) else None
            ),
        )
        repos = plan(_flex_spec(), lookup_sizes=True)["downloads_required"]
        assert [(d["repo"], d.get("filename")) for d in repos] == [
            (OTHER, None),
            (BASE, "ref2va/adapter.safetensors"),
        ]

    def test_an_unreadable_index_lists_the_base_repo_whole(
        self, plan, hub, monkeypatch
    ):
        """Never a miss: without the index the base repo stands for it, and
        the whole-repo entry covers the adapter file."""
        import dw.plan_downloads

        def offline(repo, rev):
            raise RuntimeError("offline")

        monkeypatch.setattr(dw.plan_downloads, "_fetch_modular_index", offline)
        repos = plan(_flex_spec(), lookup_sizes=True)["downloads_required"]
        assert [(d["repo"], d.get("components"), d.get("filename")) for d in repos] == [
            (BASE, None, None),
            (OTHER, None, None),
        ]

    def test_offline_with_the_index_and_its_components_cached_lists_nothing(
        self, plan, tmp_path, monkeypatch
    ):
        import dw.plan_downloads

        snapshot = tmp_path / "snapshot"
        for name in OWN:
            (snapshot / name).mkdir(parents=True)
            (snapshot / name / "config.json").write_text("{}")
        (snapshot / "modular_model_index.json").write_text(json.dumps(_modular_index()))

        def cached(repo, filename, revision, cache_dir=None):
            if repo == BASE and filename in (
                "modular_model_index.json",
                "ref2va/adapter.safetensors",
            ):
                return str(snapshot / filename)
            return None

        monkeypatch.setattr(dw.plan_downloads, "_cached_file", cached)
        repos = plan(_flex_spec())["downloads_required"]
        assert [d["repo"] for d in repos] == [OTHER]

    def test_offline_with_a_component_not_cached_lists_it_unsized(
        self, plan, tmp_path, monkeypatch
    ):
        import dw.plan_downloads

        snapshot = tmp_path / "snapshot"
        snapshot.mkdir()
        (snapshot / "modular_model_index.json").write_text(json.dumps(_modular_index()))
        monkeypatch.setattr(
            dw.plan_downloads,
            "_cached_file",
            lambda repo, filename, revision, cache_dir=None: (
                str(snapshot / filename)
                if filename == "modular_model_index.json"
                else None
            ),
        )
        [flex, other, adapter] = plan(_flex_spec())["downloads_required"]
        assert flex == {
            "repo": BASE,
            "components": ["scheduler", "tokenizer"],
            "revision": BASE_REV,
            "gb": None,
            "gated": None,
            "access_blocked": None,
        }
        assert other["repo"] == OTHER
        assert adapter["filename"] == "ref2va/adapter.safetensors"

    def test_a_string_override_moves_every_component(self, plan, hub):
        """`pretrained_model_name_or_path` as one string sends all of them
        away: nothing is left to fetch from the base repo but the adapter."""
        step = _flex_step({"pretrained_model_name_or_path": OTHER})
        repos = plan(_flex_spec(step), lookup_sizes=True)["downloads_required"]
        assert [(d["repo"], d.get("filename")) for d in repos] == [
            (OTHER, None),
            (BASE, "ref2va/adapter.safetensors"),
        ]

    def test_a_local_path_override_plans_the_base_as_without_one(
        self, plan, hub, monkeypatch
    ):
        """#1490: a component moved to a local folder fetches nothing, so the
        base repo is judged whole (cached -> no entry), not by component."""
        import dw.plan_downloads

        monkeypatch.setattr(
            dw.plan_downloads,
            "scan_models",
            lambda cache_dir: {"repos": [{"repo_id": BASE}]},
        )
        monkeypatch.setattr(
            dw.plan_downloads, "repo_download_incomplete", lambda *a, **k: False
        )
        step = _flex_step(
            {"pretrained_model_name_or_path": {"transformer_ref": "/scratch/graft"}}
        )
        repos = plan(_flex_spec(step), lookup_sizes=True)["downloads_required"]
        assert not [d for d in repos if d["repo"] == BASE and not d.get("filename")]

    def test_the_same_base_loaded_with_no_override_elsewhere_is_listed_whole(
        self, plan, hub
    ):
        plain = _flex_step()
        plain["name"] = "plain"
        del plain["pipeline"]["configuration"]["load_components"]
        repos = plan(_flex_spec(_flex_step(), plain), lookup_sizes=True)[
            "downloads_required"
        ]
        assert [(d["repo"], d.get("components"), d.get("filename")) for d in repos] == [
            (BASE, None, None),
            (OTHER, None, None),
        ]


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


class TestMissingRepoErrors:
    def test_a_missing_repo_is_an_error(self):
        required = [
            {"repo": "org/real", "gb": 1.0, "gated": False, "access_blocked": False},
            {
                "repo": "org/typo",
                "gb": None,
                "gated": None,
                "access_blocked": None,
                "not_found": True,
            },
        ]
        [error] = missing_repo_errors(required)
        assert error["path"] == "model_name"
        assert "'org/typo'" in error["message"]

    def test_unknown_is_not_an_error(self):
        required = [
            {"repo": "org/offline", "gb": None, "gated": None, "access_blocked": None}
        ]
        assert missing_repo_errors(required) == []


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

    def test_a_seed_variable_with_no_value_is_named(self):
        spec = definition()
        spec["seed"] = "variable:my_seed"
        spec["variables"] = {"my_seed": None}
        (warning,) = unseeded_cache_warnings(spec)
        assert warning.startswith("The seed variable 'my_seed' has no value")
        assert "Give 'my_seed' a default, or pass one" in warning
        assert "sets no 'seed'" not in warning

    def test_no_seed_at_all_keeps_the_original_warning(self):
        spec = definition()
        del spec["seed"]
        (warning,) = unseeded_cache_warnings(spec)
        assert warning.startswith("This workflow sets no 'seed'")


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
        import dw.plan_downloads

        monkeypatch.setattr(
            dw.plan_downloads,
            "scan_models",
            lambda cache_dir=None: {"repos": [{"repo_id": name} for name in present]},
        )
        monkeypatch.setattr(
            dw.plan_downloads, "repo_download_incomplete", lambda *a, **k: False
        )
        # The adapter names a file, so it is checked file by file
        monkeypatch.setattr(
            dw.plan_downloads,
            "_cached_file",
            lambda repo, *a, **k: "cached" if repo in present else None,
        )
        return [
            entry["repo"]
            for entry in dw.plan_downloads.downloads_required(
                self.definition(), None, None, None, False
            )
        ]

    def test_a_single_file_adapter_is_sized_as_the_file_not_the_repo(self, monkeypatch):
        """#1088: priced as its whole repo (104.7 GB for a 5.9 GB file)."""
        import dw.plan_downloads

        monkeypatch.setattr(dw.plan_downloads, "scan_models", lambda *a: {})
        monkeypatch.setattr(dw.plan_downloads, "_cached_file", lambda *a: None)
        monkeypatch.setattr(
            dw.plan_downloads,
            "_model_info",
            lambda name, *a: (104.7, False, None),
        )
        monkeypatch.setattr(
            dw.plan_downloads,
            "_file_info",
            lambda repo, filename, revision: (5.9, None),
        )
        definition = self.definition()
        definition["steps"][0]["pipeline"]["loras"][0]["subfolder"] = "loras"
        entries = dw.plan_downloads.downloads_required(
            definition, None, None, None, True
        )
        adapter = [e for e in entries if "IC-LoRA" in e["repo"]]
        assert [(e["gb"], e["filename"]) for e in adapter] == [
            (5.9, "loras/ic-lora.safetensors")
        ]

    def test_an_adapter_without_a_weight_name_is_still_the_whole_repo(
        self, monkeypatch
    ):
        import dw.plan_downloads

        monkeypatch.setattr(dw.plan_downloads, "scan_models", lambda *a: {})
        definition = self.definition()
        del definition["steps"][0]["pipeline"]["loras"][0]["weight_name"]
        entries = dw.plan_downloads.downloads_required(
            definition, None, None, None, False
        )
        assert [(e["repo"], "filename" in e) for e in entries] == [
            ("Lightricks/LTX-2.5-22b-IC-LoRA-Ingredients", False),
            ("Lightricks/LTX-2.5-Diffusers", False),
        ]

    def test_an_adapter_file_climbing_out_is_priced_as_the_repo(self, monkeypatch):
        """file_entries refuses a name that is not plain and relative, so the
        cache lookup never joins it onto the disk; the repo stands for it."""
        import dw.plan_downloads

        monkeypatch.setattr(dw.plan_downloads, "scan_models", lambda *a: {})
        looked_up = []
        monkeypatch.setattr(
            dw.plan_downloads, "_cached_file", lambda *a: looked_up.append(a)
        )
        definition = self.definition()
        definition["steps"][0]["pipeline"]["loras"][0]["subfolder"] = ".."
        entries = dw.plan_downloads.downloads_required(
            definition, None, None, None, False
        )
        assert [(e["repo"], "filename" in e) for e in entries] == [
            ("Lightricks/LTX-2.5-22b-IC-LoRA-Ingredients", False),
            ("Lightricks/LTX-2.5-Diffusers", False),
        ]
        assert looked_up == []

    def test_a_file_cached_under_an_older_snapshot_is_no_download(
        self, monkeypatch, tmp_path
    ):
        """#1392: `refs/main` moved to a snapshot without the file, but the
        blob is cached under an older one, so the loader links it with no
        download. Resolved through a `variable:` weight name, on a real
        cache layout."""
        import dw.plan_downloads
        from dw.workflow import Workflow

        repo = "org/turbo"
        name = "turbo.safetensors"
        root = tmp_path / "models--org--turbo"
        (root / "blobs").mkdir(parents=True)
        (root / "blobs" / ("a" * 64)).write_bytes(b"x")
        for commit in ("old", "new"):
            (root / "snapshots" / commit).mkdir(parents=True)
        (root / "snapshots" / "old" / name).symlink_to(root / "blobs" / ("a" * 64))
        (root / "refs").mkdir()
        (root / "refs" / "main").write_text("new")
        monkeypatch.setattr(
            dw.plan_downloads,
            "scan_models",
            lambda *a: {"repos": [{"repo_id": "Lightricks/LTX-2.5-Diffusers"}]},
        )
        monkeypatch.setattr(
            dw.plan_downloads, "repo_download_incomplete", lambda *a, **k: False
        )
        definition = self.definition()
        definition["variables"] = {"weight": name}
        lora = definition["steps"][0]["pipeline"]["loras"][0]
        lora["model_name"] = repo
        lora["weight_name"] = "variable:weight"
        expanded = Workflow(definition, str(tmp_path), None).expanded_definition({})
        assert expanded["steps"][0]["pipeline"]["loras"][0]["weight_name"] == name
        required = dw.plan_downloads.downloads_required(
            expanded, None, None, str(tmp_path), False
        )
        assert [e["repo"] for e in required] == []
        # A file no snapshot holds is still a download
        lora["weight_name"] = "other.safetensors"
        required = dw.plan_downloads.downloads_required(
            definition, None, None, str(tmp_path), False
        )
        assert [e["repo"] for e in required] == [repo]

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
            "Lightricks/LTX-2.5-Diffusers",
            "Lightricks/LTX-2.5-22b-IC-LoRA-Ingredients",
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


CATALOG_CHAINED = "workflows/templates/minimax/chained-segments.json"


class TestTiersPriceFromTheObservedDefault:
    """#1236 on the real templates: with this box's history of the default
    tier, every other tier is priced from it, not from the curated row -
    `best`, which moves `weights_dtype` too, and a `draft` whose own history
    is a single run, tempered toward the observed default scaled rather than
    toward the curated row scaled."""

    @staticmethod
    def history(by_tier):
        def answer(arguments):
            return by_tier.get(arguments.get("tier"))

        return answer

    def quote(self, plan, path, arguments, by_tier):
        template = _repo_json(path)
        return plan(template, arguments=arguments, observed=self.history(by_tier))[
            "estimate"
        ]

    def test_best_scales_the_observed_default_and_names_weights(self, plan):
        by_tier = {None: observed(minutes=5.5, runs=7)}
        args = {"segments": 2}
        standard = self.quote(plan, CATALOG_CHAINED, args, by_tier)
        best = self.quote(plan, CATALOG_CHAINED, {**args, "tier": "best"}, by_tier)
        assert (standard["minutes"], standard["basis"]) == (5.5, "observed")
        assert best["basis"] == "derived"
        assert best["scaled"] == (
            "scaled from the observed 5.5 min (4.1 min of it fixed): "
            "canvas 1344x768 vs 960x544"
        )
        assert best["differs"] == ["weights_dtype"]
        assert best["low_confidence"] is True
        # The entry's fixed share (10 of 13.5 min) held, the rest scaled by
        # (1344 x 768) / (960 x 544), as the curated row is (#1224)
        share = 10 / 13.5
        area = (1344 * 768) / (960 * 544)
        assert best["minutes"] == pytest.approx(
            5.5 * (share + (1 - share) * area), abs=0.05
        )
        assert standard["minutes"] < best["minutes"]

    def test_dialogue_short_best_scales_its_observed_default(self, plan):
        by_tier = {None: observed(minutes=38.6, runs=5)}
        best = self.quote(plan, CATALOG_DIALOGUE, {"tier": "best"}, by_tier)
        assert best["basis"] == "derived"
        assert best["scaled"].startswith("scaled from the observed 38.6 min")
        assert "curated" not in best["scaled"]
        assert best["minutes"] > 38.6

    def test_a_thin_draft_bucket_tempers_toward_the_observed_default(self, plan):
        by_tier = {
            None: observed(minutes=5.5, runs=7),
            "draft": observed(minutes=3.0, runs=1),
        }
        args = {"segments": 2}
        standard = self.quote(plan, CATALOG_CHAINED, args, by_tier)
        draft = self.quote(plan, CATALOG_CHAINED, {**args, "tier": "draft"}, by_tier)
        assert draft["basis"] == "observed"
        assert draft["tempered"] is True
        assert draft["observed_minutes"] == 3.0
        # 5.5 with its fixed share (10 of 13.5 min) held and the rest x 4/5
        # steps - not the curated 13.5 scaled
        share = 10 / 13.5
        partner = 5.5 * (share + (1 - share) * 4 / 5)
        assert draft["curated_minutes"] == pytest.approx(partner, abs=0.05)
        assert draft["minutes"] == pytest.approx(partner * 2 / 3 + 3.0 / 3, abs=0.05)
        assert (
            "scaled from the observed 5.5 min (4.1 min of it fixed): steps 4 vs 5"
            in draft["scaled"]
        )
        assert "curated" not in draft["scaled"]
        assert "tier" not in (draft.get("differs") or [])
        assert draft["minutes"] < standard["minutes"]

    def test_a_thin_bucket_with_no_default_history_tempers_toward_curated(self, plan):
        by_tier = {"draft": observed(minutes=3.0, runs=1)}
        args = {"segments": 2, "tier": "draft"}
        draft = self.quote(plan, CATALOG_CHAINED, args, by_tier)
        assert draft["tempered"] is True
        # 10 of the curated 13.5 min is the template's fixed_minutes (#1224):
        # 10 + 3.5 x (the 7.2 / 13.5 proportional factor)
        assert draft["curated_minutes"] == pytest.approx(11.87, abs=0.05)
        assert "scaled from the curated 13.5 min" in draft["scaled"]


class TestCheckedChainQuotesMoreThanUnchecked:
    """#1334 (C-F747): a checked dialogue chain whose thin history tempers
    below the curated figure still quotes more than the same chain
    unchecked, which has no history and quotes the curated figure."""

    COMMON = {
        "shot": "dialogue",
        "segments": 2,
        "num_frames": 124,
        "seed": 1262,
        "prompts": [
            "A man speaks. <d>We the people of the United States.</d>",
            "A man speaks. <d>In order to form a more perfect union.</d>",
        ],
    }
    CHECKED = {**COMMON, "retries": 1, "check": {"words_similarity_min": 0.8}}
    UNCHECKED = {**COMMON, "retries": 0, "check": False}

    @staticmethod
    def history(checked=None, unchecked=None):
        def answer(arguments):
            return unchecked if arguments.get("check") is False else checked

        return answer

    def quote(self, plan, arguments, history):
        return plan(_repo_json(CATALOG_CHAINED), arguments=arguments, observed=history)[
            "estimate"
        ]

    def test_a_thin_fast_checked_history_quotes_above_the_unchecked(self, plan):
        history = self.history(checked=observed(minutes=5.5, runs=1))
        checked = self.quote(plan, self.CHECKED, history)
        unchecked = self.quote(plan, self.UNCHECKED, history)
        assert unchecked["minutes"] == pytest.approx(12.3, abs=0.05)
        assert checked["minutes"] > unchecked["minutes"]
        # Floored at the unchecked chain (curated, no history) at its worst
        # case: the curated 13.5 for 2 of 3 segments, 4 renders for 2, plus
        # 4 Whisper passes
        assert checked["minutes"] == pytest.approx(14.87, abs=0.05)
        assert "4 renders for 2" in checked["scaled"]
        assert "words check: 4 Whisper pass(es)" in checked["scaled"]
        assert "floored at the same chain unchecked" in checked["scaled"]
        assert "checked history alone quoted 12.3 min" in checked["scaled"]

    def test_the_floor_reads_the_unchecked_chains_own_history(self, plan):
        history = self.history(
            checked=observed(minutes=5.5, runs=1),
            unchecked=observed(minutes=12.0, runs=5),
        )
        checked = self.quote(plan, self.CHECKED, history)
        unchecked = self.quote(plan, self.UNCHECKED, history)
        assert unchecked["minutes"] == 12.0
        # The twin's own 12 min plus the curated worst-case extra (14.87 -
        # 12.33), while the scaled text still names the checked worst case
        assert checked["minutes"] == pytest.approx(14.6, abs=0.05)
        assert "same chain unchecked (12 min)" in checked["scaled"]
        assert "4 renders for 2" in checked["scaled"]
        assert "words check: 4 Whisper pass(es)" in checked["scaled"]

    @pytest.mark.parametrize("twin_minutes", [5.5, 7.1, 9.0, 12.0])
    def test_strictly_above_the_twin_whatever_its_history(self, plan, twin_minutes):
        # The bounce: with both buckets thin and from a fast box, pricing the
        # checked shape on the twin's history could land at or under the
        # twin's own quote; the floor is the twin's quote plus the extra
        history = self.history(
            checked=observed(minutes=5.5, runs=1),
            unchecked=observed(minutes=twin_minutes, runs=1),
        )
        checked = self.quote(plan, self.CHECKED, history)
        unchecked = self.quote(plan, self.UNCHECKED, history)
        assert checked["minutes"] > unchecked["minutes"]
        assert "4 renders for 2" in checked["scaled"]
        assert "words check: 4 Whisper pass(es)" in checked["scaled"]

    def test_a_checked_history_above_the_floor_stands(self, plan):
        history = self.history(
            checked=observed(minutes=20.0, runs=5),
            unchecked=observed(minutes=6.0, runs=5),
        )
        checked = self.quote(plan, self.CHECKED, history)
        assert "floored" not in checked["scaled"]
        assert "from the observed 20 min" in checked["scaled"]

    def test_an_unchecked_chain_is_not_floored(self, plan):
        history = self.history(unchecked=observed(minutes=6.0, runs=1))
        unchecked = self.quote(plan, self.UNCHECKED, history)
        assert "floored" not in (unchecked.get("scaled") or "")


class TestResumedChainEstimate:
    """A chain resumed from an earlier run's kept segments prices only the
    segments it renders (chain_resume.kept_count, cost_scaling)."""

    RUN = "20261108-100000-aaaaaaaa"

    def _definition(self, resume_from):
        variables = {"segments": 4, "resume_from": None}
        definition = {
            "cost": [{"device": "cuda", "name": "card", "minutes": 40}],
            "cost_drivers": ["segments"],
            "variables": variables,
            "steps": [
                {
                    "pipeline": {
                        "chain": {
                            "segments": "variable:segments",
                            "fps": 4,
                            "save_segments": True,
                            "resume_from": "variable:resume_from",
                        },
                        "arguments": {},
                    }
                }
            ],
        }
        expanded = {"variables": {**variables, "resume_from": resume_from}}
        return definition, expanded

    @pytest.fixture
    def kept(self, tmp_path):
        """Two kept segment files of a dead run, under an active output root."""
        from dw.runs import activate_output_root, deactivate_output_root

        run = tmp_path / "outputs" / "chained" / self.RUN
        run.mkdir(parents=True)
        for index in (0, 1):
            (run / f"chained.video.0.segment-{index:03d}.mp4").write_bytes(b"")
        token = activate_output_root(str(tmp_path / "outputs"))
        yield f"chained/{self.RUN}/chained.video.0.segment-000.mp4"
        deactivate_output_root(token)

    def test_a_resolvable_resume_prices_the_rest(self, kept):
        from dw import references
        from dw.plan import estimate

        reference = references.make_ref(references.OUTPUT, kept)
        quoted = estimate(*self._definition(reference), {}, "cuda", None, None)

        assert quoted["minutes"] == pytest.approx(20.0)
        assert "2 of 4 segment(s) kept" in quoted["scaled"]

    def test_a_fan_out_resume_says_the_quote_assumes_every_iteration_resumes(
        self, kept
    ):
        from dw import references
        from dw.pipeline_processors.chain_resume import FANOUT_NOTE
        from dw.plan import estimate

        reference = references.make_ref(references.OUTPUT, kept)
        definition, expanded = self._definition(reference)
        step = definition["steps"][0]["pipeline"]
        step["arguments"]["prompt"] = "previous_result:earlier.prompts"
        quoted = estimate(definition, expanded, {}, "cuda", None, None)

        assert quoted["minutes"] == pytest.approx(20.0)
        assert FANOUT_NOTE in quoted["differs"]

        plain = estimate(*self._definition(reference), {}, "cuda", None, None)
        assert FANOUT_NOTE not in (plain.get("differs") or [])

    def test_a_reference_outside_the_chain_step_or_in_prose_is_no_fan_out(self, kept):
        from dw import references
        from dw.pipeline_processors.chain_resume import FANOUT_NOTE
        from dw.plan import estimate

        reference = references.make_ref(references.OUTPUT, kept)
        definition, expanded = self._definition(reference)
        step = definition["steps"][0]["pipeline"]
        step["arguments"]["prompt"] = "a previous_result of a long walk"
        definition["steps"].append(
            {
                "name": "other",
                "pipeline": {
                    "type": "X",
                    "arguments": {"prompt": "previous_result:chained"},
                },
            }
        )
        quoted = estimate(definition, expanded, {}, "cuda", None, None)

        assert FANOUT_NOTE not in (quoted.get("differs") or [])

    def test_an_unresolvable_resume_prices_every_segment_and_says_so(self, kept):
        from dw import references
        from dw.pipeline_processors.chain_resume import UNRESOLVED_NOTE
        from dw.plan import estimate

        missing = references.make_ref(
            references.OUTPUT, "chained/latest/none.0.segment-000.mp4"
        )
        quoted = estimate(*self._definition(missing), {}, "cuda", None, None)

        assert quoted["minutes"] == pytest.approx(40.0)
        assert UNRESOLVED_NOTE in quoted["differs"]

    def test_no_resume_is_todays_estimate(self, kept):
        from dw.plan import estimate

        quoted = estimate(*self._definition(None), {}, "cuda", None, None)

        assert quoted["minutes"] == pytest.approx(40.0)
        assert not quoted.get("differs")


class TestResumedChainWithNothingToRender(TestResumedChainEstimate):
    """A reference keeping every planned segment is not scaled, and says why (#1306)."""

    def test_the_quote_names_the_refusal_not_a_bare_resume_from(self, kept):
        from dw import references
        from dw.plan import estimate

        reference = references.make_ref(references.OUTPUT, kept)
        definition, expanded = self._definition(reference)
        expanded["variables"]["segments"] = 2
        quoted = estimate(definition, expanded, {}, "cuda", None, None)

        assert any("nothing is left to render" in d for d in quoted["differs"])
        assert quoted["low_confidence"] is True


class TestReplacePerformerPricesRenderFrames:
    """#1419: the estimate summed the source `num_frames`, so 32 short cuts
    each rendered at 124 frames were quoted far below 3 long ones."""

    def _cuts(self, count, num_frames):
        return [
            {
                "name": f"c{i}",
                "start_frame": i * num_frames,
                "num_frames": num_frames,
                "render_frames": 124,
            }
            for i in range(count)
        ]

    def test_many_short_cuts_cost_more_than_few_long_ones(self, plan):
        template = _repo_json("workflows/templates/minimax/replace-performer.json")
        many = plan(template, arguments={"cuts": self._cuts(32, 11)})["estimate"]
        few = plan(template, arguments={"cuts": self._cuts(3, 124)})["estimate"]
        assert many["minutes"] > few["minutes"]

    def test_a_cut_left_out_of_the_list_is_not_priced(self, plan):
        """An untouched cut is left out of `cuts` and replace_cuts keeps the
        take's frames for it, so the quote is for the listed cuts alone
        (0.21.0 review)."""
        template = _repo_json("workflows/templates/minimax/replace-performer.json")
        cuts = self._cuts(3, 124)
        whole = plan(template, arguments={"cuts": cuts})["estimate"]
        gap = plan(template, arguments={"cuts": [cuts[0], cuts[2]]})["estimate"]
        assert gap["minutes"] < whole["minutes"]


class TestACachedComponentFolderIsNotListed:
    """#1488: a component folder the snapshot holds whole is cached even when
    the hub's file list names small files a load never fetches."""

    def _snapshot(self, tmp_path):
        snap = tmp_path / "snap"
        (snap / "vae").mkdir(parents=True)
        (snap / "vae" / "config.json").write_text("{}")
        (snap / "vae" / "model.safetensors").write_bytes(b"x")
        (snap / "scheduler").mkdir()
        (snap / "scheduler" / "scheduler_config.json").write_text("{}")
        (snap / "modular_model_index.json").write_text("{}")
        return snap

    def test_held_folders_drop_out_despite_unfetched_siblings(
        self, tmp_path, monkeypatch
    ):
        import dw.plan_downloads as pd

        snap = self._snapshot(tmp_path)
        monkeypatch.setattr(
            pd,
            "_cached_file",
            lambda repo, name, *a, **k: (
                str(snap / name) if name == "modular_model_index.json" else None
            ),
        )
        extra = [_Sibling("vae/generation_config.json", 1000)]
        monkeypatch.setattr(pd, "model_info", lambda *a, **k: _Info(extra))
        assert (
            pd._components_entry(
                BASE,
                None,
                ["scheduler", "vae"],
                None,
                True,
                classes={"scheduler": "FlowScheduler", "vae": "AutoencoderKL"},
            )
            is None
        )

    def test_a_missing_folder_is_still_listed(self, tmp_path, monkeypatch):
        import dw.plan_downloads as pd

        snap = self._snapshot(tmp_path)
        monkeypatch.setattr(
            pd,
            "_cached_file",
            lambda repo, name, *a, **k: (
                str(snap / name) if name == "modular_model_index.json" else None
            ),
        )
        siblings = [_Sibling("tokenizer/vocab.json", GIB)]
        monkeypatch.setattr(pd, "model_info", lambda *a, **k: _Info(siblings))
        entry = pd._components_entry(
            BASE,
            None,
            ["vae", "tokenizer"],
            None,
            True,
            classes={"vae": "AutoencoderKL", "tokenizer": "T5Tokenizer"},
        )
        assert entry["components"] == ["tokenizer"] and entry["gb"] == 1.0

    def test_a_weight_bearing_folder_of_only_configs_is_still_listed(
        self, tmp_path, monkeypatch
    ):
        import dw.plan_downloads as pd

        snap = self._snapshot(tmp_path)
        (snap / "vae" / "model.safetensors").unlink()
        monkeypatch.setattr(
            pd,
            "_cached_file",
            lambda repo, name, *a, **k: (
                str(snap / name) if name == "modular_model_index.json" else None
            ),
        )
        siblings = [_Sibling("vae/model.safetensors", GIB)]
        monkeypatch.setattr(pd, "model_info", lambda *a, **k: _Info(siblings))
        entry = pd._components_entry(
            BASE, None, ["vae"], None, True, classes={"vae": "AutoencoderKL"}
        )
        assert entry["components"] == ["vae"] and entry["gb"] == 1.0


class TestLoadBytes:
    """#1580: a whole-repo estimate counts what `from_pretrained` fetches."""

    @staticmethod
    def sd15():
        from types import SimpleNamespace as S

        gb = 10**9
        rows = {
            "model_index.json": 1,
            "README.md": 14,
            "v1-5-pruned.safetensors": 7 * gb,
            "v1-5-pruned.ckpt": 7 * gb,
            "safety_checker/config.json": 1,
            "safety_checker/model.safetensors": 1 * gb,
            "safety_checker/pytorch_model.bin": 1 * gb,
            "unet/config.json": 1,
            "unet/diffusion_pytorch_model.safetensors": 3 * gb,
            "unet/diffusion_pytorch_model.non_ema.safetensors": 3 * gb,
            "unet/diffusion_pytorch_model.fp16.safetensors": 2 * gb,
            "unet/diffusion_pytorch_model.bin": 3 * gb,
            "unet/diffusion_pytorch_model.fp16.bin": 2 * gb,
        }
        return [S(rfilename=n, size=sz) for n, sz in rows.items()]

    def test_one_format_per_component_and_no_root_checkpoints(self):
        from dw.plan_downloads import _load_bytes

        assert _load_bytes(self.sd15()) == pytest.approx(4 * 10**9, abs=100)

    def test_the_requested_variant_is_the_one_counted(self):
        from dw.plan_downloads import _load_bytes

        # fp16 unet (2) + the checker, which has no fp16 file, falls back to plain (1)
        assert _load_bytes(self.sd15(), "fp16") == pytest.approx(3 * 10**9, abs=100)

    def test_a_variant_outside_the_common_names_is_matched(self):
        from types import SimpleNamespace as S

        from dw.plan_downloads import _load_bytes

        files = [
            S(rfilename="model_index.json", size=1),
            S(rfilename="unet/config.json", size=1),
            S(rfilename="unet/diffusion_pytorch_model.safetensors", size=8),
            S(rfilename="unet/diffusion_pytorch_model.int8.safetensors", size=2),
        ]
        assert _load_bytes(files, "int8") == 4

    def test_a_component_passed_as_null_is_not_counted(self):
        from dw.plan_downloads import _load_bytes

        assert _load_bytes(self.sd15(), None, {"safety_checker"}) == pytest.approx(
            3 * 10**9, abs=100
        )

    @staticmethod
    def sdxl():
        """SDXL base's shape: ONNX/OpenVINO-only folders the index omits."""
        from types import SimpleNamespace as S

        gb = 10**9
        rows = {
            "model_index.json": 1,
            "sd_xl_base_1.0.safetensors": 7 * gb,
            "unet/config.json": 1,
            "unet/diffusion_pytorch_model.safetensors": 10 * gb,
            "unet/diffusion_pytorch_model.fp16.safetensors": 5 * gb,
            "unet/openvino_model.bin": 10 * gb,
            "vae_decoder/config.json": 1,
            "vae_decoder/model.onnx": gb,
            "vae_decoder/openvino_model.bin": gb,
        }
        return [S(rfilename=n, size=sz) for n, sz in rows.items()]

    @pytest.mark.parametrize("variant, expected", [(None, 10), ("fp16", 5)])
    def test_a_folder_the_index_omits_does_not_drop_the_weights(
        self, monkeypatch, variant, expected
    ):
        import dw.plan_downloads as pd

        index = {"_class_name": "X", "unet": ["diffusers", "UNet2DConditionModel"]}
        monkeypatch.setattr(pd, "_fetch_index", lambda *a, **k: index)
        assert pd._repo_gb("org/sdxl", self.sdxl(), variant) == pytest.approx(
            expected * 10**9 / GIB, abs=0.1
        )

    def test_a_setting_or_absent_component_is_not_a_component_folder(self, monkeypatch):
        # A list-valued setting and a [null, null] optional component are not
        # components, whatever folder shares their name (#1741, #1745): their
        # pickle-only folders must not drop the safetensors of the real ones
        from types import SimpleNamespace as S

        import dw.plan_downloads as pd

        gb = 10**9
        index = {
            "_class_name": "X",
            "_hidden": ["diffusers", "UNet2DConditionModel"],
            "unet": ["diffusers", "UNet2DConditionModel"],
            "image_encoder": [None, None],
            "select_layers": [2, 5, 8],
        }
        rows = {
            "model_index.json": 1,
            "unet/diffusion_pytorch_model.safetensors": 10 * gb,
            "image_encoder/pytorch_model.bin": gb,
            "select_layers/pytorch_model.bin": gb,
        }
        siblings = [S(rfilename=n, size=sz) for n, sz in rows.items()]
        monkeypatch.setattr(pd, "_fetch_index", lambda *a, **k: index)

        assert pd._pipeline_components("org/repo") == {"unet"}
        assert pd._repo_gb("org/repo", siblings) >= 10 * gb / GIB - 0.1

    def test_an_unreadable_index_gives_no_size(self, monkeypatch):
        import dw.plan_downloads as pd

        def offline(*a, **k):
            raise OSError("offline")

        monkeypatch.setattr(pd, "_fetch_index", offline)
        assert pd._repo_gb("org/sdxl", self.sdxl()) is None

    def test_a_variant_the_repo_lacks_gives_no_size(self, monkeypatch):
        import dw.plan_downloads as pd

        monkeypatch.setattr(pd, "_fetch_index", lambda *a, **k: {"unet": ["d", "U"]})
        assert pd._repo_gb("org/sdxl", self.sdxl(), "int8") is None

    def test_a_tiny_repo_is_not_priced_at_zero(self):
        from types import SimpleNamespace as S

        from dw.plan_downloads import _repo_gb

        assert (
            _repo_gb("org/tiny", [S(rfilename="a.safetensors", size=3177927)]) == 0.01
        )

    def test_a_repo_without_a_model_index_is_summed_whole(self):
        from types import SimpleNamespace as S

        from dw.plan_downloads import _load_bytes

        files = [S(rfilename="a.safetensors", size=5), S(rfilename="a.ckpt", size=5)]
        assert _load_bytes(files) == 10

    def test_the_template_pairs_its_null_component_with_the_repo(self):
        import json

        from dw.plan_downloads import _walk

        definition = json.load(open("workflows/templates/text-to-image.json"))
        names, _v, _u, _f, _p, _w, skips = _walk(definition, None, None)
        assert skips["stable-diffusion-v1-5/stable-diffusion-v1-5"] == {
            "safety_checker"
        }
