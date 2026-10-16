"""#797: an inline workflow is priced by pipeline identity.

Observed cost history is keyed by catalog name, so an inline copy of a
template had none. `inherited_observed` finds the catalog template with the
same pipeline identity (the key `dw/vram_inheritance.py` matches on) and
quotes its runs under `basis: "inherited"`, warning where the inline
workflow's offload, quantization, frame count, attention backend, cache or
LoRA differ, and scaling its minutes by its size (#851, #866).
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

import dw.plan
from dw.plan import build_plan
from dw.server.observed_cost import observed_for
from dw.server.routes.jobs import _inherited_cost_warnings
from dw.vram_inheritance import inherited_differences
from dw.workflow import workflow_from_definition

from .test_observed_cost import MINUTE, run

PIPELINE = {
    "configuration": {"component_type": "ModularPipeline", "offload": "model"},
    "from_pretrained_arguments": {"model_name": "acme/video", "workflow": "t2v"},
}


def definition(name, pipeline=None, num_frames=124, **extra):
    return {
        "id": name,
        "variables": {"num_frames": num_frames},
        "cost_drivers": ["num_frames"],
        "steps": [
            {
                "id": "gen",
                "pipeline": pipeline or PIPELINE,
                "run": {"num_frames": "variable:num_frames"},
            }
        ],
        **extra,
    }


def state_with(history):
    """A server state whose observed costs aggregate `history` (name ->
    rows) with the real observed_for."""

    def observed(name, template, arguments=None, *, workspace=None, card=None):
        rows = history.get(name)
        return (
            observed_for(template, rows, device="cpu", arguments=arguments)
            if rows
            else None
        )

    return SimpleNamespace(observed_costs=SimpleNamespace(observed=observed))


def inherited(inline, catalog, history, arguments=None):
    from dw.server import deps

    with patch.object(
        deps, "_catalog_listing", return_value=[(n, d, False) for n, d in catalog]
    ):
        return deps.inherited_observed(
            state_with(history), None, inline, arguments or {}
        )


class TestPricedByPipelineIdentity:
    def test_an_inline_copy_inherits_the_templates_runs(self):
        template = definition("template")
        block = inherited(
            definition("inline"),
            [("templates/t", template)],
            {"templates/t": [run(8 * MINUTE, {"num_frames": 124})]},
        )

        assert block["inherited_from"] == "templates/t"
        assert block["cold_minutes"] == 8.0
        assert block["differs"] == []

    def test_a_different_checkpoint_matches_nothing(self):
        other = {
            **PIPELINE,
            "from_pretrained_arguments": {"model_name": "acme/other"},
        }
        assert (
            inherited(
                definition("inline"),
                [("templates/t", definition("t", other))],
                {"templates/t": [run(8 * MINUTE, {"num_frames": 124})]},
            )
            is None
        )

    def test_the_callers_frame_count_picks_the_bucket(self):
        history = {
            "templates/t": [
                run(8 * MINUTE, {"num_frames": 124}),
                run(16 * MINUTE, {"num_frames": 248}),
            ]
        }
        block = inherited(
            definition("inline"),
            [("templates/t", definition("t"))],
            history,
            {"num_frames": 248},
        )
        assert block["cold_minutes"] == 16.0

    def test_the_template_with_the_most_runs_wins(self):
        history = {
            "templates/a": [run(5 * MINUTE, {"num_frames": 124})],
            "templates/b": [run(9 * MINUTE, {"num_frames": 124})] * 2,
        }
        block = inherited(
            definition("inline"),
            [("templates/a", definition("a")), ("templates/b", definition("b"))],
            history,
        )
        assert block["inherited_from"] == "templates/b"

    def test_a_workflow_with_two_pipelines_is_not_priced(self):
        inline = definition("inline")
        inline["steps"].append(
            {
                "id": "other",
                "pipeline": {
                    "from_pretrained_arguments": {"model_name": "acme/second"}
                },
            }
        )
        assert (
            inherited(
                inline,
                [("templates/t", definition("t"))],
                {"templates/t": [run(8 * MINUTE, {"num_frames": 124})]},
            )
            is None
        )


class TestDifferences:
    def test_offload_quantization_and_frames_are_named(self):
        quantized = {
            "configuration": {"component_type": "ModularPipeline"},
            "from_pretrained_arguments": {
                **PIPELINE["from_pretrained_arguments"],
                "quantization_config": {"quant_type": "nf4"},
            },
        }
        differs = inherited_differences(
            definition("inline", quantized), definition("template")
        )
        assert differs == ["offload", "quantization"]

    def test_a_frame_count_the_caller_changed(self):
        differs = inherited_differences(
            definition("inline"), definition("template"), {"num_frames": 248}
        )
        assert differs == ["frame count"]

    def test_a_reference_count_other_than_the_templates_is_named(self):
        def refs(n):
            return [{"from_file": f"r{i}"} for i in range(n)]

        def with_refs(value, **top):
            d = definition("inline")
            step = d["steps"][0]
            step["pipeline"] = {
                **step["pipeline"],
                "arguments": {"references": value},
            }
            d.update(top)
            return d

        template = with_refs(refs(2))
        assert inherited_differences(with_refs(refs(2)), template) == []
        assert inherited_differences(with_refs(refs(3)), template) == [
            "reference count"
        ]
        # per-entry references, read off the step's for_each list
        entries = [{"references": refs(2)}, {"references": refs(3)}]
        each = with_refs("item:references", variables={"shots": entries})
        each["steps"][0]["for_each"] = "variable:shots"
        assert "reference count" in inherited_differences(each, template)
        # the optional form with a default, and a null source, count as the
        # run counts them
        optional = [
            {"references": refs(2) + [{"from_file": None}]},
            {"label": "no references of its own"},
        ]
        each = with_refs(
            {"item": "references", "default": refs(2)},
            variables={"shots": optional, "num_frames": 124},
        )
        each["steps"][0]["for_each"] = "variable:shots"
        assert "reference count" not in inherited_differences(each, template)
        each["variables"]["shots"][1]["references"] = refs(3)
        assert "reference count" in inherited_differences(each, template)

    def test_a_for_each_whose_entries_match_the_templates_frames_is_not_named(self):
        def with_frames(value, **top):
            d = definition("inline")
            step = d["steps"][0]
            step["pipeline"] = {
                **step["pipeline"],
                "arguments": {"num_frames": value},
            }
            d.update(top)
            return d

        template = with_frames(124)
        shots = [{"num_frames": 124}, {"num_frames": 124}]
        each = with_frames(
            "item:num_frames", variables={"shots": shots, "num_frames": 124}
        )
        each["steps"][0]["for_each"] = "variable:shots"
        assert inherited_differences(each, template) == []
        each["variables"]["shots"][1]["num_frames"] = 209
        assert inherited_differences(each, template) == ["frame count"]


class TestThePlanReportsIt:
    def test_the_estimate_is_inherited_and_the_validate_answer_warns(self, tmp_path):
        block = inherited(
            definition("inline", num_frames=248),
            [("templates/t", definition("t"))],
            {"templates/t": [run(8 * MINUTE, {"num_frames": 124})] * 3},
            {"num_frames": 124},
        )
        block["differs"] = ["offload"]
        candidate = workflow_from_definition(
            definition("inline"), str(tmp_path), str(tmp_path), str(tmp_path)
        )
        with patch.object(dw.plan, "scan_models", return_value={"repos": []}):
            plan = build_plan(
                candidate,
                {"num_frames": 124},
                device="cpu",
                lookup_sizes=False,
                observed=lambda arguments: block,
            )

        estimate = plan["estimate"]
        assert estimate["basis"] == "inherited"
        assert estimate["minutes"] == 8.0
        assert estimate["inherited_from"] == "templates/t"
        warnings = _inherited_cost_warnings(estimate)
        assert len(warnings) == 1
        assert "templates/t" in warnings[0] and "offload" in warnings[0]

    def test_an_observed_estimate_does_not_warn(self):
        assert _inherited_cost_warnings({"basis": "observed"}) == []


class TestScaledWhenItScales:
    """#821/#851: the template's minutes are for one template-sized run, so a
    bigger workflow is re-priced linearly and says why."""

    HISTORY = {"templates/t": [run(8 * MINUTE, {"num_frames": 124})]}
    CATALOG = [("templates/t", definition("t"))]

    def test_several_steps_scale_by_step_count(self):
        inline = definition("inline")
        inline["steps"] = [{**inline["steps"][0], "id": f"shot{n}"} for n in range(3)]
        block = inherited(inline, self.CATALOG, self.HISTORY)
        assert block["cold_minutes"] == 24.0
        assert block["inherited_from"] == "templates/t"
        assert "3 steps, template measured 1" in block["scaled"]

    def test_a_longer_frame_count_scales_by_frames(self):
        inline = definition("inline")
        inline["variables"] = {}
        inline["steps"][0]["run"] = {"num_frames": 248}
        block = inherited(inline, self.CATALOG, self.HISTORY)
        assert block["cold_minutes"] == 16.0
        assert "frames 248 vs 124" in block["scaled"]
        assert "frame count" in block["differs"]

    def test_steps_and_frames_multiply(self):
        inline = definition("inline")
        inline["variables"] = {}
        inline["steps"] = [
            {**inline["steps"][0], "id": f"shot{n}", "run": {"num_frames": 62}}
            for n in range(4)
        ]
        block = inherited(inline, self.CATALOG, self.HISTORY)
        assert block["cold_minutes"] == 16.0  # 4 steps x half the frames
        assert "4 steps" in block["scaled"] and "62 vs 124" in block["scaled"]

    def test_a_template_sized_workflow_is_not_scaled(self):
        block = inherited(definition("inline"), self.CATALOG, self.HISTORY)
        assert block["cold_minutes"] == 8.0
        assert "scaled" not in block

    def test_the_plan_reports_the_reason_and_low_confidence(self, tmp_path):
        inline = definition("inline")
        inline["steps"] = [{**inline["steps"][0], "id": f"shot{n}"} for n in range(3)]
        block = inherited(
            inline, self.CATALOG, {"templates/t": self.HISTORY["templates/t"] * 3}
        )
        candidate = workflow_from_definition(
            inline, str(tmp_path), str(tmp_path), str(tmp_path)
        )
        with patch.object(dw.plan, "scan_models", return_value={"repos": []}):
            plan = build_plan(
                candidate,
                {},
                device="cpu",
                lookup_sizes=False,
                observed=lambda arguments: block,
            )
        estimate = plan["estimate"]
        assert estimate["basis"] == "inherited"
        assert estimate["minutes"] == 24.0
        assert estimate["low_confidence"] is True
        assert "3 steps, template measured 1" in estimate["scaled"]
        assert "3 steps" in _inherited_cost_warnings(estimate)[0]

    def test_a_shared_driver_name_still_prices_its_bucket(self):
        block = inherited(
            definition("inline"),
            [("templates/t", definition("t"))],
            self.HISTORY,
            {"num_frames": 124},
        )
        assert block["cold_minutes"] == 8.0


def shot_list(count, num_frames=124):
    return [{"prompt": f"p{n}", "num_frames": num_frames} for n in range(count)]


def shots_definition(name, shots):
    return {
        "id": name,
        "variables": {"shots": shots},
        "cost_drivers": ["shots"],
        "steps": [
            {
                "name": "shot",
                "for_each": "variable:shots",
                "pipeline": PIPELINE,
                "run": {"num_frames": "item:num_frames", "prompt": "item:prompt"},
            }
        ],
    }


class TestAListLongerThanTheFallbackBucket:
    """A for_each step is one pipeline step per entry: an inline copy with
    more entries than the default bucket the template ran is priced from
    that bucket only once it is scaled by the expanded member count (#851
    regression - 6 x 124 frames was quoted as 3 x 124's minutes)."""

    CATALOG = [("templates/t", shots_definition("t", shot_list(3)))]
    HISTORY = {"templates/t": [run(30 * MINUTE, {"shots": shot_list(3)})]}

    def test_the_template_sized_list_is_not_scaled(self):
        block = inherited(
            shots_definition("inline", shot_list(3)), self.CATALOG, self.HISTORY
        )
        assert block["cold_minutes"] == 30.0
        assert "scaled" not in block

    def test_an_inline_six_shot_copy_scales_by_members(self):
        block = inherited(
            shots_definition("inline", shot_list(6)), self.CATALOG, self.HISTORY
        )
        assert block["cold_minutes"] == 60.0
        assert "6 steps, template measured 3" in block["scaled"]

    def test_a_six_shot_argument_scales_by_members(self):
        block = inherited(
            shots_definition("inline", shot_list(3)),
            self.CATALOG,
            self.HISTORY,
            {"shots": shot_list(6)},
        )
        assert block["cold_minutes"] == 60.0
        assert "6 steps, template measured 3" in block["scaled"]

    def test_shorter_shots_scale_by_members_and_frames(self):
        block = inherited(
            shots_definition("inline", shot_list(6, num_frames=62)),
            self.CATALOG,
            self.HISTORY,
        )
        assert block["cold_minutes"] == 30.0
        assert "6 steps" in block["scaled"] and "62 vs 124" in block["scaled"]

    def test_the_plan_quotes_it_low_confidence(self, tmp_path):
        inline = shots_definition("inline", shot_list(6))
        block = inherited(inline, self.CATALOG, self.HISTORY)
        candidate = workflow_from_definition(
            inline, str(tmp_path), str(tmp_path), str(tmp_path)
        )
        with patch.object(dw.plan, "scan_models", return_value={"repos": []}):
            plan = build_plan(
                candidate,
                {},
                device="cpu",
                lookup_sizes=False,
                observed=lambda arguments: block,
            )
        estimate = plan["estimate"]
        assert estimate["basis"] == "inherited"
        assert estimate["minutes"] == 60.0
        assert estimate["low_confidence"] is True


class TestVariableFrameCountAndWithheldReason:
    """#851 bounce: a frame count written as `variable:num_frames` that the
    template never ran at still prices from the template's default bucket, and
    a figure still withheld names the template and why."""

    HISTORY = {"templates/t": [run(8 * MINUTE, {"num_frames": 124})]}
    CATALOG = [("templates/t", definition("t"))]

    def test_a_variable_frame_count_scales_from_the_default_bucket(self):
        inline = definition("inline", num_frames=248)
        block = inherited(inline, self.CATALOG, self.HISTORY)
        assert block["cold_minutes"] == 16.0
        assert "frames 248 vs 124" in block["scaled"]

    def test_variable_frames_and_steps_multiply(self):
        inline = definition("inline", num_frames=248)
        inline["steps"] = [{**inline["steps"][0], "id": f"s{n}"} for n in range(2)]
        block = inherited(inline, self.CATALOG, self.HISTORY)
        assert block["cold_minutes"] == 32.0

    def test_a_withheld_figure_names_template_and_reason(self):
        inline = definition("inline")
        inline["variables"] = {}
        inline["steps"][0]["run"] = {"num_frames": "variable:missing"}
        block = inherited(inline, self.CATALOG, self.HISTORY)
        assert block["inherited_from"] == "templates/t"
        assert "cannot be compared" in block["withheld"]
        assert "cold_minutes" not in block

    def test_the_plan_carries_the_withheld_reason(self, tmp_path):
        inline = definition("inline")
        candidate = workflow_from_definition(
            inline, str(tmp_path), str(tmp_path), str(tmp_path)
        )
        withheld = {"inherited_from": "templates/t", "withheld": "not priced: why"}
        with patch.object(dw.plan, "scan_models", return_value={"repos": []}):
            plan = build_plan(
                candidate,
                {},
                device="cpu",
                lookup_sizes=False,
                observed=lambda arguments: withheld,
            )
        estimate = plan["estimate"]
        assert estimate["basis"] == "unknown" and estimate["minutes"] is None
        assert estimate["inherited_from"] == "templates/t"
        assert estimate["withheld"] == "not priced: why"
        assert "withheld" in _inherited_cost_warnings(estimate)[0]


def sized(name, steps=9, width=960, height=544, **extra):
    """`definition` whose step also sets inference steps and a canvas."""
    workflow = definition(name, **extra)
    workflow["variables"].update(
        {"num_inference_steps": steps, "width": width, "height": height}
    )
    workflow["cost_drivers"] += ["num_inference_steps", "width", "height"]
    workflow["steps"][0]["run"].update(
        {
            "num_inference_steps": "variable:num_inference_steps",
            "width": "variable:width",
            "height": "variable:height",
        }
    )
    return workflow


class TestScaledByStepsAndCanvas:
    """#866 stage C: the inherited figure scales by the inference-step and
    canvas ratios too, as the catalog figure does (`scale_curated`)."""

    HISTORY = {"templates/t": [run(8 * MINUTE, {})]}
    CATALOG = [("templates/t", sized("t"))]

    def test_thirty_steps_scale_by_the_step_ratio(self):
        block = inherited(sized("inline", steps=30), self.CATALOG, self.HISTORY)
        assert block["cold_minutes"] == 8.0 * 30 / 9
        assert "steps 30 vs 9" in block["scaled"]

    def test_a_larger_canvas_scales_by_the_pixel_ratio(self):
        block = inherited(
            sized("inline", width=1344, height=768), self.CATALOG, self.HISTORY
        )
        assert block["cold_minutes"] == 8.0 * (1344 * 768) / (960 * 544)
        assert "canvas 1344x768 vs 960x544" in block["scaled"]

    def test_the_callers_arguments_scale_it_and_the_axes_multiply(self):
        block = inherited(
            sized("inline"),
            self.CATALOG,
            self.HISTORY,
            {"num_inference_steps": 30, "width": 1344, "height": 768},
        )
        assert block["cold_minutes"] == pytest.approx(
            8.0 * 30 / 9 * (1344 * 768) / (960 * 544)
        )
        assert "steps" in block["scaled"] and "canvas" in block["scaled"]

    def test_a_bucket_at_those_values_is_not_scaled_again(self):
        history = {
            "templates/t": [
                run(8 * MINUTE, {}),
                run(20 * MINUTE, {"num_inference_steps": 30}),
            ]
        }
        block = inherited(sized("inline", steps=30), self.CATALOG, history)
        assert block["cold_minutes"] == 20.0
        assert "scaled" not in block

    def test_a_canvas_that_cannot_be_compared_is_withheld(self):
        inline = sized("inline")
        inline["steps"][0]["run"]["width"] = "variable:missing"
        block = inherited(inline, self.CATALOG, self.HISTORY)
        assert "canvas differs and cannot be compared" in block["withheld"]

    def test_the_plan_quotes_it_inherited_and_scaled(self, tmp_path):
        inline = sized("inline", steps=30, width=1344, height=768)
        block = inherited(inline, self.CATALOG, {"templates/t": [run(8 * MINUTE)] * 3})
        candidate = workflow_from_definition(
            inline, str(tmp_path), str(tmp_path), str(tmp_path)
        )
        with patch.object(dw.plan, "scan_models", return_value={"repos": []}):
            estimate = build_plan(
                candidate,
                {},
                device="cpu",
                lookup_sizes=False,
                observed=lambda arguments: block,
            )["estimate"]
        assert estimate["basis"] == "inherited"
        assert estimate["minutes"] == round(8.0 * 30 / 9 * 1.9765, 1)
        assert "steps 30 vs 9" in estimate["scaled"]
        assert "canvas 1344x768 vs 960x544" in estimate["scaled"]


class TestAttentionCacheAndLoraDiffer:
    """#866 stage C: the attention backend, cache and LoRA a step sets are
    named in `differs`, and a figure measured at others is low confidence."""

    def with_settings(self, name, attention=None, cache=None, lora="acme/turbo"):
        workflow = definition(name)
        pipeline = {
            **PIPELINE,
            "configuration": {
                **PIPELINE["configuration"],
                "components": {"transformer": {"attention_backend": attention}},
                "cache": cache,
            },
            "loras": [{"model_name": lora}] if lora else [],
        }
        workflow["steps"][0]["pipeline"] = pipeline
        return workflow

    def test_each_is_named(self):
        template = self.with_settings("t")
        assert inherited_differences(self.with_settings("i"), template) == []
        assert inherited_differences(
            self.with_settings("i", attention="native"), template
        ) == ["attention backend"]
        assert inherited_differences(
            self.with_settings("i", cache={"type": "fbc"}), template
        ) == ["cache"]
        assert inherited_differences(self.with_settings("i", lora=None), template) == [
            "LoRA"
        ]

    def test_the_plan_flags_it_low_confidence(self, tmp_path):
        inline = self.with_settings("inline", attention="native")
        block = inherited(
            inline,
            [("templates/t", self.with_settings("t"))],
            {"templates/t": [run(8 * MINUTE, {"num_frames": 124})] * 3},
        )
        candidate = workflow_from_definition(
            inline, str(tmp_path), str(tmp_path), str(tmp_path)
        )
        with patch.object(dw.plan, "scan_models", return_value={"repos": []}):
            estimate = build_plan(
                candidate,
                {},
                device="cpu",
                lookup_sizes=False,
                observed=lambda arguments: block,
            )["estimate"]
        assert estimate["basis"] == "inherited"
        assert estimate["minutes"] == 8.0
        assert estimate["differs"] == ["attention backend"]
        assert estimate["low_confidence"] is True


class TestTheTemplateAndAnInlineCopyAgree:
    """#866 stage C: the catalog path (`scale_curated`) and the inherited path
    quote the same request alike - an inline copy of the shipped H3
    reference-to-video and the template itself, at the same arguments."""

    ARGUMENTS = {"num_inference_steps": 30, "width": 1344, "height": 768}

    def test_the_same_minutes_within_rounding(self, tmp_path):
        import copy
        import json
        from pathlib import Path

        path = (
            Path(__file__).resolve().parents[1]
            / "workflows/templates/minimax/reference-to-video.json"
        )
        template = json.loads(path.read_text())
        curated = template["cost"][0]["minutes"]
        inline = copy.deepcopy(template)
        del inline["cost"], inline["cost_drivers"]

        def planned(definition, observed):
            candidate = workflow_from_definition(
                definition, str(tmp_path), str(tmp_path), str(tmp_path)
            )
            with patch.object(dw.plan, "scan_models", return_value={"repos": []}):
                return build_plan(
                    candidate,
                    self.ARGUMENTS,
                    device="cuda",
                    lookup_sizes=False,
                    observed=observed,
                )["estimate"]

        # The template has no run at these arguments: its curated figure,
        # scaled
        catalog = planned(template, lambda arguments: None)
        assert catalog["basis"] == "derived"

        # Its runs at the defaults took the curated minutes; the inline copy
        # inherits them, scaled from the default bucket
        block = inherited(
            inline,
            [("templates/minimax/reference-to-video", template)],
            {"templates/minimax/reference-to-video": [run(curated * MINUTE)] * 3},
            self.ARGUMENTS,
        )
        block["device"] = None
        copied = planned(inline, lambda arguments: block)
        assert copied["basis"] == "inherited"
        assert "steps 30 vs 9" in copied["scaled"]
        assert "canvas 1344x768 vs 960x544" in copied["scaled"]
        assert abs(copied["minutes"] - catalog["minutes"]) <= 0.1


class TestTheShippedTemplatesLoraIsCompared:
    """#866 stage C (bounce): H3's `loras` entry carries its repo as a
    `variable:` reference inside the list, so the comparison resolves
    references at any depth - dropping or swapping the turbo LoRA is named."""

    def template(self):
        import json
        from pathlib import Path

        path = (
            Path(__file__).resolve().parents[1]
            / "workflows/templates/minimax/reference-to-video.json"
        )
        return json.loads(path.read_text())

    def test_the_defaults_differ_in_nothing(self):
        template = self.template()
        assert inherited_differences(template, template) == []

    def test_dropping_the_lora_is_named(self):
        template = self.template()
        arguments = {"lora_model_name": None, "lora_weight_name": None}
        assert inherited_differences(template, template, arguments) == ["LoRA"]

    def test_swapping_the_lora_is_named(self):
        template = self.template()
        arguments = {"lora_model_name": "lightx2v/Some-Other-Lora"}
        assert inherited_differences(template, template, arguments) == ["LoRA"]

    def test_a_copy_without_its_null_variables_differs_in_nothing(self):
        """#866 stage C (C-F485 bounce): an inline copy that leaves out the
        variables whose default is null (`lora_alpha`) compares as if they
        resolved to null, the way the template's own declarations do."""
        template = self.template()
        copy = dict(template)
        copy["variables"] = {
            name: value
            for name, value in template["variables"].items()
            if value is not None
        }
        assert "lora_alpha" not in copy["variables"]
        assert inherited_differences(copy, template) == []
        arguments = {"lora_model_name": "lightx2v/Some-Other-Lora"}
        assert inherited_differences(copy, template, arguments) == ["LoRA"]

    def test_a_client_round_trip_differs_in_nothing(self, tmp_path):
        """#866 stage C (second C-F485 bounce): an MCP client's JSON round
        trip respells the template's `lora_scale` 1.0 as 1. The copy, sent
        the way the tester's was (seed set, prompt replaced, catalog keys
        dropped), is quoted from the template's runs with nothing named and
        no low confidence: a number compares by value, not spelling."""
        from dw.variables import same_number

        template = self.template()
        copy = same_number(template)
        assert copy["variables"]["lora_scale"] == 1
        assert isinstance(copy["variables"]["lora_scale"], int)
        for key in ("cost", "cost_drivers", "summary", "vram_estimate"):
            copy.pop(key, None)
        copy.pop("variable_constraints", None)
        copy["variables"]["seed"] = 42
        copy["variables"]["prompt"] = "a lighthouse at dusk"

        assert inherited_differences(copy, template) == []
        block = inherited(
            copy,
            [("templates/minimax/reference-to-video", template)],
            {"templates/minimax/reference-to-video": [run(8 * MINUTE)] * 3},
        )
        assert block["differs"] == []
        candidate = workflow_from_definition(
            copy, str(tmp_path), str(tmp_path), str(tmp_path)
        )
        with patch.object(dw.plan, "scan_models", return_value={"repos": []}):
            estimate = build_plan(
                candidate,
                {},
                device="cpu",
                lookup_sizes=False,
                observed=lambda arguments: {**block, "device": None},
            )["estimate"]
        assert estimate["basis"] == "inherited"
        assert not estimate.get("differs")
        assert not estimate.get("low_confidence")
        # A scale that really differs is still named
        assert inherited_differences(copy, template, {"lora_scale": 0.5}) == ["LoRA"]


def test_the_warning_agrees_in_number():
    def warned(differs):
        return _inherited_cost_warnings(
            {"basis": "inherited", "inherited_from": "templates/t", "differs": differs}
        )[0]

    assert "its LoRA differs from that template" in warned(["LoRA"])
    assert "its cache, LoRA differ from that template" in warned(["cache", "LoRA"])


def test_the_comparison_substitutes_as_the_run_does():
    """#866 stage C (arch review): `_found` reads a step through
    `replace_variables`, so a `variable:` resolving to null drops its key as
    the run does, rather than comparing a value the run never sees."""
    from dw.vram_inheritance import _found

    steps = [{"pipeline": {"loras": [{"model_name": "variable:lora"}]}}]
    assert _found(steps, "loras", {"lora": None}) == {"[{}]"}
    assert _found(steps, "loras", {"lora": "a/b"}) == {'[{"model_name": "a/b"}]'}
    # An undeclared name stays as written; the rest of the step still resolves
    mixed = [{"run": {"width": "variable:missing", "num_frames": "variable:n"}}]
    assert _found(mixed, "num_frames", {"n": 124}) == {"124"}
    assert _found(mixed, "width", {"n": 124}) == {'"variable:missing"'}
