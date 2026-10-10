"""#797: an inline workflow is priced by pipeline identity.

Observed cost history is keyed by catalog name, so an inline copy of a
template had none. `inherited_observed` finds the catalog template with the
same pipeline identity (the key `dw/vram_inheritance.py` matches on) and
quotes its runs under `basis: "inherited"`, warning where the inline
workflow's offload, quantization or frame count differ.
"""

from types import SimpleNamespace
from unittest.mock import patch

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
        with patch("dw.plan.scan_models", return_value={"repos": []}):
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
