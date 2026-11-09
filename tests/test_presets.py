"""`presets`: one variable selects a bundle of other variables' values.

A template exposes one quality dial (`tier`) that moves canvas, steps and so
on, while every variable stays individually settable: a caller's explicit
argument always wins over the bundle.
"""

import copy

import pytest

from dw.plan import build_plan
from dw.presets import apply_presets, preset_errors
from dw.security import SecurityError
from dw.validation import workflow_errors
from dw.variable_constraints import constraint_errors, constraint_warnings
from dw.workflow import workflow_from_definition

PRESETS = {
    "tier": {
        "draft": {"width": 512, "num_inference_steps": 4},
        "standard": {"width": 768, "num_inference_steps": 12},
    }
}


def definition(presets=PRESETS, **overrides):
    variables = {
        "tier": "standard",
        "width": 1024,
        "num_inference_steps": 30,
        "prompt": "a cat",
    }
    variables.update(overrides)
    return {"id": "dial", "variables": variables, "presets": copy.deepcopy(presets)}


def test_the_bundle_sets_the_variables_the_caller_did_not_pass():
    d = definition()
    variables = copy.deepcopy(d["variables"])

    apply_presets(d, {"tier": "draft"}, variables)

    assert variables["width"] == 512
    assert variables["num_inference_steps"] == 4
    assert variables["prompt"] == "a cat"


def test_a_caller_value_beats_the_bundle():
    d = definition()
    arguments = {"tier": "draft", "num_inference_steps": 30}
    variables = copy.deepcopy(d["variables"])

    apply_presets(d, arguments, variables)
    from dw.variables import set_variables

    set_variables(arguments, variables)

    assert variables["num_inference_steps"] == 30
    assert variables["width"] == 512


def test_the_default_bundle_applies_with_no_arguments():
    d = definition()
    variables = copy.deepcopy(d["variables"])

    apply_presets(d, {}, variables)

    assert variables["width"] == 768
    assert variables["num_inference_steps"] == 12


def test_an_unknown_selector_value_is_refused_naming_the_values():
    (problem,) = preset_errors(definition(), {"tier": "ultra"})

    assert problem["path"] == "arguments.tier"
    assert "draft" in problem["message"] and "standard" in problem["message"]


def test_a_bundle_naming_an_undeclared_variable_is_a_validation_error():
    d = definition({"tier": {"standard": {"nope": 1}}})

    (problem,) = preset_errors(d, None)

    assert problem["path"] == "presets.tier.standard.nope"
    assert "declared" in problem["message"]


def test_a_selector_whose_default_is_not_a_bundle_is_an_error():
    problems = preset_errors(definition(tier="gold"), None)

    assert any(p["path"] == "variables.tier" for p in problems)


def test_a_workflow_without_presets_is_unchanged():
    d = definition()
    del d["presets"]
    variables = copy.deepcopy(d["variables"])

    apply_presets(d, {"tier": "draft"}, variables)

    assert variables == d["variables"]
    assert preset_errors(d, {"tier": "draft"}) == []


def test_a_bundle_value_is_validated_like_a_caller_value():
    d = definition({"tier": {"standard": {"prompt": "bad\x00value"}}})
    variables = copy.deepcopy(d["variables"])

    with pytest.raises(SecurityError):
        apply_presets(d, {}, variables)
    (problem,) = preset_errors(d, None)
    assert problem["path"] == "presets.tier.standard.prompt"


def test_the_workflow_folds_the_bundle_for_a_run_and_for_validation(tmp_path):
    d = definition()
    d["steps"] = [{"name": "a", "task": {"command": "no_op", "arguments": {}}}]
    workflow = workflow_from_definition(d, str(tmp_path))

    variables = workflow.folded_variables({"tier": "draft"})
    assert variables["width"] == 512

    errors = workflow.validation_errors({"tier": "ultra"})
    assert any(e["path"] == "arguments.tier" for e in errors), errors


# --- every reader sees the bundle (validate checks what the run will use) ---


def resizing(presets=PRESETS, **overrides):
    """The definition with a step that reads every variable."""
    d = definition(presets, **overrides)
    d["steps"] = [
        {
            "name": "a",
            "task": {
                "command": "no_op",
                "arguments": {
                    "width": "variable:width",
                    "steps": "variable:num_inference_steps",
                },
            },
        }
    ]
    return d


@pytest.mark.parametrize("arguments", [None, {}])
def test_the_default_bundle_folds_with_no_arguments(tmp_path, arguments):
    workflow = workflow_from_definition(resizing(), str(tmp_path))

    assert workflow.folded_variables(arguments)["width"] == 768
    expanded = workflow.expanded_definition(arguments)
    assert expanded["steps"][0]["task"]["arguments"] == {"width": 768, "steps": 12}


def test_a_bad_argument_still_folds_the_default_bundle(tmp_path):
    """A bad argument leaves every argument unfolded - the declared
    defaults, of which the default bundle is one."""
    workflow = workflow_from_definition(resizing(), str(tmp_path))

    folded = workflow.folded_variables({"tier": "draft", "width": "wide"})

    assert folded["width"] == 768
    assert folded["num_inference_steps"] == 12


def test_no_arguments_and_the_default_selector_have_one_fingerprint(tmp_path):
    def plan(arguments):
        d = resizing()
        d["steps"][0]["result"] = {"content_type": "image/png"}
        candidate = workflow_from_definition(d, str(tmp_path))
        return build_plan(
            candidate,
            arguments,
            device="cuda",
            prompt_dir=str(tmp_path),
            lookup_sizes=False,
        )["fingerprint"]

    assert plan({}) == plan({"tier": "standard"})
    assert plan({}) != plan({"tier": "draft"})


def framed(draft_frames=100, standard_frames=97, snap=False):
    rule = {"modulus": 8, "remainder": 1, "reason": "VAE grid"}
    if snap:
        rule["snap"] = "up"
    return {
        "id": "framed",
        "variables": {"tier": "standard", "num_frames": 121},
        "variable_constraints": {"num_frames": rule},
        "presets": {
            "tier": {
                "draft": {"num_frames": draft_frames},
                "standard": {"num_frames": standard_frames},
            }
        },
    }


@pytest.mark.parametrize("arguments", [{}, {"tier": "draft"}, {"tier": "standard"}])
def test_a_bundle_value_breaking_a_rule_is_an_error_at_the_bundle(arguments):
    """Every bundle is checked, the chosen one or not, and only once."""
    errors = constraint_errors(framed(), arguments, supplied=set(arguments))

    assert [e["path"] for e in errors] == ["presets.tier.draft.num_frames"]
    assert "8 * n + 1" in errors[0]["message"]


def test_the_default_bundle_breaking_a_rule_is_an_error_with_no_arguments():
    errors = constraint_errors(framed(draft_frames=97, standard_frames=100), {})

    assert [e["path"] for e in errors] == ["presets.tier.standard.num_frames"]


def test_a_caller_value_over_a_bundle_is_reported_where_the_caller_wrote_it():
    arguments = {"tier": "draft", "num_frames": 100}
    errors = constraint_errors(framed(draft_frames=97), arguments, set(arguments))

    assert [e["path"] for e in errors] == ["arguments.num_frames"]


def test_a_snapped_bundle_value_warns_as_a_caller_value_would():
    d = framed(snap=True)

    assert constraint_errors(d, {"tier": "draft"}, {"tier"}) == []
    (notice,) = constraint_warnings(d, {"tier": "draft"})
    assert "100" in notice and "105" in notice
    assert constraint_warnings(d, {}) == []


def test_validation_reports_a_bundle_value_breaking_a_rule(tmp_path):
    d = framed()
    d["steps"] = [
        {
            "name": "a",
            "task": {"command": "no_op", "arguments": {"n": "variable:num_frames"}},
        }
    ]
    for arguments in ({}, {"tier": "draft"}):
        workflow = workflow_from_definition(copy.deepcopy(d), str(tmp_path))
        paths = [e["path"] for e in workflow_errors(workflow, arguments)]
        assert "presets.tier.draft.num_frames" in paths, (arguments, paths)


@pytest.mark.parametrize("arguments", [{}, {"tier": "draft"}])
def test_a_bundle_value_that_will_not_coerce_is_a_finding_not_a_raise(
    tmp_path, arguments
):
    d = resizing({"tier": {"draft": {"width": "abc"}, "standard": {"width": 768}}})
    workflow = workflow_from_definition(d, str(tmp_path))

    errors = workflow_errors(workflow, arguments)

    (problem,) = [e for e in errors if e["path"].startswith("presets.")]
    assert problem["path"] == "presets.tier.draft.width"
    assert "int" in problem["message"]


def test_a_selector_whose_default_is_not_a_string_is_refused():
    d = definition({"1": {}}, tier=1)
    d["presets"] = {"tier": {"1": {"width": 512}}}

    (problem,) = preset_errors(d, None)

    assert problem["path"] == "variables.tier"
    assert "string" in problem["message"]


def test_effective_variables_carries_the_bundle_under_the_arguments():
    from dw.presets import effective_variables

    d = definition()

    assert effective_variables(d)["width"] == 768
    values = effective_variables(d, {"tier": "draft", "width": 640})
    assert values["width"] == 640
    assert values["num_inference_steps"] == 4
    # the declared block itself is never touched
    assert d["variables"]["width"] == 1024


def test_the_memory_key_sees_the_bundle():
    from dw.server.host_ram_need import memory_key

    d = definition({"tier": {"draft": {"offload": "sequential"}, "standard": {}}})
    d["variables"]["offload"] = "none"
    d["steps"] = [
        {
            "name": "gen",
            "pipeline": {
                "configuration": {
                    "component_type": "{Fake}",
                    "offload": "variable:offload",
                },
                "from_pretrained_arguments": {"model_name": "m"},
                "arguments": {"prompt": "variable:prompt"},
            },
        }
    ]

    assert memory_key(d, {}) == memory_key(d, {"tier": "standard"})
    assert memory_key(d, {"tier": "draft"}) != memory_key(d, {})
    assert memory_key(d, {"tier": "draft"}) == memory_key(d, {"offload": "sequential"})


def test_argument_warnings_resolve_a_bundle_value():
    from dw.argument_warnings import workflow_argument_warnings

    d = {
        "id": "cut",
        "variables": {"tier": "smooth", "seam_fade_ms": 0, "audio_bleed_ms": 40},
        "presets": {"tier": {"smooth": {"seam_fade_ms": 120}, "hard": {}}},
        "steps": [
            {
                "name": "cut",
                "task": {
                    "command": "concat_videos",
                    "arguments": {
                        "seam_fade_ms": "variable:seam_fade_ms",
                        "audio_bleed_ms": "variable:audio_bleed_ms",
                    },
                },
            }
        ],
    }

    assert any("seam_fade_ms" in w for w in workflow_argument_warnings(d, {}))
    assert not any(
        "seam_fade_ms" in w for w in workflow_argument_warnings(d, {"tier": "hard"})
    )


def test_the_estimate_prices_a_default_run_at_the_default_bundle(tmp_path):
    """The curated cost was measured at the defaults, the default bundle
    included - a run with no arguments is not scaled away from it."""
    d = framed(draft_frames=49)
    d["cost"] = [{"device": "cuda", "name": "card", "vram_gb": 8, "minutes": 10}]
    d["cost_drivers"] = ["num_frames"]
    d["steps"] = [
        {
            "name": "a",
            "task": {"command": "no_op", "arguments": {"n": "variable:num_frames"}},
            "result": {"content_type": "image/png"},
        }
    ]

    def estimate(arguments):
        candidate = workflow_from_definition(copy.deepcopy(d), str(tmp_path))
        return build_plan(
            candidate,
            arguments,
            device="cuda",
            prompt_dir=str(tmp_path),
            lookup_sizes=False,
        )["estimate"]

    assert estimate({}) == estimate({"tier": "standard"})
    assert estimate({})["minutes"] == 10


@pytest.mark.parametrize("chosen", [["draft"], {"tier": "draft"}])
def test_an_unhashable_selector_value_is_a_finding_not_a_crash(chosen):
    (problem,) = preset_errors(definition(), {"tier": chosen})
    assert problem["path"] == "arguments.tier"


def test_a_bundle_naming_an_undeclared_variable_logs_no_error_when_read(caplog):
    d = definition(presets={"tier": {"standard": {"nope": 1, "width": 640}}})
    variables = copy.deepcopy(d["variables"])
    with caplog.at_level("ERROR", logger="dw"):
        apply_presets(d, {}, variables, strict=False)
    assert variables["width"] == 640
    assert not [r for r in caplog.records if r.levelname == "ERROR"]
