"""`presets`: one variable selects a bundle of other variables' values.

A template exposes one quality dial (`tier`) that moves canvas, steps and so
on, while every variable stays individually settable: a caller's explicit
argument always wins over the bundle.
"""

import copy

from dw.presets import apply_presets, preset_errors
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
    import pytest

    d = definition({"tier": {"standard": {"prompt": "bad\x00value"}}})
    variables = copy.deepcopy(d["variables"])

    with pytest.raises(Exception):
        apply_presets(d, {}, variables)


def test_the_workflow_folds_the_bundle_for_a_run_and_for_validation(tmp_path):
    d = definition()
    d["steps"] = [{"name": "a", "task": {"command": "no_op", "arguments": {}}}]
    workflow = workflow_from_definition(d, str(tmp_path))

    variables = workflow.folded_variables({"tier": "draft"})
    assert variables["width"] == 512

    errors = workflow.validation_errors({"tier": "ultra"})
    assert any(e["path"] == "arguments.tier" for e in errors), errors
