"""Declared bounds on a variable's value, refused before a run rather than
two minutes into one.

`validate_workflow(num_frames=61)` on an H3 template answered `valid: true`,
then the run loaded the weights and the turbo LoRA and failed 138.7 s later
on a check against two integers (#96). Every term in that check is a property
of the model, so the rule is declared on the workflow and checked statically,
at run time, and reported in the catalog.

The last class here is the sweep Don's approval asked for: every constraint
in the catalog, checked against the diffusers symbol it derives from, rather
than a hand-written assertion per template that would fall behind the
templates.
"""

import glob
import json
import os

import pytest

from dw.variable_constraints import (
    aligned,
    apply_constraints,
    constraint_errors,
    constraint_reference_errors,
    constraint_warnings,
    effective,
    refusal,
    resolve_constraint_references,
    snap_block,
    snap_notice,
    snapped,
    violations,
)
from dw.workflow import workflow_from_definition
from tests.test_examples import REPO_ROOT

H3 = {
    "modulus": 17,
    "remainder": 5,
    "min_frames": 124,
    "max_frames": 345,
    "snap": "up",
    "reason": "the video VAE encodes 17 * n + 5 frames",
}
# LTX-2.5 declares the grid with no `snap` - the pipeline floors an off-grid
# count, so rounding up here would be a second silent change
LTX = {"modulus": 8, "remainder": 1, "min_frames": 9}


class TestTheGrid:
    def test_a_value_on_the_grid_is_left_alone(self):
        for value in (124, 141, 345):
            assert aligned(value, H3) == value
            assert snapped(value, H3) is None
            assert violations(value, H3) == []

    def test_a_value_off_the_grid_rounds_up(self):
        assert aligned(130, H3) == 141
        assert snapped(130, H3) == 141
        assert violations(130, H3) == []

    def test_the_range_holds_for_the_rounded_value(self):
        """What the pipeline does: `align_num_frames` snaps first and the
        duration bound then holds for the aligned count. 61 becomes 73 and
        is refused; 108 becomes 124 and is accepted."""
        assert snapped(61, H3) == 73
        assert violations(61, H3) == ["must be at least 124"]
        assert snapped(346, H3) == 362
        assert violations(346, H3) == ["must be at most 345"]
        assert snapped(108, H3) == 124
        assert violations(108, H3) == []

    def test_without_snap_an_off_grid_value_is_refused_not_rounded(self):
        assert snapped(130, LTX) is None
        assert violations(130, LTX) == ["must be 8 * n + 1"]
        assert violations(121, LTX) == []

    @pytest.mark.parametrize("value", [None, "variable:n", "abc", [124], True])
    def test_what_this_pass_cannot_answer_about_is_not_a_violation(self, value):
        assert violations(value, H3) == []
        assert effective(value, H3) is None

    def test_a_numeric_string_is_still_a_number(self):
        assert violations("61", H3) == ["must be at least 124"]


class TestTheWording:
    def test_a_refusal_carries_the_rule_the_caller_could_not_guess(self):
        message = refusal("num_frames", 61, H3)
        assert "61" in message and "rounds up to 73" in message
        assert "at least 124" in message
        assert "124 to 345" in message and "17 * n + 5" in message
        assert H3["reason"] in message

    def test_a_legal_value_has_no_refusal(self):
        assert refusal("num_frames", 124, H3) is None
        assert refusal("num_frames", 130, H3) is None

    def test_a_rounded_value_says_what_it_becomes(self):
        notice = snap_notice("num_frames", 130, H3)
        assert "130" in notice and "141" in notice

    def test_a_value_the_range_still_refuses_is_not_a_notice(self):
        """Two answers to one mistake would be one too many."""
        assert snap_notice("num_frames", 61, H3) is None


def workflow_with(constraints, variables):
    return {
        "id": "constrained",
        "variable_constraints": constraints,
        "variables": variables,
        "steps": [
            {
                "name": "a",
                "task": {"command": "no_op", "arguments": {}},
            }
        ],
    }


class TestTheStaticPass:
    def test_a_stored_default_is_reported_at_variables(self):
        definition = workflow_with({"num_frames": H3}, {"num_frames": 61})

        (problem,) = constraint_errors(definition)

        assert problem["path"] == "variables.num_frames"

    def test_a_supplied_argument_is_reported_where_the_caller_wrote_it(self):
        definition = workflow_with({"num_frames": H3}, {"num_frames": 124})

        (problem,) = constraint_errors(
            definition, {"num_frames": 61}, supplied={"num_frames"}
        )

        assert problem["path"] == "arguments.num_frames"
        assert "at least 124" in problem["message"]

    def test_a_value_the_workflow_rounds_is_a_warning_not_an_error(self):
        definition = workflow_with({"num_frames": H3}, {"num_frames": 124})

        assert constraint_errors(definition, {"num_frames": 130}) == []
        (notice,) = constraint_warnings(definition, {"num_frames": 130})
        assert "141" in notice

    def test_a_workflow_declaring_nothing_is_not_checked(self):
        definition = workflow_with({}, {"num_frames": 61})
        del definition["variable_constraints"]

        assert constraint_errors(definition, {"num_frames": 61}) == []
        assert constraint_warnings(definition, {"num_frames": 61}) == []

    def test_a_constraint_naming_no_variable_is_inert(self):
        definition = workflow_with({"absent": H3}, {"num_frames": 124})

        assert constraint_errors(definition) == []


class TestTheRunTimePass:
    """The backstop for everything the static pass cannot see - an inline
    workflow, a value a parent workflow passed down."""

    def test_an_illegal_value_is_refused_before_anything_loads(self):
        variables = {"num_frames": 61}

        with pytest.raises(ValueError) as caught:
            apply_constraints(workflow_with({"num_frames": H3}, {}), variables)

        assert "at least 124" in str(caught.value)

    def test_a_rounded_value_is_rounded_in_place(self):
        variables = {"num_frames": 130}

        apply_constraints(workflow_with({"num_frames": H3}, {}), variables)

        assert variables["num_frames"] == 141

    def test_a_legal_value_is_untouched(self):
        variables = {"num_frames": 124}

        apply_constraints(workflow_with({"num_frames": H3}, {}), variables)

        assert variables["num_frames"] == 124


class TestOneShapeNotTwo:
    """A chain step's `frame_snap` may name the constraint rather than
    repeating its numbers, so a template states the rule once."""

    def test_a_reference_resolves_to_the_declared_numbers(self):
        definition = {
            "id": "chained",
            "variable_constraints": {"num_frames": H3},
            "steps": [{"name": "a", "chain": {"frame_snap": "constraint:num_frames"}}],
        }

        resolve_constraint_references(definition)

        assert definition["steps"][0]["chain"]["frame_snap"] == {
            "modulus": 17,
            "remainder": 5,
            "min_frames": 124,
            "max_frames": 345,
        }

    def test_the_snap_block_drops_what_a_chain_does_not_read(self):
        assert "snap" not in snap_block(H3) and "reason" not in snap_block(H3)

    def test_a_reference_to_nothing_is_an_error_rather_than_no_constraint(self):
        """A chain that snapped to nothing would stitch segments the
        pipeline refuses."""
        definition = {
            "id": "chained",
            "variable_constraints": {"num_frames": H3},
            "steps": [{"name": "a", "chain": {"frame_snap": "constraint:missing"}}],
        }

        (problem,) = constraint_reference_errors(definition)
        assert problem["path"] == "steps[0].chain.frame_snap"
        assert "num_frames" in problem["message"]

        with pytest.raises(ValueError):
            resolve_constraint_references(definition)

    def test_a_literal_frame_snap_is_left_alone(self):
        block = {"modulus": 17, "remainder": 5}
        definition = {
            "id": "chained",
            "steps": [{"name": "a", "chain": {"frame_snap": dict(block)}}],
        }

        resolve_constraint_references(definition)

        assert definition["steps"][0]["chain"]["frame_snap"] == block
        assert constraint_reference_errors(definition) == []


class TestValidationReportsIt:
    def test_validation_errors_carries_the_refusal(self, tmp_path):
        definition = workflow_with({"num_frames": H3}, {"num_frames": 124})

        problems = workflow_from_definition(
            definition, str(tmp_path)
        ).validation_errors({"num_frames": 61})

        assert any(
            "at least 124" in problem["message"]
            and problem["path"] == "arguments.num_frames"
            for problem in problems
        )


TEMPLATES = sorted(
    glob.glob(os.path.join(REPO_ROOT, "workflows", "**", "*.json"), recursive=True)
)


def declared_in_the_catalog():
    """Every constraint block in the catalog, as (path, variable, rule)."""
    found = []
    for path in TEMPLATES:
        with open(path, encoding="utf-8") as handle:
            definition = json.load(handle)
        for variable, rule in (definition.get("variable_constraints") or {}).items():
            found.append((os.path.relpath(path, REPO_ROOT), variable, rule))
    return found


class TestTheCatalogsNumbersAreTheLibrarys:
    """The sweep, not a hand-written assertion per template: every declared
    constraint is checked against the diffusers symbol it derives from, so a
    library change fails here rather than in a run."""

    def test_the_catalog_declares_some(self):
        assert declared_in_the_catalog(), "no template declares a constraint"

    def test_every_constraint_names_a_variable_the_workflow_declares(self):
        """A rule on a name nothing reads is checked against nothing."""
        for path in TEMPLATES:
            with open(path, encoding="utf-8") as handle:
                definition = json.load(handle)
            constraints = definition.get("variable_constraints") or {}
            variables = definition.get("variables") or {}
            for variable in constraints:
                assert variable in variables, (
                    f"{os.path.relpath(path, REPO_ROOT)} constrains '{variable}', "
                    "which it does not declare as a variable"
                )

    def test_every_default_satisfies_its_own_constraint(self):
        """A stored default outside its own rule would make the template
        unrunnable as shipped."""
        for path in TEMPLATES:
            with open(path, encoding="utf-8") as handle:
                definition = json.load(handle)
            for problem in constraint_errors(definition):
                pytest.fail(f"{os.path.relpath(path, REPO_ROOT)}: {problem}")

    def test_the_h3_frame_rule_is_the_pipelines(self):
        h3 = pytest.importorskip(
            "diffusers.modular_pipelines.minimax_h3.modular_pipeline",
            reason="diffusers without MiniMax-H3",
        )
        # The properties fall back to the released model's values when no
        # component is loaded, so the numbers need no weights
        bare = object.__new__(h3.MiniMaxH3ModularPipeline)
        modulus = bare.vae_frames_per_chunk
        remainder = bare.vae_latents_per_chunk
        fps = h3.MINIMAX_H3_FPS
        # The bounds hold for the *aligned* count, so they are the smallest
        # and largest grid values inside the duration window
        low = next(
            n
            for n in range(1, 10_000)
            if n % modulus == remainder and n / fps >= bare.min_duration
        )
        high = max(
            n
            for n in range(1, 10_000)
            if n % modulus == remainder and n / fps <= bare.max_duration
        )

        declared = [
            (path, rule)
            for path, variable, rule in declared_in_the_catalog()
            if "minimax" in path and variable == "num_frames"
        ]
        assert declared, "no MiniMax template declares a num_frames constraint"
        for path, rule in declared:
            assert rule["modulus"] == modulus, path
            assert rule["remainder"] == remainder, path
            assert rule["min_frames"] == low, path
            assert rule["max_frames"] == high, path
            # The pipeline rounds up and warns; the constraint says so too,
            # so the caller hears it before the run rather than in a log
            assert rule["snap"] == "up", path

    def test_the_ltx2_frame_rule_is_the_pipelines(self):
        ltx = pytest.importorskip(
            "diffusers.pipelines.ltx2.dfr_layout", reason="diffusers without LTX-2"
        )
        ratio = (
            ltx.resolve_canvas.__defaults__[0] if ltx.resolve_canvas.__defaults__ else 8
        )

        declared = [
            (path, rule)
            for path, variable, rule in declared_in_the_catalog()
            if "ltx2" in path and variable == "num_frames"
        ]
        assert declared, "no LTX-2.5 template declares a num_frames constraint"
        for path, rule in declared:
            # `(num_frames - 1) % temporal_compression_ratio == 0`
            assert rule["modulus"] == ratio, path
            assert rule["remainder"] == 1, path
            # `resolve_canvas` needs at least `ratio + 1` pixel frames
            assert rule["min_frames"] == ratio + 1, path
            # No snap: the LTX-2 pipelines floor an off-grid count
            # (`(n - 1) // ratio * ratio + 1`), so rounding up here would
            # hand back a longer clip than either the caller or the
            # pipeline chose
            assert "snap" not in rule, path

    def test_a_chain_step_states_the_rule_once(self):
        """Where a template declares a constraint and also snaps a chain,
        the chain names the constraint rather than repeating its numbers."""
        for path in TEMPLATES:
            with open(path, encoding="utf-8") as handle:
                text = handle.read()
            definition = json.loads(text)
            if not (definition.get("variable_constraints") or {}):
                continue
            for step in definition.get("steps") or []:
                snap = (step.get("chain") or {}).get("frame_snap")
                if snap is None:
                    continue
                assert isinstance(snap, str) and snap.startswith("constraint:"), (
                    f"{os.path.relpath(path, REPO_ROOT)} step '{step.get('name')}' "
                    "repeats the numbers its 'variable_constraints' already state"
                )
            assert constraint_reference_errors(definition) == [], path
