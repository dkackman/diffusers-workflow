"""An H3 adapter has to match the checkpoint partition its step denoises on.

`ref2va` loads `transformer_ref` and nothing else, so diffusers puts whatever
adapter it is handed onto that partition: an FL2VA turbo LoRA on a `ref2va`
step runs, succeeds, and comes back a worse clip. Four templates carried that
mistake and nothing complained (#149), and `validate_workflow` passed it
cleanly with `checked_arguments` claiming it had looked (#155).

The workflow names and the partition each denoises against are diffusers' own
and are pinned to it here. The adapter *file names* are MiniMax's hub naming
and no symbol declares them, which is why an unrecognised name is a warning
rather than a refusal - and why the catalog's own defaults are swept, as the
one place the convention is written down.
"""

import glob
import json
import os

import pytest

from dw.adapter_compatibility import (
    H3_WORKFLOWS,
    KEYFRAME_TOKEN,
    KEYFRAME_WORKFLOWS,
    REFERENCE_TOKEN,
    REFERENCE_WORKFLOWS,
    adapter_errors,
    adapter_warnings,
)
from tests.test_examples import REPO_ROOT

TEMPLATES = sorted(
    glob.glob(os.path.join(REPO_ROOT, "workflows", "**", "*.json"), recursive=True)
)


def load(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def h3_step(workflow, weight_name, name="shot"):
    return {
        "name": name,
        "pipeline": {
            "from_pretrained_arguments": {
                "model_name": "MiniMaxAI/MiniMax-H3",
                "workflow": workflow,
            },
            "loras": [
                {"model_name": "MiniMaxAI/MiniMax-H3", "weight_name": weight_name}
            ],
        },
    }


def definition(*steps):
    return {"id": "w", "steps": list(steps)}


class TestTheNamesAreDiffusers:
    def test_the_workflow_names_are_the_ones_diffusers_takes(self):
        from diffusers.modular_pipelines.minimax_h3.modular_blocks_minimax_h3 import (
            MiniMaxH3Blocks,
        )

        assert H3_WORKFLOWS == frozenset(MiniMaxH3Blocks._workflow_map)
        assert REFERENCE_WORKFLOWS | KEYFRAME_WORKFLOWS == H3_WORKFLOWS
        assert not (REFERENCE_WORKFLOWS & KEYFRAME_WORKFLOWS)

    def test_the_reference_path_denoises_on_its_own_partition(self):
        """Why an adapter cannot be shared across the two: they are two
        checkpoints, and each denoise step declares the one it loads."""
        from diffusers.modular_pipelines.minimax_h3.modular_blocks_minimax_h3 import (
            MiniMaxH3CoreDenoiseStep,
            MiniMaxH3Ref2VACoreDenoiseStep,
        )

        def components(block):
            return {getattr(spec, "name", spec) for spec in block().expected_components}

        assert "transformer_ref" in components(MiniMaxH3Ref2VACoreDenoiseStep)
        assert "transformer" not in components(MiniMaxH3Ref2VACoreDenoiseStep)
        assert "transformer" in components(MiniMaxH3CoreDenoiseStep)
        assert "transformer_ref" not in components(MiniMaxH3CoreDenoiseStep)


class TestWhatIsRefused:
    def test_a_keyframe_adapter_on_the_reference_path(self):
        errors = adapter_errors(
            definition(
                h3_step("ref2va", "minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors")
            )
        )
        assert len(errors) == 1
        assert errors[0]["path"] == "steps[0].pipeline.loras[0].weight_name"
        assert "transformer_ref" in errors[0]["message"]

    def test_a_reference_adapter_on_the_keyframe_path(self):
        """The symmetric mistake, equally invisible."""
        for workflow in sorted(KEYFRAME_WORKFLOWS):
            errors = adapter_errors(
                definition(
                    h3_step(
                        workflow,
                        "minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors",
                    )
                )
            )
            assert len(errors) == 1, workflow

    def test_the_matching_adapter_passes(self):
        assert (
            adapter_errors(
                definition(
                    h3_step(
                        "ref2va",
                        "minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors",
                    ),
                    h3_step(
                        "fl2va",
                        "minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors",
                        name="other",
                    ),
                )
            )
            == []
        )

    def test_an_unrecognised_name_is_a_warning_not_a_refusal(self):
        """The escape hatch: a reference-trained checkpoint nobody can name
        yet still gets through, with the rule stated."""
        spec = definition(h3_step("ref2va", "my-new-ref-lora.safetensors"))
        assert adapter_errors(spec) == []
        warnings = adapter_warnings(spec)
        assert len(warnings) == 1
        assert REFERENCE_TOKEN in warnings[0] and KEYFRAME_TOKEN in warnings[0]

    def test_an_unresolved_reference_is_another_passs_business(self):
        spec = definition(h3_step("ref2va", "variable:lora_weight_name"))
        assert adapter_errors(spec) == [] and adapter_warnings(spec) == []

    def test_a_pipeline_that_is_not_h3_is_not_checked(self):
        spec = definition(h3_step("something-else", "fl2v.safetensors"))
        assert adapter_errors(spec) == [] and adapter_warnings(spec) == []

    def test_the_caller_is_told_where_they_wrote_it(self):
        written = definition(h3_step("ref2va", "variable:lora_weight_name"))
        expanded = definition(
            h3_step("ref2va", "minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors")
        )
        errors = adapter_errors(
            expanded, None, written=written, supplied={"lora_weight_name"}
        )
        assert errors[0]["path"] == "arguments.lora_weight_name"

    def test_a_for_each_member_names_itself(self):
        expanded = definition(
            h3_step(
                "ref2va",
                "minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors",
                name="shot@closeup",
            )
        )
        errors = adapter_errors(expanded, [0])
        assert "member 'shot@closeup'" in errors[0]["message"]


class TestTheCatalogPairsThemRight:
    """The convention's only written record: every stored workflow's own
    default. #149 fixed four of these by hand; this is what stops the fifth."""

    @pytest.mark.parametrize("path", TEMPLATES)
    def test_no_stored_workflow_mispairs_an_adapter(self, path):
        spec = load(path)
        assert adapter_errors(spec) == [], path
        assert adapter_warnings(spec) == [], path

    def test_the_sweep_found_the_family(self):
        paired = [
            path
            for path in TEMPLATES
            if any(
                isinstance(step, dict)
                and isinstance(step.get("pipeline"), dict)
                and (step["pipeline"].get("from_pretrained_arguments") or {}).get(
                    "workflow"
                )
                in H3_WORKFLOWS
                and step["pipeline"].get("loras")
                for step in load(path).get("steps") or []
            )
        ]
        assert len(paired) >= 15, "the H3 family moved - this sweep found almost none"
