"""The pre-run check of a chain's `continuity: "guide"` (dw/guides.py)."""

import copy
import json
from pathlib import Path

import pytest

from dw.guides import guide_chain_errors
from dw.workflow import Workflow

TEMPLATE = (
    Path(__file__).parent.parent / "workflows/templates/minimax/chained-segments.json"
)


def definition(
    workflow="fl2va",
    component_type="ModularPipeline",
    chain=None,
    **arguments,
):
    return {
        "steps": [
            {
                "name": "video",
                "pipeline": {
                    "configuration": {"component_type": component_type},
                    "from_pretrained_arguments": {
                        "model_name": "MiniMaxAI/MiniMax-H3",
                        "workflow": workflow,
                    },
                    "chain": {"segments": 3, "continuity": "guide"}
                    if chain is None
                    else chain,
                    "arguments": arguments,
                },
            }
        ]
    }


def one(workflow_definition):
    errors = guide_chain_errors(workflow_definition)
    assert len(errors) == 1, errors
    return errors[0]


class TestClean:
    @pytest.mark.parametrize("workflow", ["fl2va", "t2va"])
    def test_h3_guide_chains(self, workflow):
        assert guide_chain_errors(definition(workflow, prompt="a cat")) == []

    @pytest.mark.parametrize("frames", [22, 39])
    def test_the_whole_latent_guide_lengths(self, frames):
        chain = {"segments": 2, "continuity": "guide", "guide_frames": frames}
        assert guide_chain_errors(definition(chain=chain)) == []

    def test_a_step_without_a_chain(self):
        no_chain = definition()
        del no_chain["steps"][0]["pipeline"]["chain"]
        assert guide_chain_errors(no_chain) == []

    def test_no_steps(self):
        assert guide_chain_errors({}) == []

    def test_last_frame_is_the_default_and_clean(self):
        assert guide_chain_errors(definition("ref2va", chain={"segments": 2})) == []

    def test_last_segment_on_ref2va_is_clean(self):
        chain = {"segments": 2, "continuity": "last_segment"}
        assert guide_chain_errors(definition("ref2va", chain=chain)) == []


class TestRefused:
    def test_ref2va(self):
        error = one(definition("ref2va"))

        assert error["path"] == "steps[0].pipeline.chain.continuity"
        assert "ref2va" in error["message"]

    def test_another_pipeline_class(self):
        error = one(definition(component_type="LTX2ImageToVideoPipeline"))

        assert error["path"] == "steps[0].pipeline.chain.continuity"
        assert "LTX2ImageToVideoPipeline" in error["message"]

    def test_references_in_the_arguments(self):
        error = one(definition(references=[{"image": "asset:a.png"}]))

        assert error["path"] == "steps[0].pipeline.chain.continuity"
        assert "references" in error["message"]

    @pytest.mark.parametrize("frames", [30, 5])
    def test_a_guide_length_other_than_22_or_39(self, frames):
        chain = {"segments": 2, "continuity": "guide", "guide_frames": frames}

        error = one(definition(chain=chain))

        assert error["path"] == "steps[0].pipeline.chain.guide_frames"
        assert "22" in error["message"]
        assert "39" in error["message"]

    def test_carry_frames(self):
        chain = {"segments": 2, "continuity": "guide", "carry_frames": 10}

        error = one(definition(chain=chain))

        assert error["path"] == "steps[0].pipeline.chain.carry_frames"

    def test_an_unknown_continuity(self):
        error = one(definition(chain={"segments": 2, "continuity": "teleport"}))

        assert error["path"] == "steps[0].pipeline.chain.continuity"
        assert "teleport" in error["message"]
        assert "guide" in error["message"]

    def test_every_run_mode_is_known_to_validate(self):
        from dw.pipeline_processors.chain import CONTINUITY_MODES

        for mode in CONTINUITY_MODES:
            chain = {"segments": 2, "continuity": mode}
            assert guide_chain_errors(definition("t2va", chain=chain)) == []

    def test_the_step_rule_is_the_guides_rule(self):
        # One owner: the chain's refusal carries the `guides` check's own reason
        from dw.guides import _takes_guides_problem

        step = definition("ref2va")["steps"][0]["pipeline"]
        error = one(definition("ref2va"))

        assert error["message"].endswith(_takes_guides_problem(step, {}))


class TestSkipped:
    def test_a_variable_continuity(self):
        chain = {"segments": 2, "continuity": "variable:continuity"}
        assert guide_chain_errors(definition("ref2va", chain=chain)) == []

    def test_a_variable_guide_length(self):
        chain = {
            "segments": 2,
            "continuity": "guide",
            "guide_frames": "variable:guide_frames",
        }
        assert guide_chain_errors(definition(chain=chain)) == []

    def test_guide_frames_is_ignored_on_last_frame(self):
        chain = {"segments": 2, "continuity": "last_frame", "guide_frames": 30}
        assert guide_chain_errors(definition(chain=chain)) == []


class TestThroughTheWorkflow:
    def errors(self, tmp_path, **chain):
        template = json.loads(TEMPLATE.read_text())
        workflow = Workflow(
            copy.deepcopy(template), str(tmp_path), str(tmp_path / "chained.json")
        )
        return workflow.validation_errors(chain)

    def test_a_bad_guide_length_is_refused(self, tmp_path):
        errors = self.errors(tmp_path, continuity="guide", guide_frames=30)

        assert any("guide_frames" in error["path"] for error in errors), errors

    def test_the_default_guide_chain_is_clean(self, tmp_path):
        assert self.errors(tmp_path, continuity="guide") == []


class TestOwnGuidesPlusTheChain:
    """The chain adds a guide of its own at frame 0 (chain.py GuideContinuity),
    so a step's written guides have one fewer slot than GUIDE_LIMIT."""

    @staticmethod
    def guides(count):
        return [{"video": f"g{i}.mp4", "frame": 17 * (i + 1)} for i in range(count)]

    def test_four_own_guides_leave_no_room_for_the_chain_guide(self):
        error = one(definition(prompt="a cat", guides=self.guides(4)))
        assert error["path"].endswith("continuity")
        assert "the chain adds one" in error["message"]
        assert "got 4" in error["message"]

    def test_three_own_guides_leave_room(self):
        assert (
            guide_chain_errors(definition(prompt="a cat", guides=self.guides(3))) == []
        )

    def test_an_unresolved_guides_value_is_left_to_the_run(self):
        assert (
            guide_chain_errors(definition(prompt="a cat", guides="variable:guides"))
            == []
        )

    def test_last_frame_continuity_takes_the_full_four(self):
        chain = {"segments": 2, "continuity": "last_frame"}
        assert (
            guide_chain_errors(
                definition(chain=chain, prompt="a cat", guides=self.guides(4))
            )
            == []
        )
