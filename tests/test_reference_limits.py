"""Reference limits at validation time, not after a checkpoint load.

MiniMax-H3's ref2va block refuses a reference set that breaks its limits -
but it does so once the model is up, so a caller who ran the free
validate_workflow, was quoted eight minutes and acknowledged it found out
minutes in, from a failed job (#136). The numbers here are never asserted
against literals of ours: they are read off the diffusers block that
enforces them, which is the only place they stay correct.
"""

import json

import pytest

from dw.reference_limits import (
    REFERENCE_LIMIT_BLOCKS,
    _limits,
    reference_limit_errors,
)
from dw.workflow import Workflow

H3 = "diffusers.modular_pipelines.minimax_h3"
IMAGE = f"{H3}.MiniMaxH3ImageReference"
VIDEO = f"{H3}.MiniMaxH3VideoReference"
AUDIO = f"{H3}.MiniMaxH3AudioReference"


def reference(reference_type, name="x"):
    return {"reference_type": reference_type, "from_file": f"{name}.bin"}


def workflow_with(references):
    return {
        "id": "refs",
        "steps": [
            {
                "name": "shot",
                "pipeline": {
                    "configuration": {"component_type": "ModularPipeline"},
                    "from_pretrained_arguments": {
                        "model_name": "MiniMaxAI/MiniMax-H3",
                        "workflow": "ref2va",
                    },
                    "arguments": {"prompt": "x", "references": references},
                },
                "result": {"content_type": "video/mp4"},
            }
        ],
    }


def messages(references):
    return [e["message"] for e in reference_limit_errors(workflow_with(references))]


class TestLimitsComeFromDiffusers:
    def test_the_block_declares_them(self):
        model_name, per_kind, total = _limits(H3)
        assert model_name
        assert set(per_kind) == {"image", "video", "audio"}
        assert total >= max(per_kind.values())

    def test_an_unknown_family_is_not_checked(self):
        assert H3 in REFERENCE_LIMIT_BLOCKS
        assert messages([{"reference_type": "json.JSONDecoder"}]) == []


class TestH3ReferenceSets:
    def test_audio_cannot_be_the_only_reference(self):
        found = messages([reference(AUDIO)])
        assert len(found) == 1
        assert "cannot be used on its own" in found[0]

    def test_audio_paired_with_a_picture_is_fine(self):
        assert messages([reference(IMAGE), reference(AUDIO)]) == []

    def test_too_many_of_one_kind_is_refused(self):
        _model, per_kind, _total = _limits(H3)
        over = per_kind["audio"] + 1
        found = messages(
            [reference(IMAGE)] + [reference(AUDIO, f"a{i}") for i in range(over)]
        )
        assert len(found) == 1
        assert f"at most {per_kind['audio']} audio references, got {over}" in found[0]

    def test_the_total_is_refused_too(self):
        _model, per_kind, total = _limits(H3)
        found = messages(
            [reference(IMAGE, f"i{i}") for i in range(per_kind["image"])]
            + [reference(VIDEO, f"v{i}") for i in range(per_kind["video"])]
            + [reference(AUDIO, f"a{i}") for i in range(per_kind["audio"])]
        )
        assert len(found) == 1
        assert f"at most {total} references in total" in found[0]

    def test_a_set_at_every_limit_passes(self):
        _model, per_kind, total = _limits(H3)
        references = [reference(IMAGE, f"i{i}") for i in range(per_kind["image"])]
        references += [
            reference(VIDEO, f"v{i}")
            for i in range(min(per_kind["video"], total - len(references)))
        ]
        assert messages(references) == []

    def test_an_unresolved_reference_is_left_to_another_pass(self):
        assert messages([{"reference_type": "variable:kind"}]) == []
        assert messages("variable:references") == []


class TestThroughTheTemplate:
    """dialogue-short's `shots` entries carry the references, so the error has
    to name the member and point at the step the author wrote."""

    def _template(self):
        path = "workflows/templates/minimax/dialogue-short.json"
        with open(path) as handle:
            return Workflow(json.load(handle), "outputs", path)

    def test_the_template_itself_validates(self):
        assert self._template().validation_errors() == []

    def test_an_audio_only_shot_is_refused_before_the_run(self):
        shots = [
            {
                "name": "audio_only",
                "num_frames": 124,
                "prompt": "x",
                "references": [reference(AUDIO)],
            }
        ]
        errors = self._template().validation_errors({"shots": shots})
        assert len(errors) == 1
        assert errors[0]["path"].endswith("pipeline.arguments.references")
        assert "shot@audio_only" in errors[0]["message"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
