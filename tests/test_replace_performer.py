"""The replace-performer templates (#1374): per-cut regional edit refusals and
the step-level item seed."""

import json
from pathlib import Path

import jsonschema
import pytest

from dw.workflow import workflow_from_definition
from tests.test_h3_mask import write_video

ROOT = Path(__file__).resolve().parent.parent
TEMPLATES = ROOT / "workflows" / "templates" / "minimax"
REF = "diffusers.modular_pipelines.minimax_h3.MiniMaxH3ImageReference"
GOOD = [
    {"name": "cut-1", "start_frame": 0, "num_frames": 120, "render_frames": 124},
    {"name": "cut-2", "start_frame": 120, "num_frames": 96, "render_frames": 124},
]


@pytest.fixture(scope="module")
def harness(tmp_path_factory):
    d = tmp_path_factory.mktemp("replace_performer")
    write_video(d / "src.mp4", frames=216, size=(960, 544), sample_rate=16000)
    write_video(d / "mask.mp4", frames=216, size=(960, 544), value=255)
    write_video(d / "mask_small.mp4", frames=216, size=(480, 272), value=255)
    tpl = json.loads((TEMPLATES / "replace-performer.json").read_text())
    wf = workflow_from_definition(tpl, str(d), base_dir=str(d))

    def run(cuts, **kw):
        args = {
            "source_video": "src.mp4",
            "mask_video": "mask.mp4",
            "references": [{"reference_type": REF, "from_file": "src.mp4"}],
            "cuts": cuts,
        }
        args.update(kw)
        return wf.validation_errors(args)

    return run


def cut(**kw):
    c = {"name": "c", "start_frame": 0, "num_frames": 120, "render_frames": 124}
    c.update(kw)
    return c


def text(errors):
    return "\n".join(str(e) for e in errors)


def test_two_cuts_validate_clean(harness):
    assert harness(GOOD) == []


def test_cut_with_its_own_seed_validates_clean(harness):
    assert harness([dict(GOOD[0], seed=7), GOOD[1]]) == []


def test_render_frames_off_grid_refused(harness):
    out = text(harness([cut(render_frames=125)]))
    assert "arguments.cuts[0].render_frames" in out


def test_render_frames_out_of_range_refused(harness):
    out = text(harness([cut(num_frames=100, render_frames=107)]))
    assert "arguments.cuts[0].render_frames" in out


def test_render_frames_below_num_frames_refused(harness):
    out = text(harness([cut(num_frames=150, render_frames=124)]))
    assert out
    assert "render_frames" in out or "num_frames" in out


def test_empty_cuts_refused(harness):
    assert "steps[0].for_each" in text(harness([]))


def test_too_many_cuts_refused(harness):
    cuts = [cut(name=f"c{i}", num_frames=100) for i in range(33)]
    assert "steps[0].for_each" in text(harness(cuts))


def test_cut_past_source_end_refused(harness):
    out = text(harness([cut(start_frame=200)]))
    assert "source_video" in out
    assert "mask_video" in out


def test_cut_past_source_end_is_the_trim_rule(harness):
    """The static refusal asks trim's span rule, not a copy of it (#1374)."""
    from dw.tasks.trim import span_problem

    assert span_problem(200, 120, 216) in text(harness([cut(start_frame=200)]))


def test_mask_size_is_the_regional_edit_rule(harness):
    """The static refusal asks h3_rules.media_problems, as _with_mask does."""
    from dw.pipeline_processors.h3_rules import media_problems

    expected = media_problems("mask_video", (124, 272, 480), 960, 544)
    assert expected and expected[0] in text(harness(GOOD, mask_video="mask_small.mp4"))


def test_mask_of_another_size_refused(harness):
    out = text(harness(GOOD, mask_video="mask_small.mp4"))
    assert "mask_video" in out
    assert "480x272" in out
    assert "960x544" in out


@pytest.fixture(scope="module")
def step_seed_schema():
    schema = json.loads((ROOT / "dw" / "workflow_schema.json").read_text())
    return schema["$defs"]["step"]["properties"]["seed"]


@pytest.mark.parametrize(
    "value",
    [
        3,
        "variable:x",
        "item:seed",
        {"item": "seed", "default": "variable:seed"},
        {"item": "seed", "default": 7},
    ],
)
def test_step_seed_accepts(step_seed_schema, value):
    jsonschema.validate(value, step_seed_schema)


@pytest.mark.parametrize("value", [{"default": 1}, {"item": "seed", "extra": 1}])
def test_step_seed_rejects(step_seed_schema, value):
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(value, step_seed_schema)


def _steps(**trim):
    return [
        {
            "name": "trim",
            "task": {"command": "trim_video", "arguments": trim},
        }
    ]


def test_trimmed_frames_takes_pad_to():
    from dw.hold_audio import _trimmed_frames

    ref = "previous_result:trim"
    assert _trimmed_frames(_steps(num_frames=100, pad_to=124), ref) == 124
    assert _trimmed_frames(_steps(num_frames=150, pad_to=124), ref) == 150
    assert _trimmed_frames(_steps(num_frames=100), ref) == 100


def test_cuts_template_validates(tmp_path):
    tpl = json.loads((TEMPLATES / "replace-performer-cuts.json").read_text())
    wf = workflow_from_definition(tpl, str(tmp_path), base_dir=str(tmp_path))
    assert wf is not None
