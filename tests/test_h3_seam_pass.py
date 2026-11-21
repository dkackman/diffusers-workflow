"""h3-seam-pass, the template (#1230, stage #1378): a finished take's seams
smoothed by a low-strength windowed refine. Its source is required and must be
a video, its estimate is the flat measured row, and the playbook's verdict on
it (smooths seams, not a burn fix) is in its description."""

import json
from pathlib import Path

import pytest

from dw.plan import estimate
from dw.server.catalog_shape import derive_catalog_metadata
from dw.workflow import workflow_from_definition
from tests.test_h3_mask import write_video

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "workflows" / "templates" / "minimax" / "h3-seam-pass.json"


def definition():
    return json.loads(TEMPLATE.read_text())


@pytest.fixture(scope="module")
def validate(tmp_path_factory):
    d = tmp_path_factory.mktemp("seam_pass")
    write_video(d / "take.mp4", frames=48, size=(960, 544), sample_rate=16000)
    (d / "take.wav").write_bytes(b"RIFF")
    (d / "still.png").write_bytes(b"\x89PNG")
    wf = workflow_from_definition(definition(), str(d), base_dir=str(d))
    return lambda **arguments: [str(e) for e in wf.validation_errors(arguments)]


def test_only_source_set_validates_clean(validate):
    assert validate(source="take.mp4") == []


def test_no_source_is_refused(validate):
    assert validate() != []


def test_the_document_checked_bare_leaves_its_required_track_unjudged(tmp_path):
    # #1253: no arguments does not judge a required null, so the refine's
    # "needs hold_audio" rules (refine_strength, refine_windows) stay quiet
    wf = workflow_from_definition(definition(), str(tmp_path), base_dir=str(tmp_path))
    assert [str(e) for e in wf.validation_errors(None)] == []


def test_arguments_without_source_are_refused_naming_it(validate):
    errors = validate(refine_strength=0.2)
    assert any("'source'" in e for e in errors), errors


def test_the_shared_predicate_reads_the_written_required_variable():
    from dw.hold_audio import held_by_a_required_null, judged_arguments

    written = definition()
    source = next(i for i, s in enumerate(written["steps"]) if s["name"] == "refine")
    bare = {"refine_strength": 0.3, "hold_audio": None}
    assert held_by_a_required_null(written, source, bare)
    assert judged_arguments(bare, source, written)["hold_audio"] == "variable:source"
    # A caller's arguments are judged as given
    assert judged_arguments(bare, source, written, given={}) is bare
    assert not held_by_a_required_null(written, source, {"hold_audio": "a.mp4"})


@pytest.mark.parametrize("source", ["take.wav", "still.png"])
def test_a_source_that_is_not_a_video_is_refused_at_validate(validate, source):
    errors = validate(source=source)
    assert any("steps[0]" in e and "source" in e for e in errors), errors


def test_the_refine_runs_windowed_at_the_takes_own_canvas():
    steps = {s["name"]: s for s in definition()["steps"]}
    refine = steps["refine"]["pipeline"]["arguments"]
    assert refine["refine_windows"] == {}
    assert refine["refine_strength"] == "variable:refine_strength"
    assert refine["hold_audio"] == "variable:source"
    assert refine["width"] == "previous_result:encode.width"
    assert refine["height"] == "previous_result:encode.height"
    assert refine["output"] == ["latents"]
    assert definition()["variables"]["refine_strength"] == 0.3
    assert steps["mux"]["task"]["arguments"]["audio"] == "variable:source"


def test_the_catalog_reads_it_as_a_shot_that_needs_input_media():
    meta = derive_catalog_metadata(definition())
    assert meta["shape"] == "shot"
    assert "needs-input-media" in meta["traits"]


def test_the_estimate_is_the_flat_measured_row():
    d = definition()
    quoted = estimate(d, {"variables": d["variables"]}, {}, "cuda", None, None)
    assert quoted["minutes"] == pytest.approx(d["cost"][0]["minutes"])
    assert quoted["basis"] == "catalog"


def test_the_description_states_its_cost_and_its_limit():
    text = definition()["description"]
    assert "36-40 GPU-min per minute of take" in text
    assert "16 GiB peak" in text
    assert "not a burn fix" in text
