"""The chain check's identity measure (chain_check.py): face_drop, segment
1's median face similarity to its own frame 0 minus a later segment's. It is
warn-only and off by default: measured and recorded, never failed, until a
ceiling is set - and a set ceiling fails without a re-roll.

Lip sync is not a chain measure: on 70 real H3 dialogue segments no
threshold separated a single segment's good audio from bad (ruling R9,
bench/records/narrated.md, "Sync calibration"), so `measure_lip_sync` is a
take-level task only and the chain check refuses `lip_sync_min`.

The models are replaced at chain_check's own seam (`_face_scores`), so
nothing here reads weights."""

import logging

import pytest
from PIL import Image

import dw.pipeline_processors.chain_check as cc
import dw.plan_downloads
from dw.pipeline_processors.chain import run_chain
from dw.pipeline_processors.chain_check import (
    MEASURES,
    WARN_ONLY,
    CheckResult,
    SegmentCheck,
    face_base,
    segment_check,
    with_face_drop,
)
from dw.pipeline_processors.chain_rules import chain_check_problem
from dw.tasks import face_drift
from tests.test_chain import FakePipeline, modular_output


def _frames(level, count=8):
    return [Image.new("RGB", (32, 32), (level, level, level)) for _ in range(count)]


def _scores_by_level(table):
    """A `_face_scores` stand-in: a segment's scores read off its grey level."""

    def scores(frames, reference, samples):
        return list(table[frames[0].getpixel((0, 0))[0]])

    return scores


def _passed():
    return CheckResult(True, {}, [])


def test_face_drop_is_measured_and_warn_only(monkeypatch):
    monkeypatch.setattr(
        cc, "_face_scores", _scores_by_level({100: [0.9] * 8, 120: [0.5] * 8})
    )
    base = face_base(_frames(100))
    result = with_face_drop(_passed(), _frames(120), base, SegmentCheck())
    assert result.measures["face_drop"] == pytest.approx(0.4)
    assert result.passed and result.failed == []
    assert "face_drop" in MEASURES and "face_drop" in WARN_ONLY


def test_lip_sync_is_not_a_chain_measure():
    assert "lip_sync" not in MEASURES and "lip_sync" not in WARN_ONLY
    assert not hasattr(SegmentCheck(), "lip_sync_min")
    assert not hasattr(cc, "_sync_score")


def test_face_base_excludes_frame_0(monkeypatch):
    seen = []

    def scores(frames, reference, samples):
        seen.append((list(frames), reference))
        return [0.8] * len(frames)

    monkeypatch.setattr(cc, "_face_scores", scores)
    frames = [Image.new("RGB", (32, 32), (level,) * 3) for level in range(10, 90, 10)]
    reference, median = face_base(frames)
    assert reference is frames[0] and median == pytest.approx(0.8)
    sampled, held = seen[0]
    # Segment 1's own median is read from index 1 on: frame 0 against itself
    # would match at 1.0 and lift the base every later drop is read from
    assert held is frames[0]
    assert all(frame is not frames[0] for frame in sampled)
    assert sampled == frames[1:]


def test_no_face_in_segment_1_leaves_face_drop_unmeasured(monkeypatch):
    monkeypatch.setattr(cc, "_face_scores", lambda frames, reference, samples: [])
    assert face_base(_frames(100)) is None
    result = with_face_drop(_passed(), _frames(120), None, SegmentCheck())
    assert result.measures["face_drop"] is None and result.passed


def test_a_model_that_cannot_load_records_none(monkeypatch):
    def broken(*args):
        raise RuntimeError("no YuNet in this cv2")

    monkeypatch.setattr(cc, "_face_scores", broken)
    assert face_base(_frames(100)) is None
    result = with_face_drop(
        _passed(), _frames(120), None, SegmentCheck(face_drop_max=0.1)
    )
    assert result.measures["face_drop"] is None and result.passed


def _checked_chain(segments=2, **check):
    def factory(arguments, index):
        out = modular_output(arguments, index, sample_rate=100)
        level = 100 if index == 0 else 250
        out["videos"] = [
            [Image.new("RGB", (32, 32), (level,) * 3) for _ in out["videos"][0]]
        ]
        return out

    pipeline = FakePipeline(factory)
    chain = {
        "segments": segments,
        "fps": 4,
        "trim_frames": 2,
        "check": {"jump_ratio_max": 1000, "look_ratio_max": 1000, **check},
        "retries": 1,
    }
    return pipeline, run_chain(pipeline, chain, {"num_frames": 8, "seed": 5})


def test_face_drop_max_sets_a_failing_ceiling_without_reroll(monkeypatch):
    monkeypatch.setattr(
        cc, "_face_scores", _scores_by_level({100: [0.9] * 8, 250: [0.5] * 8})
    )
    check = segment_check({"face_drop_max": 0.1, "jump_ratio_max": 1000})
    assert check.face_drop_max == 0.1

    pipeline, result = _checked_chain(face_drop_max=0.1)
    shot = result.shots[1]
    # Warn-only: failed and recorded, but never re-rolled
    assert len(pipeline.calls) == 2 and shot["attempts"] == 1
    assert shot["check_failed"] == ["face_drop"]
    assert shot["check"] == {"face_drop": pytest.approx(0.4)}


def test_faceless_segment_1_logs_once(monkeypatch, caplog):
    def no_reference_face(frames, reference, samples):
        raise face_drift.NoFaceInReference(
            "analyze_face_drift: no face found in 'reference' - give a picture "
            "with the character's face in it"
        )

    monkeypatch.setattr(cc, "_face_scores", no_reference_face)
    with caplog.at_level(logging.INFO, logger="dw"):
        _, result = _checked_chain(segments=3)
    lines = [
        record.getMessage()
        for record in caplog.records
        if "face_drop not measured" in record.getMessage()
    ]
    assert lines == [
        "Chain check: face_drop not measured - segment 1's first frame shows no face"
    ]
    # The task's message is for a caller who gave a reference; a chain's
    # caller gave none
    assert "give a picture" not in caplog.text
    assert all(shot["check"]["face_drop"] is None for shot in result.shots[1:])


def test_checked_chain_lists_face_weights_in_downloads(monkeypatch, tmp_path):
    monkeypatch.setattr(
        dw.plan_downloads, "scan_models", lambda cache_dir=None: {"repos": []}
    )

    def downloads(check):
        chain = {"segments": 2, "fps": 4}
        if check is not None:
            chain["check"] = check
        expanded = {"steps": [{"pipeline": {"chain": chain, "arguments": {}}}]}
        return dw.plan_downloads.downloads_required(
            expanded, str(tmp_path), str(tmp_path), str(tmp_path / "hub"), False
        )

    wanted = {
        (entry["repo"], entry["filename"]) for entry in face_drift.required_weights({})
    }
    assert {face_drift.SFACE_REPO} <= {repo for repo, _ in wanted}
    assert len(wanted) == 2  # SFace and YuNet
    for check in (True, {"face_drop_max": 0.2}, {}):
        listed = {(entry["repo"], entry.get("filename")) for entry in downloads(check)}
        assert wanted <= listed, check
    for check in (None, False):
        assert downloads(check) == [], check


def test_check_refuses_unknown_face_key():
    assert "unknown" in chain_check_problem({"check": {"face": None}})
    # Lip sync left the segment check after calibration (ruling R9)
    assert "unknown" in chain_check_problem({"check": {"lip_sync_min": 0.3}})
    assert "unknown" in chain_check_problem({"check": {"lip_sync_min": None}})
    assert chain_check_problem({"check": {"face_drop_max": 0.2}}) is None
    assert chain_check_problem({"check": {"face_drop_max": None}}) is None
    assert chain_check_problem({"check": {"face_drop_max": "variable:s"}}) is None
    assert "face_drop_max" in chain_check_problem({"check": {"face_drop_max": 0}})


def test_the_schema_takes_face_drop_max_and_not_face_or_lip_sync():
    import json
    from pathlib import Path

    import jsonschema

    schema = json.loads(
        (Path(__file__).parent.parent / "dw/workflow_schema.json").read_text()
    )
    chain = {"$defs": schema["$defs"], **schema["$defs"]["chain"]}
    jsonschema.validate({"segments": 2, "check": {"face_drop_max": 0.2}}, chain)
    for key in ("face", "lip_sync_min"):
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate({"segments": 2, "check": {key: None}}, chain)
