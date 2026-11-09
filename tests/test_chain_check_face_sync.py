"""The chain check's identity and lip-sync measures (chain_check.py): face_drop,
segment 1's median face similarity to its own frame 0 minus a later
segment's, and lip_sync, the mouth-motion to audio-envelope correlation.
Both are warn-only and off by default: measured and recorded, never failed,
until a ceiling is set - and a set ceiling fails without a re-roll.

The models are replaced at chain_check's own seams (`_face_scores`,
`_sync_score`), so nothing here reads weights."""

import numpy as np
import pytest
from PIL import Image

import dw.pipeline_processors.chain_check as cc
from dw.pipeline_processors.chain import run_chain
from dw.pipeline_processors.chain_check import (
    MEASURES,
    WARN_ONLY,
    CheckResult,
    SegmentCheck,
    face_base,
    segment_check,
    with_face_sync,
)
from dw.pipeline_processors.chain_rules import chain_check_problem
from tests.test_chain import FakePipeline, modular_output


def _frames(level, count=8):
    return [Image.new("RGB", (32, 32), (level, level, level)) for _ in range(count)]


def _scores_by_level(table):
    """A `_face_scores` stand-in: a segment's scores read off its grey level."""

    def scores(frames, reference, samples):
        return list(table[frames[0].getpixel((0, 0))[0]])

    return scores


def _sync(correlation):
    def score(frames, waveform, sample_rate, fps):
        return {"correlation": correlation, "lag_seconds": 0.0, "frames_with_face": 8}

    return score


def _passed():
    return CheckResult(True, {}, [])


def test_face_drop_is_measured_and_warn_only(monkeypatch):
    monkeypatch.setattr(
        cc, "_face_scores", _scores_by_level({100: [0.9] * 8, 120: [0.5] * 8})
    )
    base = face_base(_frames(100))
    result = with_face_sync(
        _passed(), _frames(120), base, None, None, None, SegmentCheck()
    )
    assert result.measures["face_drop"] == pytest.approx(0.4)
    assert result.passed and result.failed == []
    assert "face_drop" in MEASURES and "face_drop" in WARN_ONLY


def test_no_face_in_segment_1_leaves_face_drop_unmeasured(monkeypatch):
    monkeypatch.setattr(cc, "_face_scores", lambda frames, reference, samples: [])
    assert face_base(_frames(100)) is None
    result = with_face_sync(
        _passed(), _frames(120), None, None, None, None, SegmentCheck()
    )
    assert result.measures["face_drop"] is None and result.passed


def test_a_model_that_cannot_load_records_none(monkeypatch):
    def broken(*args):
        raise RuntimeError("no YuNet in this cv2")

    monkeypatch.setattr(cc, "_face_scores", broken)
    monkeypatch.setattr(cc, "_sync_score", broken)
    assert face_base(_frames(100)) is None
    audio = np.zeros((1, 800), dtype=np.float32)
    result = with_face_sync(
        _passed(), _frames(120), None, audio, 400, 4, SegmentCheck(lip_sync_min=0.5)
    )
    assert result.measures["lip_sync"] is None and result.passed


def test_lip_sync_is_measured_and_warn_only(monkeypatch):
    monkeypatch.setattr(cc, "_sync_score", _sync(0.2))
    audio = np.zeros((1, 800), dtype=np.float32)
    result = with_face_sync(
        _passed(), _frames(120), None, audio, 400, 4, SegmentCheck()
    )
    assert result.measures["lip_sync"] == 0.2
    assert result.measures["lip_sync_lag_seconds"] == 0.0
    assert result.passed
    assert "lip_sync" in MEASURES and "lip_sync" in WARN_ONLY

    floored = with_face_sync(
        _passed(), _frames(120), None, audio, 400, 4, SegmentCheck(lip_sync_min=0.5)
    )
    assert floored.failed == ["lip_sync"] and not floored.passed
    assert not floored.rerolls


def test_no_audio_leaves_lip_sync_unmeasured(monkeypatch):
    def never(*args):
        raise AssertionError("measured with no audio")

    monkeypatch.setattr(cc, "_sync_score", never)
    result = with_face_sync(
        _passed(), _frames(120), None, None, None, 4, SegmentCheck()
    )
    assert result.measures["lip_sync"] is None


def test_face_drop_max_sets_a_failing_ceiling_without_reroll(monkeypatch):
    monkeypatch.setattr(
        cc, "_face_scores", _scores_by_level({100: [0.9] * 8, 250: [0.5] * 8})
    )
    monkeypatch.setattr(cc, "_sync_score", _sync(0.7))
    check = segment_check({"face_drop_max": 0.1, "jump_ratio_max": 1000})
    assert check.face_drop_max == 0.1 and check.lip_sync_min is None

    def factory(arguments, index):
        out = modular_output(arguments, index, sample_rate=100)
        level = 100 if index == 0 else 250
        out["videos"] = [
            [Image.new("RGB", (32, 32), (level,) * 3) for _ in out["videos"][0]]
        ]
        return out

    pipeline = FakePipeline(factory)
    chain = {
        "segments": 2,
        "fps": 4,
        "trim_frames": 2,
        "check": {"face_drop_max": 0.1, "jump_ratio_max": 1000, "look_ratio_max": 1000},
        "retries": 1,
    }
    result = run_chain(pipeline, chain, {"num_frames": 8, "seed": 5})
    shot = result.shots[1]
    # Warn-only: failed and recorded, but never re-rolled
    assert len(pipeline.calls) == 2 and shot["attempts"] == 1
    assert shot["check_failed"] == ["face_drop"]
    assert shot["check"]["face_drop"] == pytest.approx(0.4)
    assert shot["check"]["lip_sync"] == 0.7


def test_check_refuses_unknown_face_key():
    assert "unknown" in chain_check_problem({"check": {"face": None}})
    assert chain_check_problem({"check": {"face_drop_max": 0.2}}) is None
    assert chain_check_problem({"check": {"lip_sync_min": 0.3}}) is None
    assert (
        chain_check_problem({"check": {"face_drop_max": None, "lip_sync_min": None}})
        is None
    )
    assert chain_check_problem({"check": {"lip_sync_min": "variable:s"}}) is None
    assert "face_drop_max" in chain_check_problem({"check": {"face_drop_max": 0}})
    assert "lip_sync_min" in chain_check_problem({"check": {"lip_sync_min": 2}})
    assert "lip_sync_min" in chain_check_problem({"check": {"lip_sync_min": True}})


def test_the_schema_takes_the_new_keys_and_not_face():
    import json
    from pathlib import Path

    import jsonschema

    schema = json.loads(
        (Path(__file__).parent.parent / "dw/workflow_schema.json").read_text()
    )
    chain = {"$defs": schema["$defs"], **schema["$defs"]["chain"]}
    jsonschema.validate(
        {"segments": 2, "check": {"face_drop_max": 0.2, "lip_sync_min": None}}, chain
    )
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"segments": 2, "check": {"face": None}}, chain)
