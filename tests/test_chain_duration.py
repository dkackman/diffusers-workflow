"""A chain asked for by its length in seconds (`duration_seconds`)."""

import pytest

from dw.cost_scaling import _driver_scaling
from dw.pipeline_processors.chain import ChainConfig, run_chain
from dw.pipeline_processors.chain_config import chain_segment_count
from dw.task_domains import frames_to_samples

from tests.test_chain import MINIMAX_SNAP, FakePipeline, modular_output


def chain(**overrides):
    return {"duration_seconds": 4.5, "fps": 4, "trim_frames": 2} | overrides


def test_duration_plans_whole_segments_and_trims_to_the_asked_length():
    # 40 s at 24 fps with 124-frame guide segments
    definition = {
        "duration_seconds": 40,
        "fps": 24,
        "continuity": "guide",
        "guide_frames": 22,
        "frame_snap": MINIMAX_SNAP,
    }
    config = ChainConfig(definition, {"num_frames": 124})
    assert config.total_frames == 960
    produced = sum(s.num_frames - s.head_trim for s in config.plan)
    assert produced >= 960

    # And the run: 4.5 s at 4 fps is 18 frames, from 8 + 6 + 6 generated
    pipeline = FakePipeline(modular_output)
    result = run_chain(pipeline, chain(), {"num_frames": 8})
    assert len(pipeline.calls) == 3
    assert len(result.frames) == 18
    assert result.audio.shape[1] == frames_to_samples(18, 4, 100) == 450
    assert sum(s["num_frames"] for s in result.shots) == 18


def test_exactly_one_of_segments_match_audio_duration():
    with pytest.raises(ValueError, match="exactly one of"):
        ChainConfig({"fps": 4}, {"num_frames": 8})
    # The duration wins over a segment count (R18); the run says so
    assert len(ChainConfig(chain(segments=2), {"num_frames": 8}).plan) == 3
    with pytest.raises(ValueError, match="exactly one of"):
        ChainConfig(chain(match_audio=True), {"num_frames": 8})
    with pytest.raises(ValueError, match="exactly one of"):
        ChainConfig(chain(match_audio=True, segments=2), {"num_frames": 8})
    # A null duration is absent
    assert len(ChainConfig(chain(duration_seconds=None, segments=2), {}).plan) == 2


def test_duration_needs_a_frame_rate_and_num_frames_and_a_positive_length():
    with pytest.raises(ValueError, match="frame rate"):
        ChainConfig({"duration_seconds": 4}, {"num_frames": 8})
    with pytest.raises(ValueError, match="num_frames"):
        ChainConfig(chain(), {})
    with pytest.raises(ValueError, match="greater than 0"):
        ChainConfig(chain(duration_seconds=0), {"num_frames": 8})


def test_the_frame_rate_may_come_from_the_pipeline_argument():
    config = ChainConfig({"duration_seconds": 4.5}, {"num_frames": 8, "frame_rate": 4})
    assert config.total_frames == 18


def test_duration_with_frame_snap_snaps_the_last_segment_up():
    definition = {"duration_seconds": 10, "fps": 24, "trim_frames": 2}
    definition["frame_snap"] = MINIMAX_SNAP
    config = ChainConfig(definition, {"num_frames": 124})
    # 240 frames: 124, then 122 more would need 120 + 2 = 122 -> snapped to 124
    assert [s.num_frames for s in config.plan] == [124, 124]
    assert all((s.num_frames - 5) % 17 == 0 for s in config.plan)


@pytest.mark.parametrize(
    "extra, match",
    [
        ({"continuity": "guide", "guide_backoff": 17}, "match_audio"),
        ({"carry_pick": {"sharpest": 3}}, "duration_seconds"),
    ],
)
def test_duration_refuses_the_trims_an_up_front_plan_cannot_count(extra, match):
    with pytest.raises(ValueError, match=match):
        ChainConfig(chain(**extra), {"num_frames": 124})


def test_chain_segment_count_is_pure_and_shared():
    assert chain_segment_count({"segments": 5}, {}) == 5
    assert chain_segment_count(chain(), {"num_frames": 8}) == 3
    assert chain_segment_count(chain(), {}) is None
    assert chain_segment_count({"match_audio": True}, {}) is None
    assert chain_segment_count({"segments": "variable:x"}, {}) is None
    assert chain_segment_count(chain(duration_seconds=None, segments=2), {}) == 2
    # The duration wins when both resolve
    assert chain_segment_count(chain(segments=2), {"num_frames": 8}) == 3


def _definition(segments, duration):
    return {
        "cost_drivers": ["segments", "duration_seconds"],
        "variables": {
            "segments": segments,
            "duration_seconds": duration,
            "num_frames": 8,
        },
        "steps": [
            {
                "pipeline": {
                    "chain": {
                        "segments": "variable:segments",
                        "duration_seconds": "variable:duration_seconds",
                        "fps": 4,
                        "trim_frames": 2,
                    },
                    "arguments": {"num_frames": "variable:num_frames"},
                }
            }
        ],
    }


def test_the_estimate_scales_by_the_segment_count_duration_implies():
    definition = _definition(3, None)
    # 4.5 s -> 3 segments, the default: unchanged and nothing flagged
    same = {
        "variables": {
            **definition["variables"],
            "segments": None,
            "duration_seconds": 4.5,
        }
    }
    factor, axes, unscaled = _driver_scaling(definition, same, {})
    assert factor == pytest.approx(1.0) and unscaled == []
    # 12.5 s at 4 fps = 50 frames -> 8, then 6 each: 1 + ceil(42/6) = 8 segments
    longer = {
        "variables": {
            **definition["variables"],
            "segments": None,
            "duration_seconds": 12.5,
        }
    }
    factor, axes, unscaled = _driver_scaling(definition, longer, {})
    assert factor == pytest.approx(8 / 3)
    assert unscaled == [] and any("segments 8 vs 3" in a for a in axes)
    # And a plain segment count scales the same way
    five = {"variables": {**definition["variables"], "segments": 6}}
    factor, _, _ = _driver_scaling(definition, five, {})
    assert factor == pytest.approx(2.0)


TEMPLATES = ("minimax", "ltx2")


def _template(family):
    import json
    from pathlib import Path

    path = (
        Path(__file__).parent.parent
        / f"workflows/templates/{family}/chained-segments.json"
    )
    return json.loads(path.read_text())


def _resolved(definition, **overrides):
    from dw.variables import replace_variables

    variables = {**definition["variables"], **overrides}
    step = replace_variables(definition["steps"][0], variables)
    chain = step["pipeline"]["chain"]
    if isinstance(chain.get("frame_snap"), str):
        # The engine resolves 'constraint:' before the chain runs
        chain["frame_snap"] = MINIMAX_SNAP
    return chain, step["pipeline"]["arguments"]


@pytest.mark.parametrize("family", TEMPLATES)
def test_a_template_with_duration_null_keeps_its_segments(family):
    chain, arguments = _resolved(_template(family))
    assert "duration_seconds" not in chain
    assert len(ChainConfig(chain, arguments).plan) == 3


@pytest.mark.parametrize("family", TEMPLATES)
def test_a_template_asked_for_seconds_plans_by_length(family):
    chain, arguments = _resolved(_template(family), segments=None, duration_seconds=20)
    config = ChainConfig(chain, arguments)
    assert config.total_frames == 480
    assert len(config.plan) == chain_segment_count(chain, arguments) >= 4


@pytest.mark.parametrize("family", TEMPLATES)
def test_a_template_estimate_scales_by_the_implied_segment_count(family):
    definition = _template(family)
    ask = {
        "variables": {
            **definition["variables"],
            "segments": None,
            "duration_seconds": 20,
        }
    }
    factor, axes, _ = _driver_scaling(definition, ask, {})
    chain, arguments = _resolved(definition, segments=None, duration_seconds=20)
    assert factor == pytest.approx(chain_segment_count(chain, arguments) / 3)
    assert any(axis.startswith("segments") for axis in axes)


def test_duration_overrides_the_templates_default_segments(monkeypatch):
    from dw.pipeline_processors import chain_config as chain_module

    logged = []
    monkeypatch.setattr(
        chain_module, "emit_log", lambda message, **data: logged.append(message)
    )
    definition = _template("minimax")
    chain_block, arguments = _resolved(definition, duration_seconds=20)
    # Only the duration was passed; segments is still the template's 3
    assert chain_block["segments"] == 3

    config = ChainConfig(chain_block, arguments)
    assert config.total_frames == 480
    assert len(config.plan) == chain_segment_count(chain_block, arguments) > 3
    produced = sum(s.num_frames - s.head_trim for s in config.plan)
    assert produced >= 480
    assert any("segments" in line and "ignored" in line for line in logged)

    ask = {"variables": {**definition["variables"], "duration_seconds": 20}}
    factor, axes, _ = _driver_scaling(definition, ask, {})
    assert factor == pytest.approx(len(config.plan) / 3)
    assert any(axis.startswith("segments") for axis in axes)


def test_a_spilled_duration_chain_trims_frames_and_audio(tmp_path):
    # A rate the segment files' AAC track can be written at
    pipeline = FakePipeline(
        lambda arguments, index: modular_output(arguments, index, sample_rate=8000),
        output_dir=str(tmp_path),
        file_prefix="wf-step",
    )
    result = run_chain(pipeline, chain(save_segments=True), {"num_frames": 8})

    assert sum(len(chunk) for chunk in result.frames) == 18
    assert result.audio.shape[1] == frames_to_samples(18, 4, 8000) == 36000
    result.frames.cleanup()


def _schema_accepts(chain_block):
    import json
    from pathlib import Path

    import jsonschema

    import dw

    schema = json.loads((Path(dw.__file__).parent / "workflow_schema.json").read_text())
    defs = schema["$defs"]
    name = next(
        k for k, v in defs.items() if "duration_seconds" in v.get("properties", {})
    )
    validator = jsonschema.Draft202012Validator(
        {"$ref": f"#/$defs/{name}", "$defs": defs}
    )
    return not list(validator.iter_errors(chain_block))


def test_the_schema_does_not_refuse_match_audio_beside_a_null_or_reference_duration():
    # presence-based, it refused a duration that substitutes to null (#1232)
    assert _schema_accepts({"match_audio": True, "duration_seconds": None})
    assert _schema_accepts({"match_audio": True, "duration_seconds": "variable:d"})
    assert not _schema_accepts({"match_audio": True, "duration_seconds": 4})
    assert not _schema_accepts({"match_audio": True, "segments": 3})


def test_the_shortened_last_segment_renders_at_its_planned_length():
    # 18 frames from 8 + 6 + 6: the last segment is asked for its 6, not the
    # step's 8, so no GPU time goes on frames the trim then drops
    pipeline = FakePipeline(modular_output)
    run_chain(pipeline, chain(), {"num_frames": 8})
    assert [call["num_frames"] for call in pipeline.calls] == [8, 8, 6]


def test_a_frame_slot_past_the_shortened_last_segment_is_refused_up_front():
    # 205 frames of 175-frame guide segments: the last needs 52 and snaps up
    # to the 124 minimum, which frame 136 is past - refused before segment 1
    definition = {
        "duration_seconds": 205 / 24,
        "fps": 24,
        "continuity": "guide",
        "frame_snap": MINIMAX_SNAP,
        "keystone": "segment_1",
        "keystone_slot": {"frame": 136},
    }
    with pytest.raises(ValueError, match="frame 136 is past segment 2's 124"):
        ChainConfig(definition, {"num_frames": 175, "image": object()})
