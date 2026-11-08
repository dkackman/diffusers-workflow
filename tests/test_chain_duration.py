"""A chain asked for by its length in seconds (`duration_seconds`)."""

import copy
import logging

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
        from dw.variable_constraints import snap_block

        name = chain["frame_snap"].removeprefix("constraint:")
        chain["frame_snap"] = snap_block(definition["variable_constraints"][name])
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


def _ltx_without_frame_snap():
    definition = copy.deepcopy(_template("ltx2"))
    del definition["steps"][0]["pipeline"]["chain"]["frame_snap"]
    return definition


def _expanded_chain(definition, caplog=None, **arguments):
    from dw.workflow import Workflow

    workflow = Workflow(definition, None, "")
    if caplog is None:
        expanded = workflow.expanded_definition(arguments)
    else:
        with caplog.at_level(logging.INFO, logger="dw"):
            expanded = workflow.expanded_definition(arguments)
    pipeline = expanded["steps"][0]["pipeline"]
    return pipeline["chain"], pipeline["arguments"]


LTX_SNAP = {"modulus": 8, "remainder": 1, "min_frames": 9}


class TestDerivedFrameSnap:
    """A chain planned by length whose block declares no `frame_snap` takes
    the one its `num_frames` variable's constraint declares, so LTX's
    shortened last segment lands on the 8n+1 grid rather than being floored
    by the pipeline (6.5 s at 24 fps: 156 frames, 121 then 37 -> 41)."""

    def test_the_ltx_template_names_its_constraint_as_the_snap(self):
        template = _template("ltx2")
        chain = template["steps"][0]["pipeline"]["chain"]
        assert chain["frame_snap"] == "constraint:num_frames"
        chain, arguments = _expanded_chain(template, duration_seconds=6.5)
        assert chain["frame_snap"] == LTX_SNAP
        config = ChainConfig(chain, arguments)
        assert config.total_frames == 156
        assert [s.num_frames for s in config.plan] == [121, 41]
        assert sum(s.num_frames - s.head_trim for s in config.plan) >= 156

    def test_a_block_with_no_frame_snap_derives_it_and_says_so(self, caplog):
        chain, arguments = _expanded_chain(
            _ltx_without_frame_snap(), caplog, duration_seconds=6.5
        )
        assert chain["frame_snap"] == LTX_SNAP
        config = ChainConfig(chain, arguments)
        assert [s.num_frames for s in config.plan] == [121, 41]
        lines = [
            r.getMessage()
            for r in caplog.records
            if "frame_snap" in r.getMessage() and "num_frames" in r.getMessage()
        ]
        assert len(lines) == 1, caplog.text
        assert "chained_image_to_video" in lines[0]

    def test_without_the_derivation_the_last_segment_is_off_grid(self):
        # What the fix guards against: the plan LTX would floor to 33 frames
        definition = _ltx_without_frame_snap()
        chain, arguments = _resolved(definition, duration_seconds=6.5)
        assert [s.num_frames for s in ChainConfig(chain, arguments).plan] == [121, 37]

    def test_the_derivation_is_pure_and_leaves_other_chains_alone(self):
        from dw.pipeline_processors.chain_config import derive_frame_snaps

        def step(chain, num_frames):
            return {
                "variable_constraints": {
                    "num_frames": LTX_SNAP,
                    "width": {"modulus": 32, "remainder": 0},
                },
                "steps": [
                    {
                        "name": "clip",
                        "pipeline": {
                            "chain": chain,
                            "arguments": {"num_frames": num_frames},
                        },
                    }
                ],
            }

        derived = step({"duration_seconds": 4, "fps": 24}, "variable:num_frames")
        assert derive_frame_snaps(derived) == [("clip", "num_frames")]
        assert derived["steps"][0]["pipeline"]["chain"]["frame_snap"] == (
            "constraint:num_frames"
        )
        matched = step({"match_audio": True, "fps": 24}, "variable:num_frames")
        assert derive_frame_snaps(matched) == [("clip", "num_frames")]
        # A literal count, a variable with no constraint, a declared snap and
        # a chain counted by segments are each left as written
        for definition in (
            step({"duration_seconds": 4, "fps": 24}, 121),
            step({"duration_seconds": 4, "fps": 24}, "variable:frames"),
            step(
                {"duration_seconds": 4, "frame_snap": MINIMAX_SNAP},
                "variable:num_frames",
            ),
            step({"segments": 3}, "variable:num_frames"),
        ):
            before = copy.deepcopy(definition)
            assert derive_frame_snaps(definition) == []
            assert definition == before

    def test_validate_warns_on_a_length_planned_chain_it_cannot_snap(self):
        from dw import validation
        from dw.workflow import Workflow

        def workflow(chain, num_frames=121):
            return Workflow(
                {
                    "steps": [
                        {
                            "name": "clip",
                            "pipeline": {
                                "configuration": {"component_type": "ModularPipeline"},
                                "from_pretrained_arguments": {
                                    "model_name": "MiniMaxAI/MiniMax-H3",
                                    "workflow": "fl2va",
                                },
                                "chain": chain,
                                "arguments": {
                                    "prompt": "a cat",
                                    "num_frames": num_frames,
                                },
                            },
                        }
                    ]
                },
                None,
                "",
            )

        warnings = validation.run_warning_check(
            workflow({"duration_seconds": 4, "fps": 24}), "frame_snap_warnings", None
        )
        assert len(warnings) == 1
        assert warnings[0].startswith("steps[0].pipeline.chain.frame_snap: ")
        assert (
            "duration_seconds" in warnings[0] and "variable_constraints" in warnings[0]
        )
        assert (
            validation.run_warning_check(
                workflow({"segments": 3}), "frame_snap_warnings", None
            )
            == []
        )
        assert (
            validation.run_warning_check(
                workflow(
                    {"duration_seconds": 4, "fps": 24, "frame_snap": MINIMAX_SNAP}
                ),
                "frame_snap_warnings",
                None,
            )
            == []
        )
        # The template declares (or derives) its snap, so it is not warned on
        for definition in (_template("ltx2"), _ltx_without_frame_snap()):
            assert (
                validation.run_warning_check(
                    Workflow(definition, None, ""),
                    "frame_snap_warnings",
                    {"duration_seconds": 6.5},
                )
                == []
            )
