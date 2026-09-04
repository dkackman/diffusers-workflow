"""
Unit tests for segment-chained pipeline execution.

A fake pipeline exposing _run_once stands in for a loaded model, returning
synthetic outputs shaped exactly like the real ones: pipeline output objects
with .frames, LTX-2 style objects with audio attributes, and modular dicts
keyed videos/audio/sampling_rate. No GPU is involved.
"""

from dataclasses import dataclass
from types import SimpleNamespace

import numpy
import pytest
import torch
from PIL import Image

from dw.pipeline_processors.chain import (
    ChainConfig,
    Segment,
    plan_segments,
    run_chain,
    snap_frames,
    validate_frame_snap,
)
from dw.result import AudioVideo

MINIMAX_SNAP = {"modulus": 17, "remainder": 5, "min_frames": 124, "max_frames": 345}


def solid_frame(color):
    return Image.new("RGB", (8, 8), color)


@dataclass
class FakeImageReference:
    image: object

    kind = "image"


@dataclass
class FakeAudioReference:
    audio: object
    sample_rate: int = None

    kind = "audio"


@dataclass
class FakeVideoReference:
    frames: object
    fps: float = None
    audio: object = None
    sample_rate: int = None

    kind = "video"


class FakePipeline:
    """Stands in for a loaded Pipeline wrapper - records every call."""

    def __init__(self, output_factory, output_dir=None, file_prefix=None):
        self.output_factory = output_factory
        self.output_dir = output_dir
        self.file_prefix = file_prefix
        self.calls = []

    def _run_once(self, arguments):
        self.calls.append(arguments)
        return self.output_factory(arguments, len(self.calls) - 1)


def video_output(arguments, index, num_frames=4):
    """A video pipeline output whose first frame reproduces its keyframe,
    the way image-to-video pipelines do."""
    color = (50 * index % 256, 100, 150)
    frames = [solid_frame(color) for _ in range(num_frames)]
    if "image" in arguments:
        frames[0] = arguments["image"]
    return SimpleNamespace(frames=[frames])


def modular_output(arguments, index, fps=4, sample_rate=100):
    """A modular pipeline's dict output - videos, audio, sampling_rate."""
    num_frames = arguments.get("num_frames", 8)
    color = (50 * index % 256, 100, 150)
    frames = [solid_frame(color) for _ in range(num_frames)]
    samples = int(num_frames / fps * sample_rate)
    audio = torch.full((1, 2, samples), float(index + 1))
    return {"videos": [frames], "audio": audio, "sampling_rate": sample_rate}


class TestSegmentsMode:
    def test_chains_the_requested_number_of_segments(self):
        pipeline = FakePipeline(video_output)

        result = run_chain(pipeline, {"segments": 3}, {"prompt": "test"})

        assert len(pipeline.calls) == 3
        assert isinstance(result, AudioVideo)

    def test_trims_the_duplicated_keyframe_from_each_later_segment(self):
        pipeline = FakePipeline(video_output)

        result = run_chain(pipeline, {"segments": 3, "trim_frames": 1}, {})

        # 4 frames, then 3 more per later segment with the keyframe copy dropped
        assert len(result.frames) == 4 + 3 + 3

    def test_trim_zero_keeps_every_frame(self):
        pipeline = FakePipeline(video_output)

        result = run_chain(pipeline, {"segments": 2, "trim_frames": 0}, {})

        assert len(result.frames) == 8

    def test_carries_the_last_frame_into_the_next_segment(self):
        pipeline = FakePipeline(video_output)

        result = run_chain(pipeline, {"segments": 2}, {})

        first_segment_last = pipeline.calls[1]["image"]
        assert isinstance(first_segment_last, Image.Image)
        # The carry is segment 0's last frame - and with trim, it never
        # appears twice in the output
        assert result.frames[3] is not result.frames[4]

    def test_the_first_segment_gets_no_injected_keyframe(self):
        pipeline = FakePipeline(video_output)

        run_chain(pipeline, {"segments": 2}, {})

        assert "image" not in pipeline.calls[0]

    def test_a_workflow_supplied_keyframe_reaches_the_first_segment(self):
        keyframe = solid_frame((1, 2, 3))
        pipeline = FakePipeline(video_output)

        run_chain(pipeline, {"segments": 2}, {"image": keyframe})

        assert pipeline.calls[0]["image"] is keyframe
        assert pipeline.calls[1]["image"] is not keyframe

    def test_per_segment_prompts_apply_in_order(self):
        pipeline = FakePipeline(video_output)
        chain = {"segments": 3, "prompts": ["one", "two", "three"]}

        run_chain(pipeline, chain, {"prompt": "unused"})

        assert [call["prompt"] for call in pipeline.calls] == ["one", "two", "three"]

    def test_the_last_prompt_covers_remaining_segments(self):
        pipeline = FakePipeline(video_output)

        run_chain(pipeline, {"segments": 3, "prompts": ["start", "rest"]}, {})

        assert [call["prompt"] for call in pipeline.calls] == ["start", "rest", "rest"]

    def test_the_original_arguments_are_not_mutated(self):
        pipeline = FakePipeline(video_output)
        arguments = {"prompt": "test"}

        run_chain(pipeline, {"segments": 2}, arguments)

        assert arguments == {"prompt": "test"}

    def test_a_video_only_chain_returns_no_audio(self):
        pipeline = FakePipeline(video_output)

        result = run_chain(pipeline, {"segments": 2}, {})

        assert result.audio is None


class TestGeneratedAudioJoining:
    def test_audio_shortens_in_step_with_the_trimmed_video(self):
        pipeline = FakePipeline(modular_output)
        chain = {"segments": 3, "trim_frames": 2, "fps": 4, "crossfade_ms": 250}

        result = run_chain(pipeline, chain, {"num_frames": 8})

        # video: 8 + 6 + 6 frames; audio must span exactly the same time
        assert len(result.frames) == 20
        assert result.audio.shape == (2, int(20 / 4 * 100))
        assert result.sample_rate == 100

    def test_the_crossfade_blends_across_each_seam(self):
        pipeline = FakePipeline(modular_output)
        chain = {"segments": 2, "trim_frames": 2, "fps": 4, "crossfade_ms": 250}

        result = run_chain(pipeline, chain, {"num_frames": 8})

        # Segment 0's audio is all 1.0, segment 1's all 2.0. The 250ms fade
        # window before the seam starts at segment 0's level and rises towards
        # segment 1's (equal-power ramps overshoot slightly on correlated
        # signals - that is expected).
        window = result.audio[0, 175:200]
        assert window[0] == pytest.approx(1.0)
        assert window[-1] > 1.9
        assert result.audio[0, 174] == pytest.approx(1.0)
        assert result.audio[0, 200] == pytest.approx(2.0)

    def test_mismatched_sample_rates_raise(self):
        def output(arguments, index):
            return modular_output(arguments, index, sample_rate=100 + index)

        pipeline = FakePipeline(output)

        with pytest.raises(ValueError, match="different sample rates"):
            run_chain(pipeline, {"segments": 2, "fps": 4}, {"num_frames": 8})

    def test_joining_audio_without_a_frame_rate_raises(self):
        pipeline = FakePipeline(modular_output)

        with pytest.raises(ValueError, match="frame rate"):
            run_chain(pipeline, {"segments": 2}, {"num_frames": 8})

    def test_the_frame_rate_argument_stands_in_for_chain_fps(self):
        pipeline = FakePipeline(modular_output)

        result = run_chain(
            pipeline, {"segments": 2}, {"num_frames": 8, "frame_rate": 4}
        )

        # default trim of 1: 8 + 7 frames -> the audio spans the same 15/4 s
        assert result.audio.shape[1] == int(15 / 4 * 100)


class TestMatchAudioMode:
    def make_arguments(self, samples=500, sample_rate=100, num_frames=8):
        subject = FakeImageReference(solid_frame((9, 9, 9)))
        voice = FakeAudioReference(torch.zeros(2, samples), sample_rate)
        return {
            "prompt": "test",
            "num_frames": num_frames,
            "references": [subject, voice],
        }

    def chain(self, **overrides):
        return {
            "match_audio": True,
            "fps": 4,
            "trim_frames": 2,
            "segment_argument": "references",
        } | overrides

    def test_the_video_length_matches_the_audio_duration(self):
        arguments = self.make_arguments()  # 5s at 4 fps -> 20 frames
        pipeline = FakePipeline(modular_output)

        result = run_chain(pipeline, self.chain(), arguments)

        assert len(result.frames) == 20
        # segments of 8 frames, minus 2 trimmed per later segment: 8+6+6
        assert len(pipeline.calls) == 3

    def test_the_final_audio_is_the_original_unsliced_track(self):
        arguments = self.make_arguments()
        pipeline = FakePipeline(modular_output)

        result = run_chain(pipeline, self.chain(), arguments)

        assert result.audio.shape == (2, 500)
        assert result.sample_rate == 100

    def test_each_segment_gets_its_frame_aligned_audio_slice(self):
        arguments = self.make_arguments()
        pipeline = FakePipeline(modular_output)

        run_chain(pipeline, self.chain(), arguments)

        slices = []
        for call in pipeline.calls:
            audio_refs = [r for r in call["references"] if r.kind == "audio"]
            assert len(audio_refs) == 1
            slices.append(audio_refs[0])

        # 8 frames at 4 fps and 100Hz -> 200 samples per slice; starts at
        # output frames 0, 6, 12 (each later slice backs up by the 2-frame trim)
        assert all(s.audio.shape[1] == 200 for s in slices)
        assert all(s.sample_rate == 100 for s in slices)

    def test_later_segments_carry_the_previous_frame_as_an_image_reference(self):
        arguments = self.make_arguments()
        pipeline = FakePipeline(modular_output)

        run_chain(pipeline, self.chain(), arguments)

        first_call_images = [
            r for r in pipeline.calls[0]["references"] if r.kind == "image"
        ]
        later_call_images = [
            r for r in pipeline.calls[1]["references"] if r.kind == "image"
        ]
        assert len(first_call_images) == 1
        # the workflow's own reference plus the carry frame - never more
        assert len(later_call_images) == 2
        assert (
            len([r for r in pipeline.calls[2]["references"] if r.kind == "image"]) == 2
        )

    def test_the_original_references_list_is_never_mutated(self):
        arguments = self.make_arguments()
        original_references = arguments["references"]
        pipeline = FakePipeline(modular_output)

        run_chain(pipeline, self.chain(), arguments)

        assert arguments["references"] is original_references
        assert len(original_references) == 2

    def test_audio_reaching_past_the_track_end_is_zero_padded(self):
        # 4.6s of audio -> 18.4 -> 18 frames. With a minimum segment length,
        # the final segment snaps up past the end of the track and its audio
        # slice is zero-padded to the full slice size.
        arguments = self.make_arguments(samples=460)
        arguments["references"][1].audio = torch.ones(2, 460)
        pipeline = FakePipeline(modular_output)
        chain = self.chain(frame_snap={"modulus": 1, "remainder": 0, "min_frames": 8})

        result = run_chain(pipeline, chain, arguments)

        assert len(result.frames) == 18
        last_slice = [r for r in pipeline.calls[-1]["references"] if r.kind == "audio"][
            0
        ]
        assert last_slice.audio.shape[1] == 200
        # real samples up to the track's end, silence past it
        assert last_slice.audio[:, :159].abs().min() == 1
        assert last_slice.audio[:, -40:].abs().max() == 0

    def test_match_audio_without_references_raises(self):
        pipeline = FakePipeline(modular_output)

        with pytest.raises(ValueError, match="references"):
            run_chain(pipeline, self.chain(), {"num_frames": 8})

    def test_match_audio_without_an_audio_reference_raises(self):
        arguments = self.make_arguments()
        arguments["references"] = [arguments["references"][0]]
        pipeline = FakePipeline(modular_output)

        with pytest.raises(ValueError, match="exactly one audio reference"):
            run_chain(pipeline, self.chain(), arguments)

    def test_match_audio_without_num_frames_raises(self):
        arguments = self.make_arguments()
        del arguments["num_frames"]
        pipeline = FakePipeline(modular_output)

        with pytest.raises(ValueError, match="num_frames"):
            run_chain(pipeline, self.chain(), arguments)

    def test_match_audio_without_a_frame_rate_raises(self):
        pipeline = FakePipeline(modular_output)
        chain = self.chain()
        del chain["fps"]

        with pytest.raises(ValueError, match="frame rate"):
            run_chain(pipeline, chain, self.make_arguments())

    def test_a_reference_without_a_sample_rate_raises(self):
        arguments = self.make_arguments()
        arguments["references"][1].sample_rate = None
        pipeline = FakePipeline(modular_output)

        with pytest.raises(ValueError, match="sample rate"):
            run_chain(pipeline, self.chain(), arguments)


class TestSaveSegments:
    """Incremental segment saving - segments spill to disk as they complete."""

    def chain(self, **overrides):
        return {
            "segments": 3,
            "trim_frames": 1,
            "fps": 4,
            "save_segments": True,
        } | overrides

    def make_pipeline(self, tmp_path, factory=video_output):
        return FakePipeline(factory, output_dir=str(tmp_path), file_prefix="wf-step")

    def test_each_segment_is_written_to_disk(self, tmp_path):
        pipeline = self.make_pipeline(tmp_path)

        run_chain(pipeline, self.chain(), {})

        files = sorted(tmp_path.glob("wf-step.0.segment-*.mp4"))
        assert len(files) == 3

    def test_the_frames_replay_from_disk(self, tmp_path):
        pipeline = self.make_pipeline(tmp_path)

        result = run_chain(pipeline, self.chain(), {})

        chunks = list(result.frames)
        assert len(chunks) == 3  # one tensor chunk per segment file
        assert sum(len(chunk) for chunk in chunks) == 4 + 3 + 3
        assert all(chunk.dtype == torch.uint8 for chunk in chunks)
        assert chunks[0].shape[1:] == (8, 8, 3)

    def test_cleanup_removes_the_segment_files(self, tmp_path):
        pipeline = self.make_pipeline(tmp_path)

        result = run_chain(pipeline, self.chain(), {})
        result.frames.cleanup()

        assert list(tmp_path.glob("*.mp4")) == []

    def test_cleanup_marks_the_frames_cleaned(self, tmp_path):
        # A cached Result pointing at these frames must not be served after
        # this - see Result.retainable, which duck-types on this attribute
        pipeline = self.make_pipeline(tmp_path)

        result = run_chain(pipeline, self.chain(), {})
        assert result.frames.cleaned is False
        result.frames.cleanup()

        assert result.frames.cleaned is True

    def test_keep_segments_survives_cleanup(self, tmp_path):
        pipeline = self.make_pipeline(tmp_path)

        result = run_chain(pipeline, self.chain(keep_segments=True), {})
        result.frames.cleanup()

        assert len(list(tmp_path.glob("*.mp4"))) == 3
        # the files are still there, so this cleanup() never counts as
        # cleaned - a cached Result stays retainable
        assert result.frames.cleaned is False

    def test_generated_audio_is_muxed_into_the_segment_files(self, tmp_path):
        import av

        # AAC only accepts standard sample rates - use one, unlike the other
        # tests' toy 100Hz
        def output(arguments, index):
            return modular_output(arguments, index, sample_rate=8000)

        pipeline = self.make_pipeline(tmp_path, output)

        run_chain(pipeline, self.chain(trim_frames=2), {"num_frames": 8})

        for path in sorted(tmp_path.glob("*.mp4")):
            with av.open(str(path)) as container:
                assert len(container.streams.audio) == 1

    def test_match_audio_tail_trims_during_replay(self, tmp_path):
        # 4.6s at 4 fps -> 18 output frames, but 20 are stored across the
        # segment files; the lazy replay stops at 18
        subject = FakeImageReference(solid_frame((9, 9, 9)))
        voice = FakeAudioReference(torch.zeros(2, int(4.6 * 8000)), 8000)
        arguments = {
            "num_frames": 8,
            "references": [subject, voice],
        }
        chain = self.chain(trim_frames=2, segment_argument="references")
        del chain["segments"]
        chain["match_audio"] = True

        def output(arguments, index):
            return modular_output(arguments, index, sample_rate=8000)

        pipeline = self.make_pipeline(tmp_path, output)

        result = run_chain(pipeline, chain, arguments)

        assert sum(len(chunk) for chunk in result.frames) == 18
        # the original track is still the soundtrack
        assert result.audio.shape == (2, int(4.6 * 8000))

    def test_repeated_chain_runs_use_distinct_files(self, tmp_path):
        # the same step chains once per cartesian iteration
        pipeline = self.make_pipeline(tmp_path)

        run_chain(pipeline, self.chain(), {})
        run_chain(pipeline, self.chain(), {})

        assert len(list(tmp_path.glob("wf-step.0.segment-*.mp4"))) == 3
        assert len(list(tmp_path.glob("wf-step.1.segment-*.mp4"))) == 3

    def test_a_rerun_does_not_clobber_a_previous_runs_segment_files(self, tmp_path):
        # A crashed or completed chain leaves salvageable segment-000.mp4
        # files behind; the per-wrapper iteration counter restarts at 0 in a
        # fresh process, so a naive path would silently overwrite them
        existing = tmp_path / "wf-step.0.segment-000.mp4"
        existing.write_bytes(b"salvageable")
        pipeline = self.make_pipeline(tmp_path)

        run_chain(pipeline, self.chain(), {})

        assert existing.read_bytes() == b"salvageable"
        deduped = sorted(tmp_path.glob("wf-step.0.segment-000*.mp4"))
        assert len(deduped) == 2

    def test_without_a_workflow_output_dir_raises(self):
        pipeline = FakePipeline(video_output)

        with pytest.raises(ValueError, match="output directory"):
            run_chain(pipeline, self.chain(), {})

    def test_without_a_frame_rate_raises(self, tmp_path):
        pipeline = self.make_pipeline(tmp_path)
        chain = self.chain()
        del chain["fps"]

        with pytest.raises(ValueError, match="frame rate"):
            run_chain(pipeline, chain, {})


class TestLastSegmentContinuity:
    def make_arguments(self, num_frames=8):
        subject = FakeImageReference(solid_frame((9, 9, 9)))
        return {"prompt": "test", "num_frames": num_frames, "references": [subject]}

    def chain(self, **overrides):
        return {
            "segments": 3,
            "continuity": "last_segment",
            "segment_argument": "references",
            "fps": 4,
            "trim_frames": 2,
        } | overrides

    def carried(self, call):
        return [r for r in call["references"] if r.kind == "video"]

    def test_later_segments_carry_the_previous_segment_as_a_video_reference(self):
        pipeline = FakePipeline(modular_output)

        run_chain(pipeline, self.chain(), self.make_arguments())

        assert self.carried(pipeline.calls[0]) == []
        # the workflow's own reference plus one carry - never more
        for call in pipeline.calls[1:]:
            assert len(call["references"]) == 2
            assert len(self.carried(call)) == 1

    def test_the_carry_holds_the_whole_segment_and_its_soundtrack(self):
        pipeline = FakePipeline(modular_output)

        run_chain(pipeline, self.chain(), self.make_arguments())

        carry = self.carried(pipeline.calls[1])[0]
        assert len(carry.frames) == 8
        # 8 frames at 4 fps and 100Hz - the segment's own generated audio
        assert carry.audio.shape == (2, 200)
        assert carry.sample_rate == 100

    def test_carry_frames_limits_the_carry_to_the_tail(self):
        pipeline = FakePipeline(modular_output)

        run_chain(pipeline, self.chain(carry_frames=4), self.make_arguments())

        carry = self.carried(pipeline.calls[1])[0]
        assert len(carry.frames) == 4
        # the soundtrack is cut to the same span, 4 frames at 4 fps and 100Hz
        assert carry.audio.shape == (2, 100)

    def test_a_carry_longer_than_the_segment_keeps_every_frame(self):
        pipeline = FakePipeline(modular_output)

        run_chain(pipeline, self.chain(carry_frames=99), self.make_arguments())

        assert len(self.carried(pipeline.calls[1])[0].frames) == 8

    def test_carry_audio_false_carries_motion_alone(self):
        pipeline = FakePipeline(modular_output)

        run_chain(pipeline, self.chain(carry_audio=False), self.make_arguments())

        carry = self.carried(pipeline.calls[1])[0]
        assert carry.audio is None
        assert len(carry.frames) == 8

    def test_an_existing_video_reference_names_the_carry_type(self):
        arguments = self.make_arguments()
        arguments["references"] = [FakeVideoReference([solid_frame((1, 1, 1))], 24.0)]
        pipeline = FakePipeline(modular_output)

        run_chain(pipeline, self.chain(), arguments)

        assert len(self.carried(pipeline.calls[1])) == 2

    def test_the_original_references_list_is_never_mutated(self):
        arguments = self.make_arguments()
        original_references = arguments["references"]
        pipeline = FakePipeline(modular_output)

        run_chain(pipeline, self.chain(), arguments)

        assert arguments["references"] is original_references
        assert len(original_references) == 1

    def test_a_segment_argument_that_is_not_a_list_raises(self):
        pipeline = FakePipeline(modular_output)

        with pytest.raises(ValueError, match="references list"):
            run_chain(
                pipeline,
                self.chain(segment_argument="image"),
                {"prompt": "test", "num_frames": 8},
            )

    def test_an_empty_references_list_raises(self):
        pipeline = FakePipeline(modular_output)

        with pytest.raises(ValueError, match="empty references list"):
            run_chain(
                pipeline,
                self.chain(),
                {"prompt": "test", "num_frames": 8, "references": []},
            )


class TestValidation:
    def test_segments_and_match_audio_together_raise(self):
        with pytest.raises(ValueError, match="exactly one of"):
            ChainConfig({"segments": 2, "match_audio": True}, {})

    def test_a_chain_without_a_length_raises(self):
        with pytest.raises(ValueError, match="exactly one of"):
            ChainConfig({}, {})

    def test_an_unknown_continuity_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown chain continuity"):
            ChainConfig({"segments": 2, "continuity": "teleport"}, {})

    def test_a_zero_carry_frames_raises(self):
        with pytest.raises(ValueError, match="carry_frames"):
            ChainConfig({"segments": 2, "carry_frames": 0}, {})

    def test_a_zero_segment_chain_raises(self):
        with pytest.raises(ValueError, match="between 1 and"):
            ChainConfig({"segments": 0}, {})

    def test_num_frames_violating_the_frame_constraint_raises(self):
        with pytest.raises(ValueError, match="frame\\s+constraint"):
            ChainConfig(
                {"segments": 2, "frame_snap": MINIMAX_SNAP}, {"num_frames": 120}
            )

    def test_batched_generation_raises(self):
        def batched(arguments, index):
            frames = [solid_frame((0, 0, 0)) for _ in range(4)]
            return SimpleNamespace(frames=[frames, list(frames)])

        pipeline = FakePipeline(batched)

        with pytest.raises(ValueError, match="exactly one video"):
            run_chain(pipeline, {"segments": 2}, {})


class TestPlanSegments:
    def test_an_exact_cover_with_full_segments(self):
        plan = plan_segments(30, 12, 1)

        assert [s.num_frames for s in plan] == [12, 12, 8]
        assert [s.head_trim for s in plan] == [0, 1, 1]
        assert [s.audio_start_frame for s in plan] == [0, 11, 22]
        assert sum(s.num_frames - s.head_trim for s in plan) == 30

    def test_minimax_frame_counts_snap_to_valid_lengths(self):
        plan = plan_segments(300, 124, 2, MINIMAX_SNAP)

        assert all((s.num_frames - 5) % 17 == 0 for s in plan)
        assert all(124 <= s.num_frames <= 345 for s in plan)
        assert sum(s.num_frames - s.head_trim for s in plan) >= 300

    def test_a_single_segment_covers_short_totals(self):
        plan = plan_segments(5, 12, 1)

        assert len(plan) == 1
        assert plan[0].num_frames == 5

    def test_short_totals_still_snap_up_to_the_minimum(self):
        plan = plan_segments(50, 124, 2, MINIMAX_SNAP)

        assert len(plan) == 1
        assert plan[0].num_frames == 124

    def test_a_segment_shorter_than_the_trim_raises(self):
        with pytest.raises(ValueError, match="cannot progress"):
            plan_segments(100, 2, 2)

    def test_an_invalid_segment_length_raises(self):
        with pytest.raises(ValueError, match="frame\\s+constraint"):
            plan_segments(300, 120, 2, MINIMAX_SNAP)


class TestSnapFrames:
    def test_no_constraint_passes_the_count_through(self):
        assert snap_frames(37, None) == 37

    def test_counts_snap_up_to_the_next_valid_value(self):
        assert snap_frames(130, MINIMAX_SNAP) == 141  # 17*8+5

    def test_a_valid_count_is_unchanged(self):
        assert snap_frames(124, MINIMAX_SNAP) == 124

    def test_counts_below_the_minimum_snap_to_it(self):
        assert snap_frames(10, MINIMAX_SNAP) == 124

    def test_a_count_past_the_maximum_raises(self):
        with pytest.raises(ValueError, match="max_frames"):
            snap_frames(350, MINIMAX_SNAP)

    def test_validate_accepts_a_valid_count(self):
        validate_frame_snap(345, MINIMAX_SNAP)

    def test_validate_rejects_wrong_modulus(self):
        with pytest.raises(ValueError):
            validate_frame_snap(125, MINIMAX_SNAP)
