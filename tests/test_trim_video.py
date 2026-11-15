"""trim_video: keep a span of a video's frames and its audio (#627)."""

from unittest.mock import patch

import numpy
import pytest
from PIL import Image

from dw.media_types import AudioVideo
from dw.shots import named_shots, shot_record, trimmed_shots
from dw.tasks.concat_videos import concat_videos
from dw.tasks.task import _COMMAND_REGISTRY
from dw.tasks.trim import span_problem, trim_video
from dw.task_domains import frames_to_samples


def frames(count):
    return [Image.new("RGB", (4, 4), (index, 0, 0)) for index in range(count)]


def clip(count, fps=24, sample_rate=48000, shots=None):
    samples = frames_to_samples(count, fps or 24, sample_rate)
    audio = numpy.arange(samples, dtype=numpy.float32).reshape(1, -1)
    return AudioVideo(frames(count), audio, sample_rate, fps=fps, shots=shots)


class TestFramesAndAudio:
    def test_keeps_exactly_the_span(self):
        result = trim_video(clip(48), 10, 20)
        assert len(result.frames) == 20
        assert [f.getpixel((0, 0))[0] for f in result.frames] == list(range(10, 30))
        assert result.fps == 24 and result.sample_rate == 48000

    @pytest.mark.parametrize("rate", [48000, 44100])
    def test_audio_matches_the_span_at_its_own_rate(self, rate):
        source = clip(48, fps=24, sample_rate=rate)
        result = trim_video(source, 7, 13)
        first = frames_to_samples(7, 24, rate)
        last = frames_to_samples(20, 24, rate)
        assert result.audio.shape == (1, last - first)
        assert result.audio[0, 0] == first
        assert result.sample_rate == rate

    def test_consecutive_trims_tile_the_track(self):
        source = clip(30, sample_rate=44100)
        a, b = trim_video(source, 0, 11), trim_video(source, 11, 19)
        joined = numpy.concatenate([a.audio, b.audio], axis=1)
        assert numpy.array_equal(joined, source.audio)

    def test_whole_clip(self):
        source = clip(24)
        result = trim_video(source, 0, 24)
        assert len(result.frames) == 24
        assert numpy.array_equal(result.audio, source.audio)

    def test_last_frame_alone(self):
        result = trim_video(clip(24), 23, 1)
        assert len(result.frames) == 1
        assert result.audio.shape[1] == 2000

    def test_plain_frame_list_gives_one_silent_clip(self):
        # A bare frame list is N results to the step, each saved as its own
        # "video" - one PIL frame handed to export_to_video (#1199)
        result = trim_video(frames(10), 2, 3, fps=12)
        assert isinstance(result, AudioVideo)
        assert len(result.frames) == 3 and result.audio is None
        assert result.fps == 12
        assert result.shots == [shot_record("video 1", 0, 3)]

    def test_audio_without_fps_is_refused(self):
        source = clip(24, fps=None)
        with pytest.raises(ValueError, match="fps"):
            trim_video(source, 0, 5)
        assert len(trim_video(source, 0, 5, fps=24).frames) == 5

    def test_silent_audio_video_stays_one(self):
        result = trim_video(AudioVideo(frames(6), None, None), 1, 2)
        assert isinstance(result, AudioVideo) and result.audio is None


class TestRefusals:
    def test_past_the_end(self):
        with pytest.raises(ValueError, match="only 24 frames"):
            trim_video(clip(24), 20, 5)

    def test_past_the_end_is_the_span_rule(self):
        """trim_video raises span_problem's message, the one owner (#1374)."""
        with pytest.raises(ValueError) as raised:
            trim_video(clip(24), 20, 5)
        assert span_problem(20, 5, 24) in str(raised.value)

    def test_span_rule(self):
        assert span_problem(19, 5, 24) is None
        assert "reaches frame 25" in span_problem(20, 5, 24)

    @pytest.mark.parametrize("count", [0, -3])
    def test_non_positive_count(self, count):
        with pytest.raises(ValueError, match="num_frames"):
            trim_video(clip(24), 0, count)

    def test_negative_start(self):
        with pytest.raises(ValueError, match="start_frame"):
            trim_video(clip(24), -1, 5)

    def test_registered_handler_refuses_too(self):
        with pytest.raises(ValueError, match="num_frames"):
            _COMMAND_REGISTRY["trim_video"](
                {}, {"video": clip(24), "start_frame": 0, "num_frames": 0}, {}
            )


class TestShots:
    def test_shots_are_clipped_across_the_boundary(self):
        shots = [
            shot_record("a", 0, 10, 0, 20000),
            shot_record("b", 10, 14, 20000, 28000),
        ]
        result = trim_video(clip(24, shots=shots), 6, 8)
        got = [(s["name"], s["start_frame"], s["num_frames"]) for s in result.shots]
        assert got == [("a", 0, 4), ("b", 4, 4)]
        assert all(s["start_sample"] is None for s in result.shots)

    def test_a_shot_outside_the_span_is_dropped(self):
        shots = [shot_record("a", 0, 10), shot_record("b", 10, 14)]
        result = trim_video(clip(24, shots=shots), 0, 10)
        assert [s["name"] for s in result.shots] == ["a"]

    def test_no_shots_in_one_shot_out(self):
        """A clip with no shots record is one shot: the trim spans it, its
        samples measured off the track it cut (#627 bounce, C-F220)."""
        result = trim_video(clip(24), 12, 8)
        assert result.shots == [shot_record("video 1", 0, 8, 0, result.audio.shape[1])]
        assert result.audio.shape[1] == frames_to_samples(8, 24, 48000)

    def test_whole_clip_and_last_frame_carry_one_shot(self):
        assert [
            (s["start_frame"], s["num_frames"])
            for s in trim_video(clip(24), 0, 24).shots
        ] == [(0, 24)]
        assert [
            (s["start_frame"], s["num_frames"])
            for s in trim_video(clip(24), 23, 1).shots
        ] == [(0, 1)]

    def test_silent_clip_shot_has_no_sample_side(self):
        result = trim_video(AudioVideo(frames(6), None, None), 1, 2)
        assert result.shots == [shot_record("video 1", 0, 2)]

    def test_a_file_names_its_shot(self, tmp_path):
        path = str(tmp_path / "ep6-cold-open.mp4")
        with patch("dw.tasks.trim.load_audio_video", return_value=clip(124)):
            result = trim_video(path, 12, 48)
        assert result.shots == [
            shot_record("ep6-cold-open.mp4", 0, 48, 0, result.audio.shape[1])
        ]

    def test_the_join_of_trims_places_shots_as_before(self):
        """music-video's edit joins the trims: the one-shot record each now
        carries lands where concat_videos placed a shotless input - same
        frames, same measured samples, and the step's names still win."""
        sources = [clip(30), clip(30)]
        trims = [trim_video(sources[0], 0, 12), trim_video(sources[1], 6, 18)]
        bare = [AudioVideo(t.frames, t.audio, t.sample_rate, fps=t.fps) for t in trims]
        joined = concat_videos(trims, trim_frames=0, fps=24)
        expected = concat_videos(bare, trim_frames=0, fps=24)
        keys = (
            "start_frame",
            "num_frames",
            "start_sample",
            "num_samples",
            "hard_cut",
            "source_index",
        )
        assert [{k: s.get(k) for k in keys} for s in joined.shots] == [
            {k: s.get(k) for k in keys} for s in expected.shots
        ]
        names = ["shot@a", "shot@b"]
        assert [s["name"] for s in named_shots(joined.shots, names)] == names

    def test_input_shots_are_not_mutated(self):
        shots = [shot_record("a", 0, 24, 0, 48000)]
        trim_video(clip(24, shots=shots), 5, 5)
        assert shots == [shot_record("a", 0, 24, 0, 48000)]


def test_trimmed_shots_keep_frames_clips_the_tail():
    shots = [shot_record("a", 0, 5, 0, 10), shot_record("b", 5, 5, 10, 10)]
    assert trimmed_shots(shots, 0, keep_frames=7) == [
        shot_record("a", 0, 5),
        shot_record("b", 5, 2),
    ]
    assert trimmed_shots(shots, 3, keep_frames=4) == [
        shot_record("a", 0, 2),
        shot_record("b", 2, 2),
    ]
    assert trimmed_shots(shots, 0) is shots


def _write_clip(path, num_frames=12, fps=24, sample_rate=16000):
    import torch
    from diffusers.utils.export_utils import encode_video

    pictures = [Image.new("RGB", (16, 16), (i * 20, 0, 0)) for i in range(num_frames)]
    track = torch.full((2, int(num_frames / fps * sample_rate)), 0.25)
    encode_video(
        pictures,
        fps=fps,
        output_path=str(path),
        audio=track,
        audio_sample_rate=sample_rate,
    )
    return str(path)


def test_for_each_over_object_entries_trims_each_file_with_its_audio(tmp_path):
    """The #1199 cold-drill cut: a 'shots' variable of {name, video} objects,
    trimmed one member per entry through 'item:video'. The variables pass
    loaded each entry's 'video' field as a frame list by the key convention -
    a variable's field is not the argument it will fill (#365) - so the trim
    got silent frames, returned a bare frame list, and the save wrote one
    'video' per frame: "'Image' object is not subscriptable"."""
    from dw.result import Result
    from dw.step_cache import step_cache
    from dw.workflow import Workflow

    first = _write_clip(tmp_path / "s01.mp4")
    second = _write_clip(tmp_path / "s02.mp4")
    definition = {
        "id": "trim_each",
        "variables": {
            "shots": [
                {"name": "s01", "video": first},
                {"name": "s02", "video": second},
            ]
        },
        "steps": [
            {
                "name": "trim",
                "for_each": "variable:shots",
                "task": {
                    "command": "trim_video",
                    "arguments": {
                        "video": "item:video",
                        "start_frame": 2,
                        "num_frames": 6,
                        "fps": 24,
                    },
                },
                "result": {"content_type": "video/mp4", "fps": 24},
            }
        ],
    }
    seen = []
    original = Result.save

    def spy_save(self, *args, **kwargs):
        seen.extend(self.result_list)
        return original(self, *args, **kwargs)

    step_cache.clear()
    workflow = Workflow(definition, str(tmp_path), str(tmp_path / "trim.json"))
    with patch.object(Result, "save", spy_save):
        workflow.run({})

    assert len(seen) == 2
    for trimmed in seen:
        assert isinstance(trimmed, AudioVideo)
        assert len(trimmed.frames) == 6
        assert trimmed.audio is not None and trimmed.audio.shape[1] > 0
    files = [f for e in workflow.manifest for f in e.get("files", [])]
    assert len(files) == 2 and all(f.endswith(".mp4") for f in files)


class TestPadTo:
    """pad_to holds the last kept frame and pads the track with silence (#1194)."""

    def test_holds_the_last_frame_to_the_count(self):
        result = trim_video(clip(200), 5, 110, pad_to=124)
        assert len(result.frames) == 124
        values = [f.getpixel((0, 0))[0] for f in result.frames]
        assert values[:110] == list(range(5, 115))
        assert values[110:] == [114] * 14

    @pytest.mark.parametrize("rate", [48000, 44100])
    def test_audio_runs_on_in_silence_to_the_same_end(self, rate):
        source = clip(200, sample_rate=rate)
        result = trim_video(source, 5, 110, pad_to=124)
        first = frames_to_samples(5, 24, rate)
        kept = frames_to_samples(115, 24, rate) - first
        padded = frames_to_samples(129, 24, rate) - first
        assert result.audio.shape == (1, padded)
        assert numpy.array_equal(
            result.audio[:, :kept], source.audio[:, first : first + kept]
        )
        assert not result.audio[:, kept:].any()
        assert abs(result.audio.shape[1] / rate - 124 / 24) < 1 / rate * 2

    def test_pad_to_equal_to_num_frames_changes_nothing(self):
        source = clip(48)
        plain, padded = trim_video(source, 4, 20), trim_video(source, 4, 20, pad_to=20)
        assert len(padded.frames) == 20
        assert numpy.array_equal(plain.audio, padded.audio)
        assert plain.shots == padded.shots

    def test_the_one_shot_covers_the_held_frames(self):
        result = trim_video(clip(200), 0, 110, pad_to=124)
        assert result.shots == [
            shot_record("video 1", 0, 124, 0, result.audio.shape[1])
        ]

    def test_the_last_carried_shot_runs_over_the_held_frames(self):
        shots = [shot_record("a", 0, 30), shot_record("b", 30, 70)]
        result = trim_video(clip(100, shots=shots), 10, 50, pad_to=60)
        assert [
            (s["name"], s["start_frame"], s["num_frames"]) for s in result.shots
        ] == [
            ("a", 0, 20),
            ("b", 20, 40),
        ]

    def test_a_silent_clip_pads_its_frames_only(self):
        result = trim_video(frames(30), 0, 10, fps=24, pad_to=24)
        assert len(result.frames) == 24 and result.audio is None

    @pytest.mark.parametrize("pad_to", [109, 1])
    def test_below_num_frames_is_refused(self, pad_to):
        with pytest.raises(ValueError, match="pad_to"):
            trim_video(clip(200), 0, 110, pad_to=pad_to)

    @pytest.mark.parametrize("pad_to", [124.5, "x", 0, -4])
    def test_a_non_whole_pad_to_is_refused(self, pad_to):
        with pytest.raises(ValueError, match="pad_to"):
            trim_video(clip(200), 0, 110, pad_to=pad_to)


class TestPadToValidation:
    @staticmethod
    def workflow(arguments):
        return {
            "id": "x",
            "steps": [
                {"name": "s", "task": {"command": "trim_video", "arguments": arguments}}
            ],
        }

    @pytest.mark.parametrize("pad_to", [109, 124.5, "x", 0])
    def test_a_bad_pad_to_is_an_error_at_its_argument(self, pad_to):
        from dw.task_domains import task_argument_errors

        arguments = {"video": "v.mp4", "start_frame": 0, "num_frames": 110}
        errors = task_argument_errors(self.workflow({**arguments, "pad_to": pad_to}))
        assert [e["path"] for e in errors] == ["steps[0].task.arguments.pad_to"], errors

    @pytest.mark.parametrize("pad_to", [110, 124, "variable:p"])
    def test_a_good_pad_to_passes(self, pad_to):
        from dw.task_domains import task_argument_errors

        arguments = {"video": "v.mp4", "start_frame": 0, "num_frames": 110}
        assert (
            task_argument_errors(self.workflow({**arguments, "pad_to": pad_to})) == []
        )
