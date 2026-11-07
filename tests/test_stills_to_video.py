"""stills_to_video: storyboard stills held for a duration into a silent clip (#1144)."""

import numpy
import pytest
from PIL import Image

from dw.media_types import AudioVideo
from dw.tasks.video_utils import (
    MAX_STILLS_FPS,
    MAX_STILLS_SECONDS,
    STILLS_SAMPLE_RATE,
    stills_to_video,
)


def still(width=512, height=512, color=(255, 0, 0)):
    return Image.new("RGB", (width, height), color)


class TestHold:
    def test_three_stills_hold_for_their_durations(self):
        video = stills_to_video([still(), still(), still()], [1, 2, 0.5], fps=24)
        assert isinstance(video, AudioVideo)
        assert video.frames.shape == (84, 512, 512, 3)
        assert video.frames.dtype == numpy.uint8
        assert video.fps == 24
        assert [shot["start_frame"] for shot in video.shots] == [0, 24, 72]
        assert [shot["num_frames"] for shot in video.shots] == [24, 48, 12]

    def test_each_still_fills_its_own_frames(self):
        video = stills_to_video(
            [still(color=(255, 0, 0)), still(color=(0, 0, 255))], [1, 1], fps=2
        )
        assert tuple(video.frames[0, 0, 0]) == (255, 0, 0)
        assert tuple(video.frames[3, 0, 0]) == (0, 0, 255)

    def test_one_duration_applies_to_all(self):
        video = stills_to_video([still(), still()], 2, fps=12)
        assert len(video.frames) == 48

    def test_a_sub_frame_duration_still_gets_one_frame(self):
        video = stills_to_video([still()], [0.01], fps=24)
        assert len(video.frames) == 1

    def test_arrays_are_accepted(self):
        array = numpy.zeros((64, 64, 3), dtype=numpy.uint8)
        video = stills_to_video([array], 1, fps=4)
        assert video.frames.shape == (4, 64, 64, 3)

    def test_clip_carries_a_silent_track_of_its_own_length(self):
        video = stills_to_video([still(), still()], [1, 0.5], fps=24)
        assert video.sample_rate == STILLS_SAMPLE_RATE
        assert video.audio.shape == (1, round(36 / 24 * STILLS_SAMPLE_RATE))
        assert not video.audio.any()


class TestRefusals:
    def test_mismatched_sizes_refused(self):
        with pytest.raises(ValueError, match=r"'images' must all.*512x512.*512x768"):
            stills_to_video([still(512, 512), still(512, 768)], 1)

    def test_wrong_duration_count_refused(self):
        with pytest.raises(ValueError, match="durations"):
            stills_to_video([still(), still(), still()], [1, 2])

    @pytest.mark.parametrize("durations", [[1, 0], [-1], 0])
    def test_non_positive_duration_refused(self, durations):
        with pytest.raises(ValueError, match="durations"):
            stills_to_video(
                [still(), still()][
                    : len(durations) if isinstance(durations, list) else 1
                ],
                durations,
            )

    def test_fps_bounds(self):
        for bad in (0, 121):
            with pytest.raises(ValueError, match="fps"):
                stills_to_video([still()], 1, fps=bad)
        assert MAX_STILLS_FPS == 120
        assert len(stills_to_video([still()], 1, fps=120).frames) == 120

    def test_total_over_600_seconds_refused(self):
        assert MAX_STILLS_SECONDS == 600
        with pytest.raises(ValueError, match="600"):
            stills_to_video([still()], [601], fps=1)

    def test_no_images_refused(self):
        with pytest.raises(ValueError, match="images"):
            stills_to_video([], 1)


class TestWithConcat:
    def test_result_concats_with_concat_videos(self):
        from dw.tasks.concat_videos import concat_videos

        video = stills_to_video([still(64, 64)], 1, fps=4)
        joined = concat_videos([video, video], fps=4)
        assert len(joined.frames) == 2 * len(video.frames)

    def test_clip_scores_through_the_assemble_and_score_chain(self):
        """assemble-and-score's own steps, in order: the joined cut's track is
        resampled as the world audio, mixed under the score and paired back
        onto the cut - a stills clip must not stop at resample_audio."""
        from dw.tasks.audio_utils import mix_audio, resample_audio
        from dw.tasks.concat_videos import concat_videos
        from dw.tasks.pair_audio import pair_audio

        video = stills_to_video([still(64, 64)] * 3, [1, 2, 0.5], fps=24)
        cut = concat_videos([video], fps=24, sample_rate=44100)
        world = resample_audio(cut, 44100)
        score = numpy.full((2, round(3.5 * 44100)), 0.1, dtype=numpy.float32)
        mixed = mix_audio([score, world], gains=[1.0, 1.8], sample_rate=44100)
        film = pair_audio(cut, mixed, sample_rate=44100, fit="video")

        assert len(film.frames) == 84
        assert film.audio.shape[-1] == round(3.5 * 44100)
        assert film.audio.any()

    def test_silent_clip_saves_as_a_video(self, tmp_path):
        from dw.result import Result

        video = stills_to_video([still(64, 64)], 1, fps=4)
        result = Result({"content_type": "video/mp4"})
        result.add_result([video])
        result.save(str(tmp_path), "silent")
        files = list(tmp_path.glob("silent*.mp4"))
        assert files and files[0].stat().st_size > 0


class TestRegistered:
    def test_command_registered(self):
        from dw.tasks.task import _COMMAND_REGISTRY

        assert "stills_to_video" in _COMMAND_REGISTRY


class TestStaticValidation:
    def _errors(self, **arguments):
        from dw.task_problems import stills_to_video_errors

        return stills_to_video_errors(arguments)

    def test_literal_bad_duration_found(self):
        assert [a for a, _ in self._errors(images=["a"], durations=[1, 0])] == [
            "durations"
        ]

    def test_fps_out_of_range_found(self):
        assert [a for a, _ in self._errors(durations=1, fps=121)] == ["fps"]

    def test_count_mismatch_found(self):
        errors = self._errors(images=["a", "b", "c"], durations=[1, 2])
        assert [a for a, _ in errors] == ["durations"]

    def test_total_found(self):
        assert self._errors(durations=[601])

    def test_references_are_left_alone(self):
        assert not self._errors(images="previous_result:x", durations="variable:d")

    def test_clean_arguments_pass(self):
        assert not self._errors(images=["a", "b"], durations=[1, 2], fps=24)


class TestHandlerLoadsLocations:
    def test_handler_reads_image_dicts(self, tmp_path):
        from dw.tasks.task import _COMMAND_REGISTRY

        path = tmp_path / "a.png"
        still(32, 32).save(path)
        handler = _COMMAND_REGISTRY["stills_to_video"]
        video = handler(
            None,
            {
                "images": [{"media_type": "image", "location": str(path)}],
                "durations": 1,
                "fps": 4,
            },
            {},
        )
        assert video.frames.shape == (4, 32, 32, 3)


class TestByteBudget:
    def test_over_budget_refused_before_allocating(self, monkeypatch):
        monkeypatch.setattr("dw.task_problems.MAX_STILLS_BYTES", 1000)
        with pytest.raises(ValueError) as error:
            stills_to_video([still(16, 16)], [1], fps=24)
        message = str(error.value)
        assert "fps" in message and "16x16" in message and "split" in message

    def test_within_budget_passes(self, monkeypatch):
        monkeypatch.setattr("dw.task_problems.MAX_STILLS_BYTES", 4 * 16 * 16 * 3)
        assert len(stills_to_video([still(16, 16)], [1], fps=4).frames) == 4

    def test_the_message_names_budget_size_and_the_fps_that_fits(self):
        # The #1199 drill's animatic: 12 boards, about 304 s at 24 fps, at
        # 1344x768 - 7290 frames, 21 GiB against the 4 GiB budget
        from dw.task_problems import stills_memory_problem

        seconds = [304 / 12] * 12
        message = stills_memory_problem(seconds, 24, 1344, 768)
        assert "4 GiB" in message and "21.0 GiB" in message
        assert "1344x768" in message and "7296 frames" in message
        # 1387 frames fit: 57 s at 24 fps, 4 fps for the whole 304 s
        assert "57 s at 24 fps" in message
        assert "'fps' to 4" in message
        assert "split" in message
        assert stills_memory_problem(seconds, 4, 1344, 768) is None


def _boards(monkeypatch, tmp_path, count, size=(1344, 768)):
    asset_dir = tmp_path / "assets"
    asset_dir.mkdir()
    for index in range(count):
        still(*size).save(asset_dir / f"board-{index}.jpg")
    monkeypatch.setenv("DW_ASSET_DIR", str(asset_dir))
    return [f"asset:board-{index}.jpg" for index in range(count)]


def _animatic_errors(tmp_path, arguments):
    import os

    from dw.workflow import workflow_from_file

    template = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "workflows",
        "templates",
        "animatic.json",
    )
    workflow = workflow_from_file(template, str(tmp_path / "out"))
    return [e for e in workflow.validation_errors(arguments) if "GiB" in e["message"]]


class TestByteBudgetAtValidate:
    """The frame count is the durations and fps the caller passes, and the
    size is the boards' own header - both known before the run, so an
    animatic over the budget is refused at validate, not ~1 s into the run
    (#1199, job f4ff0c77d018)."""

    def test_over_budget_animatic_is_refused(self, monkeypatch, tmp_path):
        boards = _boards(monkeypatch, tmp_path, 2)
        errors = _animatic_errors(
            tmp_path, {"boards": boards, "durations": [30, 30], "fps": 24}
        )
        assert len(errors) == 1
        assert errors[0]["path"] == "steps[0].task.arguments.images"
        message = errors[0]["message"]
        assert "4 GiB" in message and "1344x768" in message
        assert "1440 frames" in message and "'fps'" in message

    def test_the_same_boards_at_4_fps_validate(self, monkeypatch, tmp_path):
        boards = _boards(monkeypatch, tmp_path, 2)
        assert (
            _animatic_errors(
                tmp_path, {"boards": boards, "durations": [30, 30], "fps": 4}
            )
            == []
        )

    def test_a_board_from_an_earlier_step_is_left_to_the_run(self):
        from dw.video_size_errors import stills_memory_errors

        definition = {
            "id": "a",
            "steps": [
                {
                    "name": "stills",
                    "task": {
                        "command": "stills_to_video",
                        "arguments": {
                            "images": ["previous_result:draw"],
                            "durations": [600],
                            "fps": 120,
                        },
                    },
                }
            ],
        }
        assert stills_memory_errors(definition) == []

    def test_many_tiny_durations_cap_the_frame_count(self):
        from dw.task_problems import MAX_STILLS_FRAMES, stills_to_video_problems

        durations = [0.001] * (MAX_STILLS_FRAMES + 1)
        problems = stills_to_video_problems(durations, 24, len(durations))
        assert [a for a, _ in problems] == ["durations"]
        assert "frame" in problems[0][1]
        with pytest.raises(ValueError, match="frame"):
            stills_to_video([still(2, 2)] * len(durations), durations, fps=24)

    def test_frame_cap_applies_to_a_scalar_duration(self):
        from dw.task_problems import MAX_STILLS_FRAMES, stills_to_video_problems

        count = MAX_STILLS_FRAMES + 1
        assert stills_to_video_problems(0.001, 24, count)


class TestDurationsShapes:
    def test_numpy_durations(self):
        video = stills_to_video(
            [still(16, 16), still(16, 16)], numpy.array([1, 2]), fps=4
        )
        assert [s["num_frames"] for s in video.shots] == [4, 8]

    def test_empty_durations_refused(self):
        with pytest.raises(ValueError, match="durations"):
            stills_to_video([still(16, 16)], [], fps=4)


class TestHandlerEntries:
    def _run(self, images):
        from dw.tasks.task import _COMMAND_REGISTRY

        return _COMMAND_REGISTRY["stills_to_video"](
            None, {"images": images, "durations": 1, "fps": 2}, {}
        )

    def test_an_entry_of_several_images_is_flattened_in_order(self):
        red, blue = still(16, 16, (255, 0, 0)), still(16, 16, (0, 0, 255))
        video = self._run([[red, blue], still(16, 16, (0, 255, 0))])
        assert len(video.shots) == 3
        assert tuple(video.frames[0, 0, 0]) == (255, 0, 0)
        assert tuple(video.frames[2, 0, 0]) == (0, 0, 255)

    def test_an_entry_of_no_images_is_refused(self):
        with pytest.raises(ValueError, match="images"):
            self._run([[]])


class TestSavedManifest:
    def test_saved_shots_start_frames(self, tmp_path):
        from dw.result import Result
        from dw.shots import step_shots

        video = stills_to_video([still(16, 16)] * 3, [1, 2, 0.5], fps=24)
        result = Result({"content_type": "video/mp4", "save": True})
        result.add_result(video)
        saved = result.save(str(tmp_path), "animatic.0")
        assert list(tmp_path.glob("animatic.0*.mp4"))
        shots = step_shots(result.saved_shots, saved)
        assert [shot["start_frame"] for shot in shots] == [0, 24, 72]


class TestSilentByDesign:
    """The silent track raises no near-silent or mono-duplication warning (#1161)."""

    def test_clip_and_shots_are_marked(self):
        video = stills_to_video([still(64, 64)], [1], fps=24)
        assert video.silent_by_design is True
        assert all(shot["silent_by_design"] for shot in video.shots)

    def test_gallery_level_skips_near_silent_for_marked_shots(self):
        from dw.server.assess import level_findings

        media = {"peak_dbfs": -120.0, "mean_dbfs": -120.0}
        video = stills_to_video([still(64, 64)], [1], fps=24)
        assert level_findings(media, video.shots) == []
        names = [f["rule"] for f in level_findings(media)]
        assert "near_silent" in names

    def test_mono_duplication_quiet_only_when_asked(self, monkeypatch):
        import dw.writers as writers

        seen = []
        monkeypatch.setattr(writers, "emit_warning", lambda *a, **k: seen.append(a))
        mono = numpy.zeros((1, 100), dtype=numpy.float32)
        writers.as_audio_track(mono, quiet=True)
        assert seen == []
        writers.as_audio_track(mono)
        assert len(seen) == 1

    def test_write_time_near_silent_skipped(self, monkeypatch):
        import dw.audio_qc as qc

        calls = []
        monkeypatch.setattr(
            qc,
            "_probe_written_media",
            lambda p: {"mean_dbfs": -120.0, "peak_dbfs": -120.0},
        )
        monkeypatch.setattr(qc, "emit_warning", lambda *a, **k: calls.append(a))
        monkeypatch.setattr(qc, "remeasure_shots_after_mux", lambda *a, **k: None)
        monkeypatch.setattr(
            qc, "warn_if_written_above_full_scale", lambda *a, **k: None
        )
        video = stills_to_video([still(64, 64)], [1], fps=24)
        qc.check_written_media(
            "x.mp4",
            video,
            "video/mp4",
            video_fps=None,
            consumed_by_normalizer=False,
            headroom_warned=False,
            predicted_peak_dbfs=None,
        )
        assert calls == []
