"""The four rules validation and the run both apply, answered the same way.

Each rule used to be written twice - once in a validation checker, once in
the task that enforces it at run time - and two of the copies had drifted:
the run's frame-size refusal stopped at the first mismatch and always called
its reference "video 0", and validation's slice-past-end check lacked #557's
end-frame rounding. Each test here drives one set of inputs through both the
checker and the task and asserts they carry the same sentence, which the
shared rule in dw/task_domains.py builds.

The checkers take a fake `probe` (a keyword, not a patch target); the files
still exist because the checkers only probe a path that resolves to one.
"""

import os
import tempfile
import wave

import numpy
import pytest

from dw.dissolve_frame_errors import dissolve_frame_errors
from dw.select_validation import select_errors
from dw.slice_preflight import slice_past_end_warnings
from dw.tasks.audio_utils import slice_audio
from dw.tasks.dissolve_videos import dissolve_videos
from dw.tasks.select import select
from dw.tasks.video_utils import check_same_frame_size
from dw.video_size_errors import video_size_errors


def asset_dir_with(monkeypatch, *names):
    base_dir = tempfile.mkdtemp()
    asset_dir = os.path.join(base_dir, "assets")
    os.makedirs(asset_dir)
    for name in names:
        with open(os.path.join(asset_dir, name), "wb") as handle:
            handle.write(b"\0")
    monkeypatch.setenv("DW_ASSET_DIR", asset_dir)
    return base_dir


def fake_probe(infos):
    """A probe answering from a table keyed by file name."""

    def probe(path):
        return infos.get(os.path.basename(path))

    return probe


def task_step(command, arguments, name="join"):
    return {
        "id": "parity",
        "steps": [
            {
                "name": name,
                "task": {"command": command, "arguments": arguments},
                "result": {"content_type": "video/mp4", "fps": 6},
            }
        ],
    }


def clip(frames, width=32, height=16):
    return numpy.zeros((frames, height, width, 3), numpy.uint8)


class TestDissolveOverlap:
    def test_the_checker_and_the_task_name_the_same_short_video(self, monkeypatch):
        base_dir = asset_dir_with(monkeypatch, "a.mp4", "b.mp4", "c.mp4")
        counts = {"a.mp4": 8, "b.mp4": 5, "c.mp4": 8}
        probe = fake_probe(
            {name: {"kind": "video", "frame_count": n} for name, n in counts.items()}
        )
        definition = task_step(
            "dissolve_videos",
            {
                "videos": ["asset:a.mp4", "asset:b.mp4", "asset:c.mp4"],
                "dissolve_frames": 3,
            },
        )

        problems = dissolve_frame_errors(definition, base_dir=base_dir, probe=probe)
        with pytest.raises(ValueError) as raised:
            dissolve_videos([clip(8), clip(5), clip(8)], dissolve_frames=3)

        assert len(problems) == 1
        assert str(raised.value) in problems[0]["message"]

        from dw.task_domains import dissolve_shortfalls

        sentences = dissolve_shortfalls([8, 5, 8], 3)
        assert sentences == [
            "video 1 has 5 frames, too few for its 2 dissolve(s) of 3 frames"
        ]
        assert sentences[0] in problems[0]["message"]
        assert str(raised.value) == sentences[0]

    def test_an_unknown_count_is_skipped_but_still_counts_as_a_seam(self):
        from dw.task_domains import dissolve_shortfalls

        assert dissolve_shortfalls([None, 5, None], 3) == [
            "video 1 has 5 frames, too few for its 2 dissolve(s) of 3 frames"
        ]


class TestFrameSize:
    def test_every_mismatch_is_named_by_both(self, monkeypatch):
        base_dir = asset_dir_with(monkeypatch, "a.mp4", "b.mp4", "c.mp4")
        probe = fake_probe(
            {
                "a.mp4": {"kind": "video", "width": 32, "height": 16},
                "b.mp4": {"kind": "video", "width": 64, "height": 32},
                "c.mp4": {"kind": "video", "width": 48, "height": 24},
            }
        )
        definition = task_step(
            "concat_videos", {"videos": ["asset:a.mp4", "asset:b.mp4", "asset:c.mp4"]}
        )

        problems = video_size_errors(definition, base_dir=base_dir, probe=probe)
        with pytest.raises(ValueError) as raised:
            check_same_frame_size(
                [clip(2, 32, 16), clip(2, 64, 32), clip(2, 48, 24)], "concat_videos"
            )

        assert len(problems) == 1
        assert str(raised.value) in problems[0]["message"]
        assert "video 2 is 48x24" in str(raised.value)

        from dw.task_domains import frame_size_mismatches

        sentence = frame_size_mismatches({0: (32, 16), 1: (64, 32), 2: (48, 24)})
        assert sentence == "video 0 is 32x16, video 1 is 64x32, video 2 is 48x24"
        assert sentence in problems[0]["message"]
        assert sentence in str(raised.value)

    def test_the_reference_is_named_by_its_real_index(self, monkeypatch):
        base_dir = asset_dir_with(monkeypatch, "b.mp4", "c.mp4")
        probe = fake_probe(
            {
                "b.mp4": {"kind": "video", "width": 32, "height": 16},
                "c.mp4": {"kind": "video", "width": 64, "height": 32},
            }
        )
        definition = task_step(
            "concat_videos",
            {"videos": ["previous_result:make_a", "asset:b.mp4", "asset:c.mp4"]},
        )

        problems = video_size_errors(definition, base_dir=base_dir, probe=probe)
        with pytest.raises(ValueError) as raised:
            check_same_frame_size(
                [[], clip(2, 32, 16), clip(2, 64, 32)], "concat_videos"
            )

        assert "video 1 is 32x16, video 2 is 64x32" in problems[0]["message"]
        assert "video 1 is 32x16, video 2 is 64x32" in str(raised.value)

    def test_matching_sizes_are_no_sentence(self):
        from dw.task_domains import frame_size_mismatches

        assert frame_size_mismatches({0: (32, 16), 2: (32, 16)}) is None
        assert frame_size_mismatches({}) is None


def write_wav_samples(path, samples, sample_rate):
    tone = (numpy.sin(numpy.arange(samples) / 8.0) * 0.5 * 32767).astype("<i2")
    with wave.open(str(path), "w") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(tone.tobytes())


class TestSlicePadding:
    # 16587 samples at 8000 Hz is 2.073375 s. Fifty frames at 24 fps end at
    # sample round(50 / 24 * 8000) = 16667, 80 samples (10.0 ms) past the
    # end - the run's #557 end-rounding reaches the threshold exactly, where
    # the float-seconds arithmetic (9.958 ms) stops short of it
    SAMPLES = 16587
    RATE = 8000

    def _both(self, monkeypatch, caplog, **arguments):
        base_dir = tempfile.mkdtemp()
        asset_dir = os.path.join(base_dir, "assets")
        os.makedirs(asset_dir)
        write_wav_samples(os.path.join(asset_dir, "score.wav"), self.SAMPLES, self.RATE)
        monkeypatch.setenv("DW_ASSET_DIR", asset_dir)
        definition = task_step(
            "slice_audio", {"audio": "asset:score.wav", **arguments}, name="soundtrack"
        )
        checked = slice_past_end_warnings(definition, base_dir=base_dir)

        waveform = numpy.zeros((1, self.SAMPLES), numpy.float32)
        with caplog.at_level("WARNING"):
            slice_audio(waveform, sample_rate=self.RATE, **arguments)
        ran = [
            r.getMessage() for r in caplog.records if "past the end" in r.getMessage()
        ]
        return checked, ran

    def test_the_end_rounded_threshold_case_warns_in_both(self, monkeypatch, caplog):
        checked, ran = self._both(
            monkeypatch, caplog, start_frame=0, num_frames=50, fps=24
        )

        assert len(ran) == 1
        assert len(checked) == 1
        assert "0.01 s past the end" in ran[0]
        assert "0.01 s past the end" in checked[0]

    def test_a_seconds_slice_agrees(self, monkeypatch, caplog):
        checked, ran = self._both(
            monkeypatch, caplog, start_seconds=1.0, duration_seconds=3.0
        )

        assert len(ran) == len(checked) == 1
        assert "1.93 s past the end" in ran[0]
        assert "1.93 s past the end" in checked[0]

    def test_the_rule_is_the_padded_seconds_at_or_over_the_threshold(self):
        from dw.task_domains import SLICE_PAD_WARN_MS, slice_padding

        assert SLICE_PAD_WARN_MS == 10.0
        assert slice_padding(16587, 0, 16667, 8000) == pytest.approx(0.010)
        assert slice_padding(16588, 0, 16667, 8000) is None
        assert slice_padding(100, 0, 50, 8000) is None
        assert slice_padding(100, 0, 500, 0) is None

    def test_frames_to_samples_has_one_home(self):
        from dw import task_domains
        from dw.tasks import audio_utils

        assert audio_utils.frames_to_samples is task_domains.frames_to_samples
        assert audio_utils.SLICE_PAD_WARN_MS == task_domains.SLICE_PAD_WARN_MS


class TestSelectRules:
    @pytest.mark.parametrize(
        "arguments, key",
        [
            ({"rule": "bogus"}, "rule"),
            ({"rule": "first_above"}, "threshold"),
            ({"rule": "first_below"}, "threshold"),
            ({"rule": "index"}, "index"),
        ],
    )
    def test_the_checker_and_the_task_refuse_in_one_sentence(self, arguments, key):
        definition = task_step(
            "select",
            {"candidates": ["a", "b"], "scores": [1, 2], **arguments},
            name="pick",
        )

        problems = select_errors(definition)
        with pytest.raises(ValueError) as raised:
            select(["a", "b"], [1, 2], **arguments)

        assert len(problems) == 1
        assert problems[0]["path"] == f"steps[0].task.arguments.{key}"
        assert str(raised.value) in problems[0]["message"]

        from dw.task_domains import select_rule_problems

        sentences = select_rule_problems(arguments["rule"], None, None)
        assert sentences == [str(raised.value)]

    def test_a_rule_given_what_it_needs_has_no_problem(self):
        from dw.task_domains import (
            SELECT_RULES,
            SELECT_THRESHOLD_RULES,
            select_rule_problems,
        )

        assert SELECT_THRESHOLD_RULES < SELECT_RULES
        assert select_rule_problems("argmax", None, None) == []
        assert select_rule_problems("first_above", 0.5, None) == []
        assert select_rule_problems("index", None, 1) == []
