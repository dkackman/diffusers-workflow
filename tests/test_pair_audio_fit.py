"""A soundtrack whose length follows the cut it is laid over.

`music-video.json` sliced a hardcoded 496 frames of song (4 shots x 124) while
everything else in it followed the `shots` list, so a two-shot run produced a
deliverable holding 248 frames of picture in a 20.7 s container - audio twice
as long as video, `status: succeeded`, `warnings: []` (#142). `fit: "video"`
derives the length from the frames instead, and without it a disagreement is
at least said out loud.
"""

import json
import pathlib

import numpy
import pytest

from dw.result import AudioVideo
from dw.tasks.pair_audio import pair_audio

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

SAMPLE_RATE = 44100
FPS = 24


def song(seconds, sample_rate=SAMPLE_RATE):
    return numpy.zeros((2, int(seconds * sample_rate)), dtype=numpy.float32)


def cut(frames, fps=FPS):
    return AudioVideo([object()] * frames, None, None, fps=fps)


def samples(result):
    return result.audio.shape[1]


@pytest.fixture
def warnings_emitted():
    """The messages of the warning events a call emits - what a consumer over
    the API or MCP actually reads, rather than the server's log (#82)."""
    from dw.events import RunContext, activate_context, deactivate_context

    messages = []

    def record(event):
        if event["event"] == "warning":
            messages.append(event["message"])

    token = activate_context(RunContext(on_event=record))
    try:
        yield messages
    finally:
        deactivate_context(token)


class TestFitToTheVideo:
    def test_the_default_four_shot_length_is_unchanged(self):
        """496 frames - what the hardcoded slice used to produce."""
        result = pair_audio(cut(496), song(30), sample_rate=SAMPLE_RATE, fit="video")
        assert samples(result) == round(496 / FPS * SAMPLE_RATE)

    def test_the_other_direction_pads_and_warns(self, warnings_emitted):
        """Six shots outrun a 30 s song - the silent-padding direction the
        report could not afford to run."""
        result = pair_audio(cut(744), song(30), sample_rate=SAMPLE_RATE, fit="video")
        assert samples(result) == round(744 / FPS * SAMPLE_RATE)
        assert any("padded" in w for w in warnings_emitted)

    def test_trimming_the_track_warns_with_the_seconds_cut(self, warnings_emitted):
        """The pad direction cannot lose content; the trim direction always
        can, so it is the one that most needs saying out loud (#246). This is
        also #142's reported case: 248 frames at 24 fps is 10.33 s, from a
        30 s song."""
        result = pair_audio(cut(248), song(30), sample_rate=SAMPLE_RATE, fit="video")
        assert samples(result) == round(248 / FPS * SAMPLE_RATE)
        trimmed = [w for w in warnings_emitted if "trimmed" in w]
        assert len(trimmed) == 1
        assert "19.67 s" in trimmed[0]

    def test_a_track_that_already_matches_is_left_alone(self, warnings_emitted):
        exact = song(248 / FPS)
        result = pair_audio(cut(248), exact, sample_rate=SAMPLE_RATE, fit="video")
        assert samples(result) == exact.shape[1]
        assert warnings_emitted == []

    def test_a_sub_frame_shortfall_still_pads_and_warns(self, warnings_emitted):
        """A track a handful of samples short of the target used to be waved
        through unfitted and unwarned - the mismatch was well inside
        LENGTH_WARN_MS, a tolerance meant only for the no-'fit' mismatch
        warning. An explicit 'fit': 'video' promises an exact length
        regardless of how small the gap is (#428)."""
        wanted = round(248 / FPS * SAMPLE_RATE)
        short = numpy.zeros((2, wanted - 15), dtype=numpy.float32)
        result = pair_audio(cut(248), short, sample_rate=SAMPLE_RATE, fit="video")
        assert samples(result) == wanted
        assert any("padded" in w for w in warnings_emitted)

    def test_a_sub_frame_excess_still_trims_and_warns(self, warnings_emitted):
        wanted = round(248 / FPS * SAMPLE_RATE)
        long = numpy.zeros((2, wanted + 15), dtype=numpy.float32)
        result = pair_audio(cut(248), long, sample_rate=SAMPLE_RATE, fit="video")
        assert samples(result) == wanted
        assert any("trimmed" in w for w in warnings_emitted)

    def test_a_one_sample_pad_does_not_claim_the_cut_has_no_soundtrack(
        self, warnings_emitted
    ):
        """A 1-sample pad from ordinary rate rounding (32 kHz doubled from a
        16 kHz source, 248 f @ 24 fps) used to format as '0.00 s of silence
        ... has no soundtrack' - self-contradictory, and the advice to use a
        longer track or fewer frames cannot fix a 1-sample gap (#429)."""
        wanted = round(248 / FPS * SAMPLE_RATE)
        short = numpy.zeros((2, wanted - 1), dtype=numpy.float32)
        result = pair_audio(cut(248), short, sample_rate=SAMPLE_RATE, fit="video")
        assert samples(result) == wanted
        padded = [w for w in warnings_emitted if "padded" in w]
        assert len(padded) == 1
        assert "0.00 s" not in padded[0]
        assert "has no soundtrack" not in padded[0]
        assert "1 sample" in padded[0]

    def test_a_one_sample_trim_does_not_claim_content_is_gone(self, warnings_emitted):
        wanted = round(248 / FPS * SAMPLE_RATE)
        long = numpy.zeros((2, wanted + 1), dtype=numpy.float32)
        result = pair_audio(cut(248), long, sample_rate=SAMPLE_RATE, fit="video")
        assert samples(result) == wanted
        trimmed = [w for w in warnings_emitted if "trimmed" in w]
        assert len(trimmed) == 1
        assert "0.00 s" not in trimmed[0]
        assert "is gone from the deliverable" not in trimmed[0]
        assert "1 sample" in trimmed[0]

    def test_a_sub_frame_but_multi_sample_pad_still_omits_the_soundtrack_claim(
        self, warnings_emitted
    ):
        """The existing 15-sample (#428) case is also well under one video
        frame (41.67 ms at 24 fps) - it should get the same rounding-aware
        wording as the 1-sample case, not the frame-scale 'no soundtrack'
        claim."""
        wanted = round(248 / FPS * SAMPLE_RATE)
        short = numpy.zeros((2, wanted - 15), dtype=numpy.float32)
        result = pair_audio(cut(248), short, sample_rate=SAMPLE_RATE, fit="video")
        assert samples(result) == wanted
        padded = [w for w in warnings_emitted if "padded" in w]
        assert len(padded) == 1
        assert "has no soundtrack" not in padded[0]
        assert "15 samples" in padded[0]

    def test_a_frame_scale_pad_still_names_the_cut_as_uncovered(
        self, warnings_emitted
    ):
        """A pad at least a full video frame long is a real gap - the
        original wording, with its advice, still applies."""
        pair_audio(cut(744), song(30), sample_rate=SAMPLE_RATE, fit="video")
        padded = [w for w in warnings_emitted if "padded" in w]
        assert len(padded) == 1
        assert "has no soundtrack" in padded[0]

    def test_fps_may_be_given_when_the_frames_carry_none(self):
        result = pair_audio(
            [object()] * 248, song(30), sample_rate=SAMPLE_RATE, fps=FPS, fit="video"
        )
        assert samples(result) == round(248 / FPS * SAMPLE_RATE)

    def test_an_unknown_fit_is_refused_rather_than_ignored(self):
        with pytest.raises(ValueError, match="'fit'"):
            pair_audio(cut(248), song(30), sample_rate=SAMPLE_RATE, fit="audio")


class TestWithoutFit:
    def test_a_mismatch_is_warned_about(self, warnings_emitted):
        result = pair_audio(cut(248), song(30), sample_rate=SAMPLE_RATE)
        assert samples(result) == song(30).shape[1], "the track is left as it is"
        assert any("disagree" in w for w in warnings_emitted)

    def test_an_agreeing_pair_says_nothing(self, warnings_emitted):
        pair_audio(cut(248), song(248 / FPS), sample_rate=SAMPLE_RATE)
        assert warnings_emitted == []

    def test_unknown_fps_is_not_a_mismatch(self, warnings_emitted):
        """A frame rate this layer cannot know is not something to warn about."""
        pair_audio(cut(248, fps=None), song(30), sample_rate=SAMPLE_RATE)
        assert warnings_emitted == []


class TestTheTemplateItself:
    def test_music_video_derives_its_soundtrack(self):
        definition = json.loads(
            (REPO_ROOT / "workflows/templates/minimax/music-video.json").read_text()
        )
        steps = {s["name"]: s for s in definition["steps"]}
        assert "soundtrack" not in steps, "the hardcoded 496-frame slice is gone"
        arguments = steps["music_video"]["task"]["arguments"]
        # The whole song, by way of the gain step that gives the mux headroom (#159)
        assert arguments["audio"] == "previous_result:balanced"
        assert steps["balanced"]["task"]["arguments"]["audio"] == (
            "previous_result:write_song"
        )
        assert arguments["fit"] == "video"

    def test_no_template_hardcodes_a_soundtrack_length(self):
        """Every `slice_audio` in the catalog whose count is a literal is one
        a list cannot resize out from under - so the literal must not be the
        length of a whole cut."""
        for path in (REPO_ROOT / "workflows").rglob("*.json"):
            definition = json.loads(path.read_text())
            if not isinstance(definition, dict):
                continue
            for step in definition.get("steps", []):
                task = step.get("task") or {}
                if task.get("command") != "pair_audio":
                    continue
                audio = task.get("arguments", {}).get("audio", "")
                if not isinstance(audio, str) or not audio.startswith(
                    "previous_result:"
                ):
                    continue
                source = {s["name"]: s for s in definition["steps"]}.get(
                    audio.split(":", 1)[1]
                )
                source_task = (source or {}).get("task") or {}
                if source_task.get("command") != "slice_audio":
                    continue
                count = source_task.get("arguments", {}).get("num_frames")
                assert not isinstance(count, int), (
                    f"{path}: '{source['name']}' cuts a literal {count}-frame "
                    f"soundtrack for a cut whose length may be an argument - "
                    f"use pair_audio's 'fit': 'video' instead (#142)"
                )
