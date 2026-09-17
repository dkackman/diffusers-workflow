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
    def test_the_reported_case_is_cut_to_the_two_shot_edit(self):
        """248 frames at 24 fps is 10.33 s, from a 30 s song."""
        result = pair_audio(cut(248), song(30), sample_rate=SAMPLE_RATE, fit="video")
        assert samples(result) == round(248 / FPS * SAMPLE_RATE)

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

    def test_a_track_that_already_matches_is_left_alone(self, warnings_emitted):
        exact = song(248 / FPS)
        result = pair_audio(cut(248), exact, sample_rate=SAMPLE_RATE, fit="video")
        assert samples(result) == exact.shape[1]
        assert warnings_emitted == []

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
            pathlib.Path("workflows/templates/minimax/music-video.json").read_text()
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
        for path in pathlib.Path("workflows").rglob("*.json"):
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
