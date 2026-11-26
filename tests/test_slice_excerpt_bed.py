"""`slice_audio` `past_end: "excerpt"` and assemble-and-score's bed (#1719).

A bed is a window cut out of a longer recording, not a section kept whole:
the rest of the cut playing on past it is the point, not a lost tail, so
"excerpt" stops at the source's end the way "stop" does but warns neither
`slice_dropped_loud_tail` nor `slice_trimmed_tail`. assemble-and-score loops
that window under the mix at `bed_gain`, 0 (off) by default, so a default
run's mix is the two-track mix it always was.
"""

import json
from pathlib import Path

import numpy
import pytest

from dw.events import RunContext, activate_context, deactivate_context
from dw.task_domains import SLICE_PAST_END_CHOICES, task_argument_errors
from dw.tasks.audio_slice import slice_audio

RATE = 1000
ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "workflows" / "templates" / "assemble-and-score.json"


def tone(seconds, level=0.5, rate=RATE):
    count = int(round(seconds * rate))
    t = numpy.arange(count) / rate
    return (level * numpy.sin(2 * numpy.pi * 50 * t)).astype(numpy.float32)[None, :]


def events_of(call):
    events = []
    token = activate_context(RunContext(on_event=events.append))
    try:
        result = call()
    finally:
        deactivate_context(token)
    return result, events


def kinds(events, kind):
    return [e for e in events if e.get("kind") == kind]


class TestExcerpt:
    def test_excerpt_is_a_past_end_choice(self):
        assert "excerpt" in SLICE_PAST_END_CHOICES

    def test_a_long_loud_tail_does_not_warn(self):
        sliced, events = events_of(
            lambda: slice_audio(
                tone(40.0),
                start_seconds=2.0,
                duration_seconds=1.0,
                sample_rate=RATE,
                past_end="excerpt",
            )
        )
        assert sliced.audio.shape[1] == RATE
        assert kinds(events, "slice_dropped_loud_tail") == []
        assert kinds(events, "slice_trimmed_tail") == []
        assert kinds(events, "slice_past_end") == []

    def test_a_small_trim_does_not_warn(self):
        # 39.5 s of a 40 s source: under 10 s and 5% left, which "stop" warns
        _, events = events_of(
            lambda: slice_audio(
                tone(40.0),
                duration_seconds=39.5,
                sample_rate=RATE,
                past_end="excerpt",
            )
        )
        assert kinds(events, "slice_trimmed_tail") == []
        assert kinds(events, "slice_dropped_loud_tail") == []

    def test_the_same_window_with_stop_still_warns(self):
        _, events = events_of(
            lambda: slice_audio(
                tone(40.0), duration_seconds=20.0, sample_rate=RATE, past_end="stop"
            )
        )
        assert len(kinds(events, "slice_dropped_loud_tail")) == 1

    def test_a_window_past_the_end_is_shortened_not_padded(self):
        source = tone(3.0)
        sliced, events = events_of(
            lambda: slice_audio(
                source,
                start_seconds=2.0,
                duration_seconds=5.0,
                sample_rate=RATE,
                past_end="excerpt",
            )
        )
        assert sliced.audio.shape[1] == RATE
        numpy.testing.assert_array_equal(sliced.audio, source[:, 2 * RATE :])
        assert kinds(events, "slice_past_end") == []
        logs = [e for e in events if "excerpt" in str(e.get("message", ""))]
        assert logs, events
        assert logs[0]["asked_seconds"] == pytest.approx(5.0)

    @pytest.mark.parametrize("start", [3.0, 4.5])
    def test_a_start_at_or_past_the_end_is_refused(self, start):
        with pytest.raises(ValueError, match='past_end\' "excerpt"'):
            slice_audio(
                tone(3.0),
                start_seconds=start,
                duration_seconds=1.0,
                sample_rate=RATE,
                past_end="excerpt",
            )

    def test_validate_accepts_excerpt_and_refuses_a_zero_length(self):
        def workflow(duration):
            return {
                "steps": [
                    {
                        "name": "bed_slice",
                        "task": {
                            "command": "slice_audio",
                            "arguments": {
                                "audio": "asset:a.wav",
                                "duration_seconds": duration,
                                "past_end": "excerpt",
                            },
                        },
                    }
                ]
            }

        assert task_argument_errors(workflow(1.0)) == []
        errors = task_argument_errors(workflow(0))
        assert [e["path"] for e in errors] == [
            "steps[0].task.arguments.duration_seconds"
        ], errors


def load_template():
    with open(TEMPLATE) as f:
        return json.load(f)


class TestAssembleAndScoreBed:
    def test_bed_variables_default_off_as_floats(self):
        variables = load_template()["variables"]
        assert variables["bed_start_seconds"] == 0.0
        assert variables["bed_seconds"] == 1.0
        assert variables["bed_gain"] == 0.0
        for name in ("bed_start_seconds", "bed_seconds", "bed_gain"):
            assert isinstance(variables[name], float), name

    def test_fractional_bed_arguments_are_accepted(self):
        from dw.variables import argument_errors

        errors = argument_errors(
            load_template(),
            {"bed_gain": 0.5, "bed_start_seconds": 2.5, "bed_seconds": 1.25},
        )
        assert errors == []

    def test_the_bed_chain_feeds_the_mix(self):
        steps = {s["name"]: s for s in load_template()["steps"]}
        bed_slice = steps["bed_slice"]["task"]
        assert bed_slice["command"] == "slice_audio"
        assert bed_slice["arguments"]["audio"] == "previous_result:edit"
        assert bed_slice["arguments"]["past_end"] == "excerpt"
        assert steps["bed_resampled"]["task"]["command"] == "resample_audio"
        assert steps["bed"]["task"]["command"] == "loop_audio"
        mixed = steps["mixed"]["task"]["arguments"]
        assert mixed["audios"][-1] == "previous_result:bed"
        assert mixed["gains"][-1] == "variable:bed_gain"

    def test_cost_is_unchanged(self):
        assert load_template()["cost"][0]["minutes"] == 0.2


class TestBedThroughTheRealChain:
    """The template's audio steps run in order on a real cut's track."""

    RATE = 44100

    def cut_and_score(self):
        from dw.media_types import AudioVideo
        from dw.tasks.concat_videos import concat_videos

        rng = numpy.random.default_rng(7)
        samples = round(3.5 * self.RATE)
        frames = numpy.zeros((84, 64, 64, 3), dtype=numpy.uint8)
        # a room floor under the shot, so the bed has sound to loop
        floor = (0.05 * rng.standard_normal((2, samples))).astype(numpy.float32)
        shot = AudioVideo(frames, floor, self.RATE, fps=24)
        cut = concat_videos([shot], fps=24, sample_rate=self.RATE)
        score = numpy.full((2, samples), 0.1, dtype=numpy.float32)
        return cut, score

    def run_chain(self, bed_gain, bed_start=0.0, bed_seconds=1.0):
        from dw.tasks.audio_utils import loop_audio, mix_audio, resample_audio

        cut, score = self.cut_and_score()
        world = resample_audio(cut, self.RATE)

        def bed_chain():
            bed_slice = slice_audio(
                cut,
                start_seconds=bed_start,
                duration_seconds=bed_seconds,
                past_end="excerpt",
            )
            resampled = resample_audio(bed_slice, self.RATE)
            return loop_audio(resampled, target_frames=84, fps=24)

        bed, events = events_of(bed_chain)
        mixed = mix_audio(
            [score, world, bed],
            gains=[1.0, 1.8, bed_gain],
            sample_rate=self.RATE,
        )
        two_track = mix_audio([score, world], gains=[1.0, 1.8], sample_rate=self.RATE)
        return mixed, two_track, events

    @staticmethod
    def samples(audio):
        return getattr(audio, "audio", audio)

    def test_default_gain_leaves_the_mix_unchanged_and_quiet(self):
        mixed, two_track, events = self.run_chain(bed_gain=0.0)
        numpy.testing.assert_array_equal(self.samples(mixed), self.samples(two_track))
        assert [e for e in events if e.get("event") == "warning"] == []

    def test_a_bed_gain_adds_the_bed(self):
        mixed, two_track, _ = self.run_chain(bed_gain=1.0)
        assert self.samples(mixed).shape == self.samples(two_track).shape
        assert not numpy.array_equal(self.samples(mixed), self.samples(two_track))


class TestRefusalsNameTheBed:
    """A refusal says which template variable or step is at fault (#1719
    bounce, C-F1403): the caller set `bed_seconds`, not steps[5]."""

    def test_validate_names_the_step_and_the_variable(self, tmp_path):
        from dw.validation import workflow_errors
        from dw.workflow import workflow_from_file

        workflow = workflow_from_file(str(TEMPLATE), str(tmp_path))
        errors = workflow_errors(workflow, {"bed_seconds": 0})
        bed = [e for e in errors if "duration_seconds" in e["path"]]
        assert bed, errors
        assert "bed_slice" in bed[0]["message"], bed
        assert "bed_seconds" in bed[0]["message"], bed

    def test_a_literal_argument_names_only_the_step(self):
        workflow = {
            "steps": [
                {
                    "name": "bed_slice",
                    "task": {
                        "command": "slice_audio",
                        "arguments": {"audio": "asset:a.wav", "duration_seconds": 0},
                    },
                }
            ]
        }
        (error,) = task_argument_errors(workflow, written=workflow)
        assert error["message"].startswith("Step 'bed_slice': "), error
        assert "set by variable" not in error["message"]

    def test_failed_step_label_nests_outermost_first(self):
        from dw.workflow_run import failed_step_label, note_failed_step

        error = ValueError("boom")
        assert failed_step_label(error) == ""
        note_failed_step(error, "bed_slice")
        note_failed_step(error, "score")
        assert failed_step_label(error) == "Step 'score' > 'bed_slice': "
        assert str(error) == "boom"

    def test_a_running_step_that_fails_carries_its_name(self, monkeypatch):
        from dw import workflow_run

        def refuse(*args, **kwargs):
            slice_audio(
                tone(3.0),
                start_seconds=100.0,
                duration_seconds=1.0,
                sample_rate=RATE,
                past_end="excerpt",
            )

        monkeypatch.setattr(workflow_run, "_start_step", refuse)
        with pytest.raises(ValueError) as caught:
            workflow_run.run_step(None, None, 5, {"name": "bed_slice"}, None)
        label = workflow_run.failed_step_label(caught.value)
        assert label == "Step 'bed_slice': "
        assert 'past_end\' "excerpt"' in str(caught.value)
