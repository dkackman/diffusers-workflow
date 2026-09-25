"""A slice_audio source shorter than the requested slice, warned about at
validate time rather than only at run time - #402.

Exercises the real decode path (`probe_media` against a genuine wav file)
rather than a mock of it, so a fixture shorter than its declared slice is
the same file the run itself would have padded with silence.
"""

import os
import tempfile
import wave

import numpy

from dw.runs import activate_output_root, deactivate_output_root
from dw.slice_preflight import slice_past_end_warnings
from dw.workflow import workflow_from_definition


def write_wav(path, seconds=2.0, sample_rate=8000):
    t = numpy.arange(int(seconds * sample_rate)) / sample_rate
    samples = (numpy.sin(2 * numpy.pi * 220 * t) * 0.5 * 32767).astype("<i2")
    with wave.open(str(path), "w") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(numpy.stack([samples, samples], 1).tobytes())


def workflow_dir_with_asset(monkeypatch, name, seconds):
    base_dir = tempfile.mkdtemp()
    asset_dir = os.path.join(base_dir, "assets")
    os.makedirs(asset_dir)
    write_wav(os.path.join(asset_dir, name), seconds=seconds)
    monkeypatch.setenv("DW_ASSET_DIR", asset_dir)
    return base_dir


def slice_workflow(audio, **arguments):
    return {
        "id": "scoring",
        "steps": [
            {
                "name": "soundtrack",
                "task": {
                    "command": "slice_audio",
                    "arguments": {"audio": audio, **arguments},
                },
                "result": {"content_type": "audio/wav"},
            }
        ],
    }


class TestTheCheck:
    def test_a_frame_based_slice_past_a_short_asset_is_warned(self, monkeypatch):
        base_dir = workflow_dir_with_asset(monkeypatch, "score.wav", seconds=4.96)
        definition = slice_workflow(
            "asset:score.wav", start_frame=0, num_frames=372, fps=24
        )

        warnings = slice_past_end_warnings(definition, base_dir=base_dir)

        assert len(warnings) == 1
        assert "score.wav" in warnings[0]
        assert "4.96 s source" in warnings[0]
        assert "past the end" in warnings[0]

    def test_a_seconds_based_slice_past_a_short_asset_is_warned(self, monkeypatch):
        base_dir = workflow_dir_with_asset(monkeypatch, "voice.wav", seconds=2.0)
        definition = slice_workflow(
            "asset:voice.wav", start_seconds=0, duration_seconds=5.0
        )

        warnings = slice_past_end_warnings(definition, base_dir=base_dir)

        assert len(warnings) == 1
        assert "voice.wav" in warnings[0]

    def test_a_slice_within_the_source_validates_clean(self, monkeypatch):
        base_dir = workflow_dir_with_asset(monkeypatch, "score.wav", seconds=10.0)
        definition = slice_workflow(
            "asset:score.wav", start_frame=0, num_frames=120, fps=24
        )

        assert slice_past_end_warnings(definition, base_dir=base_dir) == []

    def test_a_sub_threshold_overrun_is_rounding_not_a_warning(self, monkeypatch):
        base_dir = workflow_dir_with_asset(monkeypatch, "score.wav", seconds=5.0)
        definition = slice_workflow(
            "asset:score.wav", start_seconds=0, duration_seconds=5.001
        )

        assert slice_past_end_warnings(definition, base_dir=base_dir) == []

    def test_no_length_given_runs_to_the_source_end_and_cannot_overrun(
        self, monkeypatch
    ):
        base_dir = workflow_dir_with_asset(monkeypatch, "score.wav", seconds=4.96)
        definition = slice_workflow("asset:score.wav", start_seconds=0)

        assert slice_past_end_warnings(definition, base_dir=base_dir) == []

    def test_a_previous_result_audio_is_left_to_the_run(self):
        definition = slice_workflow(
            "previous_result:make_track", start_frame=0, num_frames=372, fps=24
        )

        assert slice_past_end_warnings(definition) == []

    def test_an_output_reference_too_short_is_warned(self):
        output_root = tempfile.mkdtemp()
        run_dir = os.path.join(output_root, "score", "20260101-000000-abc")
        os.makedirs(run_dir)
        write_wav(os.path.join(run_dir, "bed.wav"), seconds=3.0)
        definition = slice_workflow(
            "output:score/20260101-000000-abc/bed.wav",
            start_seconds=0,
            duration_seconds=8.0,
        )

        token = activate_output_root(output_root)
        try:
            warnings = slice_past_end_warnings(definition)
        finally:
            deactivate_output_root(token)

        assert len(warnings) == 1
        assert "bed.wav" in warnings[0]

    def test_nothing_is_reported_for_a_definition_with_no_slice_step(self):
        assert slice_past_end_warnings({"steps": [{"name": "a", "task": {}}]}) == []


class TestWiredIntoTheWorkflow:
    def test_reachable_from_the_workflow_method(self, monkeypatch):
        base_dir = workflow_dir_with_asset(monkeypatch, "score.wav", seconds=4.96)
        definition = slice_workflow(
            "asset:score.wav", start_frame=0, num_frames=372, fps=24
        )
        workflow = workflow_from_definition(
            definition, os.path.join(base_dir, "workflow.json")
        )

        warnings = workflow.slice_past_end_warnings()

        assert len(warnings) == 1
        assert "score.wav" in warnings[0]
