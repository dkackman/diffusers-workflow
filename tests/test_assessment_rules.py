"""Tests for #387: the assessment rules table and the probes it is read
against.

`dw/assessment_rules.py` names, for each rule, the probe that reports the
field, the field itself (a key of that probe's per-shot/per-seam record or
of its answer), how the value is compared and the threshold. A rule reading
a field a probe does not actually emit, or naming a probe that is not a
real `returns="json"` task command, would silently find nothing - this pins
every rule to the real probes in `dw/tasks/assess.py`.
"""

import numpy
import pytest
from PIL import Image

from dw.assessment_rules import (
    COMPARATORS,
    RULES,
    SEVERITIES,
    crosses,
    finding,
    rules_for,
)
from dw.introspection import list_tasks
from dw.result import AudioVideo
from dw.shots import shot_record
from dw.tasks.assess import analyze_seams, analyze_shots, analyze_sync_drift
from dw.tasks.audio_utils import LEVEL_SPREAD_WARN_DB
from dw.tasks.task import task_command_info

FPS = 24
SAMPLE_RATE = 8000
FRAMES_PER_SHOT = 24
SAMPLES_PER_SHOT = SAMPLE_RATE  # 1 second, matching FRAMES_PER_SHOT at FPS

# Distinct brightness per shot so a seam shows a real frame jump; constant
# within a shot so its own typical delta stays at the floor
_SHOT_LEVELS = (40, 210, 40)
# Distinct tone amplitude per shot so the shots differ in level
_SHOT_AMPLITUDES = (0.02, 0.5, 0.02)


def _frame(level):
    array = numpy.full((16, 16), level, dtype=numpy.uint8)
    return Image.fromarray(array, mode="L")


def _tone(amplitude, samples, sample_rate=SAMPLE_RATE, frequency=440.0):
    t = numpy.arange(samples, dtype=numpy.float32) / sample_rate
    return (amplitude * numpy.sin(2 * numpy.pi * frequency * t)).astype(numpy.float32)


def _synthetic_video():
    """A 3-shot AudioVideo small enough to run through the real probes."""
    frames = []
    audio_chunks = []
    shots = []
    for index, (level, amplitude) in enumerate(zip(_SHOT_LEVELS, _SHOT_AMPLITUDES)):
        frames.extend(_frame(level) for _ in range(FRAMES_PER_SHOT))
        audio_chunks.append(_tone(amplitude, SAMPLES_PER_SHOT))
        shots.append(
            shot_record(
                name=f"shot{index}",
                start_frame=index * FRAMES_PER_SHOT,
                num_frames=FRAMES_PER_SHOT,
                start_sample=index * SAMPLES_PER_SHOT,
                num_samples=SAMPLES_PER_SHOT,
            )
        )
    audio = numpy.concatenate(audio_chunks)
    return AudioVideo(frames, audio, SAMPLE_RATE, fps=FPS, shots=shots)


PROBES = {
    "analyze_shots": analyze_shots,
    "analyze_seams": analyze_seams,
    "analyze_sync_drift": analyze_sync_drift,
}


def _fields(answer):
    """Every key a probe's answer or its per-shot/per-seam records carry."""
    fields = set(answer.keys())
    for records_key in ("shots", "seams"):
        for record in answer.get(records_key) or []:
            fields.update(record.keys())
    return fields


@pytest.fixture(scope="module")
def probe_answers():
    video = _synthetic_video()
    return {name: probe(video) for name, probe in PROBES.items()}


class TestRuleProbesAreRealCommands:
    def test_every_rule_names_a_registered_json_command(self):
        assessment = list_tasks()["assessment"]
        for rule in RULES:
            assert rule["probe"] in assessment
            info = task_command_info(rule["probe"])
            assert info["returns"] == "json"

    def test_assessment_is_exactly_the_three_probes(self):
        assert list_tasks()["assessment"] == sorted(
            ["analyze_shots", "analyze_seams", "analyze_sync_drift"]
        )

    def test_probes_stay_in_commands_too(self):
        commands = list_tasks()["commands"]
        for name in list_tasks()["assessment"]:
            assert name in commands


class TestRuleFieldsAreReal:
    def test_every_rule_field_is_a_real_key_the_probe_emits(self, probe_answers):
        for rule in RULES:
            answer = probe_answers[rule["probe"]]
            assert rule["field"] in _fields(answer), (
                f"{rule['name']}: {rule['field']!r} is not a key {rule['probe']} emits"
            )

    def test_every_rule_name_appears_in_its_probe_answer(self, probe_answers):
        for rule in RULES:
            answer = probe_answers[rule["probe"]]
            assert rule["name"] in answer["rules_applied"]

    def test_rules_applied_matches_rules_for(self, probe_answers):
        for probe_name, answer in probe_answers.items():
            assert answer["rules_applied"] == [r["name"] for r in rules_for(probe_name)]


class TestShotLevelSpreadThreshold:
    def test_shares_the_run_time_warning_threshold(self):
        rule = next(r for r in RULES if r["name"] == "shot_level_spread")
        assert rule["threshold"] is LEVEL_SPREAD_WARN_DB


class TestTableShape:
    def test_severities_are_declared(self):
        for rule in RULES:
            assert rule["severity"] in SEVERITIES

    def test_comparators_are_declared(self):
        for rule in RULES:
            assert rule["comparator"] in COMPARATORS

    def test_rule_names_are_unique(self):
        names = [rule["name"] for rule in RULES]
        assert len(names) == len(set(names))


class TestCrosses:
    def test_none_never_crosses(self):
        for rule in RULES:
            assert crosses(rule, None) is False

    def test_magnitude_uses_abs(self):
        rule = {"comparator": ">", "threshold": 10.0, "magnitude": True}
        assert crosses(rule, -20.0) is True
        assert crosses(rule, -5.0) is False

    def test_greater_than(self):
        rule = {"comparator": ">", "threshold": 10.0}
        assert crosses(rule, 10.0) is False
        assert crosses(rule, 10.1) is True

    def test_greater_than_or_equal(self):
        rule = {"comparator": ">=", "threshold": 10.0}
        assert crosses(rule, 10.0) is True
        assert crosses(rule, 9.9) is False

    def test_less_than(self):
        rule = {"comparator": "<", "threshold": 10.0}
        assert crosses(rule, 9.9) is True
        assert crosses(rule, 10.0) is False


class TestFinding:
    def test_shape(self):
        rule = RULES[0]
        result = finding(rule, 12.5, {"seam": 1})
        assert result == {
            "rule": rule["name"],
            "severity": rule["severity"],
            "at": {"seam": 1},
            "value": 12.5,
            "threshold": rule["threshold"],
            "says": rule["says"],
        }
