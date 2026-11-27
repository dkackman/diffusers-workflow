"""#1619: a task-only workflow is priced from its commands' declared rates
(`register_command(cost=...)`), scaled by its input media's real length."""

import wave
from unittest.mock import patch

import numpy
import pytest

import dw.media
import dw.plan_downloads
from dw.plan import build_plan
from dw.plan_pricing import task_rate_price
from dw.server.observed_cost import observed_for
from dw.tasks.registry import RegistryTable, register_command
from dw.workflow import workflow_from_definition

from .test_observed_cost import MINUTE, run

RATES = RegistryTable("cost")


def write_wav(path, seconds, sample_rate=8000):
    samples = numpy.zeros(int(seconds * sample_rate), dtype="<i2")
    with wave.open(str(path), "w") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(samples.tobytes())


def step(name, command, **arguments):
    return {"name": name, "task": {"command": command, "arguments": arguments}}


def plan_for(tmp_path, steps, observed=None):
    inline = {"id": "finish", "steps": steps}
    candidate = workflow_from_definition(
        inline, str(tmp_path), str(tmp_path), str(tmp_path)
    )
    with patch.object(dw.plan_downloads, "scan_models", return_value={"repos": []}):
        return build_plan(
            candidate, {}, device="cpu", lookup_sizes=False, observed=observed
        )["estimate"]


def expected_minutes(seconds):
    import math

    return math.ceil(round(seconds / 60 * 10, 6)) / 10


class TestTheRegistry:
    def test_a_device_consuming_command_may_not_declare_a_cost(self):
        with pytest.raises(ValueError, match="consumes a device"):
            register_command(
                "test_1619_model",
                consumes_device=True,
                cost={"fixed_seconds": 1, "per_input_second": 1},
            )
        assert "test_1619_model" not in RATES

    def test_a_cost_must_name_both_figures(self):
        with pytest.raises(ValueError):
            register_command("test_1619_bad", cost={"fixed_seconds": 1})
        with pytest.raises(ValueError):
            register_command(
                "test_1619_bad", cost={"fixed_seconds": -1, "per_input_second": 0}
            )

    def test_the_measured_finishing_commands_declare_one(self):
        for command in ("apply_lut", "film_grain", "export_timeline"):
            assert set(RATES[command]) == {"fixed_seconds", "per_input_second"}

    def test_a_model_command_declares_none(self):
        assert "separate_stems" not in RATES


def test_the_price_reads_media_by_the_location_checks_rule():
    """One predicate owns which arguments are media: the declared-rate probe
    asks dw/locations.py rather than writing the rule out again."""
    import dw.plan_pricing
    from dw.locations import is_media_argument

    assert not hasattr(dw.plan_pricing, "_is_media_argument")
    assert is_media_argument("media", "apply_lut")
    assert not is_media_argument("media")
    assert is_media_argument("latent_anchor")
    assert is_media_argument("input_video")
    assert not is_media_argument("strength", "film_grain")


class TestThePrice:
    def test_two_steps_sum_scaled_by_the_inputs_length(self, tmp_path):
        write_wav(tmp_path / "a.wav", 62)
        steps = [
            step("graded", "apply_lut", media="a.wav", palette=["#000000", "#ffffff"]),
            step("grain", "film_grain", media="previous_result:graded"),
        ]
        estimate = plan_for(tmp_path, steps)
        lut, grain = RATES["apply_lut"], RATES["film_grain"]
        seconds = (
            lut["fixed_seconds"]
            + lut["per_input_second"] * 62
            + grain["fixed_seconds"]
            + grain["per_input_second"] * 62
        )
        assert estimate["minutes"] == expected_minutes(seconds)
        assert estimate["basis"] == "derived"
        assert estimate["low_confidence"] is True
        assert "unpriced_reason" not in estimate
        assert estimate["scaled"].startswith("task rates: apply_lut ")
        assert "s/input-s over 62 s" in estimate["scaled"]
        assert "film_grain" in estimate["scaled"]

    def test_a_longer_input_prices_higher(self, tmp_path):
        write_wav(tmp_path / "short.wav", 10)
        write_wav(tmp_path / "long.wav", 120)
        short = plan_for(tmp_path, [step("g", "film_grain", media="short.wav")])
        long = plan_for(tmp_path, [step("g", "film_grain", media="long.wav")])
        assert long["minutes"] > short["minutes"]

    def test_an_earlier_steps_output_takes_the_longest_resolved_length(self, tmp_path):
        write_wav(tmp_path / "a.wav", 30)
        write_wav(tmp_path / "b.wav", 90)
        steps = [
            # Each step feeds the next, so the plan elides none of them
            step("a", "resample_audio", audio="a.wav", target_sample_rate=44100),
            step("pair", "pair_audio", video="b.wav", audio="previous_result:a"),
            step("grain", "film_grain", media="previous_result:pair"),
        ]
        estimate = plan_for(tmp_path, steps)
        resample, pair, grain = (
            RATES["resample_audio"],
            RATES["pair_audio"],
            RATES["film_grain"],
        )
        seconds = (
            resample["fixed_seconds"]
            + resample["per_input_second"] * 30
            + pair["fixed_seconds"]
            + pair["per_input_second"] * 90
            + grain["fixed_seconds"]
            + grain["per_input_second"] * 90
        )
        assert estimate["minutes"] == expected_minutes(seconds)
        assert estimate["low_confidence"] is True
        assert [line.split(":")[0] for line in estimate["differs"]] == ["grain"]
        assert "(90 s)" in estimate["differs"][0]

    def test_no_input_that_resolves_prices_the_fixed_seconds_alone(self, tmp_path):
        estimate = plan_for(tmp_path, [step("grain", "film_grain", media="gone.wav")])
        assert estimate["minutes"] == expected_minutes(
            RATES["film_grain"]["fixed_seconds"]
        )
        assert estimate["basis"] == "derived"
        assert estimate["low_confidence"] is True
        assert "fixed" in estimate["differs"][0]

    def test_an_undeclared_command_leaves_it_unknown_and_named(self, tmp_path):
        write_wav(tmp_path / "a.wav", 10)
        steps = [
            step("grain", "film_grain", media="a.wav"),
            step("split", "separate_stems", audio="a.wav"),
        ]
        estimate = plan_for(tmp_path, steps)
        assert estimate["minutes"] is None and estimate["basis"] == "unknown"
        assert estimate["unpriced_reason"].startswith(
            "no declared rate: separate_stems"
        )
        assert "film_grain" not in estimate["unpriced_reason"].split(" - ")[0]

    def test_a_pipeline_workflow_is_not_task_priced(self):
        assert task_rate_price({"steps": [{"name": "p", "pipeline": {}}]}) is None

    def test_this_boxs_history_still_wins(self, tmp_path):
        write_wav(tmp_path / "a.wav", 10)
        steps = [step("grain", "film_grain", media="a.wav")]
        definition = {"id": "finish", "steps": steps}
        block = observed_for(definition, [run(5 * MINUTE)] * 5, device="cpu")
        estimate = plan_for(tmp_path, steps, observed=lambda arguments: block)
        assert estimate["basis"] == "observed"
        assert estimate["minutes"] == 5.0


def test_the_probe_is_shared_across_steps(tmp_path):
    write_wav(tmp_path / "a.wav", 10)
    expanded = {
        "steps": [
            step("one", "film_grain", media="a.wav"),
            step("two", "film_grain", media="a.wav"),
        ]
    }
    probes = {}
    with patch.object(
        dw.media, "probe_metadata", wraps=dw.media.probe_metadata
    ) as probe:
        task_rate_price(expanded, str(tmp_path), probes)
    assert probe.call_count == 1
    assert len(probes) == 1
