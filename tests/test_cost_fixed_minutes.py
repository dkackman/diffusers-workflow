"""A `cost` entry's `fixed_minutes` stays out of the scaling (#1224)."""

import pytest

from dw.cost_scaling import _driver_scaling, _scaled, fixed_minutes
from dw.plan import _own_price


def _definition(**cost):
    return {
        "variables": {"num_inference_steps": 5, "num_frames": 124},
        "cost_drivers": ["num_inference_steps", "num_frames"],
        "cost": [{"device": "cuda", "vram_gb": 24, "minutes": 13.5, **cost}],
    }


def _quote(definition, **moved):
    variables = {**definition["variables"], **moved}
    return _own_price(definition, {"variables": variables}, {}, "cuda", {})


def test_the_fixed_share_is_not_multiplied():
    plain = _quote(_definition(), num_inference_steps=12, num_frames=345)
    fixed = _quote(
        _definition(fixed_minutes=9.0), num_inference_steps=12, num_frames=345
    )
    factor = 12 / 5 * 345 / 124
    assert plain["minutes"] == pytest.approx(13.5 * factor)
    assert fixed["minutes"] == pytest.approx(9.0 + 4.5 * factor)
    assert "9 min of it fixed" in fixed["scaled"]
    assert fixed["basis"] == "derived"


def test_a_run_at_the_defaults_is_the_curated_figure():
    assert _quote(_definition(fixed_minutes=9.0))["minutes"] == 13.5


def test_fixed_minutes_reads_the_entry_it_is_given():
    assert fixed_minutes({"minutes": 5, "fixed_minutes": 9}) == 9.0
    assert fixed_minutes(None) == 0.0
    assert fixed_minutes({"minutes": 1}) == 0.0
    assert fixed_minutes({"minutes": 1, "fixed_minutes": -2}) == 0.0
    assert fixed_minutes({"minutes": 1, "fixed_minutes": True}) == 0.0


def test_the_fixed_share_comes_from_the_entry_price_quoted():
    # only the other device's entry declares it: the cuda entry is quoted
    # from, so nothing is subtracted from its minutes
    cost = [
        {"device": "mps", "vram_gb": 0, "minutes": 5, "fixed_minutes": 4},
        {"device": "cuda", "vram_gb": 24, "minutes": 13.5},
    ]
    definition = {**_definition(), "cost": cost}
    moved = {"num_inference_steps": 10}
    cuda = _own_price(
        definition, {"variables": {**definition["variables"], **moved}}, {}, "cuda", {}
    )
    assert cuda["minutes"] == pytest.approx(13.5 * 2)
    assert "fixed" not in cuda["scaled"]
    # no entry for cpu: the first (mps) is quoted from and its share is used
    cpu = _own_price(
        definition, {"variables": {**definition["variables"], **moved}}, {}, "cpu", {}
    )
    assert cpu["basis"] == "other_device"
    assert cpu["minutes"] == pytest.approx(4 + 1 * 2)


def test_a_fixed_share_above_the_figure_is_capped():
    scaling = _driver_scaling(
        _definition(), {"variables": {"num_inference_steps": 10, "num_frames": 124}}, {}
    )
    price = {"minutes": 4.0, "basis": "catalog", "measured_on": None}
    assert _scaled(price, scaling, 99.0)["minutes"] == pytest.approx(4.0)


def test_the_schema_accepts_it_and_refuses_a_negative():
    import json
    from pathlib import Path

    import jsonschema

    schema = json.loads(
        (Path(__file__).resolve().parent.parent / "dw/workflow_schema.json").read_text()
    )
    item = schema["properties"]["cost"]["items"]
    ok = {"device": "cuda", "vram_gb": 24, "minutes": 13.5, "fixed_minutes": 9}
    jsonschema.validate(ok, item)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({**ok, "fixed_minutes": -1}, item)


def test_the_h3_chain_template_quotes_the_measured_runs_within_1_3x():
    """The 7 x 345 f, 12 step drill took 71.0 min (ab7f54f73b2f, with the check
    on); the same chain at 3 segments took 28.3 min (8a669d31258e)."""
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    definition = json.loads(
        (root / "workflows/templates/minimax/chained-segments.json").read_text()
    )
    base = {"num_inference_steps": 12, "num_frames": 345, "lora_model_name": None}
    three = _quote(definition, segments=3, **base)["minutes"]
    seven = _quote(definition, segments=7, **base)["minutes"]
    assert 28.3 / 1.3 < three < 28.3 * 1.3
    assert 71.0 / 1.3 < seven < 71.0 * 1.3


def _words_chain(check, retries=2, segments=3, **chain):
    chain = {"segments": segments, "fps": 4, "retries": retries, **chain}
    if check is not None:
        chain["check"] = check
    return {
        "variables": {},
        "steps": [{"pipeline": {"chain": chain, "arguments": {}}}],
    }


class TestWordsCheckTerm:
    ON = {"words_similarity_min": 0.8}

    def test_every_segment_attempt_is_a_pass_segment_1_included(self):
        from dw.cost_scaling import WORDS_CHECK_MINUTES, words_check_term

        minutes, axis = words_check_term(_words_chain(self.ON), {})
        # 3 segments, 1 + 2 retries each
        assert minutes == pytest.approx(9 * WORDS_CHECK_MINUTES) == pytest.approx(0.45)
        assert "9 Whisper pass" in axis and "words check" in axis

    def test_retries_scale_the_passes(self):
        from dw.cost_scaling import words_check_term

        assert words_check_term(_words_chain(self.ON, retries=0), {})[0] == (
            pytest.approx(0.15)
        )
        assert words_check_term(_words_chain(self.ON, retries=3, segments=5), {})[
            0
        ] == pytest.approx(1.0)

    @pytest.mark.parametrize(
        "check", [None, False, True, {"words_similarity_min": None}]
    )
    def test_off_is_nothing(self, check):
        from dw.cost_scaling import words_check_term

        assert words_check_term(_words_chain(check), {}) == (0.0, None)

    def test_held_audio_is_nothing(self):
        from dw.cost_scaling import words_check_term

        definition = _words_chain(self.ON, match_audio=True)
        assert words_check_term(definition, {}) == (0.0, None)

    def test_references_resolve_at_the_variables(self):
        from dw.cost_scaling import words_check_term

        definition = _words_chain({"words_similarity_min": "variable:w"})
        assert words_check_term(definition, {"w": None}) == (0.0, None)
        assert words_check_term(definition, {"w": 0.7})[0] == pytest.approx(0.45)

    def test_a_resumed_chain_leaves_the_kept_segments_out(self, monkeypatch):
        from dw.cost_scaling import words_check_term
        from dw.pipeline_processors import chain_resume

        monkeypatch.setattr(chain_resume, "kept_count", lambda chain: 2)
        definition = _words_chain(self.ON, resume_from="output:run/x.mp4")
        # 3 - 2 = 1 segment left, 1 + 2 attempts
        assert words_check_term(definition, {})[0] == pytest.approx(0.15)
        monkeypatch.setattr(chain_resume, "kept_count", lambda chain: 3)
        assert words_check_term(definition, {}) == (0.0, None)

    def test_with_words_check_adds_the_minutes_and_names_the_term(self):
        from dw.cost_scaling import with_words_check

        price = {"minutes": 10.0, "basis": "catalog", "measured_on": None}
        out = with_words_check(price, _words_chain(self.ON), {"variables": {}})
        assert out["minutes"] == pytest.approx(10.45)
        assert out["basis"] == "derived"
        assert "words check" in out["scaled"]
        assert price["minutes"] == 10.0

    def test_with_words_check_appends_to_an_existing_scaled_note(self):
        from dw.cost_scaling import with_words_check

        price = {
            "minutes": 10.0,
            "basis": "derived",
            "measured_on": None,
            "scaled": "x2 steps",
        }
        out = with_words_check(price, _words_chain(self.ON), {"variables": {}})
        assert out["scaled"].startswith("x2 steps, words check")

    def test_with_words_check_keeps_the_other_device_basis(self):
        from dw.cost_scaling import OTHER_DEVICE, with_words_check

        price = {"minutes": 10.0, "basis": OTHER_DEVICE, "measured_on": None}
        out = with_words_check(price, _words_chain(self.ON), {"variables": {}})
        assert out["basis"] == OTHER_DEVICE

    def test_with_words_check_leaves_a_price_without_minutes_alone(self):
        from dw.cost_scaling import with_words_check

        price = {"minutes": None, "basis": "unknown", "measured_on": None}
        assert with_words_check(price, _words_chain(self.ON), {"variables": {}}) == (
            price
        )

    def test_with_words_check_leaves_an_unchecked_chain_alone(self):
        from dw.cost_scaling import with_words_check

        price = {"minutes": 10.0, "basis": "catalog", "measured_on": None}
        out = with_words_check(price, _words_chain(True), {"variables": {}})
        assert out == price
