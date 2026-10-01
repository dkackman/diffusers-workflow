"""#467: assemble-and-score hardcoded peak_dbfs and exposed no way for a
caller to pass target_lufs to the 'balanced' normalize_audio step, so two
peak-normalized episodes with different crest factors (a studio-audience
laugh setting one episode's peak) could not be loudness-matched - target_lufs
alone can gain a mix down to a shared loudness, but the template gave it
nowhere to land."""

import json
import os

import pytest

TEMPLATE_PATH = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "workflows",
        "templates",
        "assemble-and-score.json",
    )
)


def load_definition():
    with open(TEMPLATE_PATH) as f:
        return json.load(f)


def steps_by_name(definition):
    return {s["name"]: s for s in definition["steps"]}


def balanced_arguments(variables):
    from dw.variables import replace_variables, resolve_variable_values

    definition = load_definition()
    merged = {**definition["variables"], **variables}
    merged = resolve_variable_values(merged)
    substituted = replace_variables(definition, merged)
    return steps_by_name(substituted)["balanced"]["task"]["arguments"]


def test_target_lufs_declared_and_defaults_to_null():
    definition = load_definition()
    assert definition["variables"]["target_lufs"] is None


def test_target_lufs_reaches_the_balanced_step():
    arguments = balanced_arguments({"target_lufs": -21})
    assert arguments["target_lufs"] == -21


def test_peak_dbfs_ceiling_is_unchanged_by_target_lufs():
    arguments = balanced_arguments({"target_lufs": -21})
    assert arguments["peak_dbfs"] == -3.0


def test_omitting_target_lufs_keeps_the_old_peak_only_result():
    # An unset (null) variable: reference is dropped from the arguments
    # entirely (dw/variables.py replace_variables), so normalize_audio falls
    # back to its own target_lufs=None default - unchanged, peak-only
    # behavior for a caller that never mentions target_lufs.
    arguments = balanced_arguments({})
    assert "target_lufs" not in arguments
    assert arguments["peak_dbfs"] == -3.0


# #497 (#474 stage 2): 'limit' is exposed beside target_lufs, so a cut can
# reach a target past the transient that would otherwise cap its gain.


def test_limit_declared_and_defaults_to_false():
    definition = load_definition()
    assert definition["variables"]["limit"] is False


def test_limit_reaches_the_balanced_step():
    arguments = balanced_arguments({"target_lufs": -16, "limit": True})
    assert arguments["limit"] is True
    assert arguments["target_lufs"] == -16
    assert arguments["peak_dbfs"] == -3.0


def test_the_default_leaves_the_balanced_step_unlimited():
    assert balanced_arguments({})["limit"] is False


def test_the_balanced_step_limits_a_laugh_track_to_the_target():
    # The real path: the arguments the template hands 'balanced', applied
    # by normalize_audio to a quiet bed with one burst 20 dB above it (the
    # laugh that capped #467's episodes). Without limit it caps; with it
    # the mix lands on -16 LUFS under a -3 dBTP ceiling.
    import numpy
    import scipy.signal

    from dw.dsp import integrated_lufs
    from dw.tasks.audio_utils import normalize_audio

    rate = 48000
    t = numpy.arange(rate * 4) / rate
    wave = 0.05 * numpy.sin(2 * numpy.pi * 1000 * t)
    wave[rate * 2 : rate * 2 + rate // 50] *= 10
    wave = numpy.tile(wave.astype(numpy.float32), (2, 1))

    def balanced(variables):
        arguments = balanced_arguments(variables)
        arguments.update(audio=wave, sample_rate=rate)
        return numpy.atleast_2d(numpy.asarray(normalize_audio(**arguments).audio))

    capped = balanced({"target_lufs": -16})
    assert integrated_lufs(capped.T, rate) < -17

    limited = balanced({"target_lufs": -16, "limit": True})
    assert integrated_lufs(limited.T, rate) == pytest.approx(-16, abs=0.5)
    true_peak = numpy.abs(scipy.signal.resample_poly(limited, 4, 1, axis=1)).max()
    assert 20 * numpy.log10(true_peak) <= -3.0 + 0.05
    assert limited.shape == wave.shape


def test_the_description_places_the_limit_ceiling_on_the_mix_not_the_film():
    # #497 v2 (#474 Q6): the limiter holds -3 dBTP on the mix; the AAC mux
    # of the film measured -2.54 dBTP, so the description must say where the
    # ceiling holds rather than promise it on the film.
    description = load_definition()["description"]
    assert "holds on the mix 'balanced' writes, not on the film" in description
    assert "about 1 dB above it" in description
