"""#467: assemble-and-score hardcoded peak_dbfs and exposed no way for a
caller to pass target_lufs to the 'balanced' normalize_audio step, so two
peak-normalized episodes with different crest factors (a studio-audience
laugh setting one episode's peak) could not be loudness-matched - target_lufs
alone can gain a mix down to a shared loudness, but the template gave it
nowhere to land."""

import json
import os

TEMPLATE_PATH = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__), "..", "workflows", "templates", "assemble-and-score.json"
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
