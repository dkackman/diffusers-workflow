"""Which reference set each checker skips, observed through its real entry
point rather than by comparing a constant.

A checker that runs on the substituted definition skips values still spelled
`variable:`/`item:` (SUBSTITUTED) or, when it also cannot see an earlier
step's result, `previous_result:`/`gather:` (UNRESOLVED). Each probe answers
whether the checker acted on a value, so swapping one site's set for the
other changes an answer here.
"""

import pytest

from dw import (
    adapter_compatibility,
    content_types,
    kernel_availability,
    probe_paths,
    reference_limits,
    reference_names,
    step_value_checks,
    subfolders,
    type_helpers,
    video_extensions,
)

_STEP_CHECKS = {
    "content_type": content_types.content_type_errors,
    "fps": step_value_checks.fps_errors,
    "subfolder": subfolders.subfolder_errors,
}


def _step_result(key):
    def probe(v, **_):
        return bool(_STEP_CHECKS[key]({"steps": [{"name": "s", "result": {key: v}}]}))

    return probe


def _kernel(v, monkeypatch, **_):
    monkeypatch.setattr(kernel_availability, "_fault_for_name", lambda name: "fault")
    return bool(kernel_availability.kernel_availability_fault(v))


def _adapter(v, **_):
    step = {
        "name": "s",
        "pipeline": {
            "from_pretrained_arguments": {"workflow": "ref2va"},
            "loras": [{"model_name": "org/lora", "weight_name": v}],
        },
    }
    return bool(adapter_compatibility.adapter_warnings({"steps": [step]}))


def _names(v, **_):
    return reference_names.reference_fault("output:" + v) is not None


def _limits(v, monkeypatch, **_):
    monkeypatch.setattr(type_helpers, "load_type_from_full_name", lambda *a: object)
    return (
        reference_limits._reference_class({"reference_type": v + ".Kind"}) is not None
    )


def _video(v, **_):
    step = {"name": "s", "task": {"arguments": {"video": v + "/clip.zip"}}}
    return bool(video_extensions.video_extension_errors({"steps": [step]}))


def _probe(v, tmp_path, **_):
    (tmp_path / (v + ".png")).write_bytes(b"x")
    return probe_paths.resolve_probe_path(v + ".png", str(tmp_path)) is not None


SUBSTITUTED_ONLY = {
    "content_types": _step_result("content_type"),
    "step_value_checks": _step_result("fps"),
    "subfolders": _step_result("subfolder"),
    "kernel_availability": _kernel,
}
UNRESOLVED_TOO = {
    "adapter_compatibility": _adapter,
    "reference_names": _names,
    "reference_limits": _limits,
    "video_extensions": _video,
    "probe_paths": _probe,
}


@pytest.mark.parametrize("checker", sorted(SUBSTITUTED_ONLY))
def test_a_substituted_only_checker_still_reports_an_earlier_steps_reference(
    checker, monkeypatch, tmp_path
):
    probe = SUBSTITUTED_ONLY[checker]
    assert probe("previous_result:x", monkeypatch=monkeypatch, tmp_path=tmp_path)
    assert not probe("variable:x", monkeypatch=monkeypatch, tmp_path=tmp_path)


@pytest.mark.parametrize("checker", sorted(UNRESOLVED_TOO))
def test_an_unresolved_checker_leaves_an_earlier_steps_reference_alone(
    checker, monkeypatch, tmp_path
):
    probe = UNRESOLVED_TOO[checker]
    assert not probe("previous_result:x", monkeypatch=monkeypatch, tmp_path=tmp_path)
    assert not probe("gather:x", monkeypatch=monkeypatch, tmp_path=tmp_path)
    assert not probe("variable:x", monkeypatch=monkeypatch, tmp_path=tmp_path)
