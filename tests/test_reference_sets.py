"""Each checker skips exactly the references its pass cannot see yet - the
set it uses is references.<SET> itself, not a copy that can drift."""

import pytest

from dw import (
    adapter_compatibility,
    content_types,
    kernel_availability,
    probe_paths,
    reference_limits,
    reference_names,
    references,
    result_fps,
    subfolders,
    video_extensions,
)


@pytest.mark.parametrize(
    "module",
    [adapter_compatibility, reference_names, reference_limits, video_extensions],
)
def test_the_unresolved_checkers_share_one_set(module):
    assert module._UNRESOLVED_PREFIXES is references.UNRESOLVED


@pytest.mark.parametrize(
    "module", [content_types, kernel_availability, result_fps, subfolders]
)
def test_the_substituted_checkers_still_check_step_results(module):
    assert module._UNRESOLVED_PREFIXES is references.SUBSTITUTED


def test_probe_paths_public_name_is_the_shared_set():
    assert probe_paths.UNRESOLVED_PREFIXES is references.UNRESOLVED
