"""Fractional frame rates are written exactly, not floored (#1109)."""

from fractions import Fraction

import av
import numpy
import pytest
import torch
from diffusers.utils import export_to_video

from dw.writers import encode_video, exact_frame_rate


def _frames(count=30):
    return numpy.zeros((count, 32, 32, 3), dtype=numpy.uint8)


def _rate(path):
    with av.open(str(path)) as container:
        return container.streams.video[0].average_rate


@pytest.mark.parametrize(
    "fps, expected",
    [
        (29.97, Fraction(30000, 1001)),
        (30000 / 1001, Fraction(30000, 1001)),
        (23.976, Fraction(24000, 1001)),
        (59.94, Fraction(60000, 1001)),
        (24, Fraction(24)),
        (24.0, Fraction(24)),
    ],
)
def test_exact_frame_rate(fps, expected):
    assert exact_frame_rate(fps) == expected


def test_encode_video_with_audio_keeps_fractional_rate(tmp_path):
    path = tmp_path / "out.mp4"
    encode_video(
        torch.from_numpy(_frames()),
        fps=30000 / 1001,
        output_path=str(path),
        audio=torch.zeros(2, 48000),
        audio_sample_rate=48000,
    )
    assert _rate(path) == Fraction(30000, 1001)
    with av.open(str(path)) as container:
        assert len(container.streams.audio) == 1


def test_encode_video_whole_rate_unchanged(tmp_path):
    path = tmp_path / "out.mp4"
    encode_video(torch.from_numpy(_frames()), fps=24, output_path=str(path))
    assert _rate(path) == 24


def test_export_to_video_keeps_fractional_rate(tmp_path):
    path = tmp_path / "out.mp4"
    export_to_video(list(_frames()), str(path), fps=30000 / 1001, macro_block_size=16)
    assert abs(float(_rate(path)) - 29.97) < 0.01


def test_no_dw_module_calls_the_diffusers_encode_video_directly():
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    pattern = re.compile(
        r"diffusers\.utils(\.export_utils)?\s+import[^\n]*\bencode_video\b"
    )
    offenders = [
        str(p.relative_to(root))
        for folder in ("dw", "dw_mcp")
        for p in (root / folder).rglob("*.py")
        if p.name != "writers.py" and pattern.search(p.read_text())
    ]
    assert offenders == []


def test_the_diffusers_hook_the_wrapper_swaps_still_exists():
    import inspect

    from diffusers.utils import export_utils

    assert callable(export_utils._import_av)
    assert 'add_stream("libx264"' in inspect.getsource(export_utils.encode_video)
