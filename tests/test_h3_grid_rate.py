"""An audio pass treats a source within a half percent of 24 fps as 24 fps (#1232)."""

from types import SimpleNamespace

import numpy

from dw.pipeline_processors.h3_guides import grid_rate, source_frame_rate
from dw.pipeline_processors.h3_rules import H3_FPS


def test_near_24_is_24_and_other_rates_are_kept():
    assert grid_rate(23.976) == H3_FPS
    assert grid_rate(24000 / 1001) == H3_FPS
    assert grid_rate(23.899) == H3_FPS
    assert grid_rate(25) == 25
    assert grid_rate(30) == 30
    assert grid_rate(None) is None


def test_a_carried_ntsc_rate_does_not_retime():
    video = SimpleNamespace(fps=24000 / 1001)
    assert source_frame_rate(video, 100) == H3_FPS


def test_the_soundtrack_length_fallback_is_snapped():
    # 143 frames against 5.984s of padded AAC is 23.899 fps
    audio = numpy.zeros((1, int(5.984 * 48000)), dtype=numpy.float32)
    video = SimpleNamespace(audio=audio, sample_rate=48000)
    assert source_frame_rate(video, 143) == H3_FPS
    assert source_frame_rate(video, 150) == 150 / 5.984
