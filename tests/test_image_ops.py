"""Tests for dw/tasks/image_ops.py (#775): the luma, alpha and per-frame
helpers grade, finish and lut share, and that moving them onto it changed no
output byte.
"""

from pathlib import Path

import numpy
from PIL import Image

from dw.media_types import AudioVideo
from dw.tasks import image_ops
from dw.tasks.finish import film_grain, grain_frame, sharpen_image
from dw.tasks.grade import grade_image
from dw.tasks.image_ops import join_alpha, luma, per_frame, split_alpha
from dw.tasks.lut import LUMA_WEIGHTS, apply_lut

FIXTURE = Path(__file__).parent / "fixtures" / "image_ops_outputs.npz"
PALETTE = ["#102030", "#a05020", "#f0e0c0"]


def _source():
    """24x16 RGBA noise whose alpha varies and is never all 255."""
    rng = numpy.random.default_rng(0)
    array = rng.integers(0, 256, size=(16, 24, 4), dtype=numpy.uint8)
    array[..., 3] = (numpy.arange(16 * 24) % 200 + 20).reshape(16, 24)
    return Image.fromarray(array, mode="RGBA")


def _outputs(image):
    return {
        "grade": grade_image(
            image,
            exposure=0.4,
            contrast=1.2,
            saturation=1.3,
            temperature=0.3,
            tint=-0.2,
            highlights=-0.3,
            shadows=0.4,
            whites=0.2,
            blacks=-0.1,
            clarity=0.5,
            vignette=0.3,
            fade=0.2,
        ),
        "sharpen": sharpen_image(image, amount=1.5, radius=2.0),
        "grain_frame": grain_frame(
            image, numpy.random.default_rng(7), amount=0.3, size=1.0, chroma=0.5
        ),
        "film_grain": film_grain(image, amount=0.3, size=2.0, chroma=0.0, seed=7),
        "lut": apply_lut(image, palette=PALETTE, strength=0.8),
    }


class TestLuma:
    def test_lut_re_exports_the_one_float64_weights(self):
        assert LUMA_WEIGHTS is image_ops.LUMA_WEIGHTS
        assert LUMA_WEIGHTS.dtype == numpy.float64

    def test_float32_input_stays_float32_and_matches_float32_weights(self):
        rgb = numpy.random.default_rng(1).random((5, 6, 3)).astype(numpy.float32)
        old = numpy.tensordot(rgb, LUMA_WEIGHTS.astype(numpy.float32), axes=([-1], [0]))
        result = luma(rgb)
        assert result.dtype == numpy.float32
        assert numpy.array_equal(result, old)

    def test_float64_input_matches_the_matmul(self):
        rgb = numpy.random.default_rng(2).random((5, 6, 3))
        result = luma(rgb)
        assert result.dtype == numpy.float64
        assert numpy.allclose(result, rgb @ LUMA_WEIGHTS, atol=1e-12, rtol=0)


class TestAlpha:
    def test_rgba_round_trip_keeps_the_alpha_bytes(self):
        image = _source()
        rgb, alpha = split_alpha(image)
        assert rgb.dtype == numpy.float32
        out = join_alpha(rgb, alpha)
        assert out.mode == "RGBA"
        assert numpy.array_equal(numpy.asarray(out), numpy.asarray(image))

    def test_rgb_has_no_alpha(self):
        rgb, alpha = split_alpha(_source().convert("RGB"))
        assert alpha is None
        assert join_alpha(rgb, alpha).mode == "RGB"


class TestPerFrame:
    def test_a_single_image_is_processed_as_itself(self):
        image = _source()
        assert per_frame(image, lambda frame: frame.resize((4, 4))).size == (4, 4)

    def test_a_video_comes_back_as_a_video_with_the_same_frames_and_audio(self):
        audio = numpy.full((2, 50), 0.25, dtype=numpy.float32)
        source = AudioVideo([_source() for _ in range(3)], audio, 100, fps=12)
        result = per_frame(source, lambda frame: frame.resize((4, 4)))
        assert isinstance(result, AudioVideo)
        assert len(result.frames) == 3
        assert result.fps == 12
        assert result.sample_rate == 100
        assert numpy.array_equal(result.audio, audio)

    def test_a_frame_list_comes_back_as_a_video(self):
        result = per_frame([_source(), _source()], lambda frame: frame)
        assert isinstance(result, AudioVideo)
        assert len(result.frames) == 2


class TestOutputsAreByteIdentical:
    """The fixture was captured from grade, finish and lut before they moved
    onto image_ops; never regenerate it to make this pass."""

    def test_every_command_matches_the_pre_refactor_bytes(self):
        image = _source()
        expected = numpy.load(FIXTURE)
        outputs = _outputs(image)
        assert set(outputs) == set(expected.files)
        alpha = numpy.asarray(image)[..., 3]
        for name, out in outputs.items():
            assert out.mode == "RGBA", name
            assert numpy.array_equal(numpy.asarray(out), expected[name]), name
            assert numpy.array_equal(numpy.asarray(out)[..., 3], alpha), name
