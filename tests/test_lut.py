"""Tests for the apply_lut task command and its strict .cube parser (#603)."""

import os
from unittest.mock import patch

import numpy
import pytest
from PIL import Image

from dw.media_types import AudioVideo
from dw.security import InvalidInputError, PathTraversalError
from dw.task_domains import task_argument_errors
from dw.tasks.lut import MAX_CUBE_BYTES, CubeError, apply_lut, load_lut, parse_cube
from dw.tasks.task import Task
from dw.trust import TRUST_WORKFLOWS_ENV_VAR


def _cube(mapping, size=2):
    """A .cube text whose entry at (r, g, b) is mapping(r, g, b), each 0..1,
    listed red fastest."""
    text = f"LUT_3D_SIZE {size}\n"
    steps = [i / (size - 1) for i in range(size)]
    for b in steps:
        for g in steps:
            for r in steps:
                text += " ".join(f"{v:.6f}" for v in mapping(r, g, b)) + "\n"
    return text


def _identity(r, g, b):
    return (r, g, b)


def _swap(r, g, b):
    """Channel swap: red and blue trade places."""
    return (b, g, r)


def _invert(r, g, b):
    return (1 - r, 1 - g, 1 - b)


def _write(directory, name, text):
    path = os.path.join(directory, name)
    data = text if isinstance(text, bytes) else text.encode("utf-8")
    with open(path, "wb") as handle:
        handle.write(data)
    return path


def _gradient(width=32, height=24):
    """Every pixel a different colour, including both ends of each channel."""
    x = numpy.linspace(0, 255, width)
    y = numpy.linspace(0, 255, height)
    red = numpy.tile(x, (height, 1))
    green = numpy.tile(y[:, None], (1, width))
    blue = 255 - red
    array = numpy.stack([red, green, blue], axis=2).round().astype(numpy.uint8)
    return Image.fromarray(array, mode="RGB")


def _corners():
    """The eight corners of the RGB cube, one pixel each."""
    values = [(r, g, b) for r in (0, 255) for g in (0, 255) for b in (0, 255)]
    return Image.fromarray(numpy.array([values], dtype=numpy.uint8), mode="RGB")


class TestApply:
    def test_an_identity_lut_returns_the_input(self, tmp_path):
        image = _gradient()
        path = _write(tmp_path, "identity.cube", _cube(_identity, size=17))
        out = apply_lut(image, path)
        assert numpy.array_equal(numpy.asarray(out), numpy.asarray(image))

    def test_a_channel_swap_maps_every_corner_exactly(self, tmp_path):
        path = _write(tmp_path, "swap.cube", _cube(_swap, size=5))
        out = numpy.asarray(apply_lut(_corners(), path))
        expected = numpy.asarray(_corners())[..., ::-1]
        assert numpy.array_equal(out, expected)

    def test_the_lookup_interpolates_between_entries(self, tmp_path):
        """A 2-point invert LUT is linear, so trilinear interpolation inverts
        every value, not only the corners."""
        image = _gradient()
        path = _write(tmp_path, "invert.cube", _cube(_invert, size=2))
        out = numpy.asarray(apply_lut(image, path)).astype(int)
        assert numpy.abs(out - (255 - numpy.asarray(image).astype(int))).max() <= 1

    def test_strength_blends_original_and_result(self, tmp_path):
        image = _gradient()
        path = _write(tmp_path, "invert.cube", _cube(_invert))
        original = numpy.asarray(image)
        assert numpy.array_equal(
            numpy.asarray(apply_lut(image, path, strength=0)), original
        )
        half = numpy.asarray(apply_lut(image, path, strength=0.5)).astype(int)
        # Halfway between a value and its inverse is mid grey
        assert numpy.abs(half - 127.5).max() <= 1
        quarter = numpy.asarray(apply_lut(image, path, strength=0.25)).astype(int)
        expected = original * 0.75 + (255 - original.astype(float)) * 0.25
        assert numpy.abs(quarter - expected).max() <= 1

    def test_alpha_passes_through(self, tmp_path):
        image = _gradient().convert("RGBA")
        alpha = numpy.arange(24 * 32, dtype=numpy.uint32).reshape(24, 32) % 256
        image.putalpha(Image.fromarray(alpha.astype(numpy.uint8), mode="L"))
        path = _write(tmp_path, "invert.cube", _cube(_invert))
        out = apply_lut(image, path)
        assert out.mode == "RGBA"
        assert numpy.array_equal(
            numpy.asarray(out.getchannel("A")), numpy.asarray(image.getchannel("A"))
        )

    def test_a_video_keeps_its_frames_fps_and_audio(self, tmp_path):
        frames = [_gradient() for _ in range(3)]
        audio = numpy.full((2, 50), 0.25, dtype=numpy.float32)
        video = AudioVideo(frames, audio, 100, fps=12)
        path = _write(tmp_path, "invert.cube", _cube(_invert))
        out = Task({"command": "apply_lut", "arguments": {}}, "cpu").run(
            {"media": video, "lut": path}
        )
        assert isinstance(out, AudioVideo)
        assert len(out.frames) == 3
        assert out.fps == 12
        assert out.sample_rate == 100
        assert numpy.array_equal(out.audio, audio)
        first = numpy.asarray(out.frames[0]).astype(int)
        assert (
            numpy.abs(first - (255 - numpy.asarray(frames[0]).astype(int))).max() <= 1
        )

    def test_the_command_runs_on_an_image(self, tmp_path):
        image = _gradient()
        path = _write(tmp_path, "identity.cube", _cube(_identity, size=3))
        out = Task({"command": "apply_lut", "arguments": {}}, "cpu").run(
            {"media": image, "lut": path, "strength": 1.0}
        )
        assert numpy.array_equal(numpy.asarray(out), numpy.asarray(image))

    def test_the_command_refuses_a_missing_lut(self):
        with pytest.raises(ValueError, match="'lut' is required"):
            Task({"command": "apply_lut", "arguments": {}}, "cpu").run(
                {"media": _gradient()}
            )


def _errors(arguments):
    definition = {
        "id": "apply-lut-domains",
        "steps": [
            {
                "name": "look",
                "task": {"command": "apply_lut", "arguments": arguments},
                "result": {"content_type": "image/png"},
            }
        ],
    }
    return task_argument_errors(definition)


class TestStrengthDomain:
    @pytest.mark.parametrize("strength", [-0.1, 1.5])
    def test_strength_outside_zero_to_one_refuses_at_validate(self, strength):
        errors = _errors(
            {"media": "asset:a.png", "lut": "asset:a.cube", "strength": strength}
        )
        assert [e["path"] for e in errors] == ["steps[0].task.arguments.strength"]
        assert "between 0.0 and 1.0" in errors[0]["message"]

    def test_strength_inside_refuses_nothing(self):
        assert not _errors(
            {"media": "asset:a.png", "lut": "asset:a.cube", "strength": 0.5}
        )

    def test_a_run_time_strength_is_checked_too(self, tmp_path):
        path = _write(tmp_path, "identity.cube", _cube(_identity))
        with pytest.raises(ValueError, match="strength"):
            Task({"command": "apply_lut", "arguments": {}}, "cpu").run(
                {"media": _gradient(), "lut": path, "strength": 2}
            )


def _rows(size, value="0.5 0.5 0.5"):
    return "\n".join([value] * size**3) + "\n"


class TestParserRefusals:
    """Every refusal names the file and the line."""

    def _refusal(self, text, match):
        with pytest.raises(CubeError, match=match) as refusal:
            parse_cube(text, "look.cube")
        assert "look.cube" in str(refusal.value)
        return str(refusal.value)

    def test_a_well_formed_header_parses(self):
        text = (
            '# a comment\nTITLE "a look"\n\nDOMAIN_MIN 0 0 0\nDOMAIN_MAX 1 1 1\n'
            "LUT_3D_SIZE 2\n" + _rows(2)
        )
        assert parse_cube(text, "look.cube").shape == (2, 2, 2, 3)

    @pytest.mark.parametrize("size", [1, 66, 200])
    def test_a_size_out_of_bounds(self, size):
        message = self._refusal(f"LUT_3D_SIZE {size}\n", f"LUT_3D_SIZE {size}")
        assert "line 1" in message and "2..65" in message

    def test_a_missing_row(self):
        text = "LUT_3D_SIZE 2\n" + "\n".join(["0 0 0"] * 7) + "\n"
        message = self._refusal(text, "calls for 8")
        assert "line 8" in message and "7 data rows" in message

    def test_an_extra_row(self):
        text = "LUT_3D_SIZE 2\n" + _rows(2) + "0 0 0\n"
        assert "line 10" in self._refusal(text, "more than the 8")

    @pytest.mark.parametrize("bad", ["nan", "inf", "-inf", "NaN"])
    def test_a_value_that_is_not_finite(self, bad):
        text = "LUT_3D_SIZE 2\n0 0 0\n" + f"0.5 {bad} 0.5\n" + _rows(2)
        assert "line 3" in self._refusal(text, "not a finite number")

    @pytest.mark.parametrize("bad", ["1.01", "-0.01"])
    def test_a_value_out_of_range(self, bad):
        text = "LUT_3D_SIZE 2\n" + f"{bad} 0 0\n" + _rows(2)
        assert "line 2" in self._refusal(text, "outside 0..1")

    def test_a_row_without_three_values(self):
        text = "LUT_3D_SIZE 2\n0.5 0.5\n" + _rows(2)
        assert "line 2" in self._refusal(text, "2 values, not 3")

    def test_an_unknown_keyword(self):
        text = "LUT_3D_SIZE 2\nLUT_3D_INPUT_RANGE 0 1\n" + _rows(2)
        assert "line 2" in self._refusal(text, "unknown keyword")

    def test_a_1d_lut(self):
        text = "LUT_1D_SIZE 4\n" + "\n".join(["0 0 0"] * 4) + "\n"
        assert "line 1" in self._refusal(text, "1D LUT")

    def test_a_duplicate_size(self):
        text = "LUT_3D_SIZE 2\nLUT_3D_SIZE 2\n" + _rows(2)
        assert "line 2" in self._refusal(text, "second time")

    def test_a_size_after_the_data(self):
        text = "LUT_3D_SIZE 2\n0 0 0\nLUT_3D_SIZE 2\n"
        assert "line 3" in self._refusal(text, "after the data")

    def test_data_before_the_size(self):
        assert "line 1" in self._refusal("0 0 0\nLUT_3D_SIZE 2\n", "before LUT_3D_SIZE")

    def test_no_size_at_all(self):
        self._refusal('TITLE "x"\n', "no LUT_3D_SIZE")

    @pytest.mark.parametrize(
        "line", ["DOMAIN_MIN 0.1 0 0", "DOMAIN_MAX 1 1 2", "DOMAIN_MAX 1 1"]
    )
    def test_a_domain_other_than_zero_to_one(self, line):
        text = f"LUT_3D_SIZE 2\n{line}\n" + _rows(2)
        assert "line 2" in self._refusal(text, "DOMAIN_M")


class TestFileRefusals:
    def test_an_oversize_file(self, tmp_path):
        path = tmp_path / "huge.cube"
        with open(path, "wb") as handle:
            handle.truncate(MAX_CUBE_BYTES + 1)
        with pytest.raises(CubeError, match="byte limit") as refusal:
            load_lut(str(path))
        assert "huge.cube" in str(refusal.value)
        assert str(tmp_path) not in str(refusal.value)

    def test_content_that_is_not_utf8(self, tmp_path):
        path = _write(tmp_path, "latin.cube", b'LUT_3D_SIZE 2\nTITLE "caf\xe9"\n')
        with pytest.raises(CubeError, match="line 2: not UTF-8") as refusal:
            load_lut(path)
        assert str(tmp_path) not in str(refusal.value)

    def test_a_parse_refusal_names_the_file_not_its_directory(self, tmp_path):
        path = _write(tmp_path, "bad.cube", "LUT_3D_SIZE 200\n")
        with pytest.raises(CubeError) as refusal:
            load_lut(path)
        assert str(refusal.value).startswith("bad.cube line 1")
        assert str(tmp_path) not in str(refusal.value)

    @pytest.mark.parametrize("name", ["look.png", "look.3dl", "look"])
    def test_a_wrong_extension(self, tmp_path, name):
        path = _write(tmp_path, name, _cube(_identity))
        with pytest.raises(InvalidInputError, match="must be a .cube") as refusal:
            load_lut(path)
        assert str(tmp_path) not in str(refusal.value)

    def test_a_missing_file(self, tmp_path):
        with pytest.raises(InvalidInputError, match="no .cube file") as refusal:
            load_lut(str(tmp_path / "absent.cube"))
        assert str(tmp_path) not in str(refusal.value)


class TestPathContainment:
    """A literal path is confined to the media roots, as any media argument."""

    @pytest.fixture
    def roots(self, tmp_path, monkeypatch):
        monkeypatch.setenv(TRUST_WORKFLOWS_ENV_VAR, "0")
        allowed = tmp_path / "assets"
        allowed.mkdir()
        with patch("dw.locations.media_roots", return_value=[str(allowed)]):
            yield allowed

    def test_a_path_inside_a_root_loads(self, roots):
        path = _write(roots, "look.cube", _cube(_identity))
        assert load_lut(path).shape == (2, 2, 2, 3)

    def test_a_path_outside_every_root_refuses(self, roots, tmp_path):
        outside = tmp_path / "elsewhere"
        outside.mkdir()
        path = _write(outside, "look.cube", _cube(_identity))
        with pytest.raises(PathTraversalError, match="outside every directory") as r:
            load_lut(path)
        # Only the location the caller wrote; never a root's path
        assert str(roots) not in str(r.value)

    def test_dot_dot_refuses(self, roots):
        with pytest.raises(PathTraversalError):
            load_lut(str(roots) + "/../elsewhere/look.cube")

    def test_a_relative_dot_dot_refuses(self, roots):
        with pytest.raises(PathTraversalError):
            load_lut("../look.cube")

    def test_a_url_refuses_as_a_url(self, roots):
        # A lut from a variable never met validation; it gets the same reason
        with pytest.raises(InvalidInputError, match="never fetched from a URL"):
            load_lut("https://example.com/look.cube")

    def test_the_command_refuses_too(self, roots, tmp_path):
        path = _write(tmp_path, "look.cube", _cube(_identity))
        with pytest.raises(PathTraversalError):
            Task({"command": "apply_lut", "arguments": {}}, "cpu").run(
                {"media": _gradient(), "lut": path}
            )
