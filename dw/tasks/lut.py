"""
apply_lut: a 3D colour lookup table from a .cube file, applied on CPU.

The .cube parser is strict (#603, Q2): it reads the 3D subset of the format
and refuses anything else, naming the file and the line, rather than guessing
at what a file it does not understand meant. A file is untrusted user input -
an uploaded asset - so it is size-capped before it is read.

The lookup itself is Pillow's (ImageFilter.Color3DLUT, trilinear); this module
owns only the parse that feeds it.

A refusal names the file by its base name only: the path an asset: or
output: reference resolves to is the server's filesystem layout, and the
message reaches API and MCP callers.
"""

import math
import os

import numpy as np
from PIL import ImageFilter

from .finish import _join_alpha, _split_alpha

# A 65-point LUT, the largest the parser takes, is about 9 MB of text at
# generous precision; anything far past that is not a LUT
MAX_CUBE_BYTES = 16 * 1024 * 1024
MIN_LUT_SIZE = 2
MAX_LUT_SIZE = 65

# The keywords a 3D .cube may carry. LUT_1D_SIZE is refused by name, every
# other keyword as unknown
_KEYWORDS = {"TITLE", "LUT_3D_SIZE", "DOMAIN_MIN", "DOMAIN_MAX"}


class CubeError(ValueError):
    """A .cube file the strict parser refuses."""


def _floats(tokens, where, what):
    try:
        values = [float(token) for token in tokens]
    except ValueError:
        raise CubeError(f"{where}: {what} is not three numbers")
    if len(values) != 3:
        raise CubeError(f"{where}: {what} has {len(values)} values, not 3")
    return values


def parse_cube(text, name):
    """The lookup table a .cube file's text describes.

    Args:
        text: The file's contents
        name: The file's name, for the refusal messages

    Returns:
        A float32 array of shape (size, size, size, 3), indexed
        [blue, green, red]: the format lists red fastest, so the rows
        reshape straight into that order

    Raises:
        CubeError: On anything outside the strict 3D subset, naming the
            file and line
    """
    size = None
    domain = {}
    rows = []
    last_line = 0
    for number, raw in enumerate(text.splitlines(), start=1):
        last_line = number
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        where = f"{name} line {number}"
        tokens = line.split()
        keyword = tokens[0]
        try:
            float(keyword)
            is_data = True
        except ValueError:
            is_data = False

        if not is_data:
            if keyword == "LUT_1D_SIZE":
                raise CubeError(
                    f"{where}: LUT_1D_SIZE - a 1D LUT, and apply_lut takes "
                    f"only a 3D LUT (LUT_3D_SIZE)"
                )
            if keyword not in _KEYWORDS:
                raise CubeError(
                    f"{where}: unknown keyword {keyword[:40]!r} - a .cube "
                    f"may carry only TITLE, LUT_3D_SIZE, DOMAIN_MIN, "
                    f"DOMAIN_MAX, # comments and data rows"
                )
            if rows:
                raise CubeError(
                    f"{where}: {keyword} after the data rows - every keyword "
                    f"comes before the data"
                )
            if keyword == "TITLE":
                continue
            if keyword == "LUT_3D_SIZE":
                if size is not None:
                    raise CubeError(f"{where}: LUT_3D_SIZE given a second time")
                if len(tokens) != 2 or not tokens[1].isdigit():
                    raise CubeError(f"{where}: LUT_3D_SIZE takes one whole number")
                size = int(tokens[1])
                if not MIN_LUT_SIZE <= size <= MAX_LUT_SIZE:
                    raise CubeError(
                        f"{where}: LUT_3D_SIZE {size} is outside "
                        f"{MIN_LUT_SIZE}..{MAX_LUT_SIZE}"
                    )
                continue
            # DOMAIN_MIN / DOMAIN_MAX
            if keyword in domain:
                raise CubeError(f"{where}: {keyword} given a second time")
            values = _floats(tokens[1:], where, keyword)
            expected = 0.0 if keyword == "DOMAIN_MIN" else 1.0
            if any(value != expected for value in values):
                raise CubeError(
                    f"{where}: {keyword} must be {expected:g} {expected:g} "
                    f"{expected:g} - only a 0..1 domain is supported"
                )
            domain[keyword] = values
            continue

        if size is None:
            raise CubeError(f"{where}: a data row before LUT_3D_SIZE")
        if len(rows) == size**3:
            raise CubeError(
                f"{where}: more than the {size**3} data rows LUT_3D_SIZE "
                f"{size} calls for"
            )
        values = _floats(tokens, where, "the data row")
        for value in values:
            if not math.isfinite(value):
                raise CubeError(f"{where}: {value} is not a finite number")
            if not 0.0 <= value <= 1.0:
                raise CubeError(f"{where}: {value:g} is outside 0..1")
        rows.append(values)

    if size is None:
        raise CubeError(f"{name}: no LUT_3D_SIZE line")
    if len(rows) != size**3:
        raise CubeError(
            f"{name} line {last_line}: the file ends after {len(rows)} data "
            f"rows, and LUT_3D_SIZE {size} calls for {size**3}"
        )
    return np.asarray(rows, dtype=np.float32).reshape(size, size, size, 3)


def read_cube(path):
    """Parse a .cube file already validated to lie inside the media roots.

    Size-capped before a byte is read, and decoded as strict UTF-8.
    """
    name = os.path.basename(path)
    size = os.path.getsize(path)
    if size > MAX_CUBE_BYTES:
        raise CubeError(
            f"{name}: {size} bytes, over the {MAX_CUBE_BYTES} byte limit "
            f"for a .cube file"
        )
    with open(path, "rb") as handle:
        data = handle.read(MAX_CUBE_BYTES + 1)
    if len(data) > MAX_CUBE_BYTES:
        raise CubeError(
            f"{name}: over the {MAX_CUBE_BYTES} byte limit for a .cube file"
        )
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        line = data.count(b"\n", 0, error.start) + 1
        raise CubeError(f"{name} line {line}: not UTF-8 text")
    return parse_cube(text, name)


def load_lut(lut):
    """The lookup table a `lut` argument names.

    An asset:/output: reference reaches here already resolved to a path; a
    literal path arrives raw. Either way it is confined to the media roots
    (the workflow directory, the asset libraries and the output root) and
    must be a .cube, the precedent load_audio_video set. No refusal names
    an absolute path the caller did not write.
    """
    from ..locations import validate_media_path
    from ..security import (
        ALLOWED_LUT_EXTENSIONS,
        InvalidInputError,
        validate_file_extension,
    )

    if not isinstance(lut, str):
        raise ValueError(
            f"apply_lut: 'lut' is a .cube file - an asset: or output: "
            f"reference or a path - not {type(lut).__name__}"
        )
    path = validate_media_path(lut, None, "a LUT", require_exists=False)
    try:
        validate_file_extension(path, ALLOWED_LUT_EXTENSIONS)
    except InvalidInputError:
        extension = os.path.splitext(path)[1] or "no extension"
        raise InvalidInputError(
            f"apply_lut: 'lut' must be a .cube file, not {extension} "
            f"({os.path.basename(path)})"
        )
    if not os.path.isfile(path):
        raise InvalidInputError(
            f"apply_lut: no .cube file at {os.path.basename(path)!r}"
        )
    return read_cube(path)


def color_lut(table):
    """Pillow's trilinear 3D lookup filter for a parsed table.

    Pillow takes the table indexed [blue, green, red] with red fastest - the
    .cube row order parse_cube already returns - and the same 2..65 sizes.
    """
    return ImageFilter.Color3DLUT(table.shape[0], table)


def apply_lut(media, lut, strength=1.0):
    """Apply a 3D lookup table from a .cube file to an image.

    Each pixel's colour is looked up in the table with trilinear
    interpolation, and the result blended with the original by `strength`.

    Args:
        media: PIL Image to colour
        lut: The .cube file: an asset: or output: reference, or a path
            inside the workflow's directory, the asset libraries or the
            output root. A strict 3D .cube - LUT_3D_SIZE 2..65, domain 0..1,
            values in 0..1 - or the step refuses, naming the file and line.
            (Also an already-built color_lut, which is how the command
            handler reads the file once for a whole video)
        strength: 0.0 returns the original, 1.0 the full LUT result; values
            between blend the two

    Returns:
        PIL Image, the same size, RGB (RGBA when the input had alpha, which
        passes through untouched)
    """
    if not isinstance(lut, ImageFilter.Color3DLUT):
        lut = color_lut(load_lut(lut))
    rgb, alpha = _split_alpha(media)
    if strength != 0:
        looked_up = np.asarray(media.convert("RGB").filter(lut), dtype=np.float32)
        rgb = rgb + float(strength) * (looked_up - rgb)
    return _join_alpha(rgb, alpha)
