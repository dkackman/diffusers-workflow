"""Frames out of a file by seeking, not by decoding it whole - what
get_output_frames needs so an agent can look at a moment, a contact sheet
or the frame pair either side of a seam (#193)."""

import numpy
import pytest

from dw.media_frames import (
    contact_sheet,
    frames_at,
    resolve_crop_box,
    seam_tiles,
    video_shape,
)


def write_ramp_mp4(path, frames=24, fps=6, width=32, height=16):
    """A clip whose frame N is a flat grey of value N*10, so a returned
    frame says which one it is. For frames > 25, wraps to stay in uint8."""
    import av

    container = av.open(str(path), "w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
    stream.options = {"crf": "0", "preset": "ultrafast"}  # lossless, so grey survives
    for index in range(frames):
        pixels = numpy.full((height, width, 3), (index % 26) * 10, numpy.uint8)
        frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def write_shifted_ramp_mp4(path, frames=60, fps=6, width=32, height=16, offset=5):
    """Like `write_ramp_mp4`, but every packet's pts is shifted by `offset`
    frames - the edit-list / non-zero-start shape a real muxer can write,
    which PyAV surfaces as a non-zero `stream.start_time`. Frame N is grey
    `(N % 25) * 10`, wrapping so the values stay in a byte at 60 frames."""
    import av

    container = av.open(str(path), "w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
    stream.options = {"crf": "0", "preset": "ultrafast", "g": "10"}

    def shifted(packets):
        for packet in packets:
            if packet.pts is not None:
                packet.pts += offset
            if packet.dts is not None:
                packet.dts += offset
            container.mux(packet)

    for index in range(frames):
        pixels = numpy.full((height, width, 3), (index % 25) * 10, numpy.uint8)
        frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
        shifted(stream.encode(frame))
    shifted(stream.encode())
    container.close()


def write_split_mp4(path, frames=12, fps=6, width=64, height=32, left=50, right=200):
    """Every frame's left half is a flat grey of `left`, its right half
    `right` - a positional fixture. `write_ramp_mp4`'s frames are each one
    flat value, which proves *which frame* was read but not *which region*
    of it - a crop of a uniform frame looks identical wherever the box sits.
    This is what a crop-position test actually needs to fail under the old
    bug (crop applied to the wrong pixels, or to the assembled/downscaled
    result rather than the source frame)."""
    import av

    container = av.open(str(path), "w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
    stream.options = {"crf": "0", "preset": "ultrafast"}
    half = width // 2
    pixels = numpy.empty((height, width, 3), numpy.uint8)
    pixels[:, :half] = left
    pixels[:, half:] = right
    for _ in range(frames):
        frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def grey_of(image):
    return int(numpy.asarray(image.convert("L")).mean().round())


def test_video_shape_reads_the_container(tmp_path):
    write_ramp_mp4(tmp_path / "ramp.mp4", frames=24, fps=6)

    shape = video_shape(str(tmp_path / "ramp.mp4"))

    assert shape == {"frame_count": 24, "fps": 6.0, "width": 32, "height": 16}


def test_frames_at_seeks_to_seconds_and_frame_indexes(tmp_path):
    write_ramp_mp4(tmp_path / "ramp.mp4", frames=24, fps=6)

    tiles = frames_at(str(tmp_path / "ramp.mp4"), [0.0, 2.0, "frame:23"])

    assert [t["frame"] for t in tiles] == [0, 12, 23]
    assert [t["seconds"] for t in tiles] == pytest.approx([0.0, 2.0, 23 / 6])
    assert [grey_of(t["image"]) for t in tiles] == pytest.approx([0, 120, 230], abs=6)
    assert tiles[1]["label"] == "00:02.0 (frame 12)"


def test_frames_at_is_correct_when_the_stream_has_a_non_zero_start(tmp_path):
    path = tmp_path / "shifted.mp4"
    write_shifted_ramp_mp4(path, frames=60, fps=6, offset=5)

    import av

    with av.open(str(path)) as container:
        start_time = container.streams.video[0].start_time
    assert start_time not in (None, 0), "fixture didn't actually shift pts"

    tiles = frames_at(str(path), ["frame:20", "frame:40", "frame:55"])

    assert [t["frame"] for t in tiles] == [20, 40, 55]
    assert [grey_of(t["image"]) for t in tiles] == pytest.approx(
        [(n % 25) * 10 for n in (20, 40, 55)], abs=6
    )


def test_a_moment_past_the_end_is_refused(tmp_path):
    write_ramp_mp4(tmp_path / "ramp.mp4", frames=6, fps=6)

    with pytest.raises(ValueError, match="past the end"):
        frames_at(str(tmp_path / "ramp.mp4"), [5.0])
    with pytest.raises(ValueError, match="past the end"):
        frames_at(str(tmp_path / "ramp.mp4"), ["frame:6"])


def test_a_contact_sheet_tiles_evenly_spaced_frames(tmp_path):
    write_ramp_mp4(tmp_path / "ramp.mp4", frames=24, fps=6, width=64, height=32)

    tile = contact_sheet(str(tmp_path / "ramp.mp4"), count=4, tile_width=64)

    assert tile["label"] == "contact sheet, 4 frames"
    assert tile["image"].width == 64 * 2  # 4 tiles, two columns
    assert tile["image"].height == 32 * 2
    # first tile is frame 0, last is frame 23 - sampled away from the
    # timestamp stamp in the top-left corner of each tile
    first = tile["image"].crop((32, 16, 64, 32))
    last = tile["image"].crop((96, 48, 128, 64))
    assert grey_of(first) < 20
    assert grey_of(last) > 200


def test_seam_tiles_pair_the_frames_either_side_of_each_boundary(tmp_path):
    write_ramp_mp4(tmp_path / "ramp.mp4", frames=24, fps=6)

    tiles = seam_tiles(
        str(tmp_path / "ramp.mp4"),
        boundaries=[8, 16],
        names=["shot@a", "shot@b", "shot@c"],
        tile_width=32,
    )

    assert [t["label"] for t in tiles] == [
        "seam 1: shot@a | shot@b",
        "seam 2: shot@b | shot@c",
    ]
    assert [t["frame"] for t in tiles] == [8, 16]
    image = tiles[0]["image"]
    assert image.width == 64 and image.height == 16
    assert grey_of(image.crop((0, 0, 32, 16))) == pytest.approx(70, abs=6)  # frame 7
    assert grey_of(image.crop((32, 0, 64, 16))) == pytest.approx(80, abs=6)  # frame 8


def test_a_boundary_outside_the_clip_is_refused(tmp_path):
    write_ramp_mp4(tmp_path / "ramp.mp4", frames=6, fps=6)

    with pytest.raises(ValueError, match="boundary"):
        seam_tiles(str(tmp_path / "ramp.mp4"), boundaries=[0])
    with pytest.raises(ValueError, match="boundary"):
        seam_tiles(str(tmp_path / "ramp.mp4"), boundaries=[6])


def test_seam_tiles_wanted_skips_decoding_the_other_seams(tmp_path):
    """A caller after seam 2 of a three-shot cut should not pay to decode
    and compose the other seam too - `wanted` still validates and labels
    against the full boundaries/names, but only builds the seam asked
    for (#193 follow-up: seam_tiles used to build every seam regardless).

    Needs more than one keyframe to show the saving - write_ramp_mp4's
    default GOP has a single keyframe at frame 0, so every seek lands
    there anyway and decode work is identical either way; a multi-keyframe
    clip (write_shifted_ramp_mp4's g=10) is what lets `wanted` actually
    skip the decode work between the seam it wasn't asked for and the one
    it was."""
    import av
    from unittest import mock

    write_shifted_ramp_mp4(tmp_path / "ramp.mp4", frames=60, fps=6, offset=0)
    path = str(tmp_path / "ramp.mp4")

    tiles = seam_tiles(path, boundaries=[10, 50], names=["a", "b", "c"], wanted={2})

    assert [t["label"] for t in tiles] == ["seam 2: b | c"]
    assert [t["frame"] for t in tiles] == [50]

    original_decode = av.container.InputContainer.decode

    def counting(counter):
        def decode(self, *args, **kwargs):
            for frame in original_decode(self, *args, **kwargs):
                counter[0] += 1
                yield frame

        return decode

    full_count = [0]
    with mock.patch.object(av.container.InputContainer, "decode", counting(full_count)):
        seam_tiles(path, boundaries=[10, 50], names=["a", "b", "c"])

    filtered_count = [0]
    with mock.patch.object(
        av.container.InputContainer, "decode", counting(filtered_count)
    ):
        seam_tiles(path, boundaries=[10, 50], names=["a", "b", "c"], wanted={2})

    assert filtered_count[0] < full_count[0]


def test_frames_at_reuses_a_given_shape(tmp_path, monkeypatch):
    """The gallery route computes `video_shape` once and hands it to
    frames_at/contact_sheet/seam_tiles - a `shape` already in hand must not
    trigger another container open (and, lacking a header frame count,
    another full decode to count) inside the selector function."""
    import dw.media_frames as media_frames

    write_ramp_mp4(tmp_path / "ramp.mp4", frames=24, fps=6)
    path = str(tmp_path / "ramp.mp4")

    calls = [0]
    original = media_frames.video_shape

    def counting_shape(p):
        calls[0] += 1
        return original(p)

    monkeypatch.setattr(media_frames, "video_shape", counting_shape)

    shape = media_frames.video_shape(path)
    assert calls[0] == 1

    media_frames.frames_at(path, [0.0, "frame:12"], shape=shape)

    assert calls[0] == 1


def test_a_sub_tile_is_never_upscaled_past_the_source(tmp_path):
    """`_fit_width` scaled a 32-wide source up to a 320-wide tile - a
    blurred enlargement that costs bytes and shows nothing the source
    holds. A tile is at most the source's own width."""
    write_ramp_mp4(tmp_path / "ramp.mp4", frames=12, fps=6, width=32, height=16)

    sheet = contact_sheet(str(tmp_path / "ramp.mp4"), 4, tile_width=320)
    assert sheet["image"].width <= 32 * 4

    seams = seam_tiles(str(tmp_path / "ramp.mp4"), boundaries=[6], tile_width=320)
    assert seams[0]["image"].width == 64


def test_a_wanted_seam_outside_the_cut_is_refused(tmp_path):
    """`wanted={3}` on a two-seam cut answered `[]` - a 200 with no tiles
    that read as "nothing to show". A seam number the cut does not have is
    an error naming the range."""
    write_ramp_mp4(tmp_path / "ramp.mp4", frames=24, fps=6)

    with pytest.raises(ValueError, match="1..2"):
        seam_tiles(str(tmp_path / "ramp.mp4"), boundaries=[8, 16], wanted={3})
    with pytest.raises(ValueError, match="1..2"):
        seam_tiles(str(tmp_path / "ramp.mp4"), boundaries=[8, 16], wanted={0, 1})


def test_a_non_finite_moment_is_refused(tmp_path):
    """`at=inf` reached `int(round(inf * fps))` and raised OverflowError -
    a 500 from the route rather than the 400 every other bad moment gets."""
    write_ramp_mp4(tmp_path / "ramp.mp4", frames=12, fps=6)

    for moment in (float("inf"), float("-inf"), float("nan")):
        with pytest.raises(ValueError):
            frames_at(str(tmp_path / "ramp.mp4"), [moment])


def test_a_contact_sheet_over_the_frame_cap_is_refused(tmp_path):
    from dw.media_frames import MAX_CONTACT_SHEET_FRAMES

    # The clip must hold more frames than the cap itself, else the
    # clip-length clamp would bring the requested count under the cap
    # before it is ever checked (see the next test).
    write_ramp_mp4(tmp_path / "ramp.mp4", frames=MAX_CONTACT_SHEET_FRAMES + 24, fps=6)

    with pytest.raises(ValueError, match=f"{MAX_CONTACT_SHEET_FRAMES}"):
        contact_sheet(str(tmp_path / "ramp.mp4"), MAX_CONTACT_SHEET_FRAMES + 1)


def test_a_contact_sheet_count_clamped_under_the_cap_is_not_refused(tmp_path):
    """count=100 on a clip with fewer frames than the cap clamps down to the
    clip's own frame_count first, and that clamped number is well under
    MAX_CONTACT_SHEET_FRAMES - so it must succeed rather than 400 on the
    caller's raw, pre-clamp number (#193 review, finding 3)."""
    from dw.media_frames import MAX_CONTACT_SHEET_FRAMES

    write_ramp_mp4(tmp_path / "ramp.mp4", frames=24, fps=6)
    assert 24 < MAX_CONTACT_SHEET_FRAMES

    result = contact_sheet(str(tmp_path / "ramp.mp4"), 100)

    assert len(result["frames"]) == 24


def test_more_seams_than_the_cap_are_refused(tmp_path):
    from dw.media_frames import MAX_SEAMS

    write_ramp_mp4(tmp_path / "ramp.mp4", frames=MAX_SEAMS + 3, fps=6)
    boundaries = list(range(1, MAX_SEAMS + 2))  # MAX_SEAMS + 1 seams

    with pytest.raises(ValueError, match=f"{MAX_SEAMS}"):
        seam_tiles(str(tmp_path / "ramp.mp4"), boundaries=boundaries)
    # a `wanted` subset under the cap is still served
    tiles = seam_tiles(str(tmp_path / "ramp.mp4"), boundaries=boundaries, wanted={1, 2})
    assert len(tiles) == 2


def test_read_frames_fits_each_frame_as_it_is_decoded(tmp_path):
    from dw.media_frames import _read_frames

    write_ramp_mp4(tmp_path / "ramp.mp4", frames=24, fps=6)
    seen = []

    def fit(image, index):
        seen.append((image.size, index))
        return image.resize((8, 4))

    found = _read_frames(str(tmp_path / "ramp.mp4"), [0, 5, 23], fit=fit)

    assert seen == [
        ((32, 16), 0),
        ((32, 16), 5),
        ((32, 16), 23),
    ]  # ran per frame, at source size
    assert all(image.size == (8, 4) for image in found.values())


def test_a_contact_sheet_never_holds_a_full_size_frame(tmp_path, monkeypatch):
    """The point of the cap and the fitter together: a 1080p clip's contact
    sheet is built from tiles, not from a list of 1080p images."""
    import dw.media_frames as module

    write_ramp_mp4(tmp_path / "ramp.mp4", frames=24, fps=6, width=64, height=32)
    sizes = []
    real_read = module._read_frames

    def spying_read(path, indexes, fit=None):
        found = real_read(path, indexes, fit=fit)
        sizes.extend(image.size for image in found.values())
        return found

    monkeypatch.setattr(module, "_read_frames", spying_read)

    contact_sheet(str(tmp_path / "ramp.mp4"), 4, tile_width=16)

    assert sizes and all(size == (16, 8) for size in sizes)


def test_a_seek_that_lands_past_its_target_recovers_from_the_top(tmp_path):
    """A VFR or off-rate file can make the seek land *after* the wanted
    frame. The reader must then read from the top rather than scan to EOF
    and report the frame as undecodable."""
    import av
    from unittest import mock

    write_shifted_ramp_mp4(tmp_path / "gop.mp4", frames=60, fps=6, offset=0)
    real_open = av.open

    class Overshooting:
        def __init__(self, inner):
            self._inner = inner
            self.seeks = []

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def __enter__(self):
            self._inner.__enter__()
            return self

        def __exit__(self, *exc):
            return self._inner.__exit__(*exc)

        def seek(self, offset, **kwargs):
            self.seeks.append(offset)
            stream = kwargs["stream"]
            # first seek lands two keyframes late; a later seek is honest
            late = int(20 / 6 / stream.time_base) if len(self.seeks) == 1 else 0
            return self._inner.seek(offset + late, **kwargs)

    proxies = []

    def opening(*args, **kwargs):
        proxies.append(Overshooting(real_open(*args, **kwargs)))
        return proxies[-1]

    with mock.patch("dw.media_frames.av.open", side_effect=opening):
        tiles = frames_at(str(tmp_path / "gop.mp4"), ["frame:35"])

    assert grey_of(tiles[0]["image"]) == pytest.approx((35 % 25) * 10, abs=6)
    assert len(proxies[-1].seeks) == 2  # the overshoot, then the recovery


def test_contact_sheet_cells_are_stamped_with_their_timestamp(tmp_path):
    """A flat grey cell has zero variance; a stamped one does not. The stamp
    is what lets a reader of the image, not the text, say which cell is
    which."""
    write_ramp_mp4(tmp_path / "ramp.mp4", frames=24, fps=6, width=64, height=32)

    sheet = contact_sheet(str(tmp_path / "ramp.mp4"), 4, tile_width=64)

    columns = 2  # _default_columns(4)
    for cell in range(4):
        row, col = divmod(cell, columns)
        corner = sheet["image"].crop((col * 64, row * 32, col * 64 + 32, row * 32 + 16))
        assert numpy.asarray(corner.convert("L")).std() > 5, (
            f"cell {cell} carries no stamp"
        )
    # the stamp sits in a corner: the opposite corner is still the flat grey
    cell = sheet["image"].crop((32, 16, 64, 32))
    assert numpy.asarray(cell.convert("L")).std() < 2


def test_a_seam_tile_reports_the_mean_difference_across_the_join(tmp_path):
    write_ramp_mp4(tmp_path / "ramp.mp4", frames=24, fps=6)

    tiles = seam_tiles(str(tmp_path / "ramp.mp4"), boundaries=[8], tile_width=32)

    # frame 7 is grey 70, frame 8 grey 80: a flat difference of 10
    assert tiles[0]["difference"] == pytest.approx(10.0, abs=6)


def test_resolve_crop_box_converts_to_a_pillow_box_and_clamps():
    assert resolve_crop_box([10, 5, 20, 8], width=100, height=50) == (10, 5, 30, 13)
    # a box that overruns the frame is clamped, not refused
    assert resolve_crop_box([90, 40, 50, 50], width=100, height=50) == (90, 40, 100, 50)


def test_resolve_crop_box_refuses_malformed_input():
    with pytest.raises(ValueError, match="x, y, width, height"):
        resolve_crop_box([1, 2, 3], width=100, height=50)
    with pytest.raises(ValueError, match="x, y, width, height"):
        resolve_crop_box(["a", "b", "c", "d"], width=100, height=50)


def test_resolve_crop_box_refuses_a_non_positive_origin_or_size():
    with pytest.raises(ValueError, match="non-negative origin"):
        resolve_crop_box([-1, 0, 10, 10], width=100, height=50)
    with pytest.raises(ValueError, match="non-negative origin"):
        resolve_crop_box([0, 0, 0, 10], width=100, height=50)


def test_resolve_crop_box_refuses_an_origin_outside_the_frame():
    with pytest.raises(ValueError, match="outside the"):
        resolve_crop_box([100, 0, 10, 10], width=100, height=50)
    with pytest.raises(ValueError, match="outside the"):
        resolve_crop_box([0, 50, 10, 10], width=100, height=50)


def test_frames_at_crop_box_cuts_the_source_frame_before_it_is_returned(tmp_path):
    write_split_mp4(tmp_path / "split.mp4", width=64, height=32, left=50, right=200)

    left = frames_at(str(tmp_path / "split.mp4"), [0.0], crop_box=(0, 0, 32, 32))
    right = frames_at(str(tmp_path / "split.mp4"), [0.0], crop_box=(32, 0, 64, 32))

    assert left[0]["image"].size == (32, 32)
    assert grey_of(left[0]["image"]) == pytest.approx(50, abs=3)
    assert grey_of(right[0]["image"]) == pytest.approx(200, abs=3)


def test_contact_sheet_crop_box_is_cut_from_each_source_frame_before_tiling(tmp_path):
    """The tester's original repro: a crop box in source pixels must select
    the same region regardless of the sheet's own tile_width, and must not
    be checked against the assembled (and possibly much smaller) sheet."""
    write_split_mp4(tmp_path / "split.mp4", width=64, height=32, left=50, right=200)
    path = str(tmp_path / "split.mp4")
    crop_box = (32, 0, 64, 32)  # right half, in the 64-wide source

    wide = contact_sheet(path, count=4, tile_width=64, crop_box=crop_box)
    narrow = contact_sheet(path, count=4, tile_width=8, crop_box=crop_box)

    # every source frame here is flat within its half, so a correctly
    # cropped-then-stamped cell reads close to 200 wherever the stamp
    # doesn't sit, at either tile_width - including a sheet far smaller
    # than the crop box's own source-pixel coordinates.
    for sheet in (wide, narrow):
        corner = sheet["image"].crop(
            (0, sheet["image"].height // 2, sheet["image"].width, sheet["image"].height)
        )
        assert grey_of(corner) > 150


def test_seam_tiles_crop_box_is_cut_from_each_source_frame_before_pairing(tmp_path):
    write_split_mp4(tmp_path / "split.mp4", width=64, height=32, left=50, right=200)
    path = str(tmp_path / "split.mp4")

    tiles = seam_tiles(
        str(path), boundaries=[6], tile_width=32, crop_box=(32, 0, 64, 32)
    )

    # both sides of the seam are cropped to the right half of a uniformly
    # split frame, so the whole paired image reads near 200 - a stray crop
    # of the left half, or of the assembled pair, would not.
    assert grey_of(tiles[0]["image"]) == pytest.approx(200, abs=6)
