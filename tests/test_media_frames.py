"""Frames out of a file by seeking, not by decoding it whole - what
get_output_frames needs so an agent can look at a moment, a contact sheet
or the frame pair either side of a seam (#193)."""

import numpy
import pytest

from dw.media_frames import contact_sheet, frames_at, seam_tiles, video_shape


def write_ramp_mp4(path, frames=24, fps=6, width=32, height=16):
    """A clip whose frame N is a flat grey of value N*10, so a returned
    frame says which one it is."""
    import av

    container = av.open(str(path), "w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
    stream.options = {"crf": "0", "preset": "ultrafast"}  # lossless, so grey survives
    for index in range(frames):
        pixels = numpy.full((height, width, 3), index * 10, numpy.uint8)
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
    write_ramp_mp4(tmp_path / "ramp.mp4", frames=24, fps=6)

    tile = contact_sheet(str(tmp_path / "ramp.mp4"), count=4, tile_width=32)

    assert tile["label"] == "contact sheet, 4 frames"
    assert tile["image"].width == 32 * 2  # 4 tiles, two columns
    assert tile["image"].height == 16 * 2
    # first tile is frame 0, last is frame 23
    first = tile["image"].crop((0, 0, 32, 16))
    last = tile["image"].crop((32, 16, 64, 32))
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

    assert [t["label"] for t in tiles] == ["seam 1: shot@a | shot@b", "seam 2: shot@b | shot@c"]
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
