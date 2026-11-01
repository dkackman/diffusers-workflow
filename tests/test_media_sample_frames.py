"""dw.media.sample_frames: a clip at a model's frame rate and size, by source
frame index (MMAudio's CLIP 8 fps / 384 and Synchformer 25 fps / 224 inputs)."""

import av
import numpy
import pytest

from dw.media import sample_frames

SECONDS = 2.0
WIDTH, HEIGHT = 64, 48


def level(index):
    return (index * 7) % 200 + 20


def write_clip(path, fps, frames, frame_pixels):
    container = av.open(str(path), "w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width, stream.height, stream.pix_fmt = WIDTH, HEIGHT, "yuv420p"
    stream.options = {"crf": "0", "preset": "ultrafast"}
    for index in range(frames):
        frame = av.VideoFrame.from_ndarray(frame_pixels(index), format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def grey(index):
    return numpy.full((HEIGHT, WIDTH, 3), level(index), numpy.uint8)


@pytest.fixture(params=[30, 24])
def clip(request, tmp_path):
    path = tmp_path / f"clip{request.param}.mp4"
    write_clip(path, request.param, int(request.param * SECONDS), grey)
    return str(path), request.param


@pytest.mark.parametrize("fps,expected", [(8, 16), (25, 50)])
def test_counts_follow_the_target_rate(clip, fps, expected):
    path, _ = clip

    assert len(sample_frames(path, fps, SECONDS, 16)) == expected


def test_a_short_clip_returns_only_the_frames_it_has(clip):
    path, source_fps = clip
    count = int(source_fps * SECONDS)
    wanted = [round(k / 8 * source_fps) for k in range(24)]

    frames = sample_frames(path, 8, 3.0, 16)

    assert len(frames) == sum(1 for i in wanted if i < count) == 16


@pytest.mark.parametrize("fps", [8, 25])
def test_frame_k_comes_from_the_rounded_source_index(clip, fps):
    path, source_fps = clip

    frames = sample_frames(path, fps, SECONDS, 16)

    for k, frame in enumerate(frames):
        source = round(k / fps * source_fps)
        assert frame.mean() == pytest.approx(level(source), abs=3)


def test_clip_path_is_square_uint8(clip):
    path, _ = clip

    frames = sample_frames(path, 8, SECONDS, 384)

    assert frames.shape == (16, 384, 384, 3)
    assert frames.dtype == numpy.uint8


def test_small_square_resize_keeps_uint8(clip):
    path, _ = clip

    frames = sample_frames(path, 8, SECONDS, 32)

    assert frames.shape == (16, 32, 32, 3) and frames.dtype == numpy.uint8


def test_synchformer_path_is_float_in_unit_range(clip):
    path, _ = clip

    frames = sample_frames(path, 25, SECONDS, 224, crop=True)

    assert frames.shape == (50, 224, 224, 3)
    assert frames.dtype == numpy.float32
    assert frames.min() >= -1.0 and frames.max() <= 1.0
    assert frames[0].mean() == pytest.approx(level(0) / 127.5 - 1, abs=0.05)


def test_centre_crop_takes_the_middle(tmp_path):
    def thirds(_index):
        pixels = numpy.full((HEIGHT, WIDTH, 3), 20, numpy.uint8)
        pixels[:, WIDTH // 3 : 2 * WIDTH // 3] = 220
        return pixels

    write_clip(tmp_path / "thirds.mp4", 30, 30, thirds)

    cropped = sample_frames(str(tmp_path / "thirds.mp4"), 8, 1.0, 32, crop=True)
    squashed = sample_frames(str(tmp_path / "thirds.mp4"), 8, 1.0, 32)

    # 64x48 -> short side 32 -> 43x32 -> a 32-wide window is the middle,
    # whose centre is bright and whose edges are only the bright third's slope
    centre = cropped[0][:, 12:20].mean()
    assert centre > 0.6
    assert cropped[0][:, :2].mean() < -0.5
    # without the crop the whole width is squeezed in, so the edges are dark
    # too but the dark side bands are wider in the crop's view of the same frame
    assert squashed[0][:, :2].mean() < 60


def test_bad_arguments_raise(clip):
    path, _ = clip

    for args in [(0, 1.0, 8), (8, 0, 8), (8, 1.0, 0), (-1, 1.0, 8)]:
        with pytest.raises(ValueError):
            sample_frames(path, *args)


def test_audio_only_file_raises(tmp_path):
    path = tmp_path / "tone.wav"
    container = av.open(str(path), "w")
    stream = container.add_stream("pcm_s16le", rate=8000)
    stream.layout = "mono"
    frame = av.AudioFrame.from_ndarray(
        numpy.zeros((1, 800), numpy.int16), format="s16", layout="mono"
    )
    frame.sample_rate = 8000
    for packet in stream.encode(frame):
        container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()

    with pytest.raises(ValueError):
        sample_frames(str(path), 8, 1.0, 16)
