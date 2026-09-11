"""probe_media reports what the server knows about a generated file and
would otherwise not say - an agent cannot listen, so duration and level
are the only way it checks an audio deliverable."""

import math

import numpy
import pytest

from dw.media_info import probe_media


def write_wav(path, seconds=2.0, sample_rate=8000, amplitude=0.5):
    import wave

    t = numpy.arange(int(seconds * sample_rate)) / sample_rate
    samples = (numpy.sin(2 * numpy.pi * 220 * t) * amplitude * 32767).astype("<i2")
    with wave.open(str(path), "w") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(numpy.stack([samples, samples], 1).tobytes())


def write_mp4(path, frames=12, fps=6, width=32, height=16, with_audio=True):
    import av

    container = av.open(str(path), "w")
    video = container.add_stream("libx264", rate=fps)
    video.width, video.height, video.pix_fmt = width, height, "yuv420p"
    audio = container.add_stream("aac", rate=8000) if with_audio else None
    if audio is not None:
        audio.layout = "stereo"
    for _ in range(frames):
        frame = av.VideoFrame.from_ndarray(
            numpy.zeros((height, width, 3), numpy.uint8), format="rgb24"
        )
        for packet in video.encode(frame):
            container.mux(packet)
    if audio is not None:
        total = 8000 * frames // fps
        t = numpy.arange(total) / 8000
        sine = (numpy.sin(2 * numpy.pi * 220 * t) * 0.25).astype(numpy.float32)
        tone = numpy.stack([sine, sine])
        for start in range(0, total, 1024):
            chunk = av.AudioFrame.from_ndarray(
                numpy.ascontiguousarray(tone[:, start : start + 1024]),
                format="fltp",
                layout="stereo",
            )
            chunk.sample_rate = 8000
            chunk.pts = start
            for packet in audio.encode(chunk):
                container.mux(packet)
        for packet in audio.encode():
            container.mux(packet)
    for packet in video.encode():
        container.mux(packet)
    container.close()


def test_a_wav_reports_duration_rate_channels_and_level(tmp_path):
    write_wav(tmp_path / "score.wav", seconds=2.0, sample_rate=8000, amplitude=0.5)

    info = probe_media(str(tmp_path / "score.wav"))

    assert info["kind"] == "audio"
    assert info["duration_seconds"] == pytest.approx(2.0, abs=0.01)
    assert info["sample_rate"] == 8000
    assert info["channels"] == 2
    # a 0.5-amplitude sine peaks at -6 dBFS and sits at -9 dBFS rms
    assert info["peak_dbfs"] == pytest.approx(-6.0, abs=0.2)
    assert info["mean_dbfs"] == pytest.approx(-9.0, abs=0.2)


def test_a_video_reports_its_picture_and_its_soundtrack(tmp_path):
    write_mp4(tmp_path / "shot.mp4", frames=12, fps=6, width=32, height=16)

    info = probe_media(str(tmp_path / "shot.mp4"))

    assert info["kind"] == "video"
    assert info["frame_count"] == 12
    assert info["fps"] == pytest.approx(6.0)
    assert (info["width"], info["height"]) == (32, 16)
    assert info["duration_seconds"] == pytest.approx(2.0, abs=0.1)
    assert info["sample_rate"] == 8000
    assert info["channels"] == 2
    # AAC's lossy encode shifts the peak beyond the raw -12 dBFS the tone was
    # written at; widen the tolerance rather than the wav assertions above.
    assert info["peak_dbfs"] == pytest.approx(-12.0, abs=1.0)


def test_a_container_with_no_upfront_frame_count_still_reads_both_passes(
    tmp_path,
):
    """Matroska doesn't write a frame count into the stream header the way
    mp4 does, so `video.frames` comes back 0 and probe_media must count
    frames by decoding - in the same pass that measures the soundtrack, since
    a second decode pass over an already-exhausted demuxer reads nothing."""
    import av

    path = tmp_path / "shot.mkv"
    write_mp4(path, frames=12, fps=6, width=32, height=16)

    with av.open(str(path)) as container:
        assert container.streams.video[0].frames == 0, (
            "fixture assumption broken: this container format now writes "
            "a frame count up front, so it no longer exercises the "
            "fallback-counting path probe_media relies on"
        )

    info = probe_media(str(path))

    assert info["kind"] == "video"
    assert info["frame_count"] == 12
    assert info["peak_dbfs"] == pytest.approx(-12.0, abs=1.0)


def test_a_silent_video_has_no_audio_fields(tmp_path):
    write_mp4(tmp_path / "mute.mp4", with_audio=False)

    info = probe_media(str(tmp_path / "mute.mp4"))

    assert info["kind"] == "video"
    assert "sample_rate" not in info
    assert "peak_dbfs" not in info


def test_silence_is_clamped_not_minus_infinity(tmp_path):
    write_wav(tmp_path / "quiet.wav", amplitude=0.0)

    info = probe_media(str(tmp_path / "quiet.wav"))

    assert info["peak_dbfs"] == -120.0
    assert not math.isinf(info["mean_dbfs"])


def test_a_file_that_is_not_media_answers_none(tmp_path):
    (tmp_path / "notes.txt").write_text("not media")

    assert probe_media(str(tmp_path / "notes.txt")) is None


def test_unsigned_8bit_silence_is_not_reported_as_loud(tmp_path):
    """u8 PCM is offset-binary - silence is the byte 128, not 0 - so dividing
    raw samples by iinfo.max without recentering reports silence around
    -6 dBFS instead of the floor."""
    import wave

    path = tmp_path / "quiet-u8.wav"
    with wave.open(str(path), "w") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(1)
        handle.setframerate(8000)
        handle.writeframes(bytes([128]) * 8000)

    info = probe_media(str(path))

    assert info["peak_dbfs"] == -120.0


def test_a_damaged_track_still_reports_header_fields(tmp_path):
    """A file that opens fine but fails mid-decode (a track damaged after
    the header was written) must not 500 the metadata route - it should
    fall back to the header-level fields and drop the fields that require
    a full decode."""
    path = tmp_path / "shot.mkv"
    write_mp4(path, frames=12, fps=6, width=32, height=16)

    raw = bytearray(path.read_bytes())
    mid = len(raw) // 2
    for i in range(mid, len(raw), 64):
        raw[i] = (raw[i] + 137) % 256
    path.write_bytes(bytes(raw))

    info = probe_media(str(path))

    assert info is not None
    assert info["kind"] == "video"
    assert "peak_dbfs" not in info
    assert "frame_count" not in info
