"""extract_audio hands back a soundtrack, or a named slice of one, as WAV -
what get_output_audio needs for a muxed video and for a track too long to
send whole."""

import io
import wave
from unittest import mock

import numpy
import pytest

from dw.media_audio import NoSoundtrack, extract_audio, media_duration
from tests.test_media_info import write_mp4, write_wav


def read_wav(data):
    with wave.open(io.BytesIO(data)) as handle:
        frames = handle.readframes(handle.getnframes())
        samples = numpy.frombuffer(frames, dtype="<i2").reshape(-1, handle.getnchannels())
        return handle.getframerate(), samples


def test_a_video_soundtrack_comes_back_whole_as_wav(tmp_path):
    write_mp4(tmp_path / "shot.mp4", frames=12, fps=6)  # 2 s of tone at 8 kHz

    data, info = extract_audio(str(tmp_path / "shot.mp4"))

    rate, samples = read_wav(data)
    assert rate == 8000
    assert samples.shape[1] == 2
    assert info["channels"] == 2
    assert info["sample_rate"] == 8000
    assert info["excerpt"] is False
    assert info["of_seconds"] == pytest.approx(2.0, abs=0.1)
    assert info["duration_seconds"] == pytest.approx(info["of_seconds"], abs=0.05)
    # the tone is there, not silence
    assert numpy.abs(samples).max() > 1000


def test_an_excerpt_is_cut_where_asked_and_says_so(tmp_path):
    write_wav(tmp_path / "score.wav", seconds=4.0)

    data, info = extract_audio(str(tmp_path / "score.wav"), start=1.0, duration=0.5)

    rate, samples = read_wav(data)
    assert samples.shape[0] == pytest.approx(0.5 * rate, abs=rate * 0.02)
    assert info["excerpt"] is True
    assert info["start"] == 1.0
    assert info["duration_seconds"] == pytest.approx(0.5, abs=0.02)
    assert info["of_seconds"] == pytest.approx(4.0, abs=0.05)


def test_an_excerpt_past_the_end_is_clipped_to_it(tmp_path):
    write_wav(tmp_path / "score.wav", seconds=2.0)

    data, info = extract_audio(str(tmp_path / "score.wav"), start=1.5, duration=5.0)

    rate, samples = read_wav(data)
    assert samples.shape[0] == pytest.approx(0.5 * rate, abs=rate * 0.02)
    assert info["duration_seconds"] == pytest.approx(0.5, abs=0.02)


def test_a_start_past_the_end_is_refused(tmp_path):
    write_wav(tmp_path / "score.wav", seconds=2.0)

    with pytest.raises(ValueError, match="past the end"):
        extract_audio(str(tmp_path / "score.wav"), start=3.0, duration=1.0)


def test_a_non_positive_duration_is_refused(tmp_path):
    write_wav(tmp_path / "score.wav", seconds=2.0)

    with pytest.raises(ValueError, match="duration"):
        extract_audio(str(tmp_path / "score.wav"), start=0.0, duration=0.0)


def test_a_silent_video_has_no_soundtrack(tmp_path):
    write_mp4(tmp_path / "mute.mp4", frames=6, fps=6, with_audio=False)

    with pytest.raises(NoSoundtrack):
        extract_audio(str(tmp_path / "mute.mp4"))


def test_media_duration_reads_the_header_without_decoding(tmp_path):
    """The route's "serve an audio file whole" fast path wants only the
    length, not a level measurement - media_duration is the container's own
    header figure, the same number extract_audio calls `total`."""
    write_wav(tmp_path / "score.wav", seconds=2.0)

    assert media_duration(str(tmp_path / "score.wav")) == pytest.approx(2.0, abs=0.01)


def test_an_excerpt_does_not_decode_unnecessary_frames(tmp_path):
    """Extracting a short excerpt from a long track must not decode the
    entire file: only frames up to the excerpt end should be yielded by
    container.decode(). This pins the bug where break only exits the inner
    chunk loop, leaving the outer frame loop running to EOF."""
    import av

    # Create a 20-second WAV file - long enough to see the difference
    write_wav(tmp_path / "long.wav", seconds=20.0)

    # Count frames decoded by wrapping the decode method
    frame_count = [0]  # Use list to capture in nested function

    original_decode = av.container.InputContainer.decode

    def counting_decode(self, *args, **kwargs):
        for frame in original_decode(self, *args, **kwargs):
            frame_count[0] += 1
            yield frame

    # Patch and extract a 0.5s excerpt starting at 1.0s
    with mock.patch.object(
        av.container.InputContainer, "decode", counting_decode
    ):
        extract_audio(str(tmp_path / "long.wav"), start=1.0, duration=0.5)

    # With 8 kHz sample rate and ~1024-sample chunks, 0.5s is ~4 chunks
    # from one frame. The full file has 20s = 160000 samples = ~156 frames.
    # If the bug exists, frame_count would be 150+; if fixed, should be ~5-10.
    assert frame_count[0] < 50, (
        f"Decoded {frame_count[0]} frames for a 0.5s excerpt from a 20s file; "
        "expected < 50 (likely bug: outer loop not breaking at stop)"
    )


def write_shifted_two_tone_mp4(path, offset_seconds=1.0, sample_rate=8000):
    """A 4 s AAC soundtrack whose first two seconds are a 220 Hz tone and
    last two are 1760 Hz, with every muxed packet's pts/dts shifted by
    `offset_seconds` - the edit-list / non-zero-start shape a real muxer
    writes, which PyAV surfaces as a non-zero `stream.start_time` (the
    audio twin of `tests/test_media_frames.py::write_shifted_ramp_mp4`).
    The frequency step is what lets a test tell *which* second it got."""
    import av

    container = av.open(str(path), "w")
    audio = container.add_stream("aac", rate=sample_rate)
    audio.layout = "stereo"
    offset = int(offset_seconds * sample_rate)  # audio time_base is 1/rate

    def shifted(packets):
        for packet in packets:
            if packet.pts is not None:
                packet.pts += offset
            if packet.dts is not None:
                packet.dts += offset
            container.mux(packet)

    total = 4 * sample_rate
    t = numpy.arange(total) / sample_rate
    hz = numpy.where(t < 2.0, 220.0, 1760.0)
    sine = (numpy.sin(2 * numpy.pi * hz * t) * 0.25).astype(numpy.float32)
    tone = numpy.stack([sine, sine])
    for start in range(0, total, 1024):
        chunk = av.AudioFrame.from_ndarray(
            numpy.ascontiguousarray(tone[:, start : start + 1024]),
            format="fltp",
            layout="stereo",
        )
        chunk.sample_rate = sample_rate
        chunk.pts = start
        shifted(audio.encode(chunk))
    shifted(audio.encode())
    container.close()


def dominant_hz(samples, rate):
    mono = samples[:, 0].astype(numpy.float64)
    spectrum = numpy.abs(numpy.fft.rfft(mono))
    return numpy.fft.rfftfreq(mono.shape[0], 1 / rate)[spectrum.argmax()]


def test_an_excerpt_is_anchored_on_the_streams_start_time(tmp_path):
    """A stream whose packets carry a pts offset (an edit list, a non-zero
    start) used to have its seek and its per-frame clock both read against
    the container's zero rather than the stream's own start: `start=2.5`
    came back from about 1.5 s - the wrong tone, silently - and `start=0.5`
    came back as a 0.13 s stub. Anchor on `stream.start_time`, as
    `_read_frames` does for video."""
    write_shifted_two_tone_mp4(tmp_path / "shifted.mp4", offset_seconds=1.0)
    import av

    with av.open(str(tmp_path / "shifted.mp4")) as container:
        assert container.streams.audio[0].start_time not in (None, 0), (
            "fixture did not produce a shifted start_time"
        )

    data, info = extract_audio(str(tmp_path / "shifted.mp4"), start=2.5, duration=0.5)
    rate, samples = read_wav(data)
    assert samples.shape[0] == pytest.approx(0.5 * rate, abs=rate * 0.05)
    assert dominant_hz(samples, rate) == pytest.approx(1760, abs=20)

    data, info = extract_audio(str(tmp_path / "shifted.mp4"), start=0.5, duration=0.5)
    rate, samples = read_wav(data)
    assert samples.shape[0] == pytest.approx(0.5 * rate, abs=rate * 0.05)
    assert dominant_hz(samples, rate) == pytest.approx(220, abs=20)


def test_the_resampler_is_flushed_for_an_excerpt_too(tmp_path):
    """The resampler was flushed only for a whole-track request, so an
    excerpt whose last samples the resampler still held ended short. Flush
    always; what the flush yields past `stop` is dropped."""
    from av.audio.resampler import AudioResampler

    write_wav(tmp_path / "score.wav", seconds=4.0)
    flushed = []
    original = AudioResampler.resample

    def observing(self, frame):
        chunks = original(self, frame)
        if frame is None:
            flushed.append(True)
            import av

            # pretend the resampler held a whole second more than it did:
            # only what fits before `stop` may come back
            extra = av.AudioFrame.from_ndarray(
                numpy.zeros((1, 8000 * 2), "<i2"), format="s16", layout="stereo"
            )
            extra.sample_rate = 8000
            return list(chunks) + [extra]
        return chunks

    with mock.patch.object(AudioResampler, "resample", observing):
        data, info = extract_audio(str(tmp_path / "score.wav"), start=1.0, duration=0.5)

    assert flushed == [True]
    rate, samples = read_wav(data)
    assert samples.shape[0] == pytest.approx(0.5 * rate, abs=rate * 0.02)
