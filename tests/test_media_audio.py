"""extract_audio hands back a soundtrack, or a named slice of one, as WAV -
what get_output_audio needs for a muxed video and for a track too long to
send whole."""

import io
import wave

import numpy
import pytest

from dw.media_audio import NoSoundtrack, extract_audio
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
