"""probe_media reports what the server knows about a generated file and
would otherwise not say - an agent cannot listen, so duration and level
are the only way it checks an audio deliverable."""

import math

import numpy
import pytest

from dw.media_info import probe_media, probe_metadata


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


class TestLoudness:
    """integrated_lufs and true_peak_dbfs, #361 - a level measured over the
    whole track rather than a single sample."""

    def test_a_known_tone_reads_within_half_a_lu_of_its_reference(self, tmp_path):
        # The known reference is pyloudnorm's own measurement of the exact
        # waveform write_wav encodes - probe_media's answer, reached through
        # a full decode of the file it wrote, must agree with a direct
        # measurement of the source samples to within codec/quantization
        # noise (the wav here is lossless, so this is tight).
        import pyloudnorm

        seconds, rate, amplitude = 2.0, 8000, 0.5
        write_wav(
            tmp_path / "tone.wav",
            seconds=seconds,
            sample_rate=rate,
            amplitude=amplitude,
        )
        t = numpy.arange(int(seconds * rate)) / rate
        tone = numpy.sin(2 * numpy.pi * 220 * t) * amplitude
        reference = pyloudnorm.Meter(rate).integrated_loudness(
            numpy.stack([tone, tone], axis=1)
        )

        info = probe_media(str(tmp_path / "tone.wav"))

        assert info["integrated_lufs"] == pytest.approx(reference, abs=0.5)

    def test_silence_reports_lufs_as_none_not_minus_infinity(self, tmp_path):
        write_wav(tmp_path / "quiet.wav", amplitude=0.0)

        info = probe_media(str(tmp_path / "quiet.wav"))

        assert info["integrated_lufs"] is None
        assert info["true_peak_dbfs"] == -120.0

    def test_a_clip_shorter_than_the_gating_block_reports_lufs_as_none(self, tmp_path):
        write_wav(tmp_path / "short.wav", seconds=0.1, sample_rate=8000, amplitude=0.5)

        info = probe_media(str(tmp_path / "short.wav"))

        assert info["integrated_lufs"] is None
        # true peak is still a single-sample-independent measurement, and is
        # defined for any nonempty track regardless of length
        assert info["true_peak_dbfs"] < 0.0

    def test_true_peak_is_reported_alongside_sample_peak(self, tmp_path):
        write_wav(tmp_path / "score.wav", seconds=2.0, sample_rate=8000, amplitude=0.5)

        info = probe_media(str(tmp_path / "score.wav"))

        assert info["true_peak_dbfs"] == pytest.approx(-6.0, abs=0.5)


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


class TestEnvelope:
    """The per-second level: what says *where* in a track something is,
    rather than only how loud the whole of it was."""

    def write_gapped_wav(self, path, seconds=4.0, sample_rate=8000, silent_second=2):
        """A tone with one second of silence punched out of the middle of it."""
        import wave

        t = numpy.arange(int(seconds * sample_rate)) / sample_rate
        samples = (numpy.sin(2 * numpy.pi * 220 * t) * 0.5 * 32767).astype("<i2")
        samples[silent_second * sample_rate : (silent_second + 1) * sample_rate] = 0
        with wave.open(str(path), "w") as handle:
            handle.setnchannels(2)
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            handle.writeframes(numpy.stack([samples, samples], 1).tobytes())

    def test_it_is_off_unless_asked_for(self, tmp_path):
        write_wav(tmp_path / "score.wav", seconds=2.0)

        assert "envelope" not in probe_media(str(tmp_path / "score.wav"))

    def test_one_entry_per_second_of_the_track(self, tmp_path):
        self.write_gapped_wav(tmp_path / "score.wav", seconds=4.0)

        envelope = probe_media(str(tmp_path / "score.wav"), envelope=True)["envelope"]

        assert envelope["interval_seconds"] == 1.0
        assert len(envelope["rms_dbfs"]) == 4
        assert len(envelope["peak_dbfs"]) == 4

    def test_it_finds_the_second_the_sound_stops_in(self, tmp_path):
        self.write_gapped_wav(tmp_path / "score.wav", seconds=4.0, silent_second=2)

        envelope = probe_media(str(tmp_path / "score.wav"), envelope=True)["envelope"]

        assert envelope["rms_dbfs"][2] == pytest.approx(-120.0)
        assert envelope["peak_dbfs"][2] == pytest.approx(-120.0)
        for second in (0, 1, 3):
            assert envelope["rms_dbfs"][second] > -20.0

    def test_a_video_soundtrack_gets_one_too(self, tmp_path):
        write_mp4(tmp_path / "shot.mp4", frames=12, fps=6)

        info = probe_media(str(tmp_path / "shot.mp4"), envelope=True)

        assert info["kind"] == "video"
        assert info["frame_count"] == 12
        # 2 s of soundtrack - a lossy codec's own priming/padding can decode
        # a fraction of a second past the nominal duration, but that trailing
        # fragment is folded into the last full bin rather than reported on
        # its own (#277)
        assert len(info["envelope"]["rms_dbfs"]) == 2
        assert info["envelope"]["rms_dbfs"][0] > -30.0
        assert info["peak_dbfs"] < 0.0

    def test_a_silent_video_has_no_envelope_to_report(self, tmp_path):
        write_mp4(tmp_path / "mute.mp4", frames=12, fps=6, with_audio=False)

        assert "envelope" not in probe_media(str(tmp_path / "mute.mp4"), envelope=True)

    def test_a_genuine_partial_last_second_stands_on_its_own(self, tmp_path):
        """A real duration that isn't a whole number of seconds gets a
        shorter final bin, not merged away into the second before it - the
        fold in #277 is for a codec's own decode-past-duration fragment, not
        for real trailing content (#278)."""
        write_wav(tmp_path / "tail.wav", seconds=2.5, sample_rate=8000, amplitude=0.5)

        info = probe_media(str(tmp_path / "tail.wav"), envelope=True)

        assert info["duration_seconds"] == pytest.approx(2.5, abs=0.01)
        envelope = info["envelope"]
        assert len(envelope["rms_dbfs"]) == 3
        # the half-second tail still carries the same tone, not silence
        assert envelope["rms_dbfs"][2] > -20.0

    def test_the_seconds_sum_back_to_the_whole_track(self, tmp_path):
        """A bin holds the same sums the whole-track level is made of, so
        recombining them has to land on the level the track reports."""
        self.write_gapped_wav(tmp_path / "score.wav", seconds=4.0)

        info = probe_media(str(tmp_path / "score.wav"), envelope=True)

        assert max(info["envelope"]["peak_dbfs"]) == pytest.approx(
            info["peak_dbfs"], abs=0.01
        )
        power = numpy.mean([10 ** (db / 10) for db in info["envelope"]["rms_dbfs"]])
        assert 10 * math.log10(power) == pytest.approx(info["mean_dbfs"], abs=0.1)


class TestProbeMetadata:
    """probe_metadata answers the same header-level fields probe_media does,
    for a fraction of the cost - validation only ever needs metadata, never
    a level or a loudness figure, so it should never pay for a full decode
    of a file it is only checking the shape of (B9)."""

    PARITY_KEYS = ("kind", "width", "height", "duration_seconds")

    @pytest.mark.parametrize(
        "extension",
        ["mp4", "mkv"],
        ids=["header-frame-count", "no-header-frame-count"],
    )
    def test_frame_count_and_shape_match_probe_media(self, tmp_path, extension):
        # mp4 writes libx264's frame count into the stream header; mkv does
        # not (video.frames comes back 0), so probe_media falls back to a
        # full decode to count frames there - the exact case probe_metadata
        # has to match without paying for that decode.
        path = tmp_path / f"shot.{extension}"
        write_mp4(path, frames=12, fps=6, width=32, height=16)

        decoded = probe_media(str(path))
        header = probe_metadata(str(path))

        assert header["frame_count"] == decoded["frame_count"] == 12
        for key in self.PARITY_KEYS:
            assert header[key] == decoded[key]
        assert header["fps"] == pytest.approx(decoded["fps"])

    def test_an_audio_file_matches_probe_media_on_the_shared_keys(self, tmp_path):
        write_wav(tmp_path / "score.wav", seconds=2.0, sample_rate=8000)

        decoded = probe_media(str(tmp_path / "score.wav"))
        header = probe_metadata(str(tmp_path / "score.wav"))

        assert header["kind"] == decoded["kind"] == "audio"
        assert header["sample_rate"] == decoded["sample_rate"] == 8000
        assert header["channels"] == decoded["channels"] == 2
        assert header["duration_seconds"] == pytest.approx(
            decoded["duration_seconds"], abs=0.01
        )

    def test_it_carries_no_loudness_keys(self, tmp_path):
        write_wav(tmp_path / "score.wav", seconds=2.0, sample_rate=8000)

        info = probe_metadata(str(tmp_path / "score.wav"))

        for key in ("peak_dbfs", "mean_dbfs", "integrated_lufs", "true_peak_dbfs"):
            assert key not in info

    def test_a_video_soundtrack_is_never_decoded_for_its_level(self, tmp_path):
        write_mp4(tmp_path / "shot.mp4", frames=12, fps=6, with_audio=True)

        info = probe_metadata(str(tmp_path / "shot.mp4"))

        assert info["kind"] == "video"
        assert info["sample_rate"] == 8000
        for key in ("peak_dbfs", "mean_dbfs", "integrated_lufs", "true_peak_dbfs"):
            assert key not in info

    def test_a_file_that_is_not_media_answers_none(self, tmp_path):
        (tmp_path / "notes.txt").write_text("not media")

        assert probe_metadata(str(tmp_path / "notes.txt")) is None

    def test_it_never_calls_container_decode(self, tmp_path, monkeypatch):
        # Neither branch - header count present (mp4) or counted by
        # demuxing (mkv) - may call decode: that is the whole point of a
        # metadata-only probe. Patched on the av class itself (not a
        # string "dw..." target), so any call anywhere raises.
        import av

        def _boom(self, *args, **kwargs):
            raise AssertionError("probe_metadata must not decode")

        monkeypatch.setattr(av.container.InputContainer, "decode", _boom)

        write_mp4(tmp_path / "shot.mp4", frames=12, fps=6, with_audio=True)
        write_mp4(tmp_path / "shot.mkv", frames=12, fps=6, with_audio=True)

        info_mp4 = probe_metadata(str(tmp_path / "shot.mp4"))
        info_mkv = probe_metadata(str(tmp_path / "shot.mkv"))

        assert info_mp4["frame_count"] == 12
        assert info_mkv["frame_count"] == 12

    def test_a_damaged_track_still_reports_header_fields(self, tmp_path):
        path = tmp_path / "shot.mkv"
        write_mp4(path, frames=12, fps=6, width=32, height=16)

        raw = bytearray(path.read_bytes())
        mid = len(raw) // 2
        for i in range(mid, len(raw), 64):
            raw[i] = (raw[i] + 137) % 256
        path.write_bytes(bytes(raw))

        info = probe_metadata(str(path))

        assert info is not None
        assert info["kind"] == "video"

    def test_a_codec_outside_the_allowlist_falls_back_to_a_decode(
        self, tmp_path, monkeypatch
    ):
        # Demuxing counts one packet per frame exactly only for a codec dw
        # actually writes; anything else must fall back to probe_media's
        # real decode rather than trust a count that might not hold
        # (review round 1, B9). `_video_codec_name` is its own function
        # precisely so this can be exercised without a fixture encoded
        # with a genuinely unlisted codec.
        import dw.media_info as media_info_module

        path = tmp_path / "shot.mkv"
        write_mp4(path, frames=12, fps=6, width=32, height=16)
        monkeypatch.setattr(
            media_info_module, "_video_codec_name", lambda video: "flv1"
        )

        info = probe_metadata(str(path))

        assert info["frame_count"] == 12

    def test_a_demux_failure_falls_back_to_a_decode(self, tmp_path, monkeypatch):
        # A demux that raises partway through must not silently drop
        # frame_count when probe_media could still supply it (review round
        # 1, B9) - it falls back to a real decode instead. `decode()` itself
        # calls `demux()` internally (proven empirically), so the fake only
        # breaks the *first* call - probe_metadata's own attempt - and lets
        # every later one (probe_media's fallback decode, on a freshly
        # opened container) run for real; otherwise no fallback could ever
        # succeed, decode and demux failing together.
        import av

        orig_demux = av.container.InputContainer.demux
        state = {"failed_once": False}

        def _boom_once(self, *args, **kwargs):
            if not state["failed_once"]:
                state["failed_once"] = True
                raise RuntimeError("demux exploded")
            return orig_demux(self, *args, **kwargs)

        monkeypatch.setattr(av.container.InputContainer, "demux", _boom_once)

        path = tmp_path / "shot.mkv"
        write_mp4(path, frames=12, fps=6, width=32, height=16)

        info = probe_metadata(str(path))

        assert state["failed_once"]
        assert info["frame_count"] == 12
