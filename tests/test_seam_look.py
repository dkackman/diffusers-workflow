"""Tests for the look measures on analyze_seams records (#1031).

Each frame's lightness, contrast, saturation and detail are taken at decode
(`look_stats`), on a file and on an in-memory clip alike, and every seam
record carries the steps across it. These are measures only: no rule reads
them. The probes run for real on synthetic frames and mp4 files written with
PyAV, as in tests/test_assess.py.
"""

import numpy
import pytest
from PIL import Image, ImageFilter

from dw.media_types import AudioVideo
from dw.shots import shot_record
from dw.tasks.assess import (
    LOOK_MEASURES,
    Media,
    analyze_seams,
    look_stats,
    media_from,
    read_media,
    seams_answer,
)
from dw.tasks.dissolve_videos import dissolve_videos

LOOK_FIELDS = (
    "saturation_step",
    "lightness_step",
    "contrast_step",
    "look_ratio",
    "look_ratio_measure",
    "detail_ratio",
)


def scene(num_frames, size=(128, 72), seed=0, gain=1.0):
    """A colourful, textured picture with a little per-frame noise, so the
    shot has motion of its own; `gain` brightens every frame from the first
    by that factor per frame (a ramp)."""
    rng = numpy.random.default_rng(seed)
    width, height = size
    x = numpy.linspace(0, 1, width)[None, :, None]
    y = numpy.linspace(0, 1, height)[:, None, None]
    base = numpy.concatenate(
        [60 + 120 * x + 0 * y, 80 + 100 * y + 0 * x, 140 - 80 * x + 0 * y], axis=-1
    )
    texture = 25 * ((numpy.indices((height, width)).sum(axis=0) // 4) % 2)[..., None]
    frames = []
    for index in range(num_frames):
        arr = (base + texture) * (gain**index) + rng.normal(0, 1.5, base.shape)
        frames.append(numpy.clip(arr, 0, 255).astype(numpy.uint8))
    return frames


def saturate(frames, factor):
    out = []
    for frame in frames:
        rgb = frame.astype(numpy.float32)
        grey = rgb.mean(axis=-1, keepdims=True)
        out.append(numpy.clip(grey + factor * (rgb - grey), 0, 255).astype(numpy.uint8))
    return out


def darken(frames, factor):
    return [
        numpy.clip(frame.astype(numpy.float32) * factor, 0, 255).astype(numpy.uint8)
        for frame in frames
    ]


def blur(frames, radius=2):
    return [
        numpy.asarray(Image.fromarray(frame).filter(ImageFilter.GaussianBlur(radius)))
        for frame in frames
    ]


def write_mp4(path, frames, fps=24):
    import av

    height, width = frames[0].shape[:2]
    container = av.open(str(path), "w")
    video = container.add_stream("libx264", rate=fps)
    video.width, video.height, video.pix_fmt = width, height, "yuv420p"
    video.options = {"crf": "12"}
    for arr in frames:
        for packet in video.encode(av.VideoFrame.from_ndarray(arr, format="rgb24")):
            container.mux(packet)
    for packet in video.encode():
        container.mux(packet)
    container.close()
    return str(path)


def two_shots(count):
    return [shot_record("a", 0, count), shot_record("b", count, count)]


def seam_of(frames_a, frames_b, tmp_path=None, name="cut.mp4"):
    """The one seam record of frames_a then frames_b, read from an mp4 when
    `tmp_path` is given, otherwise from the in-memory clip."""
    frames = list(frames_a) + list(frames_b)
    shots = two_shots(len(frames_a))
    if tmp_path is not None:
        answer = analyze_seams(write_mp4(tmp_path / name, frames), shots=shots)
    else:
        answer = analyze_seams(AudioVideo(frames, None, None, fps=24), shots=shots)
    (seam,) = answer["seams"]
    return seam, answer


@pytest.fixture
def trusted(monkeypatch):
    monkeypatch.setenv("DW_TRUST_WORKFLOWS", "1")


class TestLookStats:
    def test_a_flat_colour(self):
        frame = numpy.full((72, 128, 3), (200, 100, 50), dtype=numpy.uint8)
        lightness, contrast, saturation, detail = look_stats(frame)
        assert lightness == pytest.approx(0.2126 * 200 + 0.7152 * 100 + 0.0722 * 50)
        assert contrast == pytest.approx(0.0, abs=1e-4)
        assert saturation == pytest.approx(150.0)
        assert detail == pytest.approx(0.0, abs=1e-4)

    def test_black_and_white_halves(self):
        frame = numpy.zeros((72, 128, 3), dtype=numpy.uint8)
        frame[:, 64:] = 255
        lightness, contrast, saturation, detail = look_stats(frame)
        assert lightness == pytest.approx(127.5, abs=0.01)
        assert contrast == pytest.approx(127.5, abs=0.01)
        assert saturation == 0.0
        # only the two columns beside the edge differ from their blur
        assert detail == pytest.approx(2 * (255 / 3) / 128, abs=0.01)

    def test_order_and_dtype(self):
        assert LOOK_MEASURES == ("lightness", "contrast", "saturation", "detail")
        stats = look_stats(numpy.zeros((4, 4, 3), dtype=numpy.uint8))
        assert stats.dtype == numpy.float32 and stats.shape == (4,)

    def test_file_and_in_memory_agree(self, tmp_path, trusted):
        frames = scene(8, size=(256, 144))
        from_file = read_media(write_mp4(tmp_path / "s.mp4", frames))
        in_memory = media_from(AudioVideo(frames, None, None, fps=24))
        assert from_file.looks.shape == in_memory.looks.shape == (8, 4)
        assert from_file.looks.dtype == numpy.float32
        # yuv420 and two scalers apart, but the same reduction
        numpy.testing.assert_allclose(
            from_file.looks.mean(axis=0), in_memory.looks.mean(axis=0), atol=3.0
        )

    def test_thumbs_unchanged(self, tmp_path, trusted):
        media = read_media(write_mp4(tmp_path / "s.mp4", scene(4)))
        assert media.thumbs.shape == (4, 36, 64)
        assert media.thumbs.dtype == numpy.uint8


class TestSeamLook:
    def test_saturation_cut_against_a_same_clip_seam(self, tmp_path, trusted):
        clip = scene(24)
        graded, _ = seam_of(clip, saturate(clip, 1.5), tmp_path, "graded.mp4")
        same, _ = seam_of(clip, clip, tmp_path, "same.mp4")

        assert graded["saturation_step"] > 10
        assert graded["look_ratio_measure"] == "saturation"
        assert graded["look_ratio"] > 3 * max(same["look_ratio"], 1.0)
        assert abs(same["saturation_step"]) < 3

    def test_a_darker_copy_steps_lightness_down(self, tmp_path, trusted):
        clip = scene(24)
        seam, _ = seam_of(clip, darken(clip, 0.7), tmp_path)
        assert seam["lightness_step"] < -10
        assert seam["contrast_step"] < 0

    def test_a_step_inside_a_ramp_scores_low(self):
        ramp = scene(48, gain=1.01)
        seam, _ = seam_of(ramp[:24], ramp[24:])
        assert seam["lightness_step"] > 1
        assert seam["look_ratio"] < 2

    def test_a_softened_copy_has_less_detail(self, tmp_path, trusted):
        clip = scene(24)
        soft, _ = seam_of(clip, blur(clip), tmp_path, "soft.mp4")
        same, _ = seam_of(clip, clip, tmp_path, "same.mp4")
        assert soft["detail_ratio"] < 0.8
        assert same["detail_ratio"] == pytest.approx(1.0, abs=0.1)

    def test_grey_input_has_no_saturation(self):
        grey = [
            numpy.full((72, 128, 3), value, dtype=numpy.uint8) for value in [90] * 12
        ]
        darker = [numpy.full((72, 128, 3), 60, dtype=numpy.uint8)] * 12
        seam, _ = seam_of(grey, darker)
        assert all(field in seam for field in LOOK_FIELDS)
        assert seam["saturation_step"] == 0
        assert seam["lightness_step"] == pytest.approx(-30, abs=0.5)
        # a flat picture has no detail to divide by
        assert seam["detail_ratio"] is None

    def test_frameless_media_gives_nulls(self):
        media = Media(None, None, None, 24)
        answer = seams_answer(media, two_shots(12), "argument")
        (seam,) = answer["seams"]
        assert {field: seam[field] for field in LOOK_FIELDS} == dict.fromkeys(
            LOOK_FIELDS
        )

    def test_single_shot_has_no_seams(self):
        answer = analyze_seams(AudioVideo(scene(12), None, None, fps=24))
        assert answer["seams"] == []

    def test_windows_stop_at_the_shots(self):
        """A one-frame shot either side still gives a step, from that frame."""
        looks = numpy.zeros((3, 4), dtype=numpy.float32)
        looks[:, LOOK_MEASURES.index("saturation")] = [10, 10, 40]
        looks[:, LOOK_MEASURES.index("detail")] = 2.0
        media = Media(
            numpy.zeros((3, 36, 64), numpy.uint8), None, None, 24, looks=looks
        )
        shots = [shot_record("a", 0, 1), shot_record("b", 1, 1), shot_record("c", 2, 1)]
        first, second = seams_answer(media, shots, "argument")["seams"]
        assert first["saturation_step"] == 0
        assert second["saturation_step"] == 30
        assert second["look_ratio"] == 30  # no span to rate, so the floor
        assert second["detail_ratio"] == 1

    def test_a_dissolve_is_measured_across_its_fade(self):
        fps = 24
        clips = [
            AudioVideo(
                [Image.fromarray(frame) for frame in scene(36, seed=1)], None, None, fps
            ),
            AudioVideo(
                [Image.fromarray(frame) for frame in saturate(scene(36, seed=2), 1.6)],
                None,
                None,
                fps,
            ),
        ]
        joined = dissolve_videos(clips, dissolve_frames=12, fps=fps)
        (seam,) = analyze_seams(joined)["seams"]
        assert seam["kind"] == "dissolve"
        assert all(seam[field] is not None for field in LOOK_FIELDS)
        assert seam["saturation_step"] > 10

    def test_no_new_rules_or_findings(self, tmp_path, trusted):
        clip = scene(24)
        _seam, graded = seam_of(clip, saturate(clip, 1.6), tmp_path)
        names = set(graded["rules_applied"]) | {
            entry["rule"] for entry in graded["rules_skipped"]
        }
        names |= {entry["rule"] for entry in graded["findings"]}
        assert not any("look" in name or "detail" in name for name in names)


def _looks_media(saturation, fps=24):
    """A Media whose only picture data is a look per frame: saturation as
    given, the other measures fixed, so window means are exact."""
    looks = numpy.zeros((len(saturation), 4), dtype=numpy.float32)
    looks[:, LOOK_MEASURES.index("saturation")] = saturation
    looks[:, LOOK_MEASURES.index("lightness")] = 100
    looks[:, LOOK_MEASURES.index("detail")] = 2.0
    thumbs = numpy.zeros((len(saturation), 36, 64), numpy.uint8)
    return Media(thumbs, None, None, fps, looks=looks)


def _parts(segments, measure):
    seams = sum(
        after["first"][measure] - before["last"][measure]
        for before, after in zip(segments, segments[1:])
    )
    inside = sum(entry["last"][measure] - entry["first"][measure] for entry in segments)
    total = segments[-1]["last"][measure] - segments[0]["first"][measure]
    return seams, inside, total


class TestSegmentLooks:
    """`segment_looks` on analyze_seams (#1324): each shot's look over the
    first and last second of its own new frames."""

    def test_steps_and_ramps_telescope(self):
        """Three 48-frame shots, each ramping +1 saturation per frame and
        stepping +20 at each seam: the parts are known and sum to the whole."""
        saturation = []
        for shot in range(3):
            saturation += [50 + 20 * shot + frame for frame in range(48)]
        media = _looks_media(saturation)
        shots = [shot_record(name, 48 * i, 48) for i, name in enumerate("abc")]
        answer = seams_answer(media, shots, "argument")
        segments = answer["segment_looks"]

        assert [entry["shot"] for entry in segments] == ["a", "b", "c"]
        assert [entry["between_frames"] for entry in segments] == [
            [0, 47],
            [48, 95],
            [96, 143],
        ]
        for entry in segments:
            assert set(entry["first"]) == set(entry["last"]) == set(LOOK_MEASURES)
            # mean of frames 24..47 minus mean of frames 0..23
            assert entry["last"]["saturation"] - entry["first"]["saturation"] == 24
            assert entry["last"]["lightness"] == entry["first"]["lightness"] == 100
        for before, after in zip(segments, segments[1:]):
            # first second after (ramp start + 20) against the last second before
            assert after["first"]["saturation"] - before["last"]["saturation"] == (
                pytest.approx(20 - 24)
            )
        for measure in LOOK_MEASURES:
            seams, inside, total = _parts(segments, measure)
            assert seams + inside == pytest.approx(total, abs=1e-6)
        assert _parts(segments, "saturation") == (
            pytest.approx(-8),
            pytest.approx(72),
            pytest.approx(64),
        )

    def test_decoded_take_telescopes_and_agrees_with_the_seam_steps(self):
        """The real path: frames through look_stats, graded more saturated
        at each seam."""
        clips = [saturate(scene(36, seed=i), 1.0 + 0.3 * i) for i in range(3)]
        frames = [frame for clip in clips for frame in clip]
        shots = [shot_record(name, 36 * i, 36) for i, name in enumerate("abc")]
        answer = analyze_seams(AudioVideo(frames, None, None, fps=24), shots=shots)
        segments = answer["segment_looks"]
        assert len(segments) == 3
        for measure in LOOK_MEASURES:
            seams, inside, total = _parts(segments, measure)
            assert seams + inside == pytest.approx(total, abs=0.01)
        for seam, before, after in zip(answer["seams"], segments, segments[1:]):
            difference = after["first"]["saturation"] - before["last"]["saturation"]
            assert difference > 5 and seam["saturation_step"] > 5
        assert all(
            isinstance(value, float)
            for entry in segments
            for side in ("first", "last")
            for value in entry[side].values()
        )

    def test_a_short_segment_is_halved(self):
        # 10 frames at 24 fps is under two windows: halves of 5
        saturation = list(range(10)) + [100] * 48
        media = _looks_media(saturation)
        shots = [shot_record("a", 0, 10), shot_record("b", 10, 48)]
        short, _long = seams_answer(media, shots, "argument")["segment_looks"]
        assert short["between_frames"] == [0, 9]
        assert short["first"]["saturation"] == 2  # mean of 0..4
        assert short["last"]["saturation"] == 7  # mean of 5..9

    def test_a_one_frame_segment_is_both_halves(self):
        media = _looks_media([10, 40, 40])
        shots = [shot_record("a", 0, 1), shot_record("b", 1, 2)]
        first, _second = seams_answer(media, shots, "argument")["segment_looks"]
        assert first["between_frames"] == [0, 0]
        assert first["first"]["saturation"] == first["last"]["saturation"] == 10

    def test_first_starts_after_an_overlap(self):
        """A dissolve's fade frames belong to neither segment's windows."""
        saturation = [10] * 48 + [500] * 12 + [70] * 48
        media = _looks_media(saturation)
        shots = [
            shot_record("a", 0, 48),
            {**shot_record("b", 48, 60), "overlap_frames": 12},
        ]
        a, b = seams_answer(media, shots, "argument")["segment_looks"]
        assert a["between_frames"] == [0, 47]
        assert b["between_frames"] == [60, 107]
        assert b["first"]["saturation"] == 70
        assert a["last"]["saturation"] == 10

    def test_first_starts_after_a_dissolve_videos_fade(self):
        clips = [
            AudioVideo(
                [Image.fromarray(frame) for frame in scene(36, seed=1)], None, None, 24
            ),
            AudioVideo(
                [Image.fromarray(frame) for frame in scene(36, seed=2)], None, None, 24
            ),
        ]
        joined = dissolve_videos(clips, dissolve_frames=12, fps=24)
        answer = analyze_seams(joined)
        _a, b = answer["segment_looks"]
        (shot_b,) = [shot for shot in joined.shots if shot.get("overlap_frames")]
        assert b["between_frames"][0] == shot_b["start_frame"] + 12

    def test_a_segment_with_no_new_frames_is_null(self):
        media = _looks_media([10] * 24)
        shots = [
            shot_record("a", 0, 12),
            {**shot_record("b", 12, 12), "overlap_frames": 12},
        ]
        _a, b = seams_answer(media, shots, "argument")["segment_looks"]
        assert b == {"shot": "b", "between_frames": None, "first": None, "last": None}

    def test_frameless_media_gives_an_empty_list(self):
        media = Media(None, None, None, 24)
        answer = seams_answer(media, two_shots(12), "argument")
        assert answer["segment_looks"] == []
        assert len(answer["seams"]) == 1

    def test_audio_only_gives_an_empty_list(self):
        audio = numpy.zeros((1, 48000), dtype=numpy.float32)
        media = Media(None, audio, 48000, None, audio_seconds=1.0)
        answer = seams_answer(media, two_shots(12), "argument")
        assert answer["segment_looks"] == []

    def test_one_shot_gives_an_empty_list(self):
        answer = analyze_seams(AudioVideo(scene(12), None, None, fps=24))
        assert answer["segment_looks"] == []

    def test_existing_seam_fields_unchanged(self):
        clip = scene(24)
        seam, answer = seam_of(clip, saturate(clip, 1.5))
        assert set(answer) == {
            "seams",
            "segment_looks",
            "findings",
            "rules_applied",
            "rules_skipped",
            "shots_source",
        }
        assert all(field in seam for field in LOOK_FIELDS)
        assert "segment_looks" not in seam
