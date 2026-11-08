"""look_hold (#1191): pixel corrections on a chain segment's frames."""

import numpy
import pytest

from dw.pipeline_processors.chain_look import (
    LookHold,
    ease_creep,
    ease_texture,
    look,
    pickup_grade,
    precompensate,
    texture,
    tone,
)

H = W = 64


def gradient(scale=1.0, shift=(0, 0, 0)):
    ramp = numpy.linspace(40, 200, W, dtype=numpy.float32)
    base = numpy.tile(ramp, (H, 1))
    frame = numpy.stack([base, base * 0.9 + 10, base * 0.8 + 20], axis=-1)
    mean = frame.mean()
    frame = mean + (frame - mean) * scale + numpy.array(shift, dtype=numpy.float32)
    return numpy.clip(frame, 0, 255).astype(numpy.uint8)


def clip_of(frame, count=5):
    return numpy.stack([frame] * count)


def tinted(frame, cb):
    from dw.pipeline_processors.chain_look import _float, _from_ycc, _to_ycc, _uint8

    ycc = _to_ycc(_float(frame))
    ycc[..., 1] += cb / 255.0
    return _uint8(_from_ycc(ycc))


def test_pickup_grade_matches_the_opening_to_the_carried_frame():
    carried = gradient()
    frames = clip_of(tinted(carried, 10))
    graded = pickup_grade(frames, carried)
    assert numpy.abs(tone(graded[1]) - tone(carried)).max() * 255 <= 1.0
    assert numpy.abs(tone(graded[-1]) - tone(carried)).max() * 255 <= 1.0
    assert (graded[0] == frames[0]).all()  # the keyframe copy is left alone


def test_pickup_grade_skips_a_cut_and_a_flat_frame():
    carried = gradient()
    cut = clip_of((carried * 0.3).astype(numpy.uint8))
    assert (pickup_grade(cut, carried) == cut).all()
    flat = clip_of(numpy.full((H, W, 3), 90, numpy.uint8))
    assert (pickup_grade(flat, carried) == flat).all()


def test_creep_brings_the_last_frame_to_the_scene_look_within_tolerance():
    scene = look(gradient())
    frames = numpy.stack([gradient(1.0), gradient(1.2), gradient(1.4)])
    assert look(frames[-1])[0] - scene[0] > 0.03
    eased = ease_creep(frames, scene)
    assert look(eased[-1])[0] - scene[0] <= 0.03
    assert (eased[0] == frames[0]).all()


def test_creep_never_brightens_and_never_touches_a_darker_shot():
    scene = look(gradient())
    duller = clip_of(gradient(0.6))
    assert (ease_creep(duller, scene) == duller).all()
    frames = numpy.stack([gradient(1.0), gradient(1.5)])
    eased = ease_creep(frames, scene)
    assert look(eased[-1])[0] <= look(frames[-1])[0]
    assert abs(float(eased[-1].mean()) - float(frames[-1].mean())) < 3


def test_texture_ease_leaves_edges_alone():
    rng = numpy.random.default_rng(0)
    field = numpy.full((H, W, 3), 128.0)
    field[:, W // 2 :] = 200.0  # a sharp step edge
    noisy = field.copy()
    noisy[:, : W // 2 - 4] += rng.normal(0, 25, (H, W // 2 - 4, 1))  # faint noise
    noisy[:, W // 2 + 4 :] += rng.normal(0, 25, (H, W // 2 - 4, 1))
    frame = numpy.clip(noisy, 0, 255).astype(numpy.uint8)
    frames = clip_of(frame)
    before = texture(frame)
    assert before > 0.03
    eased = ease_texture(frames, 0.0)
    after = texture(eased[-1])
    assert after < before
    assert after >= before * 0.5 - 1e-3  # at most half
    # the edge columns survive
    assert (
        eased[-1][:, W // 2 - 2 : W // 2 + 2] == frame[:, W // 2 - 2 : W // 2 + 2]
    ).all()


@pytest.mark.parametrize("height,width", [(512, 512), (1088, 1920)])
def test_texture_ease_leaves_edges_alone_at_full_resolution(height, width):
    rng = numpy.random.default_rng(0)
    mid = width // 2
    field = numpy.full((height, width, 3), 128.0)
    field[:, mid:] = 200.0
    noisy = field.copy()
    noisy[:, : mid - 4] += rng.normal(0, 25, (height, mid - 4, 1))
    noisy[:, mid + 4 :] += rng.normal(0, 25, (height, width - mid - 4, 1))
    frame = numpy.clip(noisy, 0, 255).astype(numpy.uint8)
    eased = ease_texture(clip_of(frame, 3), 0.0)
    band = slice(mid - 4, mid + 4)
    change = numpy.abs(eased[-1][:, band].astype(int) - frame[:, band].astype(int))
    assert change.max() <= 1
    assert texture(eased[-1]) < texture(frame)  # the flat-field noise still drops


def test_texture_ease_leaves_the_keyframe_copy_alone():
    rng = numpy.random.default_rng(1)
    frame = numpy.clip(128 + rng.normal(0, 25, (H, W, 3)), 0, 255).astype(numpy.uint8)
    frames = clip_of(frame, 4)
    eased = ease_texture(frames, 0.0)
    assert (eased[0] == frames[0]).all()
    assert (eased[-1] != frames[-1]).any()


def test_texture_ease_leaves_a_smooth_shot_alone():
    frames = clip_of(gradient())
    assert (ease_texture(frames, texture(gradient())) == frames).all()


def test_precompensate_is_identity_at_boost_one():
    frame = gradient(1.3)
    assert (precompensate(frame, (1.0, 1.0)) == frame).all()
    assert (precompensate(frame, (0.8, 0.5)) == frame).all()  # clamped up to 1


def test_precompensate_divides_and_clamps():
    frame = gradient(1.3)
    out = precompensate(frame, (1.1, 1.0))
    assert look(out)[0] < look(frame)[0]
    big = precompensate(frame, (9.0, 9.0))
    capped = precompensate(frame, (1.25, 1.25))
    assert (big == capped).all()


def test_a_cut_opening_does_not_move_the_boost():
    from PIL import Image

    hold = LookHold(
        dict.fromkeys(("pickup", "creep", "texture", "precompensate"), True)
    )
    carry = Image.fromarray(gradient())
    hold.correct([carry] * 3, None, True)
    hold.feed(carry)
    cut = Image.fromarray((gradient() * 0.3).astype(numpy.uint8))
    hold.correct([carry, cut, cut], carry, False)
    assert hold.boost() == (1.0, 1.0)
    drift = Image.fromarray(gradient(1.2))
    hold.feed(carry)
    hold.correct([carry, drift, drift], carry, False)
    assert hold.boost()[0] > 1.1


def test_a_greyscale_feed_still_learns_the_contrast_boost():
    # The fed frame has no chroma, so the chroma gain is unmeasurable (stays
    # 1.0) - but that must not discard the contrast measure (#1242)
    from PIL import Image

    def grey(scale):
        ramp = numpy.linspace(60, 180, W, dtype=numpy.float32)
        mean = ramp.mean()
        row = numpy.clip(mean + (ramp - mean) * scale, 0, 255).astype(numpy.uint8)
        return Image.fromarray(numpy.tile(row[None, :, None], (H, 1, 3)))

    hold = LookHold(
        dict.fromkeys(("pickup", "creep", "texture", "precompensate"), True)
    )
    carry = grey(1.0)
    hold.correct([carry] * 3, None, True)
    hold.feed(carry)
    rendered = grey(1.3)
    hold.correct([carry, rendered, rendered], carry, False)
    contrast, chroma = hold.boost()
    assert contrast > 1.15
    assert chroma == 1.0


def test_a_near_greyscale_feed_does_not_learn_a_chroma_boost():
    # Nonzero but faint chroma (jpeg/vae residue) is still "about 0": the
    # opening/fed chroma ratio is noise and must stay 1.0 (#1242)
    from PIL import Image

    def tinted(scale, tint):
        ramp = numpy.linspace(60, 180, W, dtype=numpy.float32)
        mean = ramp.mean()
        row = numpy.clip(mean + (ramp - mean) * scale, 0, 255)
        rgb = numpy.tile(row[None, :, None], (H, 1, 3))
        rgb[..., 0] += tint
        return Image.fromarray(numpy.clip(rgb, 0, 255).astype(numpy.uint8))

    hold = LookHold(
        dict.fromkeys(("pickup", "creep", "texture", "precompensate"), False)
    )
    hold.flags["precompensate"] = True
    carry = tinted(1.0, 1)
    hold.correct([carry] * 3, None, True)
    hold.feed(carry)
    rendered = tinted(1.3, 6)
    hold.correct([carry, rendered, rendered], carry, False)
    contrast, chroma = hold.boost()
    assert contrast > 1.15
    assert chroma == 1.0


def test_correct_reports_what_each_part_measured_and_applied():
    # #1251: the notes carry the learned boost, pickup outcome, creep gains and
    # texture cuts, so a consumer can read them off the seam log line
    from PIL import Image

    hold = LookHold(
        dict.fromkeys(("pickup", "creep", "texture", "precompensate"), True)
    )
    carry = Image.fromarray(gradient())
    hold.correct([carry] * 3, None, True)
    hold.feed(carry)
    drift = Image.fromarray(gradient(1.2))
    hold.correct([carry, drift, drift], carry, False)
    text = " | ".join(hold.notes)
    for expected in (
        "precompensate divided out",
        "measured opening/fed ratio",
        "pickup",
        "creep contrast gain",
        "texture cut",
    ):
        assert expected in text
    cut = Image.fromarray((gradient() * 0.3).astype(numpy.uint8))
    hold.feed(carry)
    hold.correct([carry, cut, cut], carry, False)
    assert "cut" in " | ".join(hold.notes)


def test_a_grey_feed_after_coloured_ones_leaves_the_chroma_boost_alone():
    # An unmeasurable gain adds no sample: a grey feed after a learned chroma
    # boost must not average it back toward 1.0 (#1242)
    from PIL import Image

    ramp = numpy.linspace(60, 180, W, dtype=numpy.float32).astype(numpy.uint8)
    grey = Image.fromarray(numpy.tile(ramp[None, :, None], (H, 1, 3)))
    hold = LookHold(
        dict.fromkeys(("pickup", "creep", "texture", "precompensate"), True)
    )
    hold.correct([grey] * 3, None, True)
    # What two earlier coloured feeds learned
    hold._ratios[0].extend([1.0, 1.0])
    hold._ratios[1].extend([1.2, 1.2])
    hold.feed(grey)
    hold.correct([grey, grey, grey], grey, False)
    assert hold.boost()[1] == pytest.approx(1.2)
    assert len(hold._ratios[0]) == 3  # the contrast gain was still measured
    assert len(hold._ratios[1]) == 2
