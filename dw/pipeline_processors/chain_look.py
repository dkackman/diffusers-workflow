"""Pixel-domain look holding for a long chain (#1191, #1033, #1067).

Each finished segment's frames can be corrected before its carry is taken, so a
long chain keeps the look of its first segment: a grade at the pickup, an
ease-out of saturation/contrast creep, an ease-out of faint texture, and a
keyframe pre-compensated for the boost a model measurably adds. Arrays are
(n, h, w, 3) uint8 in and out; the work is float32 inside, clipped on the way
out. Numpy and torch only.

The ideas come from Smite79/MiniMax-H3-LongVideos (a no-redistribution
licence; concepts only, no code taken).
"""

import numpy
import torch
from PIL import Image

POOL = 64  # tone and look are read off a frame pooled to POOL x POOL by area
QUANTILES = numpy.linspace(0.0, 1.0, 17)
FLAT_SPREAD = 0.02  # a frame whose luma spans less than this has no tone to match
CUT_JUMP = (
    0.25  # per-channel mean quantile gap past which an opening is a cut, not drift
)
LUT_SIZE = 1024
TEXTURE_AMPLITUDE = 0.12  # detail stronger than this (in luma) is structure
TEXTURE_GRADIENT = 0.03  # edge threshold on blurred luma, as at a 3 px box
BOOST_MAX = 1.25  # precompensate clamps each gain to [1, BOOST_MAX]
_EPS = 1e-4


def _float(frames):
    return numpy.asarray(frames, dtype=numpy.float32) / 255.0


def _uint8(pixels):
    return numpy.clip(numpy.rint(pixels * 255.0), 0, 255).astype(numpy.uint8)


def _to_ycc(rgb):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return numpy.stack(
        [y, 0.5 + (b - y) * 0.564, 0.5 + (r - y) * 0.713], axis=-1
    ).astype(numpy.float32)


def _from_ycc(ycc):
    y, cb, cr = ycc[..., 0], ycc[..., 1] - 0.5, ycc[..., 2] - 0.5
    r = y + cr / 0.713
    b = y + cb / 0.564
    g = (y - 0.299 * r - 0.114 * b) / 0.587
    return numpy.stack([r, g, b], axis=-1).astype(numpy.float32)


def _pooled(frame):
    """A frame (uint8 h, w, 3) pooled to POOL x POOL by area, as float YCbCr."""
    pixels = torch.from_numpy(_float(frame)).permute(2, 0, 1)[None]
    pooled = torch.nn.functional.adaptive_avg_pool2d(pixels, POOL)
    return _to_ycc(pooled[0].permute(1, 2, 0).numpy())


def tone(frame):
    """A frame's (3, 17) YCbCr quantiles - 0, 1/16, ..., 1 per channel."""
    ycc = _pooled(frame).reshape(-1, 3)
    return numpy.quantile(ycc, QUANTILES, axis=0).T.astype(numpy.float32)


def look(frame):
    """(luma spread between the 5th and 95th percentile, mean chroma deviation
    from 0.5) of a frame."""
    ycc = _pooled(frame)
    low, high = numpy.quantile(ycc[..., 0], [0.05, 0.95])
    chroma = float(numpy.abs(ycc[..., 1:] - 0.5).mean())
    return float(high - low), chroma


def _lut(source, target):
    """A monotone map from `source` quantiles onto `target` quantiles, sampled
    on LUT_SIZE points of [0, 1]; the ends shift by the end offsets."""
    grid = numpy.linspace(0.0, 1.0, LUT_SIZE, dtype=numpy.float32)
    xs = numpy.concatenate([[0.0], source, [1.0]])
    fs = numpy.concatenate(
        [
            [numpy.clip(target[0] - source[0], 0, 1)],
            target,
            [numpy.clip(1.0 + target[-1] - source[-1], 0, 1)],
        ]
    )
    xs = numpy.maximum.accumulate(xs)
    out = numpy.interp(grid, xs, numpy.maximum.accumulate(fs))
    return numpy.maximum.accumulate(out).astype(numpy.float32)


def _apply_lut(channel, lut):
    return numpy.interp(channel, numpy.linspace(0.0, 1.0, LUT_SIZE), lut)


def is_cut(opening_tone, reference_tone):
    """True when an opening's tone is past CUT_JUMP from the reference's."""
    gap = numpy.abs(opening_tone - reference_tone).mean(axis=1).max()
    return float(gap) > CUT_JUMP


def pickup_grade(frames, carried):
    """Match frame 1 to the carried frame by quantile LUT, and carry the same
    mapping to the last frame, interpolating the LUT per frame between.

    The mapping matches the carried frame's chroma as well as its tone, so a
    carried frame more saturated than the opening raises the opening's
    saturation - that is the match, not a side effect (#1242). Frame 0 is the
    keyframe copy and stays as it is. Returned unchanged when
    there are under two frames (frame 0 plus one more), when a frame is flat,
    or when the opening differs from the carried frame by more than CUT_JUMP
    (a cut, not drift).
    """
    frames = numpy.asarray(frames)
    count = len(frames)
    if count < 2:
        return frames
    carried_tone = tone(numpy.asarray(carried))
    first_tone, last_tone = tone(frames[1]), tone(frames[-1])
    for each in (carried_tone, first_tone, last_tone):
        if each[0, -1] - each[0, 0] < FLAT_SPREAD:
            return frames
    # Known limit (#1242): is_cut reads a fade or exposure ramp past CUT_JUMP
    # as a cut, so a pickup across one is skipped, not graded
    if is_cut(first_tone, carried_tone):
        return frames

    first_luts = [_lut(first_tone[c], carried_tone[c]) for c in range(3)]
    # last_luts refits first_luts on the last frame's quantiles (it maps
    # last_tone where first_luts would), so it is nearly first_luts: the ramp
    # from first to last moves little by design
    last_luts = [
        _lut(
            last_tone[c],
            _apply_lut(last_tone[c], first_luts[c]),
        )
        for c in range(3)
    ]
    out = frames.copy()
    for index in range(1, count):
        weight = (index - 1) / (count - 2) if count > 2 else 0.0
        ycc = _to_ycc(_float(frames[index]))
        for c in range(3):
            lut = (1 - weight) * first_luts[c] + weight * last_luts[c]
            ycc[..., c] = _apply_lut(ycc[..., c], lut)
        out[index] = _uint8(_from_ycc(ycc))
    return out


def ease_creep(frames, scene_look, tolerance=0.03):
    """Scale contrast (about the frame mean) and chroma (about 0.5) down by a
    factor running linearly from 1 at the first frame to the needed gain at the
    last, when the last frame's look exceeds the scene's by more than the
    tolerance. Gains are at most 1: a darker or paler shot is untouched."""
    frames = numpy.asarray(frames)
    count = len(frames)
    if count < 2:
        return frames
    spread, chroma = look(frames[-1])
    scene_spread, scene_chroma = scene_look
    contrast_gain = chroma_gain = 1.0
    if spread - scene_spread > tolerance and spread > _EPS:
        contrast_gain = min(1.0, scene_spread / spread)
    if chroma - scene_chroma > tolerance and chroma > _EPS:
        chroma_gain = min(1.0, scene_chroma / chroma)
    if contrast_gain == 1.0 and chroma_gain == 1.0:
        return frames
    out = frames.copy()
    for index in range(count):
        ramp = index / (count - 1)
        out[index] = _scaled(
            frames[index],
            1.0 + (contrast_gain - 1.0) * ramp,
            1.0 + (chroma_gain - 1.0) * ramp,
        )
    return out


def _scaled(frame, contrast, chroma):
    """Contrast about the frame's mean luma and chroma about 0.5, as factors."""
    ycc = _to_ycc(_float(frame))
    mean = ycc[..., 0].mean()
    ycc[..., 0] = mean + (ycc[..., 0] - mean) * contrast
    ycc[..., 1:] = 0.5 + (ycc[..., 1:] - 0.5) * chroma
    return _uint8(_from_ycc(ycc))


def _detail(luma):
    """(luma - its box blur, mask of faint detail in flat areas). The blur
    radius is about height / 64; edges and strong detail fall outside the
    mask. An edge softer than about 0.09 at a 3 px box is flat to this mask
    and counts as texture (#1242). A blurred step's gradient falls as 1 / (2 radius + 1), so the edge
    test scales it back to the 3 px box TEXTURE_GRADIENT was set at: the mask
    reads the same at any resolution."""
    radius = max(1, luma.shape[0] // 64)
    tensor = torch.from_numpy(luma)[None, None]
    padded = torch.nn.functional.pad(tensor, (radius,) * 4, mode="replicate")
    blurred = torch.nn.functional.avg_pool2d(padded, 2 * radius + 1, stride=1)[0, 0]
    blurred = blurred.numpy()
    gy, gx = numpy.gradient(blurred)
    flat = numpy.hypot(gx, gy) * (2 * radius + 1) < TEXTURE_GRADIENT * 3
    detail = luma - blurred
    return detail, flat & (numpy.abs(detail) < TEXTURE_AMPLITUDE)


def texture(frame):
    """The mean of the faint detail |luma - blur(luma)| over the flat, faint
    area of a frame; 0 when there is none."""
    detail, mask = _detail(_to_ycc(_float(frame))[..., 0])
    return float(numpy.abs(detail)[mask].mean()) if mask.any() else 0.0


def scene_texture(frames):
    """The roughest of a segment's first, middle and last frames."""
    frames = numpy.asarray(frames)
    picks = {0, len(frames) // 2, len(frames) - 1}
    return max(texture(frames[i]) for i in picks)


def _eased(frame, cut):
    ycc = _to_ycc(_float(frame))
    detail, mask = _detail(ycc[..., 0])
    ycc[..., 0] -= cut * mask * detail
    return _uint8(_from_ycc(ycc))


def _cut_for(frame, limit, max_cut):
    """The smallest cut in [0, max_cut] that brings the frame's texture to
    `limit`, by bisection; max_cut when even that falls short."""
    if texture(frame) <= limit:
        return 0.0
    if texture(_eased(frame, max_cut)) > limit:
        return max_cut
    low, high = 0.0, max_cut
    for _ in range(10):
        middle = (low + high) / 2
        if texture(_eased(frame, middle)) > limit:
            low = middle
        else:
            high = middle
    return high


def ease_texture(frames, scene_texture, tolerance=0.03, max_cut=0.5):
    """Subtract a fraction (at most max_cut) of the masked faint detail at
    whichever end exceeds the scene's texture by more than the tolerance,
    interpolating the fraction linearly across the shot. Frame 0 is the keyframe
    copy and stays as it is; the ends are frame 1 and the last."""
    frames = numpy.asarray(frames)
    count = len(frames)
    if count < 2:
        return frames
    limit = scene_texture + tolerance
    ends = (_cut_for(frames[1], limit, max_cut), _cut_for(frames[-1], limit, max_cut))
    if not any(ends):
        return frames
    out = frames.copy()
    for index in range(1, count):
        weight = (index - 1) / (count - 2) if count > 2 else 0.0
        cut = ends[0] + (ends[1] - ends[0]) * weight
        if cut > 0:
            out[index] = _eased(frames[index], cut)
    return out


def precompensate(frame, boost):
    """Divide contrast and chroma by the measured boost (contrast_gain,
    chroma_gain), each clamped to [1, BOOST_MAX]; (1, 1) is the identity."""
    contrast, chroma = (min(max(float(g), 1.0), BOOST_MAX) for g in boost)
    if contrast == 1.0 and chroma == 1.0:
        return frame
    return _scaled(numpy.asarray(frame), 1.0 / contrast, 1.0 / chroma)


class LookHold:
    """What a chain keeps between segments: segment 1's look and texture, the
    running boost, and the keyframe the next segment was fed."""

    def __init__(self, flags):
        self.flags = flags
        self.scene_look = None
        self.scene_texture = None
        self._ratios = []
        self._fed = None
        self._fed_tone = None

    def boost(self):
        if not self._ratios:
            return (1.0, 1.0)
        return tuple(float(v) for v in numpy.mean(self._ratios, axis=0))

    def correct(self, frames, carried, first):
        """One segment's PIL frames, corrected. Segment 1 only sets the scene;
        later ones also learn the boost from their rendered opening."""
        stack = numpy.stack([numpy.asarray(f.convert("RGB")) for f in frames])
        opening = stack[min(1, len(stack) - 1)]
        if first:
            self.scene_look, self.scene_texture = look(opening), scene_texture(stack)
            return frames
        if self._fed is not None and not is_cut(tone(opening), self._fed_tone):
            rendered = look(opening)
            # Each gain is learned on its own: a fed frame with no chroma (a
            # greyscale keyframe) says nothing of the chroma boost but still
            # measures the contrast one, and the other way round (#1242)
            self._ratios.append(
                [r / f if f > _EPS else 1.0 for r, f in zip(rendered, self._fed)]
            )
        if self.flags["pickup"] and carried is not None:
            stack = pickup_grade(stack, numpy.asarray(carried.convert("RGB")))
        if self.flags["creep"]:
            stack = ease_creep(stack, self.scene_look)
        if self.flags["texture"]:
            stack = ease_texture(stack, self.scene_texture)
        return [Image.fromarray(each) for each in stack]

    def feed(self, carry):
        """The keyframe for the next segment: the carry, pre-compensated by the
        boost so far. Records its look to measure the next opening against."""
        if self.flags["precompensate"]:
            pixels = numpy.asarray(carry.convert("RGB"))
            carry = Image.fromarray(precompensate(pixels, self.boost()))
        pixels = numpy.asarray(carry.convert("RGB"))
        self._fed, self._fed_tone = look(pixels), tone(pixels)
        return carry
