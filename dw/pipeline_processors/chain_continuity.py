"""The three ways a chain carries continuity from one segment into the
next (`continuity`): the last frame as a keyframe, the last segment as a
video reference, or the last segment's tail as a frame-0 H3 guide.

Each mode is a class with `extract(artifact)`, the carry taken off a finished
segment, and `inject(arguments, carry, segment_argument)`, the carry laid
into the next segment's arguments. `CONTINUITY_MODES` names them on
`h3_rules.CHAIN_CONTINUITY_MODES`, which validation reads too. Split out of
`chain.py`, which imports this module and never the reverse.
"""

import sys
from dataclasses import dataclass

import numpy
import torch

from ..dsp import as_channels_samples
from ..media_types import AudioVideo
from ..task_domains import frames_to_samples
from ..tasks.video_utils import extract_frame, frames_as_pil_list
from .chain_pick import sharpest_index
from .h3_rules import CHAIN_CONTINUITY_MODES, GUIDES_INPUT


class LastFrameContinuity:
    """Carry the last frame of each segment into the next as its keyframe."""

    def __init__(self, config):
        self.config = config

    def extract(self, artifact):
        # With carry_pick, the sharpest of the last N frames; `picked_back`
        # is how many frames after it the segment drops (0 = the last frame)
        self.picked_back = 0
        if self.config.carry_pick:
            frames = frames_as_pil_list(artifact)
            self.picked_back = sharpest_index(frames, self.config.carry_pick)
            return frames[len(frames) - 1 - self.picked_back]
        return extract_frame(artifact, -1)

    def inject(self, arguments, carry, segment_argument):
        target = arguments.get(segment_argument)
        if isinstance(target, list):
            # A references list - the carry frame is appended as an image
            # reference alongside the workflow's own references
            arguments[segment_argument] = _with_carry_reference(target, carry)
        else:
            arguments[segment_argument] = carry


class LastSegmentContinuity:
    """Carry the whole previous segment into the next as a video reference.

    A still frame carries pose and colour and nothing else. A reference-
    conditioned pipeline (MiniMax H3's ref2va) can take the previous segment
    itself - its frames and the soundtrack generated with them - which carries
    motion, camera and voice across the seam instead of just appearance.

    Only the tail of the segment is worth carrying: 'carry_frames' bounds it,
    and the soundtrack is cut to the same span so the reference's own audio and
    video stay aligned. The generated media is already at the pipeline's own
    rates, so the reference declares no rate of its own and nothing is
    resampled on the way back in.
    """

    def __init__(self, config):
        self.config = config

    def extract(self, artifact):
        frames = frames_as_pil_list(artifact)
        audio, sample_rate = _generated_audio(artifact)

        carry_frames = self.config.carry_frames
        if carry_frames is not None and carry_frames < len(frames):
            if audio is not None:
                if self.config.fps is None:
                    raise ValueError(
                        "Trimming a last_segment carry needs the frame rate - "
                        "set 'fps' on the chain or a 'frame_rate' pipeline argument"
                    )
                samples = frames_to_samples(carry_frames, self.config.fps, sample_rate)
                audio = audio[:, -samples:]
            frames = frames[-carry_frames:]

        if not self.config.carry_audio:
            audio, sample_rate = None, None

        return _SegmentCarry(frames, audio, sample_rate)

    def inject(self, arguments, carry, segment_argument):
        target = arguments.get(segment_argument)
        if not isinstance(target, list):
            raise ValueError(
                f"The 'last_segment' continuity carries a video reference, so "
                f"'{segment_argument}' must be a references list, not a "
                f"{type(target).__name__}"
            )
        arguments[segment_argument] = _with_carry_video(target, carry)


class GuideContinuity:
    """Lay the previous segment's tail into the next as a frame-0 H3 guide.

    The next segment renders its first guide_frames frames over the carried
    ones and goes on from there, so motion, camera and (with carry_audio, the
    held soundtrack) voice continue through the seam; those frames are the
    previous segment's again, and are trimmed at assembly. On fl2va the
    keyframe is the guide's first frame - the previous segment's frame
    -guide_frames - so the keyframe and the guide agree.
    """

    def __init__(self, config):
        self.config = config

    def extract(self, artifact):
        count = self.config.guide_frames
        # With backoff the held span ends one block early; the segment's tail
        # past it is dropped at assembly, so the join lands inside the copy
        back = self.config.guide_backoff
        all_frames = frames_as_pil_list(artifact)
        end = len(all_frames) - back
        if end < count:
            raise ValueError(
                f"A guide chain with guide_backoff {back} needs segments of at "
                f"least {count + back} frames, and this one made {len(all_frames)}"
            )
        frames = all_frames[end - count : end]
        if not self.config.carry_audio:
            return _SegmentCarry(frames, None, None)

        audio, sample_rate = _generated_audio(artifact)
        if audio is None:
            raise ValueError(
                "A guide chain with carry_audio holds the previous segment's "
                "audio, and this segment generated none - set carry_audio false"
            )
        if self.config.fps is None:
            raise ValueError(
                "Holding a guide chain's audio needs the frame rate - set 'fps' "
                "on the chain or a 'frame_rate' pipeline argument"
            )
        samples = frames_to_samples(count, self.config.fps, sample_rate)
        stop = audio.shape[1] - frames_to_samples(back, self.config.fps, sample_rate)
        return _SegmentCarry(
            frames, audio[:, max(stop - samples, 0) : stop], sample_rate
        )

    def inject(self, arguments, carry, segment_argument, keystone=None):
        if carry.latents is not None:
            # The experimental latent carry (#1150): the tail's own latents
            # stand in for encoding its frames and audio; the frames stay for
            # the length check and the keyframe below
            guide = {"video": carry.frames, "frame": 0, "latents": carry.latents}
            if carry.audio_latents is not None:
                guide["audio_latents"] = carry.audio_latents
            self._lay_in(arguments, guide, carry, segment_argument, keystone)
            return
        if carry.audio is not None:
            video = AudioVideo(
                carry.frames,
                torch.from_numpy(numpy.ascontiguousarray(carry.audio)),
                carry.sample_rate,
                fps=self.config.fps,
            )
        else:
            video = carry.frames
        guide = {"video": video, "frame": 0, "audio": carry.audio is not None}
        self._lay_in(arguments, guide, carry, segment_argument, keystone)

    @staticmethod
    def _lay_in(arguments, guide, carry, segment_argument, keystone):
        # A copy: iteration arguments share nested values
        arguments[GUIDES_INPUT] = list(arguments.get(GUIDES_INPUT) or []) + [guide]
        # fl2va's keyframe - t2va passes none, and gets none
        if arguments.get(segment_argument) is not None:
            # A keystone (#1187) is segment 1's frame 0 in place of the guide's
            arguments[segment_argument] = (
                keystone if keystone is not None else carry.frames[0]
            )


# The names are h3_rules.CHAIN_CONTINUITY_MODES, which validation reads too;
# strict, so a class added here without its name there fails at import
CONTINUITY_MODES = dict(
    zip(
        CHAIN_CONTINUITY_MODES,
        (LastFrameContinuity, LastSegmentContinuity, GuideContinuity),
        strict=True,
    )
)


@dataclass
class _SegmentCarry:
    """The part of a finished segment a last_segment or guide chain
    conditions on."""

    frames: list
    audio: object  # (channels, samples) numpy, or None
    sample_rate: int
    # The experimental latent carry (#1150): the frames' own latents and
    # audio latents rows, or None (h3_guides.latent_guide_tail)
    latents: object = None
    audio_latents: object = None


def _with_carry_reference(references, carry):
    """A copy of a references list with the carry frame appended as an image
    reference of the same type the workflow already uses."""
    image_reference = next(
        (
            reference
            for reference in references
            if getattr(reference, "kind", None) == "image"
        ),
        None,
    )
    if image_reference is None:
        raise ValueError(
            "Cannot carry a frame into a references list that has no image "
            "reference to model the new one on"
        )
    return list(references) + [type(image_reference)(image=carry)]


def _with_carry_video(references, carry):
    """A copy of a references list with the carry segment appended as a video
    reference of the same family the workflow already uses."""
    reference_type = _video_reference_type(references)
    arguments = {"frames": carry.frames}
    if carry.audio is not None:
        arguments["audio"] = torch.from_numpy(carry.audio)
        arguments["sample_rate"] = carry.sample_rate
    return list(references) + [reference_type(**arguments)]


def _video_reference_type(references):
    """The video reference class of the family the workflow's references come from.

    A workflow that already passes a video reference names the class outright.
    Otherwise it is the video-kind class living beside the references it does
    pass - the chain never imports a pipeline's reference types itself, the way
    _with_carry_reference models its carry on the list it was given.
    """
    if not references:
        raise ValueError(
            "Cannot carry a segment into an empty references list - a "
            "last_segment chain needs the workflow's own references to model "
            "the carry on"
        )

    for reference in references:
        if getattr(reference, "kind", None) == "video":
            return type(reference)

    module = sys.modules.get(type(references[0]).__module__, None)
    for candidate in vars(module).values() if module else ():
        if isinstance(candidate, type) and getattr(candidate, "kind", None) == "video":
            return candidate

    raise ValueError(
        f"Cannot carry a segment as a video reference - no video reference type "
        f"found alongside {type(references[0]).__name__}"
    )


def _generated_audio(artifact):
    """The audio generated with a segment, as (channels, samples) numpy."""
    if isinstance(artifact, AudioVideo) and artifact.audio is not None:
        return as_channels_samples(artifact.audio), artifact.sample_rate
    return None, None
