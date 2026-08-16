"""Segment-chained pipeline execution - videos of arbitrary length from
pipelines that generate short clips.

A "chain" block on a pipeline step runs the pipeline once per segment,
carries visual continuity from each segment into the next (the last frame
becomes the next segment's keyframe), trims the duplicated boundary frames,
and stitches the segments' frames and audio into one video.

Two ways to specify the length:
- segments: N - run the pipeline N times as configured
- match_audio: true - derive the total frame count from the audio reference
  in the step's arguments, slice that audio into frame-aligned per-segment
  chunks, and mux the final video with the original, unsliced track - so the
  soundtrack has no seams at all

The chain runs inside one cartesian iteration, so it composes with
previous_result fan-out: three keyframes in, three chained videos out.
"""

import gc
import logging
import math
from dataclasses import dataclass

import torch

from .. import empty_device_cache
from ..result import AudioVideo, get_artifact_list
from ..tasks.audio_utils import (
    as_channels_samples,
    equal_power_crossfade_join,
    frames_to_samples,
    slice_samples,
)
from ..tasks.video_utils import extract_frame, frames_as_pil_list

logger = logging.getLogger("dw")

# A runaway segment count is a configuration error - kept in the spirit of
# previous_results.MAX_ITERATIONS
MAX_SEGMENTS = 1000


class LastFrameContinuity:
    """Carry the last frame of each segment into the next as its keyframe."""

    def extract(self, artifact):
        return extract_frame(artifact, -1)

    def inject(self, arguments, carry, segment_argument):
        target = arguments.get(segment_argument)
        if isinstance(target, list):
            # A references list - the carry frame is appended as an image
            # reference alongside the workflow's own references
            arguments[segment_argument] = _with_carry_reference(target, carry)
        else:
            arguments[segment_argument] = carry


# Later modes - conditioning on the tail frames of the previous segment, or a
# full video reference carrying frames and audio - register here
CONTINUITY_MODES = {"last_frame": LastFrameContinuity}


@dataclass
class Segment:
    """One planned pipeline invocation of a chain."""

    index: int
    num_frames: int  # frames this segment generates; None - use the step's own
    audio_start_frame: int  # generation-timeline frame its audio slice starts at
    head_trim: int  # frames dropped from its head on the output timeline


def run_chain(pipeline, chain_definition, arguments):
    """Run a pipeline's chain and stitch the segments into one video.

    Args:
        pipeline: The loaded Pipeline wrapper - each segment goes through its
            _run_once, so prompt handling matches an unchained run
        chain_definition: The step's "chain" block
        arguments: Fully resolved arguments for one iteration of the step

    Returns:
        A single AudioVideo holding the stitched frames, and either the joined
        generated audio, the original match_audio track, or no audio at all
    """
    config = ChainConfig(chain_definition, arguments)
    continuity = CONTINUITY_MODES[config.continuity]()

    frames = []  # PIL frames on the output timeline
    audio = None  # joined generated audio, (channels, samples) float32
    audio_rate = None
    carry = None

    for segment in config.plan:
        segment_arguments = dict(arguments)

        if config.prompts:
            segment_arguments["prompt"] = config.prompts[
                min(segment.index, len(config.prompts) - 1)
            ]

        if config.source_audio is not None:
            segment_arguments["num_frames"] = segment.num_frames
            segment_arguments["references"] = _sliced_references(
                config, segment, arguments["references"]
            )

        if segment.index > 0:
            continuity.inject(segment_arguments, carry, config.segment_argument)

        logger.info(
            f"Chain segment {segment.index + 1}/{len(config.plan)}"
            + (f": {segment.num_frames} frames" if segment.num_frames else "")
        )

        output = pipeline._run_once(segment_arguments)
        artifact = _single_artifact(output)

        carry = continuity.extract(artifact)
        segment_frames = frames_as_pil_list(artifact)
        segment_audio, segment_rate = _generated_audio(artifact)

        frames.extend(segment_frames[segment.head_trim :])

        if config.source_audio is None and segment_audio is not None:
            audio, audio_rate = _joined_audio(
                audio, audio_rate, segment_audio, segment_rate, segment, config
            )

        # The segment's raw output is finished with - the frames live on as
        # PIL images and the carry frame is extracted. Free it before the next
        # segment needs the accelerator.
        del output, artifact, segment_frames, segment_audio
        gc.collect()
        empty_device_cache()

    if config.source_audio is not None:
        # The video matches the track's duration; the original, unsliced audio
        # is muxed in so the soundtrack has no seams
        frames = frames[: config.total_frames]
        return AudioVideo(frames, config.source_audio, config.source_rate)

    return AudioVideo(frames, audio, audio_rate)


class ChainConfig:
    """Validated chain settings plus the planned segments for one run."""

    def __init__(self, chain_definition, arguments):
        segments = chain_definition.get("segments", None)
        match_audio = bool(chain_definition.get("match_audio", False))
        if (segments is not None) == match_audio:
            raise ValueError("A chain needs exactly one of 'segments' or 'match_audio'")

        self.continuity = chain_definition.get("continuity", "last_frame")
        if self.continuity not in CONTINUITY_MODES:
            known = ", ".join(sorted(CONTINUITY_MODES))
            raise ValueError(
                f"Unknown chain continuity '{self.continuity}' - expected one of {known}"
            )

        self.segment_argument = chain_definition.get("segment_argument", "image")
        self.trim_frames = int(chain_definition.get("trim_frames", 1))
        self.crossfade_ms = float(chain_definition.get("crossfade_ms", 75))
        self.prompts = chain_definition.get("prompts", None)
        self.fps = _resolve_fps(chain_definition, arguments)
        self.frame_snap = chain_definition.get("frame_snap", None)

        self.source_audio = None
        self.source_rate = None
        self.audio_reference = None
        self.total_frames = None

        if match_audio:
            self._plan_from_audio(arguments)
        else:
            segments = int(segments)
            if not 1 <= segments <= MAX_SEGMENTS:
                raise ValueError(
                    f"Chain 'segments' must be between 1 and {MAX_SEGMENTS}, got {segments}"
                )
            num_frames = arguments.get("num_frames", None)
            if num_frames is not None:
                validate_frame_snap(int(num_frames), self.frame_snap)
            self.plan = [
                Segment(
                    index,
                    int(num_frames) if num_frames is not None else None,
                    0,
                    self.trim_frames if index > 0 else 0,
                )
                for index in range(segments)
            ]

    def _plan_from_audio(self, arguments):
        """Derive the segment plan from the audio reference's duration."""
        if self.fps is None:
            raise ValueError(
                "A match_audio chain needs the frame rate - set 'fps' on the "
                "chain or a 'frame_rate' pipeline argument"
            )

        num_frames = arguments.get("num_frames", None)
        if num_frames is None:
            raise ValueError(
                "A match_audio chain needs 'num_frames' in the step's arguments "
                "as the per-segment length"
            )

        reference = _find_audio_reference(arguments)
        self.audio_reference = reference
        self.source_audio = as_channels_samples(reference.audio)
        self.source_rate = reference.sample_rate
        if self.source_rate is None:
            raise ValueError("The chain's audio reference has no sample rate")

        total_samples = self.source_audio.shape[1]
        self.total_frames = max(1, round(total_samples / self.source_rate * self.fps))
        self.plan = plan_segments(
            self.total_frames, int(num_frames), self.trim_frames, self.frame_snap
        )
        duration = total_samples / self.source_rate
        logger.info(
            f"Chaining to match {duration:.2f}s of audio: {self.total_frames} "
            f"frames across {len(self.plan)} segments"
        )


def plan_segments(total_frames, segment_frames, trim_frames, frame_snap=None):
    """Plan the segments that cover a total frame count.

    Every segment generates segment_frames frames except possibly the last,
    which shrinks to what remains - snapped up to a count the pipeline accepts.
    Each segment after the first has trim_frames dropped from its head, so it
    contributes segment_frames - trim_frames new frames to the output.

    Args:
        total_frames: Frames the stitched output must cover
        segment_frames: Frames a full segment generates
        trim_frames: Head frames dropped from every segment after the first
        frame_snap: Optional dict with modulus/remainder and min/max_frames
            describing the counts the pipeline accepts

    Returns:
        List of Segment
    """
    if segment_frames <= trim_frames:
        raise ValueError(
            f"Segments of {segment_frames} frames cannot progress past a head "
            f"trim of {trim_frames} frames"
        )
    validate_frame_snap(segment_frames, frame_snap)

    plan = []
    covered = 0
    while covered < total_frames:
        if len(plan) >= MAX_SEGMENTS:
            raise ValueError(f"Chain would exceed {MAX_SEGMENTS} segments")

        head_trim = trim_frames if plan else 0
        needed = (total_frames - covered) + head_trim
        if needed >= segment_frames:
            num_frames = segment_frames
        else:
            # The last segment generates only what remains, snapped up to a
            # count the pipeline accepts; the overshoot is trimmed at the end
            num_frames = snap_frames(needed, frame_snap)

        plan.append(Segment(len(plan), num_frames, covered - head_trim, head_trim))
        covered += num_frames - head_trim

    return plan


def snap_frames(count, frame_snap):
    """The smallest frame count the pipeline accepts that covers count."""
    if not frame_snap:
        return count

    modulus = frame_snap["modulus"]
    remainder = frame_snap["remainder"]
    target = max(count, frame_snap.get("min_frames", 1))

    steps = max(0, math.ceil((target - remainder) / modulus))
    snapped = steps * modulus + remainder
    while snapped < target:
        snapped += modulus

    max_frames = frame_snap.get("max_frames", None)
    if max_frames is not None and snapped > max_frames:
        raise ValueError(
            f"Cannot snap {count} frames into the pipeline's accepted range - "
            f"the next valid count {snapped} exceeds max_frames {max_frames}"
        )
    return snapped


def validate_frame_snap(num_frames, frame_snap):
    """Check a configured num_frames against the pipeline's constraint."""
    if not frame_snap:
        return

    modulus = frame_snap["modulus"]
    remainder = frame_snap["remainder"]
    problems = []
    if (num_frames - remainder) % modulus != 0:
        problems.append(f"counts must be {modulus}*n+{remainder}")
    min_frames = frame_snap.get("min_frames", None)
    if min_frames is not None and num_frames < min_frames:
        problems.append(f"at least {min_frames}")
    max_frames = frame_snap.get("max_frames", None)
    if max_frames is not None and num_frames > max_frames:
        problems.append(f"at most {max_frames}")

    if problems:
        raise ValueError(
            f"num_frames {num_frames} does not satisfy the pipeline's frame "
            f"constraint: {'; '.join(problems)}"
        )


def _resolve_fps(chain_definition, arguments):
    """The frame rate used for audio math - explicit, or the pipeline's own."""
    fps = chain_definition.get("fps", arguments.get("frame_rate", None))
    return float(fps) if fps is not None else None


def _find_audio_reference(arguments):
    """The single audio reference a match_audio chain slices per segment."""
    references = arguments.get("references", None)
    if not isinstance(references, list):
        raise ValueError(
            "A match_audio chain needs a 'references' argument holding the "
            "audio reference to match"
        )

    audio_references = [
        reference
        for reference in references
        if getattr(reference, "kind", None) == "audio"
    ]
    if len(audio_references) != 1:
        raise ValueError(
            f"A match_audio chain needs exactly one audio reference, "
            f"found {len(audio_references)}"
        )
    return audio_references[0]


def _sliced_references(config, segment, references):
    """A copy of the references list with the segment's audio slice swapped in.

    The original list and reference objects are never touched - iteration
    arguments share nested values, so they must not be mutated in place.
    """
    start = frames_to_samples(segment.audio_start_frame, config.fps, config.source_rate)
    length = frames_to_samples(segment.num_frames, config.fps, config.source_rate)
    piece = slice_samples(config.source_audio, start, length)

    sliced = type(config.audio_reference)(
        audio=torch.from_numpy(piece), sample_rate=config.source_rate
    )
    return [
        sliced if reference is config.audio_reference else reference
        for reference in references
    ]


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


def _single_artifact(output):
    """The one video artifact a chain segment must produce.

    Modular pipelines asked for extra outputs return them alongside the video -
    those are dropped here. More than one video means batched generation, which
    a chain cannot stitch.
    """
    artifacts = get_artifact_list(output)
    videos = [artifact for artifact in artifacts if _is_video_artifact(artifact)]

    if len(videos) != 1:
        raise ValueError(
            f"A chained pipeline must generate exactly one video per segment, "
            f"got {len(videos)} - batched generation cannot be chained"
        )
    if len(artifacts) > 1:
        logger.debug(f"Chain segment dropped {len(artifacts) - 1} non-video output(s)")
    return videos[0]


def _is_video_artifact(artifact):
    if isinstance(artifact, AudioVideo):
        return True
    if isinstance(artifact, list) and artifact:
        return not isinstance(artifact[0], str)
    return hasattr(artifact, "ndim") and artifact.ndim >= 3


def _generated_audio(artifact):
    """The audio generated with a segment, as (channels, samples) numpy."""
    if isinstance(artifact, AudioVideo) and artifact.audio is not None:
        return as_channels_samples(artifact.audio), artifact.sample_rate
    return None, None


def _joined_audio(audio, audio_rate, segment_audio, segment_rate, segment, config):
    """Fold one segment's generated audio into the accumulated track.

    The samples matching the segment's trimmed head frames are cut off and
    used as crossfade material against the tail of the accumulated audio, so
    the audio timeline shortens by exactly as much as the video's.
    """
    if audio is None:
        return segment_audio, segment_rate

    if segment_rate != audio_rate:
        raise ValueError(
            f"Chain segments generated audio at different sample rates: "
            f"{audio_rate} then {segment_rate}"
        )

    if segment.head_trim > 0 and config.fps is None:
        raise ValueError(
            "Joining generated audio needs the frame rate - set 'fps' on the "
            "chain or a 'frame_rate' pipeline argument"
        )

    trim_samples = (
        frames_to_samples(segment.head_trim, config.fps, audio_rate)
        if segment.head_trim
        else 0
    )
    head = segment_audio[:, :trim_samples]
    body = segment_audio[:, trim_samples:]
    return (
        equal_power_crossfade_join(audio, head, body, audio_rate, config.crossfade_ms),
        audio_rate,
    )
