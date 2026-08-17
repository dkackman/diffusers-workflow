"""Segment-chained pipeline execution - videos of arbitrary length from
pipelines that generate short clips.

A "chain" block on a pipeline step runs the pipeline once per segment,
carries continuity from each segment into the next, trims the duplicated
boundary frames, and stitches the segments' frames and audio into one video.

Two ways to carry continuity:
- last_frame - the last frame becomes the next segment's keyframe, which is
  what a keyframe-conditioned pipeline takes
- last_segment - the previous segment's frames and the soundtrack generated
  with them become a video reference, which carries motion, camera and voice
  across the seam rather than appearance alone

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
import os
import sys
from dataclasses import dataclass

import numpy
import torch
from diffusers.utils import encode_video, is_av_available

from .. import empty_device_cache
from ..result import AudioVideo, get_artifact_list
from ..security import validate_output_path
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

    def __init__(self, config):
        self.config = config

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


CONTINUITY_MODES = {
    "last_frame": LastFrameContinuity,
    "last_segment": LastSegmentContinuity,
}


@dataclass
class _SegmentCarry:
    """The part of a finished segment a last_segment chain conditions on."""

    frames: list
    audio: object  # (channels, samples) numpy, or None
    sample_rate: int


@dataclass
class Segment:
    """One planned pipeline invocation of a chain."""

    index: int
    num_frames: int  # frames this segment generates; None - use the step's own
    audio_start_frame: int  # generation-timeline frame its audio slice starts at
    head_trim: int  # frames dropped from its head on the output timeline


class SegmentedFrames:
    """Chained segments spilled to disk, replayed at save time.

    Iterating yields one uint8 (frames, height, width, 3) torch tensor per
    segment file - the chunk shape encode_video streams from - so the final
    video is written holding only one segment in memory at a time.
    """

    def __init__(self, paths, total_frames=None, keep_files=False):
        """
        Args:
            paths: The segment files, in output order
            total_frames: Frames to yield in total - the match_audio tail trim.
                None yields every stored frame
            keep_files: Leave the segment files in place after cleanup()
        """
        self.paths = list(paths)
        self.total_frames = total_frames
        self.keep_files = keep_files

    def __len__(self):
        """Chunk count - one per segment file."""
        return len(self.paths)

    def __iter__(self):
        remaining = self.total_frames
        for path in self.paths:
            frames = _decode_segment(path)
            if remaining is not None:
                frames = frames[:remaining]
                remaining -= len(frames)
            if len(frames):
                yield frames
            if remaining == 0:
                return

    def cleanup(self):
        """Remove the segment files once the final video is safely written."""
        if self.keep_files:
            return
        for path in self.paths:
            try:
                os.remove(path)
            except FileNotFoundError:
                pass


class SegmentSpill:
    """Writes each completed segment to disk as a playable mp4.

    Files are named {prefix}.{iteration}.segment-{index:03d}.mp4 in the
    workflow's output directory - a crashed chain leaves them behind, ready to
    salvage with gather_videos + concat_videos.
    """

    def __init__(self, pipeline, config):
        output_dir = getattr(pipeline, "output_dir", None)
        file_prefix = getattr(pipeline, "file_prefix", None)
        if not output_dir or not file_prefix:
            raise ValueError(
                "save_segments needs the workflow's output directory - it is "
                "only available when the chain runs through a workflow"
            )
        if not is_av_available():
            raise ValueError(
                "save_segments writes mp4 segment files with PyAV - install "
                "it with: pip install av"
            )
        if config.fps is None:
            raise ValueError(
                "save_segments needs the frame rate to encode segment files - "
                "set 'fps' on the chain or a 'frame_rate' pipeline argument"
            )

        # The same step can chain more than once (previous_result fan-out) -
        # a per-wrapper counter keeps each iteration's files apart
        iteration = getattr(pipeline, "_chain_iteration", -1) + 1
        pipeline._chain_iteration = iteration

        self.output_dir = output_dir
        self.base_name = f"{file_prefix}.{iteration}"
        self.fps = config.fps
        self.paths = []

    def write(self, frames, audio, sample_rate):
        """Encode one trimmed segment to disk and record its path.

        Args:
            frames: The segment's on-timeline PIL frames
            audio: The segment's on-timeline generated audio as
                (channels, samples) numpy, or None - muxed in so a crashed
                chain leaves fully playable segments
            sample_rate: Sample rate of that audio
        """
        path = validate_output_path(
            os.path.join(
                self.output_dir,
                f"{self.base_name}.segment-{len(self.paths):03d}.mp4",
            ),
            self.output_dir,
        )

        audio_track = None
        if audio is not None and audio.shape[1] and sample_rate is not None:
            audio_track = torch.from_numpy(numpy.ascontiguousarray(audio))

        encode_video(
            frames,
            fps=self.fps,
            output_path=path,
            audio=audio_track,
            audio_sample_rate=sample_rate if audio_track is not None else None,
        )
        self.paths.append(path)
        logger.info(f"Saved chain segment to {path}")


def _decode_segment(path):
    """Read a segment file back as a uint8 (frames, height, width, 3) tensor."""
    import av

    with av.open(path) as container:
        frames = [
            frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)
        ]
    return torch.from_numpy(numpy.stack(frames, axis=0))


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
    continuity = CONTINUITY_MODES[config.continuity](config)

    # With save_segments, each completed segment is written to disk and its
    # frames freed, bounding memory to one segment - a crash leaves the
    # finished segments behind as playable files
    spill = SegmentSpill(pipeline, config) if config.save_segments else None

    frames = []  # PIL frames on the output timeline (unspilled chains)
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

        kept_frames = segment_frames[segment.head_trim :]
        if spill is not None:
            spill.write(
                kept_frames,
                _on_timeline_audio(segment_audio, segment, config, segment_rate),
                segment_rate,
            )
        else:
            frames.extend(kept_frames)

        if config.source_audio is None and segment_audio is not None:
            audio, audio_rate = _joined_audio(
                audio, audio_rate, segment_audio, segment_rate, segment, config
            )

        # The segment's raw output is finished with - the frames live on
        # (in RAM or on disk) and the carry frame is extracted. Free it
        # before the next segment needs the accelerator.
        del output, artifact, segment_frames, segment_audio, kept_frames
        gc.collect()
        empty_device_cache()

    if spill is not None:
        # match_audio overshoots by design - the tail trim happens as the
        # lazy frames replay, so the files themselves stay whole
        frames = SegmentedFrames(
            spill.paths,
            config.total_frames if config.source_audio is not None else None,
            config.keep_segments,
        )

    if config.source_audio is not None:
        # The video matches the track's duration; the original, unsliced audio
        # is muxed in so the soundtrack has no seams
        if spill is None:
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
        self.carry_frames = chain_definition.get("carry_frames", None)
        if self.carry_frames is not None:
            self.carry_frames = int(self.carry_frames)
            if self.carry_frames < 1:
                raise ValueError(
                    f"Chain 'carry_frames' must be at least 1, got {self.carry_frames}"
                )
        self.carry_audio = bool(chain_definition.get("carry_audio", True))
        self.trim_frames = int(chain_definition.get("trim_frames", 1))
        self.crossfade_ms = float(chain_definition.get("crossfade_ms", 75))
        self.prompts = chain_definition.get("prompts", None)
        self.fps = _resolve_fps(chain_definition, arguments)
        self.frame_snap = chain_definition.get("frame_snap", None)
        self.save_segments = bool(chain_definition.get("save_segments", False))
        self.keep_segments = bool(chain_definition.get("keep_segments", False))

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


def _on_timeline_audio(segment_audio, segment, config, segment_rate):
    """The part of a segment's generated audio that survives the head trim.

    Muxed into the segment's spill file so a crashed chain leaves fully
    playable segments; the final soundtrack still comes from the accumulated
    crossfaded track (or the original match_audio track).
    """
    if segment_audio is None:
        return None
    trim_samples = frames_to_samples(segment.head_trim, config.fps, segment_rate)
    return segment_audio[:, trim_samples:]


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
