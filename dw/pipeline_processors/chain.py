"""Segment-chained pipeline execution - videos of arbitrary length from
pipelines that generate short clips.

A "chain" block on a pipeline step runs the pipeline once per segment,
carries continuity from each segment into the next, trims the duplicated
boundary frames, and stitches the segments' frames and audio into one video.

Three ways to carry continuity:
- last_frame - the last frame becomes the next segment's keyframe, which is
  what a keyframe-conditioned pipeline takes
- last_segment - the previous segment's frames and the soundtrack generated
  with them become a video reference, which carries motion, camera and voice
  across the seam rather than appearance alone
- guide - the previous segment's last guide_frames frames (and, with
  carry_audio, their audio) are laid in at frame 0 of the next as an H3 guide,
  and trimmed off it again, so the next segment continues the motion itself.
  MiniMax-H3 t2va and fl2va only: guides stay off ref2va

Two ways to specify the length:
- segments: N - run the pipeline N times as configured
- match_audio: true - derive the total frame count from the step's
  `hold_audio` track, or else its one audio reference, slice that audio into
  frame-aligned per-segment chunks, and mux the final video with the original,
  unsliced track - so the soundtrack has no seams at all. A held track is
  sliced into each segment's `hold_audio`, so every segment generates to its
  own piece of it (dw/pipeline_processors/h3_blocks.py)

The chain runs inside one cartesian iteration, so it composes with
previous_result fan-out: three keyframes in, three chained videos out.
"""

import gc
import logging
import math
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass

import numpy
import torch
from PIL import Image
from diffusers.utils import encode_video, is_av_available

from .. import empty_device_cache
from ..media_types import AudioVideo, fit_codec_padding
from ..output_extraction import get_artifact_list
from ..writers import frames_for_encoding, output_file_path
from ..events import emit_log
from ..shots import shot_record, without_samples
from ..dsp import as_channels_samples, slice_samples
from ..task_domains import frames_to_samples
from ..tasks.joins import equal_power_crossfade_join
from ..tasks.video_utils import extract_frame, frames_as_pil_list
from .h3_blocks import (
    CHAIN_CONTINUITY_MODES,
    GUIDE_CHAIN_DEFAULT,
    GUIDE_CHAIN_RULE,
    GUIDE_CONTINUITY,
    GUIDES_INPUT,
    HOLD_AUDIO_INPUT,
    guide_chain_problems,
    guides_refusal,
    hold_audio_reference,
)

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
        frames = frames_as_pil_list(artifact)[-count:]
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
        return _SegmentCarry(frames, audio[:, -samples:], sample_rate)

    def inject(self, arguments, carry, segment_argument):
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
        # A copy: iteration arguments share nested values
        arguments[GUIDES_INPUT] = list(arguments.get(GUIDES_INPUT) or []) + [guide]
        # fl2va's keyframe - t2va passes none, and gets none
        if arguments.get(segment_argument) is not None:
            arguments[segment_argument] = carry.frames[0]


# The names are h3_blocks.CHAIN_CONTINUITY_MODES, which validation reads too;
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

    def __init__(self, paths, total_frames=None, keep_files=False, stored_frames=None):
        """
        Args:
            paths: The segment files, in output order
            total_frames: Frames to yield in total - the match_audio tail trim.
                None yields every stored frame
            keep_files: Leave the segment files in place after cleanup()
            stored_frames: Frames across all the files, when known - what
                frame_count reports, since len() counts files
        """
        self.paths = list(paths)
        self.stored_frames = stored_frames
        self.total_frames = total_frames
        self.keep_files = keep_files
        # True once cleanup() has actually removed the files (not when
        # keep_files skipped that) - a cached Result pointing at a cleaned-up
        # SegmentedFrames must not be served to a later miss, since the files
        # a replay would open are gone. See Result.retainable.
        self.cleaned = False

    def __len__(self):
        """Chunk count - one per segment file."""
        return len(self.paths)

    @property
    def frame_count(self):
        """Frames the replay yields, or None when the files were not counted."""
        if self.stored_frames is None:
            return None
        if self.total_frames is None:
            return self.stored_frames
        return min(self.stored_frames, self.total_frames)

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
        self.cleaned = True


class SavedFrames(Sequence):
    """The frames of a chain's finished video, read back from the file it saved.

    Once SegmentedFrames.cleanup() has removed the segment files, a later step
    that names the chain's result (previous_result:) gets these instead: a
    frame sequence - length is the frame count - decoded from the final file
    the first time a frame is asked for, so a chain nobody consumes never pays
    for the decode.
    """

    def __init__(self, path, count=None, cleaned=True):
        self.path = path
        self._count = count
        self._frames = None
        # Result.retainable: the segment files this replaced are gone, and a
        # cache entry must not outlive the file this one reads
        self.cleaned = cleaned

    def _load(self):
        if self._frames is None:
            from ..media import decode_rgb_frames

            self._frames = [Image.fromarray(f) for f in decode_rgb_frames(self.path)]
            self._count = len(self._frames)
        return self._frames

    def __len__(self):
        if self._count is None:
            self._load()
        return self._count

    def __getitem__(self, index):
        return self._load()[index]


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
        self.frame_count = 0

    def write(self, frames, audio, sample_rate):
        """Encode one trimmed segment to disk and record its path.

        Args:
            frames: The segment's on-timeline PIL frames
            audio: The segment's on-timeline generated audio as
                (channels, samples) numpy, or None - muxed in so a crashed
                chain leaves fully playable segments
            sample_rate: Sample rate of that audio
        """
        # A crashed or completed chain leaves segment files behind on disk,
        # ready to salvage - and the per-wrapper iteration counter (base_name)
        # restarts at 0 in a fresh process, so a rerun must dedupe through the
        # same '-N' convention every other output goes through rather than
        # silently overwriting them.
        path = output_file_path(
            self.output_dir,
            f"{self.base_name}.segment-{len(self.paths):03d}.mp4",
        )

        audio_track = None
        if audio is not None and audio.shape[1] and sample_rate is not None:
            audio_track = torch.from_numpy(numpy.ascontiguousarray(audio))

        encode_video(
            frames_for_encoding(frames),
            fps=self.fps,
            output_path=path,
            audio=audio_track,
            audio_sample_rate=sample_rate if audio_track is not None else None,
        )
        self.paths.append(path)
        self.frame_count += len(frames)
        logger.info(f"Saved chain segment to {path}")


def _decode_segment(path):
    """Read a segment file back as a uint8 (frames, height, width, 3) tensor."""
    from ..media import decode_rgb_frames

    return torch.from_numpy(numpy.stack(decode_rgb_frames(path), axis=0))


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
    # Step.run resolved any previous_result references in the chain's prompts,
    # which the chain block cannot express on its own
    config = ChainConfig(
        chain_definition,
        arguments,
        getattr(pipeline, "chain_prompts", None),
        getattr(pipeline, "base_dir", None),
    )
    continuity = CONTINUITY_MODES[config.continuity](config)
    if config.continuity == GUIDE_CONTINUITY:
        _check_guide_chain(pipeline, arguments, chain_definition, config)

    # With save_segments, each completed segment is written to disk and its
    # frames freed, bounding memory to one segment - a crash leaves the
    # finished segments behind as playable files
    spill = SegmentSpill(pipeline, config) if config.save_segments else None

    frames = []  # PIL frames on the output timeline (unspilled chains)
    audio = None  # joined generated audio, (channels, samples) float32
    audio_rate = None
    carry = None
    # One shot per segment, measured as the picture and track grow (#378);
    # the frames are counted rather than read off `frames`, which a spilled
    # chain never fills
    shots = []
    frame_count = 0

    try:
        for segment in config.plan:
            segment_arguments = dict(arguments)

            if config.prompts:
                segment_arguments["prompt"] = config.prompts[
                    min(segment.index, len(config.prompts) - 1)
                ]

            if config.source_audio is not None:
                segment_arguments["num_frames"] = segment.num_frames
                if config.holds_audio:
                    segment_arguments[HOLD_AUDIO_INPUT] = _audio_slice(config, segment)
                else:
                    segment_arguments["references"] = _sliced_references(
                        config, segment, arguments["references"]
                    )

            if segment.index > 0:
                continuity.inject(segment_arguments, carry, config.segment_argument)

            logger.info(
                f"Chain segment {segment.index + 1}/{len(config.plan)}"
                + (f": {segment.num_frames} frames" if segment.num_frames else "")
            )

            # The denoise counter restarts for every segment - without this the
            # bar rewinds to zero with nothing saying why
            pipeline.segment_label = f"segment {segment.index + 1}/{len(config.plan)}"
            output = pipeline._run_once(segment_arguments)
            artifact = _single_artifact(output)

            carry = continuity.extract(artifact)
            segment_frames = frames_as_pil_list(artifact)
            segment_audio, segment_rate = _generated_audio(artifact)
            # A segment's own generated audio can run a codec-padding sliver
            # short of the frames it was asked for - the same gap #197 fixed for
            # a decoded file (_decode_audio_video) and for an in-memory
            # previous_result shot (Result.save). A chained segment goes through
            # neither of those, so the shortfall was surviving here uncorrected
            # and compounding once per segment (#408).
            if segment_audio is not None and segment_rate and config.fps:
                segment_audio = fit_codec_padding(
                    segment_audio, len(segment_frames), config.fps, segment_rate
                )

            kept_frames = segment_frames[segment.head_trim :]
            start_sample = audio.shape[1] if audio is not None else 0
            shots.append(
                shot_record(
                    f"segment {segment.index + 1}",
                    frame_count,
                    len(kept_frames),
                    start_sample,
                )
            )
            frame_count += len(kept_frames)
            if spill is not None:
                spill.write(
                    kept_frames,
                    _on_timeline_audio(segment_audio, segment, config, segment_rate),
                    segment_rate,
                )
            else:
                frames.extend(kept_frames)

            if config.source_audio is None and segment_audio is not None:
                applied = {}
                audio, audio_rate = _joined_audio(
                    audio,
                    audio_rate,
                    segment_audio,
                    segment_rate,
                    segment,
                    config,
                    applied,
                )
                if segment.index > 0:
                    # The seam's own blend, so assess_output can tell it from a
                    # dropout in the content (#660)
                    shots[-1]["trim_frames"] = segment.head_trim
                    if "crossfade_ms" in applied:
                        shots[-1]["crossfade_ms"] = applied["crossfade_ms"]
                    emit_log(
                        f"Chain seam {segment.index}/{len(config.plan) - 1}: trimmed "
                        f"{segment.head_trim} head frame(s), crossfade "
                        + (
                            f"{applied['crossfade_ms']} ms"
                            if "crossfade_ms" in applied
                            else "none (the guide held the audio)"
                            if config.guide_holds_audio
                            else "none (no head material)"
                        )
                    )

            shots[-1]["num_samples"] = (
                audio.shape[1] if audio is not None else 0
            ) - start_sample

            # The segment's raw output is finished with - the frames live on
            # (in RAM or on disk) and the carry frame is extracted. Free it
            # before the next segment needs the accelerator.
            del output, artifact, segment_frames, segment_audio, kept_frames
            gc.collect()
            empty_device_cache()
    finally:
        # The label belongs to the run that set it - on a raise or a cancel
        # as much as on the way out
        pipeline.segment_label = None

    if spill is not None:
        # match_audio overshoots by design - the tail trim happens as the
        # lazy frames replay, so the files themselves stay whole
        frames = SegmentedFrames(
            spill.paths,
            config.total_frames if config.source_audio is not None else None,
            config.keep_segments,
            spill.frame_count,
        )

    if config.source_audio is not None:
        # The video matches the track's duration; the original, unsliced audio
        # is muxed in so the soundtrack has no seams
        if spill is None:
            frames = frames[: config.total_frames]
        return AudioVideo(
            frames,
            config.source_audio,
            config.source_rate,
            fps=config.fps,
            shots=_trimmed_shots(shots, config.total_frames),
        )

    if audio is None:
        shots = without_samples(shots)
    return AudioVideo(frames, audio, audio_rate, fps=config.fps, shots=shots)


def _check_guide_chain(pipeline, arguments, chain_definition, config):
    """Refuse a guide chain the H3 guides cannot take before the first segment
    renders, and note once in the job log the settings it does not read."""
    rule = GUIDE_CHAIN_RULE
    if arguments.get("references") is not None:
        raise ValueError(f"{rule}, and this step passes references")
    loaded = getattr(pipeline, "pipeline", None)
    if loaded is not None:
        refusal = guides_refusal(loaded)
        if refusal:
            raise ValueError(f"{rule}: {refusal}")
    if "trim_frames" in chain_definition:
        emit_log(
            f"Guide chain: trim_frames is ignored - each segment after the first "
            f"drops its {config.guide_frames} guide frames instead"
        )
    if config.guide_holds_audio and "crossfade_ms" in chain_definition:
        emit_log(
            "Guide chain: crossfade_ms is ignored - the guide holds the audio "
            "across each seam (set carry_audio false to crossfade instead)"
        )


def _trimmed_shots(shots, total_frames):
    """A match_audio chain's shots, cut where its overshooting picture is.

    The soundtrack is the caller's own, laid under whole rather than built
    segment by segment, so no shot has a stretch of it to measure: the sample
    side is cleared, not derived.
    """
    trimmed = []
    for shot in without_samples(shots):
        if shot["start_frame"] >= total_frames:
            break
        end = min(shot["start_frame"] + shot["num_frames"], total_frames)
        trimmed.append({**shot, "num_frames": end - shot["start_frame"]})
    return trimmed


class ChainConfig:
    """Validated chain settings plus the planned segments for one run."""

    def __init__(
        self, chain_definition, arguments, resolved_prompts=None, base_dir=None
    ):
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
        # A guide chain trims the guide it laid in, not trim_frames, and a
        # held soundtrack has nothing to crossfade
        self.guide_frames = None
        self.guide_holds_audio = False
        head_trim = self.trim_frames
        if self.continuity == GUIDE_CONTINUITY:
            problems = guide_chain_problems(chain_definition)
            if problems:
                raise ValueError(
                    "; ".join(f"Chain {message}" for _, message in problems)
                )
            self.guide_frames = chain_definition.get(
                "guide_frames", GUIDE_CHAIN_DEFAULT
            )
            head_trim = self.guide_frames
            if self.carry_audio:
                self.guide_holds_audio = True
                self.crossfade_ms = 0.0
        self.head_trim = head_trim
        self.prompts = resolved_prompts or chain_definition.get("prompts", None)
        self.fps = _resolve_fps(chain_definition, arguments)
        self.frame_snap = chain_definition.get("frame_snap", None)
        self.save_segments = bool(chain_definition.get("save_segments", False))
        self.keep_segments = bool(chain_definition.get("keep_segments", False))

        self.source_audio = None
        self.source_rate = None
        self.audio_reference = None
        self.holds_audio = False
        self.total_frames = None

        if match_audio:
            self._plan_from_audio(arguments, base_dir)
        else:
            segments = int(segments)
            if not 1 <= segments <= MAX_SEGMENTS:
                raise ValueError(
                    f"Chain 'segments' must be between 1 and {MAX_SEGMENTS}, got {segments}"
                )
            num_frames = arguments.get("num_frames", None)
            if num_frames is not None:
                validate_frame_snap(int(num_frames), self.frame_snap)
                if self.guide_frames is not None and int(num_frames) <= head_trim:
                    raise ValueError(
                        f"Segments of {num_frames} frames cannot progress past "
                        f"a {head_trim}-frame guide"
                    )
            self.plan = [
                Segment(
                    index,
                    int(num_frames) if num_frames is not None else None,
                    0,
                    head_trim if index > 0 else 0,
                )
                for index in range(segments)
            ]

    def _plan_from_audio(self, arguments, base_dir=None):
        """Derive the segment plan from the matched track's duration - the
        step's `hold_audio` when it names one, else its audio reference."""
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

        held = arguments.get(HOLD_AUDIO_INPUT)
        if held is not None:
            # Read once here, sliced per segment - each slice is a reference
            # already built, which the segment's own hold takes as it is
            reference = hold_audio_reference(held, base_dir)
            self.holds_audio = True
        else:
            reference = _find_audio_reference(arguments)
        self.audio_reference = reference
        self.source_audio = as_channels_samples(reference.audio)
        self.source_rate = reference.sample_rate
        if self.source_rate is None:
            raise ValueError("The chain's matched audio has no sample rate")

        total_samples = self.source_audio.shape[1]
        self.total_frames = max(1, round(total_samples / self.source_rate * self.fps))
        self.plan = plan_segments(
            self.total_frames, int(num_frames), self.head_trim, self.frame_snap
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
    """The single audio reference a match_audio chain with no `hold_audio`
    slices per segment."""
    references = arguments.get("references", None)
    if not isinstance(references, list):
        raise ValueError(
            "A match_audio chain needs a 'hold_audio' track, or a 'references' "
            "argument holding the audio reference to match"
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


def _audio_slice(config, segment):
    """The segment's piece of the matched track, as a new reference of the
    matched one's type - the original is never touched."""
    start = frames_to_samples(segment.audio_start_frame, config.fps, config.source_rate)
    length = frames_to_samples(segment.num_frames, config.fps, config.source_rate)
    piece = slice_samples(config.source_audio, start, length)
    return type(config.audio_reference)(
        audio=torch.from_numpy(piece), sample_rate=config.source_rate
    )


def _sliced_references(config, segment, references):
    """A copy of the references list with the segment's audio slice swapped in.

    The original list and reference objects are never touched - iteration
    arguments share nested values, so they must not be mutated in place.
    """
    sliced = _audio_slice(config, segment)
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


def _joined_audio(
    audio, audio_rate, segment_audio, segment_rate, segment, config, applied=None
):
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
        equal_power_crossfade_join(
            audio, head, body, audio_rate, config.crossfade_ms, applied=applied
        ),
        audio_rate,
    )
