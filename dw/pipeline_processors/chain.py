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

Three ways to specify the length:
- segments: N - run the pipeline N times as configured
- duration_seconds: S - plan the segments that cover S seconds at the chain's
  frame rate, then trim the picture and the generated audio to exactly S
- match_audio: true - derive the total frame count from the step's
  `hold_audio` track, or else its one audio reference, slice that audio into
  frame-aligned per-segment chunks, and mux the final video with the original,
  unsliced track - so the soundtrack has no seams at all. A held track is
  sliced into each segment's `hold_audio`, so every segment generates to its
  own piece of it (dw/pipeline_processors/h3_hold.py)

The chain runs inside one cartesian iteration, so it composes with
previous_result fan-out: three keyframes in, three chained videos out.
"""

import copy
import gc
import logging
import math
import os
import sys
from dataclasses import dataclass

import numpy
import torch

from .. import empty_device_cache
from ..media_types import AudioVideo, fit_codec_padding
from ..output_extraction import get_artifact_list
from ..events import emit_log, emit_warning
from ..shots import shot_record, without_samples
from ..dsp import as_channels_samples, slice_samples
from ..task_domains import frames_to_samples
from ..tasks.joins import equal_power_crossfade_join
from ..tasks.video_utils import extract_frame, frames_as_pil_list
from .chain_spill import (  # noqa: F401 - SegmentedFrames and SavedFrames re-exported
    SavedFrames,
    SegmentedFrames,
    SegmentSpill,
    spilled_frames,
)
from .chain_check import (
    best_attempt,
    check_segment,
    reroll_seed,
    segment_check,
    texture_samples,
)
from .chain_look import LookHold
from .chain_pick import sharpest_index
from .h3_anchor import anchor_refusal
from .h3_guides import guides_refusal
from .h3_hold import hold_audio_reference
from .chain_rules import (
    CHAIN_CAPSTONE,
    CHAIN_CAPSTONE_FROM_END,
    chain_capstone_problem,
    chain_check_problem,
    chain_look_hold_flags,
    chain_look_hold_problem,
)
from .h3_rules import (
    ANCHOR_NONE,
    ANCHOR_SEGMENT_1,
    CHAIN_CONTINUITY_MODES,
    GUIDES_INPUT,
    GUIDE_CHAIN_DEFAULT,
    GUIDE_CHAIN_RULE,
    GUIDE_CONTINUITY,
    HOLD_AUDIO_INPUT,
    LATENT_ANCHOR_INPUT,
    LATENT_ANCHOR_STRENGTH_INPUT,
    anchor_strength_problem,
    KEYSTONE_NONE,
    KEYSTONE_SEGMENT_1,
    chain_anchor_problem,
    chain_carry_pick_problem,
    chain_keystone_problem,
    guide_backoff_problems,
    guide_chain_problems,
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
    if config.anchor == ANCHOR_SEGMENT_1:
        _check_anchor_chain(pipeline, arguments)

    # With save_segments, each completed segment is written to disk and its
    # frames freed, bounding memory to one segment - a crash leaves the
    # finished segments behind as playable files
    spill = SegmentSpill(pipeline, config) if config.save_segments else None

    frames = []  # PIL frames on the output timeline (unspilled chains)
    audio = None  # joined generated audio, (channels, samples) float32
    audio_rate = None
    carry = None
    # The corrected frames' look-hold state, and the last carry before any
    # pre-compensation (what a later segment's opening is graded to)
    hold = LookHold(config.look_hold) if config.look_hold else None
    raw_carry = None
    # Segment 1's decoded frames, the latent anchor of every later segment
    # under `anchor: "segment_1"` (#1166) - the same frame objects the output
    # holds, so keeping them costs no copy on an unspilled chain
    anchor_frames = None
    # Segment 1's frame 0 alone, the keyframe of every later segment under
    # `keystone: "segment_1"` (#1187)
    keystone_frame = None
    # Segment 1's capstone frame, under the experimental capstone (#1187)
    capstone_frame = None
    # One shot per segment, measured as the picture and track grow (#378);
    # the frames are counted rather than read off `frames`, which a spilled
    # chain never fills
    shots = []
    frame_count = 0

    # The step's seed, which a re-roll offsets (chain_check.reroll_seed); the
    # previous segment's kept frames, which a check measures the seam
    # against, and segment 1's texture samples, which it measures drift by
    seed = _step_seed(arguments)
    previous_kept = None
    segment_1_texture = None

    try:
        for segment in config.plan:
            logger.info(
                f"Chain segment {segment.index + 1}/{len(config.plan)}"
                + (f": {segment.num_frames} frames" if segment.num_frames else "")
            )
            state = _ChainState(
                carry,
                raw_carry,
                hold,
                anchor_frames,
                keystone_frame,
                seed,
                capstone_frame,
            )
            if config.check is not None and segment.index > 0:
                attempt, checked = _checked_attempt(
                    pipeline,
                    arguments,
                    config,
                    segment,
                    continuity,
                    state,
                    previous_kept,
                    segment_1_texture,
                )
            else:
                attempt = _render_attempt(
                    pipeline, arguments, config, segment, continuity, state
                )
                checked = None

            # Only the kept attempt reaches the timeline: its carry, its
            # look-hold state, its frames and its audio
            carry, raw_carry = attempt.carry, attempt.raw_carry
            hold = attempt.hold
            if segment.index == 0:
                anchor_frames, keystone_frame, capstone_frame = _from_segment_1(
                    config, attempt.rendered
                )
            if config.check is not None:
                previous_kept = attempt.kept
                if segment.index == 0:
                    segment_1_texture = texture_samples(attempt.kept)
            segment_audio, segment_rate = attempt.audio, attempt.rate

            tail_trim = attempt.tail_trim
            if config.carry_pick and segment.index < len(config.plan) - 1:
                emit_log(
                    f"Chain seam {segment.index + 1}/{len(config.plan) - 1}: "
                    f"carry_pick took frame {len(attempt.rendered) - tail_trim} of "
                    f"{len(attempt.rendered)} (sharpest of the last "
                    f"{config.carry_pick}), dropping {tail_trim} after it"
                )

            kept_frames = attempt.kept
            start_sample = audio.shape[1] if audio is not None else 0
            shots.append(
                shot_record(
                    f"segment {segment.index + 1}",
                    frame_count,
                    len(kept_frames),
                    start_sample,
                )
            )
            if config.check is not None:
                _record_check(shots[-1], segment, config, checked)
            frame_count += len(kept_frames)
            if spill is not None:
                spill.write(
                    kept_frames,
                    _on_timeline_audio(segment_audio, segment, config, segment_rate),
                    segment_rate,
                )
                if config.keep_segments:
                    # The file as written ('-2' deduped on a rerun), which is
                    # how repair-segment finds a take's own files (#1169)
                    shots[-1]["segment_file"] = os.path.basename(spill.paths[-1])
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
                    _record_seam(shots[-1], segment, config, applied)

            shots[-1]["num_samples"] = (
                audio.shape[1] if audio is not None else 0
            ) - start_sample

            # The segment's raw output is finished with - the frames live on
            # (in RAM or on disk) and the carry frame is extracted. Free it
            # before the next segment needs the accelerator.
            del attempt, segment_audio, kept_frames
            gc.collect()
            empty_device_cache()
    finally:
        # The label belongs to the run that set it - on a raise or a cancel
        # as much as on the way out
        pipeline.segment_label = None
        pipeline.latent_anchor_source = None

    return _stitched(config, spill, frames, audio, audio_rate, shots)


@dataclass
class _ChainState:
    """What a segment is rendered from: the previous segment's carry (and,
    under look_hold, the carry before pre-compensation and the hold's state),
    segment 1's anchor frames and keystone, and the step's seed. A re-roll
    renders from exactly this again."""

    carry: object
    raw_carry: object
    hold: object  # LookHold or None
    anchor_frames: list
    keystone_frame: object
    seed: int  # None when the step passes neither a generator nor a seed
    capstone_frame: object = None


@dataclass
class _Attempt:
    """One render of a segment and everything derived from it, before any of
    it is committed to the chain's timeline."""

    rendered: list  # its frames as rendered (look-held), before any trim
    kept: list  # the frames it adds to the timeline: tail and head trimmed
    audio: object  # (channels, samples) numpy, tail trimmed, or None
    rate: int
    carry: object
    raw_carry: object
    hold: object  # the look-hold state after this segment, or None
    tail_trim: int
    seed: int = None


def _render_attempt(
    pipeline, arguments, config, segment, continuity, state, number=0, attempts=1
):
    """Render one attempt of a segment from `state`. Attempt 0 runs on the
    step's own generator; attempt n > 0 on `chain_check.reroll_seed`'s seed.
    The look-hold state is forked when more than one attempt may run, so a
    discarded attempt leaves nothing behind in it."""
    segment_arguments = _segment_arguments(
        arguments,
        config,
        segment,
        continuity,
        state.carry,
        state.anchor_frames,
        state.keystone_frame,
    )
    if segment.index > 0 and state.capstone_frame is not None:
        _add_capstone(segment_arguments, state.capstone_frame, config.capstone)
    seed = _reseeded(segment_arguments, state.seed, number) if number else None

    # The denoise counter restarts for every segment - without this the
    # bar rewinds to zero with nothing saying why
    pipeline.segment_label = f"segment {segment.index + 1}/{len(config.plan)}" + (
        f", attempt {number + 1}/{attempts}" if number else ""
    )
    # Names the anchor in the job log when this segment carries one
    pipeline.latent_anchor_source = (
        "segment 1" if segment.index > 0 and state.anchor_frames is not None else None
    )
    output = pipeline._run_once(segment_arguments)
    artifact = _single_artifact(output)
    del output

    hold, raw_carry = state.hold, state.raw_carry
    if hold is not None:
        if attempts > 1:
            hold = copy.deepcopy(hold)
        artifact = _look_held(artifact, hold, raw_carry, segment.index == 0)
    carry = continuity.extract(artifact)
    if hold is not None and config.continuity == "last_frame":
        raw_carry, carry = carry, hold.feed(carry)
    rendered = frames_as_pil_list(artifact)
    segment_audio, segment_rate = _generated_audio(artifact)
    # A segment's own generated audio can run a codec-padding sliver
    # short of the frames it was asked for - the same gap #197 fixed for
    # a decoded file (_decode_audio_video) and for an in-memory
    # previous_result shot (Result.save). A chained segment goes through
    # neither of those, so the shortfall was surviving here uncorrected
    # and compounding once per segment (#408).
    if segment_audio is not None and segment_rate and config.fps:
        segment_audio = fit_codec_padding(
            segment_audio, len(rendered), config.fps, segment_rate
        )

    segment_frames = rendered
    tail_trim = config.tail_trim + getattr(continuity, "picked_back", 0)
    if tail_trim and segment.index < len(config.plan) - 1:
        # The carry is already taken; the tail it backed off from is
        # replaced by the next segment's fresh take (#1189), and a
        # picked frame ends its segment (#1032)
        segment_frames = segment_frames[:-tail_trim]
        if segment_audio is not None:
            segment_audio = segment_audio[
                :,
                : segment_audio.shape[1]
                - frames_to_samples(tail_trim, config.fps, segment_rate),
            ]
    return _Attempt(
        rendered,
        segment_frames[segment.head_trim :],
        segment_audio,
        segment_rate,
        carry,
        raw_carry,
        hold,
        tail_trim,
        seed,
    )


def _checked_attempt(
    pipeline,
    arguments,
    config,
    segment,
    continuity,
    state,
    previous_kept,
    segment_1_texture,
):
    """Render a segment, check it against the one before and segment 1, and
    re-roll it from the same state up to `config.retries` times while it
    fails. Returns (the kept attempt, its record): the attempt with the
    fewest failed measures, then the lowest jump ratio (`best_attempt`), and
    {"attempts", "result", "kept"}. Only the best attempt so far is held."""
    attempts = 1 + config.retries
    results = []
    best, best_index = None, None
    for number in range(attempts):
        if number:
            emit_log(
                f"Chain segment {segment.index + 1}/{len(config.plan)}: check "
                f"failed ({_failures(results[-1])}) - re-rolling, attempt "
                f"{number + 1} of {attempts}"
                + (
                    f" at seed {reroll_seed(state.seed, number)}"
                    if state.seed is not None
                    else " (the step passes no seed to vary)"
                )
            )
            gc.collect()
            empty_device_cache()
        attempt = _render_attempt(
            pipeline, arguments, config, segment, continuity, state, number, attempts
        )
        results.append(
            check_segment(previous_kept, attempt.kept, segment_1_texture, config.check)
        )
        kept_index = best_attempt(results)
        if kept_index == number:
            best, best_index = attempt, number
        del attempt
        # A texture-only failure is recorded and warned, never re-rolled (R38)
        if not results[-1].rerolls:
            break
    return best, {
        "attempts": len(results),
        "result": results[best_index],
        "kept": best_index,
        "seed": best.seed,
    }


def _failures(result):
    return ", ".join(f"{name} {result.measures.get(name)}" for name in result.failed)


def _record_check(shot, segment, config, checked):
    """Put a checked segment's attempts on its shot - and, when every attempt
    failed, the measures its kept attempt failed, with one job warning that
    reaches the caller. Segment 1 is never checked: one attempt."""
    if checked is None:
        shot["attempts"] = 1
        return
    shot["attempts"] = checked["attempts"]
    result, kept = checked["result"], checked["kept"]
    if kept:
        # The seed the kept re-roll ran at, so the run can be repeated
        shot["seed"] = checked.get("seed")
    if result.passed:
        return
    # Texture alone ends the loop on attempt 1 without trying the budget
    exhausted = result.rerolls
    shot["check_failed"] = list(result.failed)
    emit_warning(
        f"Chain segment {segment.index + 1}/{len(config.plan)} failed its check "
        + (
            f"on every attempt ({_failures(result)}); kept attempt {kept + 1} of "
            f"{checked['attempts']}, the closest"
            if exhausted
            else f"({_failures(result)}); texture drift is not re-rolled, "
            "a fresh seed inherits the carry"
        ),
        kind="chain_check_failed",
        segment=segment.index + 1,
        measures=list(result.failed),
        values={name: result.measures.get(name) for name in result.failed},
        kept_attempt=kept + 1,
        attempts=checked["attempts"],
    )


def _step_seed(arguments):
    """The seed a chain's segments run at: the step's generator's (a
    workflow naming none drew one before the run, `workflow_run`, and seeded
    the generator with it), else a `seed` argument, else None."""
    generator = arguments.get("generator")
    if isinstance(generator, torch.Generator):
        return generator.initial_seed()
    seed = arguments.get("seed")
    if isinstance(seed, int) and not isinstance(seed, bool):
        return seed
    return None


def _reseeded(segment_arguments, seed, number):
    """Point a re-roll's arguments at attempt `number`'s seed - a fresh
    generator on the step generator's device, or the `seed` argument - and
    return that seed; None, with the arguments as they were, without one.
    The step's own generator is left where attempt 0 left it, so the
    segments after a re-roll run as they would have."""
    if seed is None:
        return None
    rerolled = reroll_seed(seed, number)
    generator = segment_arguments.get("generator")
    if isinstance(generator, torch.Generator):
        segment_arguments["generator"] = torch.Generator(
            device=generator.device
        ).manual_seed(rerolled)
    else:
        segment_arguments["seed"] = rerolled
    return rerolled


def _look_held(artifact, hold, raw_carry, first):
    """The segment with its frames look-corrected (#1191), audio untouched -
    the corrected frames are what the carry is taken from and what the
    stitched video is made of."""
    frames = hold.correct(frames_as_pil_list(artifact), raw_carry, first)
    if isinstance(artifact, AudioVideo):
        # A shallow copy with the new frames: audio, rate and shots as they were
        held = copy.copy(artifact)
        held.frames = frames
        return held
    return frames


def _stitched(config, spill, frames, audio, audio_rate, shots):
    """The chain's one AudioVideo: the frames (replayed from disk when
    spilled), and either the joined generated audio, the original match_audio
    track, or no audio at all."""
    if spill is not None:
        frames = spilled_frames(config, spill)

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

    if config.duration_seconds is not None:
        # The plan overshoots by design; the picture and the generated audio
        # are cut to the asked length together
        if spill is None:
            frames = frames[: config.total_frames]
        if audio is not None:
            audio = audio[
                :, : frames_to_samples(config.total_frames, config.fps, audio_rate)
            ]
        shots = _trimmed_shots(
            shots, config.total_frames, audio.shape[1] if audio is not None else None
        )
    elif audio is None:
        shots = without_samples(shots)
    return AudioVideo(frames, audio, audio_rate, fps=config.fps, shots=shots)


def _from_segment_1(config, rendered):
    """What later segments take from segment 1's frames as rendered: its
    frames as the latent anchor, its frame 0 as the keystone (#1187) and its
    late capstone frame (the experimental capstone) - each None when off."""
    if config.capstone and config.capstone["from_end"] > len(rendered):
        raise ValueError(
            f"Chain {CHAIN_CAPSTONE} from_end {config.capstone['from_end']} is "
            f"past segment 1's {len(rendered)} frames"
        )
    return (
        rendered if config.anchor == ANCHOR_SEGMENT_1 else None,
        rendered[0] if config.keystone == KEYSTONE_SEGMENT_1 else None,
        rendered[-config.capstone["from_end"]] if config.capstone else None,
    )


def _add_capstone(segment_arguments, frame, capstone):
    """Put segment 1's capstone frame one latent past this segment's end, as
    a guide that enters the forward below its sigma gate (#1187, h3_windows)."""
    guide = {
        "video": [frame],
        "past_end": True,
        "enter_below_sigma": capstone["enter_below_sigma"],
    }
    segment_arguments[GUIDES_INPUT] = list(
        segment_arguments.get(GUIDES_INPUT) or []
    ) + [guide]


def _segment_arguments(
    arguments, config, segment, continuity, carry, anchor_frames=None, keystone=None
):
    """One segment's arguments: the step's, with this segment's prompt, its
    slice of a match_audio track, the previous segment's carry, and - under
    `anchor: "segment_1"`, on every segment after the first - segment 1's
    frames as its `latent_anchor`."""
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
        if keystone is not None:
            continuity.inject(
                segment_arguments, carry, config.segment_argument, keystone=keystone
            )
        else:
            continuity.inject(segment_arguments, carry, config.segment_argument)
        if anchor_frames is not None:
            segment_arguments[LATENT_ANCHOR_INPUT] = anchor_frames
            if config.anchor_strength is not None:
                segment_arguments[LATENT_ANCHOR_STRENGTH_INPUT] = config.anchor_strength
    return segment_arguments


def _record_seam(shot, segment, config, applied):
    """Record a seam's own blend on its shot, so assess_output can tell it
    from a dropout in the content (#660), and say so in the job log."""
    shot["trim_frames"] = segment.head_trim
    if "crossfade_ms" in applied:
        shot["crossfade_ms"] = applied["crossfade_ms"]
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


def _check_anchor_chain(pipeline, arguments):
    """Refuse an `anchor: "segment_1"` chain whose segments could not run the
    latent anchor step - a loaded pipeline that is not MiniMax-H3 t2va/fl2va,
    or a step that passes references (ref2va) - before segment 1 renders."""
    if arguments.get("references") is not None:
        raise ValueError(
            "Chain anchor 'segment_1' cannot be combined with references - its "
            "segments would run ref2va, which has no anchor step"
        )
    loaded = getattr(pipeline, "pipeline", None)
    if loaded is not None:
        refusal = anchor_refusal(loaded)
        if refusal:
            raise ValueError(f"Chain anchor 'segment_1': {refusal}")


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


def _trimmed_shots(shots, total_frames, total_samples=None):
    """A length-planned chain's shots, cut where its overshooting picture is.

    A match_audio soundtrack is the caller's own, laid under whole rather than
    built segment by segment, so no shot has a stretch of it to measure: the
    sample side is cleared, not derived. A duration_seconds chain's generated
    audio is cut to `total_samples`, and each shot's stretch of it with it.
    """
    trimmed = []
    for shot in shots if total_samples is not None else without_samples(shots):
        if shot["start_frame"] >= total_frames:
            break
        end = min(shot["start_frame"] + shot["num_frames"], total_frames)
        cut = {**shot, "num_frames": end - shot["start_frame"]}
        if total_samples is not None:
            start = min(shot["start_sample"], total_samples)
            cut["num_samples"] = min(shot["num_samples"], total_samples - start)
        trimmed.append(cut)
    return trimmed


class ChainConfig:
    """Validated chain settings plus the planned segments for one run."""

    def __init__(
        self, chain_definition, arguments, resolved_prompts=None, base_dir=None
    ):
        segments = chain_definition.get("segments", None)
        match_audio = bool(chain_definition.get("match_audio", False))
        duration = chain_definition.get("duration_seconds", None)
        # A resolved duration wins over a segment count: a template carries a
        # default `segments`, and asking for seconds must not need it nulled
        counted = segments is not None or duration is not None
        if counted == match_audio:
            raise ValueError(
                "A chain needs exactly one of 'segments', 'match_audio' or "
                "'duration_seconds'"
            )
        if segments is not None and duration is not None:
            emit_log(
                f"Chain 'segments' ({segments}) ignored: 'duration_seconds' "
                f"({float(duration):g}) sets the segment count"
            )

        self.continuity = chain_definition.get("continuity", "last_frame")
        if self.continuity not in CONTINUITY_MODES:
            known = ", ".join(sorted(CONTINUITY_MODES))
            raise ValueError(
                f"Unknown chain continuity '{self.continuity}' - expected one of {known}"
            )

        # Segment 1 as every later segment's latent statistics anchor (#1166)
        problem = chain_anchor_problem(chain_definition)
        if problem:
            raise ValueError(f"Chain {problem}")
        self.anchor = chain_definition.get("anchor", ANCHOR_NONE)
        # How far each later segment's match goes (#1190); None is the full one
        self.anchor_strength = chain_definition.get("anchor_strength")
        problem = (
            anchor_strength_problem(self.anchor_strength)
            if (self.anchor_strength is not None)
            else None
        )
        if problem:
            raise ValueError(f"Chain {problem}")
        if self.anchor_strength is not None:
            self.anchor_strength = float(self.anchor_strength)

        # Segment 1's frame 0 as every later keyframe (#1187)
        problem = chain_keystone_problem(chain_definition)
        if problem:
            raise ValueError(f"Chain {problem}")
        self.keystone = chain_definition.get("keystone", KEYSTONE_NONE)

        # Segment 1's late frame past every later segment's end (#1187), a spike
        problem = chain_capstone_problem(chain_definition)
        if problem:
            raise ValueError(f"Chain {problem}")
        capstone = chain_definition.get(CHAIN_CAPSTONE)
        self.capstone = capstone and {
            "enter_below_sigma": float(capstone["enter_below_sigma"]),
            "from_end": int(capstone.get("from_end", CHAIN_CAPSTONE_FROM_END)),
        }

        # The sharpest of the last N frames as the keyframe (#1032)
        problem = chain_carry_pick_problem(
            chain_definition,
            arguments.get("num_frames") if isinstance(arguments, dict) else None,
        )
        if problem:
            raise ValueError(f"Chain {problem}")
        pick = chain_definition.get("carry_pick")
        self.carry_pick = int(pick["sharpest"]) if pick else 0

        # Pixel corrections on each segment before its carry (#1191)
        problem = chain_look_hold_problem(chain_definition)
        if problem:
            raise ValueError(f"Chain {problem}")
        self.look_hold = chain_look_hold_flags(chain_definition)

        # The per-segment check and its re-roll budget; absent is off and 0
        problem = chain_check_problem(chain_definition)
        if problem:
            raise ValueError(f"Chain {problem}")
        self.check = segment_check(chain_definition.get("check"))
        self.retries = int(chain_definition.get("retries", 0)) if self.check else 0

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
        # Frames dropped from the tail of every segment but the last (#1189)
        self.guide_backoff = 0
        self.tail_trim = 0
        problems = guide_backoff_problems(chain_definition)
        if problems:
            raise ValueError("; ".join(f"Chain {message}" for _, message in problems))
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
            self.guide_backoff = int(chain_definition.get("guide_backoff", 0))
            self.tail_trim = self.guide_backoff
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
        self.duration_seconds = float(duration) if duration is not None else None

        if match_audio:
            self._plan_from_audio(arguments, base_dir)
        elif duration is not None:
            self.total_frames, self.plan = _duration_plan(chain_definition, arguments)
            logger.info(
                f"Chaining to {float(duration):g}s: {self.total_frames} frames "
                f"across {len(self.plan)} segments"
            )
        else:
            segments = int(segments)
            if not 1 <= segments <= MAX_SEGMENTS:
                raise ValueError(
                    f"Chain 'segments' must be between 1 and {MAX_SEGMENTS}, got {segments}"
                )
            num_frames = arguments.get("num_frames", None)
            if num_frames is not None:
                validate_frame_snap(int(num_frames), self.frame_snap)
                if (
                    self.guide_frames is not None
                    and int(num_frames) <= head_trim + self.tail_trim
                ):
                    raise ValueError(
                        f"Segments of {num_frames} frames cannot progress past "
                        f"a {head_trim}-frame guide"
                        + (
                            f" and a {self.tail_trim}-frame back-off"
                            if self.tail_trim
                            else ""
                        )
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


def _head_trim(chain_definition):
    """Head frames each segment after the first drops: the guide a `guide`
    chain laid in, else `trim_frames`."""
    if chain_definition.get("continuity", "last_frame") == GUIDE_CONTINUITY:
        return int(chain_definition.get("guide_frames", GUIDE_CHAIN_DEFAULT))
    return int(chain_definition.get("trim_frames", 1))


def _duration_plan(chain_definition, arguments):
    """(total_frames, plan) for a `duration_seconds` chain - the same plan a
    `match_audio` chain makes from a track, from the asked length instead."""
    duration = float(chain_definition["duration_seconds"])
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(
            f"Chain 'duration_seconds' must be greater than 0, got {duration:g}"
        )
    fps = _resolve_fps(chain_definition, arguments)
    if fps is None:
        raise ValueError(
            "A duration_seconds chain needs the frame rate - set 'fps' on the "
            "chain or a 'frame_rate' pipeline argument"
        )
    num_frames = arguments.get("num_frames", None)
    if num_frames is None:
        raise ValueError(
            "A duration_seconds chain needs 'num_frames' in the step's arguments "
            "as the per-segment length"
        )
    total_frames = max(1, round(duration * fps))
    plan = plan_segments(
        total_frames,
        int(num_frames),
        _head_trim(chain_definition),
        chain_definition.get("frame_snap", None),
    )
    return total_frames, plan


def chain_segment_count(chain_definition, arguments):
    """How many segments a chain block runs, or None when it cannot be told
    from the block and the step's arguments alone (`match_audio`, an
    unresolved reference, or a missing frame rate or `num_frames`). A
    `duration_seconds` wins over `segments`. Pure: the run's `ChainConfig`
    and the cost estimate both ask it."""
    segments = chain_definition.get("segments", None)
    duration = chain_definition.get("duration_seconds", None)
    if chain_definition.get("match_audio") or (segments is None and duration is None):
        return None
    try:
        if duration is None:
            return int(segments)
        frame_snap = chain_definition.get("frame_snap", None)
        definition = {**chain_definition}
        if not isinstance(frame_snap, dict):
            definition.pop("frame_snap", None)
        return len(_duration_plan(definition, arguments)[1])
    except (TypeError, ValueError):
        return None


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
