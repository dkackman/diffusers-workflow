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
import os
from dataclasses import dataclass, field

import torch

from .. import empty_device_cache
from ..media_types import AudioVideo, fit_codec_padding
from ..output_extraction import get_artifact_list
from ..events import emit_log, emit_warning
from ..shots import shot_record, without_samples
from ..dsp import slice_samples
from ..task_domains import frames_to_samples
from ..tasks.joins import equal_power_crossfade_join
from ..tasks.video_utils import frames_as_pil_list
from .chain_spill import (  # noqa: F401 - SegmentedFrames and SavedFrames re-exported
    SavedFrames,
    SegmentedFrames,
    SegmentSpill,
    spilled_frames,
)
from .chain_check import (
    best_attempt,
    check_segment,
    measure_frames,
    reroll_seed,
    texture_samples,
)
from .chain_config import ChainConfig
from .chain_continuity import CONTINUITY_MODES, _generated_audio
from .chain_look import LookHold
from .h3_anchor import anchor_refusal
from .h3_guides import guides_refusal, latent_guide_tail
from .chain_rules import CHAIN_CAPSTONE, CHAIN_LATENT_CARRY, place_keystone
from .h3_rules import (
    ANCHOR_SEGMENT_1,
    GUIDES_INPUT,
    GUIDE_CHAIN_RULE,
    GUIDE_CONTINUITY,
    HOLD_AUDIO_INPUT,
    LATENT_ANCHOR_INPUT,
    LATENT_ANCHOR_STRENGTH_INPUT,
    KEYSTONE_SEGMENT_1,
)

logger = logging.getLogger("dw")


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

    timeline = _Timeline()
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
                # Keep the kept segment's small measures, not its frames
                if segment.index == 0:
                    segment_1_texture = texture_samples(attempt.kept)
                    previous_kept = measure_frames(attempt.kept)
                else:
                    previous_kept = checked["result"].frame_data
            _log_carry_pick(config, segment, attempt)
            _log_seam_holds(config, segment, attempt, state)
            _lay_down(timeline, config, segment, attempt, checked, spill)

            # The segment's raw output is finished with - the frames live on
            # (in RAM or on disk) and the carry frame is extracted. Free it
            # before the next segment needs the accelerator.
            del attempt
            gc.collect()
            empty_device_cache()
    finally:
        # The label belongs to the run that set it - on a raise or a cancel
        # as much as on the way out
        pipeline.segment_label = None
        pipeline.latent_anchor_source = None

    return _stitched(
        config,
        spill,
        timeline.frames,
        timeline.audio,
        timeline.audio_rate,
        timeline.shots,
    )


@dataclass
class _Timeline:
    """The chain's output as it grows: the frames on the output timeline
    (an unspilled chain's), the joined generated audio, (channels, samples)
    float32, and its rate, and one shot per segment, measured as the picture
    and track grow (#378). The frames are counted rather than read off
    `frames`, which a spilled chain never fills."""

    frames: list = field(default_factory=list)
    audio: object = None
    audio_rate: int = None
    shots: list = field(default_factory=list)
    frame_count: int = 0


def _log_carry_pick(config, segment, attempt):
    """Say in the job log which frame carry_pick took at a seam."""
    tail_trim = attempt.tail_trim
    if config.carry_pick and segment.index < len(config.plan) - 1:
        emit_log(
            f"Chain seam {segment.index + 1}/{len(config.plan) - 1}: "
            f"carry_pick took frame {len(attempt.rendered) - tail_trim} of "
            f"{len(attempt.rendered)} (sharpest of the last "
            f"{config.carry_pick}), dropping {tail_trim} after it"
        )


def _log_seam_holds(config, segment, attempt, state):
    """Say in the job log what keystone and look_hold did at a seam: a segment
    after the first is rendered from the keystone frame and corrected by the
    look hold; the first only sets the scene, and its carry may be
    pre-compensated for the next."""
    seam = f"Chain seam {segment.index}/{len(config.plan) - 1}"
    if segment.index > 0 and state.keystone_frame is not None:
        emit_log(
            f"{seam}: keystone applied - segment {segment.index + 1} opens on "
            "segment 1's frame 0"
        )
    if attempt.hold is None:
        return
    parts = ", ".join(name for name, on in config.look_hold.items() if on)
    if segment.index == 0:
        emit_log(
            f"Chain segment 1/{len(config.plan)}: look_hold read the scene look "
            f"from segment 1 ({parts}); segment 1 is left as rendered"
        )
    else:
        emit_log(f"{seam}: look_hold corrected segment {segment.index + 1} ({parts})")


def _lay_down(timeline, config, segment, attempt, checked, spill):
    """Commit a kept attempt to the timeline: its shot, its frames (to the
    spill when there is one) and its generated audio, joined on the seam."""
    kept_frames = attempt.kept
    segment_audio, segment_rate = attempt.audio, attempt.rate
    audio = timeline.audio
    start_sample = audio.shape[1] if audio is not None else 0
    shot = shot_record(
        f"segment {segment.index + 1}",
        timeline.frame_count,
        len(kept_frames),
        start_sample,
    )
    timeline.shots.append(shot)
    if config.check is not None:
        _record_check(shot, segment, config, checked)
    timeline.frame_count += len(kept_frames)
    if spill is not None:
        spill.write(
            kept_frames,
            _on_timeline_audio(segment_audio, segment, config, segment_rate),
            segment_rate,
        )
        if config.keep_segments:
            # The file as written ('-2' deduped on a rerun), which is how
            # repair-segment finds a take's own files (#1169)
            shot["segment_file"] = os.path.basename(spill.paths[-1])
    else:
        timeline.frames.extend(kept_frames)

    if config.source_audio is None and segment_audio is not None:
        applied = {}
        timeline.audio, timeline.audio_rate = _joined_audio(
            audio,
            timeline.audio_rate,
            segment_audio,
            segment_rate,
            segment,
            config,
            applied,
        )
        if segment.index > 0:
            _record_seam(shot, segment, config, applied)

    audio = timeline.audio
    shot["num_samples"] = (audio.shape[1] if audio is not None else 0) - start_sample


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
    if config.latent_carry:
        segment_arguments["output"] = _with_latent_outputs(
            segment_arguments.get("output")
        )

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
    taken = _taken_latents(output) if config.latent_carry else None
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
    if taken is not None:
        carry = _with_latent_tail(carry, taken, config, len(rendered))
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
            else f"({_failures(result)}); kept attempt {kept + 1} of "
            f"{checked['attempts']}; texture drift is not re-rolled, "
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


# The outputs a latent-carry segment adds to its call (#1150): diffusers'
# H3 after-denoise leaves the target's own latents in the state, unpacked
LATENT_OUTPUTS = ("latents", "audio_latents")


def _with_latent_outputs(output):
    """A segment's `output` names with the latents asked for too: the step's
    own (a string is one name; none is the chain's videos, audio and rate)."""
    names = [output] if isinstance(output, str) else list(output or [])
    if not names:
        names = ["videos", "audio", "sampling_rate"]
    return names + [name for name in LATENT_OUTPUTS if name not in names]


def _taken_latents(output):
    """(latents, audio_latents) taken off a latent-carry segment's output, on
    the CPU, before the one video is picked out of it; audio_latents None
    when the segment made no audio."""
    latents = output.pop("latents", None) if isinstance(output, dict) else None
    if not isinstance(latents, torch.Tensor):
        raise ValueError(
            f"Chain {CHAIN_LATENT_CARRY} hands on each segment's latents, and "
            "this segment returned none - it needs a MiniMax-H3 pipeline"
        )
    audio = output.pop("audio_latents", None)
    return latents.detach().cpu(), (
        audio.detach().cpu() if isinstance(audio, torch.Tensor) else None
    )


def _with_latent_tail(carry, taken, config, num_frames):
    """The guide carry with the same span's own latents on it (#1150): the
    video tail, and the audio rows when the chain carries audio."""
    latents, audio = taken
    video, audio_rows = latent_guide_tail(
        latents,
        audio if config.carry_audio else None,
        num_frames,
        config.guide_frames,
        config.guide_backoff,
    )
    carry.latents, carry.audio_latents = video, audio_rows
    return carry


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
        if keystone is not None and config.keystone_slot is not None:
            # Out of the keyframe's slot, which is the guide's first latent's
            continuity.inject(segment_arguments, carry, config.segment_argument)
            place_keystone(segment_arguments, keystone, config.keystone_slot)
        elif keystone is not None:
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
