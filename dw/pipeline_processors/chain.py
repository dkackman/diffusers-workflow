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
    WORDS_MEASURES,
    audio_dbfs,
    best_attempt,
    check_audio,
    check_segment,
    check_words,
    face_base,
    face_record,
    failures_text as _failures,
    measure_frames,
    reroll_seed,
    shown_value as _shown_value,
    texture_base,
    with_face_drop,
    with_words,
)
from . import chain_resume
from .chain_config import ChainConfig, trimmed_take_shots
from .chain_continuity import CONTINUITY_MODES, _generated_audio
from .chain_look import LookHold
from .h3_anchor import anchor_refusal
from .h3_guides import guides_refusal, latent_guide_tail, levelled_tail
from .chain_rules import (
    CHAIN_CAPSTONE,
    CHAIN_LATENT_CARRY,
    KEYSTONE_SLOT_LAST,
    KEYSTONE_SLOT_LATENT,
    place_keystone,
    segment_1_keystone,
)
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
    # The block's previous_result: lists, Step.run resolved (resolve_chain_lists)
    config = ChainConfig(
        chain_definition,
        arguments,
        getattr(pipeline, "chain_prompts", None),
        getattr(pipeline, "base_dir", None),
        getattr(pipeline, "chain_lines", None),
        getattr(pipeline, "chain_keystone_images", None),
    )
    continuity = CONTINUITY_MODES[config.continuity](config)
    if config.continuity == GUIDE_CONTINUITY:
        _check_guide_chain(pipeline, arguments, chain_definition, config)
    if config.anchor == ANCHOR_SEGMENT_1:
        _check_anchor_chain(pipeline, arguments)
    _warn_idle_keystone(config, arguments)

    # With save_segments, each completed segment is written to disk and its
    # frames freed, bounding memory to one segment - a crash leaves the
    # finished segments behind as playable files
    spill = SegmentSpill(pipeline, config) if config.save_segments else None

    timeline = _Timeline()
    carry = None
    # The corrected frames' look-hold state, and the last carry before any
    # pre-compensation (what a later segment's opening is graded to)
    hold = LookHold(config.look_hold) if config.look_hold else None
    level = None  # a levelled latent carry's reference, set by the first tail
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
    # What the keystone does at each seam, for the job log; None when off
    keystone_note = _keystone_note(config, arguments)

    # The step's seed, which a re-roll offsets (chain_check.reroll_seed); the
    # previous segment's kept frames, which a check measures the seam
    # against, and segment 1's fine-band texture and face base (frame 0 and
    # its median match), which it measures drift by - each measured once
    seed = _step_seed(arguments)
    previous_kept = None
    segment_1_texture = segment_1_face = None

    # An earlier run's kept segments, copied in as segments 1..k
    resumed = chain_resume.load_or_hint(config, spill, hold, pipeline)
    if resumed is not None:
        resumed.lay_down(timeline)
        carry, raw_carry = resumed.carry, resumed.raw_carry
        anchor_frames, keystone_frame = resumed.anchor_frames, resumed.keystone_frame
        capstone_frame = resumed.capstone_frame
        previous_kept = resumed.previous_kept
        segment_1_texture = resumed.segment_1_texture
        segment_1_face = resumed.segment_1_face

    try:
        for segment in config.plan[resumed.count if resumed else 0 : config.end_index]:
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
                level,
            )
            if config.check is not None and (
                segment.index > 0 or config.checks_segment_1
            ):
                attempt, checked = _checked_attempt(
                    pipeline,
                    arguments,
                    config,
                    segment,
                    continuity,
                    state,
                    previous_kept,
                    (segment_1_texture, segment_1_face),
                )
            else:
                attempt = _render_attempt(
                    pipeline, arguments, config, segment, continuity, state
                )
                checked = None

            # Only the kept attempt reaches the timeline: its carry, its
            # look-hold state, its frames and its audio
            carry, raw_carry = attempt.carry, attempt.raw_carry
            hold, level = attempt.hold, attempt.level
            if segment.index == 0:
                anchor_frames, keystone_frame, capstone_frame = _from_segment_1(
                    config, attempt.rendered, attempt.latent_0
                )
            if config.check is not None:
                # Keep the kept segment's small measures, not its frames
                if segment.index == 0:
                    segment_1_texture = texture_base(attempt.kept)
                    segment_1_face = face_base(attempt.kept)
                    previous_kept = measure_frames(attempt.kept)
                else:
                    previous_kept = checked["result"].frame_data
            if resumed is not None:
                resumed.check_rendered_audio(config, segment, attempt.audio)
            _log_carry_pick(config, segment, attempt)
            _log_seam_holds(config, segment, attempt, state, keystone_note)
            _lay_down(timeline, config, segment, attempt, checked, spill)

            # The segment's raw output is finished with - the frames live on
            # (in RAM or on disk) and the carry frame is extracted. Free it
            # before the next segment needs the accelerator.
            del attempt
            gc.collect()
            empty_device_cache()
    except Exception:
        # The finished segments on disk are where a rerun can pick up
        chain_resume.warn_resumable(spill)
        raise
    finally:
        # The label belongs to the run that set it - on a raise or a cancel
        # as much as on the way out
        pipeline.segment_label = None
        pipeline.latent_anchor_source = None

    return _stitched_or_hint(config, spill, timeline)


def _stitched_or_hint(config, spill, timeline):
    """`_stitched` over the finished timeline. Every segment is on disk by
    now, so a failed final encode is a chain that can resume with all of
    them kept: it draws the resume hint a failed segment does."""
    try:
        return _stitched(
            config,
            spill,
            timeline.frames,
            timeline.audio,
            timeline.audio_rate,
            timeline.shots,
        )
    except Exception:
        chain_resume.warn_resumable(spill)
        raise


def _warn_idle_keystone(config, arguments):
    """Warn that `keystone: "segment_1"` does nothing on a t2va step, which
    has no keyframe argument for the keystone to replace (#1232)."""
    if (
        config.keystone == KEYSTONE_SEGMENT_1
        and config.keystone_slot is None
        and arguments.get(config.segment_argument) is None
    ):
        emit_warning(
            f"Chain keystone 'segment_1' has no effect on this step: it takes "
            f"no {config.segment_argument!r} keyframe (t2va), so every later "
            f"segment is generated without the keystone frame"
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


def _keystone_note(config, arguments):
    """What `keystone: "segment_1"` does to each later segment, as the job
    log says it - where its keystone_slot puts segment 1's frame 0 - or None
    when it is off or has no effect (a t2va step with no keyframe to replace,
    which run_chain has warned of)."""
    if config.keystone != KEYSTONE_SEGMENT_1:
        return None
    slot = config.keystone_slot
    if slot is None:
        if arguments.get(config.segment_argument) is None:
            return None
        return "opens on segment 1's frame 0"
    if config.keystone_images:
        return "ends on keystone_images[{entry}] (its last_image)"
    if slot == KEYSTONE_SLOT_LAST:
        return "ends on segment 1's frame 0 (its last_image)"
    if slot == KEYSTONE_SLOT_LATENT:
        return "holds segment 1's frame-0 latent over its guide's latent 0"
    return f"is guided by segment 1's frame 0 at frame {slot['frame']}"


def _log_seam_holds(config, segment, attempt, state, keystone_note=None):
    """Say in the job log what keystone and look_hold did at a seam: a segment
    after the first is rendered from the keystone frame (where
    `_keystone_note` says) and corrected by the look hold; the first only
    sets the scene, and its carry may be pre-compensated for the next."""
    seam = f"Chain seam {segment.index}/{len(config.plan) - 1}"
    keystone = config.keystone_for(segment.index, state.keystone_frame)
    if keystone is not None and keystone_note:
        note = keystone_note.format(entry=segment.index)
        emit_log(f"{seam}: keystone applied - segment {segment.index + 1} {note}")
    if attempt.hold is None:
        return
    parts = ", ".join(name for name, on in config.look_hold.items() if on)
    if segment.index == 0:
        emit_log(
            f"Chain segment 1/{len(config.plan)}: look_hold read the scene look "
            f"from segment 1 ({parts}); segment 1 is left as rendered"
        )
    else:
        detail = "; ".join(attempt.hold.notes)
        emit_log(
            f"{seam}: look_hold corrected segment {segment.index + 1} ({parts})"
            + (f" - {detail}" if detail else "")
        )


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
    _record_level(shot, segment, config, attempt)
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
    level: object = None  # the first carried tail's latent_level, when levelled


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
    latent_0: object = None  # segment 1's frame-0 latent, a latent keystone's
    level: object = None  # the level later tails are pulled to, when levelled


def _render_attempt(
    pipeline, arguments, config, segment, continuity, state, number=0, attempts=1
):
    """Render one attempt of a segment from `state`. Attempt 0 runs on the
    step's own generator; attempt n > 0 on `chain_check.reroll_seed`'s seed.
    The look-hold state is forked when more than one attempt may run, so a
    discarded attempt leaves nothing behind in it."""
    # Under anchor_every only every n-th later segment takes these two
    anchored = config.anchors(segment.index)
    segment_arguments = _segment_arguments(
        arguments,
        config,
        segment,
        continuity,
        state.carry,
        state.anchor_frames,
        config.keystone_for(segment.index, state.keystone_frame),
    )
    if anchored and state.capstone_frame is not None:
        _add_capstone(segment_arguments, state.capstone_frame, config.capstone)
    seed = _reseeded(segment_arguments, state.seed, number) if number else None
    # A latent keystone is segment 1's own frame-0 latent
    latent_0 = segment.index == 0 and config.latent_keystone
    if config.latent_carry or latent_0:
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
    taken = _taken_latents(output) if config.latent_carry or latent_0 else None
    # The frames as PIL once: the look hold, the carry and the timeline then
    # all take them by identity rather than converting again
    artifact = _with_pil_frames(_single_artifact(output))
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
    if config.latent_carry:
        carry = _with_latent_tail(carry, taken, config, len(rendered))
    level = state.level
    if config.latent_level:
        level = levelled_tail(carry, level, segment.index)
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
        taken[0][:, :, :1].clone() if latent_0 else None,
        level,
    )


def _checked_attempt(
    pipeline,
    arguments,
    config,
    segment,
    continuity,
    state,
    previous_kept,
    segment_1,
):
    """Render a segment, check it against the one before and segment 1 - its
    (texture, face) bases - and
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
        kept_audio = _kept_audio(attempt, segment, config)
        result = (
            check_audio(kept_audio, config.check)
            if segment.index == 0
            else check_segment(
                previous_kept,
                attempt.kept,
                segment_1[0],
                config.check,
                audio=kept_audio,
            )
        )
        if config.checks_words:
            result = with_words(
                result, _segment_words(kept_audio, attempt, segment, config)
            )
        result = with_face_drop(result, attempt.kept, segment_1[1], config.check)
        results.append(result)
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
        # Every attempt failed its words: a fresh seed did not fix them, so
        # the cause may be in the carry (#1262)
        "every_attempt_failed": all(
            "words" in r.failed or "words_opening" in r.failed for r in results
        ),
    }


def _segment_words(kept_audio, attempt, segment, config):
    """The words measure of one attempt (#1262): `check_words` on its kept
    audio against the segment's line, the line before it given for the
    seam-head rule - or None when the segment has no line or no audio, so
    it is never failed for words."""
    line = config.segment_line(segment.index)
    if line is None or kept_audio is None or attempt.rate is None:
        return None
    previous = config.segment_line(segment.index - 1) if segment.index else None
    # The whole audio and the span the head trim cuts off, so a failing line
    # can be told apart from one whose opening the trim cut (#1300)
    head_seconds = segment.head_trim / config.fps if config.fps else 0.0
    return check_words(
        kept_audio,
        attempt.rate,
        line,
        previous,
        config.check.words_similarity_min,
        full_audio=attempt.audio if head_seconds else None,
        head_seconds=head_seconds,
    )


def _record_check(shot, segment, config, checked):
    """Put a checked segment's attempts on its shot - and, when every attempt
    failed, the measures its kept attempt failed, with one job warning that
    reaches the caller. An unchecked segment 1 (no audio floor to hold it
    to) has one attempt."""
    if checked is None:
        shot["attempts"] = 1
        return
    shot["attempts"] = checked["attempts"]
    result, kept = checked["result"], checked["kept"]
    if kept:
        # The seed the kept re-roll ran at, so the run can be repeated
        shot["seed"] = checked.get("seed")
    shot["check"] = face_record(result)
    if "words" in result.measures:
        # The kept attempt's words, null for a segment with no line (#1262)
        shot["words"] = result.measures["words"]
    if result.passed:
        return
    # Texture alone ends the loop on attempt 1 without trying the budget
    exhausted = result.rerolls
    shot["check_failed"] = list(result.failed)
    words = (
        result.measures.get("words")
        if WORDS_MEASURES.intersection(result.failed)
        else None
    )
    detail = {}
    if words is not None:
        detail = {
            "heard": words["heard"],
            "similarity": words["similarity"],
            "findings": words["findings"],
        }
        if checked.get("every_attempt_failed"):
            detail["every_attempt_failed"] = True
            if "words" not in result.failed:
                # Only the opening failed: the similarity passed (#1363)
                detail["note"] = (
                    "every attempt opened past the line's first words (the "
                    "similarity passed): the head trim may cut it or it is "
                    "never spoken - try a different seed or prompt, or for a "
                    "long monologue use a narrated take"
                )
            elif segment.index == 0:
                # Segment 1 has no carry to blame and repair-segment cannot
                # repair it (#1363)
                detail["note"] = (
                    "every attempt failed its words: re-render segment 1 with "
                    "a different seed or a reworded prompt"
                )
            else:
                detail["note"] = (
                    "every attempt failed its words, so a fresh seed is not "
                    "the fix: the cause may be in the carry - repair the "
                    "segment with templates/minimax/repair-segment rather "
                    "than more retries"
                )
    emit_warning(
        f"Chain segment {segment.index + 1}/{len(config.plan)} failed its check "
        + (
            f"on every attempt ({_failures(result)}); kept attempt {kept + 1} of "
            f"{checked['attempts']}, the closest"
            if exhausted
            else f"({_failures(result)}); kept attempt {kept + 1} of "
            f"{checked['attempts']}; a warn-only measure is not re-rolled "
            "(texture drift is in the carry, which a fresh seed inherits)"
        )
        + (f" - {detail['note']}" if "note" in detail else ""),
        kind="chain_check_failed",
        segment=segment.index + 1,
        measures=list(result.failed),
        values={name: _shown_value(result, name) for name in result.failed},
        kept_attempt=kept + 1,
        attempts=checked["attempts"],
        **detail,
    )


def _kept_audio(attempt, segment, config):
    """The generated audio a kept attempt puts on the timeline - its span
    past the head trim - or None when it has none or the chain holds its
    audio, which is not the segment's to measure. With no frame rate the
    span cannot be placed, and the join refuses the chain itself."""
    if config.audio_held or attempt.audio is None or config.fps is None:
        return None
    return _on_timeline_audio(attempt.audio, segment, config, attempt.rate)


def _record_level(shot, segment, config, attempt):
    """Put the kept segment's generated audio level on its shot, and warn
    `chain_segment_silent` when it reads under the chain's floor - with the
    check on or off (#1221), so a silent segment is never kept unsaid."""
    level = audio_dbfs(_kept_audio(attempt, segment, config))
    if level is None:
        return
    shot["audio_dbfs"] = level
    floor = config.audio_floor
    if floor is None or level >= floor:
        return
    emit_warning(
        f"Chain segment {segment.index + 1}/{len(config.plan)} is silent: its "
        f"generated audio reads {level} dBFS, under the {floor:g} dBFS floor"
        + (
            " - turn check on with retries to re-roll it, or set check "
            "audio_dbfs_min null if the step asks for silence"
            if config.check is None
            else "; set check audio_dbfs_min null if the step asks for silence"
        ),
        kind="chain_segment_silent",
        segment=segment.index + 1,
        audio_dbfs=level,
        floor=floor,
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


def _with_pil_frames(artifact):
    """The segment's artifact with its frames as PIL images, converted once:
    an AudioVideo as a shallow copy (audio, rate and shots as they were), any
    other video as the list of its frames."""
    frames = frames_as_pil_list(artifact)
    if isinstance(artifact, AudioVideo):
        converted = copy.copy(artifact)
        converted.frames = frames
        return converted
    return frames


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
    # A probe's take ends at its last rendered segment (stop_after)
    total_frames, shots = config.take_frames(shots)

    if config.source_audio is not None:
        # The video matches the track's duration; the original, unsliced audio
        # is muxed in so the soundtrack has no seams
        if spill is None:
            frames = frames[:total_frames]
        return AudioVideo(
            frames,
            config.take_track(total_frames),
            config.source_rate,
            fps=config.fps,
            shots=trimmed_take_shots(shots, total_frames),
        )

    if config.duration_seconds is not None:
        # The plan overshoots by design; the picture and the generated audio
        # are cut to the asked length together
        if spill is None:
            frames = frames[:total_frames]
        if audio is not None:
            audio = audio[:, : frames_to_samples(total_frames, config.fps, audio_rate)]
        shots = trimmed_take_shots(
            shots, total_frames, audio.shape[1] if audio is not None else None
        )
    elif audio is None:
        shots = without_samples(shots)
    return AudioVideo(frames, audio, audio_rate, fps=config.fps, shots=shots)


def _copied(frame):
    """`frame` as its own object when it is copyable (a PIL image)."""
    return frame.copy() if hasattr(frame, "copy") else frame


def _from_segment_1(config, rendered, latent_0=None):
    """What later segments take from segment 1's frames as rendered: its
    frames as the latent anchor, its frame 0 as the keystone (#1187; with its
    frame-0 latent `latent_0` under keystone_slot "latent") and its late
    capstone frame (the experimental capstone) - each None when off."""
    if config.capstone and config.capstone["from_end"] > len(rendered):
        raise ValueError(
            f"Chain {CHAIN_CAPSTONE} from_end {config.capstone['from_end']} is "
            f"past segment 1's {len(rendered)} frames"
        )
    return (
        rendered if config.anchor == ANCHOR_SEGMENT_1 else None,
        # A copy: the output holds rendered[0], and a later segment's
        # processing must not reach it through the shared object (#1232)
        segment_1_keystone(config.keystone_slot, _copied(rendered[0]), latent_0)
        if config.keystone == KEYSTONE_SEGMENT_1
        else None,
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
    video tail, and the audio rows when the chain carries generated audio - a
    held track is each segment's own slice, so its guide carries video only."""
    latents, audio = taken
    video, audio_rows = latent_guide_tail(
        latents,
        audio if config.carry_audio and not config.audio_held else None,
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

    if config.total_frames is not None:
        # A length-planned chain (match_audio, duration_seconds) renders its
        # shortened last segment at its planned length, not the step's
        segment_arguments["num_frames"] = segment.num_frames
    if config.source_audio is not None:
        if config.holds_audio:
            segment_arguments[HOLD_AUDIO_INPUT] = config.audio_slice(segment)
        else:
            segment_arguments["references"] = config.sliced_references(
                segment, arguments["references"]
            )

    if segment.index > 0:
        if keystone is not None and config.keystone_slot is not None:
            # Out of the keyframe's slot, which is the guide's first latent's
            continuity.inject(segment_arguments, carry, config.segment_argument)
            place_keystone(
                segment_arguments,
                keystone,
                config.keystone_slot,
                config.segment_argument,
            )
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
    elif keystone is not None:  # segment 1's own keystone_images entry
        place_keystone(segment_arguments, keystone, config.keystone_slot)
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
