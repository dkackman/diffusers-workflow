"""A chain resumed from an earlier run's kept segments (`resume_from`).

A chain run with `save_segments` writes each finished segment as a playable
mp4 (`chain_spill.SegmentSpill`) and a chain that dies leaves them on disk;
its failure warning (`chain_resume_hint`, `warn_resumable`) quotes the first
file's output reference. Naming that file in `resume_from` makes a new run of
the same chain take that run's segment set (`chain_repair.segment_files`, in
index order, complete) as its segments 1..k and render only k+1..N:

- The plan does not move: every kept file must hold the frames the plan puts
  on the timeline for its segment (`check_plan`), and k must be under N.
- The kept files are copied, never moved, into this run's spill under its
  own names, so the new take is self-contained and `keep_segments` names
  them on its shots (each also carries `resumed: true`); the originals stay.
- The carry is read off the last kept file. `last_frame` takes its last
  frame: the file already ends where a carry_pick cut it. `guide` takes its
  last `guide_frames` frames and, with `carry_audio`, their decoded audio:
  the file ends where a guide back-off cut it, so no back-off is taken
  again. That first resumed seam's audio guide is the decoded AAC track,
  re-encoded with no latents (Don, 2026-11-08), which the `chain_resumed`
  warning says; every later seam is a live carry. The latent carry is
  refused with a resume (`chain_rules.chain_resume_problem`).
- `anchor`, `keystone` and the capstone read segment 1 from the first kept
  file. The kept segments are not checked again; with `check`, the first
  rendered segment is checked against the last kept one as usual.
- `look_hold` cannot be re-derived faithfully: `LookHold` learns its boost
  from each segment's rendered opening against the keyframe it was fed, and
  a kept file holds only the corrected frames. So the hold reads the scene
  from the first kept file, as segment 1 sets it, and is fed the last kept
  file's carry; the boost restarts at none.
- The kept segments' audio is laid on the timeline whole, end to end: each
  file holds only its span past the head trim, so the crossfade material of
  their own seams is gone; the first rendered seam is joined as usual.
- The step generator restarts at the step's seed, so the rendered segments
  are not the renders the dead run would have made next.

`kept_count` is what the cost estimate asks (`cost_scaling`): the segments a
resolvable `resume_from` skips, 0 when it is unset or does not resolve yet.
"""

import logging
import os
import shutil
from dataclasses import dataclass

import numpy

from .. import references
from ..events import emit_warning
from ..media import probe_metadata
from ..media_types import fit_codec_padding
from ..runs import output_root, resolve_output_reference
from ..security import PathTraversalError, SecurityError, validate_path
from ..shots import shot_record
from ..task_domains import frames_to_samples
from ..tasks.chain_repair import segment_files
from ..tasks.video_utils import load_audio_video
from ..writers import output_file_path
from .chain_check import measure_frames, texture_base
from .chain_continuity import _SegmentCarry
from .chain_rules import CHAIN_RESUME_FROM, resume_reference_problem
from .h3_rules import ANCHOR_SEGMENT_1, GUIDE_CONTINUITY, KEYSTONE_SEGMENT_1

logger = logging.getLogger("dw")

# The estimate's note when `resume_from` names a file that is not there yet
UNRESOLVED_NOTE = "priced for every segment; resume_from is not resolvable at validate"


@dataclass
class ResumeState:
    """What a resumed chain starts from: the kept files copied into its
    spill, their shots and joined audio, and the chain state the last of
    them hands the first rendered segment."""

    kept: list  # the copied files, in this run's spill
    count: int
    shots: list
    audio: object  # (channels, samples) numpy, or None
    audio_rate: int
    frame_count: int
    carry: object  # a frame (last_frame) or a _SegmentCarry (guide)
    raw_carry: object
    anchor_frames: list
    keystone_frame: object
    capstone_frame: object
    previous_kept: object  # the last kept segment's check measures, or None
    segment_1_texture: object
    warning: str

    def lay_down(self, timeline):
        """Put the kept segments on the chain's timeline."""
        timeline.shots.extend(self.shots)
        timeline.frame_count = self.frame_count
        timeline.audio, timeline.audio_rate = self.audio, self.audio_rate


def resolved_reference(value):
    """The file a `resume_from` value names, validated: refused as a
    location unless it is an output reference, then resolved against the
    run's output root."""
    problem = resume_reference_problem(value)
    if problem:
        raise PathTraversalError(problem)
    return resolve_output_reference(value)


def kept_count(chain):
    """How many segments a chain block's `resume_from` keeps, for the cost
    estimate: 0 when it is unset or does not resolve (yet)."""
    value = chain.get(CHAIN_RESUME_FROM)
    if not references.is_ref(references.OUTPUT, value):
        return 0
    try:
        return len(segment_files(resolve_output_reference(value)))
    except (SecurityError, OSError, ValueError):
        return 0


def check_plan(kept_paths, plan, tail_trim=0, slack=0):
    """Refuse kept files the plan would not have made: k must be under the
    plan's N, and each file must hold the frames its segment puts on the
    timeline - its `num_frames` less its head trim and the tail trim (a
    kept segment is never the last), within `slack` frames short for a
    carry_pick that may end a segment early. Raises ValueError naming the
    segment, the kept count and the planned count."""
    if len(kept_paths) >= len(plan):
        raise ValueError(
            f"Chain {CHAIN_RESUME_FROM} found {len(kept_paths)} kept segment(s) "
            f"and the plan has {len(plan)}: nothing is left to render"
        )
    for segment, path in zip(plan, kept_paths):
        if segment.num_frames is None:
            raise ValueError(
                f"Chain {CHAIN_RESUME_FROM} checks the kept segments against "
                "the plan, which needs the step's num_frames"
            )
        planned = segment.num_frames - segment.head_trim - tail_trim
        kept = (probe_metadata(path) or {}).get("frame_count")
        if kept is None or not planned - slack <= kept <= planned:
            raise ValueError(
                f"Chain {CHAIN_RESUME_FROM}: kept segment {segment.index + 1} "
                f"({os.path.basename(path)}) holds {kept} frames and the plan "
                f"puts {planned} on the timeline for segment "
                f"{segment.index + 1} - a resume never changes the plan, so "
                "resume with the settings the earlier run used"
            )


def load(config, spill, hold=None):
    """Copy the kept segments `config.resume_from` names into `spill` and
    return the state the chain continues from, or None with no resume.
    `hold`, the chain's LookHold, is brought up to the last kept segment."""
    if config.resume_from is None:
        return None
    named = resolved_reference(config.resume_from)
    sources = segment_files(named)
    check_plan(
        sources,
        config.plan,
        config.tail_trim,
        config.carry_pick - 1 if config.carry_pick else 0,
    )
    kept = _Kept(config)
    for path in sources:
        spill.frame_count += kept.add(path, _copied_into(spill, path))
    last = kept.last
    if hold is not None:
        # The scene, as segment 1 sets it; the boost is not recoverable
        hold.correct(kept.first.frames, None, True)
    carry, raw_carry = _carry(config, last, hold)
    capstone = _capstone(config, kept.first)
    warning = _resumed_warning(config, len(sources), config.resume_from)
    emit_warning(
        warning,
        kind="chain_resumed",
        kept=len(sources),
        segments=len(config.plan),
        resume_from=config.resume_from,
    )
    return ResumeState(
        kept=list(spill.paths),
        count=len(sources),
        shots=kept.shots,
        audio=kept.audio,
        audio_rate=kept.audio_rate,
        frame_count=kept.frame_count,
        carry=carry,
        raw_carry=raw_carry,
        anchor_frames=kept.first.frames if config.anchor == ANCHOR_SEGMENT_1 else None,
        keystone_frame=kept.first.frames[0].copy()
        if config.keystone == KEYSTONE_SEGMENT_1
        else None,
        capstone_frame=capstone,
        previous_kept=measure_frames(last.frames) if config.check else None,
        segment_1_texture=texture_base(kept.first.frames) if config.check else None,
        warning=warning,
    )


class _Kept:
    """The kept segments as they are copied: their shots and joined audio,
    the first and last decoded (what segment 1 and the carry are read off)."""

    def __init__(self, config):
        self.config = config
        self.shots = []
        self.audio = None
        self.audio_rate = None
        self.frame_count = 0
        self.first = None
        self.last = None

    def add(self, source, copy):
        """Decode one copied file onto the kept timeline; returns its frames."""
        video = load_audio_video(copy)
        count = len(video.frames)
        start_sample = self.audio.shape[1] if self.audio is not None else 0
        self._join_audio(video, count)
        shot = shot_record(
            f"segment {len(self.shots) + 1}",
            self.frame_count,
            count,
            start_sample,
            resumed=True,
        )
        shot["num_samples"] = (
            self.audio.shape[1] if self.audio is not None else 0
        ) - start_sample
        if self.config.keep_segments:
            shot["segment_file"] = os.path.basename(copy)
        self.shots.append(shot)
        self.frame_count += count
        logger.info(f"Chain resume: kept {source} as {copy}")
        if self.first is None:
            self.first = video
        self.last = video
        return count

    def _join_audio(self, video, count):
        """Lay a kept file's own track end to end on the joined one - the
        generated audio only: a held or matched track is the chain's own."""
        if self.config.source_audio is not None or video.audio is None:
            return
        audio = fit_codec_padding(
            numpy.asarray(video.audio, dtype=numpy.float32),
            count,
            self.config.fps,
            video.sample_rate,
        )
        if self.audio is None:
            self.audio, self.audio_rate = audio, video.sample_rate
            return
        if video.sample_rate != self.audio_rate:
            raise ValueError(
                f"Chain {CHAIN_RESUME_FROM}: the kept segments' audio runs at "
                f"different sample rates, {self.audio_rate} then {video.sample_rate}"
            )
        self.audio = numpy.concatenate([self.audio, audio], axis=1)


def _copied_into(spill, source):
    """Copy one kept file into this run's spill under its own next name."""
    target = output_file_path(
        spill.output_dir, f"{spill.base_name}.segment-{len(spill.paths):03d}.mp4"
    )
    shutil.copyfile(source, target)
    spill.paths.append(target)
    return target


def _carry(config, last, hold):
    """(carry, raw carry) read off the last kept segment's decoded file."""
    if config.continuity != GUIDE_CONTINUITY:
        frame = last.frames[-1]
        if hold is not None:
            return hold.feed(frame), frame
        return frame, None
    count = config.guide_frames
    if len(last.frames) < count:
        raise ValueError(
            f"Chain {CHAIN_RESUME_FROM}: the last kept segment holds "
            f"{len(last.frames)} frames, fewer than the {count}-frame guide"
        )
    frames = last.frames[-count:]
    if not config.carry_audio:
        return _SegmentCarry(frames, None, None), None
    if last.audio is None:
        raise ValueError(
            f"Chain {CHAIN_RESUME_FROM}: a guide chain with carry_audio holds the "
            "previous segment's audio, and the last kept file has none"
        )
    audio = fit_codec_padding(
        numpy.asarray(last.audio, dtype=numpy.float32),
        len(last.frames),
        config.fps,
        last.sample_rate,
    )
    samples = frames_to_samples(count, config.fps, last.sample_rate)
    return _SegmentCarry(frames, audio[:, -samples:], last.sample_rate), None


def _capstone(config, first):
    """Segment 1's capstone frame, read off the first kept file: rendered
    frame -from_end, which the file holds less its tail trim."""
    if not config.capstone:
        return None
    index = len(first.frames) + config.tail_trim - config.capstone["from_end"]
    if not 0 <= index < len(first.frames):
        raise ValueError(
            f"Chain {CHAIN_RESUME_FROM}: the capstone frame is not in the first "
            "kept segment's file"
        )
    return first.frames[index]


def _resumed_warning(config, count, reference):
    message = f"Resumed {count} of {len(config.plan)} segments from {reference}"
    if config.continuity == GUIDE_CONTINUITY and config.carry_audio:
        message += (
            f"; the first resumed seam's audio guide is segment {count}'s "
            "decoded track (AAC, re-encoded, no latents) rather than a live "
            "carry"
        )
    return message


def failure_hint(spill):
    """The output reference of the first segment a failed chain spilled, or
    None when it spilled none or the file is not under the output root."""
    if spill is None or not spill.paths:
        return None
    root = output_root()
    try:
        path = validate_path(spill.paths[0], root)
    except SecurityError:
        return None
    relative = os.path.relpath(path, root).replace(os.sep, "/")
    return references.make_ref(references.OUTPUT, relative)


def warn_resumable(spill):
    """Tell the caller a failed chain's finished segments can be resumed
    from, naming the reference `resume_from` takes."""
    hint = failure_hint(spill)
    if hint is None:
        return
    emit_warning(
        f"The chain stopped after {len(spill.paths)} finished segment(s), "
        f"saved beside the take; set the chain's {CHAIN_RESUME_FROM} to "
        f"{hint} to render only the rest",
        kind="chain_resume_hint",
        segments=len(spill.paths),
        resume_from=hint,
    )
