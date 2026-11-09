"""A chain resumed from an earlier run's kept segments (`resume_from`).

A chain run with `save_segments` writes each finished segment as a playable
mp4 (`chain_spill.SegmentSpill`) and a chain that dies leaves them on disk;
its failure warning (`chain_resume_hint`, `warn_resumable`) quotes the first
file's output reference. Naming that file in `resume_from` makes a new run of
the same chain take that run's segment set (`chain_repair.segment_files`, in
index order, complete) as its segments 1..k and render only k+1..N:

- Each iteration reads its own files. A previous_result fan-out runs one
  chain block once per iteration, and the spill numbers them
  `<prefix>.<iteration>.segment-NNN.mp4` with a counter on the step's
  Pipeline wrapper, which is new each job, so iteration i of the resumed run
  is iteration i of the dead one. The reference names one iteration's file
  (the one the hint quoted); every iteration takes the siblings with its own
  number beside it (`own_segments`). One with none there, or with a
  complete set (it finished) renders in full - the named one too, in a
  fan-out; a lone chain with every segment kept is refused - and says so in a `chain_resumed` warning with `kept: 0`.

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
  file (see the anchor and capstone approximations below). The kept segments are not checked again; with `check`, the first
  rendered segment is checked against the last kept one as usual.
- `look_hold` cannot be re-derived faithfully: `LookHold` learns its boost
  from each segment's rendered opening against the keyframe it was fed, and
  a kept file holds only the corrected frames. So the hold reads the scene
  from the first kept file, as segment 1 sets it, and is fed the last kept
  file's carry; the boost restarts at none.
- The kept segments' audio is laid on the timeline whole, end to end: each
  file holds only its span past the head trim, so the crossfade material of
  their own seams is gone, and each kept seam gets the declick fade the chain
  gives a seam with no head material (`equal_power_crossfade_join` with an
  empty head) rather than the `crossfade_ms` blend; the first rendered seam
  is joined as usual. All kept files carry audio or none does, and the
  rendered segments must agree with them (`check_rendered_audio`): a mixed
  set would leave the track shorter than the picture and offset from it, and
  a silent kept set under voiced segments means the earlier run was not this
  plan, so both are refused rather than padded with silence.
- The step generator restarts at the step's seed, so the rendered segments
  are not the renders the dead run would have made next.

Known approximations, none of them silent reuse:

- Under `carry_pick` the capstone frame is read off segment 1's file by its
  tail trim alone, so it is off by segment 1's unknown `picked_back`. Prefer
  not to resume a chain that sets `capstone` with `carry_pick`; the frame is
  close, not exact.
- `anchor` frames are segment 1's kept file, which lacks its tail trim
  (`tail_trim`, plus `picked_back` under `carry_pick`), while a live run's
  anchor is segment 1 as rendered, so a resumed anchor is shorter than the
  live one.
- A failure inside `load` draws a `chain_resume_hint` only when it had already
  accepted kept files into the spill (a resumable prefix), and the hint counts
  and names only those: a file `_Kept.add` refuses (its decode disagrees with
  its header, say) has its copy taken back out of the spill first. A refusal
  of the first file, or one before any copy (`check_plan`), has none to name,
  and the earlier run's files are the ones `resume_from` names.
- A spill's `-2` dedupe suffix (`...segment-000-2.mp4`, flat layout) is not
  matched by `SEGMENT_FILE`: moot while each job has its own run directory.
- The iteration number counts `run_chain` calls on one Pipeline wrapper; an
  in-step retry on the same wrapper would shift the numbers, which shows as a
  `chain_resumed` warning with `kept: 0` (a render in full), never as reuse.
- `resume_from` resolves a `latest` or `vN` run id as of before this run
  started, so iteration 0's copies into the new run never become the
  reference of iteration 1.
- The estimate (`kept_count`) prices every fan-out iteration at the named
  iteration's count; see `FANOUT_NOTE`.

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
from ..runs import is_run_id, output_root, resolve_output_reference
from ..security import PathTraversalError, SecurityError, validate_path
from ..shots import shot_record
from ..task_domains import frames_to_samples
from ..tasks.chain_repair import SEGMENT_FILE, segment_files
from ..tasks.joins import equal_power_crossfade_join
from ..tasks.video_utils import load_audio_video
from ..writers import output_file_path
from .chain_check import audio_dbfs, measure_frames, texture_base
from .chain_continuity import _SegmentCarry
from .chain_rules import CHAIN_RESUME_FROM, resume_reference_problem
from .h3_rules import ANCHOR_SEGMENT_1, GUIDE_CONTINUITY, KEYSTONE_SEGMENT_1

logger = logging.getLogger("dw")

# The estimate's note when `resume_from` names a file that is not there yet
UNRESOLVED_NOTE = "priced for every segment; resume_from is not resolvable at validate"

# The estimate's note when the resuming step fans out over previous_result
FANOUT_NOTE = (
    "the resume quote assumes every fan-out iteration resumes by the named "
    "iteration's segment count; an iteration whose files are missing, or that "
    "finished, renders in full and costs that many segments more"
)


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

    def check_rendered_audio(self, config, segment, audio):
        """Refuse a rendered segment that disagrees with the kept ones on
        audio: the joined track would be shorter than the picture and start
        at the wrong frame."""
        if config.source_audio is not None:
            return
        kept, rendered = self.audio is not None, audio is not None
        if kept == rendered:
            return
        raise ValueError(
            f"Chain {CHAIN_RESUME_FROM}: segment {segment.index + 1} rendered "
            + ("no audio" if kept else "audio")
            + f" and the {self.count} kept segment(s) carry "
            + ("audio" if kept else "none")
            + " - a resume never changes the plan, so resume with the "
            "settings the earlier run used"
        )


def resolved_reference(value, before=None):
    """The file a `resume_from` value names, validated: refused as a
    location unless it is an output reference, then resolved against the
    run's output root. `before`, the current run id, narrows `latest` to the
    runs that started before it."""
    problem = resume_reference_problem(value)
    if problem:
        raise PathTraversalError(problem)
    return resolve_output_reference(value, before=before)


def _current_run_id(spill):
    """The run id of the directory this run spills into, or None when the
    layout has none (flat)."""
    name = os.path.basename(os.path.normpath(spill.output_dir))
    return name if is_run_id(name) else None


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
    named = resolved_reference(config.resume_from, _current_run_id(spill))
    sources = own_segments(named, spill)
    if sources is None or (
        len(sources) >= len(config.plan) and not _names_iteration(named, spill)
    ):
        _warn_rendered_in_full(config, spill, sources)
        return None
    _check_stop_after(config, len(sources))
    check_plan(
        sources,
        config.plan,
        config.tail_trim,
        config.carry_pick - 1 if config.carry_pick else 0,
    )
    _check_kept_audio(config, sources)
    kept = _Kept(config)
    for path in sources:
        copy = _copied_into(spill, path)
        try:
            count = kept.add(path, copy)
        except Exception:
            # A refused file is no part of a resumable prefix: the hint a
            # failure here draws names only the files already accepted
            _discard_copy(spill, copy)
            raise
        spill.frame_count += count
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


def _check_stop_after(config, kept):
    """Refuse a probe's `stop_after` at or under the segments a resume keeps:
    it counts from segment 1, so this run would render nothing."""
    if config.stop_after is not None and kept >= config.stop_after:
        raise ValueError(
            f"Chain {CHAIN_RESUME_FROM} keeps {kept} segment(s) and stop_after "
            f"{config.stop_after} ends the take at segment {config.stop_after}: "
            f"nothing is left to render - clear stop_after to continue the "
            f"take to its {len(config.plan)} planned segments, or raise it"
        )


def _iteration_base(named):
    """(stem, iteration) of a spilled segment file's name, `<stem>.<n>`
    before `.segment-NNN.mp4`, or None for a name a spill did not make."""
    match = SEGMENT_FILE.match(os.path.basename(named))
    if match is None:
        return None
    stem, dot, number = match["base"].rpartition(".")
    if not dot or not number.isdigit():
        return None
    return stem, int(number)


def _names_iteration(named, spill):
    """Whether a finished set is refused rather than rendered in full: the
    reference names this spill's own iteration of a chain that does not fan
    out. A fan-out's finished iteration renders in full whichever iteration
    the reference names (#1307)."""
    if getattr(spill, "fan_out", False):
        return False
    base = _iteration_base(named)
    return base is None or base[1] == spill.iteration


def own_segments(named, spill):
    """This iteration's kept segment files beside the one `named` names: the
    siblings numbered with this spill's iteration, in index order, or None
    when it left none there. A name a spill did not make is taken whole."""
    base = _iteration_base(named)
    if base is None or base[1] == spill.iteration:
        return segment_files(named)
    stem, _ = base
    own = os.path.join(
        os.path.dirname(named), f"{stem}.{spill.iteration}.segment-000.mp4"
    )
    if not os.path.isfile(own):
        return None
    return segment_files(own)


def _warn_rendered_in_full(config, spill, sources):
    """Say an iteration took nothing from the reference: none of its files
    are there, or it finished in the earlier run."""
    why = (
        "left no segment files beside it"
        if sources is None
        else "finished in the earlier run"
    )
    emit_warning(
        f"Chain iteration {spill.iteration} {why}, so it rendered all "
        f"{len(config.plan)} segments in full rather than resuming from "
        f"{config.resume_from}",
        kind="chain_resumed",
        kept=0,
        segments=len(config.plan),
        resume_from=config.resume_from,
    )


def _check_kept_audio(config, sources):
    """Refuse a kept set where some files carry audio and some do not, read
    off the headers before anything is copied."""
    if config.source_audio is not None:
        return
    voiced = [
        (probe_metadata(path) or {}).get("sample_rate") is not None for path in sources
    ]
    if all(voiced) or not any(voiced):
        return
    named = {True: [], False: []}
    for path, has in zip(sources, voiced):
        named[has].append(os.path.basename(path))
    raise ValueError(
        f"Chain {CHAIN_RESUME_FROM}: the kept segments carry audio in "
        f"{', '.join(named[True])} and none in {', '.join(named[False])} - "
        "laid end to end the track would be shorter than the picture and "
        "offset from it"
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
        header = (probe_metadata(copy) or {}).get("frame_count")
        if header is not None and header != count:
            # check_plan went by the header; the timeline goes by the decode
            raise ValueError(
                f"Chain {CHAIN_RESUME_FROM}: kept file {os.path.basename(source)} "
                f"decodes to {count} frames but its header says {header}"
            )
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
        self._record_level(shot, video)
        if self.config.keep_segments:
            shot["segment_file"] = os.path.basename(copy)
        self.shots.append(shot)
        self.frame_count += count
        logger.info(f"Chain resume: kept {source} as {copy}")
        if self.first is None:
            self.first = video
        self.last = video
        return count

    def _record_level(self, shot, video):
        """The kept file's generated audio level on its shot, and the silent
        floor warning the rendered segments get (they were measured in the
        dead run, from the unencoded audio; this reads the file's AAC)."""
        if self.config.audio_held or video.audio is None:
            return
        level = audio_dbfs(video.audio)
        if level is None:
            return
        shot["audio_dbfs"] = level
        floor = self.config.audio_floor
        if floor is None or level >= floor:
            return
        emit_warning(
            f"Chain kept segment {len(self.shots) + 1}/{len(self.config.plan)} "
            f"is silent: its audio reads {level} dBFS, under the {floor:g} dBFS "
            "floor",
            kind="chain_segment_silent",
            segment=len(self.shots) + 1,
            audio_dbfs=level,
            floor=floor,
        )

    def _join_audio(self, video, count):
        """Lay a kept file's own track end to end on the joined one, declicked
        at the seam - the generated audio only: a held or matched track is
        the chain's own."""
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
        # No head material survives in a kept file: the chain's own seam
        # treatment for that case, the declick fade, not the crossfade
        self.audio = equal_power_crossfade_join(
            self.audio, audio[:, :0], audio, self.audio_rate, self.config.crossfade_ms
        )


def _copied_into(spill, source):
    """Copy one kept file into this run's spill under its own next name."""
    target = output_file_path(
        spill.output_dir, f"{spill.base_name}.segment-{len(spill.paths):03d}.mp4"
    )
    shutil.copyfile(source, target)
    spill.paths.append(target)
    return target


def _discard_copy(spill, copy):
    """Take a refused kept file's copy back out of this run's spill."""
    spill.paths.remove(copy)
    try:
        os.remove(copy)
    except OSError:
        logger.warning(f"Chain resume: could not remove refused copy {copy}")


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
