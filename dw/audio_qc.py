"""The audio-level checks a saved deliverable gets: headroom before the write,
and, from one decode of the file just written, clipping and near-silence.

Each speaks through `emit_warning` and decides nothing - what level a
deliverable should sit at is the workflow's to decide. `Result.save_artifact`
runs them in a fixed order (see its post-write block).
"""

import logging
import os

import numpy

from .dsp import dbfs
from .events import emit_warning

logger = logging.getLogger("dw")


# A deliverable this close to full scale has no headroom left: an mp3 or AAC
# encode of it decodes above 0 dBFS and clips, which is why a track measured
# at +1.3 dBFS in the gallery can have been written from samples that never
# exceeded 1.0. Same ceiling `match_levels` holds a gain to
# (MATCH_CEILING_DBFS in dw/tasks/joins.py)
HEADROOM_WARN_DBFS = -0.5


def _peak_dbfs(waveform):
    """The loudest sample of anything saveable as audio, in dBFS, or None
    when it cannot be measured cheaply (no samples, a lazily-decoded
    reader, a shape nothing here recognises)."""
    try:
        if hasattr(waveform, "detach"):
            waveform = waveform.detach().float().cpu().numpy()
        samples = numpy.asarray(waveform)
        if samples.size == 0 or not numpy.issubdtype(samples.dtype, numpy.number):
            return None
        peak = float(numpy.abs(samples).max())
    except Exception:
        logger.debug("Could not measure the peak of a saved track", exc_info=True)
        return None
    return dbfs(peak)


def warn_without_headroom(waveform, file_name, emit=True, lossless=False):
    """Say when the soundtrack about to be written is at or over full scale.

    A clipped deliverable is invisible to the consumer this server is built
    for: the job succeeds, and an agent that cannot listen has `peak_dbfs`
    and no rule to read it against - `get_gallery_metadata` teaches the
    near-silent end of the range and said nothing about the other one (#158).
    A warning rather than a change to the mix: what level a deliverable
    should sit at is the workflow's to decide, and `normalize_audio` is the
    step that decides it.

    `emit=False` measures and returns the predicted peak without emitting
    the warning - used for a video mux, where the caller holds the emission
    until the post-encode probe (`warn_if_written_above_full_scale`) has a
    ground-truth answer, and only falls back to this prediction if that
    probe could not measure the written file at all (#174 amendment).

    `lossless=True` is a plain wav/aiff/flac save: soundfile writes those as
    integer PCM by default, which clips a sample outside [-1, 1] immediately
    rather than merely risking it on some later lossy encode (#295) - the
    message says so, and the caller does not suppress the post-write
    ground-truth check for this file the way it does for an mp3/ogg/opus save.
    """
    peak = _peak_dbfs(waveform)
    if peak is None or peak < HEADROOM_WARN_DBFS:
        return None
    if emit:
        risk = (
            "the write itself clips samples above full scale to 0 dBFS - the "
            "file just written is already clipped"
            if lossless
            else "an mp3 or AAC encode of it decodes above 0 dBFS and clips"
        )
        emit_warning(
            f"The soundtrack written to {file_name} peaks at {peak:+.1f} dBFS, "
            f"which leaves no headroom below full scale - {risk}. Add a "
            f"'normalize_audio' step (peak_dbfs: -1) before the step that "
            f"saves it, or 'match_levels' on the join that made it.",
            kind="audio_no_headroom",
            file=file_name,
            peak_dbfs=round(peak, 2),
        )
    return peak


# The written file, decoded, is the only measurement that is the consumer's
# own. `warn_without_headroom` measures the waveform handed to the writer,
# and the encoder is downstream of that: a track normalized to exactly
# -1.0 dBFS came back out of an AAC mux at +0.94, so the deliverable of a
# clean default run of `music-video` was above full scale and nothing
# warned, because the number the check read was -1.0 (#158, #159, #161)
CLIPPED_WARN_DBFS = 0.0


# Distinguishes "no info supplied, probe it yourself" from "already probed,
# and it came back empty/unmeasurable" - a caller that decoded the file once
# (save_artifact, sharing one pass between this and warn_if_written_near_silent)
# passes the dict (or None on failure) through instead of paying for a second
# full decode of the same file (#262 - two independent probes, each a full
# audio+video decode, doubled the 'saving' phase's wall clock)
_UNPROBED = object()


def _probe_written_media(output_path):
    """Decode the just-written file once. `None` on any failure to probe."""
    try:
        from .media import probe_media

        return probe_media(output_path) or {}
    except Exception:
        logger.debug(
            f"Could not measure the written level of {output_path}", exc_info=True
        )
        return None


def warn_if_written_above_full_scale(
    output_path, already_warned=False, info=_UNPROBED, lossless=False
):
    """Say when the file just written decodes above full scale.

    The overshoot a lossy encode adds is material-dependent - about 0.1 dB
    on the mp3s measured for #159 and about 1.9 dB on the AAC mux of the
    same song - so no amount of headroom chosen up front can be known to be
    enough. Reading the file back is what closes that: whatever the encoder
    did, this is the number a consumer's decoder will see.

    Silent when `warn_without_headroom` has already spoken for this file -
    but only for a plain audio save into a lossy container (mp3/ogg/opus).
    That suppression assumed the encoder only ever adds overshoot, which held
    for the mp3s #159/#161 measured but is backwards for an H3 video mux:
    #174 measured that family's AAC mux landing *under* full scale after
    starting over it, so the pre-encode warning was right and the caller's
    suppression hid the post-encode check that would have said so. It is
    also wrong for a wav/aiff/flac save: soundfile's default integer PCM
    subtype clips a sample outside [-1, 1] at write time, so the pre-write
    prediction and the post-write ground truth are two different facts about
    the same file rather than a duplicate of one, and both are worth reading
    (#295). The caller decides which case applies (`content_type`), not this
    function.

    `info` lets a caller that already decoded the file (`_probe_written_media`)
    hand the result in rather than have this probe it again. Best effort
    either way: a file that will not probe is not a level problem, and a
    deliverable that is already written is not worth failing a finished run
    over.
    """
    if already_warned:
        return None
    if info is _UNPROBED:
        info = _probe_written_media(output_path)
    if info is None:
        return None
    peak = info.get("peak_dbfs")
    if peak is None or peak < CLIPPED_WARN_DBFS:
        return peak
    name = os.path.basename(output_path)
    if lossless:
        cause = (
            "The write itself clipped it - there was no headroom left below full scale"
        )
    else:
        cause = "The encode adds its own overshoot on top of the level it was handed"
    emit_warning(
        f"{name} decodes at {peak:+.2f} dBFS - above full scale, so it "
        f"clips on playback. {cause}, so the fix is more headroom before "
        f"the file is written: a 'normalize_audio' step at 'peak_dbfs: -3' "
        f"ahead of the step that saves it. A mux into a video needs more "
        f"of it than an audio file does.",
        kind="audio_clipped",
        file=name,
        peak_dbfs=round(peak, 2),
    )
    return peak


# The mean level get_gallery_metadata's own hint text already teaches as
# near-silent (#158) - mirrored here as the check nothing was actually
# running: a succeeded job could hand back a track this quiet with
# warnings: [] (#261)
NEAR_SILENT_WARN_DBFS = -40.0


# Above this, a peak means "quiet but not empty" rather than "check for a
# defect" - s02's -18.5 dBFS peaks (an ambience-only shot) clear it, #261's
# -68.7 dBFS Bark clip and S-F077's -60 dBFS normalize do not (#358)
NEAR_SILENT_QUIET_NOT_EMPTY_DBFS = -30.0


def warn_if_written_near_silent(
    output_path, already_warned=False, info=_UNPROBED, source_already_quiet=False
):
    """Say when the file just written decodes as near-silent.

    The other end of the range `warn_if_written_above_full_scale` guards:
    `get_gallery_metadata` already taught mean_dbfs below -40 on a track
    that should be full as a near-silent render, but nothing emitted a
    warning for it at save time, so a job could succeed and hand back a
    clip nobody could hear with `warnings: []` (#261).

    `info` is the same shared-probe handoff `warn_if_written_above_full_scale`
    takes. Best effort either way: a file that will not probe is not a
    level problem, and a deliverable that is already written is not worth
    failing a finished run over.

    `source_already_quiet` is set by a pass-through task (slice_audio) whose
    *source* material was already this quiet going in - a slice of room
    tone is not a defect the slice introduced, and the warning's own wording
    ("check the step that generated it... an unintended near-zero gain
    upstream") is aimed at a step that could plausibly have caused the
    level, which a plain cut out of an already-quiet recording did not (#309).

    The trigger stays mean-only (#358): a wordless, ambience-only shot (paws,
    husks scraping, water) reads a low mean with real peaks - -54 dBFS mean,
    -18 to -31 dBFS peaks - and split perfectly with dialogue presence, not
    with silence, costing an investigation every run. The fix is the message,
    not the gate: a peak above `NEAR_SILENT_QUIET_NOT_EMPTY_DBFS` says so
    plainly rather than reusing the "check the step that generated it"
    wording aimed at a genuinely empty render (#261's -68.7 dBFS, S-F077's
    -60 dBFS).
    """
    if already_warned or source_already_quiet:
        return None
    if info is _UNPROBED:
        info = _probe_written_media(output_path)
    if info is None:
        return None
    mean = info.get("mean_dbfs")
    if mean is None or mean >= NEAR_SILENT_WARN_DBFS:
        return mean
    name = os.path.basename(output_path)
    peak = info.get("peak_dbfs")
    if peak is not None and peak >= NEAR_SILENT_QUIET_NOT_EMPTY_DBFS:
        message = (
            f"{name} decodes at a mean level of {mean:+.2f} dBFS but peaks "
            f"at {peak:+.2f} dBFS: quiet overall, not empty. Expected for "
            f"an ambience-only shot; a concern only if this was meant to "
            f"carry speech or music."
        )
    else:
        message = (
            f"{name} decodes at a mean level of {mean:+.2f} dBFS - near-silent "
            f"for a deliverable meant to be heard. Check the step that "
            f"generated it: an empty or malformed prompt, a source model that "
            f"produced no meaningful audio for this input, or an unintended "
            f"near-zero gain upstream ('normalize_audio' or 'match_levels')."
        )
    fields = {"kind": "audio_near_silent", "file": name, "mean_dbfs": round(mean, 2)}
    if peak is not None:
        fields["peak_dbfs"] = round(peak, 2)
    emit_warning(message, **fields)
    return mean
