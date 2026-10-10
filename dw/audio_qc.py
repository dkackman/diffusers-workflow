"""The audio-level checks a saved deliverable gets: headroom before the write,
and, from one decode of the file just written, clipping and near-silence.

Each speaks through `emit_warning` and decides nothing - what level a
deliverable should sit at is the workflow's to decide. `check_written_media`
runs the post-write ones in a fixed order; `Result.save_artifact` calls it
once a write succeeds.
"""

import logging
import os

import numpy

from .dsp import dbfs
from .content_types import LOSSY_AUDIO_CONTENT_TYPES
from .events import emit_log, emit_warning
from .media_types import AUDIO_FIT_TOLERANCE_SECONDS

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
        f"ahead of the step that saves it (a template's intermediate shot has "
        f"no such step to edit: set the join's 'match_levels' instead). A "
        f"mux into a video needs more of it than an audio file does. For a "
        f"file already written, the 'relevel-clip' template re-levels it "
        f"without a re-render.",
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


def remeasure_shots_after_mux(artifact, probed_info, output_path, video_fps):
    """Re-measure a joined video's shot map against what its file decodes to.

    `video_fps` is called only when the video carries shots and the probe
    read an audio stream - it emits `fps_mismatch` on every call, so the call
    count is part of the behaviour.
    """
    # The shot map (#426) was re-measured against the fitted
    # in-memory track before this file was even encoded - a
    # prediction, not the file's own ground truth. A lossy mux can
    # still trim or pad past that (AAC's frame alignment cost the
    # #426 repro 29-30 samples on top of what fitting alone
    # accounted for), so once the file is probed the shots are
    # re-measured again against what actually decodes from it - the
    # same length assess.py's read_media() trims audio to
    # (`audio_stream_seconds`, the audio *stream's* own reported
    # duration - not the container's `duration_seconds`, which can
    # disagree with it by a handful of samples on a lossy mux and
    # would leave a residual overrun the probe still reports).
    if not (
        getattr(artifact, "shots", None)
        and probed_info
        and probed_info.get("audio_stream_seconds") is not None
        and probed_info.get("sample_rate")
    ):
        return
    from .shots import measured_num_samples

    written_samples = int(
        round(probed_info["audio_stream_seconds"] * probed_info["sample_rate"])
    )
    measured_num_samples(artifact.shots, written_samples)
    frames = getattr(artifact, "frames", None)
    frame_count = 0 if frames is None else len(frames)
    fps = video_fps(artifact)
    if not (frame_count and fps):
        return
    expected_samples = int(round(frame_count / fps * probed_info["sample_rate"]))
    shortfall = expected_samples - written_samples
    # Only a residual the save-time fit (fit_codec_padding)
    # would have padded: past its tolerance the track was
    # left at its own length, audio_video_length_mismatch
    # already names that gap, and "the mux trimmed it" would
    # misexplain it.
    tolerance = AUDIO_FIT_TOLERANCE_SECONDS * probed_info["sample_rate"]
    name = os.path.basename(output_path)
    if 0 < shortfall < probed_info["sample_rate"] / fps:
        # Under a frame: the encoder's alignment on every
        # joined deliverable, not something a caller can act
        # on - logged, with the shots already re-measured
        emit_log(
            f"{name}'s soundtrack "
            f"decodes {shortfall} sample(s) short of its "
            f"{frame_count}-frame grid after muxing; the shot "
            "map is measured against what it decodes to",
            file=name,
            shortfall_samples=shortfall,
        )
    elif 0 < shortfall <= tolerance:
        emit_warning(
            f"{name}'s soundtrack decodes "
            f"{shortfall} sample(s) short of its {frame_count}-frame "
            f"grid after muxing, even though it was padded to the "
            f"grid before encoding - the mux itself (commonly AAC's "
            f"frame alignment) trimmed it further. The shot map has "
            f"been re-measured against what the file actually "
            f"decodes to, so it stays accurate, but a consumer "
            f"reading exact sample counts should expect this small "
            f"residual gap.",
            kind="joined_audio_short_after_mux",
            file=name,
            shortfall_samples=shortfall,
            written_samples=written_samples,
            expected_samples=expected_samples,
        )


def written_peak_already_warned(content_type, consumed_by_normalizer, headroom_warned):
    """Whether the pre-write headroom check already spoke for this file.

    Only a plain audio save suppresses the post-write check; a lossless one
    only when a normalizer consumed it, a lossy one also when the pre-write
    warning fired. A video gets the ground-truth check, unless a normalizer
    or a level-matching join consumes it: that step resets the level it
    ships at, so the file's own written peak is not the deliverable's (#671).
    """
    if not content_type.startswith("audio"):
        return consumed_by_normalizer
    if content_type not in LOSSY_AUDIO_CONTENT_TYPES:
        return consumed_by_normalizer
    return headroom_warned or consumed_by_normalizer


def warn_held_prediction(output_path, predicted_peak_dbfs):
    """Speak the pre-encode headroom prediction for a muxed soundtrack the
    probe could not measure, so the warning it was held for is not lost."""
    emit_warning(
        f"The soundtrack written to {os.path.basename(output_path)} "
        f"was predicted to peak at {predicted_peak_dbfs:+.1f} "
        f"dBFS before encoding, and the written file could not be "
        f"re-measured to confirm whether the mux corrected it - add "
        f"a 'normalize_audio' step (peak_dbfs: -1) before the step "
        f"that saves it, or 'match_levels' on the join that made it.",
        kind="audio_no_headroom",
        file=os.path.basename(output_path),
        peak_dbfs=round(predicted_peak_dbfs, 2),
    )


def check_written_media(
    output_path,
    artifact,
    content_type,
    *,
    video_fps,
    consumed_by_normalizer,
    headroom_warned,
    predicted_peak_dbfs,
):
    """The post-write checks on an audio or video file, in their fixed order:
    probe once, shot re-measure, clipping, the held prediction, near-silence.

    `video_fps` is the saving `Result`'s bound method, called by
    `remeasure_shots_after_mux` only.
    """
    # One decode pass shared between the checks below (#262) -
    # each used to probe the file independently, which for a video
    # is a full audio+video decode and doubled the 'saving' phase's
    # wall clock for no second answer
    probed_info = _probe_written_media(output_path)
    is_video = content_type.startswith("video")
    if is_video:
        remeasure_shots_after_mux(artifact, probed_info, output_path, video_fps)
    written_peak = warn_if_written_above_full_scale(
        output_path,
        already_warned=written_peak_already_warned(
            content_type, consumed_by_normalizer, headroom_warned
        ),
        info=probed_info,
        lossless=content_type.startswith("audio")
        and content_type not in LOSSY_AUDIO_CONTENT_TYPES,
    )
    # A video's pre-encode prediction was held rather than emitted
    # (#174 amendment): the post-encode probe is the ground truth,
    # so a clean or genuinely-clipped result each get exactly one
    # answer - nothing here, or `audio_clipped` from the probe
    # itself. The only time the held prediction is worth anything is
    # when the probe could not measure the file at all, in which
    # case it is the one signal available and is surfaced late
    # rather than dropped silently
    #
    # Note the gap this leaves: a written peak between
    # HEADROOM_WARN_DBFS (around -0.5) and CLIPPED_WARN_DBFS (0.0)
    # produces no warning here - the post-encode probe only speaks
    # when the file is genuinely at or over full scale, so a mux
    # that predicted risk but measured merely close-but-clean says
    # nothing. That is the file's own ground truth, not a threshold
    # bug.
    #
    # A video a normalizer or level-matching join consumes skips both:
    # that step resets the level it ships at, so neither the written
    # peak nor the prediction is the deliverable's (#671)
    if (
        is_video
        and headroom_warned
        and not consumed_by_normalizer
        and written_peak is None
    ):
        warn_held_prediction(output_path, predicted_peak_dbfs)
    source_mean_dbfs = getattr(artifact, "source_mean_dbfs", None)
    warn_if_written_near_silent(
        output_path,
        info=probed_info,
        source_already_quiet=(
            source_mean_dbfs is not None and source_mean_dbfs < NEAR_SILENT_WARN_DBFS
        ),
    )
