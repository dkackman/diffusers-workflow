"""Assess a finished output in the server process (#388).

`GET /api/gallery/{name}/assess` runs the assessment probes
(`dw/tasks/assess.py`) against one gallery file or asset without queueing
anything: the probes are CPU-only and read a file, so the route is a sync
`def` FastAPI runs in its threadpool, beside whatever job holds the GPU.

One request is one decode: the file is streamed once into a `Media` and
every applicable probe reads that - bar `analyze_held_frames`, which decodes
the picture a second time for its cut detector, and only when the file
holds a frame for a cut to explain. The shot boundaries are the ones the
caller's root recorded - the run manifest for an output, the sidecar
`keep_output` wrote for an asset (#393) - resolved here rather than by the
probe, which cannot know which workspace the file belongs to.

What comes back are places to look, not verdicts; nothing acts on a
finding (`dw/assessment_rules.py`).
"""

import json

from .. import audio_qc, dsp
from ..assessment_rules import finding, sort_findings
from ..tasks.assess import (
    held_frames_answer,
    read_media,
    seams_answer,
    shots_answer,
    sync_drift_answer,
)

PROBES = {
    "analyze_shots": shots_answer,
    "analyze_seams": seams_answer,
    "analyze_sync_drift": sync_drift_answer,
    "analyze_held_frames": held_frames_answer,
}

# The probes whose answer takes `detail` (held_frames' d series)
DETAILED = ("analyze_held_frames",)


def _run(name, media, records, source, detail):
    if name in DETAILED:
        return PROBES[name](media, records, source, detail=detail)
    return PROBES[name](media, records, source)


def unknown_probe(probe):
    """The 400 detail for a probe outside the whitelist, or None."""
    if probe is None or probe in PROBES:
        return None
    return f"Unknown probe {probe!r} - one of {', '.join(PROBES)}"


def _dedupe_findings(findings):
    """Findings with exact duplicates dropped, order preserved.

    `_shot_span_findings` (`dw/tasks/assess.py`) is shared by the shot
    probes and raises the identical finding from each one that ran, so a
    file with a span overrun listed it three times in the compact merge
    (#427) - one true finding, not three.
    """
    seen = set()
    deduped = []
    for found in findings:
        key = json.dumps(found, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(found)
    return deduped


def _not_applicable(kind, media, records):
    """Why each probe cannot say anything about this file: {probe: why}."""
    if media is None:
        if kind != "image":
            return dict.fromkeys(PROBES, "not a video or audio file")
        reasons = dict.fromkeys(
            PROBES, "a still has no shots, seams or soundtrack to measure"
        )
        reasons["analyze_held_frames"] = "a still has no frames to repeat"
        return reasons
    reasons = {}
    if media.thumbs is None:
        reasons["analyze_held_frames"] = "no picture - an audio-only file"
    if media.audio is None:
        reasons["analyze_shots"] = "no audio track"
        reasons["analyze_sync_drift"] = "no audio track"
    elif media.thumbs is None:
        reasons["analyze_sync_drift"] = "no picture to sync against"
    if not records or len(records) < 2:
        reasons["analyze_seams"] = (
            "no shot boundaries recorded for this file - analyze_seams "
            "needs two or more shots"
        )
    return reasons


LEVEL_RULES = ("full_scale", "near_silent")


def _file_level(media):
    """The peak and mean (rms) dBFS of a Media's whole soundtrack, as
    `level_findings` reads them from the gallery's probe."""
    return {
        "peak_dbfs": dsp.level_dbfs(media.audio, "peak"),
        "mean_dbfs": dsp.level_dbfs(media.audio, "rms"),
    }


def assess(path, kind, shots, probe=None, detail=False):
    """Run the applicable probes over one file.

    path: the validated file; kind: its MEDIA_KINDS entry ("video", "audio",
    "image", ...); shots: the shot records its root recorded, or None.

    With `probe`, that probe's full answer. Otherwise every applicable
    probe's findings merged, with `rules_applied`, `rules_skipped` and
    `not_applicable`; `detail` adds each probe's full answer under `probes`.
    """
    media = read_media(path) if kind in ("audio", "video") else None
    records = [dict(shot) for shot in shots] if shots else None
    source = "manifest" if records else "none"
    reasons = _not_applicable(kind, media, records)

    if probe is not None:
        # Asked by name: a probe the file cannot feed at all says why; one
        # it can (a shotless seam pass included) answers in full, its own
        # rules_skipped saying what could not be measured
        blocked = reasons.get(probe)
        if media is None or (blocked and probe != "analyze_seams"):
            return {"probe": probe, "not_applicable": {probe: blocked}}
        return {"probe": probe, **_run(probe, media, records, source, detail)}

    answers = {
        name: _run(name, media, records, source, detail)
        for name in PROBES
        if name not in reasons
    }
    # The whole-file level rules (full_scale, near_silent) are the gallery's
    # too (#807); a file with no soundtrack has no level to read
    level = media is not None and media.audio is not None
    level_found = level_findings(_file_level(media), records) if level else []
    body = {
        "shots_source": source,
        "findings": sort_findings(
            _dedupe_findings(
                level_found
                + [found for answer in answers.values() for found in answer["findings"]]
            )
        ),
        "rules_applied": (list(LEVEL_RULES) if level else [])
        + [rule for answer in answers.values() for rule in answer["rules_applied"]],
        "rules_skipped": [
            {"probe": name, **skipped}
            for name, answer in answers.items()
            for skipped in answer["rules_skipped"]
        ],
        "not_applicable": reasons,
    }
    if detail:
        body["probes"] = answers
    return body


__all__ = ["PROBES", "assess", "unknown_probe"]


def level_findings(media, shots=None):
    """The level problems a probed file's soundtrack shows, as findings:
    a peak at full scale (an encode may clip it) and a near-silent mean
    (info when real peaks say quiet-not-empty, as an ambience-only shot
    reads). The thresholds are dw/audio_qc.py's, read at call time, so
    the gallery and the write-time warnings cannot disagree."""
    if not media:
        return []
    found = []
    peak, mean = media.get("peak_dbfs"), media.get("mean_dbfs")
    if peak is not None and peak >= audio_qc.CLIPPED_WARN_DBFS:
        rule = {
            "name": "full_scale",
            "severity": "warn",
            "threshold": audio_qc.CLIPPED_WARN_DBFS,
            # The file measured is the written one, so past full scale it
            # already clips; the fix is warn_if_written_above_full_scale's
            "says": "at or above full scale, so it clips on playback - a "
            "'normalize_audio' step at 'peak_dbfs: -3' ahead of the step that "
            "saves it; a mux into a video needs more headroom than that (the job's "
            "'audio_clipped' warning names how much). For a "
            "file already written, run the 'relevel-clip' template on it: no "
            "re-render",
        }
        found.append(finding(rule, peak, None))
    if (
        mean is not None
        and mean < audio_qc.NEAR_SILENT_WARN_DBFS
        and not audio_qc.all_shots_silent_by_design(shots)
    ):
        quiet_not_empty = (
            peak is not None and peak >= audio_qc.NEAR_SILENT_QUIET_NOT_EMPTY_DBFS
        )
        rule = {
            "name": "near_silent",
            "severity": "info" if quiet_not_empty else "warn",
            "threshold": audio_qc.NEAR_SILENT_WARN_DBFS,
            "says": (
                "quiet overall but with real peaks - expected for an "
                "ambience-only track"
                if quiet_not_empty
                else "near-silent for a track meant to be heard; check the "
                "step that made it, or raise it with normalize_audio"
            ),
        }
        found.append(finding(rule, mean, None))
    return sort_findings(found)
