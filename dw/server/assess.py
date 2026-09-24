"""Assess a finished output in the server process (#388).

`GET /api/gallery/{name}/assess` runs the assessment probes
(`dw/tasks/assess.py`) against one gallery file or asset without queueing
anything: the probes are CPU-only and read a file, so the route is a sync
`def` FastAPI runs in its threadpool, beside whatever job holds the GPU.

One request is one decode: the file is streamed once into a `Media` and
every applicable probe reads that. The shot boundaries are the ones the
caller's root recorded - the run manifest for an output, the sidecar
`keep_output` wrote for an asset (#393) - resolved here rather than by the
probe, which cannot know which workspace the file belongs to.

What comes back are places to look, not verdicts; nothing acts on a
finding (`dw/assessment_rules.py`).
"""

from ..tasks.assess import (
    read_media,
    seams_answer,
    shots_answer,
    sync_drift_answer,
)

PROBES = {
    "analyze_shots": shots_answer,
    "analyze_seams": seams_answer,
    "analyze_sync_drift": sync_drift_answer,
}


def unknown_probe(probe):
    """The 400 detail for a probe outside the whitelist, or None."""
    if probe is None or probe in PROBES:
        return None
    return f"Unknown probe {probe!r} - one of {', '.join(PROBES)}"


def _not_applicable(kind, media, records):
    """Why each probe cannot say anything about this file: {probe: why}."""
    if media is None:
        why = (
            "a still has no shots, seams or soundtrack to measure"
            if kind == "image"
            else "not a video or audio file"
        )
        return dict.fromkeys(PROBES, why)
    reasons = {}
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
        return {"probe": probe, **PROBES[probe](media, records, source)}

    answers = {
        name: run(media, records, source)
        for name, run in PROBES.items()
        if name not in reasons
    }
    body = {
        "shots_source": source,
        "findings": [
            found for answer in answers.values() for found in answer["findings"]
        ],
        "rules_applied": [
            rule for answer in answers.values() for rule in answer["rules_applied"]
        ],
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
