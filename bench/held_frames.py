"""Held-frame probe over H3 takes (#1527 stage A, #1551): does a take repeat
every 2nd or 4th frame (12 fps shown as 24), and how often, in Don's own work?

The measure (#1527 plan v1, *Design -> The measure*):

- d_i is the mean absolute difference of the 64x36 grey thumbnails (0-255) of
  frames i and i-1 (`dw.tasks.assess.read_media`), for i >= 1.
- Frame i is **held** when d_i < eps and max(d_{i-1}, d_{i+1}) >=
  `TYPICAL_DELTA_FLOOR` (2.0): it repeats its predecessor while the picture
  around it is clearly moving. A static shot holds nothing.
- Frames within one frame of a hard cut (`dw.tasks.scene_cuts.content_cuts`)
  don't count, so a cut's own frames are never a hold.
- **Clusters** are runs of holds whose gaps are <= `CLUSTER_GAP` frames, of at
  least `CLUSTER_MIN` holds, reported as start and end seconds with the count.

Beside the rule, every file reports its **hold runs**: runs of consecutive
d < eps bounded on both sides by a clearly moving d. A run of 1 is a frame shown
twice; a run of 3 is a frame shown four times. The rule as written counts the
edges of a run only (a run's middle frames have still neighbours), so the run
lengths, not the held count, say whether holds come in twos or fours.

    python bench/held_frames.py scan WORKSPACE_ROOT [--workspace NAME ...]
        [--out held-frames.json] [--jobs N] [--all-video]
    python bench/held_frames.py file VIDEO [VIDEO ...]
    python bench/held_frames.py report held-frames.json

`scan` walks each named workspace's `outputs/` read-only - it opens files for
reading and writes only `--out` (default `/tmp/held-frames.json`), never inside
a workspace. It lists every video; a video is an H3 output when its run's
realized `workflow.json`, or a sub-workflow it names, loads `MiniMaxAI/MiniMax-H3`.
Only H3 videos are measured unless `--all-video`. `report` prints the record's
tables from the JSON, so a rerun's numbers can be compared line for line.
`file` measures loose files (a scratch copy made choppy with ffmpeg, say).
"""

import argparse
import concurrent.futures
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

EPSILONS = (0.5, 1.0, 1.5)
MOVING = 2.0  # dw.tasks.assess.TYPICAL_DELTA_FLOOR; checked equal at import below
CUT_GUARD = 1
CLUSTER_GAP = 4
CLUSTER_MIN = 2
PERCENTILES = (10, 25, 50, 75, 90)
GATE_HELD = 7  # plan v1, Gate after A
STATIC_SHARE = 0.1  # below this moving share a take is near-static

# Don's workspaces, counted as demand (plan v1, Corpus)
DEMAND = (
    "known-issues",
    "h3-quality",
    "constitution-walk",
    "constitution-walk-2",
    "constitution-walk-3",
    "inner-space",
    "soviet-anthem",
    "russian-anthem",
    "shedding-skin-quartet",
    "acorn-wars",
    "sci0fi",
    "default",
)
NOT_DEMAND = ("qa-", "regression-", "drill-", "probe-")

H3_MODEL = "MiniMaxAI/MiniMax-H3"
H3_REPO = re.compile(r'"MiniMaxAI/MiniMax-H3"')
VIDEO_SUFFIXES = (".mp4", ".mov", ".webm", ".mkv")


# ------------------------------------------------------------------ the measure


def deltas(thumbs):
    """d: d[i] = mean |thumbs[i] - thumbs[i-1]| for i >= 1; d[0] is nan."""
    thumbs = np.asarray(thumbs, dtype=np.float32)
    d = np.full(len(thumbs), np.nan, dtype=np.float64)
    if len(thumbs) > 1:
        d[1:] = np.mean(np.abs(np.diff(thumbs, axis=0)), axis=(1, 2))
    return d


def excluded(frame_count, cuts, guard=CUT_GUARD):
    """Frames within `guard` of a cut frame (the frame where the picture
    jumps), plus frame 0 (it has no d)."""
    out = {0}
    for cut in cuts:
        out.update(range(max(0, cut - guard), min(frame_count, cut + guard + 1)))
    return out


def held(d, eps, skip=frozenset(), moving=MOVING):
    """The held frames: d_i < eps with a neighbouring d >= `moving`. A missing
    neighbour (either end) is not moving."""
    out = []
    n = len(d)
    for i in range(1, n):
        if i in skip or not d[i] < eps:
            continue
        before = d[i - 1] if i - 1 >= 1 else np.nan
        after = d[i + 1] if i + 1 < n else np.nan
        if np.nanmax([before, after, -1.0]) >= moving:
            out.append(i)
    return out


def clusters(frames, fps, gap=CLUSTER_GAP, minimum=CLUSTER_MIN):
    """Runs of held frame indices whose gaps (frames between two holds) are
    <= `gap`, of at least `minimum` holds: [{start, end, held, first, last}]
    with start/end in seconds."""
    runs, current = [], []
    for frame in sorted(frames):
        if current and frame - current[-1] - 1 > gap:
            runs.append(current)
            current = []
        current.append(frame)
    if current:
        runs.append(current)
    rate = fps or 24.0
    return [
        {
            "start": round(run[0] / rate, 3),
            "end": round(run[-1] / rate, 3),
            "held": len(run),
            "first": run[0],
            "last": run[-1],
        }
        for run in runs
        if len(run) >= minimum
    ]


def hold_runs(d, eps, skip=frozenset(), moving=MOVING):
    """Lengths of runs of consecutive d < eps that a d >= `moving` bounds on
    both sides, none of whose frames is skipped. 1 = a frame shown twice, 3 =
    shown four times. Also the spacing between successive runs' starts."""
    lengths, starts = [], []
    n = len(d)
    i = 1
    while i < n:
        if d[i] < eps:
            j = i
            while j + 1 < n and d[j + 1] < eps:
                j += 1
            bounded = (
                i - 1 >= 1
                and d[i - 1] >= moving
                and j + 1 < n
                and d[j + 1] >= moving
                and not any(k in skip for k in range(i, j + 1))
            )
            if bounded:
                lengths.append(j - i + 1)
                starts.append(i)
            i = j + 1
        else:
            i += 1
    spacing = [b - a for a, b in zip(starts, starts[1:])]
    return lengths, spacing


def measure(thumbs, fps, cuts):
    """Every number the record takes for one take's thumbnails and cuts."""
    d = deltas(thumbs)
    n = len(d)
    skip = excluded(n, cuts)
    counted = np.array([d[i] for i in range(1, n) if i not in skip])
    out = {
        "frames": n,
        "fps": fps,
        "cuts": sorted(cuts),
        "d_percentiles": (
            {str(p): round(float(np.percentile(counted, p)), 3) for p in PERCENTILES}
            if counted.size
            else {}
        ),
        "moving_ratio": (
            round(float(np.mean(counted >= MOVING)), 3) if counted.size else 0.0
        ),
        "by_eps": {},
    }
    for eps in EPSILONS:
        frames = held(d, eps, skip)
        lengths, spacing = hold_runs(d, eps, skip)
        out["by_eps"][str(eps)] = {
            "held": len(frames),
            "held_ratio": round(len(frames) / n, 4) if n else 0.0,
            "clusters": clusters(frames, fps),
            "run_lengths": dict(sorted(Counter(lengths).items())),
            "run_spacing": dict(sorted(Counter(spacing).most_common(6))),
        }
    return out


def measure_file(path):
    """read_media's thumbnails and content_cuts' cuts for one file."""
    from dw.media_readers import iter_rgb_frames
    from dw.tasks.assess import read_media
    from dw.tasks.scene_cuts import content_cuts

    media = read_media(str(path))
    if media.thumbs is None:
        return {"error": "no video stream"}
    cuts = [frame for frame, _ in content_cuts(iter_rgb_frames(str(path)))]
    return measure(media.thumbs, media.fps, cuts)


# ------------------------------------------------------------ the workspace walk


def _load_json(path):
    try:
        with open(path) as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def _resolve(value, variables):
    while isinstance(value, str) and value.startswith("variable:"):
        name = value[len("variable:") :]
        if name not in variables:
            return None
        value = variables[name]
    return value


def _h3_steps(workflow):
    """{step name: step} for the workflow's steps whose pipeline loads H3."""
    out = {}
    for step in (workflow or {}).get("steps") or []:
        pipeline = step.get("pipeline") if isinstance(step, dict) else None
        if not isinstance(pipeline, dict):
            continue
        model = (pipeline.get("from_pretrained_arguments") or {}).get("model_name")
        if _resolve(model, workflow.get("variables") or {}) == H3_MODEL:
            out[step.get("name")] = step
    return out


def _sub_workflow_is_h3(key, search):
    for root in search:
        for candidate in (root / f"{key}.json", root / key):
            if candidate.is_file():
                try:
                    return bool(H3_REPO.search(candidate.read_text()))
                except OSError:
                    return False
    return None


def run_facts(run_dir, search):
    """What a run directory says about its videos: is it H3, its step
    counts, and {file relative to run: (step, shots)}."""
    manifest = _load_json(run_dir / "manifest.json") or {}
    workflow = _load_json(run_dir / "workflow.json") or {}
    variables = dict(workflow.get("variables") or {})
    variables.update(manifest.get("arguments") or {})
    h3 = _h3_steps(workflow)
    subs = (manifest.get("workflow") or {}).get("sub_workflows") or {}
    h3_subs = [key for key in subs if _sub_workflow_is_h3(key, search)]
    unknown_subs = [key for key in subs if _sub_workflow_is_h3(key, search) is None]
    steps = set()
    for step in h3.values():
        count = _resolve(
            (step.get("pipeline", {}).get("arguments") or {}).get(
                "num_inference_steps"
            ),
            variables,
        )
        if isinstance(count, int):
            steps.add(count)
    if not steps and (h3_subs or h3):
        count = variables.get("num_inference_steps")
        if isinstance(count, int):
            steps.add(count)
    files = {}
    for entry in manifest.get("steps") or []:
        for name in entry.get("files") or []:
            files[name] = (entry.get("step"), entry.get("shots"))
    task_of = {
        step.get("name"): (step.get("task") or {}).get("command")
        for step in workflow.get("steps") or []
        if isinstance(step, dict) and step.get("task")
    }
    return {
        "h3": bool(h3 or h3_subs),
        "h3_unknown": bool(unknown_subs) and not (h3 or h3_subs),
        "workflow": (manifest.get("workflow") or {}).get("id") or workflow.get("id"),
        "sub_workflows": sorted(subs),
        "h3_steps": sorted(h3),
        "steps": sorted(steps),
        "files": files,
        "task_of": task_of,
        "run_id": manifest.get("run_id") or run_dir.name,
        "has_manifest": bool(manifest),
    }


def _base_step(name):
    return (name or "").split("@", 1)[0]


def mode_of(relative, facts):
    """single | chain segment | joined | cuts | derived | unlisted (the
    manifest does not name it) | no manifest (a run with only workflow.json)."""
    if ".segment-" in Path(relative).name:
        return "chain segment"
    if relative not in facts["files"]:
        return "unlisted" if facts["has_manifest"] else "no manifest"
    step, shots = facts["files"][relative]
    shots = shots or []
    chained = any(isinstance(s, dict) and s.get("segment_file") for s in shots)
    if (
        _base_step(step) in facts["h3_steps"]
        or _base_step(step) not in (facts["task_of"])
    ):
        # a pipeline step (or a composed sub-workflow step)
        return "joined" if chained and len(shots) > 1 else "single"
    if len(shots) > 1:
        return "joined" if chained else "cuts"
    return "derived"


def workspace_outputs(root, name):
    return root / "outputs" if name == "default" else root / name / "outputs"


def walk(root, names, search):
    """Every video under each workspace's outputs, with its run's facts."""
    rows = []
    for name in names:
        outputs = workspace_outputs(root, name)
        if not outputs.is_dir():
            print(f"warning: no outputs for workspace {name}", file=sys.stderr)
            continue
        facts_by_run = {}
        for dirpath, dirnames, filenames in os.walk(outputs):
            dirnames.sort()
            for filename in sorted(filenames):
                if not filename.lower().endswith(VIDEO_SUFFIXES):
                    continue
                path = Path(dirpath) / filename
                run_dir = next(
                    (
                        parent
                        for parent in [path.parent, *path.parents]
                        if (parent / "manifest.json").is_file()
                        or (parent / "workflow.json").is_file()
                    ),
                    None,
                )
                if run_dir is not None and outputs not in run_dir.parents:
                    run_dir = None
                if run_dir is None:
                    facts = None
                else:
                    if run_dir not in facts_by_run:
                        facts_by_run[run_dir] = run_facts(run_dir, search)
                    facts = facts_by_run[run_dir]
                rows.append(
                    {
                        "workspace": name,
                        "demand": name in DEMAND,
                        "gallery_name": str(path.relative_to(outputs)),
                        "path": str(path),
                        "run": facts["run_id"] if facts else None,
                        "workflow": facts["workflow"] if facts else None,
                        "sub_workflows": facts["sub_workflows"] if facts else [],
                        "h3": (facts["h3"] if facts else None),
                        "h3_unknown": facts["h3_unknown"] if facts else True,
                        "mode": (
                            mode_of(str(path.relative_to(run_dir)), facts)
                            if facts
                            else "unlisted"
                        ),
                        "steps": facts["steps"] if facts else [],
                    }
                )
    return rows


def _measure_row(row):
    try:
        return measure_file(row["path"])
    except Exception as error:  # a broken file is a row, not a crash
        return {"error": f"{type(error).__name__}: {error}"}


def scan(root, names, out, jobs, all_video, search):
    rows = walk(root, names, search)
    todo = [row for row in rows if row["h3"] or all_video]
    print(f"{len(rows)} videos, measuring {len(todo)}", file=sys.stderr)
    with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as pool:
        for done, (row, result) in enumerate(
            zip(todo, pool.map(_measure_row, todo, chunksize=1)), 1
        ):
            row.update(result)
            if done % 25 == 0:
                print(f"  {done}/{len(todo)}", file=sys.stderr)
    with open(out, "w") as handle:
        json.dump({"root": str(root), "workspaces": names, "rows": rows}, handle)
    print(f"wrote {out}", file=sys.stderr)


# ---------------------------------------------------------------------- report


def _max_cluster(row, eps):
    found = row["by_eps"][eps]["clusters"]
    return max((c["held"] for c in found), default=0)


def report(data, eps="1.0"):
    """The record's tables as markdown lines."""
    rows = data["rows"]
    lines = []
    listed = Counter((r["workspace"], bool(r["h3"])) for r in rows)
    lines.append("### Corpus: videos per workspace")
    lines.append("")
    lines.append(
        "| Workspace | Demand | H3 videos | Other videos | Measured | Errors |"
    )
    lines.append("|---|---|---|---|---|---|")
    for name in data["workspaces"]:
        mine = [r for r in rows if r["workspace"] == name]
        lines.append(
            f"| {name} | {'yes' if name in DEMAND else 'no'} | {listed[(name, True)]} "
            f"| {listed[(name, False)]} | {sum('by_eps' in r for r in mine)} "
            f"| {sum('error' in r for r in mine)} |"
        )
    measured = [r for r in rows if "by_eps" in r and r["demand"] and r["h3"]]
    lines.append("")
    lines.append(
        f"### Held frames per mode and schedule (demand H3 takes, n={len(measured)})"
    )
    lines.append("")
    lines.append(
        "| Mode | Steps | Takes | held=0 @0.5 | held=0 @1.0 | held=0 @1.5 "
        "| median held @1.0 | max held @1.0 | takes with a cluster @1.0 "
        "| takes >= 7 held @1.0 | d p50 median |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    groups = {}
    for r in measured:
        key = (r["mode"], ",".join(map(str, r["steps"])) or "?")
        groups.setdefault(key, []).append(r)
    for (mode, steps), group in sorted(groups.items()):
        counts = {
            e: [g["by_eps"][e]["held"] for g in group] for e in map(str, EPSILONS)
        }
        p50 = [g["d_percentiles"].get("50") for g in group if g["d_percentiles"]]
        lines.append(
            f"| {mode} | {steps} | {len(group)} "
            + " ".join(f"| {sum(c == 0 for c in counts[e])}" for e in counts)
            + f" | {np.median(counts[eps]):g} | {max(counts[eps])} "
            f"| {sum(bool(g['by_eps'][eps]['clusters']) for g in group)} "
            f"| {sum(c >= 7 for c in counts[eps])} "
            f"| {np.median(p50) if p50 else float('nan'):.2f} |"
        )
    lines.append("")
    lines.append(f"### Hold run lengths (demand H3 takes, eps {eps})")
    lines.append("")
    lines.append(
        "A run of 1 is a frame shown twice, 3 is a frame shown four times "
        "(runs bounded by moving frames on both sides)."
    )
    lines.append("")
    runs, spacing = Counter(), Counter()
    for r in measured:
        runs.update({int(k): v for k, v in r["by_eps"][eps]["run_lengths"].items()})
        spacing.update({int(k): v for k, v in r["by_eps"][eps]["run_spacing"].items()})
    lines.append("| Run length | Runs |")
    lines.append("|---|---|")
    for length, count in sorted(runs.items()):
        lines.append(f"| {length} | {count} |")
    lines.append("")
    lines.append("| Spacing between run starts (frames) | Count (top 6 per take) |")
    lines.append("|---|---|")
    for gap, count in sorted(spacing.items()):
        lines.append(f"| {gap} | {count} |")
    lines.append("")
    lines.append(f"### Highest-count takes (eps {eps})")
    lines.append("")
    lines.append(_take_table_header())
    ranked = sorted(
        measured,
        key=lambda r: (-r["by_eps"][eps]["held"], -_max_cluster(r, eps), r["path"]),
    )
    for r in ranked[:5]:
        lines.append(_take_row(r, eps))
    lines.append("")
    lines.append(f"### Nearest the 6 and 12 held marks (eps {eps})")
    lines.append("")
    lines.append(_take_table_header())
    near = sorted(
        (r for r in measured if r["by_eps"][eps]["held"] > 0),
        key=lambda r: (
            min(abs(r["by_eps"][eps]["held"] - 6), abs(r["by_eps"][eps]["held"] - 12)),
            r["path"],
        ),
    )
    shown = {r["path"] for r in ranked[:5]}
    for r in [r for r in near if r["path"] not in shown][:3]:
        lines.append(_take_row(r, eps))
    lines.append("")
    lines.append(f"### The gate: takes with >= 7 held frames (eps {eps})")
    lines.append("")
    lines.append(
        f"A take whose moving share (counted frames with d >= {MOVING:g}) is under "
        f"{STATIC_SHARE:g} is a near-static shot: its holds are the still frames "
        "either side of a rare jump, the static case the gate excludes."
    )
    lines.append("")
    lines.append(
        "| Workspace | Gallery name | Mode | Steps | Held | Moving share "
        "| Static | Run lengths | Spacing (frames: count) |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    gate = [r for r in ranked if r["by_eps"][eps]["held"] >= GATE_HELD]
    for r in gate:
        by = r["by_eps"][eps]
        lines.append(
            f"| {r['workspace']} | `{r['gallery_name']}` | {r['mode']} "
            f"| {','.join(map(str, r['steps'])) or '?'} | {by['held']} "
            f"| {r['moving_ratio']:.3f} | {'yes' if _static(r) else 'no'} "
            f"| {_counts(by['run_lengths'])} | {_counts(by['run_spacing'])} |"
        )
    moving = [r for r in gate if not _static(r)]
    clustered = [r for r in measured if r["by_eps"][eps]["clusters"]]
    lines.append("")
    lines.append(
        f"Gate (eps {eps}): {len(clustered)} takes have a cluster; "
        f"{len(moving)} non-static takes have >= {GATE_HELD} held frames."
    )
    return lines


def _static(row):
    return row["moving_ratio"] < STATIC_SHARE


def _counts(mapping):
    return ", ".join(f"{k}: {v}" for k, v in mapping.items()) or "none"


def _take_table_header():
    return (
        "| Workspace | Gallery name | Mode | Steps | Frames | held @0.5/1.0/1.5 "
        "| Clusters (start-end s: held) | d p10/p50/p90 |\n|---|---|---|---|---|---|---|---|"
    )


def _take_row(r, eps):
    held_counts = "/".join(str(r["by_eps"][e]["held"]) for e in map(str, EPSILONS))
    found = r["by_eps"][eps]["clusters"]
    cluster_text = (
        ", ".join(f"{c['start']:.2f}-{c['end']:.2f}: {c['held']}" for c in found[:4])
        + (f" (+{len(found) - 4})" if len(found) > 4 else "")
    ) or "none"
    p = r["d_percentiles"]
    return (
        f"| {r['workspace']} | `{r['gallery_name']}` | {r['mode']} "
        f"| {','.join(map(str, r['steps'])) or '?'} | {r['frames']} | {held_counts} "
        f"| {cluster_text} | {p.get('10')}/{p.get('50')}/{p.get('90')} |"
    )


# ------------------------------------------------------------------------ main


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("scan")
    s.add_argument("root", type=Path)
    s.add_argument("--workspace", action="append")
    s.add_argument("--out", default="/tmp/held-frames.json")
    s.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    s.add_argument("--all-video", action="store_true")
    f = sub.add_parser("file")
    f.add_argument("videos", nargs="+")
    r = sub.add_parser("report")
    r.add_argument("json")
    r.add_argument("--eps", default="1.0")
    args = parser.parse_args(argv)

    if args.command == "scan":
        root = args.root.resolve()
        out = Path(args.out).resolve()
        if root in out.parents:
            parser.error("--out must not be inside the workspace root")
        import dw  # the checkout whose library names sub-workflows

        repo = Path(dw.__file__).resolve().parent.parent
        search = [repo / "workflows", repo / "dw" / "workflows"]
        names = args.workspace or list(DEMAND)
        search += [workspace_outputs(root, n).parent / "workflows" for n in names]
        scan(root, names, out, args.jobs, args.all_video, search)
    elif args.command == "file":
        for video in args.videos:
            result = measure_file(video)
            summary = {
                e: (v["held"], len(v["clusters"]), v["run_lengths"])
                for e, v in result.get("by_eps", {}).items()
            }
            print(
                json.dumps(
                    {
                        "file": video,
                        "frames": result.get("frames"),
                        "cuts": result.get("cuts"),
                        "d_percentiles": result.get("d_percentiles"),
                        "held, clusters, run lengths by eps": summary,
                        "error": result.get("error"),
                    }
                )
            )
    else:
        with open(args.json) as handle:
            print("\n".join(report(json.load(handle), args.eps)))


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from dw.tasks.assess import TYPICAL_DELTA_FLOOR

    assert TYPICAL_DELTA_FLOOR == MOVING, "the moving floor moved in dw.tasks.assess"
    main()
