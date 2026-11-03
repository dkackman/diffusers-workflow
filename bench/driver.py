"""Long-form bench driver: submit arms to the bench dw.serve, wait, and
measure each output through the server's compare and assess routes (the
same routes the MCP tools call, so the tester can repeat any number).
Stdlib only; never fetches media files."""

import argparse
import json
import math
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from verdict import noise_floor, verdict  # noqa: E402

DEFAULT_SERVER = "http://127.0.0.1:18766"
DEFAULT_TOKEN_FILE = "~/.dw-bench-token"
MAX_COMPARE_BYTES = 4_000_000
TERMINAL = ("succeeded", "failed", "cancelled")


@dataclass
class Arm:
    name: str
    scene: str
    workflow: str  # path under bench/workflows/
    arguments: dict
    seed: int


@dataclass
class Scene:
    name: str
    crop: list
    moments: list  # "frame:N" at segments 1, middle and last
    face_reference: str | None = None
    lines: list | None = None


def load_arms(path):
    data = json.loads(Path(path).read_text())
    return [
        Arm(
            name=a["name"],
            scene=a["scene"],
            workflow=a["workflow"],
            arguments=a.get("arguments") or {},
            seed=a["seed"],
        )
        for a in data
    ]


def load_scenes(path):
    data = json.loads(Path(path).read_text())
    return {
        s["name"]: Scene(
            name=s["name"],
            crop=s["crop"],
            moments=s["moments"],
            face_reference=s.get("face_reference"),
            lines=s.get("lines"),
        )
        for s in data
    }


def http(method, path, server, token, payload=None, params=None):
    """One JSON request; the single seam the tests fake."""
    url = server.rstrip("/") + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    data = None
    if payload is not None:
        data = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read() or b"null")
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{method} {path}: {e.code} {e.read().decode()[:500]}")


def health(server, token):
    return http("GET", "/api/health", server, token)


def submit(arm, server, token, workspace=None):
    workflow = json.loads((HERE / "workflows" / arm.workflow).read_text())
    payload = {
        "workflow": workflow,
        "arguments": {**arm.arguments, "seed": arm.seed},
        "note": f"bench arm {arm.name} (scene {arm.scene}, seed {arm.seed})",
        "acknowledged_cost": True,
    }
    params = {"workspace": workspace} if workspace else None
    job = http("POST", "/api/jobs", server, token, payload, params)
    return job["id"]


def wait(job_id, server, token, timeout_s, poll_s=15):
    deadline = time.monotonic() + timeout_s
    while True:
        job = http("GET", f"/api/jobs/{job_id}", server, token)
        if job.get("status") in TERMINAL:
            return job
        if time.monotonic() >= deadline:
            raise TimeoutError(f"job {job_id} still {job.get('status')} after {timeout_s}s")
        time.sleep(poll_s)


def output_name(job):
    """The gallery name of the job's deliverable video: the last video file
    in the manifest, `final` subfolder first."""
    kinds = job.get("output_kinds") or {}
    videos = [
        (entry.get("subfolder") == "final", name)
        for entry in job.get("manifest") or []
        for name in entry.get("files") or []
        if kinds.get(name, "video") == "video"
    ]
    finals = [name for final, name in videos if final]
    best = (finals or [name for _, name in videos] or [None])[-1]
    if best is None:
        raise ValueError(f"job {job.get('id')} wrote no video")
    return best


def _mean(values):
    return sum(values) / len(values) if values else None


def _stats(out, values, name_mean, name_max):
    values = [v for v in values if v is not None]
    if values:
        out[name_mean] = _mean(values)
        out[name_max] = max(values)


def measure(job, scene, server, token, workspace=None):
    name = output_name(job)
    params = {"probe": "analyze_seams"}
    if workspace:
        params["workspace"] = workspace
    seams = http(
        "GET", f"/api/gallery/{urllib.parse.quote(name, safe='/')}/assess",
        server, token, params=params,
    ).get("seams") or []
    out = {}
    _stats(out, [s.get("jump_ratio") for s in seams], "seam_jump_mean", "seam_jump_max")
    _stats(out, [s.get("look_ratio") for s in seams], "seam_look_mean", "seam_look_max")
    _stats(
        out, [abs(s["level_step_db"]) for s in seams if s.get("level_step_db") is not None],
        "seam_level_step_db_mean", "seam_level_step_db_max",
    )
    payload = {
        "names": [name],
        "max_dimension": 384,
        "max_total_bytes": MAX_COMPARE_BYTES,
        "at": scene.moments,
        "crop": scene.crop,
        "texture_bands": True,
    }
    compared = http(
        "POST", "/api/gallery/compare", server, token, payload,
        {"workspace": workspace} if workspace else None,
    )
    row = (compared.get("texture") or [{}])[0]
    cells = [c.get("texture") for c in row.get("cells") or []]
    if len(cells) >= 2 and cells[0] and cells[-1]:
        # Two-sided: blur (ratio < 1) and crust (ratio > 1) both count as
        # drift, so the measure is |ln(last / first)| and lower is better.
        for band in ("fine", "mid", "coarse"):
            luma = cells[0].get("luma") or {}, cells[-1].get("luma") or {}
            first, last = luma[0].get(band), luma[1].get(band)
            if first and last and first > 0 and last > 0:
                out[f"texture_{band}_seg_last"] = abs(math.log(last / first))
    if job.get("started_at") is not None and job.get("finished_at") is not None:
        out["wall_minutes"] = (job["finished_at"] - job["started_at"]) / 60.0
    return out


def run_arms(arms, scenes, server, token, do_measure, timeout_s, workspace=None, out_path=None):
    ids = {arm.name: submit(arm, server, token, workspace) for arm in arms}
    records = []

    def flush():
        if out_path:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            Path(out_path).write_text(json.dumps(records, indent=2))

    for arm in arms:
        record = {"arm": arm.name, "scene": arm.scene, "seed": arm.seed,
                  "job_id": ids[arm.name]}
        try:
            job = wait(ids[arm.name], server, token, timeout_s)
            record["status"] = job.get("status")
            record["error"] = job.get("error")
            if job.get("status") == "succeeded":
                record["output"] = output_name(job)
                if do_measure:
                    record["measures"] = measure(job, scenes[arm.scene], server, token, workspace)
        except Exception as exc:  # one arm's failure must not lose the others
            record.setdefault("status", "error")
            record["error"] = f"{type(exc).__name__}: {exc}"
        records.append(record)
        flush()
    return records


def report(records):
    lines = []
    keys = sorted({k for r in records for k in r.get("measures", {})})
    lines.append("| arm | scene | seed | status | " + " | ".join(keys) + " |")
    lines.append("|" + "---|" * (4 + len(keys)))
    for r in records:
        m = r.get("measures", {})
        cells = [f"{m[k]:.3f}" if k in m else "" for k in keys]
        lines.append(f"| {r['arm']} | {r['scene']} | {r['seed']} | {r['status']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _token(args):
    if args.token_file:
        return Path(args.token_file).expanduser().read_text().strip()
    if os.environ.get("BENCH_TOKEN"):
        return os.environ["BENCH_TOKEN"]
    path = Path(DEFAULT_TOKEN_FILE).expanduser()
    if path.is_file():
        return path.read_text().strip()
    raise SystemExit(
        f"no bench token: pass --token-file, set BENCH_TOKEN, or write {DEFAULT_TOKEN_FILE}"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default=os.environ.get("BENCH_SERVER", DEFAULT_SERVER))
    parser.add_argument("--token-file")
    parser.add_argument("--workspace")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="submit arms, wait, optionally measure")
    run.add_argument("arms")
    run.add_argument("--scenes", default=str(HERE / "scenes.json"))
    run.add_argument("--measure", action="store_true")
    run.add_argument("--out")
    run.add_argument("--timeout", type=int, default=7200)
    rep = sub.add_parser("report", help="markdown tables from records")
    rep.add_argument("records", nargs="+")
    args = parser.parse_args(argv)
    if args.command == "report":
        for path in args.records:
            print(f"### {path}\n")
            print(report(json.loads(Path(path).read_text())))
            print()
        return 0
    token = _token(args)
    records = run_arms(
        load_arms(args.arms), load_scenes(args.scenes) if args.measure else {},
        args.server, token, args.measure, args.timeout, args.workspace, args.out,
    )
    if not args.out:
        print(json.dumps(records, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
