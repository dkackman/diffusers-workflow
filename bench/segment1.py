"""Segment-1 drift labels (ruling R35).

No counter the bench runs touches segment 1 of a chain, so every arm of one
scene at one seed should open on the same segment 1. A cold-loaded worker
gives that canonical segment 1; a worker that has run some jobs can move it
into an alternate output (`nondeterminism-report.md`). This hashes segment 1
of a record's output and labels the record `clean` (the cold-load canonical
fingerprint for its scene and seed) or `drifted`.

The fingerprint is the md5 of the decoded-frame framemd5 hashes of video
frames 0-59 (2.5 s) and of AAC frames 0-129 (4.16 s at 32 kHz), written
`vvvvvvvv/aaaaaaaa`. Both sit inside segment 1 and before any seam, including
a `guide_backoff 17` arm's shorter segment 1 (107 kept frames, 4.46 s; at 140
AAC frames the window would cross that seam). Decoded H.264 is bit-exact
across ffmpeg builds, but decoded AAC is not (lem's ffmpeg 7.0.2 and a Mac
build give different audio halves for the same file), so take every
fingerprint of one table, canonical entries included, with one ffmpeg.

An arm whose arguments name a LoRA that runs on every segment
(`antiburn_model_name`) changes segment 1 by design; its key carries
`+lora`, and it is labelled against its own canonical entry when there is one.

Usage (on the host that holds the outputs, stdlib only):

    python3 segment1.py --root ~/bench-workspace/outputs RECORD.json ...
    python3 segment1.py --fingerprints fp.json --canonical C.json \
        --arms ARMS.json ... RECORD.json ...
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

VIDEO_FRAMES = 60
AUDIO_FRAMES = 130
SEGMENT1_ARGUMENTS = ("antiburn_model_name",)


def framemd5_hashes(text):
    """The per-frame hashes of one stream's framemd5 output, in order."""
    return [
        line.split(",")[-1].strip()
        for line in text.splitlines()
        if line.strip() and not line.startswith("#")
    ]


def _digest(hashes, count, what):
    if len(hashes) < count:
        raise ValueError(f"{what} has {len(hashes)} frames, segment 1 needs {count}")
    return hashlib.md5("".join(hashes[:count]).encode()).hexdigest()[:8]


def fingerprint_from_framemd5(
    video_text, audio_text, video_frames=VIDEO_FRAMES, audio_frames=AUDIO_FRAMES
):
    """`vvvvvvvv/aaaaaaaa` from the video and audio framemd5 outputs."""
    return "/".join(
        (
            _digest(framemd5_hashes(video_text), video_frames, "video"),
            _digest(framemd5_hashes(audio_text), audio_frames, "audio"),
        )
    )


def _framemd5(path, ffmpeg, stream, frames):
    # One stream per call: ffmpeg ends every output stream when the first
    # one reaches its -frames cap.
    command = [
        ffmpeg,
        "-v",
        "error",
        "-i",
        str(path),
        "-map",
        f"0:{stream}:0",
        f"-frames:{stream}",
        str(frames),
        "-f",
        "framemd5",
        "-",
    ]
    return subprocess.run(command, capture_output=True, text=True, check=True).stdout


def fingerprint_file(path, ffmpeg="ffmpeg"):
    return fingerprint_from_framemd5(
        _framemd5(path, ffmpeg, "v", VIDEO_FRAMES),
        _framemd5(path, ffmpeg, "a", AUDIO_FRAMES),
    )


def segment1_key(scene, seed, arguments=None):
    key = f"{scene}@{seed}"
    if any((arguments or {}).get(name) for name in SEGMENT1_ARGUMENTS):
        key += "+lora"
    return key


def label(fingerprint, canonical):
    """`clean`, `drifted`, or `no canonical` against one canonical entry."""
    if not canonical:
        return "no canonical"
    return "clean" if fingerprint == canonical["fingerprint"] else "drifted"


def classify(records, fingerprints, canonical, arguments_by_arm=None):
    """One row per succeeded record: arm, job, key, fingerprint, label."""
    arguments_by_arm = arguments_by_arm or {}
    rows = []
    for record in records:
        if record.get("status", "succeeded") != "succeeded":
            continue
        key = segment1_key(
            record["scene"], record["seed"], arguments_by_arm.get(record["arm"])
        )
        fingerprint = fingerprints.get(record["output"])
        rows.append(
            {
                "arm": record["arm"],
                "job_id": record.get("job_id"),
                "key": key,
                "fingerprint": fingerprint,
                "label": (
                    label(fingerprint, canonical.get(key))
                    if fingerprint
                    else "no output"
                ),
            }
        )
    return rows


def _records(paths):
    records = []
    for path in paths:
        records.extend(json.loads(Path(path).read_text()))
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("records", nargs="+", help="bench record files")
    parser.add_argument("--root", help="outputs folder; fingerprints each output")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--fingerprints", help="JSON {output: fingerprint}")
    parser.add_argument("--canonical", help="JSON {scene@seed: {fingerprint}}")
    parser.add_argument("--arms", nargs="*", default=[], help="arms files")
    args = parser.parse_args(argv)
    records = _records(args.records)
    if args.root:
        fingerprints = {}
        for record in records:
            if record.get("status", "succeeded") != "succeeded":
                continue
            try:
                fingerprints[record["output"]] = fingerprint_file(
                    Path(args.root).expanduser() / record["output"], args.ffmpeg
                )
            except (OSError, ValueError, subprocess.CalledProcessError) as exc:
                print(f"{record['arm']}: {exc}", file=sys.stderr)
        if not args.canonical:
            print(json.dumps(fingerprints, indent=1))
            return 0
    elif args.fingerprints:
        fingerprints = json.loads(Path(args.fingerprints).read_text())
    else:
        parser.error("one of --root or --fingerprints is needed")
    canonical = json.loads(Path(args.canonical).read_text()) if args.canonical else {}
    arguments = {
        arm["name"]: arm.get("arguments") or {}
        for path in args.arms
        for arm in json.loads(Path(path).read_text())
    }
    print("| arm | job | key | segment 1 | label |")
    print("|---|---|---|---|---|")
    for row in classify(records, fingerprints, canonical, arguments):
        print(
            f"| {row['arm']} | `{row['job_id']}` | {row['key']} | "
            f"`{row['fingerprint']}` | {row['label']} |"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
