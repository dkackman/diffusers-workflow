"""The chain check's `face_drop`, measured on a finished one-shot take (#1571).

usage: python bench/face_drop_spans.py <video> [--span 7.2] [--until 60]

A chain records, per segment, segment 1's median SFace similarity to its
frame 0 minus that segment's median similarity to the same frame 0, each the
median of FACE_SAMPLES evenly spaced frames (`chain_check.with_face_drop`).
A hybrid-windows take has no segments, so this cuts it into spans of the
chain's segment length (`--span`, seconds; the 86 s arc v4 chain is 12
segments, about 7.2 s each) and scores each span the same way, with the same
`face_drift.face_similarities`. The chain arm itself, measured by this script,
is the check that the spans read like its segments.
"""

import json
import sys

import cv2
import numpy

from dw.pipeline_processors.chain_check import FACE_SAMPLES
from dw.tasks.face_drift import face_similarities


def frames_of(path, until):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames = []
    while len(frames) < int(round(until * fps)):
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return frames, fps


def spans(frames, fps, span):
    size = int(round(span * fps))
    return [frames[i : i + size] for i in range(0, len(frames), size)]


def main(argv):
    span = float(argv[argv.index("--span") + 1]) if "--span" in argv else 7.2
    until = float(argv[argv.index("--until") + 1]) if "--until" in argv else 60.0
    frames, fps = frames_of(argv[0], until)
    reference = frames[0]
    medians = []
    for part in spans(frames, fps, span):
        scores = face_similarities(part, reference, FACE_SAMPLES)
        medians.append(float(numpy.median(scores)) if scores else None)
    base = medians[0]
    drops = [None if m is None else round(base - m, 3) for m in medians[1:]]
    print(
        json.dumps(
            {
                "video": argv[0],
                "span_s": span,
                "until_s": until,
                "span_medians": [None if m is None else round(m, 3) for m in medians],
                "face_drop": drops,
                "face_drop_max": max((d for d in drops if d is not None), default=None),
            }
        )
    )


if __name__ == "__main__":
    main(sys.argv[1:])
