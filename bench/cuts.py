"""Cuts inside a chain's segments (Task 13 Step 5, bench/records/check.md).

The chain check (dw/pipeline_processors/chain_check.py) measures a segment at
its seam: the one frame step across it and a LOOK_WINDOW of frames either
side. A segment that cuts to another scene inside itself, past those windows, passes
it. This reads a finished take and its shots and reports, per segment, the
largest single-frame step inside the segment against the segment's own typical
step (`analyze_seams`' arithmetic: the 90th percentile, floored at
TYPICAL_DELTA_FLOOR) - the `jump_ratio` of a seam nobody marked - so a cut
ceiling for the check can be set from the bench's own takes.

Usage (on lem, from a checkout, with the dw venv):

    python bench/cuts.py TAKE.mp4 ...
"""

import json
import sys

import numpy

from dw.runs import shots_beside
from dw.tasks.assess import (
    TYPICAL_DELTA_FLOOR,
    TYPICAL_DELTA_PERCENTILE,
    read_media,
)


def inner_cut(thumbs, start, count):
    """(ratio, frame) of the largest frame step inside thumbs[start:start +
    count] over the span's floored typical step, or (None, None)."""
    span = thumbs[start : start + count].astype(numpy.int16)
    if span.shape[0] < 3:
        return None, None
    deltas = numpy.abs(numpy.diff(span, axis=0)).mean(axis=(1, 2))
    typical = max(
        TYPICAL_DELTA_FLOOR, float(numpy.percentile(deltas, TYPICAL_DELTA_PERCENTILE))
    )
    at = int(deltas.argmax())
    return round(float(deltas[at]) / typical, 2), start + at + 1


def take_cuts(path):
    media = read_media(path)
    shots = shots_beside(path) or []
    rows = []
    for index, shot in enumerate(shots):
        ratio, frame = inner_cut(
            media.thumbs, int(shot["start_frame"]), int(shot["num_frames"])
        )
        rows.append(
            {
                "segment": index + 1,
                "inner_cut_ratio": ratio,
                "at_frame": frame,
                "attempts": shot.get("attempts"),
                "check_failed": shot.get("check_failed"),
            }
        )
    return rows


def main(argv):
    out = {path: take_cuts(path) for path in argv}
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
