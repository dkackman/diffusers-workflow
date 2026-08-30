"""Flag a poisoned load in a chained MiniMax-H3 run.

A corrupted weight load leaves the transformer emitting noise, which the VAE
decodes faithfully - the run completes, burns its full wall clock, and writes
segments that are pure static. Noise is incompressible, so it shows up in the
file size and in the neighbouring-pixel delta long before anyone watches it.

Run it against the output directory while a chain is still going: a bad
segment-000 means the whole run is lost and is worth killing early.

    python local_inputs/check_segments.py outputs
"""

import glob
import os
import sys

import av
import numpy


def spatial_delta(path):
    """Mean absolute difference between horizontally neighbouring pixels."""
    with av.open(path) as container:
        frames = [
            frame.to_ndarray(format="rgb24")
            for index, frame in enumerate(container.decode(video=0))
            if index % 20 == 0
        ]
    stack = numpy.stack(frames).astype(numpy.int16)
    return numpy.abs(numpy.diff(stack, axis=2)).mean()


def main(output_dir):
    paths = sorted(glob.glob(os.path.join(output_dir, "*segment-*.mp4")))
    if not paths:
        print(f"no segment files in {output_dir}")
        return 0

    bad = 0
    for path in paths:
        megabytes = os.path.getsize(path) / 1e6
        delta = spatial_delta(path)
        # A clean segment sits near 3.5; noise near 9.5. 6.0 separates them with
        # room to spare in both directions
        verdict = "NOISE - bad load" if delta > 6.0 else "ok"
        bad += verdict != "ok"
        print(
            f"{os.path.basename(path):55s} {megabytes:5.2f} MB  delta={delta:5.2f}  {verdict}"
        )

    print(f"\n{len(paths) - bad}/{len(paths)} segments ok")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "outputs"))
