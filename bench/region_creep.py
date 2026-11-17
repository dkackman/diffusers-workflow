"""Subject-region sharpness creep over one clip (#1399).

The stand-in for bench/face_creep.py when the face is hidden (the #1326 astronaut's
mirrored visor: YuNet finds a face in 2 of 140 samples of cs-0). The measure is the
same Laplacian variance, on the gray crop of a fixed box instead of a detected face:
the centre box (x 35-65 %, y 5-60 % of the frame, the helmet and visor of a centred
chest-up framing) and the whole frame. Sampled every 0.25 s. Prints the median of
each 1 s window and the creep (last full 1 s window over the first).

usage: python -I bench/region_creep.py <label>=<clip.mp4> ...
"""

import sys

import cv2
import numpy as np

STEP = 0.25
BOX = (0.35, 0.65, 0.05, 0.60)


def lap(gray):
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def samples(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    every = max(int(round(STEP * fps)), 1)
    out, i = [], 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i % every == 0:
            g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            h, w = g.shape
            x0, x1, y0, y1 = BOX
            box = g[int(y0 * h) : int(y1 * h), int(x0 * w) : int(x1 * w)]
            out.append((i / fps, lap(box), lap(g)))
        i += 1
    return out, n / fps


def window(s, k, a, b):
    v = [x[k] for x in s if a <= x[0] < b]
    return float(np.median(v)) if v else float("nan")


def main():
    for arg in sys.argv[1:]:
        label, path = arg.split("=", 1)
        s, dur = samples(path)
        last = int(np.floor(dur)) - 1
        print(f"{label} ({dur:.2f} s, {len(s)} samples)")
        for k, name in ((1, "centre box"), (2, "whole frame")):
            per = [window(s, k, a, a + 1) for a in range(last + 1)]
            print(f"  {name} per 1 s: " + ", ".join(f"{v:.0f}" for v in per))
            print(
                f"  {name} creep: {per[0]:.0f} -> {per[last]:.0f} = x{per[last] / per[0]:.2f}"
            )


main()
