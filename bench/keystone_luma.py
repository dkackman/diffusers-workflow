"""#1394 step 1 (#1391 plan v4, Q6): the brightness of a segment-2 probe take.

    python bench/keystone_luma.py TAKE.mp4 [TAKE.mp4 ...] [--first-new 22]

For each take it prints the mean luma (BT.601 gray, 0-255) of frame 0, of the
carried guide frames (0 to first-new - 1), and of the new frames (first-new to
the end), the new frames' 24-frame window means, and their mean HSV saturation
(0-255). The probe's guide is segment 1's 22-frame tail, so the new frames
start at 22.

The numbering check compares each bright-keystone take's new-frame mean with
the normal-keystone take's: roycho87's method (#1391 issuecomment-6922271175,
after NerdNotes test 19, CC BY-SA 4.0).
"""

import argparse

import cv2
import numpy as np


def frame_means(path):
    cap = cv2.VideoCapture(path)
    luma, sat = [], []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            luma.append(float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()))
            sat.append(float(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[..., 1].mean()))
    finally:
        cap.release()
    return np.array(luma), np.array(sat)


def summarize(luma, sat, first_new=22):
    if len(luma) <= first_new:
        raise ValueError(f"{len(luma)} frames: no new frames after {first_new}")
    return {
        "frames": len(luma),
        "frame0": luma[0],
        "guide": luma[:first_new].mean(),
        "new": luma[first_new:].mean(),
        "windows": [luma[i : i + 24].mean() for i in range(0, len(luma), 24)],
        "new_saturation": sat[first_new:].mean(),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("takes", nargs="+")
    parser.add_argument("--first-new", type=int, default=22)
    args = parser.parse_args(argv)
    for path in args.takes:
        s = summarize(*frame_means(path), first_new=args.first_new)
        windows = " ".join(f"{w:.2f}" for w in s["windows"])
        print(
            f"{path}: frames={s['frames']} frame0={s['frame0']:.2f} "
            f"guide={s['guide']:.2f} new={s['new']:.3f} "
            f"new_sat={s['new_saturation']:.2f} windows24=[{windows}]"
        )


if __name__ == "__main__":
    main()
