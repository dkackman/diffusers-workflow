"""Face-crop sharpness in 1 s windows around a cut, and identity across it.

The measure is bench/face_windows.py's (YuNet's largest face, Laplacian variance
of the gray crop, SFace cosine), in 1 s windows sampled every 0.25 s.

usage:
  python -I bench/cut_windows.py <yunet.onnx> <sface.onnx> pair <A.mp4> <B.mp4>
      A's per-second sharpness, B's, the step from A's last 1 s to B's first 1 s,
      when B first reaches A's end level, and SFace across the cut.
  python -I bench/cut_windows.py <yunet.onnx> <sface.onnx> clip <clip.mp4> <cut_s>
      per-second sharpness of one clip, the step at cut_s, SFace across it.
  python -I bench/cut_windows.py <yunet.onnx> <sface.onnx> still <image> ...
      face sharpness of each still.
"""

import sys

import cv2
import numpy as np

STEP = 0.25


def face(det, frame):
    h, w = frame.shape[:2]
    det.setInputSize((w, h))
    _, faces = det.detect(frame)
    if faces is None or len(faces) == 0:
        return None
    return max(faces, key=lambda f: f[2] * f[3])


def sharpness(frame, f):
    x, y, w, h = [int(v) for v in f[:4]]
    crop = frame[max(y, 0) : y + h, max(x, 0) : x + w]
    return cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()


def samples(det, rec, path):
    """(t, sharpness, feature) every STEP seconds, frames with a face only."""
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
            f = face(det, fr)
            if f is not None:
                out.append((i / fps, sharpness(fr, f), rec.feature(rec.alignCrop(fr, f))))
        i += 1
    return out, n / fps


def window(s, a, b):
    v = [x[1] for x in s if a <= x[0] < b]
    return float(np.median(v)) if v else float("nan")


def mean_feat(s, a, b):
    v = [x[2] for x in s if a <= x[0] < b]
    return np.mean(v, axis=0).astype(np.float32) if v else None


def per_second(label, s, dur):
    print(f"{label} ({dur:.2f} s, {len(s)} face samples)")
    for a in range(int(np.ceil(dur))):
        print(f"  {a:>3}-{a + 1:<3} s: sharp {window(s, a, a + 1):.0f}")


def across(rec, fa, fb):
    if fa is None or fb is None:
        return float("nan")
    return rec.match(fa, fb, cv2.FaceRecognizerSF_FR_COSINE)


def main():
    yunet, sface, mode = sys.argv[1:4]
    det = cv2.FaceDetectorYN.create(yunet, "", (320, 320), 0.6)
    rec = cv2.FaceRecognizerSF.create(sface, "")
    args = sys.argv[4:]
    if mode == "still":
        for p in args:
            fr = cv2.imread(p)
            f = face(det, fr)
            print(f"{p}: sharp {sharpness(fr, f):.0f}" if f is not None else f"{p}: no face")
        return
    if mode == "pair":
        sa, da = samples(det, rec, args[0])
        sb, db = samples(det, rec, args[1])
        per_second("A", sa, da)
        per_second("B", sb, db)
        end_a = window(sa, da - 1, da)
        head_b = window(sb, 0, 1)
        reach = next(
            (a for a in range(int(np.ceil(db))) if window(sb, a, a + 1) >= end_a), None
        )
        sim = across(rec, mean_feat(sa, da - 1, da), mean_feat(sb, 0, 1))
    else:
        s, d = samples(det, rec, args[0])
        cut = float(args[1])
        per_second("clip", s, d)
        end_a = window(s, cut - 1, cut)
        head_b = window(s, cut, cut + 1)
        reach = next(
            (a - cut for a in np.arange(cut, d, 1.0) if window(s, a, a + 1) >= end_a),
            None,
        )
        sim = across(rec, mean_feat(s, cut - 1, cut), mean_feat(s, cut, cut + 1))
    print(f"last 1 s before cut: {end_a:.0f}   first 1 s after: {head_b:.0f}")
    print(f"step: {head_b / end_a:.2f}x ({(head_b - end_a) / end_a * 100:+.0f}%)")
    print(
        "after side first reaches the before level: "
        + (f"window starting {reach:g} s after the cut" if reach is not None else "never")
    )
    print(f"SFace across the cut (mean feature, 1 s each side): {sim:.3f}")


main()
