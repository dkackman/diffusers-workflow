"""Face sharpness creep and identity hold over one clip (#1399).

The measure is bench/cut_windows.py's (YuNet's largest face, Laplacian variance
of the gray crop, SFace cosine), sampled every 0.25 s. Prints the median
sharpness of each 1 s window, the total creep (last full 1 s window over the
first), and SFace between the mean feature of the first 1 s and of the last 1 s.

usage: python -I bench/face_creep.py <yunet.onnx> <sface.onnx> <label>=<clip.mp4> ...
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


def samples(det, rec, path):
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
                x, y, w, h = [int(v) for v in f[:4]]
                crop = fr[max(y, 0) : y + h, max(x, 0) : x + w]
                sharp = cv2.Laplacian(
                    cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F
                ).var()
                out.append((i / fps, sharp, rec.feature(rec.alignCrop(fr, f))))
        i += 1
    return out, n / fps, n // every + 1


def window(s, a, b):
    v = [x[1] for x in s if a <= x[0] < b]
    return float(np.median(v)) if v else float("nan")


def mean_feat(s, a, b):
    v = [x[2] for x in s if a <= x[0] < b]
    return np.mean(v, axis=0).astype(np.float32) if v else None


def main():
    yunet, sface = sys.argv[1:3]
    det = cv2.FaceDetectorYN.create(yunet, "", (320, 320), 0.6)
    rec = cv2.FaceRecognizerSF.create(sface, "")
    for arg in sys.argv[3:]:
        label, path = arg.split("=", 1)
        s, dur, total = samples(det, rec, path)
        last = int(np.floor(dur)) - 1
        print(f"{label} ({dur:.2f} s, face in {len(s)} of {total} samples)")
        per = [window(s, a, a + 1) for a in range(last + 1)]
        print("  per 1 s: " + ", ".join(f"{v:.0f}" for v in per))
        first, end = per[0], per[last]
        print(f"  creep: {first:.0f} -> {end:.0f} = x{end / first:.2f}")
        fa, fb = mean_feat(s, 0, 1), mean_feat(s, last, last + 1)
        sim = (
            rec.match(fa, fb, cv2.FaceRecognizerSF_FR_COSINE)
            if fa is not None and fb is not None
            else float("nan")
        )
        print(f"  SFace first 1 s vs last 1 s: {sim:.3f}")


main()
