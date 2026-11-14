"""Face-crop sharpness / saturation / identity per 10 s window, and a face strip.

usage: python -I bench/face_windows.py <yunet.onnx> <sface.onnx> <out_strip.jpg> <label>=<video> ...
"""

import sys

import cv2
import numpy as np

WINDOWS = [(0, 10), (20, 30), (40, 50), (60, 70), (80, 90)]
STRIP_AT = [0.5, 40, 60, 85]


def face(det, frame):
    h, w = frame.shape[:2]
    det.setInputSize((w, h))
    _, faces = det.detect(frame)
    if faces is None or len(faces) == 0:
        return None
    return max(faces, key=lambda f: f[2] * f[3])


def read_at(cap, fps, t):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
    ok, frame = cap.read()
    return frame if ok else None


def main():
    yunet, sface, strip_path = sys.argv[1:4]
    det = cv2.FaceDetectorYN.create(yunet, "", (320, 320), 0.6)
    rec = cv2.FaceRecognizerSF.create(sface, "")
    rows, strips = {}, []
    for arg in sys.argv[4:]:
        label, path = arg.split("=", 1)
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        ref = None
        f0 = read_at(cap, fps, 0.5)
        ff = face(det, f0)
        ref = rec.feature(rec.alignCrop(f0, ff))
        out = []
        for a, b in WINDOWS:
            sharp, sat, sim = [], [], []
            for t in np.arange(a, b, 0.5):
                fr = read_at(cap, fps, float(t))
                if fr is None:
                    continue
                f = face(det, fr)
                if f is None:
                    continue
                x, y, w, h = [int(v) for v in f[:4]]
                crop = fr[max(y, 0) : y + h, max(x, 0) : x + w]
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                sharp.append(cv2.Laplacian(gray, cv2.CV_64F).var())
                sat.append(cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)[..., 1].mean())
                feat = rec.feature(rec.alignCrop(fr, f))
                sim.append(rec.match(ref, feat, cv2.FaceRecognizerSF_FR_COSINE))
            out.append(
                (
                    f"{a}-{b} s",
                    np.median(sharp) if sharp else float("nan"),
                    np.mean(sat) if sat else float("nan"),
                    np.mean(sim) if sim else float("nan"),
                    len(sharp),
                )
            )
        rows[label] = out
        tiles = []
        for t in STRIP_AT:
            fr = read_at(cap, fps, t)
            f = face(det, fr)
            x, y, w, h = [int(v) for v in f[:4]]
            cx, cy, s = x + w // 2, y + h // 2, int(max(w, h) * 0.9)
            tile = fr[max(cy - s, 0) : cy + s, max(cx - s, 0) : cx + s]
            tile = cv2.resize(tile, (320, 320))
            cv2.putText(tile, f"{label} {t:g}s", (8, 24), 0, 0.7, (255, 255, 255), 2)
            tiles.append(tile)
        strips.append(np.hstack(tiles))
    cv2.imwrite(strip_path, np.vstack(strips), [cv2.IMWRITE_JPEG_QUALITY, 90])
    for label, out in rows.items():
        print(label)
        for w, sh, sa, si, n in out:
            print(f"  {w}: sharp {sh:.0f}  sat {sa:.1f}  sim {si:.2f}  (n={n})")


main()
