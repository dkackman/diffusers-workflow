"""Face table for the Kandinsky S2V Pro gate (#1339), in the 2026-11-12 format.

usage: python -I bench/ks2v_gate.py <yunet.onnx> <sface.onnx> <narration.wav> <sheet.jpg>
           <times> <label>=<video> ... [--ref <still.jpg>]

<times> is a comma list of seconds (the 2026-11-12 table used 0.5,15,30,45,60,75,85).
For each video, at each time: SFace cosine to the 0.5 s face, mean HSV saturation of the
YuNet face crop, and the Laplacian variance of that crop in grey at native scale. Then
still-mouth bins up to the last sampled time, rounded up: the YuNet mouth box's grey diff per frame
against the narration's RMS, in half-second bins, flagged when speech > 0.7x its median
and mouth motion < 0.45x its median. The sheet is one row of face crops per video.
With --ref, each video also gets an SFace row against that still's face (the input still,
the reference `face-drift-check` uses).
"""

import sys

import cv2
import numpy as np
import soundfile as sf


def largest(det, frame):
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


def crop(frame, f):
    x, y, w, h = [int(v) for v in f[:4]]
    return frame[max(y, 0) : y + h, max(x, 0) : x + w]


def mouth_box(f, frame):
    # YuNet landmarks: right mouth corner (10, 11), left mouth corner (12, 13)
    rx, ry, lx, ly = f[10], f[11], f[12], f[13]
    width = abs(lx - rx)
    cx, cy = (rx + lx) / 2, (ry + ly) / 2
    x0, x1 = int(cx - 0.75 * width), int(cx + 0.75 * width)
    y0, y1 = int(cy - 0.6 * width), int(cy + 0.6 * width)
    h, w = frame.shape[:2]
    return max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)


def still_mouth_bins(det, path, wav, until):
    audio, sr = sf.read(wav, always_2d=True)
    audio = audio.mean(axis=1)
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    prev, motion = None, []
    while len(motion) < until * fps:
        ok, frame = cap.read()
        if not ok:
            break
        f = largest(det, frame)
        if f is None:
            motion.append(np.nan)
            prev = None
            continue
        x0, y0, x1, y1 = mouth_box(f, frame)
        gray = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (48, 32)).astype(np.float32)
        motion.append(np.abs(gray - prev).mean() if prev is not None else np.nan)
        prev = gray
    motion = np.array(motion)
    per_bin = int(round(fps / 2))
    bins = len(motion) // per_bin
    mouth = np.array(
        [np.nanmean(motion[i * per_bin : (i + 1) * per_bin]) for i in range(bins)]
    )
    half = sr // 2
    speech = np.array(
        [np.sqrt(np.mean(audio[i * half : (i + 1) * half] ** 2)) for i in range(bins)]
    )
    flagged = (speech > 0.7 * np.median(speech)) & (mouth < 0.45 * np.nanmedian(mouth))
    return int(flagged.sum()), bins


def main():
    argv = sys.argv[1:]
    still = None
    if "--ref" in argv:
        i = argv.index("--ref")
        still = argv[i + 1]
        del argv[i : i + 2]
    yunet, sface, wav, sheet = argv[0:4]
    times = [float(t) for t in argv[4].split(",")]
    det = cv2.FaceDetectorYN.create(yunet, "", (320, 320), 0.6)
    rec = cv2.FaceRecognizerSF.create(sface, "")
    still_feat = None
    if still is not None:
        img = cv2.imread(still)
        still_feat = rec.feature(rec.alignCrop(img, largest(det, img)))
    rows = []
    for arg in argv[5:]:
        label, path = arg.split("=", 1)
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        f0 = read_at(cap, fps, times[0])
        ref = rec.feature(rec.alignCrop(f0, largest(det, f0)))
        sim, sim_still, sat, lap, tiles = [], [], [], [], []
        for t in times:
            fr = read_at(cap, fps, t)
            f = largest(det, fr) if fr is not None else None
            if f is None:
                sim.append(np.nan), sat.append(np.nan), lap.append(np.nan)
                sim_still.append(np.nan)
                tiles.append(np.zeros((256, 256, 3), np.uint8))
                continue
            c = crop(fr, f)
            feat = rec.feature(rec.alignCrop(fr, f))
            sim.append(rec.match(ref, feat, cv2.FaceRecognizerSF_FR_COSINE))
            if still_feat is not None:
                sim_still.append(
                    rec.match(still_feat, feat, cv2.FaceRecognizerSF_FR_COSINE)
                )
            sat.append(cv2.cvtColor(c, cv2.COLOR_BGR2HSV)[..., 1].mean())
            lap.append(
                cv2.Laplacian(cv2.cvtColor(c, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
            )
            x, y, w, h = [int(v) for v in f[:4]]
            cx, cy, s = x + w // 2, y + h // 2, int(max(w, h) * 0.9)
            tile = cv2.resize(
                fr[max(cy - s, 0) : cy + s, max(cx - s, 0) : cx + s], (256, 256)
            )
            cv2.putText(tile, f"{label} {t:g}s", (6, 22), 0, 0.6, (255, 255, 255), 2)
            tiles.append(tile)
        flagged, bins = still_mouth_bins(det, path, wav, np.ceil(times[-1]))
        rows.append(np.hstack(tiles))
        print(f"{label}: {frames} frames at {fps:g} fps ({frames / fps:.2f} s)")
        print("  SFace " + "/".join(f"{v:.2f}" for v in sim))
        if still_feat is not None:
            print("  SFace to --ref " + "/".join(f"{v:.2f}" for v in sim_still))
        print("  saturation " + "/".join(f"{v:.1f}" for v in sat))
        print("  Laplacian " + "/".join(f"{v:.0f}" for v in lap))
        print(f"  still-mouth bins {flagged} of {bins}")
    cv2.imwrite(sheet, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])


main()
