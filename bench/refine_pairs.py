"""Measure a refined take against its unrefined source (Task 17 Step 4).

A refine's output is decoded fresh, so its run records no shots and the
server's assess route finds no seams in it. This reads both files in-process
(the code behind the compare and assess routes, as round 3's moments pass
did) and measures the refined take at the source's own seams:

- `analyze_seams` at every seam the source's run recorded: jump, look and
  level step, and `typical_delta` (the within-shot frame delta beside each
  seam, the bench's motion measure);
- texture drift against the first moment, `|ln(band / band@first)|`, on the
  scene crop at every moment (and a face crop where the scene has one);
- motion: the mean absolute frame-to-frame delta of the grey thumbnails,
  over the whole take;
- the refined take against its source at each moment: PSNR and SSIM.

Run on lem from the bench checkout, with the deployed dw on the path:

    PYTHONPATH=~/dw-bench ~/diffusers-workflow/venv/bin/python \\
        bench/refine_pairs.py ~/bench-workspace/outputs pairs.json --out out.json

`pairs.json` is a list of {arm, scene, source, refined, moments}: output
names relative to the outputs root, and moments as "frame:N".
"""

import argparse
import json
import math
import sys
from pathlib import Path

# Face crops (x, y, w, h) for the scenes with a face to watch
FACE_CROPS = {"s1-static": [360, 30, 240, 240]}
SCENE_CROPS = {
    "s1-static": [0, 0, 240, 150],
    "s2-arc": [0, 0, 240, 150],
    "s3-dialogue-walk": [0, 0, 240, 150],
}


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def seam_measures(path, shots):
    from dw.server.assess import assess

    seams = assess(path, "video", shots, probe="analyze_seams").get("seams") or []
    out = {"seams": [], "measures": {}}
    for seam in seams:
        out["seams"].append(
            {
                key: seam.get(key)
                for key in (
                    "seconds",
                    "jump_ratio",
                    "look_ratio",
                    "level_step_db",
                    "typical_delta",
                )
            }
        )
    m = out["measures"]
    for key, name in (("jump_ratio", "seam_jump"), ("look_ratio", "seam_look")):
        values = [s.get(key) for s in seams if s.get(key) is not None]
        if values:
            m[f"{name}_mean"], m[f"{name}_max"] = _mean(values), max(values)
    levels = [
        abs(s["level_step_db"]) for s in seams if s.get("level_step_db") is not None
    ]
    if levels:
        m["seam_level_step_db_mean"], m["seam_level_step_db_max"] = (
            _mean(levels),
            max(levels),
        )
    deltas = [s.get("typical_delta") for s in seams if s.get("typical_delta")]
    if deltas:
        m["typical_delta_mean"] = _mean(deltas)
    return out


def motion(path):
    import numpy

    from dw.tasks.assess import read_media

    media = read_media(path)
    thumbs = media.thumbs.astype(numpy.float32)
    deltas = numpy.abs(numpy.diff(thumbs, axis=0)).mean(axis=(1, 2))
    return float(deltas.mean()), [float(d) for d in deltas]


def _source(name, path):
    from dw.media_readers import video_shape

    return {"name": name, "kind": "video", "path": path, "shape": video_shape(path)}


def texture_series(path, name, moments, crop):
    from dw import media_compare

    answer = media_compare.compare(
        [_source(name, path)], moments, crop, 0, "shared", 384, texture_bands=True
    )
    cells = answer["texture"][0]["cells"]
    first = (cells[0].get("texture") or {}).get("luma") or {}
    series = {}
    for band in ("fine", "mid", "coarse"):
        row = []
        for cell in cells:
            luma = (cell.get("texture") or {}).get("luma") or {}
            a, b = first.get(band), luma.get(band)
            row.append(abs(math.log(b / a)) if a and b and a > 0 and b > 0 else None)
        series[band] = row
    return series


def fidelity(source_path, refined_path, moments):
    from dw import media_compare

    answer = media_compare.compare(
        [_source("source", source_path), _source("refined", refined_path)],
        moments,
        None,
        0,
        None,
        384,
    )
    cells = answer["metrics"][0]["cells"]
    return {
        "psnr_db": [c.get("psnr_db") for c in cells],
        "ssim": [c.get("ssim") for c in cells],
    }


def measure_take(path, name, shots, scene, moments):
    seams = seam_measures(path, shots)
    m = dict(seams["measures"])
    series = texture_series(path, name, moments, SCENE_CROPS[scene])
    for band, row in series.items():
        if row and row[-1] is not None:
            m[f"texture_{band}_seg_last"] = row[-1]
    face = None
    if scene in FACE_CROPS:
        face = texture_series(path, name, moments, FACE_CROPS[scene])
        if face["fine"][-1] is not None:
            m["face_fine_seg_last"] = face["fine"][-1]
    motion_mean, _ = motion(path)
    return {
        "measures": m,
        "seams": seams["seams"],
        "texture": series,
        "face_texture": face,
        "motion_mean": motion_mean,
    }


def measure_pair(root, pair):
    from dw.runs import recorded_shots

    shots = recorded_shots(str(root), pair["source"])
    if not shots:
        raise ValueError(f"{pair['source']} recorded no shots")
    source_path = str(Path(root) / pair["source"])
    refined_path = str(Path(root) / pair["refined"])
    moments = pair["moments"]
    return {
        "arm": pair["arm"],
        "scene": pair["scene"],
        "moments": moments,
        "source": measure_take(
            source_path, pair["source"], shots, pair["scene"], moments
        ),
        "refined": measure_take(
            refined_path, pair["refined"], shots, pair["scene"], moments
        ),
        "fidelity": fidelity(source_path, refined_path, moments),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outputs", help="the bench workspace's outputs root")
    parser.add_argument("pairs")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    root = Path(args.outputs).expanduser()
    results = []
    for pair in json.loads(Path(args.pairs).read_text()):
        try:
            results.append(measure_pair(root, pair))
        except Exception as exc:  # one pair's failure must not lose the others
            results.append(
                {"arm": pair["arm"], "error": f"{type(exc).__name__}: {exc}"}
            )
        Path(args.out).write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
