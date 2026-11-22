"""bench/held_frames.py (#1551): the held-frame measure on synthetic stacks,
and the workspace walk's H3 detection and mode, without decoding."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

BENCH = Path(__file__).resolve().parent.parent / "bench"
sys.path.insert(0, str(BENCH))
import held_frames as hf  # noqa: E402

RNG = np.random.default_rng(1551)


def unique(count):
    """Distinct thumbnails, far apart (d ~ 70), away from 0/255 so noise
    never clips."""
    return [RNG.integers(20, 236, size=(36, 64)).astype(np.uint8) for _ in range(count)]


def stack(frames):
    return np.stack(frames)


def test_floor_matches_assess():
    from dw.tasks.assess import TYPICAL_DELTA_FLOOR

    assert hf.MOVING == TYPICAL_DELTA_FLOOR


def test_duplicate_every_second_frame_is_held():
    frames = [f for u in unique(24) for f in (u, u)]  # 48 frames, 12 shown as 24
    result = hf.measure(stack(frames), 24.0, cuts=[])
    for eps in ("0.5", "1.0", "1.5"):
        by = result["by_eps"][eps]
        assert by["held"] == 24  # frames 1, 3, ..., 47
        assert by["run_lengths"] == {1: 22}  # 1 and 47: a side has no d
        assert len(by["clusters"]) == 1
        cluster = by["clusters"][0]
        assert (cluster["first"], cluster["last"], cluster["held"]) == (1, 47, 24)
        assert cluster["end"] == pytest.approx(47 / 24, abs=1e-3)


def test_duplicate_every_fourth_frame_holds_edges_and_runs_of_three():
    frames = [f for u in unique(12) for f in (u, u, u, u)]
    result = hf.measure(stack(frames), 24.0, cuts=[])
    by = result["by_eps"]["1.0"]
    # the neighbour rule marks only a run's edge frames (3, 5, 7, 9, ...)
    assert hf.held(hf.deltas(stack(frames)), 1.0)[:4] == [3, 5, 7, 9]
    assert by["run_lengths"] == {3: 10}
    assert by["run_spacing"] == {4: 9}


def test_static_shot_holds_nothing():
    still = unique(1)[0]
    result = hf.measure(stack([still] * 60), 24.0, cuts=[])
    for eps in ("0.5", "1.0", "1.5"):
        assert result["by_eps"][eps]["held"] == 0
        assert result["by_eps"][eps]["clusters"] == []
        assert result["by_eps"][eps]["run_lengths"] == {}


def test_slow_moving_shot_holds_nothing():
    # every frame moves by ~1 grey level: under eps, but no neighbour moves
    base = unique(1)[0].astype(np.int16)
    frames = [np.clip(base + k, 0, 255).astype(np.uint8) for k in range(10)]
    assert hf.measure(stack(frames), 24.0, cuts=[])["by_eps"]["1.5"]["held"] == 0


def test_hard_cut_frames_do_not_count():
    a, b = unique(2)
    frames = [a] * 20 + [b] * 20  # the cut is frame 20
    d = hf.deltas(stack(frames))
    # without the guard the static frames either side of the cut look held
    assert hf.held(d, 1.0) == [19, 21]
    result = hf.measure(stack(frames), 24.0, cuts=[20])
    assert result["by_eps"]["1.0"]["held"] == 0
    assert result["by_eps"]["1.0"]["run_lengths"] == {}
    assert hf.excluded(40, [20]) == {0, 19, 20, 21}


def test_content_cuts_finds_the_synthetic_cut():
    pytest.importorskip("cv2")
    from dw.tasks.scene_cuts import content_cuts

    dark = np.full((72, 128, 3), 30, np.uint8)
    bright = np.zeros((72, 128, 3), np.uint8)
    bright[..., 0] = 220
    cuts = [frame for frame, _ in content_cuts([dark] * 12 + [bright] * 12)]
    assert cuts == [12]


def test_noisy_duplicate_is_held_above_the_noise():
    frames = []
    for u in unique(20):
        noise = np.round(RNG.normal(0, 0.9, size=u.shape)).astype(np.int16)
        frames += [u, (u.astype(np.int16) + noise).astype(np.uint8)]
    d = hf.deltas(stack(frames))
    assert 0.5 < np.nanmedian(d[1::2]) < 1.0  # h264-like jitter on the repeat
    result = hf.measure(stack(frames), 24.0, cuts=[])
    assert result["by_eps"]["0.5"]["held"] == 0
    assert result["by_eps"]["1.0"]["held"] == 20
    assert result["by_eps"]["1.5"]["held"] == 20


def test_clusters_split_on_gaps_over_four_and_drop_lone_holds():
    found = hf.clusters([10, 12, 17, 40, 80, 82], fps=24.0)
    assert [(c["first"], c["last"], c["held"]) for c in found] == [
        (10, 17, 3),
        (80, 82, 2),
    ]
    assert found[0]["start"] == pytest.approx(10 / 24, abs=1e-3)


def test_d_percentiles_skip_cut_frames():
    a, b = unique(2)
    result = hf.measure(stack([a] * 10 + [b] * 10), 24.0, cuts=[10])
    assert result["d_percentiles"]["90"] == 0.0


# --------------------------------------------------------------- the walk


def _run(outputs, identity, run_id, workflow, manifest_steps, files, subs=None):
    run = outputs / identity / run_id
    for relative in files:
        (run / relative).parent.mkdir(parents=True, exist_ok=True)
        (run / relative).write_bytes(b"")
    (run / "workflow.json").write_text(json.dumps(workflow))
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "workflow": {"id": identity, "sub_workflows": subs or {}},
                "arguments": {"steps": 8},
                "steps": manifest_steps,
            }
        )
    )
    return run


H3_STEP = {
    "name": "shot",
    "pipeline": {
        "from_pretrained_arguments": {
            "model_name": "variable:model",
            "workflow": "t2va",
        },
        "arguments": {"num_inference_steps": "variable:steps"},
    },
}


def test_walk_lists_h3_videos_with_mode_and_steps(tmp_path):
    root = tmp_path / "ws-root"
    outputs = root / "h3-quality" / "outputs"
    workflow = {
        "id": "take",
        "variables": {"model": hf.H3_MODEL, "steps": 30},
        "steps": [
            H3_STEP,
            {"name": "join", "task": {"command": "concat_videos"}},
        ],
    }
    shots = [{"name": "a"}, {"name": "b"}]
    _run(
        outputs,
        "take",
        "r1",
        workflow,
        [
            {"step": "shot@a", "files": ["intermediate/a.mp4"]},
            {"step": "join", "files": ["final/joined.mp4"], "shots": shots},
        ],
        ["intermediate/a.mp4", "final/joined.mp4", "final/x.segment-1.mp4"],
    )
    other = {
        "id": "ltx",
        "steps": [
            {
                "name": "s",
                "pipeline": {"from_pretrained_arguments": {"model_name": "x/y"}},
            }
        ],
    }
    _run(outputs, "ltx", "r2", other, [{"step": "s", "files": ["v.mp4"]}], ["v.mp4"])

    rows = {r["gallery_name"]: r for r in hf.walk(root, ["h3-quality"], [])}
    assert set(rows) == {
        "take/r1/intermediate/a.mp4",
        "take/r1/final/joined.mp4",
        "take/r1/final/x.segment-1.mp4",
        "ltx/r2/v.mp4",
    }
    assert rows["take/r1/intermediate/a.mp4"]["mode"] == "single"
    assert rows["take/r1/final/joined.mp4"]["mode"] == "cuts"
    assert rows["take/r1/final/x.segment-1.mp4"]["mode"] == "chain segment"
    assert rows["take/r1/intermediate/a.mp4"]["steps"] == [8]  # argument wins
    assert rows["take/r1/final/joined.mp4"]["h3"] is True
    assert rows["ltx/r2/v.mp4"]["h3"] is False
    assert all(r["demand"] for r in rows.values())


def test_sub_workflow_h3_and_chained_join(tmp_path):
    library = tmp_path / "library"
    (library / "minimax").mkdir(parents=True)
    (library / "minimax" / "chain.json").write_text(
        json.dumps({"steps": [{"pipeline": {"model_name": hf.H3_MODEL}}]})
    )
    root = tmp_path / "root"
    workflow = {"id": "outer", "steps": [{"name": "chain", "workflow": {}}]}
    shots = [{"name": "a", "segment_file": "s1"}, {"name": "b", "segment_file": "s2"}]
    _run(
        root / "outputs",
        "outer",
        "r1",
        workflow,
        [{"step": "chain", "files": ["out.mp4"], "shots": shots}],
        ["out.mp4"],
        subs={"minimax/chain": "abc"},
    )
    (row,) = hf.walk(root, ["default"], [library])
    assert row["h3"] is True
    assert row["mode"] == "joined"


def test_scan_refuses_output_inside_the_workspace_root(tmp_path):
    with pytest.raises(SystemExit):
        hf.main(["scan", str(tmp_path), "--out", str(tmp_path / "x.json")])


def test_report_reads_the_gate_off_non_static_takes():
    choppy = hf.measure(stack([f for u in unique(12) for f in (u, u)]), 24.0, [])
    a, b = unique(2)
    # one jump: its still neighbours look held, and cluster
    still = hf.measure(stack([a] * 30 + [b] + [a] * 30), 24.0, [])
    rows = [
        {
            "workspace": "h3-quality",
            "demand": True,
            "h3": True,
            "mode": "single",
            "steps": [20],
            "gallery_name": name,
            "path": name,
            **result,
        }
        for name, result in (("choppy.mp4", choppy), ("still.mp4", still))
    ]
    lines = hf.report({"workspaces": ["h3-quality"], "rows": rows})
    assert still["by_eps"]["1.0"]["held"] == 2 and hf._static(rows[1])
    assert lines[-1] == (
        "Gate (eps 1.0): 2 takes have a cluster; 1 non-static takes have >= 7 held frames."
    )


def test_a_run_with_only_workflow_json_is_still_read(tmp_path):
    run = tmp_path / "outputs" / "trailer" / "r1"
    (run / "intermediate").mkdir(parents=True)
    (run / "intermediate" / "shot@s01.mp4").write_bytes(b"")
    workflow = {
        "id": "trailer",
        "variables": {"model": hf.H3_MODEL},
        "steps": [H3_STEP],
    }
    (run / "workflow.json").write_text(json.dumps(workflow))
    (row,) = hf.walk(tmp_path, ["default"], [])
    assert row["h3"] is True
    assert row["mode"] == "no manifest"
