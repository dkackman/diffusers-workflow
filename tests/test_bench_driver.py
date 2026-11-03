import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bench"))

import driver  # noqa: E402


def test_load_arms_reads_every_field(tmp_path):
    path = tmp_path / "arms.json"
    path.write_text(json.dumps([{"name": "a", "scene": "s", "workflow": "w.json",
                                 "arguments": {"prompt": "x"}, "seed": 7}]))
    (arm,) = driver.load_arms(str(path))
    assert (arm.name, arm.scene, arm.workflow, arm.arguments, arm.seed) == (
        "a", "s", "w.json", {"prompt": "x"}, 7)


def test_submit_carries_seed_and_names_the_arm(monkeypatch, tmp_path):
    seen = {}

    def fake(method, path, server, token, payload=None, params=None):
        seen.update(method=method, path=path, payload=payload, token=token)
        return {"id": "job-1"}

    monkeypatch.setattr(driver, "http", fake)
    monkeypatch.setattr(driver, "HERE", tmp_path)
    (tmp_path / "workflows").mkdir()
    (tmp_path / "workflows" / "w.json").write_text('{"steps": []}')
    arm = driver.Arm("armA", "s", "w.json", {"prompt": "x"}, 11)
    assert driver.submit(arm, "http://h", "tok") == "job-1"
    assert seen["path"] == "/api/jobs"
    assert seen["payload"]["arguments"] == {"prompt": "x", "seed": 11}
    assert "armA" in seen["payload"]["note"]
    assert seen["payload"]["acknowledged_cost"] is True


def test_measure_calls_compare_with_the_scene_crop_and_texture_bands(monkeypatch):
    calls = []
    cell = lambda fine, mid: {"texture": {"luma": {"fine": fine, "mid": mid}}}

    def fake(method, path, server, token, payload=None, params=None):
        calls.append((method, path, payload, params))
        if path.endswith("/assess"):
            return {"probe": "analyze_seams", "seams": [
                {"jump_ratio": 2.0, "look_ratio": 1.0, "level_step_db": -3.0},
                {"jump_ratio": 4.0, "look_ratio": 3.0, "level_step_db": 1.0}]}
        return {"texture": [{"cells": [cell(10.0, 20.0), cell(8.0, 20.0), cell(5.0, 30.0)]}]}

    monkeypatch.setattr(driver, "http", fake)
    scene = driver.Scene("s", [1, 2, 64, 64], ["frame:1", "frame:50", "frame:99"])
    job = {"id": "j", "manifest": [{"files": ["wf/v1/out.mp4"], "subfolder": "final"}],
           "output_kinds": {"wf/v1/out.mp4": "video"}, "started_at": 0, "finished_at": 120}
    m = driver.measure(job, scene, "http://h", "tok")

    assess = next(c for c in calls if c[1].endswith("/assess"))
    assert assess[3]["probe"] == "analyze_seams"
    compare = next(c for c in calls if c[1] == "/api/gallery/compare")
    assert compare[2]["names"] == ["wf/v1/out.mp4"]
    assert compare[2]["crop"] == [1, 2, 64, 64]
    assert compare[2]["at"] == scene.moments
    assert compare[2]["texture_bands"] is True
    assert m["seam_jump_mean"] == 3.0 and m["seam_jump_max"] == 4.0
    assert m["seam_look_mean"] == 2.0 and m["seam_look_max"] == 3.0
    assert m["seam_level_step_db_max"] == 3.0
    assert m["texture_fine_seg_last"] == 0.5 and m["texture_mid_seg_last"] == 1.5
    assert m["wall_minutes"] == 2.0
