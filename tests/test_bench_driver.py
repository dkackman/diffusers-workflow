import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bench"))

import pytest  # noqa: E402
import driver  # noqa: E402


def test_load_arms_reads_every_field(tmp_path):
    path = tmp_path / "arms.json"
    path.write_text(
        json.dumps(
            [
                {
                    "name": "a",
                    "scene": "s",
                    "workflow": "w.json",
                    "arguments": {"prompt": "x"},
                    "seed": 7,
                }
            ]
        )
    )
    (arm,) = driver.load_arms(str(path))
    assert (arm.name, arm.scene, arm.workflow, arm.arguments, arm.seed) == (
        "a",
        "s",
        "w.json",
        {"prompt": "x"},
        7,
    )


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

    def cell(fine, mid, coarse=4.0):
        return {"texture": {"luma": {"fine": fine, "mid": mid, "coarse": coarse}}}

    def fake(method, path, server, token, payload=None, params=None):
        calls.append((method, path, payload, params))
        if path.endswith("/assess"):
            return {
                "probe": "analyze_seams",
                "seams": [
                    {"jump_ratio": 2.0, "look_ratio": 1.0, "level_step_db": -3.0},
                    {"jump_ratio": 4.0, "look_ratio": 3.0, "level_step_db": 1.0},
                ],
            }
        return {
            "texture": [{"cells": [cell(10.0, 20.0), cell(8.0, 20.0), cell(5.0, 30.0)]}]
        }

    monkeypatch.setattr(driver, "http", fake)
    scene = driver.Scene("s", [1, 2, 64, 64], ["frame:1", "frame:50", "frame:99"])
    job = {
        "id": "j",
        "manifest": [{"files": ["wf/v1/out.mp4"], "subfolder": "final"}],
        "output_kinds": {"wf/v1/out.mp4": "video"},
        "started_at": 0,
        "finished_at": 120,
    }
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
    assert m["texture_fine_seg_last"] == pytest.approx(math.log(2))
    assert m["texture_mid_seg_last"] == pytest.approx(math.log(1.5))
    assert m["texture_coarse_seg_last"] == 0.0
    assert m["wall_minutes"] == 2.0


def _texture_measure(monkeypatch, first, last):
    def cell(v):
        return {"texture": {"luma": {"fine": v, "mid": 1.0, "coarse": 1.0}}}

    def fake(method, path, server, token, payload=None, params=None):
        if path.endswith("/assess"):
            return {"seams": []}
        return {"texture": [{"cells": [cell(first), cell(last)]}]}

    monkeypatch.setattr(driver, "http", fake)
    scene = driver.Scene("s", None, ["frame:1", "frame:2"])
    job = {
        "manifest": [{"files": ["o.mp4"], "subfolder": "final"}],
        "output_kinds": {"o.mp4": "video"},
    }
    return driver.measure(job, scene, "http://h", "tok")


def test_texture_drift_is_two_sided(monkeypatch):
    blur = _texture_measure(monkeypatch, 10.0, 5.0)["texture_fine_seg_last"]
    crust = _texture_measure(monkeypatch, 10.0, 20.0)["texture_fine_seg_last"]
    assert blur == pytest.approx(crust) and blur > 0
    assert _texture_measure(monkeypatch, 10.0, 10.0)["texture_fine_seg_last"] == 0.0
    assert "texture_fine_seg_last" not in _texture_measure(monkeypatch, 0.0, 5.0)
    assert "texture_fine_seg_last" not in _texture_measure(monkeypatch, 10.0, None)


def test_run_arms_writes_each_record_as_it_lands(monkeypatch, tmp_path):
    arms = [
        driver.Arm("a", "s", "w.json", {}, 1),
        driver.Arm("b", "s", "w.json", {}, 2),
    ]
    monkeypatch.setattr(driver, "submit", lambda arm, *a: f"id-{arm.name}")
    out = tmp_path / "sub" / "rec.json"
    seen = {}

    def fake_wait(job_id, *a):
        if job_id == "id-b":
            seen["after_first"] = json.loads(out.read_text())
            raise RuntimeError("poll died")
        return {"status": "succeeded"}

    monkeypatch.setattr(driver, "wait", fake_wait)
    monkeypatch.setattr(driver, "output_name", lambda job: "o.mp4")
    records = driver.run_arms(arms, {}, "http://h", "tok", False, 1, out_path=str(out))
    assert [r["arm"] for r in seen["after_first"]] == ["a"]
    saved = json.loads(out.read_text())
    assert [r["arm"] for r in saved] == ["a", "b"]
    assert "poll died" in saved[1]["error"] and records == saved


def test_load_arms_reads_optional_suffix_and_moments(tmp_path):
    path = tmp_path / "arms.json"
    path.write_text(
        json.dumps(
            [
                {
                    "name": "a",
                    "scene": "s",
                    "workflow": "w.json",
                    "seed": 7,
                    "workflow_id_suffix": "-noise-b",
                    "moments": ["frame:1"],
                }
            ]
        )
    )
    (arm,) = driver.load_arms(str(path))
    assert arm.workflow_id_suffix == "-noise-b" and arm.moments == ["frame:1"]


def test_submit_suffixes_the_workflow_id(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(
        driver,
        "http",
        lambda m, p, s, t, payload=None, params=None: (
            seen.update(payload=payload) or {"id": "j"}
        ),
    )
    monkeypatch.setattr(driver, "HERE", tmp_path)
    (tmp_path / "workflows").mkdir()
    (tmp_path / "workflows" / "w.json").write_text('{"id": "Wf", "steps": []}')
    arm = driver.Arm("a", "s", "w.json", {}, 1, workflow_id_suffix="-noise-b")
    driver.submit(arm, "http://h", "tok")
    assert seen["payload"]["workflow"]["id"] == "Wf-noise-b"


def test_measure_uses_the_arm_moments_override(monkeypatch):
    calls = []

    def fake(method, path, server, token, payload=None, params=None):
        calls.append(payload)
        return {"seams": []} if path.endswith("/assess") else {"texture": []}

    monkeypatch.setattr(driver, "http", fake)
    scene = driver.Scene("s", [0, 0, 8, 8], ["frame:1", "frame:2"])
    job = {"id": "j", "manifest": [{"files": ["o.mp4"], "subfolder": "final"}]}
    driver.measure(job, scene, "http://h", "tok", moments=["frame:9"])
    assert calls[-1]["at"] == ["frame:9"]


def _fake_server(compare_calls):
    def fake(method, path, server, token, payload=None, params=None):
        if path.startswith("/api/jobs/"):
            return {
                "id": path.rsplit("/", 1)[1],
                "status": "succeeded",
                "started_at": 0,
                "finished_at": 600,
                "manifest": [{"subfolder": "final", "files": ["out/x.mp4"]}],
                "output_kinds": {"out/x.mp4": "video"},
            }
        if path.endswith("/assess"):
            return {"seams": [{"jump_ratio": 2.0, "look_ratio": 1.0}]}
        compare_calls.append(payload)
        cell = {"texture": {"luma": {"fine": 10.0, "mid": 10.0, "coarse": 10.0}}}
        return {"texture": [{"cells": [cell, cell]}]}

    return fake


def test_measure_records_remeasures_succeeded_jobs_with_current_crops(
    monkeypatch, tmp_path
):
    calls = []
    monkeypatch.setattr(driver, "http", _fake_server(calls))
    scenes = tmp_path / "scenes.json"
    scenes.write_text(
        json.dumps(
            [{"name": "s", "crop": [5, 6, 7, 8], "moments": ["frame:3", "frame:9"]}]
        )
    )
    rec = tmp_path / "rec.json"
    rec.write_text(
        json.dumps(
            [
                {
                    "arm": "a",
                    "scene": "s",
                    "seed": 1,
                    "job_id": "j1",
                    "status": "succeeded",
                    "measures": {"stale": 1.0},
                },
                {
                    "arm": "b",
                    "scene": "s",
                    "seed": 1,
                    "job_id": "j2",
                    "status": "failed",
                    "error": "boom",
                },
            ]
        )
    )
    token = tmp_path / "t"
    token.write_text("tok")
    argv = ["--token-file", str(token), "measure", str(rec), "--scenes", str(scenes)]
    assert driver.main(argv) == 0
    saved = json.loads(rec.read_text())
    assert saved[0]["measures"]["seam_jump_mean"] == 2.0
    assert "stale" not in saved[0]["measures"]
    assert saved[0]["measures"]["wall_minutes"] == 10.0
    assert saved[1] == {
        "arm": "b",
        "scene": "s",
        "seed": 1,
        "job_id": "j2",
        "status": "failed",
        "error": "boom",
    }
    assert calls[0]["crop"] == [5, 6, 7, 8] and calls[0]["at"] == ["frame:3", "frame:9"]


def test_measure_records_takes_arm_moments_from_the_arms_file(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(driver, "http", _fake_server(calls))
    scenes = {"s": driver.Scene("s", [0, 0, 9, 9], ["frame:1", "frame:2"])}
    arms = [driver.Arm("a", "s", "w.json", {}, 1, moments=["frame:62", "frame:99"])]
    records = [{"arm": "a", "scene": "s", "job_id": "j1", "status": "succeeded"}]
    driver.measure_records(records, scenes, arms, "http://h", "tok")
    assert calls[0]["at"] == ["frame:62", "frame:99"]
