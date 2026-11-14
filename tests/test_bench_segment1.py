import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))

import pytest  # noqa: E402
import segment1  # noqa: E402

HEADER = (
    b"#format: frame checksums\n"
    b"#version: 2\n"
    b"#hash: MD5\n"
    b"#tb 0: 1/24\n"
    b"#media_type 0: video\n"
    b"#codec_id 0: rawvideo\n"
    b"#dimensions 0: 960x544\n"
    b"#sar 0: 1/1\n"
    b"#stream#, dts,        pts, duration,     size, hash\n"
)


def framemd5_bytes(frames, salt=b""):
    """A framemd5 output with one line per frame, each a distinct hash."""
    lines = [
        b"0, %10d, %10d, %8d, %8d, %s"
        % (
            i,
            i,
            1,
            783360,
            hashlib.md5(salt + bytes([i % 256, i // 256])).hexdigest().encode(),
        )
        for i in range(frames)
    ]
    return HEADER + b"\n".join(lines) + b"\n"


def expected(video, audio):
    def digest(text, count):
        hashes = [
            line.split(",")[-1].strip()
            for line in text.splitlines()
            if not line.startswith("#")
        ]
        return hashlib.md5("".join(hashes[:count]).encode()).hexdigest()[:8]

    return f"{digest(video, 60)}/{digest(audio, 130)}"


def test_fingerprint_hashes_only_segment_one_frames():
    video = framemd5_bytes(124).decode()
    audio = framemd5_bytes(161, b"a").decode()
    fp = segment1.fingerprint_from_framemd5(video, audio)
    assert fp == expected(video, audio)
    # A departure after the window (segment 2) leaves the fingerprint alone.
    later_video = video.replace(video.splitlines()[-1].split(", ")[-1], "0" * 32)
    later_audio = framemd5_bytes(161, b"a").decode().rsplit("\n", 2)[0] + "\n"
    assert segment1.fingerprint_from_framemd5(later_video, later_audio) == fp
    # A departure inside segment 1 changes it.
    first = video.splitlines()[9].split(", ")[-1]
    assert (
        segment1.fingerprint_from_framemd5(video.replace(first, "f" * 32), audio) != fp
    )


def test_fingerprint_refuses_a_short_stream():
    with pytest.raises(ValueError, match="audio has 78 frames"):
        segment1.fingerprint_from_framemd5(
            framemd5_bytes(60).decode(), framemd5_bytes(78).decode()
        )


def test_fingerprint_file_runs_one_stream_per_call_without_a_shell(monkeypatch):
    calls = []
    outputs = {
        "v": framemd5_bytes(60).decode(),
        "a": framemd5_bytes(130, b"a").decode(),
    }

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        stream = command[command.index("-map") + 1].split(":")[1]
        return subprocess.CompletedProcess(command, 0, stdout=outputs[stream])

    monkeypatch.setattr(segment1.subprocess, "run", fake_run)
    fp = segment1.fingerprint_file("x.mp4", ffmpeg="/opt/ffmpeg")
    assert fp == expected(outputs["v"], outputs["a"])
    assert [c[0][c[0].index("-map") + 1] for c in calls] == ["0:v:0", "0:a:0"]
    assert all(isinstance(c[0], list) and not c[1].get("shell") for c in calls)
    assert "-frames:v" in calls[0][0] and "-frames:a" in calls[1][0]


def test_classify_labels_clean_drifted_and_lora_arms():
    canonical = {
        "s1-static@1166": {"fingerprint": "aaaa/bbbb"},
        "s1-static@1166+lora": {"fingerprint": "cccc/dddd"},
    }
    records = [
        {
            "arm": "ctl",
            "scene": "s1-static",
            "seed": 1166,
            "output": "o1",
            "job_id": "1",
        },
        {
            "arm": "arm",
            "scene": "s1-static",
            "seed": 1166,
            "output": "o2",
            "job_id": "2",
        },
        {
            "arm": "lora",
            "scene": "s1-static",
            "seed": 1166,
            "output": "o3",
            "job_id": "3",
        },
        {
            "arm": "seed2",
            "scene": "s1-static",
            "seed": 2,
            "output": "o4",
            "job_id": "4",
        },
        {"arm": "gone", "scene": "s1-static", "seed": 1166, "status": "failed"},
    ]
    fingerprints = {
        "o1": "aaaa/bbbb",
        "o2": "aaaa/eeee",
        "o3": "cccc/dddd",
        "o4": "x/y",
    }
    rows = segment1.classify(
        records,
        fingerprints,
        canonical,
        {"lora": {"antiburn_model_name": "Imagineer13/H3-Anti-Burn-LoRA"}},
    )
    assert [(r["arm"], r["key"], r["label"]) for r in rows] == [
        ("ctl", "s1-static@1166", "clean"),
        ("arm", "s1-static@1166", "drifted"),
        ("lora", "s1-static@1166+lora", "clean"),
        ("seed2", "s1-static@2", "no canonical"),
    ]


def test_the_committed_labels_match_the_committed_fingerprints():
    records = Path(__file__).resolve().parent.parent / "bench" / "records"
    canonical = json.loads((records / "segment1-canonical.json").read_text())
    fingerprints = json.loads((records / "segment1-fingerprints.json").read_text())
    for key, entry in canonical.items():
        assert entry["fingerprint"] in fingerprints.values(), key
