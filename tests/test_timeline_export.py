"""Tests for export_timeline (dw/tasks/timeline_export.py, #1197).

Header probing is replaced by a table of facts per file, so no video is
decoded: the files are placeholder bytes, which is all the zip needs.
"""

import json
import os
import zipfile
from fractions import Fraction

import pytest

import dw.media
import dw.tasks.timeline_export as te
from dw.locations import location_errors
from dw.security import InvalidInputError, SecurityError
from dw.shots import shot_record
from dw.tasks.task import task_command_info
from dw.trust import TRUST_WORKFLOWS_ENV_VAR

NTSC = 24000 / 1001


@pytest.fixture
def facts(monkeypatch):
    table = {}
    monkeypatch.setattr(dw.media, "probe_metadata", lambda path: table.get(path))
    return table


def a_file(directory, name, facts, frames, fps=24.0, sample_rate=48000):
    path = os.path.join(str(directory), name)
    with open(path, "wb") as file:
        file.write(b"mp4:" + name.encode())
    facts[path] = {
        "fps": fps,
        "frame_count": frames,
        "sample_rate": sample_rate,
    }
    return path


def shots(*spans):
    return [
        shot_record(f"shot {i}", start, count) for i, (start, count) in enumerate(spans)
    ]


def bundle_of(video, tmp_path, **kwargs):
    out = tmp_path / "out.otioz"
    te.export_timeline(video, **kwargs).save(str(out))
    archive = zipfile.ZipFile(out)
    return archive, json.loads(archive.read("content.otio"))


def frames_of(item):
    return item["source_range"]["duration"]["value"]


def test_the_bundle_has_its_three_members(tmp_path, facts):
    video = a_file(tmp_path, "joined.mp4", facts, 72)
    archive, content = bundle_of(
        video, tmp_path, shots=shots((0, 24), (24, 48)), name="cut"
    )
    names = archive.namelist()
    assert archive.read("version.txt").decode().strip() == "1.0.0"
    assert "content.otio" in names
    assert names.count("media/joined.mp4") == 1
    assert len([n for n in names if n.startswith("media/")]) == 1
    assert content["OTIO_SCHEMA"] == "Timeline.1"
    assert content["name"] == "cut"


def test_one_clip_per_shot_on_a_video_and_an_audio_track(tmp_path, facts):
    video = a_file(tmp_path, "joined.mp4", facts, 72)
    _, content = bundle_of(video, tmp_path, shots=shots((0, 24), (24, 48)))
    stack = content["tracks"]
    assert stack["OTIO_SCHEMA"] == "Stack.1"
    assert [t["kind"] for t in stack["children"]] == ["Video", "Audio"]
    for track in stack["children"]:
        assert track["OTIO_SCHEMA"] == "Track.1"
        clips = track["children"]
        assert [c["OTIO_SCHEMA"] for c in clips] == ["Clip.1", "Clip.1"]
        assert [c["name"] for c in clips] == ["shot 0", "shot 1"]
        assert [c["source_range"]["start_time"]["value"] for c in clips] == [0, 24]
        reference = clips[0]["media_reference"]
        assert reference["OTIO_SCHEMA"] == "ExternalReference.1"
        assert reference["target_url"] == "media/joined.mp4"
        assert reference["available_range"]["duration"]["value"] == 72


def test_a_silent_video_has_no_audio_track(tmp_path, facts):
    video = a_file(tmp_path, "mute.mp4", facts, 24, sample_rate=None)
    _, content = bundle_of(video, tmp_path)
    assert [t["kind"] for t in content["tracks"]["children"]] == ["Video"]


def test_clips_and_gaps_sum_to_the_videos_duration(tmp_path, facts):
    video = a_file(tmp_path, "joined.mp4", facts, 100)
    _, content = bundle_of(video, tmp_path, shots=shots((10, 20), (50, 30)))
    for track in content["tracks"]["children"]:
        kinds = [c["OTIO_SCHEMA"] for c in track["children"]]
        assert kinds == ["Gap.1", "Clip.1", "Gap.1", "Clip.1", "Gap.1"]
        assert sum(frames_of(c) for c in track["children"]) == 100


def test_a_video_with_no_shots_is_one_clip(tmp_path, facts):
    video = a_file(tmp_path, "joined.mp4", facts, 50)
    _, content = bundle_of(video, tmp_path)
    clips = content["tracks"]["children"][0]["children"]
    assert [frames_of(c) for c in clips] == [50]


def test_a_fractional_rate_is_the_exact_rational(tmp_path, facts):
    video = a_file(tmp_path, "ntsc.mp4", facts, 48, fps=23.976)
    _, content = bundle_of(video, tmp_path, shots=shots((0, 24), (24, 24)))
    for item in content["tracks"]["children"][0]["children"]:
        for part in (
            item["source_range"]["start_time"],
            item["source_range"]["duration"],
        ):
            assert part["rate"] == NTSC
            assert Fraction(part["rate"]).limit_denominator(1001) == Fraction(
                24000, 1001
            )
            assert isinstance(part["value"], int)


def test_sources_give_each_clip_its_own_file_over_its_full_range(tmp_path, facts):
    video = a_file(tmp_path, "joined.mp4", facts, 60)
    a = a_file(tmp_path, "a.mp4", facts, 24)
    b = a_file(tmp_path, "b.mp4", facts, 36)
    archive, content = bundle_of(
        video, tmp_path, shots=shots((0, 24), (24, 36)), sources=[a, b]
    )
    clips = content["tracks"]["children"][0]["children"]
    assert [c["media_reference"]["target_url"] for c in clips] == [
        "media/a.mp4",
        "media/b.mp4",
    ]
    assert [frames_of(c) for c in clips] == [24, 36]
    assert [c["source_range"]["start_time"]["value"] for c in clips] == [0, 0]
    assert {"media/a.mp4", "media/b.mp4"} <= set(archive.namelist())


def test_sources_must_match_the_shots(tmp_path, facts):
    video = a_file(tmp_path, "joined.mp4", facts, 60)
    a = a_file(tmp_path, "a.mp4", facts, 24)
    with pytest.raises(Exception, match="one per shot"):
        te.export_timeline(video, shots=shots((0, 24), (24, 36)), sources=[a])


def test_same_named_sources_stay_apart(tmp_path, facts):
    (tmp_path / "x").mkdir()
    (tmp_path / "y").mkdir()
    video = a_file(tmp_path, "joined.mp4", facts, 48)
    a = a_file(tmp_path / "x", "s.mp4", facts, 24)
    b = a_file(tmp_path / "y", "s.mp4", facts, 24)
    archive, content = bundle_of(
        video, tmp_path, shots=shots((0, 24), (24, 24)), sources=[a, b]
    )
    urls = [
        c["media_reference"]["target_url"]
        for c in content["tracks"]["children"][0]["children"]
    ]
    assert len(set(urls)) == 2
    assert set(urls) <= set(archive.namelist())


def test_the_bundle_path_is_confined_by_the_result_writer(tmp_path, facts):
    # The writer joins the result's name onto the step's output directory
    # and confines it there (dw/writers.py output_file_path) before save()
    # is ever called - save() re-validating against a base derived from the
    # path itself would confine nothing
    from dw.writers import output_file_path

    video = a_file(tmp_path, "joined.mp4", facts, 24)
    bundle = te.export_timeline(video)
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(SecurityError):
        bundle.save(output_file_path(str(out), "../escape.otioz"))
    assert not (tmp_path / "escape.otioz").exists()


def test_save_does_not_derive_its_own_confinement(tmp_path, facts, monkeypatch):
    calls = []
    monkeypatch.setattr(
        te,
        "validate_output_path",
        lambda path, base: calls.append(base) or path,
        raising=False,
    )
    video = a_file(tmp_path, "joined.mp4", facts, 24)
    te.export_timeline(video).save(str(tmp_path / "out.otioz"))
    assert calls == []


def test_manifest_shots_are_found_through_a_location_dict(tmp_path, facts, monkeypatch):
    video = a_file(tmp_path, "joined.mp4", facts, 72)
    seen = []

    def beside(path):
        seen.append(path)
        return shots((0, 24), (24, 48))

    monkeypatch.setattr("dw.runs.shots_beside", beside)
    _, content = bundle_of({"location": video}, tmp_path)
    clips = content["tracks"]["children"][0]["children"]
    assert [frames_of(c) for c in clips] == [24, 48]
    assert seen == [video]


@pytest.mark.parametrize("sources", ["a.mp4", {"a": 1}, 3])
def test_sources_must_be_a_list(tmp_path, facts, sources):
    video = a_file(tmp_path, "joined.mp4", facts, 24)
    with pytest.raises(InvalidInputError, match="sources"):
        te.export_timeline(video, sources=sources)


@pytest.mark.parametrize(
    "records, match",
    [
        ("shot", "shots"),
        (["shot"], "shots\\[0\\]"),
        ([{"name": "a"}], "start_frame"),
        ([{"start_frame": "x", "num_frames": 4}], "start_frame"),
        ([{"start_frame": 0, "num_frames": None}], "num_frames"),
        ([{"start_frame": -1, "num_frames": 4}], "start_frame"),
        ([{"start_frame": 0, "num_frames": 0}], "num_frames"),
    ],
)
def test_malformed_shot_records_are_refused(tmp_path, facts, records, match):
    video = a_file(tmp_path, "joined.mp4", facts, 24)
    with pytest.raises(InvalidInputError, match=match):
        te.export_timeline(video, shots=records)


def test_overlapping_shots_are_refused(tmp_path, facts):
    # A track lays its clips end to end, so an overlap would lengthen the
    # timeline past the video and repeat frames - clips and gaps must add
    # up to the video's length
    video = a_file(tmp_path, "joined.mp4", facts, 72)
    with pytest.raises(InvalidInputError, match="overlap"):
        te.export_timeline(video, shots=shots((0, 30), (24, 48)))


def test_member_names_never_come_from_a_path(tmp_path, facts):
    video = a_file(tmp_path, "joined.mp4", facts, 24)
    bundle = te.export_timeline(video)
    for arcname in bundle.media:
        assert arcname.startswith("media/") and ".." not in arcname


def test_an_unreadable_video_is_refused(monkeypatch, tmp_path, facts):
    monkeypatch.setenv(TRUST_WORKFLOWS_ENV_VAR, "0")
    with pytest.raises(SecurityError):
        te.export_timeline("/etc/hostname")


def test_a_clip_in_memory_is_refused(facts):
    with pytest.raises(Exception, match="stored video"):
        te.export_timeline(object())


def test_sources_is_a_media_argument(monkeypatch, tmp_path):
    monkeypatch.setenv(TRUST_WORKFLOWS_ENV_VAR, "0")
    definition = {
        "steps": [
            {
                "name": "t",
                "task": {
                    "command": "export_timeline",
                    "arguments": {"video": "asset:a.mp4", "sources": ["/etc/hostname"]},
                },
            }
        ]
    }
    errors = location_errors(definition, base_dir=str(tmp_path))
    assert [e["path"] for e in errors] == ["steps[0].task.arguments.sources[0]"]


def test_the_command_is_registered_with_an_otioz_content_type():
    info = task_command_info("export_timeline")
    assert info is not None
    from dw.content_types import guess_extension

    assert guess_extension("application/x-otioz") == ".otioz"
