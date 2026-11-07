"""chain_repair_cut: re-render one chain segment in place between its
neighbours (#1169) - the splice arithmetic and the cut."""

import os
import shutil

import numpy
import pytest
from PIL import Image

from dw.media_types import AudioVideo
from dw.tasks.chain_repair import (
    chain_repair_cut,
    chain_repair_splice,
    repair_geometry,
    segment_files,
)
from dw.tasks.task import _COMMAND_REGISTRY
from dw.tasks.video_utils import load_audio_video
from dw.task_domains import frames_to_samples

# A 5-segment guide chain at the chained-segments template's defaults: 124
# frames a segment, 22 carried, so every segment after the first is 102
CHAIN = [124, 102, 102, 102, 102]


def splice_length(segment_frames, segment, geometry):
    """The take's frame count after the splice the template makes: the
    segments before k, the kept span of the render, segment k+1 from the
    overrun on, and the segments after it."""
    k = segment
    return (
        sum(segment_frames[: k - 1])
        + geometry["keep_frames"]
        + segment_frames[k]
        - geometry["tail_start"]
        + sum(segment_frames[k + 1 :])
    )


class TestGeometry:
    def test_default_chain_middle_segment(self):
        geometry = repair_geometry(CHAIN, 3, 175, 22, 39)
        assert geometry["minimum_num_frames"] == 175
        assert geometry["overrun"] == 12
        assert geometry["tail_frame"] == 136
        assert geometry["head_start"] == 80
        assert geometry["keep_start"] == 22
        assert geometry["keep_frames"] == 114
        assert splice_length(CHAIN, 3, geometry) == sum(CHAIN)

    def test_equal_holds_match_the_parked_plan(self):
        # plan v1 on #1169: 124/22, segment 3 of 5 -> render 158, tail at 136
        geometry = repair_geometry(CHAIN, 3, 158, 22, 22)
        assert geometry["minimum_num_frames"] == 158
        assert geometry["tail_frame"] == 136
        assert geometry["tail_start"] == 12

    @pytest.mark.parametrize(
        "frames, segment",
        [
            (CHAIN, 2),
            (CHAIN, 4),
            ([124, 102, 102], 2),
            ([345, 306, 306, 200], 2),  # a short last segment (duration_seconds)
            ([345, 306, 306, 200], 3),
            ([124, 122, 122, 122], 2),  # last_frame continuity, trim_frames 2
            ([124, 85, 85, 85], 3),  # guide_frames 39
        ],
    )
    @pytest.mark.parametrize("head, tail", [(22, 22), (22, 39), (39, 56)])
    def test_the_splice_keeps_the_take_length_and_the_grid(
        self, frames, segment, head, tail
    ):
        shortest = head + frames[segment - 1] + tail
        num_frames = shortest + (5 - shortest) % 17
        try:
            geometry = repair_geometry(frames, segment, num_frames, head, tail)
        except ValueError as error:
            # Only a neighbour too short for the tail guide may refuse
            assert "tail guide would be" in str(error)
            return
        assert splice_length(frames, segment, geometry) == sum(frames)
        assert geometry["total_frames"] == sum(frames)
        assert geometry["tail_frame"] % 17 == 0
        assert geometry["tail_frame"] + tail == num_frames
        assert 0 <= geometry["overrun"] < 17
        # The kept span sits exactly between the two guides
        assert geometry["keep_start"] == head
        assert (
            geometry["keep_start"] + geometry["keep_frames"] == geometry["tail_frame"]
        )

    def test_a_longer_render_reaches_further_into_the_next_segment(self):
        geometry = repair_geometry(CHAIN, 3, 192, 22, 39)
        assert geometry["overrun"] == 29
        assert splice_length(CHAIN, 3, geometry) == sum(CHAIN)

    @pytest.mark.parametrize("segment", [1, 5, 0, 6])
    def test_the_first_and_last_segment_are_refused(self, segment):
        with pytest.raises(ValueError, match="one-sided repair"):
            repair_geometry(CHAIN, segment, 175, 22, 39)

    def test_a_render_too_short_names_the_smallest(self):
        with pytest.raises(ValueError, match="pass num_frames=175"):
            repair_geometry(CHAIN, 3, 158, 22, 39)

    def test_an_off_grid_render_is_refused(self):
        with pytest.raises(ValueError, match="17n \\+ 5"):
            repair_geometry(CHAIN, 3, 176, 22, 39)

    @pytest.mark.parametrize("name, head, tail", [("head", 21, 39), ("tail", 22, 30)])
    def test_a_hold_off_the_latent_grid_is_refused(self, name, head, tail):
        with pytest.raises(ValueError, match=f"{name}_frames"):
            repair_geometry(CHAIN, 3, 175, head, tail)

    def test_a_short_next_segment_is_refused(self):
        with pytest.raises(ValueError, match="tail guide would be"):
            repair_geometry([124, 102, 102, 40], 3, 175, 22, 39)

    def test_two_segments_are_too_few(self):
        with pytest.raises(ValueError, match="at least 3"):
            repair_geometry([124, 102], 2, 175, 22, 39)


FPS = 24
RATE = 48000
TAKE = "Wf-step.0-0.0.mp4"


def _frames(start, count, size):
    """Frames whose red channel is their index on the take."""
    frames = numpy.zeros((count, size[1], size[0], 3), dtype=numpy.uint8)
    frames[..., 0] = (numpy.arange(start, start + count) % 250)[:, None, None]
    return frames


def write_chain(directory, lengths, base="Wf-step.0", size=(64, 32), suffix=""):
    """Segment files as SegmentSpill names them, each frame's red channel its
    index on the take and its audio a level of its own."""
    from dw.writers import encode_video
    import torch

    paths, start = [], 0
    for index, count in enumerate(lengths):
        samples = frames_to_samples(count, FPS, RATE)
        audio = numpy.full((1, samples), 0.1 + 0.05 * index, dtype=numpy.float32)
        path = os.path.join(directory, f"{base}.segment-{index:03d}{suffix}.mp4")
        encode_video(
            torch.from_numpy(_frames(start, count, size)),
            fps=FPS,
            output_path=path,
            audio=torch.from_numpy(audio),
            audio_sample_rate=RATE,
        )
        paths.append(path)
        start += count
    return paths


def write_run(
    root,
    lengths,
    take_frames=None,
    *,
    match_audio=False,
    crossfade_ms=None,
    manifest=True,
    suffix="",
    size=(64, 32),
):
    """A chain run as the engine leaves it: segment files and the take in
    <run>/final, and a manifest whose shots name the segment files.

    `take_frames` below the files' total is a duration_seconds take (cut on
    replay); `match_audio` lays a 44.1 kHz track under it whole, whose shots
    carry no sample side; `crossfade_ms` records the chain's blend on every
    seam."""
    import json

    import torch

    from dw.writers import encode_video

    final = os.path.join(root, "run", "final")
    os.makedirs(final, exist_ok=True)
    paths = write_chain(final, lengths, size=size, suffix=suffix)
    total = sum(lengths) if take_frames is None else take_frames
    rate = 44100 if match_audio else RATE
    noise = numpy.random.default_rng(7).uniform(
        -0.5, 0.5, (1, frames_to_samples(total, FPS, rate))
    )
    take = os.path.join(final, TAKE)
    encode_video(
        torch.from_numpy(_frames(0, total, size)),
        fps=FPS,
        output_path=take,
        audio=torch.from_numpy(noise.astype(numpy.float32)),
        audio_sample_rate=rate,
    )
    shots, start = [], 0
    for index, count in enumerate(lengths):
        count = min(count, total - start)
        shot = {
            "name": f"segment {index + 1}",
            "start_frame": start,
            "num_frames": count,
            "start_sample": None,
            "num_samples": None,
            "segment_file": os.path.basename(paths[index]),
        }
        if not match_audio:
            first = frames_to_samples(start, FPS, rate)
            shot["start_sample"] = first
            shot["num_samples"] = frames_to_samples(start + count, FPS, rate) - first
        if index and crossfade_ms:
            shot["crossfade_ms"] = crossfade_ms
            shot["trim_frames"] = 22
        shots.append(shot)
        start += count
    if manifest:
        with open(os.path.join(root, "run", "manifest.json"), "w") as file:
            json.dump(
                {
                    "steps": [
                        {"step": "step", "files": [f"final/{TAKE}"], "shots": shots}
                    ]
                },
                file,
            )
    return {"take": take, "segments": paths, "shots": shots}


def stand_in_render(cut, level=0.7):
    """The render as H3 would return it: the head guide, new white frames,
    the tail guide, over a track of its own."""
    new = [Image.new("RGB", cut["head_guide"].frames[0].size, (255, 255, 255))] * (
        cut["tail_frame"] - cut["keep_start"]
    )
    frames = list(cut["head_guide"].frames) + new + list(cut["tail_guide"].frames)
    samples = frames_to_samples(len(frames), FPS, RATE)
    return AudioVideo(
        frames, numpy.full((1, samples), level, dtype=numpy.float32), RATE, fps=FPS
    )


def repaired(run, segment=2, num_frames=107, tail_frames=22):
    cut = chain_repair_cut(
        chain_take=run["take"],
        segment=segment,
        num_frames=num_frames,
        tail_frames=tail_frames,
    )
    render = stand_in_render(cut)
    spliced = chain_repair_splice(
        chain_take=cut["chain_take"],
        video=render,
        segment=segment,
        start_frame=cut["start_frame"],
        end_frame=cut["end_frame"],
        keep_start=cut["keep_start"],
    )
    return cut, spliced, load_audio_video(run["take"])


def red(frame):
    return numpy.asarray(frame)[10, 10, 0]


@pytest.fixture
def run(tmp_path):
    return write_run(str(tmp_path), [40, 51, 51, 51])


class TestSegmentFiles:
    def test_the_take_shots_name_the_files(self, tmp_path):
        # A stale set from another run, under the names the regex matches,
        # beside the run's own '-2' deduped files the take's shots name
        made = write_run(str(tmp_path), [40, 51, 51, 51], suffix="-2")
        write_chain(str(tmp_path / "run" / "final"), [5, 5, 5], size=(32, 16))
        cut = chain_repair_cut(
            chain_take=made["take"], segment=2, num_frames=107, tail_frames=22
        )
        assert cut["segment_files"] == made["segments"]
        assert cut["head_guide"].frames[0].size == (64, 32)

    def test_the_take_is_found_from_a_segment_file(self, run):
        cut = chain_repair_cut(
            chain_output=run["segments"][1], segment=2, num_frames=107, tail_frames=22
        )
        assert cut["chain_take"] == run["take"]

    def test_a_segment_file_of_another_take_is_refused(self, run, tmp_path):
        other = write_chain(str(tmp_path), [5], base="Wf-other.0")
        with pytest.raises(ValueError, match="not one of"):
            chain_repair_cut(
                chain_take=run["take"],
                chain_output=other[0],
                segment=2,
                num_frames=107,
                tail_frames=22,
            )

    def test_neither_input_is_refused(self):
        with pytest.raises(ValueError, match="chain_take"):
            chain_repair_cut(segment=2, num_frames=107)

    def test_without_a_manifest_a_segment_file_needs_the_take(self, tmp_path):
        made = write_run(str(tmp_path), [40, 51, 51, 51], manifest=False)
        with pytest.raises(ValueError, match="chain_take"):
            chain_repair_cut(
                chain_output=made["segments"][0],
                segment=2,
                num_frames=107,
                tail_frames=22,
            )

    def _edit_shots(self, made, edit):
        import json

        path = os.path.join(
            os.path.dirname(os.path.dirname(made["take"])), "manifest.json"
        )
        with open(path) as file:
            manifest = json.load(file)
        edit(manifest["steps"][0]["shots"])
        with open(path, "w") as file:
            json.dump(manifest, file)

    def _cut(self, made):
        return chain_repair_cut(
            chain_take=made["take"], segment=2, num_frames=107, tail_frames=22
        )

    def test_a_take_without_keep_segments_names_that_fix(self, tmp_path):
        made = write_run(str(tmp_path), [40, 51, 51, 51])
        self._edit_shots(
            made, lambda shots: [shot.pop("segment_file") for shot in shots]
        )
        with pytest.raises(ValueError, match="keep_segments: true"):
            self._cut(made)

    def test_a_take_copied_away_names_chain_output(self, tmp_path):
        made = write_run(str(tmp_path), [40, 51, 51, 51], manifest=False)
        with pytest.raises(ValueError, match="copied away.*chain_output"):
            self._cut(made)

    def test_a_take_beside_a_manifest_but_not_its_segments_names_chain_output(
        self, tmp_path
    ):
        made = write_run(str(tmp_path), [40, 51, 51, 51])
        run = os.path.dirname(os.path.dirname(made["take"]))
        copy_run = tmp_path / "copy"
        os.makedirs(copy_run / "final")
        shutil.copy(made["take"], copy_run / "final")
        shutil.copy(os.path.join(run, "manifest.json"), copy_run)
        with pytest.raises(ValueError, match="copied away.*chain_output"):
            chain_repair_cut(
                chain_take=str(copy_run / "final" / os.path.basename(made["take"])),
                segment=2,
                num_frames=107,
                tail_frames=22,
            )

    def test_an_already_repaired_take_names_the_original(self, tmp_path):
        made = write_run(str(tmp_path), [40, 51, 51, 51])

        def mark(shots):
            for shot in shots:
                shot.pop("segment_file")
            shots[1]["name"] = "segment 2 (repaired)"

        self._edit_shots(made, mark)
        with pytest.raises(ValueError, match="already a repaired take.*original"):
            self._cut(made)

    def test_without_a_manifest_the_siblings_are_found_by_name(self, tmp_path):
        # The by-name fallback finds the files only; the take must still
        # have shots, so the cut refuses it
        made = write_run(str(tmp_path), [40, 51, 51, 51], manifest=False)
        assert segment_files(made["segments"][2]) == made["segments"]
        with pytest.raises(ValueError, match="chain's own output with its shots"):
            chain_repair_cut(
                chain_take=made["take"],
                chain_output=made["segments"][2],
                segment=2,
                num_frames=107,
                tail_frames=22,
            )

    def test_siblings_that_disagree_are_refused(self, tmp_path):
        # Two runs' files mixed by the flat layout: one is another size
        made = write_run(str(tmp_path), [40, 51, 51, 51], manifest=False)
        (other,) = write_chain(str(tmp_path), [51], size=(32, 16))
        os.replace(other, made["segments"][3])
        with pytest.raises(ValueError, match="disagree"):
            chain_repair_cut(
                chain_take=made["take"],
                chain_output=made["segments"][0],
                segment=2,
                num_frames=107,
                tail_frames=22,
            )

    def test_a_missing_segment_is_refused(self, tmp_path):
        made = write_run(str(tmp_path), [40, 51, 51, 51], manifest=False)
        os.remove(made["segments"][1])
        with pytest.raises(ValueError, match="segment-001 is missing"):
            segment_files(made["segments"][0])

    def test_the_neighbours_are_counted_by_their_decode(self, run, monkeypatch):
        # The header is not read for the neighbours the cut decodes anyway
        import dw.tasks.chain_repair as chain_repair

        monkeypatch.setattr(
            chain_repair, "probe_metadata", lambda path: {"frame_count": 999}
        )
        cut = chain_repair_cut(
            chain_take=run["take"], segment=2, num_frames=107, tail_frames=22
        )
        assert cut["segment_frames"][0] == 40 and cut["segment_frames"][2] == 51


class TestCut:
    def test_the_cut(self, run):
        # 51-frame segments: head 22 + 51 + tail 22 = 95 -> 107, overrun 12
        cut = chain_repair_cut(
            chain_take=run["take"],
            segment=2,
            num_frames=107,
            tail_frames=22,
            width=64,
            height=32,
        )
        assert cut["segments"] == 4
        assert cut["overrun"] == 12 and cut["tail_frame"] == 85
        assert cut["start_frame"] == 40 and cut["end_frame"] == 103
        assert cut["total_frames"] == 193
        assert len(cut["head_guide"].frames) == 22
        assert len(cut["tail_guide"].frames) == 22
        assert cut["head_guide"].audio is not None
        assert cut["tail_guide"].audio is not None
        assert cut["keyframe"] is cut["head_guide"].frames[0]
        # The head guide is segment 1's last 22 frames, the tail guide
        # segment 3's from the overrun on
        assert abs(int(red(cut["head_guide"].frames[0])) - 18) <= 3
        assert abs(int(red(cut["tail_guide"].frames[0])) - (91 + 12)) <= 3

    def test_a_canvas_other_than_the_chains_is_refused(self, run):
        with pytest.raises(ValueError, match="chain's canvas"):
            chain_repair_cut(
                chain_take=run["take"],
                segment=2,
                num_frames=107,
                tail_frames=22,
                width=960,
                height=544,
            )

    def test_the_last_segment_is_refused(self, run):
        with pytest.raises(ValueError, match="one-sided repair"):
            chain_repair_cut(
                chain_take=run["take"], segment=4, num_frames=107, tail_frames=22
            )

    def test_registered(self):
        assert "chain_repair_cut" in _COMMAND_REGISTRY
        assert "chain_repair_splice" in _COMMAND_REGISTRY


class TestSplice:
    """Apart from the repaired span, the output IS the take."""

    def test_the_take_outside_the_span(self, tmp_path):
        made = write_run(str(tmp_path), [40, 51, 51, 51], crossfade_ms=75)
        cut, spliced, take = repaired(made)
        assert len(spliced.frames) == len(take.frames) == 193
        start, end = cut["start_frame"], cut["end_frame"]
        for index in list(range(start)) + list(range(end, 193)):
            assert numpy.array_equal(
                numpy.asarray(spliced.frames[index]), numpy.asarray(take.frames[index])
            ), index
        assert all(red(frame) == 255 for frame in spliced.frames[start:end])
        # Every other seam's crossfaded audio, bit for bit
        s0 = made["shots"][1]["start_sample"]
        s1 = made["shots"][2]["start_sample"] + frames_to_samples(12, FPS, RATE)
        assert spliced.audio.shape == take.audio.shape
        assert numpy.array_equal(spliced.audio[:, :s0], take.audio[:, :s0])
        assert numpy.array_equal(spliced.audio[:, s1:], take.audio[:, s1:])
        assert spliced.sample_rate == take.sample_rate
        # And its shots: every one outside the repaired pair is the take's,
        # its seam records kept; no shot names a segment file the splice
        # output does not have
        assert len(spliced.shots) == len(take.shots)
        own = [
            {k: v for k, v in shot.items() if k != "segment_file"}
            for shot in take.shots
        ]
        for index in (0, 3):
            assert spliced.shots[index] == own[index], index
        assert not any("segment_file" in shot for shot in spliced.shots)

    def test_the_repaired_seams_are_recorded_as_the_chain_records_one(self, tmp_path):
        made = write_run(str(tmp_path), [40, 51, 51, 51], crossfade_ms=75)
        cut, spliced, take = repaired(made)
        repaired_shot, resumed = spliced.shots[1], spliced.shots[2]
        # The render's head guide trimmed off before the span, and the take's
        # own segment 3 head the span replaced
        assert repaired_shot["trim_frames"] == cut["keep_start"]
        assert resumed["trim_frames"] == cut["end_frame"] - take.shots[2]["start_frame"]
        assert repaired_shot["crossfade_ms"] == resumed["crossfade_ms"] == 75.0

    def test_the_untouched_crossfaded_seams_keep_their_record(self, tmp_path):
        from dw.tasks.assess import analyze_seams

        made = write_run(str(tmp_path), [40, 51, 51, 51, 51], crossfade_ms=75)
        cut, spliced, take = repaired(made)
        before = analyze_seams(take)["seams"]
        after = analyze_seams(spliced)["seams"]
        assert len(after) == len(before) == 4
        for index in (2, 3):
            assert after[index]["crossfade_ms"] == before[index]["crossfade_ms"] == 75

    def test_the_span_is_the_render_blended_as_the_chain_blended(self, tmp_path):
        made = write_run(str(tmp_path), [40, 51, 51, 51], crossfade_ms=75)
        cut, spliced, take = repaired(made)
        s0 = made["shots"][1]["start_sample"]
        s1 = made["shots"][2]["start_sample"] + frames_to_samples(12, FPS, RATE)
        window = int(0.075 * RATE)
        # In from the take's own track, out into it again, the chain's 75 ms
        assert spliced.audio[0, s0] == pytest.approx(take.audio[0, s0], abs=1e-6)
        assert numpy.allclose(spliced.audio[:, s0 + window : s1 - window], 0.7)
        assert not numpy.allclose(spliced.audio[:, s1 - window // 2], 0.7)
        assert spliced.shots[1]["crossfade_ms"] == 75.0
        assert spliced.shots[2]["crossfade_ms"] == 75.0

    def test_a_guide_chain_seam_gets_the_declick(self, run):
        cut, spliced, take = repaired(run)
        assert spliced.shots[1]["crossfade_ms"] == 3.0
        assert spliced.shots[2]["crossfade_ms"] == 3.0

    def test_a_duration_take_keeps_its_frame_count(self, tmp_path):
        # duration_seconds: the spill holds 193 frames, the take 172
        for segment in (2, 3):
            made = write_run(str(tmp_path / str(segment)), [40, 51, 51, 51], 172)
            cut, spliced, take = repaired(made, segment=segment)
            assert len(take.frames) == 172
            assert len(spliced.frames) == 172
            assert spliced.audio.shape == take.audio.shape

    def test_a_match_audio_take_keeps_its_source_track(self, tmp_path):
        made = write_run(str(tmp_path), [40, 51, 51, 51], 180, match_audio=True)
        cut, spliced, take = repaired(made)
        assert take.sample_rate == 44100
        assert spliced.sample_rate == 44100
        assert numpy.array_equal(spliced.audio, take.audio)
        assert len(spliced.frames) == 180

    def test_the_repaired_seams_are_measured(self, run):
        from dw.tasks.assess import analyze_seams

        cut, spliced, take = repaired(run)
        names = [shot["name"] for shot in spliced.shots]
        assert names == ["segment 1", "segment 2 (repaired)", "segment 3", "segment 4"]
        assert spliced.shots[1]["start_frame"] == cut["start_frame"]
        assert spliced.shots[2]["start_frame"] == cut["end_frame"]
        assert sum(shot["num_frames"] for shot in spliced.shots) == 193
        assert not any(shot.get("hard_cut") for shot in spliced.shots)
        assert all(not seam["hard_cut"] for seam in analyze_seams(spliced)["seams"])

    @pytest.mark.parametrize("match_audio", [False, True])
    def test_a_take_without_shots_is_refused_by_the_cut(self, tmp_path, match_audio):
        # Before any render step runs: the cut is the template's first step
        made = write_run(
            str(tmp_path), [40, 51, 51, 51], match_audio=match_audio, manifest=False
        )
        with pytest.raises(ValueError, match="chain's own output with its shots"):
            chain_repair_cut(
                chain_take=made["take"],
                chain_output=made["segments"][0],
                segment=2,
                num_frames=107,
                tail_frames=22,
            )

    @pytest.mark.parametrize("match_audio", [False, True])
    def test_a_take_without_its_shots_is_refused(self, tmp_path, match_audio):
        # The shots go after the cut: the splice refuses on its own too
        made = write_run(str(tmp_path), [40, 51, 51, 51], match_audio=match_audio)
        cut = chain_repair_cut(
            chain_take=made["take"], segment=2, num_frames=107, tail_frames=22
        )
        os.remove(os.path.join(str(tmp_path), "run", "manifest.json"))
        with pytest.raises(ValueError, match="chain's own output with its shots"):
            chain_repair_splice(
                chain_take=cut["chain_take"],
                video=stand_in_render(cut),
                segment=2,
                start_frame=cut["start_frame"],
                end_frame=cut["end_frame"],
                keep_start=cut["keep_start"],
            )

    def test_a_render_on_another_canvas_is_refused(self, run):
        cut = chain_repair_cut(
            chain_take=run["take"], segment=2, num_frames=107, tail_frames=22
        )
        render = stand_in_render(cut)
        render.frames = [frame.resize((32, 16)) for frame in render.frames]
        with pytest.raises(ValueError, match="canvas"):
            chain_repair_splice(
                chain_take=run["take"],
                video=render,
                segment=2,
                start_frame=cut["start_frame"],
                end_frame=cut["end_frame"],
                keep_start=cut["keep_start"],
            )


TEMPLATES = (
    __import__("pathlib").Path(__file__).resolve().parent.parent
    / "workflows"
    / "templates"
    / "minimax"
)
TEMPLATE = TEMPLATES / "repair-segment.json"


def template_errors(arguments=None):
    import json

    from dw.workflow import Workflow

    definition = json.loads(TEMPLATE.read_text())
    workflow = Workflow(definition, str(TEMPLATE.parent), str(TEMPLATE))
    return workflow.validation_errors(arguments)


def chain_blocks(definition):
    if isinstance(definition, dict):
        if isinstance(definition.get("chain"), dict):
            yield definition["chain"]
        for value in definition.values():
            yield from chain_blocks(value)
    elif isinstance(definition, list):
        for value in definition:
            yield from chain_blocks(value)


class TestTemplate:
    def test_validates_with_its_defaults(self):
        assert template_errors() == []

    def test_validates_with_sample_arguments(self):
        assert (
            template_errors(
                {
                    "segment": 3,
                    "num_frames": 192,
                    "tail_frames": 39,
                    "prompt": "A glass figure turns towards the light.",
                    "seed": 7,
                }
            )
            == []
        )

    def test_segment_1_is_refused_before_the_run(self):
        errors = template_errors({"segment": 1})
        assert any("segment" in str(error) for error in errors), errors

    @pytest.mark.parametrize(
        "arguments", [{"head_frames": 39}, {"tail_frames": 56}, {"tail_frames": 5}]
    )
    def test_holds_past_the_vram_fold_are_refused(self, arguments):
        errors = template_errors(arguments)
        assert any(next(iter(arguments)) in str(error) for error in errors), errors

    def test_the_vram_fold_is_an_upper_bound_on_every_allowed_render(self):
        import json

        definition = json.loads(TEMPLATE.read_text())
        rate = definition["vram_estimate"]["bytes_per_voxel"]
        constraints = definition["variable_constraints"]
        head = constraints["head_frames"]
        tail = constraints["tail_frames"]
        assert head["min_frames"] == head["max_frames"] == 22
        assert (tail["min_frames"], tail["max_frames"]) == (22, 39)
        frames = constraints["num_frames"]
        for num_frames in range(frames["min_frames"], frames["max_frames"] + 1, 17):
            # The t2va rate on the render and on both guides' frames
            assert rate * num_frames >= 28.71 * (num_frames + 22 + 39), num_frames
        assert "exact" not in definition["vram_estimate"]["reason"]

    def test_the_segment_bound_says_it_is_no_frame_count(self):
        import json

        constraints = json.loads(TEMPLATE.read_text())["variable_constraints"]
        assert "not a frame count" in constraints["segment"]["reason"]

    def test_the_steps_cut_render_and_splice_the_take(self):
        import json

        definition = json.loads(TEMPLATE.read_text())
        steps = {s["name"]: s for s in definition["steps"]}
        assert list(steps) == ["cut", "repair", "splice"]
        assert steps["cut"]["task"]["command"] == "chain_repair_cut"
        assert steps["cut"]["task"]["arguments"]["chain_take"] == (
            "variable:chain_take"
        )
        guides = steps["repair"]["pipeline"]["arguments"]["guides"]
        assert [g["frame"] for g in guides] == [0, "previous_result:cut.tail_frame"]
        assert all(g["audio"] is True for g in guides)
        splice = steps["splice"]["task"]
        assert splice["command"] == "chain_repair_splice"
        assert splice["arguments"]["chain_take"] == "previous_result:cut.chain_take"
        assert splice["arguments"]["video"] == "previous_result:repair"
        # No made-up output: path stands in for the caller's take
        assert definition["variables"]["chain_take"] is None
        assert definition["variables"]["chain_output"] is None

    def test_the_default_render_fits_the_default_chain(self):
        import json

        variables = json.loads(TEMPLATE.read_text())["variables"]
        geometry = repair_geometry(
            CHAIN,
            variables["segment"],
            variables["num_frames"],
            variables["head_frames"],
            variables["tail_frames"],
        )
        assert geometry["minimum_num_frames"] == variables["num_frames"]
        assert variables["tail_frames"] > variables["head_frames"]
        assert variables["seed"] is None

    @pytest.mark.parametrize(
        "name",
        [
            "chained-segments.json",
            "chain-matched-to-audio.json",
            "chain-matched-and-aligned.json",
            "chain-video-continuity.json",
        ],
    )
    def test_the_h3_chains_can_keep_their_segments(self, name):
        import json

        definition = json.loads((TEMPLATES / name).read_text())
        assert definition["variables"]["keep_segments"] is False
        (chain,) = list(chain_blocks(definition))
        assert chain["keep_segments"] == "variable:keep_segments"
        assert chain["save_segments"] in ("variable:keep_segments", True)
        assert "repair-segment" in definition["description"]


@pytest.mark.parametrize("name", ["save_segments", "keep_segments"])
@pytest.mark.parametrize(
    "value, valid",
    [(True, True), ("variable:keep_segments", True), ("false", False)],
)
def test_the_segment_flags_take_a_boolean_or_a_variable(name, value, valid):
    # A literal "false" would read as bool("false") == True
    import json

    import jsonschema

    schema = json.loads(
        (TEMPLATES.parent.parent.parent / "dw" / "workflow_schema.json").read_text()
    )

    def find(node):
        if isinstance(node, dict):
            if name in node and isinstance(node[name], dict) and "type" in node[name]:
                return node[name]
            for child in node.values():
                if (found := find(child)) is not None:
                    return found
        elif isinstance(node, list):
            for child in node:
                if (found := find(child)) is not None:
                    return found
        return None

    entry = find(schema)
    errors = list(jsonschema.Draft7Validator(entry).iter_errors(value))
    assert (not errors) == valid, errors


def test_the_cut_feeds_the_splice_in_a_workflow(run, tmp_path, monkeypatch):
    """The template's task steps as a workflow, with a stand-in for the H3
    render: the record's fields reach the splice as one value each, and the
    saved splice keeps the take's length."""
    import av

    from dw.workflow import Workflow

    monkeypatch.setenv("DW_TRUST_WORKFLOWS", "1")
    definition = {
        "id": "repair-wiring",
        "steps": [
            {
                "name": "cut",
                "task": {
                    "command": "chain_repair_cut",
                    "arguments": {
                        "chain_take": run["take"],
                        "segment": 2,
                        "num_frames": 107,
                        "tail_frames": 22,
                    },
                },
            },
            {
                # Stands in for the render: 107 frames of the take with audio
                "name": "repair",
                "task": {
                    "command": "trim_video",
                    "arguments": {
                        "video": run["take"],
                        "start_frame": 18,
                        "num_frames": 107,
                        "fps": 24,
                    },
                },
            },
            {
                "name": "splice",
                "task": {
                    "command": "chain_repair_splice",
                    "arguments": {
                        "chain_take": "previous_result:cut.chain_take",
                        "video": "previous_result:repair",
                        "segment": 2,
                        "start_frame": "previous_result:cut.start_frame",
                        "end_frame": "previous_result:cut.end_frame",
                        "keep_start": "previous_result:cut.keep_start",
                    },
                },
                "result": {"content_type": "video/mp4", "fps": 24},
            },
        ],
    }
    out = tmp_path / "out"
    Workflow(definition, str(out), "").run({})
    saved = [p for p in out.rglob("*.mp4")]
    assert len(saved) == 1, saved
    with av.open(str(saved[0])) as container:
        frames = sum(1 for _ in container.decode(video=0))
    assert frames == 193
