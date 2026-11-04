"""chain_repair_cut: re-render one chain segment in place between its
neighbours (#1169) - the splice arithmetic and the cut."""

import os

import numpy
import pytest
from PIL import Image

from dw.media_types import AudioVideo
from dw.tasks.chain_repair import chain_repair_cut, repair_geometry, segment_files
from dw.tasks.concat_videos import concat_videos
from dw.tasks.task import _COMMAND_REGISTRY
from dw.tasks.trim import trim_video
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


def write_chain(directory, lengths, base="Wf-step.0", size=(64, 32)):
    """Segment files as SegmentSpill names them, each frame's red channel its
    index on the take and its audio a ramp of the take's sample index."""
    from dw.writers import encode_video
    import torch

    paths, start = [], 0
    for index, count in enumerate(lengths):
        frames = numpy.zeros((count, size[1], size[0], 3), dtype=numpy.uint8)
        frames[..., 0] = (numpy.arange(start, start + count) % 250)[:, None, None]
        samples = frames_to_samples(count, FPS, RATE)
        audio = numpy.full((1, samples), 0.1 + 0.05 * index, dtype=numpy.float32)
        path = os.path.join(directory, f"{base}.segment-{index:03d}.mp4")
        encode_video(
            torch.from_numpy(frames),
            fps=FPS,
            output_path=path,
            audio=torch.from_numpy(audio),
            audio_sample_rate=RATE,
        )
        paths.append(path)
        start += count
    return paths


@pytest.fixture
def chain(tmp_path):
    return write_chain(str(tmp_path), [40, 51, 51, 51])


class TestSegmentFiles:
    def test_finds_every_sibling_in_order(self, chain, tmp_path):
        # Another chain's files, and an unrelated file, beside them
        write_chain(str(tmp_path), [5], base="Wf-other.0")
        (tmp_path / "Wf-step.0-0.0.mp4").write_bytes(b"")
        assert segment_files(chain[2]) == chain

    def test_a_file_that_is_no_segment_is_refused(self, tmp_path):
        path = tmp_path / "take.mp4"
        path.write_bytes(b"")
        with pytest.raises(ValueError, match="save_segments and keep_segments"):
            segment_files(str(path))

    def test_a_missing_segment_is_refused(self, chain):
        os.remove(chain[1])
        with pytest.raises(ValueError, match="segment-001 is missing"):
            segment_files(chain[0])


class TestCut:
    def test_the_cut_and_the_splice(self, chain):
        # 51-frame segments: head 22 + 51 + tail 22 = 95 -> 107, overrun 12
        cut = chain_repair_cut(chain[0], 2, 107, 22, 22, width=64, height=32)
        assert cut["segments"] == 4
        assert cut["segment_frames"] == [40, 51, 51, 51]
        assert cut["overrun"] == 12 and cut["tail_frame"] == 85
        assert len(cut["head_guide"].frames) == 22
        assert len(cut["tail_guide"].frames) == 22
        assert cut["head_guide"].audio is not None
        assert cut["tail_guide"].audio is not None
        assert cut["keyframe"] is cut["head_guide"].frames[0]
        assert cut["before"] == chain[:1]
        assert len(cut["after"]) == 2
        assert len(cut["after"][0].frames) == 51 - 12
        assert cut["after"][1:] == chain[3:]

        # A stand-in render: the head guide, new frames, the tail guide
        new = [Image.new("RGB", (64, 32), (255, 255, 255))] * cut["keep_frames"]
        render_frames = (
            list(cut["head_guide"].frames) + new + list(cut["tail_guide"].frames)
        )
        samples = frames_to_samples(len(render_frames), FPS, RATE)
        render = AudioVideo(
            render_frames,
            numpy.zeros((1, samples), dtype=numpy.float32),
            RATE,
            fps=FPS,
        )
        assert len(render.frames) == 107
        kept = trim_video(render, cut["keep_start"], cut["keep_frames"])
        before = concat_videos(cut["before"], fps=FPS, continuous_joins=True)
        after = concat_videos(cut["after"], fps=FPS, continuous_joins=True)
        take = concat_videos([before, kept, after], fps=FPS, continuous_joins=True)
        assert len(take.frames) == sum(cut["segment_frames"])
        assert not any(shot.get("hard_cut") for shot in take.shots)
        # The take resumes on segment 3's own frame 12 right after the repair
        resume = 40 + cut["keep_frames"]
        red = take.frames[resume].getpixel((10, 10))[0]
        assert abs(red - (40 + 51 + 12)) <= 3

    def test_a_canvas_other_than_the_chains_is_refused(self, chain):
        with pytest.raises(ValueError, match="chain's canvas"):
            chain_repair_cut(chain[0], 2, 107, 22, 22, width=960, height=544)

    def test_the_last_segment_is_refused(self, chain):
        with pytest.raises(ValueError, match="one-sided repair"):
            chain_repair_cut(chain[0], 4, 107, 22, 22)

    def test_registered(self):
        assert "chain_repair_cut" in _COMMAND_REGISTRY


TEMPLATE = (
    __import__("pathlib").Path(__file__).resolve().parent.parent
    / "workflows"
    / "templates"
    / "minimax"
    / "repair-segment.json"
)


def template_errors(arguments=None):
    import json

    from dw.workflow import Workflow

    definition = json.loads(TEMPLATE.read_text())
    workflow = Workflow(definition, str(TEMPLATE.parent), str(TEMPLATE))
    return workflow.validation_errors(arguments)


class TestTemplate:
    def test_validates_with_its_defaults(self):
        assert template_errors() == []

    def test_validates_with_sample_arguments(self):
        assert (
            template_errors(
                {
                    "segment": 3,
                    "num_frames": 192,
                    "tail_frames": 56,
                    "prompt": "A glass figure turns towards the light.",
                    "seed": 7,
                }
            )
            == []
        )

    def test_segment_1_is_refused_before_the_run(self):
        errors = template_errors({"segment": 1})
        assert any("segment" in str(error) for error in errors), errors

    def test_the_splice_is_continuous_and_the_guides_hold_audio(self):
        import json

        steps = {s["name"]: s for s in json.loads(TEMPLATE.read_text())["steps"]}
        assert steps["cut"]["task"]["command"] == "chain_repair_cut"
        guides = steps["repair"]["pipeline"]["arguments"]["guides"]
        assert [g["frame"] for g in guides] == [0, "previous_result:cut.tail_frame"]
        assert all(g["audio"] is True for g in guides)
        for name in ("before", "after", "splice"):
            arguments = steps[name]["task"]["arguments"]
            assert steps[name]["task"]["command"] == "concat_videos"
            assert arguments["continuous_joins"] is True
        assert steps["splice"]["task"]["arguments"]["videos"] == [
            "previous_result:before",
            "previous_result:keep",
            "previous_result:after",
        ]

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


def test_the_cut_feeds_the_template_steps_in_a_workflow(chain, tmp_path, monkeypatch):
    """The template's task steps as a workflow, with a stand-in for the H3
    render: a record's list and number fields reach trim_video and
    concat_videos as one value each, and the splice keeps the take's length."""
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
                        "chain_output": chain[0],
                        "segment": 2,
                        "num_frames": 107,
                        "tail_frames": 22,
                    },
                },
            },
            {
                # Stands in for the render: its kept span is the head guide's
                # frames from overrun on (a span of the right shape)
                "name": "keep",
                "task": {
                    "command": "trim_video",
                    "arguments": {
                        "video": "previous_result:cut.head_guide",
                        "start_frame": "previous_result:cut.overrun",
                        "num_frames": 10,
                        "fps": 24,
                    },
                },
            },
            {
                "name": "before",
                "task": {
                    "command": "concat_videos",
                    "arguments": {
                        "videos": "previous_result:cut.before",
                        "fps": 24,
                        "continuous_joins": True,
                    },
                },
            },
            {
                "name": "after",
                "task": {
                    "command": "concat_videos",
                    "arguments": {
                        "videos": "previous_result:cut.after",
                        "fps": 24,
                        "continuous_joins": True,
                    },
                },
            },
            {
                "name": "splice",
                "task": {
                    "command": "concat_videos",
                    "arguments": {
                        "videos": [
                            "previous_result:before",
                            "previous_result:keep",
                            "previous_result:after",
                        ],
                        "fps": 24,
                        "continuous_joins": True,
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
    # segment 1 (40) + the 10-frame stand-in + segments 3 (51 - 12) and 4 (51)
    assert frames == 40 + 10 + 39 + 51
