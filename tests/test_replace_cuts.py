"""replace_cuts: re-rendered cuts go back into the take, whose other frames
and whole soundtrack stay (0.21.0 review).

replace-performer joined only the cuts it was given and paired the take's
whole track under them, so a cut left out of the middle was missing from the
result and every later cut played against the wrong sound. The chain is run
here on a take whose every frame and every sample says where it came from.
"""

import copy
import json
from pathlib import Path
from unittest.mock import patch

import numpy
import pytest
from PIL import Image

from dw.media_types import AudioVideo
from dw.task_domains import frames_to_samples, task_argument_errors
from dw.tasks.concat_videos import concat_videos
from dw.tasks.pair_audio import pair_audio
from dw.tasks.task import _COMMAND_REGISTRY
from dw.tasks.trim import cut_list_problem, replace_cuts, trim_video
from tests.test_h3_mask import write_video

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "workflows" / "templates" / "minimax" / "replace-performer.json"
FPS, RATE, TOTAL = 24, 48000, 360
CUTS = [
    {"name": "cut-1", "start_frame": 0, "num_frames": 120, "render_frames": 124},
    {"name": "cut-2", "start_frame": 120, "num_frames": 120, "render_frames": 124},
    {"name": "cut-3", "start_frame": 240, "num_frames": 120, "render_frames": 124},
]
MIDDLE_LEFT_OUT = [CUTS[0], CUTS[2]]


def take(total=TOTAL, shots=None):
    """A take that encodes its own time: frame i is the colour (i % 256,
    i // 256, 0), and sample s of its track has the value s."""
    frames = [Image.new("RGB", (4, 4), (i % 256, i // 256, 0)) for i in range(total)]
    samples = frames_to_samples(total, FPS, RATE)
    audio = numpy.arange(samples, dtype=numpy.float32).reshape(1, -1)
    return AudioVideo(frames, audio, RATE, fps=FPS, shots=shots)


def source_frame(frame):
    """The take frame a result frame shows, and whether a render drew it."""
    red, green, blue = frame.getpixel((0, 0))
    return red + 256 * green, blue == 255


def render(clip):
    """The H3 step, stubbed: every frame marked as rendered (blue 255), the
    picture otherwise the source's, the held audio passed through."""
    marked = []
    for frame in clip.frames:
        red, green, _ = frame.getpixel((0, 0))
        marked.append(Image.new("RGB", frame.size, (red, green, 255)))
    return AudioVideo(marked, clip.audio, clip.sample_rate, fps=clip.fps)


def rendered_cuts(source, cuts):
    """The template's per-cut chain: source_cut, swap (stubbed), cut."""
    clips = []
    for cut in cuts:
        padded = trim_video(
            source, cut["start_frame"], cut["num_frames"], pad_to=cut["render_frames"]
        )
        clips.append(trim_video(render(padded), 0, cut["num_frames"]))
    return clips


def sound_under(result, frame):
    """The source time, in frames, of the sound a result frame plays over."""
    sample = result.audio[0, frames_to_samples(frame, FPS, RATE)]
    return float(sample) * FPS / RATE


class TestPictureAndSoundAgree:
    def test_the_join_it_replaced_slid_later_cuts_under_the_wrong_sound(self):
        """What 0.21.0's template did with the middle cut left out: 240
        frames, and frame 120 shows take frame 240 over the take's sound at
        frame 120 - five seconds apart."""
        source = take()
        joined = concat_videos(
            rendered_cuts(source, MIDDLE_LEFT_OUT), trim_frames=0, crossfade_ms=0
        )
        old = pair_audio(joined, source, fit="video")
        assert len(old.frames) == 240
        assert source_frame(old.frames[120])[0] == 240
        assert sound_under(old, 120) == 120

    def test_a_cut_left_out_of_the_middle_stays_the_takes(self):
        source = take()
        result = replace_cuts(
            source, rendered_cuts(source, MIDDLE_LEFT_OUT), MIDDLE_LEFT_OUT
        )
        assert len(result.frames) == TOTAL
        assert numpy.array_equal(result.audio, source.audio)
        assert result.fps == FPS and result.sample_rate == RATE
        for index, frame in enumerate(result.frames):
            shown, was_rendered = source_frame(frame)
            assert shown == index
            assert was_rendered == (not 120 <= index < 240), index
            assert sound_under(result, index) == index

    @pytest.mark.parametrize("boundary", [120, 240])
    def test_each_cut_boundary_is_frame_exact(self, boundary):
        source = take()
        result = replace_cuts(
            source, rendered_cuts(source, MIDDLE_LEFT_OUT), MIDDLE_LEFT_OUT
        )
        for index in (boundary - 1, boundary):
            assert source_frame(result.frames[index])[0] == index
            assert sound_under(result, index) == index
        before = source_frame(result.frames[boundary - 1])[1]
        after = source_frame(result.frames[boundary])[1]
        assert before != after

    @pytest.mark.parametrize(
        "cuts",
        [CUTS, CUTS[:1], CUTS[1:2], CUTS[2:], CUTS[:2], CUTS[1:]],
        ids=["all", "first", "middle", "last", "first-two", "last-two"],
    )
    def test_any_subset_of_the_cuts_keeps_the_takes_length_and_track(self, cuts):
        source = take()
        result = replace_cuts(source, rendered_cuts(source, cuts), cuts)
        assert len(result.frames) == TOTAL
        assert numpy.array_equal(result.audio, source.audio)
        listed = {
            index
            for cut in cuts
            for index in range(
                cut["start_frame"], cut["start_frame"] + cut["num_frames"]
            )
        }
        for index, frame in enumerate(result.frames):
            assert source_frame(frame) == (index, index in listed)

    def test_a_cut_that_is_part_of_a_shot_leaves_the_rest_of_it(self):
        source = take()
        cuts = [
            {"name": "a", "start_frame": 50, "num_frames": 30, "render_frames": 124}
        ]
        result = replace_cuts(source, rendered_cuts(source, cuts), cuts)
        drawn = [i for i, f in enumerate(result.frames) if source_frame(f)[1]]
        assert drawn == list(range(50, 80))

    def test_a_padded_render_is_cut_back_to_the_cuts_length(self):
        """The clip's first num_frames frames are used, so the held frames a
        pad_to render ends on never reach the take."""
        source = take()
        cut = CUTS[1]
        padded = render(trim_video(source, 120, 120, pad_to=124))
        result = replace_cuts(source, [padded], [cut])
        assert len(result.frames) == TOTAL
        assert [source_frame(f)[0] for f in result.frames] == list(range(TOTAL))

    def test_the_takes_shots_are_carried(self):
        shots = [
            {"name": "one", "start_frame": 0, "num_frames": 200},
            {"name": "two", "start_frame": 200, "num_frames": 160},
        ]
        source = take(shots=shots)
        result = replace_cuts(source, rendered_cuts(source, CUTS[:1]), CUTS[:1])
        assert [s["name"] for s in result.shots] == ["one", "two"]
        assert result.shots is not source.shots

    def test_the_take_is_left_as_it_was(self):
        source = take()
        replace_cuts(source, rendered_cuts(source, CUTS[:1]), CUTS[:1])
        assert not any(source_frame(frame)[1] for frame in source.frames)


class TestRefusals:
    def test_out_of_order_says_to_list_in_the_takes_order(self):
        cuts = [CUTS[2], CUTS[0]]
        problem = cut_list_problem(cuts)
        assert "cuts[1] ('cut-1') starts at frame 0" in problem
        assert "before cuts[0] ('cut-3') at frame 240" in problem
        assert "list the cuts in the take's order" in problem
        assert "left out of the list" in problem
        source = take()
        with pytest.raises(ValueError, match="take's order"):
            replace_cuts(source, rendered_cuts(source, cuts), cuts)

    def test_overlap_says_how_to_fix_it(self):
        cuts = [CUTS[0], dict(CUTS[1], start_frame=100)]
        problem = cut_list_problem(cuts)
        assert "cuts[1] ('cut-2') starts at frame 100" in problem
        assert "inside cuts[0] ('cut-1'), which runs to frame 119" in problem
        assert "Start it at frame 120 or later" in problem
        assert "merge the two" in problem
        source = take()
        with pytest.raises(ValueError, match="cannot cover the same frames"):
            replace_cuts(source, rendered_cuts(source, cuts), cuts)

    def test_the_same_cut_twice_is_an_overlap(self):
        assert "inside" in cut_list_problem([CUTS[0], CUTS[0]])

    @pytest.mark.parametrize(
        "cuts",
        [CUTS, MIDDLE_LEFT_OUT, CUTS[1:], [], "variable:cuts", None],
        ids=["all", "gap", "tail", "empty", "reference", "none"],
    )
    def test_an_ordered_list_with_or_without_gaps_is_no_problem(self, cuts):
        assert cut_list_problem(cuts) is None

    def test_an_unresolved_span_is_not_judged(self):
        cuts = [CUTS[2], dict(CUTS[0], start_frame="item:start")]
        assert cut_list_problem(cuts) is None

    def test_a_clip_count_that_differs_is_refused(self):
        source = take()
        with pytest.raises(ValueError, match=r"1 clip\(s\) for 2 cut\(s\)"):
            replace_cuts(source, rendered_cuts(source, CUTS[:1]), CUTS[:2])

    def test_a_cut_past_the_takes_end_is_refused(self):
        source = take()
        cut = dict(CUTS[2], num_frames=121)
        clip = render(trim_video(source, 236, 124))
        with pytest.raises(ValueError, match="the clip has only 360 frames"):
            replace_cuts(source, [clip], [cut])

    def test_a_clip_shorter_than_its_cut_is_refused(self):
        source = take()
        short = trim_video(source, 0, 100)
        with pytest.raises(ValueError, match="100 frames, fewer than"):
            replace_cuts(source, [short], CUTS[:1])

    def test_a_clip_of_another_size_is_refused(self):
        source = take()
        big = AudioVideo([Image.new("RGB", (8, 8))] * 120, None, None, fps=FPS)
        with pytest.raises(ValueError, match="8x8 and the take is 4x4"):
            replace_cuts(source, [big], CUTS[:1])

    @pytest.mark.parametrize("cut", [{"name": "x"}, "cut-1", {"start_frame": 0}])
    def test_a_cut_without_a_span_is_refused(self, cut):
        source = take()
        with pytest.raises(ValueError, match="replace_cuts"):
            replace_cuts(source, rendered_cuts(source, CUTS[:1]), [cut])

    def test_no_take_and_no_cuts_are_refused(self):
        with pytest.raises(ValueError, match="'video'"):
            replace_cuts(None, [], CUTS[:1])
        with pytest.raises(ValueError, match="'cuts'"):
            replace_cuts(take(), [], [])


class TestValidate:
    """The same rule, before anything runs (task_argument_errors)."""

    @staticmethod
    def errors(cuts, clips=None):
        clips = ["previous_result:cut@x"] * len(cuts) if clips is None else clips
        step = {
            "name": "replaced",
            "task": {
                "command": "replace_cuts",
                "arguments": {"video": "take.mp4", "clips": clips, "cuts": cuts},
            },
        }
        return task_argument_errors({"steps": [step]})

    def test_registered(self):
        assert "replace_cuts" in _COMMAND_REGISTRY

    def test_a_gap_is_clean(self):
        assert self.errors(MIDDLE_LEFT_OUT) == []

    def test_out_of_order_is_an_error_at_cuts(self):
        (error,) = self.errors([CUTS[1], CUTS[0]])
        assert error["path"] == "steps[0].task.arguments.cuts"
        assert "list the cuts in the take's order" in error["message"]

    def test_overlap_is_an_error_at_cuts(self):
        (error,) = self.errors([CUTS[0], dict(CUTS[1], start_frame=119)])
        assert error["path"] == "steps[0].task.arguments.cuts"
        assert "merge the two" in error["message"]

    def test_a_clip_count_that_differs_is_an_error_at_clips(self):
        (error,) = self.errors(CUTS, clips=["previous_result:cut@x"])
        assert error["path"] == "steps[0].task.arguments.clips"


# ---------------------------------------------------------------------------
# The template itself, through the engine: for_each, gather and the file read
# ---------------------------------------------------------------------------

LEVELS = (40, 120, 200)  # each cut of the take file, as a flat grey


def _write_take(path):
    """A real mp4 of three 24-frame cuts, each its own grey, with a track."""
    import av

    container = av.open(str(path), "w")
    video = container.add_stream("libx264", rate=FPS)
    video.width, video.height, video.pix_fmt = 64, 48, "yuv420p"
    audio = container.add_stream("aac", rate=16000)
    for level in LEVELS:
        pixels = numpy.full((48, 64, 3), level, dtype=numpy.uint8)
        for _ in range(24):
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            container.mux(video.encode(frame))
    container.mux(video.encode())
    ramp = numpy.linspace(-0.5, 0.5, 16000 * 3, dtype=numpy.float32).reshape(1, -1)
    frame = av.AudioFrame.from_ndarray(ramp, format="flt", layout="mono")
    frame.sample_rate = 16000
    container.mux(audio.encode(frame))
    container.mux(audio.encode())
    container.close()
    return path.name


def _stubbed_template():
    """replace-performer with its H3 step replaced by a task that passes the
    trimmed source through: every other step is the template's own."""
    template = json.loads(TEMPLATE.read_text())
    definition = {
        "id": "replace_performer_stubbed",
        "variables": {"source_video": None, "mask_video": None, "cuts": []},
        "steps": [],
    }
    for step in template["steps"]:
        step = copy.deepcopy(step)
        if step["name"] == "swap":
            step = {
                "name": "swap",
                "for_each": "variable:cuts",
                "task": {
                    "command": "trim_video",
                    "arguments": {
                        "video": "previous_result:source_cut",
                        "start_frame": 0,
                        "num_frames": "item:render_frames",
                    },
                },
            }
        definition["steps"].append(step)
    return definition


def _run_stubbed(tmp_path, cuts):
    from dw.result import Result
    from dw.step_cache import step_cache
    from dw.workflow import Workflow

    name = _write_take(tmp_path / "take.mp4")
    write_video(tmp_path / "mask.mp4", frames=72, value=255)
    seen = []
    original = Result.save

    def spy_save(self, *args, **kwargs):
        if (self.result_definition or {}).get("subfolder") == "final":
            seen.extend(self.result_list)
        return original(self, *args, **kwargs)

    step_cache.clear()
    workflow = Workflow(_stubbed_template(), str(tmp_path), str(tmp_path / "wf.json"))
    with patch.object(Result, "save", spy_save):
        workflow.run({"source_video": name, "mask_video": "mask.mp4", "cuts": cuts})
    (result,) = seen
    return result


def _grey(frame):
    return float(numpy.asarray(frame.convert("L")).mean())


FILE_CUTS = [
    {"name": "cut-1", "start_frame": 0, "num_frames": 24, "render_frames": 30},
    {"name": "cut-2", "start_frame": 24, "num_frames": 24, "render_frames": 30},
    {"name": "cut-3", "start_frame": 48, "num_frames": 24, "render_frames": 30},
]


@pytest.mark.parametrize(
    "cuts",
    [FILE_CUTS, [FILE_CUTS[0], FILE_CUTS[2]], FILE_CUTS[2:]],
    ids=["all", "middle-left-out", "last-only"],
)
def test_the_template_returns_the_whole_take_whatever_cuts_are_listed(tmp_path, cuts):
    """The template's own steps around a stubbed render: the result is the
    take's length, each cut where the take has it, under the take's track."""
    from dw.tasks.video_utils import load_audio_video

    result = _run_stubbed(tmp_path, cuts)
    source = load_audio_video(str(tmp_path / "take.mp4"))
    assert isinstance(result, AudioVideo)
    assert len(result.frames) == len(source.frames) == 72
    assert numpy.array_equal(result.audio, source.audio)
    for index, level in enumerate(LEVELS):
        for frame in (index * 24, index * 24 + 23):
            assert abs(_grey(result.frames[frame]) - level) < 12, frame


def test_the_template_ends_on_replace_cuts_over_the_whole_take():
    template = json.loads(TEMPLATE.read_text())
    last = template["steps"][-1]
    assert last["task"] == {
        "command": "replace_cuts",
        "arguments": {
            "video": "variable:source_video",
            "clips": "gather:cut",
            "cuts": "variable:cuts",
        },
    }
    commands = [s.get("task", {}).get("command") for s in template["steps"]]
    assert "concat_videos" not in commands and "pair_audio" not in commands
