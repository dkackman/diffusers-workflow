"""MiniMax-H3 audio pass (#1192): a `source_video` with no mask and no
`hold_audio` keeps every video row and generates only the audio. The rules,
the block insertion, the two blocks on a synthetic pipeline state, the call's
argument preparation and the result's frames."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dw.hold_audio import mask_errors
from dw.pipeline_processors import h3_mask
from dw.pipeline_processors.h3_hold import core_denoise_sequences
from dw.pipeline_processors.h3_rules import (
    AUDIO_PASS_BLOCK,
    AUDIO_PASS_ROWS,
    HELD_VIDEO_OUTPUT,
    HELD_VIDEO_RATE_OUTPUT,
    LISTEN_SCALE_INPUT,
    REGENERATE_MASK,
    RELEASE_VIDEO_BLOCK,
    SOURCE_BLOCK,
    SOURCE_ROWS,
    edits_region,
    is_audio_pass,
    listen_canvas,
    mask_problems,
    refine_problems,
)

torch = pytest.importorskip("torch")
minimax = pytest.importorskip("diffusers.modular_pipelines.minimax_h3")
from diffusers.modular_pipelines.minimax_h3.denoise import (  # noqa: E402
    MiniMaxH3LoopSchedulerStep,
)
from diffusers.modular_pipelines.modular_pipeline import PipelineState  # noqa: E402
from diffusers.schedulers import MiniMaxH3Scheduler  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS = ("t2va", "fl2va", "ref2va")
# A 128x64 source, 5 frames (the test canvas, not H3's real grid)
WIDTH, HEIGHT, FRAMES = 128, 64, 5


def source_frames(width=WIDTH, height=HEIGHT, frames=FRAMES, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (frames, height, width, 3), dtype=np.uint8)


def rated(frames, fps=24):
    """Frames carrying the rate they play at, as a loaded video file's do -
    an audio pass refuses a source with no rate (R25)."""
    from dw.media_types import FrameList

    return FrameList(list(frames), fps)


def pipeline_with_blocks():
    pipeline = minimax.MiniMaxH3Blocks().get_workflow("t2va").init_pipeline()
    h3_mask.insert_mask_blocks(pipeline)
    return pipeline


def calling_step(pipeline, chained=False):
    return SimpleNamespace(
        name="sound",
        pipeline=pipeline,
        base_dir=None,
        segment_label="segment 1" if chained else None,
    )


def with_mask(pipeline, arguments, chained=False):
    from dw.pipeline_processors.pipeline import Pipeline

    return Pipeline._with_mask(calling_step(pipeline, chained), arguments)


def audio_pass_arguments(**overrides):
    arguments = {
        "source_video": rated(source_frames()),
        "prompt": "a voice says <d>[English] hello</d>",
        "output": ["videos", "audio", "sampling_rate"],
    }
    arguments.update(overrides)
    return {k: v for k, v in arguments.items() if v is not None}


# The three the brief names


def test_the_output_frames_are_the_sources_own():
    """The call carries the full-size source, the pipeline returns it beside
    its decode, and the result plays the source's pixels - not the decode's -
    under the generated track, at the rate H3 reports."""
    from dw.media_types import rated_videos
    from dw.output_extraction import modular_artifacts
    from PIL import Image

    source = source_frames()
    arguments = with_mask(
        pipeline_with_blocks(), audio_pass_arguments(source_video=rated(source))
    )
    assert arguments["output"] == [
        "videos",
        "audio",
        "sampling_rate",
        HELD_VIDEO_OUTPUT,
        HELD_VIDEO_RATE_OUTPUT,
    ]

    # What the pipeline would hand back: a VAE round trip that is not the source
    decoded = [Image.fromarray(np.zeros((HEIGHT, WIDTH, 3), np.uint8))] * FRAMES
    generated = torch.randn(1, 2, 4800)
    result = {
        "videos": rated_videos([decoded], 24),
        "audio": generated,
        "sampling_rate": 48000,
        HELD_VIDEO_OUTPUT: arguments[HELD_VIDEO_OUTPUT],
        HELD_VIDEO_RATE_OUTPUT: arguments[HELD_VIDEO_RATE_OUTPUT],
    }
    artifacts = modular_artifacts(result)

    assert len(artifacts) == 1
    played = artifacts[0]
    assert np.array_equal(np.stack([np.asarray(f) for f in played.frames]), source)
    assert played.fps == 24
    assert torch.equal(played.audio, generated[0])
    assert played.sample_rate == 48000


def test_listen_scale_half_runs_the_kept_rows_at_half_canvas():
    source = source_frames()
    arguments = with_mask(
        pipeline_with_blocks(),
        audio_pass_arguments(source_video=rated(source), listen_scale=0.5),
    )
    assert (arguments["width"], arguments["height"]) == (WIDTH // 2, HEIGHT // 2)
    assert arguments["source_video"].shape == (FRAMES, HEIGHT // 2, WIDTH // 2, 3)
    assert arguments["source_video"].dtype == np.uint8
    # The audio's length follows the frame count, which the scale leaves alone
    assert arguments["num_frames"] == FRAMES
    # The picture handed back is still the full-size source
    assert np.array_equal(arguments[HELD_VIDEO_OUTPUT], source)


def test_a_mask_or_hold_audio_keeps_todays_meaning():
    edit = {
        "source_video": "asset:take.mp4",
        "mask_image": "asset:m.png",
        "refine_strength": 0.9,
        "hold_audio": "asset:take.mp4",
        "num_inference_steps": 5,
    }
    assert not is_audio_pass(edit)
    assert mask_problems(edit) == [] and refine_problems(edit) == []

    whole = {k: v for k, v in edit.items() if k != "mask_image"}
    assert not is_audio_pass(whole)
    assert mask_problems(whole) == [] and refine_problems(whole) == []

    # A mask with no hold_audio is still refused for its soundtrack
    unheld = {k: v for k, v in edit.items() if k != "hold_audio"}
    assert not is_audio_pass(unheld)
    assert any("hold_audio" in p for p in refine_problems(unheld))

    # hold_audio with no refine_strength is still refused for the strength
    no_strength = {k: v for k, v in whole.items() if k != "refine_strength"}
    assert any("refine_strength" in p for p in mask_problems(no_strength))

    # And the call: a regional edit gets no held video and keeps its canvas
    from PIL import Image

    mask = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    mask[:8, :8] = 255
    arguments = with_mask(
        pipeline_with_blocks(),
        audio_pass_arguments(
            mask_image=Image.fromarray(mask), refine_strength=0.5, hold_audio=object()
        ),
    )
    assert HELD_VIDEO_OUTPUT not in arguments
    assert REGENERATE_MASK in arguments
    assert arguments["output"] == ["videos", "audio", "sampling_rate"]
    assert (arguments["width"], arguments["height"]) == (WIDTH, HEIGHT)


# Rules


class TestRules:
    def test_a_source_with_no_mask_and_no_hold_is_an_audio_pass(self):
        assert is_audio_pass({"source_video": "asset:take.mp4"})
        assert is_audio_pass({"source_video": "asset:take.mp4", "hold_audio": None})
        assert not is_audio_pass({"source_video": None})
        assert not is_audio_pass({"prompt": "x"})

    def test_an_audio_pass_needs_no_refine_strength(self):
        assert mask_problems({"source_video": "asset:take.mp4"}) == []

    @pytest.mark.parametrize("scale", [1, 1.0, 0.5, 0.25, "variable:scale"])
    def test_each_listen_scale_is_accepted(self, scale):
        arguments = {"source_video": "asset:take.mp4", LISTEN_SCALE_INPUT: scale}
        assert mask_problems(arguments) == []

    @pytest.mark.parametrize("scale", [0.3, 2, 0, True, "half"])
    def test_another_listen_scale_is_refused(self, scale):
        arguments = {"source_video": "asset:take.mp4", LISTEN_SCALE_INPUT: scale}
        problems = mask_problems(arguments)
        assert len(problems) == 1
        assert "listen_scale is one of 1, 0.5, 0.25" in problems[0]

    @pytest.mark.parametrize(
        "name, value",
        [("refine_strength", 0.9), ("refine_sigmas", [1.0, 0.0])],
    )
    def test_a_refine_argument_is_refused_once(self, name, value):
        arguments = {"source_video": "asset:take.mp4", name: value}
        problems = mask_problems(arguments)
        assert len(problems) == 1
        assert name in problems[0] and "audio pass" in problems[0]

    def test_refine_does_not_also_demand_hold_audio_on_an_audio_pass(self):
        arguments = {
            "source_video": "asset:take.mp4",
            "refine_strength": 0.9,
            "num_inference_steps": 5,
        }
        assert not any("hold_audio" in p for p in refine_problems(arguments))

    def test_listen_scale_on_a_regional_edit_is_refused(self):
        arguments = {
            "source_video": "asset:take.mp4",
            "mask_image": "asset:m.png",
            "refine_strength": 0.9,
            LISTEN_SCALE_INPUT: 0.5,
        }
        problems = mask_problems(arguments)
        assert len(problems) == 1 and "listen_scale" in problems[0]

    def test_listen_scale_without_a_source_is_refused(self):
        problems = mask_problems({LISTEN_SCALE_INPUT: 0.5})
        assert len(problems) == 1 and "audio pass" in problems[0]
        assert edits_region({LISTEN_SCALE_INPUT: 0.5}, True)
        assert not edits_region({LISTEN_SCALE_INPUT: 0.5}, False)

    def test_a_chained_audio_pass_is_refused(self):
        problems = mask_problems({"source_video": "asset:take.mp4"}, chained=True)
        assert len(problems) == 1 and "chain" in problems[0]

    def test_latents_beside_the_source_is_refused(self):
        problems = mask_problems({"source_video": "asset:take.mp4", "latents": "x"})
        assert len(problems) == 1 and "latents" in problems[0]

    def test_the_listen_canvas_snaps_to_32_px(self):
        assert listen_canvas(960, 544, 1) == (960, 544)
        assert listen_canvas(960, 544, 0.5) == (512, 288)
        assert listen_canvas(960, 544, 0.25) == (224, 128)
        assert listen_canvas(544, 960, 0.25) == (128, 224)
        assert listen_canvas(64, 32, 0.25) == (64, 32)


def h3_step(**arguments):
    return {
        "name": "sound",
        "pipeline": {
            "configuration": {"component_type": "ModularPipeline"},
            "from_pretrained_arguments": {"workflow": "t2va"},
            "arguments": arguments,
        },
    }


class TestStatic:
    def test_an_audio_pass_validates(self):
        assert mask_errors({"steps": [h3_step(source_video="asset:take.mp4")]}) == []

    def test_a_bad_listen_scale_is_refused_at_the_source_path(self):
        errors = mask_errors(
            {"steps": [h3_step(source_video="asset:take.mp4", listen_scale=0.3)]}
        )
        assert len(errors) == 1
        assert errors[0]["path"].endswith("source_video")
        assert "listen_scale" in errors[0]["message"]

    def test_a_silent_source_is_fine_for_an_audio_pass(self, tmp_path):
        # Adding a soundtrack is the point, so a source with none is no problem
        from dw.media import probe_metadata
        import av

        path = tmp_path / "silent.mp4"
        with av.open(str(path), "w") as container:
            stream = container.add_stream("libx264", rate=24)
            stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
            for _ in range(5):
                frame = av.VideoFrame.from_ndarray(
                    np.zeros((48, 64, 3), np.uint8), format="rgb24"
                )
                container.mux(stream.encode(frame))
            container.mux(stream.encode())
        errors = mask_errors(
            {"steps": [h3_step(source_video="silent.mp4")]},
            None,
            str(tmp_path),
            probe=probe_metadata,
        )
        assert errors == []


# Insertion


@pytest.fixture
def built():
    blocks_ = minimax.MiniMaxH3Blocks()
    result = [("whole", blocks_.init_pipeline())]
    for name in WORKFLOWS:
        result.append((name, blocks_.get_workflow(name).init_pipeline()))
    return result


class TestInsertion:
    def test_the_pair_lands_before_denoise_and_after_denoise(self, built):
        for label, pipeline in built:
            assert not h3_mask.passes_audio(pipeline), label
            assert h3_mask.insert_mask_blocks(pipeline) is True, label
            assert h3_mask.passes_audio(pipeline), label
            for prefix, sequence in core_denoise_sequences(pipeline):
                names = list(sequence.sub_blocks)
                at = names.index(prefix + AUDIO_PASS_BLOCK)
                assert names[at + 1] == prefix + "denoise", label
                assert names.index(prefix + SOURCE_BLOCK) < at, label
                release = names.index(prefix + RELEASE_VIDEO_BLOCK)
                assert at < release < names.index(prefix + "after_denoise"), label

    def test_a_second_call_adds_nothing(self, built):
        for label, pipeline in built:
            h3_mask.insert_mask_blocks(pipeline)
            before = [list(s.sub_blocks) for _, s in core_denoise_sequences(pipeline)]
            h3_mask.insert_mask_blocks(pipeline)
            after = [list(s.sub_blocks) for _, s in core_denoise_sequences(pipeline)]
            assert after == before, label

    def test_the_inserted_blocks_are_the_step_classes(self):
        from dw.pipeline_processors.h3_mask_steps import (
            DwH3AudioPassStep,
            DwH3ReleaseVideoStep,
        )

        assert h3_mask.audio_pass_blocks() == (DwH3AudioPassStep, DwH3ReleaseVideoStep)

    def test_listen_scale_is_a_pipeline_input(self, built):
        for label, pipeline in built:
            h3_mask.insert_mask_blocks(pipeline)
            names = [param.name for param in pipeline._blocks.inputs]
            assert LISTEN_SCALE_INPUT in names, label
            # The held frames and their rate are inputs, so the call returns them
            assert HELD_VIDEO_OUTPUT in names and HELD_VIDEO_RATE_OUTPUT in names, label

    def test_the_hold_and_the_pass_coexist(self):
        from dw.pipeline_processors.h3_hold import insert_audio_hold

        pipeline = minimax.MiniMaxH3Blocks().get_workflow("fl2va").init_pipeline()
        insert_audio_hold(pipeline)
        h3_mask.insert_mask_blocks(pipeline)
        names = list(pipeline._blocks.sub_blocks)
        assert names.index("denoise.dw_refine_schedule") < names.index(
            "denoise." + AUDIO_PASS_BLOCK
        )


# The blocks, on a synthetic state


VIDEO_ROWS, AUDIO_ROWS, CONDITION, TEXT, CHANNELS = 6, 4, 2, 3, 8


def audio_pass_state(held=True, condition=CONDITION, steps=4):
    """A state as `set_timesteps` leaves it: the stock video (shift 12) and
    audio (shift 3) schedules, `condition` leading video rows, the source
    rows encoded behind them."""
    video = MiniMaxH3Scheduler(shift=12.0)
    audio = MiniMaxH3Scheduler(shift=3.0)
    video.set_timesteps(steps)
    audio.set_timesteps(steps)
    total = condition + VIDEO_ROWS
    state = PipelineState()
    generator = torch.Generator().manual_seed(0)
    state.set("latents", torch.randn(total, CHANNELS, generator=generator))
    state.set("audio_latents", torch.randn(AUDIO_ROWS, CHANNELS, generator=generator))
    state.set(SOURCE_ROWS, state.get("latents")[condition:].clone())
    state.set("num_condition_video_rows", condition)
    state.set("num_condition_audio_rows", 0)
    state.set("timesteps", video.timesteps)
    state.set("audio_timesteps", audio.timesteps)
    state.set("video_indices", torch.arange(total))
    state.set("audio_indices", torch.arange(total, total + AUDIO_ROWS))
    state.set(
        "text_indices", torch.arange(total + AUDIO_ROWS, total + AUDIO_ROWS + TEXT)
    )
    state.set("width", WIDTH)
    state.set("height", HEIGHT)
    state.set("row_timestep_plan", ["stock"])
    state.set(HELD_VIDEO_OUTPUT, source_frames() if held else None)
    components = SimpleNamespace(
        _execution_device="cpu", scheduler=video, audio_scheduler=audio
    )
    return components, state


def run_pass(components, state):
    from dw.pipeline_processors.h3_mask_steps import DwH3AudioPassStep

    return DwH3AudioPassStep()(components, state)


def row_timesteps(plan_entry):
    timesteps, indices = plan_entry
    return timesteps[indices]


class TestAudioPassBlock:
    def test_without_a_held_video_it_does_nothing(self):
        components, state = audio_pass_state(held=False)
        run_pass(components, state)
        assert state.get("num_condition_video_rows") == CONDITION
        assert state.get("row_timestep_plan") == ["stock"]

    def test_every_video_row_becomes_a_conditioning_row(self):
        components, state = audio_pass_state()
        run_pass(components, state)
        assert state.get("num_condition_video_rows") == CONDITION + VIDEO_ROWS
        assert state.get(AUDIO_PASS_ROWS) == VIDEO_ROWS

    def test_the_plan_pins_video_clean_and_steps_audio_from_full_noise(self):
        components, state = audio_pass_state()
        run_pass(components, state)
        plan = state.get("row_timestep_plan")
        audio_timesteps = state.get("audio_timesteps")
        assert len(plan) == len(state.get("timesteps"))
        total = CONDITION + VIDEO_ROWS
        for step, entry in enumerate(plan):
            rows = row_timesteps(entry)
            assert torch.all(rows[:total] == 1.0), step
            assert torch.allclose(
                rows[total : total + AUDIO_ROWS],
                audio_timesteps[step].float().expand(AUDIO_ROWS),
            ), step
        # The stock audio schedule starts from noise: t = 1 - sigma = 0
        assert float(audio_timesteps[0]) == 0.0

    def test_no_source_rows_is_refused(self):
        components, state = audio_pass_state()
        state.set(SOURCE_ROWS, None)
        with pytest.raises(ValueError, match="source"):
            run_pass(components, state)

    def test_the_loop_steps_audio_and_never_a_video_row(self):
        """`update`, the diffusers loop step, over the widened count: the
        video rows end the loop as the exact source latent, and the audio rows
        move."""
        components, state = audio_pass_state()
        run_pass(components, state)
        latents_before = state.get("latents").clone()
        audio_before = state.get("audio_latents").clone()
        update = MiniMaxH3LoopSchedulerStep()
        block_state = SimpleNamespace(
            latents=state.get("latents"),
            audio_latents=state.get("audio_latents"),
            num_condition_video_rows=state.get("num_condition_video_rows"),
            num_condition_audio_rows=0,
            audio_timesteps=state.get("audio_timesteps"),
        )
        generator = torch.Generator().manual_seed(1)
        for i, t in enumerate(state.get("timesteps")):
            block_state.noise_pred = torch.randn(
                1, *block_state.latents.shape, generator=generator
            )
            block_state.audio_noise_pred = torch.randn(
                1, *block_state.audio_latents.shape, generator=generator
            )
            update(components, block_state, i, t)
        assert torch.equal(block_state.latents, latents_before)
        assert not torch.equal(block_state.audio_latents, audio_before)

    def test_the_release_puts_the_count_back(self):
        from dw.pipeline_processors.h3_mask_steps import DwH3ReleaseVideoStep

        components, state = audio_pass_state()
        run_pass(components, state)
        DwH3ReleaseVideoStep()(components, state)
        assert state.get("num_condition_video_rows") == CONDITION
        assert state.get(AUDIO_PASS_ROWS) == 0

    def test_the_release_is_a_no_op_off_the_audio_pass(self):
        from dw.pipeline_processors.h3_mask_steps import DwH3ReleaseVideoStep

        components, state = audio_pass_state(held=False)
        run_pass(components, state)
        DwH3ReleaseVideoStep()(components, state)
        assert state.get("num_condition_video_rows") == CONDITION

    def test_it_emits_what_it_held(self):
        from dw.events import RunContext, activate_context, deactivate_context

        components, state = audio_pass_state()
        events = []
        token = activate_context(RunContext(on_event=events.append))
        try:
            run_pass(components, state)
        finally:
            deactivate_context(token)
        logs = [e for e in events if e.get("event") == "log"]
        assert len(logs) == 1
        assert logs[0]["held_video_rows"] == VIDEO_ROWS
        assert logs[0]["listen_scale"] == 1
        assert "Audio pass" in logs[0]["message"]


# The call


class TestTheCall:
    def test_an_unsized_audio_pass_renders_the_sources_canvas(self):
        arguments = with_mask(pipeline_with_blocks(), audio_pass_arguments())
        assert (arguments["width"], arguments["height"], arguments["num_frames"]) == (
            WIDTH,
            HEIGHT,
            FRAMES,
        )
        assert arguments["source_video"].shape == (FRAMES, HEIGHT, WIDTH, 3)

    def test_listen_scale_one_leaves_the_canvas(self):
        arguments = with_mask(
            pipeline_with_blocks(), audio_pass_arguments(listen_scale=1)
        )
        assert (arguments["width"], arguments["height"]) == (WIDTH, HEIGHT)
        assert arguments["source_video"] is arguments[HELD_VIDEO_OUTPUT]

    def test_a_quarter_listen_scale_snaps_the_canvas(self):
        arguments = with_mask(
            pipeline_with_blocks(), audio_pass_arguments(listen_scale=0.25)
        )
        assert (arguments["width"], arguments["height"]) == (64, 32)

    def test_the_step_canvas_is_still_the_sources(self):
        with pytest.raises(ValueError, match="must match"):
            with_mask(
                pipeline_with_blocks(),
                audio_pass_arguments(width=WIDTH // 2, listen_scale=0.5),
            )

    def test_no_videos_output_asks_for_no_held_video(self):
        arguments = with_mask(
            pipeline_with_blocks(), audio_pass_arguments(output=["audio"])
        )
        assert arguments["output"] == ["audio"]

    def test_a_refine_strength_is_refused_at_the_call(self):
        with pytest.raises(ValueError, match="audio pass"):
            with_mask(pipeline_with_blocks(), audio_pass_arguments(refine_strength=0.5))

    def test_a_pipeline_without_the_pass_blocks_refuses(self, monkeypatch):
        from dw.pipeline_processors import pipeline_h3

        monkeypatch.setattr(pipeline_h3, "passes_audio", lambda pipeline: False)
        with pytest.raises(ValueError, match="audio pass"):
            with_mask(pipeline_with_blocks(), audio_pass_arguments())

    def test_area_resampling_keeps_a_flat_colour(self):
        flat = np.full((3, 64, 128, 3), 77, dtype=np.uint8)
        resized = h3_mask.listen_frames(flat, 32, 16)
        assert resized.shape == (3, 16, 32, 3)
        assert np.all(resized == 77)

    def test_area_resampling_averages_each_cell(self):
        frames = np.zeros((1, 2, 2, 3), dtype=np.uint8)
        frames[0, 0, 0] = 200
        assert h3_mask.listen_frames(frames, 1, 1)[0, 0, 0, 0] == 50


class TestHeldVideos:
    def test_an_np_batch_is_replaced_as_floats(self):
        from dw.output_extraction import held_videos

        source = source_frames()
        videos = np.zeros((2, FRAMES, HEIGHT, WIDTH, 3), dtype=np.float32)
        held = held_videos(videos, source)
        assert held.shape == videos.shape
        assert np.allclose(held[1] * 255, source)

    def test_a_result_without_the_held_video_is_todays(self):
        from dw.output_extraction import modular_artifacts

        frames = [[object()]]
        artifacts = modular_artifacts({"videos": frames})
        assert artifacts == frames


# The template (#1192): the bench draft at bench/workflows/add-sound.json,
# promoted once Don's ear passed it at listen_scale 1 (2026-11-06)


TEMPLATE = ROOT / "workflows" / "templates" / "minimax" / "add-sound.json"
DRAFT = ROOT / "bench" / "workflows" / "add-sound.json"


class TestAddSoundTemplate:
    def definition(self):
        return json.loads(TEMPLATE.read_text(encoding="utf-8"))

    @pytest.mark.parametrize("path", [TEMPLATE, DRAFT])
    def test_it_validates(self, path):
        from dw.workflow import Workflow

        definition = json.loads(path.read_text(encoding="utf-8"))
        assert Workflow(definition, None, str(path)).validation_errors({}) == []

    @pytest.mark.parametrize("scale", [1.0, 0.5, 0.25])
    def test_every_listen_scale_validates(self, scale):
        from dw.workflow import Workflow

        workflow = Workflow(self.definition(), None, str(TEMPLATE))
        assert workflow.validation_errors({"listen_scale": scale}) == []

    def test_it_runs_without_a_distillation_lora_on_the_dialogue_schedule(self):
        variables = self.definition()["variables"]
        assert variables["lora_model_name"] is None
        assert variables["num_inference_steps"] == 12
        assert (variables["video_shift"], variables["audio_shift"]) == (12.0, 3.0)

    def test_it_ships_at_listen_scale_1(self):
        # The only scale Don's ear passed outright ("very good lip sync")
        assert self.definition()["variables"]["listen_scale"] == 1.0

    def test_it_is_an_audio_pass(self):
        step = self.definition()["steps"][0]["pipeline"]["arguments"]
        assert step["source_video"] == "variable:source_video"
        assert step["listen_scale"] == "variable:listen_scale"
        for name in ("hold_audio", "mask_image", "mask_video", "refine_strength"):
            assert name not in step

    def test_it_runs_the_drafts_step(self):
        # Promoted as measured: the step Don listened to, unchanged
        draft = json.loads(DRAFT.read_text(encoding="utf-8"))
        (shipped,) = self.definition()["steps"]
        (measured,) = draft["steps"]
        assert shipped["pipeline"] == measured["pipeline"]

    def test_the_description_says_where_lip_sync_holds(self):
        description = self.definition()["description"]
        # 0.5 is the fastest scale that keeps lip sync; 0.25 loses it
        assert "'listen_scale' 0.5" in description
        assert "fastest" in description
        assert "0.25 loses lip sync" in description
        # and a new line over an on-screen speaker reads as voice-over
        assert "not a re-voice" in description
        assert "voice-over" in description

    def test_it_quotes_the_measured_cost(self):
        (entry,) = self.definition()["cost"]
        assert (entry["device"], entry["name"]) == ("cuda", "RTX 3090")
        # bench/records/audio.md: 188 s warm at listen_scale 1
        assert entry["minutes"] == 3.1
        assert "listen_scale" in self.definition()["cost_drivers"]


# The source's own frame rate (R25): conditioning is retimed to H3's 24 fps,
# the result plays the source's own frames at the source's own rate


def pass_result(arguments, seconds, sample_rate=48000):
    """What the pipeline would hand back for `arguments`: a decode that is not
    the source, `seconds` of generated audio, and the held keys echoed."""
    from dw.media_types import rated_videos
    from dw.pipeline_processors.h3_rules import HELD_VIDEO_RATE_OUTPUT
    from PIL import Image

    count = arguments["num_frames"]
    canvas = (arguments["height"], arguments["width"], 3)
    decoded = [Image.fromarray(np.zeros(canvas, np.uint8))] * count
    generated = torch.randn(1, 2, round(seconds * sample_rate))
    return {
        "videos": rated_videos([decoded], 24),
        "audio": generated,
        "sampling_rate": sample_rate,
        HELD_VIDEO_OUTPUT: arguments[HELD_VIDEO_OUTPUT],
        HELD_VIDEO_RATE_OUTPUT: arguments[HELD_VIDEO_RATE_OUTPUT],
    }, generated


class TestSourceFrameRate:
    def test_a_30_fps_source_plays_at_30_under_audio_of_its_own_length(self):
        from dw.output_extraction import modular_artifacts
        from dw.pipeline_processors.h3_rules import RENDER_GRID
        from dw.variable_constraints import aligned

        # 124 frames at 30 fps is 4.13 s; at a relabelled 24 it would be 5.17 s
        source = source_frames(frames=124)
        arguments = with_mask(
            pipeline_with_blocks(), audio_pass_arguments(source_video=rated(source, 30))
        )
        conditioning = aligned(round(124 / 30 * 24), RENDER_GRID)
        assert arguments["num_frames"] == conditioning == 107
        assert arguments["source_video"].shape[0] == conditioning
        assert np.array_equal(arguments[HELD_VIDEO_OUTPUT], source)

        result, _ = pass_result(arguments, conditioning / 24)
        (played,) = modular_artifacts(result)
        assert played.fps == 30
        assert len(played.frames) == 124
        assert np.array_equal(np.stack([np.asarray(f) for f in played.frames]), source)
        assert played.audio.shape[-1] == round(124 / 30 * 48000)

    def test_a_24_fps_source_is_unchanged(self):
        from dw.output_extraction import modular_artifacts

        source = source_frames()
        arguments = with_mask(
            pipeline_with_blocks(), audio_pass_arguments(source_video=rated(source, 24))
        )
        assert arguments["num_frames"] == FRAMES
        assert arguments["source_video"] is arguments[HELD_VIDEO_OUTPUT]
        assert np.array_equal(arguments["source_video"], source)

        result, generated = pass_result(arguments, 0.1)
        (played,) = modular_artifacts(result)
        assert played.fps == 24
        assert torch.equal(played.audio, generated[0])

    @pytest.mark.parametrize(
        "frames, fps",
        [(124, 30), (100, 30), (60, 25), (240, 60), (50, 12), (124, 23.976)],
    )
    def test_the_retimed_conditioning_lands_on_the_h3_grid(self, frames, fps):
        from dw.pipeline_processors.h3_rules import RENDER_GRID, retimed_indices
        from dw.variable_constraints import aligned

        indices = retimed_indices(frames, fps)
        assert len(indices) == aligned(len(indices), RENDER_GRID)
        # It covers the source's duration, with less than one grid step over
        assert len(indices) >= round(frames / fps * 24)
        assert len(indices) - round(frames / fps * 24) < 17
        # Nearest-frame in time, never past the source's last frame
        assert all(0 <= i < frames for i in indices)
        assert indices == sorted(indices)
        assert indices[0] == 0

    def test_the_conditioning_frames_are_the_nearest_source_frames(self):
        source = source_frames(width=32, height=32, frames=124)
        arguments = with_mask(
            pipeline_with_blocks(),
            audio_pass_arguments(source_video=rated(source, 30)),
        )
        for i in (0, 1, 4, 50, 98):
            nearest = min(123, int(i * 30 / 24 + 0.5))
            assert np.array_equal(arguments["source_video"][i], source[nearest])

    def test_a_soundtrack_times_a_source_that_carries_no_rate(self):
        from dw.media_types import AudioVideo
        from dw.pipeline_processors.h3_rules import HELD_VIDEO_RATE_OUTPUT

        source = source_frames(frames=124)
        clip = AudioVideo(list(source), np.zeros((2, 124 * 1600)), 48000)
        arguments = with_mask(
            pipeline_with_blocks(), audio_pass_arguments(source_video=clip)
        )
        assert arguments[HELD_VIDEO_RATE_OUTPUT] == pytest.approx(30)
        assert arguments["num_frames"] == 107

    def test_a_source_with_no_rate_and_no_soundtrack_is_refused(self):
        with pytest.raises(ValueError, match="frame rate"):
            with_mask(
                pipeline_with_blocks(),
                audio_pass_arguments(source_video=source_frames()),
            )

    def test_an_off_grid_source_at_another_rate_is_not_refused_for_the_grid(self):
        # The conditioning count is retimed onto the grid, so the source's own
        # count need not be on it; at 24 fps it still must
        arguments = with_mask(
            pipeline_with_blocks(),
            audio_pass_arguments(source_video=rated(source_frames(frames=100), 30)),
        )
        assert arguments["num_frames"] == 90
        with pytest.raises(ValueError, match="17n"):
            with_mask(
                pipeline_with_blocks(),
                audio_pass_arguments(source_video=rated(source_frames(frames=100), 24)),
            )

    def test_validate_reads_the_rate_for_the_grid_check(self, tmp_path):
        (tmp_path / "take.mp4").write_bytes(b"")

        def probe(fps):
            return lambda path: {
                "kind": "video",
                "frame_count": 100,
                "width": WIDTH,
                "height": HEIGHT,
                "fps": fps,
            }

        step = {"steps": [h3_step(source_video="take.mp4")]}
        assert mask_errors(step, None, str(tmp_path), probe=probe(30)) == []
        errors = mask_errors(step, None, str(tmp_path), probe=probe(24))
        assert len(errors) == 1 and "17n" in errors[0]["message"]


class TestReviewFindings:
    def test_a_mask_edit_is_never_read_as_an_audio_pass(self):
        # A mask with no hold_audio: _check_refine refuses it before the call
        # today, but _with_mask itself must not take it for an audio pass
        from PIL import Image

        mask = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        mask[:8, :8] = 255
        arguments = with_mask(
            pipeline_with_blocks(),
            audio_pass_arguments(mask_image=Image.fromarray(mask), refine_strength=0.5),
        )
        assert HELD_VIDEO_OUTPUT not in arguments
        assert REGENERATE_MASK in arguments

    @pytest.mark.parametrize("width, height", [(960, 544), (1344, 768), (544, 960)])
    def test_a_quarter_listen_canvas_keeps_the_aspect(self, width, height):
        canvas = listen_canvas(width, height, 0.25)
        assert all(side % 32 == 0 and side >= 32 for side in canvas)
        error = abs(canvas[0] / canvas[1] - width / height) / (width / height)
        assert error < 0.07

    def test_listen_scale_on_a_non_h3_step_is_refused_at_validate(self):
        step = {
            "name": "still",
            "pipeline": {
                "configuration": {"component_type": "FluxPipeline"},
                "arguments": {"prompt": "x", LISTEN_SCALE_INPUT: 0.5},
            },
        }
        errors = mask_errors({"steps": [step]})
        assert errors
        assert all(e["path"].endswith(LISTEN_SCALE_INPUT) for e in errors)
        assert any("MiniMax-H3" in e["message"] for e in errors)
