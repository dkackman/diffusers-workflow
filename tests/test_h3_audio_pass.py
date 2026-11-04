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
        "source_video": source_frames(),
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
        pipeline_with_blocks(), audio_pass_arguments(source_video=source)
    )
    assert arguments["output"] == [
        "videos",
        "audio",
        "sampling_rate",
        HELD_VIDEO_OUTPUT,
    ]

    # What the pipeline would hand back: a VAE round trip that is not the source
    decoded = [Image.fromarray(np.zeros((HEIGHT, WIDTH, 3), np.uint8))] * FRAMES
    generated = torch.randn(1, 2, 4800)
    result = {
        "videos": rated_videos([decoded], 24),
        "audio": generated,
        "sampling_rate": 48000,
        HELD_VIDEO_OUTPUT: arguments[HELD_VIDEO_OUTPUT],
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
        audio_pass_arguments(source_video=source, listen_scale=0.5),
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
        assert listen_canvas(960, 544, 0.5) == (480, 288)
        assert listen_canvas(960, 544, 0.25) == (256, 128)
        assert listen_canvas(48, 32, 0.25) == (32, 32)


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
        assert (arguments["width"], arguments["height"]) == (32, 32)

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


# The draft template


DRAFT = ROOT / "bench" / "workflows" / "add-sound.json"


class TestDraftTemplate:
    def definition(self):
        return json.loads(DRAFT.read_text(encoding="utf-8"))

    def test_it_is_not_in_the_catalog(self):
        assert not (
            ROOT / "workflows" / "templates" / "minimax" / "add-sound.json"
        ).exists()

    def test_it_validates(self):
        from dw.workflow import Workflow

        assert Workflow(self.definition(), None, str(DRAFT)).validation_errors({}) == []

    def test_it_runs_without_a_distillation_lora_on_the_dialogue_schedule(self):
        variables = self.definition()["variables"]
        assert variables["lora_model_name"] is None
        assert variables["num_inference_steps"] == 12
        assert (variables["video_shift"], variables["audio_shift"]) == (12.0, 3.0)
        assert variables["listen_scale"] == 1

    def test_it_is_an_audio_pass(self):
        step = self.definition()["steps"][0]["pipeline"]["arguments"]
        assert "source_video" in step
        for name in ("hold_audio", "mask_image", "mask_video", "refine_strength"):
            assert name not in step
