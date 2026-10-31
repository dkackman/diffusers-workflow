"""MiniMax-H3 refine schedule (#620): the sigma list, the block that swaps the
schedules and re-noises the video, its anchors, and the refusals."""

import copy
import json
import os
import tempfile
from types import SimpleNamespace

import pytest
import torch

from dw.hold_audio import refine_strength_errors
from dw.pipeline_processors import h3_hold, h3_rules
from dw.pipeline_processors.h3_hold import (
    blocks,
    core_denoise_sequences,
    insert_audio_hold,
    refines,
)
from dw.pipeline_processors.h3_rules import (
    HOLD_BLOCK,
    REFINE_BLOCK,
    REFINE_SIGMAS_INPUT,
    REFINE_STRENGTH_INPUT,
    refine_problems,
)
from dw.pipeline_processors.pipeline import Pipeline
from dw.workflow import workflow_from_definition

minimax = pytest.importorskip("diffusers.modular_pipelines.minimax_h3")
from diffusers.modular_pipelines.modular_pipeline import PipelineState  # noqa: E402
from diffusers.modular_pipelines.minimax_h3.before_denoise import (  # noqa: E402
    MiniMaxH3SetTimestepsStep,
)
from diffusers.schedulers import MiniMaxH3Scheduler  # noqa: E402
from diffusers.utils.torch_utils import randn_tensor  # noqa: E402
from dw.pipeline_processors.h3_hold_steps import (  # noqa: E402
    explicit_sigmas,
    refine_sigmas,
    shifted_sigma_grid,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKFLOWS = ("t2va", "fl2va", "ref2va")
NOISE_AUG = 0.999  # MiniMaxH3ModularPipeline.keyframe_noise_aug
VIDEO_ROWS, AUDIO_ROWS, TEXT_ROWS, WIDTH = 6, 4, 2, 8


def pipelines():
    """[(label, ModularPipeline)]: the whole graph and each pruned workflow."""
    blocks_ = minimax.MiniMaxH3Blocks()
    result = [("whole", blocks_.init_pipeline())]
    for name in WORKFLOWS:
        result.append((name, blocks_.get_workflow(name).init_pipeline()))
    return result


@pytest.fixture
def built():
    return pipelines()


def components(shift=6.0):
    return SimpleNamespace(
        scheduler=MiniMaxH3Scheduler(shift=shift),
        audio_scheduler=MiniMaxH3Scheduler(shift=3.0),
        keyframe_noise_aug=NOISE_AUG,
        _execution_device=torch.device("cpu"),
    )


def make_state(
    steps=5,
    strength=0.2,
    condition_video=0,
    condition_audio=AUDIO_ROWS,
    seed=None,
):
    """Rows: video_indices 0..5, audio 6..9, text 10..11; `condition_audio`
    leading audio rows held (all of them by default)."""
    state = PipelineState()
    state.set("latents", torch.randn(VIDEO_ROWS, WIDTH, generator=torch.manual_seed(1)))
    state.set(
        "audio_latents",
        torch.randn(AUDIO_ROWS, WIDTH, generator=torch.manual_seed(2)),
    )
    state.set("video_indices", torch.arange(VIDEO_ROWS))
    state.set("audio_indices", torch.arange(VIDEO_ROWS, VIDEO_ROWS + AUDIO_ROWS))
    state.set(
        "text_indices",
        torch.arange(VIDEO_ROWS + AUDIO_ROWS, VIDEO_ROWS + AUDIO_ROWS + TEXT_ROWS),
    )
    state.set("num_condition_video_rows", condition_video)
    state.set("num_condition_audio_rows", condition_audio)
    state.set("num_inference_steps", steps)
    if strength is not None:
        state.set(REFINE_STRENGTH_INPUT, strength)
    if seed is not None:
        state.set("generator", torch.Generator().manual_seed(seed))
    return state


def run(comps, state, set_timesteps=True):
    """The real SetTimesteps block, then the refine block."""
    if set_timesteps:
        MiniMaxH3SetTimestepsStep()(comps, state)
    blocks()[1]()(comps, state)
    return state


# 1. The sigma list


class TestRefineSigmas:
    def test_five_points_at_shift_six(self):
        sigmas = refine_sigmas(0.2, 5, 6)
        assert [round(float(s), 3) for s in sigmas] == [0.2, 0.157, 0.109, 0.057, 0.0]

    def test_six_points_at_shift_six(self):
        sigmas = refine_sigmas(0.2, 6, 6)
        assert [round(float(s), 3) for s in sigmas] == [
            0.2,
            0.166,
            0.129,
            0.089,
            0.046,
            0.0,
        ]

    @pytest.mark.parametrize("points", [2, 5, 6, 30])
    @pytest.mark.parametrize("shift", [1, 3, 6, 12])
    def test_strictly_decreasing_from_strength_to_exactly_zero(self, points, shift):
        sigmas = refine_sigmas(0.2, points, shift)
        assert len(sigmas) == points
        assert float(sigmas[0]) == pytest.approx(0.2)
        assert float(sigmas[-1]) == 0.0
        assert bool((sigmas[1:] < sigmas[:-1]).all())


class TestSigmaGridDrift:
    """`refine_sigmas` copies the shifted spacing stock `set_timesteps` gives a
    full grid; the copy is pinned against the installed scheduler."""

    @pytest.mark.parametrize("steps", [2, 5, 6, 8, 30, 50])
    @pytest.mark.parametrize(
        "shift", [MiniMaxH3Scheduler()._shift, 6.0, 3.0], ids=["default", "6", "3"]
    )
    def test_the_full_grid_is_stock_set_timesteps(self, steps, shift):
        scheduler = MiniMaxH3Scheduler(shift=shift)
        scheduler.set_timesteps(steps)
        ours = shifted_sigma_grid(steps, shift).float()
        assert ours.shape == scheduler.sigmas.shape
        assert torch.allclose(ours, scheduler.sigmas.cpu(), rtol=0, atol=1e-6)

    def test_the_refine_sigmas_sit_on_that_spacing(self):
        # Past the pinned endpoints, refine_sigmas is the grid from its start
        shift = 6.0
        start = 0.2 / (shift - (shift - 1) * 0.2)
        grid = shifted_sigma_grid(5, shift, start).float()
        sigmas = refine_sigmas(0.2, 5, shift)
        assert torch.equal(sigmas[1:-1], grid[1:-1])


# 2. The schedule survives set_timesteps


class TestSchedule:
    @pytest.mark.parametrize("shift", [6.0, 3.0])
    @pytest.mark.parametrize("steps", [5, 6])
    def test_the_refine_sigmas_replace_the_schedule(self, shift, steps):
        comps = components(shift)
        state = run(comps, make_state(steps=steps))

        expected = refine_sigmas(0.2, steps, shift)
        assert torch.allclose(comps.scheduler.sigmas.float(), expected, atol=1e-6)
        timesteps = state.get("timesteps")
        assert len(timesteps) == steps - 1
        assert torch.allclose(timesteps.float(), 1 - expected[:-1], atol=1e-6)

    def test_the_default_schedule_is_what_the_block_replaces(self):
        comps = components()
        state = make_state(steps=5)
        MiniMaxH3SetTimestepsStep()(comps, state)
        full = comps.scheduler.sigmas.clone()
        blocks()[1]()(comps, state)
        assert not torch.equal(full, comps.scheduler.sigmas)


# 3. The audio schedule and the row plan


class TestPlan:
    def test_audio_timesteps_match_the_videos(self):
        comps = components()
        state = run(comps, make_state(steps=5))
        video, audio = state.get("timesteps"), state.get("audio_timesteps")
        assert len(audio) == len(video) == 4
        assert torch.allclose(audio.float(), video.float(), atol=1e-6)

    def test_one_plan_entry_per_timestep(self):
        state = run(components(), make_state(steps=6))
        assert len(state.get("row_timestep_plan")) == len(state.get("timesteps")) == 5

    def test_condition_video_rows_sit_at_the_larger_of_t_and_the_noise_aug(self):
        k = 2
        state = run(
            components(),
            make_state(steps=5, condition_video=k, condition_audio=AUDIO_ROWS),
        )
        for t, (unique, inverse) in zip(
            state.get("timesteps"), state.get("row_timestep_plan")
        ):
            condition = unique[inverse[:k]]
            assert torch.allclose(condition, torch.full((k,), max(float(t), NOISE_AUG)))
            generated = unique[inverse[k:VIDEO_ROWS]]
            assert torch.allclose(generated, torch.full_like(generated, float(t)))
            held = unique[inverse[VIDEO_ROWS : VIDEO_ROWS + AUDIO_ROWS]]
            assert torch.equal(held, torch.ones(AUDIO_ROWS))


# 4. Every audio row must be held


class TestHeldAudioRequired:
    def test_an_unheld_audio_row_raises(self):
        state = make_state(condition_audio=AUDIO_ROWS - 1)
        with pytest.raises(ValueError, match="1 of 4 audio rows are not held"):
            run(components(), state)

    def test_no_held_audio_raises(self):
        with pytest.raises(ValueError, match="hold_audio"):
            run(components(), make_state(condition_audio=0))

    def test_every_row_held_does_not_raise(self):
        run(components(), make_state(condition_audio=AUDIO_ROWS))


# 5 and 6. The re-noise


class TestRenoise:
    def test_fl2va_condition_video_rows_are_untouched(self):
        k = 2
        state = make_state(condition_video=k, condition_audio=AUDIO_ROWS, seed=3)
        before = state.get("latents").clone()
        run(components(), state)
        after = state.get("latents")
        assert torch.equal(after[:k], before[:k])
        assert not torch.equal(after[k:], before[k:])

    def test_ref2va_reference_rows_in_front_are_untouched(self):
        k = 3
        # reference audio rows in front and the held track behind: all held
        state = make_state(condition_video=k, condition_audio=AUDIO_ROWS, seed=4)
        before = state.get("latents").clone()
        run(components(), state)
        after = state.get("latents")
        assert after.shape == before.shape
        assert torch.equal(after[:k], before[:k])
        assert not torch.equal(after[k:], before[k:])

    def test_t2va_every_row_is_renoised(self):
        state = make_state(seed=5)
        before = state.get("latents").clone()
        run(components(), state)
        assert not torch.equal(state.get("latents"), before)

    @pytest.mark.parametrize("k", [0, 2])
    def test_generated_rows_follow_the_formula(self, k):
        strength = 0.3
        state = make_state(strength=strength, condition_video=k, seed=7)
        clean = state.get("latents")[k:].clone()
        run(components(), state)

        noise = randn_tensor(
            clean.shape,
            generator=torch.Generator().manual_seed(7),
            dtype=torch.float32,
        )
        expected = (1 - strength) * clean + strength * noise
        assert torch.allclose(state.get("latents")[k:], expected, atol=1e-6)

    def test_the_same_seed_gives_the_same_latents(self):
        first = run(components(), make_state(seed=11)).get("latents")
        second = run(components(), make_state(seed=11)).get("latents")
        assert torch.equal(first, second)

    def test_a_different_seed_differs(self):
        first = run(components(), make_state(seed=11)).get("latents")
        second = run(components(), make_state(seed=12)).get("latents")
        assert not torch.equal(first, second)


# 7. Anchors


class TestAnchors:
    def test_inserted_before_denoise_in_every_shape(self, built):
        for label, pipeline in built:
            assert insert_audio_hold(pipeline) is True, label
            assert refines(pipeline), label
            for prefix, sequence in core_denoise_sequences(pipeline):
                names = list(sequence.sub_blocks)
                refine = names.index(prefix + REFINE_BLOCK)
                assert names[refine + 1] == prefix + "denoise", label
                assert refine > names.index(prefix + "set_timesteps"), label

    def test_second_call_does_not_duplicate(self, built):
        for label, pipeline in built:
            insert_audio_hold(pipeline)
            before = [list(s.sub_blocks) for _, s in core_denoise_sequences(pipeline)]
            assert insert_audio_hold(pipeline) is True
            after = [list(s.sub_blocks) for _, s in core_denoise_sequences(pipeline)]
            assert before == after, label
            for names in after:
                assert (
                    names.count(REFINE_BLOCK) + names.count("denoise." + REFINE_BLOCK)
                    == 1
                )

    def test_refine_strength_is_a_pipeline_input(self, built):
        for label, pipeline in built:
            insert_audio_hold(pipeline)
            names = [param.name for param in pipeline._blocks.inputs]
            assert REFINE_STRENGTH_INPUT in names, label

    def test_a_pipeline_without_the_blocks_does_not_refine(self):
        assert not refines(minimax.MiniMaxH3Blocks().init_pipeline())
        other = SimpleNamespace(_blocks=SimpleNamespace(sub_blocks={}))
        assert refines(other) is False
        assert refines(object()) is False

    def test_a_sequence_without_denoise_holds_but_does_not_refine(self):
        pipeline = minimax.MiniMaxH3Blocks().get_workflow("t2va").init_pipeline()
        pipeline._blocks.sub_blocks.pop("denoise.denoise")
        assert insert_audio_hold(pipeline) is True
        assert h3_hold.holds_audio(pipeline)
        assert refines(pipeline) is False
        assert HOLD_BLOCK in "".join(
            n for _, s in core_denoise_sequences(pipeline) for n in s.sub_blocks
        )

    def test_module_exports_stay_in_sync(self):
        assert h3_rules.REFINE_BEFORE == "denoise"
        assert h3_rules.REFINE_BLOCK == "dw_refine_schedule"
        assert blocks()[1].__name__ == "DwH3RefineScheduleStep"


# 8. Refusals


def h3_step(workflow="t2va", **arguments):
    return {
        "name": "video",
        "pipeline": {
            "configuration": {"component_type": "ModularPipeline"},
            "from_pretrained_arguments": {"workflow": workflow},
            "arguments": arguments,
        },
    }


def valid(**overrides):
    arguments = {
        "refine_strength": 0.2,
        "latents": "previous_result:up",
        "hold_audio": "previous_result:base.audio",
        "num_inference_steps": 5,
    }
    arguments.update(overrides)
    return {k: v for k, v in arguments.items() if v is not None}


def errors_for(step):
    return refine_strength_errors({"steps": [step]})


def step_with(workflow="t2va", **overrides):
    arguments = valid()
    for key, value in overrides.items():
        if value is None:
            arguments.pop(key, None)
        else:
            arguments[key] = value
    return h3_step(workflow, **arguments)


class TestStaticRefusals:
    def test_a_valid_step_is_clean(self):
        assert errors_for(step_with()) == []

    @pytest.mark.parametrize("value", [0, 1.0, -0.2, 1.5, "0.2", True])
    def test_a_strength_outside_zero_one_or_not_a_number_is_refused(self, value):
        errors = errors_for(step_with(refine_strength=value))
        assert len(errors) == 1
        assert errors[0]["path"].endswith("refine_strength")

    def test_a_deferred_strength_is_not_refused_for_being_a_string(self):
        for value in ("previous_result:pick.strength", "variable:strength"):
            assert errors_for(step_with(refine_strength=value)) == []

    def test_no_latents_is_refused(self):
        errors = errors_for(step_with(latents=None))
        assert len(errors) == 1
        assert "latents" in errors[0]["message"]
        assert errors[0]["path"].endswith("refine_strength")

    def test_no_hold_audio_is_refused(self):
        errors = errors_for(step_with(hold_audio=None))
        assert len(errors) == 1
        assert "hold_audio" in errors[0]["message"]
        assert errors[0]["path"].endswith("refine_strength")

    def test_a_non_h3_component_type_is_refused(self):
        step = step_with()
        step["pipeline"]["configuration"]["component_type"] = "StableDiffusionPipeline"
        errors = errors_for(step)
        assert len(errors) == 1
        assert "StableDiffusionPipeline" in errors[0]["message"]
        assert errors[0]["path"].endswith("refine_strength")

    @pytest.mark.parametrize("workflow", ["i2v", "t2v"])
    def test_a_non_h3_workflow_is_refused(self, workflow):
        errors = errors_for(step_with(workflow))
        assert len(errors) == 1
        assert workflow in errors[0]["message"]

    @pytest.mark.parametrize("workflow", WORKFLOWS)
    def test_the_h3_workflows_are_accepted(self, workflow):
        assert errors_for(step_with(workflow)) == []

    def test_one_inference_step_is_refused(self):
        errors = errors_for(step_with(num_inference_steps=1))
        assert len(errors) == 1
        assert "num_inference_steps" in errors[0]["message"]
        assert errors[0]["path"].endswith("refine_strength")

    def test_no_refine_strength_says_nothing(self):
        assert errors_for(step_with(refine_strength=None, latents=None)) == []


def template(*parts):
    with open(os.path.join(ROOT, "workflows", "templates", *parts)) as f:
        return json.load(f)


def refine_errors(definition):
    workflow = workflow_from_definition(definition, tempfile.mkdtemp())
    return [
        problem
        for problem in workflow.validation_errors()
        if "refine_strength" in problem["path"]
    ]


class TestValidationEntry:
    def test_sd15_template_refuses_refine_strength(self):
        definition = copy.deepcopy(template("text-to-image.json"))
        step = definition["steps"][0]
        assert (
            step["pipeline"]["configuration"]["component_type"]
            == "StableDiffusionPipeline"
        )
        step["pipeline"]["arguments"].update(valid())

        problems = refine_errors(definition)

        assert len(problems) == 1
        assert "StableDiffusionPipeline" in problems[0]["message"]
        assert problems[0]["path"].endswith("refine_strength")

    def test_an_invalid_strength_surfaces_through_the_workflow(self):
        definition = copy.deepcopy(template("text-to-image.json"))
        step = definition["steps"][0]
        step["pipeline"]["arguments"].update(valid(refine_strength=1.5))

        assert any("1.5" in problem["message"] for problem in refine_errors(definition))


# 8 (runtime). Pipeline._check_refine


def ns(pipeline):
    return SimpleNamespace(name="video", pipeline=pipeline, base_dir=None)


def runtime_arguments(**overrides):
    arguments = {
        "refine_strength": 0.2,
        "latents": torch.zeros(1),
        "hold_audio": object(),
        "num_inference_steps": 5,
    }
    for key, value in overrides.items():
        if value is None:
            arguments.pop(key, None)
        else:
            arguments[key] = value
    return arguments


@pytest.fixture
def refining():
    pipeline = minimax.MiniMaxH3Blocks().get_workflow("t2va").init_pipeline()
    insert_audio_hold(pipeline)
    return ns(pipeline)


class TestRuntimeCheck:
    def test_a_non_h3_pipeline_raises(self):
        with pytest.raises(ValueError, match="MiniMax-H3"):
            Pipeline._check_refine(ns(object()), runtime_arguments())

    def test_a_pipeline_without_the_blocks_raises(self):
        pipeline = minimax.MiniMaxH3Blocks().get_workflow("t2va").init_pipeline()
        with pytest.raises(ValueError, match="cannot refine"):
            Pipeline._check_refine(ns(pipeline), runtime_arguments())

    def test_missing_latents_raises(self, refining):
        with pytest.raises(ValueError, match="latents"):
            Pipeline._check_refine(refining, runtime_arguments(latents=None))

    def test_missing_hold_audio_raises(self, refining):
        with pytest.raises(ValueError, match="hold_audio"):
            Pipeline._check_refine(refining, runtime_arguments(hold_audio=None))

    def test_a_strength_out_of_range_raises(self, refining):
        with pytest.raises(ValueError, match="1.5"):
            Pipeline._check_refine(refining, runtime_arguments(refine_strength=1.5))

    def test_one_inference_step_raises(self, refining):
        with pytest.raises(ValueError, match="num_inference_steps"):
            Pipeline._check_refine(refining, runtime_arguments(num_inference_steps=1))

    def test_a_valid_call_passes(self, refining):
        assert Pipeline._check_refine(refining, runtime_arguments()) is None

    def test_no_refine_strength_returns_without_checking(self):
        assert (
            Pipeline._check_refine(
                ns(object()), runtime_arguments(refine_strength=None)
            )
            is None
        )


# 9. No refine_strength: nothing changes


class TestNoRefine:
    def test_the_block_leaves_the_state_as_set_timesteps_left_it(self):
        comps = components()
        state = make_state(steps=5, strength=None, seed=9)
        MiniMaxH3SetTimestepsStep()(comps, state)
        names = ("latents", "timesteps", "audio_timesteps", "row_timestep_plan")
        before = {name: state.get(name) for name in names}
        sigmas = comps.scheduler.sigmas
        audio_sigmas = comps.audio_scheduler.sigmas
        latents = before["latents"].clone()

        blocks()[1]()(comps, state)

        assert torch.equal(state.get("latents"), latents)
        for name in names:
            assert state.get(name) is before[name], name
        assert comps.scheduler.sigmas is sigmas
        assert comps.audio_scheduler.sigmas is audio_sigmas
        assert len(state.get("timesteps")) == 4

    def test_unheld_audio_does_not_matter_without_refine(self):
        comps = components()
        state = make_state(strength=None, condition_audio=0)
        run(comps, state)
        assert len(state.get("timesteps")) == 4


class TestOneOwnerOfTheRules:
    """The run-time check raises what `refine_problems` says, the same rules
    the static check reports - so a rule changed there reaches both."""

    @pytest.mark.parametrize(
        "overrides",
        [
            {"latents": None},
            {"hold_audio": None},
            {"refine_strength": 1.5},
            {"refine_strength": "0.2"},
            {"num_inference_steps": 1},
        ],
    )
    def test_the_runtime_message_is_the_owners(self, refining, overrides):
        arguments = runtime_arguments(**overrides)
        problems = refine_problems(arguments)
        assert problems
        with pytest.raises(ValueError) as raised:
            Pipeline._check_refine(refining, arguments)
        assert str(raised.value) == f"Step 'video': {problems[0]}"


# 10. refine_tiles (#1043)

TILES = {"width": 1344, "height": 768, "overlap": 256}


def tiled(tiles=TILES, **overrides):
    return step_with(refine_tiles=tiles, **overrides)


def messages(step):
    return [error["message"] for error in errors_for(step)]


def tile_errors(step):
    return [e for e in errors_for(step) if e["path"].endswith("refine_tiles")]


def upscale_workflow(up_width=1920, up_height=1088, tiles=TILES, **overrides):
    refine = tiled(tiles, **overrides)
    refine["name"] = "refine"
    refine["pipeline"]["arguments"]["latents"] = "previous_result:up"
    return {
        "steps": [
            h3_step(),
            {
                "name": "up",
                "task": {
                    "command": "upscale_h3_latents",
                    "arguments": {
                        "latents": "previous_result:video.latents",
                        "width": up_width,
                        "height": up_height,
                    },
                },
            },
            refine,
        ]
    }


class TestTileStatic:
    def test_tiles_without_a_strength_are_refused_at_the_tiles_path(self):
        errors = errors_for(tiled(refine_strength=None))
        assert len(errors) == 1
        assert errors[0]["path"].endswith("arguments.refine_tiles")
        assert errors[0]["message"].startswith("refine_tiles tiles a refine_strength")

    @pytest.mark.parametrize(
        "tiles, fragment",
        [
            ({"width": 1000, "height": 544, "overlap": 256}, "multiple of 32"),
            ({"width": 1920, "height": 1088, "overlap": 256}, "1920x1088"),
            ({"width": 1344, "height": 768, "overlap": 128}, "128"),
            ({"width": 1344, "height": 768, "overlap": 400}, "400"),
            ({"width": 1344, "height": 768, "overlap": 384}, "384"),
            ({"width": 1344, "height": 768, "overlap": 160}, "160"),
            ({"width": 1344, "height": 768, "overlap": 250}, "multiple of 32"),
            ({"width": "wide", "height": 768, "overlap": 256}, "'wide'"),
            ({"width": 1344, "height": 768.0, "overlap": 256}, "768.0"),
            ({"width": 1344, "height": 768, "overlap": True}, "True"),
            ({"width": 1344, "height": 768}, "missing overlap"),
            ({"width": 1344, "height": 768, "overlap": 256, "pad": 1}, "pad"),
            ("1344x768", "mapping"),
        ],
    )
    def test_a_bad_tile_is_refused_naming_refine_tiles(self, tiles, fragment):
        errors = tile_errors(tiled(tiles))
        assert errors
        assert all(e["message"].startswith("refine_tiles") for e in errors)
        assert any(fragment in e["message"] for e in errors)
        assert len(errors) == len(errors_for(tiled(tiles)))

    @pytest.mark.parametrize(
        "tiles",
        [
            {"width": 1344, "height": 768, "overlap": 256},
            {"width": 768, "height": 1088, "overlap": 192},
            {"width": 768, "height": 768, "overlap": 352},
        ],
    )
    def test_a_boundary_tile_is_clean(self, tiles):
        assert errors_for(tiled(tiles)) == []

    def test_an_unresolved_value_skips_its_checks(self):
        tiles = {"width": "variable:w", "height": 768, "overlap": "variable:o"}
        assert errors_for(tiled(tiles)) == []
        assert errors_for(tiled("variable:tiles")) == []

    @pytest.mark.parametrize(
        "argument, value",
        [
            ("image", {"location": "a.png"}),
            ("last_image", "asset:a.png"),
            ("references", [{"reference_type": "image"}]),
            ("guides", [{"video": "asset:a.mp4"}]),
        ],
    )
    def test_a_reference_keyframe_or_guide_is_refused(self, argument, value):
        step = tiled(**{argument: value})
        found = [m for m in messages(step) if argument in m]
        assert len(found) == 1
        assert found[0].startswith("refine_tiles")

    def test_a_null_reference_does_not_count(self):
        assert errors_for(tiled(image=None, guides=None)) == []

    def test_a_picture_in_the_prompt_is_refused(self):
        found = messages(tiled(prompt="<Picture 1> walks"))
        assert len(found) == 1 and "<Picture N>" in found[0]

    def test_a_tile_over_a_known_canvas_is_refused(self):
        found = messages(tiled(width=1280, height=704))
        assert len(found) == 1
        assert "larger than the 1280x704 canvas" in found[0]

    def test_a_tile_over_one_axis_is_refused(self):
        tall = {"width": 768, "height": 1344, "overlap": 192}
        found = messages(tiled(tall, width=1920, height=1088))
        assert len(found) == 1
        assert "768x1344 is larger than the 1920x1088 canvas" in found[0]

    def test_the_canvas_comes_from_the_upscale_step(self):
        assert refine_strength_errors(upscale_workflow()) == []
        problems = refine_strength_errors(upscale_workflow(1280, 704))
        assert any("1280x704 canvas" in p["message"] for p in problems)
        assert all(p["path"].endswith("refine_tiles") for p in problems)

    def test_a_disagreeing_size_is_refused_against_the_upscale_step(self):
        problems = refine_strength_errors(upscale_workflow(width=1344))
        assert len(problems) == 1
        assert "width 1344" in problems[0]["message"]
        assert "1920x1088" in problems[0]["message"]

    def test_an_agreeing_size_is_clean(self):
        assert refine_strength_errors(upscale_workflow(width=1920, height=1088)) == []

    def test_an_unknown_canvas_skips_the_canvas_checks(self):
        assert errors_for(tiled()) == []

    def test_a_non_h3_step_is_refused_naming_refine_tiles(self):
        step = tiled()
        step["pipeline"]["configuration"]["component_type"] = "StableDiffusionPipeline"
        found = tile_errors(step)
        assert len(found) == 1
        assert "refine_tiles is a MiniMax-H3 argument" in found[0]["message"]

    def test_a_tiles_step_keeps_the_strength_rules(self):
        errors = errors_for(tiled(hold_audio=None))
        assert len(errors) == 1
        assert errors[0]["path"].endswith("refine_strength")

    def test_the_happy_workflow_is_clean(self):
        definition = copy.deepcopy(upscale_workflow())
        workflow = workflow_from_definition(definition, tempfile.mkdtemp())
        assert [
            p
            for p in workflow.validation_errors()
            if "refine_" in str(p.get("path")) or "refine_" in p["message"]
        ] == []


class TestTileRuntime:
    def canvas(self, height=1088, width=1920):
        return torch.zeros(1, 16, 8, height // 16, width // 16)

    def test_tiles_without_a_strength_raise_naming_refine_tiles(self, run):
        with pytest.raises(ValueError, match="refine_tiles tiles a refine_strength"):
            run(refine_strength=None)

    def test_a_non_h3_pipeline_names_refine_tiles(self):
        with pytest.raises(ValueError, match="refine_tiles is a MiniMax-H3"):
            Pipeline._check_refine(
                ns(object()),
                runtime_arguments(
                    refine_strength=None, refine_tiles=TILES, latents=self.canvas()
                ),
            )

    @pytest.fixture
    def run(self, refining):
        def call(tiles=TILES, canvas=None, **overrides):
            arguments = runtime_arguments(
                refine_tiles=tiles,
                latents=self.canvas() if canvas is None else canvas,
                **overrides,
            )
            return Pipeline._check_refine(refining, arguments)

        return call

    def test_a_valid_tiled_call_passes(self, run):
        assert run() is None

    def test_an_off_limit_tile_raises(self, run):
        with pytest.raises(ValueError, match="refine_tiles.*1920x1088"):
            run({"width": 1920, "height": 1088, "overlap": 256})

    def test_a_low_overlap_raises(self, run):
        with pytest.raises(ValueError, match="refine_tiles overlap.*128"):
            run({"width": 1344, "height": 768, "overlap": 128})

    def test_a_tile_over_the_tensor_canvas_raises(self, run):
        with pytest.raises(ValueError, match="1280x704 canvas"):
            run(canvas=self.canvas(704, 1280))

    def test_a_size_disagreeing_with_the_tensor_raises(self, run):
        with pytest.raises(ValueError, match="width 1344 disagrees"):
            run(width=1344, height=1088)

    def test_a_guide_raises(self, run):
        with pytest.raises(ValueError, match="refine_tiles.*guides"):
            run(guides=[{"video": "a.mp4"}])


def full_frame_workflow(up_width=1920, up_height=1088, **sizes):
    workflow = upscale_workflow(up_width, up_height)
    arguments = workflow["steps"][2]["pipeline"]["arguments"]
    del arguments["refine_tiles"]
    arguments.update(sizes)
    return workflow


class TestFullFrameStatic:
    def test_an_omitted_size_over_larger_latents_is_refused(self):
        problems = refine_strength_errors(full_frame_workflow())
        assert len(problems) == 1
        assert problems[0]["path"].endswith("refine_strength")
        assert "1920x1088" in problems[0]["message"]
        assert "no width" in problems[0]["message"]

    def test_a_disagreeing_height_is_refused(self):
        problems = refine_strength_errors(full_frame_workflow(width=1920, height=768))
        assert len(problems) == 1
        assert "height 768" in problems[0]["message"]

    def test_a_matching_size_is_clean(self):
        assert (
            refine_strength_errors(full_frame_workflow(width=1920, height=1088)) == []
        )

    def test_latents_at_the_layout_default_need_no_size(self):
        assert refine_strength_errors(full_frame_workflow(1344, 768)) == []

    def test_an_unknown_canvas_is_not_checked(self):
        assert errors_for(step_with()) == []


# 12. refine_sigmas - an explicit schedule (#1082)

# The regenerate recipe: H3's stock 4-step grid from 0.909375
RECIPE = [0.909375, 0.857143, 0.666667, 0]
RECIPE_STRENGTH = 0.909375


def explicit(**overrides):
    arguments = {
        "refine_strength": RECIPE_STRENGTH,
        "num_inference_steps": len(RECIPE),
        "refine_sigmas": list(RECIPE),
    }
    arguments.update(overrides)
    return arguments


BROKEN = [
    pytest.param({"refine_sigmas": [0.909375]}, "at least 2 points", id="one-point"),
    pytest.param(
        {"refine_sigmas": [0.909375, 0.857143, 0.857143, 0]},
        "strictly decreasing",
        id="repeated-point",
    ),
    pytest.param(
        {"refine_sigmas": [0.909375, 0.666667, 0.857143, 0]},
        "strictly decreasing",
        id="rising",
    ),
    pytest.param(
        {"refine_sigmas": [0.9, 0.857143, 0.666667, 0]},
        "starts at refine_strength",
        id="off-strength",
    ),
    pytest.param(
        {"refine_sigmas": [0.909375, 0.857143, 0.666667, 0.01]},
        "ends at 0",
        id="not-to-zero",
    ),
    pytest.param(
        {"num_inference_steps": 5}, "set num_inference_steps to 4", id="count"
    ),
    pytest.param(
        {"num_inference_steps": None}, "sets no num_inference_steps", id="no-steps"
    ),
    pytest.param(
        {"refine_sigmas": "0.9,0.5,0"}, "not a list of numbers", id="not-a-list"
    ),
    pytest.param(
        {"refine_sigmas": [0.909375, "half", 0]},
        "not a list of numbers",
        id="not-numbers",
    ),
    pytest.param(
        {"refine_sigmas": [0.909375, True, 0]},
        "not a list of numbers",
        id="a-bool",
    ),
]


class TestSigmaListStatic:
    def test_the_regenerate_recipe_is_clean(self):
        assert errors_for(step_with(**explicit())) == []

    @pytest.mark.parametrize("overrides, fragment", BROKEN)
    def test_a_broken_rule_is_refused_at_the_sigmas_path(self, overrides, fragment):
        errors = errors_for(step_with(**explicit(**overrides)))
        assert errors, overrides
        assert all(e["path"].endswith("arguments.refine_sigmas") for e in errors)
        assert any(fragment in e["message"] for e in errors), errors

    def test_the_first_point_may_differ_within_the_tolerance(self):
        sigmas = [RECIPE_STRENGTH + 5e-7, 0.857143, 0.666667, 0]
        assert errors_for(step_with(**explicit(refine_sigmas=sigmas))) == []

    def test_an_integer_zero_and_a_float_zero_both_end_it(self):
        for zero in (0, 0.0):
            sigmas = [0.909375, 0.5, zero]
            assert (
                errors_for(
                    step_with(**explicit(refine_sigmas=sigmas, num_inference_steps=3))
                )
                == []
            )

    def test_sigmas_without_a_strength_are_refused(self):
        errors = errors_for(step_with(**explicit(refine_strength=None)))
        assert len(errors) == 1
        assert errors[0]["path"].endswith("arguments.refine_sigmas")
        assert errors[0]["message"].startswith(
            "refine_sigmas schedules a refine_strength pass"
        )

    def test_a_deferred_list_is_not_checked(self):
        step = step_with(**explicit(refine_sigmas="variable:sigmas"))
        assert errors_for(step) == []

    def test_sigmas_with_tiles_are_clean(self):
        assert errors_for(tiled(**explicit())) == []

    def test_sigmas_on_a_non_h3_step_are_refused(self):
        step = step_with(**explicit())
        step["pipeline"]["configuration"]["component_type"] = "StableDiffusionPipeline"
        errors = errors_for(step)
        assert any(e["path"].endswith("arguments.refine_sigmas") for e in errors)

    def test_the_strength_rules_still_hold_beside_sigmas(self):
        errors = errors_for(step_with(**explicit(hold_audio=None)))
        assert any(
            e["path"].endswith("refine_strength") and "hold_audio" in e["message"]
            for e in errors
        )

    def test_no_sigmas_keeps_the_old_rules(self):
        assert refine_problems(valid()) == []


class TestSigmaListRuntime:
    def test_the_regenerate_recipe_passes(self, refining):
        assert Pipeline._check_refine(refining, runtime_arguments(**explicit())) is None

    @pytest.mark.parametrize("overrides, fragment", BROKEN)
    def test_a_broken_rule_raises(self, refining, overrides, fragment):
        with pytest.raises(ValueError, match="refine_sigmas") as raised:
            Pipeline._check_refine(refining, runtime_arguments(**explicit(**overrides)))
        assert str(raised.value).startswith("Step 'video': ")

    def test_sigmas_without_a_strength_raise(self, refining):
        with pytest.raises(ValueError, match="refine_sigmas schedules"):
            Pipeline._check_refine(
                refining, runtime_arguments(**explicit(refine_strength=None))
            )

    def test_a_non_h3_pipeline_names_refine_sigmas(self):
        with pytest.raises(ValueError, match="refine_sigmas is a MiniMax-H3"):
            Pipeline._check_refine(
                ns(object()), runtime_arguments(**explicit(refine_strength=None))
            )


class TestSigmaListOneOwner:
    @pytest.mark.parametrize("overrides, fragment", BROKEN)
    def test_the_runtime_message_is_the_owners(self, refining, overrides, fragment):
        arguments = runtime_arguments(**explicit(**overrides))
        problems = refine_problems(arguments)
        assert any(fragment in problem for problem in problems)
        with pytest.raises(ValueError) as raised:
            Pipeline._check_refine(refining, arguments)
        assert str(raised.value) == f"Step 'video': {problems[0]}"

    @pytest.mark.parametrize("overrides, fragment", BROKEN)
    def test_the_static_messages_are_the_owners(self, overrides, fragment):
        arguments = valid(**explicit(**overrides))
        statics = [e["message"] for e in errors_for(h3_step(**arguments))]
        assert statics == refine_problems(arguments)


def explicit_state(sigmas=RECIPE, strength=RECIPE_STRENGTH, seed=4, **kwargs):
    state = make_state(steps=len(sigmas), strength=strength, seed=seed, **kwargs)
    state.set(REFINE_SIGMAS_INPUT, list(sigmas))
    return state


class TestSigmaListSchedule:
    def test_explicit_sigmas_are_the_list_verbatim(self):
        assert explicit_sigmas(RECIPE, RECIPE_STRENGTH).tolist() == pytest.approx(
            RECIPE
        )

    def test_the_ends_are_pinned_to_strength_and_zero(self):
        sigmas = explicit_sigmas([0.9093755, 0.5, 0.0], 0.909375)
        assert sigmas[0].item() == pytest.approx(0.909375, abs=1e-7)
        assert sigmas[-1].item() == 0.0

    def test_fewer_than_two_points_raise(self):
        with pytest.raises(ValueError, match="at least 2 points"):
            explicit_sigmas([0.9], 0.9)

    @pytest.mark.parametrize("shift", [1.0, 6.0])
    def test_the_block_runs_the_list_unshifted(self, shift):
        comps = components(shift)
        state = run(comps, explicit_state())
        for scheduler in (comps.scheduler, comps.audio_scheduler):
            assert scheduler.sigmas.tolist() == pytest.approx(RECIPE, abs=1e-6)
        assert len(state.get("timesteps")) == len(RECIPE) - 1
        assert torch.equal(state.get("timesteps"), state.get("audio_timesteps"))

    def test_the_plan_is_rebuilt_one_entry_per_denoise_step(self):
        state = run(components(), explicit_state())
        assert len(state.get("row_timestep_plan")) == len(RECIPE) - 1

    def test_the_renoise_is_at_the_strength(self):
        comps = components()
        state = make_state(steps=4, strength=RECIPE_STRENGTH, seed=4)
        clean = state.get("latents").clone()
        state.set(REFINE_SIGMAS_INPUT, list(RECIPE))
        run(comps, state)
        noise = randn_tensor(
            clean.shape,
            generator=torch.Generator().manual_seed(4),
            device=torch.device("cpu"),
            dtype=clean.dtype,
        )
        expected = (1 - RECIPE_STRENGTH) * clean + RECIPE_STRENGTH * noise
        assert torch.allclose(state.get("latents"), expected, atol=1e-5)

    def test_a_shorter_list_is_not_the_shift_grid(self):
        comps = components()
        sigmas = [0.5, 0.25, 0]
        run(comps, explicit_state(sigmas=sigmas, strength=0.5))
        assert comps.scheduler.sigmas.tolist() == pytest.approx(sigmas, abs=1e-6)
        assert comps.scheduler.sigmas.tolist() != pytest.approx(
            refine_sigmas(0.5, 3, 6.0).tolist(), abs=1e-6
        )

    def test_without_the_list_the_shift_grid_runs(self):
        comps = components()
        run(comps, make_state(steps=5, strength=0.2))
        assert comps.scheduler.sigmas.tolist() == pytest.approx(
            refine_sigmas(0.2, 5, 6.0).tolist(), abs=1e-6
        )
