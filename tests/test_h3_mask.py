"""MiniMax-H3 regional edit (#1062/#1077): the mask-to-rows arithmetic, the block
insertion, the blend, and the argument rules."""

import copy
import json
import os
import tempfile
from types import SimpleNamespace

import numpy as np
import pytest

from dw.hold_audio import mask_errors
from dw.pipeline_processors import h3_mask
from dw.pipeline_processors.h3_hold import core_denoise_sequences
from dw.pipeline_processors.h3_mask import (
    _dilate,
    _pool,
    binarized,
    edits_regions,
    insert_mask_blocks,
    keep_rows,
    regenerate_cells,
)
from dw.pipeline_processors.h3_rules import (
    REGENERATE_MASK,
    BLEND_BLOCK,
    KEEP_ROWS,
    REFINE_ROW_NOISE,
    SOURCE_BLOCK,
    SOURCE_ROWS,
    edits_region,
    is_h3_step,
    mask_problems,
    media_problems,
    refine_problems,
)
from dw.workflow import workflow_from_definition

torch = pytest.importorskip("torch")
minimax = pytest.importorskip("diffusers.modular_pipelines.minimax_h3")
from diffusers.modular_pipelines.modular_pipeline import PipelineState  # noqa: E402
from diffusers.schedulers import MiniMaxH3Scheduler  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKFLOWS = ("t2va", "fl2va", "ref2va")
PATCH = (1, 2, 2)

# A 128x128 pixel canvas is an 8x8 latent, a 4x4 grid of 2x2 patches (32 px a cell)
SIDE = 128
LATENT_SIDE = SIDE // 16


def pixel_mask(frames, *white):
    """(frames, SIDE, SIDE) uint8 with each `(frame, y, x)` in `white` set to 255."""
    mask = np.zeros((frames, SIDE, SIDE), dtype=np.uint8)
    for frame, y, x in white:
        mask[frame, y, x] = 255
    return mask


def cells(mask, latent_frames=3, grow=0):
    return regenerate_cells(
        binarized(mask), latent_frames, LATENT_SIDE, LATENT_SIDE, PATCH, grow
    )


# 1. Arithmetic


class TestBinarized:
    def test_rgb_thresholds_luma_at_128(self):
        frame = np.zeros((1, 1, 4, 3), dtype=np.uint8)
        frame[0, 0, 0] = (255, 255, 255)
        frame[0, 0, 1] = (128, 128, 128)
        frame[0, 0, 2] = (127, 127, 127)
        frame[0, 0, 3] = (0, 0, 0)
        assert binarized(frame).tolist() == [[[True, True, False, False]]]

    def test_rgb_uses_luma_not_a_channel(self):
        frame = np.zeros((1, 1, 2, 3), dtype=np.uint8)
        frame[0, 0, 0] = (255, 0, 0)  # luma 76: below
        frame[0, 0, 1] = (0, 255, 0)  # luma 150: above
        assert binarized(frame).tolist() == [[[False, True]]]

    def test_rgba_ignores_alpha(self):
        frame = np.zeros((1, 1, 1, 4), dtype=np.uint8)
        frame[0, 0, 0] = (0, 0, 0, 255)
        assert not binarized(frame).any()

    def test_a_mask_image_binarizes_as_its_frame_does(self):
        from PIL import Image

        frame = np.zeros((1, 2, 8, 3), dtype=np.uint8)
        frame[0, :, :4] = (255, 255, 255)
        frame[0, :, 4] = (128, 128, 128)
        frame[0, :, 5] = (0, 255, 0)
        image = Image.fromarray(frame[0])
        assert binarized(image).tolist() == binarized(frame).tolist()
        assert binarized(image)[0, 0].tolist() == [True] * 6 + [False] * 2

    def test_a_2d_frame_stack_thresholds_directly(self):
        frames = np.array([[[0, 127, 128, 255]]], dtype=np.uint8)
        assert binarized(frames).tolist() == [[[False, False, True, True]]]


class TestRegenerateCells:
    def test_the_grid_is_the_patch_row_grid(self):
        grid = cells(pixel_mask(6))
        assert grid.shape == (3, 4, 4)
        assert grid.dtype == bool

    def test_a_corner_pixel_without_grow_marks_one_cell(self):
        grid = cells(pixel_mask(6, (0, 0, 0)), grow=0)
        assert grid.sum() == 1
        assert grid[0, 0, 0]

    def test_the_opposite_corner_lands_in_the_last_cell(self):
        grid = cells(pixel_mask(6, (5, SIDE - 1, SIDE - 1)), grow=0)
        assert grid.sum() == 1
        assert grid[2, 3, 3]

    def test_grow_one_makes_a_clipped_cube_at_a_corner(self):
        grid = cells(pixel_mask(6, (0, 0, 0)), grow=1)
        assert grid.sum() == 8
        assert grid[:2, :2, :2].all()

    def test_grow_in_the_interior_makes_a_full_3x3x3_cube(self):
        # latent frames 5, pixel frame 2 of 5 -> cell 2; pixel (64, 64) -> cell (2, 2)
        grid = cells(pixel_mask(5, (2, 64, 64)), latent_frames=5, grow=1)
        assert grid.shape == (5, 4, 4)
        assert grid[1:4, 1:4, 1:4].all()
        assert grid.sum() == 27

    def test_grow_two_widens_further(self):
        one = cells(pixel_mask(6, (0, 0, 0)), grow=1).sum()
        two = cells(pixel_mask(6, (0, 0, 0)), grow=2).sum()
        assert two == 27 and two > one

    def test_a_single_frame_mask_covers_every_latent_frame(self):
        mask = pixel_mask(1, (0, 0, 0))
        grid = cells(mask, latent_frames=3, grow=0)
        assert grid[:, 0, 0].all()
        assert grid.sum() == 3

    def test_a_single_frame_mask_matches_the_same_mask_repeated(self):
        one = pixel_mask(1, (0, 70, 100))
        many = np.repeat(one, 6, axis=0)
        assert (cells(one, grow=1) == cells(many, grow=1)).all()

    def test_an_empty_mask_regenerates_nothing(self):
        assert not cells(pixel_mask(6), grow=1).any()

    def test_a_full_mask_regenerates_everything(self):
        assert cells(np.full((6, SIDE, SIDE), 255, np.uint8), grow=0).all()

    def test_the_flattened_order_is_frame_major_then_row_major(self):
        # cell (frame 1, row 2, column 3)
        mask = pixel_mask(6, (2, 2 * 32, 3 * 32))
        flat = cells(mask, grow=0).reshape(-1)
        assert flat.sum() == 1
        assert np.flatnonzero(flat).tolist() == [1 * 16 + 2 * 4 + 3]

    def test_a_patch_time_stride_pools_latent_frames(self):
        grid = regenerate_cells(
            binarized(pixel_mask(8, (0, 0, 0))), 4, 8, 8, (2, 2, 2), 0
        )
        assert grid.shape == (2, 4, 4)
        assert grid.sum() == 1

    def test_pool_counts_a_boundary_pixel_toward_both_cells(self):
        # 5 pixels into 2 cells: cell 0 covers [0, 3), cell 1 covers [2, 5)
        mask = np.zeros((1, 1, 5), dtype=bool)
        mask[0, 0, 2] = True
        assert _pool(mask, (1, 1, 2)).tolist() == [[[True, True]]]

    def test_pool_to_more_cells_than_pixels_still_covers_each(self):
        mask = np.array([[[True, False]]])
        assert _pool(mask, (1, 1, 4)).tolist() == [[[True, True, False, False]]]

    def test_dilate_is_clipped_and_diagonal(self):
        grid = np.zeros((3, 3), dtype=bool)
        grid[0, 0] = True
        assert _dilate(grid, 1).tolist() == [
            [True, True, False],
            [True, True, False],
            [False, False, False],
        ]

    def test_dilate_zero_is_a_no_op(self):
        grid = np.zeros((2, 2, 2), dtype=bool)
        grid[0, 0, 0] = True
        assert (_dilate(grid, 0) == grid).all()


class TestKeepRows:
    def keep(self, mask, grow=0):
        return keep_rows(binarized(mask), 3, LATENT_SIDE, LATENT_SIDE, PATCH, grow)

    def test_it_is_the_complement_of_the_regenerated_cells(self):
        mask = pixel_mask(6, (0, 0, 0), (3, 70, 70))
        for grow in (0, 1, 2):
            regenerate = cells(mask, grow=grow).reshape(-1)
            keep = self.keep(mask, grow)
            assert keep.shape == (3 * 4 * 4,)
            assert (keep == ~regenerate).all()

    def test_an_empty_mask_keeps_every_row(self):
        assert self.keep(pixel_mask(6)).all()

    def test_a_corner_pixel_keeps_all_but_one_row(self):
        keep = self.keep(pixel_mask(6, (0, 0, 0)))
        assert (~keep).sum() == 1
        assert not keep[0]


def test_fit_source_frames_pads_with_the_last_frame():
    from dw.pipeline_processors.h3_mask_steps import fit_source_frames

    frames = np.arange(3, dtype=np.uint8).reshape(3, 1, 1, 1) * np.ones(
        (3, 2, 2, 3), np.uint8
    )
    fitted = fit_source_frames(frames, 5)
    assert fitted.shape == (5, 2, 2, 3)
    assert (fitted[:3] == frames).all()
    assert (fitted[3] == frames[2]).all() and (fitted[4] == frames[2]).all()


def test_fit_source_frames_truncates_and_passes_an_exact_fit():
    from dw.pipeline_processors.h3_mask_steps import fit_source_frames

    frames = np.arange(4, dtype=np.uint8).reshape(4, 1, 1, 1)
    assert fit_source_frames(frames, 2).reshape(-1).tolist() == [0, 1]
    assert fit_source_frames(frames, 4).reshape(-1).tolist() == [0, 1, 2, 3]


# 2. Insertion


@pytest.fixture
def built():
    blocks_ = minimax.MiniMaxH3Blocks()
    result = [("whole", blocks_.init_pipeline())]
    for name in WORKFLOWS:
        result.append((name, blocks_.get_workflow(name).init_pipeline()))
    return result


class TestInsertion:
    def test_blocks_land_before_set_timesteps_and_after_update(self, built):
        for label, pipeline in built:
            assert not edits_regions(pipeline), label
            assert insert_mask_blocks(pipeline) is True, label
            assert edits_regions(pipeline), label
            sequences = core_denoise_sequences(pipeline)
            assert sequences, label
            for prefix, sequence in sequences:
                names = list(sequence.sub_blocks)
                source = names.index(prefix + SOURCE_BLOCK)
                assert source < names.index(prefix + "set_timesteps"), label
                loop = list(sequence.sub_blocks[prefix + "denoise"].sub_blocks)
                assert loop[loop.index("update") + 1] == BLEND_BLOCK, label

    def test_the_source_block_sits_right_before_set_timesteps(self, built):
        for label, pipeline in built:
            insert_mask_blocks(pipeline)
            for prefix, sequence in core_denoise_sequences(pipeline):
                names = list(sequence.sub_blocks)
                assert names.index(prefix + SOURCE_BLOCK) + 1 == names.index(
                    prefix + "set_timesteps"
                ), label

    def test_the_flat_shape_prefixes_the_source_block(self):
        pipeline = minimax.MiniMaxH3Blocks().get_workflow("t2va").init_pipeline()
        insert_mask_blocks(pipeline)
        assert "denoise." + SOURCE_BLOCK in pipeline._blocks.sub_blocks

    def test_the_nested_shape_does_not_prefix_it(self):
        pipeline = minimax.MiniMaxH3Blocks().init_pipeline()
        insert_mask_blocks(pipeline)
        prefixes = {p for p, _ in core_denoise_sequences(pipeline)}
        assert prefixes == {""}
        for _, sequence in core_denoise_sequences(pipeline):
            assert SOURCE_BLOCK in sequence.sub_blocks

    def test_a_second_call_adds_nothing(self, built):
        for label, pipeline in built:
            insert_mask_blocks(pipeline)

            def shape():
                return [
                    (list(s.sub_blocks), list(s.sub_blocks[p + "denoise"].sub_blocks))
                    for p, s in core_denoise_sequences(pipeline)
                ]

            before = shape()
            assert insert_mask_blocks(pipeline) is True
            assert shape() == before, label
            for names, loop in before:
                assert sum(SOURCE_BLOCK in n for n in names) == 1
                assert loop.count(BLEND_BLOCK) == 1

    def test_the_inserted_blocks_are_the_step_classes(self, built):
        from dw.pipeline_processors.h3_mask_steps import (
            DwH3MaskBlendStep,
            DwH3SourceLatentsStep,
        )

        assert h3_mask.blocks() == (DwH3SourceLatentsStep, DwH3MaskBlendStep)
        _, pipeline = built[1]
        insert_mask_blocks(pipeline)
        for prefix, sequence in core_denoise_sequences(pipeline):
            assert isinstance(
                sequence.sub_blocks[prefix + SOURCE_BLOCK], DwH3SourceLatentsStep
            )
            loop = sequence.sub_blocks[prefix + "denoise"]
            assert isinstance(loop.sub_blocks[BLEND_BLOCK], DwH3MaskBlendStep)

    def test_source_video_and_mask_are_pipeline_inputs(self, built):
        for label, pipeline in built:
            insert_mask_blocks(pipeline)
            names = [param.name for param in pipeline._blocks.inputs]
            assert "source_video" in names, label
            assert "mask_grow" in names, label

    def test_a_pipeline_without_the_blocks_does_not_edit(self):
        assert not edits_regions(minimax.MiniMaxH3Blocks().init_pipeline())
        assert edits_regions(object()) is False
        assert insert_mask_blocks(object()) is False

    def test_a_sequence_without_update_gets_neither_block(self):
        pipeline = minimax.MiniMaxH3Blocks().get_workflow("t2va").init_pipeline()
        loop = pipeline._blocks.sub_blocks["denoise.denoise"]
        loop.sub_blocks.pop("update")
        assert insert_mask_blocks(pipeline) is False
        assert "denoise." + SOURCE_BLOCK not in pipeline._blocks.sub_blocks
        assert not edits_regions(pipeline)

    def test_every_anchor_is_in_the_installed_diffusers(self):
        pipeline = minimax.MiniMaxH3Blocks().get_workflow("t2va").init_pipeline()
        sequence = pipeline._blocks
        assert "denoise.set_timesteps" in sequence.sub_blocks
        assert "update" in sequence.sub_blocks["denoise.denoise"].sub_blocks

    def test_components_load_inserts_the_mask_blocks(self):
        # components.py inserts them where it inserts the hold and the guides
        import inspect

        from dw.pipeline_processors import components

        source = inspect.getsource(components)
        assert "insert_mask_blocks(component)" in source
        assert components.insert_mask_blocks is insert_mask_blocks


# 3. The blend


ROWS, WIDTH = 6, 4


def comps(sigmas):
    scheduler = MiniMaxH3Scheduler(shift=6.0)
    scheduler.set_timesteps(sigmas=sigmas)
    return SimpleNamespace(scheduler=scheduler)


def block_state(keep, condition=0, seed=0):
    generator = torch.Generator().manual_seed(seed)
    total = condition + ROWS
    return SimpleNamespace(
        latents=torch.randn(total, WIDTH, generator=generator),
        num_condition_video_rows=condition,
        dw_source_rows=torch.randn(ROWS, WIDTH, generator=generator),
        dw_keep_rows=keep,
        dw_refine_row_noise=torch.randn(ROWS, WIDTH, generator=generator),
    )


def blend():
    from dw.pipeline_processors.h3_mask_steps import DwH3MaskBlendStep

    return DwH3MaskBlendStep()


KEEP = torch.tensor([True, False, True, True, False, True])


def at_last_step(condition=0):
    components = comps([0.4, 0.2, 0.0])
    state = block_state(KEEP, condition)
    # as `update` leaves it after the final step
    components.scheduler._step_index = 2
    return components, state


class TestBlend:
    def test_names_match_the_rules(self):
        state = block_state(KEEP)
        assert hasattr(state, SOURCE_ROWS)
        assert hasattr(state, KEEP_ROWS)
        assert hasattr(state, REFINE_ROW_NOISE)

    def test_at_sigma_zero_kept_rows_are_the_source_exactly(self):
        components, state = at_last_step()
        before = state.latents.clone()
        blend()(components, state, 1, components.scheduler.timesteps[-1])
        assert torch.equal(state.latents[KEEP], state.dw_source_rows[KEEP])
        assert torch.equal(state.latents[~KEEP], before[~KEEP])

    def test_before_the_last_step_kept_rows_are_the_source_noised_to_sigma(self):
        components = comps([0.4, 0.2, 0.0])
        components.scheduler._step_index = 1
        state = block_state(KEEP)
        blend()(components, state, 0, components.scheduler.timesteps[0])
        sigma = components.scheduler.sigmas[1]
        expected = (
            1 - sigma
        ) * state.dw_source_rows + sigma * state.dw_refine_row_noise
        assert torch.allclose(state.latents[KEEP], expected[KEEP], atol=1e-6)

    def test_the_step_index_is_i_plus_one_when_the_scheduler_has_none(self):
        components = comps([0.4, 0.2, 0.0])
        assert components.scheduler.step_index is None
        state = block_state(KEEP)
        blend()(components, state, 1, components.scheduler.timesteps[1])
        assert torch.equal(state.latents[KEEP], state.dw_source_rows[KEEP])

    def test_conditioning_rows_are_never_touched(self):
        components, state = at_last_step(condition=3)
        before = state.latents.clone()
        blend()(components, state, 1, components.scheduler.timesteps[-1])
        assert torch.equal(state.latents[:3], before[:3])
        assert torch.equal(state.latents[3:][KEEP], state.dw_source_rows[KEEP])
        assert torch.equal(state.latents[3:][~KEEP], before[3:][~KEEP])

    def test_the_latents_are_written_in_place(self):
        components, state = at_last_step()
        tensor = state.latents
        blend()(components, state, 1, components.scheduler.timesteps[-1])
        assert state.latents is tensor

    def test_no_keep_is_a_no_op(self):
        components, state = at_last_step()
        state.dw_keep_rows = None
        before = state.latents.clone()
        out_components, out_state = blend()(
            components, state, 1, components.scheduler.timesteps[-1]
        )
        assert out_components is components and out_state is state
        assert torch.equal(state.latents, before)

    def test_no_keep_attribute_is_a_no_op(self):
        components = comps([0.4, 0.0])
        state = SimpleNamespace(latents=torch.zeros(2, 2))
        blend()(components, state, 0, components.scheduler.timesteps[0])
        assert torch.equal(state.latents, torch.zeros(2, 2))

    def test_keep_without_noise_raises(self):
        components, state = at_last_step()
        state.dw_refine_row_noise = None
        with pytest.raises(ValueError, match="refine"):
            blend()(components, state, 1, components.scheduler.timesteps[-1])

    def test_a_mini_loop_ends_on_the_source_in_kept_rows(self):
        sigmas = [0.5, 0.35, 0.2, 0.08, 0.0]
        components = comps(sigmas)
        scheduler = components.scheduler
        state = block_state(KEEP, condition=2, seed=3)
        source, noise = state.dw_source_rows, state.dw_refine_row_noise
        # what the refine block does: re-noise the generated rows from the source
        state.latents[2:] = scheduler.scale_noise(source, 1 - sigmas[0], noise)
        generator = torch.Generator().manual_seed(11)

        for i, t in enumerate(scheduler.timesteps):
            output = torch.randn(state.latents.shape, generator=generator)
            state.latents = scheduler.step(output, t, state.latents).prev_sample
            before_regenerated = state.latents[2:][~KEEP].clone()
            blend()(components, state, i, t)
            sigma = scheduler.sigmas[i + 1]
            kept = state.latents[2:][KEEP]
            expected = (1 - sigma) * source[KEEP] + sigma * noise[KEEP]
            assert torch.allclose(kept, expected, atol=1e-6), i
            assert torch.equal(state.latents[2:][~KEEP], before_regenerated), i

        assert float(scheduler.sigmas[-1]) == 0.0
        assert torch.equal(state.latents[2:][KEEP], source[KEEP])


# 4. The source block


class TestSourceBlock:
    def test_no_source_video_returns_the_state_untouched(self):
        from dw.pipeline_processors.h3_mask_steps import DwH3SourceLatentsStep

        state = PipelineState()
        latents = torch.randn(ROWS, WIDTH)
        state.set("latents", latents)
        for name in ("height", "width", "num_frames", "num_latent_frames"):
            state.set(name, 16)
        state.set("latent_height", 1)
        state.set("latent_width", 1)
        components = SimpleNamespace()

        out_components, out_state = DwH3SourceLatentsStep()(components, state)

        assert out_components is components and out_state is state
        assert state.get("latents") is latents
        assert state.get(SOURCE_ROWS) is None
        assert state.get(KEEP_ROWS) is None

    def run_block(self, monkeypatch, mask, strength):
        """The block over a 3-latent-frame SIDE canvas with the VAE encode
        stubbed, and the events it emitted."""
        from dw.events import RunContext, activate_context, deactivate_context
        from dw.pipeline_processors import h3_mask_steps

        monkeypatch.setattr(
            h3_mask_steps.encoders,
            "encode_vae_condition",
            lambda *a: torch.zeros(1, 1, 3, LATENT_SIDE, LATENT_SIDE),
        )
        state = PipelineState()
        state.set("latents", torch.randn(3 * 4 * 4, 4))
        state.set("source_video", np.zeros((9, SIDE, SIDE, 3), dtype=np.uint8))
        state.set(REGENERATE_MASK, mask)
        state.set("mask_grow", 0)
        state.set("refine_strength", strength)
        for name, value in (("height", SIDE), ("width", SIDE), ("num_frames", 9)):
            state.set(name, value)
        state.set("num_latent_frames", 3)
        state.set("latent_height", LATENT_SIDE)
        state.set("latent_width", LATENT_SIDE)
        components = SimpleNamespace(
            vae=None,
            pixel_mean=None,
            pixel_std=None,
            keyframe_encode_seed=0,
            patch_size=PATCH,
            _execution_device="cpu",
        )
        events = []
        token = activate_context(RunContext(on_event=events.append))
        try:
            h3_mask_steps.DwH3SourceLatentsStep()(components, state)
        finally:
            deactivate_context(token)
        return state, [e for e in events if e.get("event") == "log"]

    def test_a_masked_edit_emits_its_cells_grow_and_strength(self, monkeypatch):
        # #1077: a job's events show the regional edit ran and how hard
        mask = binarized(pixel_mask(1, (0, 0, 0)))
        state, logs = self.run_block(monkeypatch, mask, 0.9)

        assert len(logs) == 1
        log = logs[0]
        assert log["regenerated_cells"] == 3 and log["kept_cells"] == 45
        assert log["mask_grow"] == 0 and log["refine_strength"] == 0.9
        assert "3 of 48 patch cells" in log["message"]
        assert "refine_strength 0.9" in log["message"]
        assert int((~state.get(KEEP_ROWS)).sum()) == 3

    def test_an_unmasked_edit_says_it_re_denoises_the_whole_take(self, monkeypatch):
        state, logs = self.run_block(monkeypatch, None, 0.5)

        assert len(logs) == 1
        assert logs[0]["regenerated_cells"] is None
        assert "whole source_video at refine_strength 0.5" in logs[0]["message"]
        assert state.get(KEEP_ROWS) is None


# 5. Rule refusals


def args(**overrides):
    arguments = {
        "source_video": "asset:clip",
        "mask_video": "asset:mask",
        "mask_grow": 1,
        "refine_strength": 0.9,
    }
    arguments.update(overrides)
    return {k: v for k, v in arguments.items() if v is not None}


def one(problems, name):
    assert len(problems) == 1, problems
    assert name in problems[0]
    return problems[0]


class TestMaskProblems:
    def test_a_valid_set_is_clean(self):
        assert mask_problems(args()) == []
        assert mask_problems(args(mask_video=None, mask_image="asset:m")) == []
        assert mask_problems(args(mask_grow=None)) == []
        assert mask_problems(args(mask_grow=0)) == []

    def test_a_source_alone_is_clean(self):
        assert mask_problems(args(mask_video=None, mask_grow=None)) == []

    def test_no_regional_edit_argument_says_nothing(self):
        assert mask_problems({"prompt": "x"}) == []

    def test_both_masks_are_refused(self):
        problems = mask_problems(args(mask_image="asset:m"))
        one(problems, "mask_video")
        assert "mask_image" in problems[0]

    def test_a_mask_without_source_video_is_refused(self):
        one(
            mask_problems(args(source_video=None, refine_strength=None)), "source_video"
        )
        one(
            mask_problems(
                args(source_video=None, mask_video=None, mask_image="a", mask_grow=None)
            ),
            "mask_image",
        )

    def test_mask_grow_without_a_mask_is_refused(self):
        one(mask_problems(args(mask_video=None)), "mask_grow")

    @pytest.mark.parametrize("grow", [-1, 1.5, "2", True, 2.0])
    def test_a_bad_mask_grow_is_refused(self, grow):
        one(mask_problems(args(mask_grow=grow)), "mask_grow")

    def test_a_deferred_mask_grow_is_not_refused(self):
        assert mask_problems(args(mask_grow="variable:grow")) == []

    def test_source_video_without_refine_strength_is_refused(self):
        one(mask_problems(args(refine_strength=None)), "refine_strength")

    def test_source_video_with_latents_is_refused(self):
        one(mask_problems(args(latents="previous_result:up")), "latents")

    def test_a_chained_step_is_refused(self):
        problems = mask_problems(args(), chained=True)
        assert len(problems) == 1
        assert "chained" in problems[0] or "chain" in problems[0]

    def test_a_chained_step_without_an_edit_says_nothing(self):
        assert mask_problems({"prompt": "x"}, chained=True) == []


class TestOtherRules:
    def test_refine_no_longer_demands_latents_with_a_source_video(self):
        arguments = {
            "refine_strength": 0.9,
            "hold_audio": "previous_result:base.audio",
            "num_inference_steps": 5,
        }
        assert any("latents" in p for p in refine_problems(arguments))
        arguments["source_video"] = "asset:clip"
        assert refine_problems(arguments) == []

    def test_a_size_mismatch_is_reported(self):
        problems = media_problems("source_video", (9, 64, 96), 128, 128)
        assert len(problems) == 1
        assert "96x64" in problems[0] and "128x128" in problems[0]

    def test_a_frame_mismatch_is_reported(self):
        problems = media_problems("mask_video", (5, 128, 128), 128, 128, 9)
        assert len(problems) == 1
        assert "5 frames" in problems[0] and "num_frames=9" in problems[0]

    def test_a_match_is_clean(self):
        assert media_problems("source_video", (9, 128, 128), 128, 128, 9) == []

    def test_a_mask_image_ignores_frames(self):
        assert media_problems("mask_image", (1, 128, 128), 128, 128, 9) == []

    def test_a_mask_image_still_checks_size(self):
        assert len(media_problems("mask_image", (1, 64, 64), 128, 128, 9)) == 1

    def test_no_frame_count_skips_the_frame_check(self):
        assert media_problems("source_video", (3, 128, 128), 128, 128) == []

    def test_an_inpaint_mask_image_on_a_non_h3_step_is_not_a_regional_edit(self):
        assert edits_region({"mask_image": "x"}, False) is False
        assert edits_region({"mask_image": "x"}, True) is True
        assert edits_region({"mask_video": "x"}, False) is False
        assert edits_region({"mask_grow": 1}, False) is False

    def test_source_video_always_is(self):
        assert edits_region({"source_video": "x"}, False) is True
        assert edits_region({"source_video": "x"}, True) is True

    def test_nothing_given_is_not(self):
        assert edits_region({"source_video": None, "mask_image": None}, True) is False

    def test_is_h3_step_reads_the_workflow(self):
        for workflow in WORKFLOWS:
            assert is_h3_step({"from_pretrained_arguments": {"workflow": workflow}})
        assert not is_h3_step({"from_pretrained_arguments": {"workflow": "i2v"}})
        assert not is_h3_step({"from_pretrained_arguments": {}})
        assert not is_h3_step({})


# 5b. Validation end to end


def h3_step(workflow="t2va", chain=None, **arguments):
    pipeline = {
        "configuration": {"component_type": "ModularPipeline"},
        "from_pretrained_arguments": {"workflow": workflow},
        "arguments": arguments,
    }
    if chain is not None:
        pipeline["chain"] = chain
    return {"name": "video", "pipeline": pipeline}


def template(name):
    with open(os.path.join(ROOT, "workflows", "templates", name)) as f:
        return json.load(f)


def edit_errors(definition):
    workflow = workflow_from_definition(definition, tempfile.mkdtemp())
    return [
        problem
        for problem in workflow.validation_errors()
        if any(n in problem["path"] for n in ("source_video", "mask_"))
    ]


def h3_definition(**arguments):
    definition = copy.deepcopy(template("text-to-image.json"))
    pipeline = definition["steps"][0]["pipeline"]
    pipeline["configuration"]["component_type"] = "ModularPipeline"
    pipeline["from_pretrained_arguments"] = {
        "model_name": "MiniMaxAI/MiniMax-H3",
        "workflow": "t2va",
    }
    pipeline["arguments"].update(arguments)
    return definition


class TestStaticEdit:
    def test_a_valid_step_is_clean(self):
        assert mask_errors({"steps": [h3_step(**args())]}) == []

    def test_a_mask_without_source_is_refused_at_the_mask_path(self):
        errors = mask_errors(
            {"steps": [h3_step(mask_video="asset:m", refine_strength=0.9)]}
        )
        assert len(errors) == 1
        assert errors[0]["path"].endswith("mask_video")
        assert "source_video" in errors[0]["message"]

    def test_the_path_is_the_first_argument_the_step_passes(self):
        errors = mask_errors({"steps": [h3_step(**args(mask_grow=-1))]})
        assert len(errors) == 1
        assert errors[0]["path"].endswith("source_video")
        assert "mask_grow" in errors[0]["message"]

    def test_a_chained_step_is_refused(self):
        errors = mask_errors({"steps": [h3_step(chain={"segments": 2}, **args())]})
        assert len(errors) == 1

    def test_a_non_h3_workflow_is_refused(self):
        errors = mask_errors({"steps": [h3_step("i2v", **args())]})
        assert errors and "i2v" in errors[0]["message"]

    def test_an_h3_step_without_edit_arguments_is_clean(self):
        assert mask_errors({"steps": [h3_step(prompt="x")]}) == []

    def test_a_non_h3_inpaint_mask_image_is_not_flagged(self):
        definition = copy.deepcopy(template("inpaint.json"))
        arguments = definition["steps"][0]["pipeline"]["arguments"]
        assert "mask_image" in arguments
        assert mask_errors(definition) == []
        assert edit_errors(definition) == []

    def test_a_mask_without_source_surfaces_through_the_workflow(self):
        definition = h3_definition(mask_image="asset:m", refine_strength=0.9)
        errors = edit_errors(definition)
        assert errors
        assert any(e["path"].endswith("mask_image") for e in errors)
        assert any("source_video" in e["message"] for e in errors)

    def test_the_check_is_registered(self):
        from dw import validation

        source = open(validation.__file__).read()
        assert '"regional_edit",' in source
        assert (
            "mask_errors(c.expanded, c.source_indices, c.base_dir, probe=c.probe)"
            in (source)
        )


# 5c. Validation reads the files: size, frame count and an empty mask (#1077)


def write_video(path, frames=9, size=(64, 48), value=0, sample_rate=None):
    """A real h264 mp4 of `frames` flat frames at `value`, `size` (w, h), with a
    soundtrack when `sample_rate` is given."""
    import av

    container = av.open(str(path), "w")
    video = container.add_stream("libx264", rate=8)
    video.width, video.height, video.pix_fmt = size[0], size[1], "yuv420p"
    audio = container.add_stream("aac", rate=sample_rate) if sample_rate else None
    pixels = np.full((size[1], size[0], 3), value, dtype=np.uint8)
    for _ in range(frames):
        frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
        container.mux(video.encode(frame))
    container.mux(video.encode())
    if audio is not None:
        samples = np.zeros((1, sample_rate * frames // 8), dtype=np.float32)
        frame = av.AudioFrame.from_ndarray(samples, format="flt", layout="mono")
        frame.sample_rate = sample_rate
        container.mux(audio.encode(frame))
        container.mux(audio.encode())
    container.close()
    return path.name


def write_mask(path, size=(64, 48), value=255):
    from PIL import Image

    Image.new("L", size, value).save(path)
    return path.name


def probed_errors(tmp_path, **arguments):
    from dw.media import probe_metadata

    arguments.setdefault("refine_strength", 0.9)
    return mask_errors(
        {"steps": [h3_step(**arguments)]}, None, str(tmp_path), probe=probe_metadata
    )


class TestStaticMedia:
    def test_a_fitting_edit_is_clean(self, tmp_path):
        source = write_video(tmp_path / "src.mp4")
        mask = write_mask(tmp_path / "m.png")
        assert probed_errors(tmp_path, source_video=source, mask_image=mask) == []

    def test_the_canvas_defaults_to_the_source(self, tmp_path):
        source = write_video(tmp_path / "src.mp4")
        assert probed_errors(tmp_path, source_video=source) == []

    def test_a_source_off_the_canvas_is_refused(self, tmp_path):
        source = write_video(tmp_path / "src.mp4")
        errors = probed_errors(tmp_path, source_video=source, width=128, height=48)
        assert len(errors) == 1
        assert errors[0]["path"].endswith("source_video")
        assert "64x48" in errors[0]["message"] and "128x48" in errors[0]["message"]

    def test_a_source_frame_mismatch_is_refused(self, tmp_path):
        source = write_video(tmp_path / "src.mp4", frames=9)
        errors = probed_errors(tmp_path, source_video=source, num_frames=17)
        assert len(errors) == 1
        assert "num_frames=17" in errors[0]["message"]

    def test_a_mask_image_of_another_size_is_refused_at_its_path(self, tmp_path):
        source = write_video(tmp_path / "src.mp4")
        mask = write_mask(tmp_path / "m.png", size=(32, 32))
        errors = probed_errors(tmp_path, source_video=source, mask_image=mask)
        assert len(errors) == 1
        assert errors[0]["path"].endswith("mask_image")
        assert "32x32" in errors[0]["message"]

    def test_an_all_black_mask_image_is_refused(self, tmp_path):
        source = write_video(tmp_path / "src.mp4")
        mask = write_mask(tmp_path / "m.png", value=0)
        errors = probed_errors(tmp_path, source_video=source, mask_image=mask)
        assert len(errors) == 1
        assert errors[0]["path"].endswith("mask_image")
        assert "all black" in errors[0]["message"]

    def test_an_all_black_mask_video_is_refused(self, tmp_path):
        source = write_video(tmp_path / "src.mp4")
        mask = write_video(tmp_path / "m.mp4", value=0)
        errors = probed_errors(tmp_path, source_video=source, mask_video=mask)
        assert len(errors) == 1
        assert errors[0]["path"].endswith("mask_video")
        assert "all black" in errors[0]["message"]

    def test_a_white_mask_video_that_fits_is_clean(self, tmp_path):
        source = write_video(tmp_path / "src.mp4")
        mask = write_video(tmp_path / "m.mp4", value=255)
        assert probed_errors(tmp_path, source_video=source, mask_video=mask) == []

    def test_a_mask_video_frame_mismatch_is_refused(self, tmp_path):
        source = write_video(tmp_path / "src.mp4", frames=9)
        mask = write_video(tmp_path / "m.mp4", frames=5, value=255)
        errors = probed_errors(tmp_path, source_video=source, mask_video=mask)
        assert len(errors) == 1
        assert errors[0]["path"].endswith("mask_video")
        assert "5 frames" in errors[0]["message"]

    def test_a_mask_video_that_is_an_image_is_refused(self, tmp_path):
        source = write_video(tmp_path / "src.mp4")
        mask = write_mask(tmp_path / "m.png")
        errors = probed_errors(tmp_path, source_video=source, mask_video=mask)
        assert len(errors) == 1
        assert errors[0]["path"].endswith("mask_video")
        assert "1 frames" in errors[0]["message"]

    def test_without_a_probe_only_the_rules_are_checked(self, tmp_path):
        source = write_video(tmp_path / "src.mp4")
        mask = write_mask(tmp_path / "m.png", value=0)
        step = h3_step(source_video=source, mask_image=mask, refine_strength=0.9)
        assert mask_errors({"steps": [step]}, None, str(tmp_path)) == []

    def test_a_previous_result_is_left_to_the_call(self, tmp_path):
        errors = probed_errors(
            tmp_path,
            source_video="previous_result:base.video",
            mask_video="previous_result:mask",
        )
        assert errors == []

    def test_the_size_refusal_surfaces_through_the_workflow(self, tmp_path):
        source = write_video(tmp_path / "src.mp4")
        mask = write_mask(tmp_path / "m.png", value=0)
        definition = h3_definition(
            source_video=source,
            mask_image=mask,
            refine_strength=0.9,
            hold_audio=source,
            width=64,
            height=48,
            num_frames=9,
        )
        workflow = workflow_from_definition(
            definition, str(tmp_path), base_dir=str(tmp_path)
        )
        errors = [
            e
            for e in workflow.validation_errors()
            if any(n in e["path"] for n in ("source_video", "mask_"))
        ]
        assert any("all black" in e["message"] for e in errors), errors


class TestStrengthOne:
    def test_one_is_accepted_with_a_source_video(self):
        arguments = {
            "refine_strength": 1.0,
            "source_video": "asset:clip.mp4",
            "hold_audio": "asset:clip.mp4",
            "num_inference_steps": 5,
        }
        assert refine_problems(arguments) == []

    def test_one_is_refused_without_a_source_video(self):
        arguments = {
            "refine_strength": 1.0,
            "latents": "previous_result:up",
            "hold_audio": "previous_result:base.audio",
            "num_inference_steps": 5,
        }
        problems = refine_problems(arguments)
        assert len(problems) == 1 and "source_video" in problems[0]

    def test_above_one_is_refused_with_a_source_video(self):
        arguments = {
            "refine_strength": 1.5,
            "source_video": "asset:clip.mp4",
            "hold_audio": "asset:clip.mp4",
        }
        problems = refine_problems(arguments)
        assert len(problems) == 1 and "(0, 1]" in problems[0]

    def test_the_schedule_at_one_is_the_stock_grid(self):
        from dw.pipeline_processors.h3_hold_steps import (
            refine_sigmas,
            shifted_sigma_grid,
        )

        sigmas = refine_sigmas(1.0, 5, 12)
        stock = shifted_sigma_grid(5, 12.0).float()
        stock[-1] = 0.0
        assert sigmas[0] == 1.0
        assert np.allclose(sigmas.numpy(), stock.numpy())

    def test_a_validated_step_at_one_is_clean(self):
        step = h3_step(
            **args(refine_strength=1.0, hold_audio="asset:clip.mp4"),
            num_inference_steps=5,
        )
        from dw.hold_audio import refine_strength_errors

        assert refine_strength_errors({"steps": [step]}) == []


# The call: Pipeline._with_mask


def editing_step(pipeline, chained=False):
    return SimpleNamespace(
        name="video",
        pipeline=pipeline,
        base_dir=None,
        segment_label="segment 1" if chained else None,
    )


@pytest.fixture
def editing():
    pipeline = minimax.MiniMaxH3Blocks().get_workflow("t2va").init_pipeline()
    h3_mask.insert_mask_blocks(pipeline)
    return pipeline


def call_arguments(**overrides):
    from PIL import Image

    mask = np.zeros((32, 48, 3), dtype=np.uint8)
    mask[:8, :8] = 255
    arguments = {
        "source_video": [Image.new("RGB", (48, 32)) for _ in range(5)],
        "mask_image": Image.fromarray(mask),
        "refine_strength": 0.5,
        "hold_audio": object(),
    }
    arguments.update(overrides)
    return {k: v for k, v in arguments.items() if v is not None}


class TestTheCall:
    def call(self, pipeline, chained=False, **overrides):
        from dw.pipeline_processors.pipeline import Pipeline

        step = editing_step(pipeline, chained)
        return Pipeline._with_mask(step, call_arguments(**overrides))

    def test_the_mask_becomes_the_internal_bool_frames(self, editing):
        arguments = self.call(editing)
        assert "mask_image" not in arguments
        mask = arguments[REGENERATE_MASK]
        assert mask.shape == (1, 32, 48) and mask.dtype == bool
        assert mask[0, :8, :8].all() and not mask[0, 8:, 8:].any()
        assert arguments["source_video"].shape == (5, 32, 48, 3)

    def test_an_unsized_step_renders_the_sources_canvas(self, editing):
        arguments = self.call(editing)
        assert (arguments["width"], arguments["height"], arguments["num_frames"]) == (
            48,
            32,
            5,
        )

    def test_a_canvas_mismatch_is_refused(self, editing):
        with pytest.raises(ValueError, match="must match"):
            self.call(editing, width=64)

    def test_an_all_black_mask_is_refused(self, editing):
        from PIL import Image

        with pytest.raises(ValueError, match="all black"):
            self.call(editing, mask_image=Image.new("RGB", (48, 32)))

    def test_a_chained_step_is_refused(self, editing):
        with pytest.raises(ValueError, match="chain"):
            self.call(editing, chained=True)

    def test_a_pipeline_without_the_blocks_refuses_source_video(self):
        with pytest.raises(ValueError, match="MiniMax-H3"):
            self.call(object())

    def test_an_inpaint_mask_image_passes_through(self):
        from dw.pipeline_processors.pipeline import Pipeline

        arguments = {"mask_image": object(), "prompt": "x"}
        step = editing_step(object())
        assert Pipeline._with_mask(step, arguments) is arguments
