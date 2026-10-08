"""MiniMax-H3 guides (#611): the aligned lengths, the guide rows spliced into the
packed layout, the blocks dw inserts and the call-time checks on `guides`."""

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from dw.media_types import AudioVideo
from dw.pipeline_processors import h3_guides, h3_hold, h3_rules
from dw.pipeline_processors import pipeline as pipeline_module
from dw.pipeline_processors.h3_guides import (
    LAYOUT_ANCHORS,
    default_num_frames,
    fit_guide_frames,
    guide_audio_latents,
    guide_audio_positions,
    guide_audio_span,
    guide_audio_waveform,
    guide_blocks,
    guide_frames_array,
    guide_position_ids,
    guides_refusal,
    insert_guides,
    layout_anchor_problem,
    pad_guide_audio,
    splice_guide_audio_rows,
    splice_guide_rows,
    takes_guides,
)
from dw.pipeline_processors.h3_hold import insert_audio_hold
from dw.pipeline_processors.h3_rules import (
    GUIDE_AUDIO_BLOCK,
    GUIDE_AUDIO_ROWS,
    GUIDE_CONDITION_BLOCK,
    GUIDE_FRAMES_PER_CHUNK,
    GUIDE_LATENTS_BLOCK,
    GUIDE_LATENTS_PER_CHUNK,
    GUIDE_LIMIT,
    RENDER_GRID,
    guide_end_problem,
    guide_frame_problem,
    guide_latent_frames,
    snap_guide_length,
)
from dw.pipeline_processors.pipeline import Pipeline
from dw.variable_constraints import aligned_down

minimax = pytest.importorskip("diffusers.modular_pipelines.minimax_h3")
from diffusers.modular_pipelines.minimax_h3 import before_denoise  # noqa: E402
from diffusers.modular_pipelines.modular_pipeline import PipelineState  # noqa: E402

Stock = before_denoise.MiniMaxH3PrepareLayoutStep

TEXT = 10
LATENT_FRAMES = 37
LAT_H, LAT_W = 4, 6
PATCH = (1, 2, 2)
AUDIO_LATENTS = 8
CHANNELS = 2
AUDIO_TAG, VIDEO_TAG = 7, 9
ROWS_PER_FRAME = (LAT_H // 2) * (LAT_W // 2)


def stock_layout(anchors=()):
    tags = torch.ones(TEXT, dtype=torch.long)
    return Stock.build_packed_sequence(
        tags,
        LATENT_FRAMES,
        LAT_H,
        LAT_W,
        AUDIO_LATENTS,
        PATCH,
        CHANNELS,
        AUDIO_TAG,
        VIDEO_TAG,
        anchors,
    )[:6]


def target_time():
    return before_denoise._temporal_position_grid(LATENT_FRAMES, float(TEXT))


def frame_grid():
    return before_denoise._frame_position_grid(LAT_H, LAT_W, 2, 2)[0]


def guide_positions(frame, latents):
    return guide_position_ids(target_time(), frame_grid(), frame // 17 * 5, latents)


def assert_layouts_equal(a, b):
    for x, y in zip(a[:5], b[:5]):
        assert torch.equal(x, y)
    assert a[5] == b[5]


# 1. Lengths, frames, ends


class TestLengths:
    @pytest.mark.parametrize(
        "n, snapped",
        [
            (1, 1),
            (2, 1),
            (3, 1),
            (4, 1),
            (5, 5),
            (6, 5),
            (21, 5),
            (22, 22),
            (23, 22),
            (30, 22),
            (39, 39),
            (40, 39),
            (56, 56),
        ],
    )
    def test_snap_guide_length(self, n, snapped):
        assert snap_guide_length(n) == snapped

    @pytest.mark.parametrize("n", range(22, 200))
    def test_long_snap_is_the_constraint_owners_floor(self, n):
        # The 17m + 5 tail is variable_constraints' grid, not a copy of it
        assert snap_guide_length(n) == aligned_down(n, RENDER_GRID)

    def test_default_num_frames_is_the_stock_layout_default(self):
        stock = before_denoise.MiniMaxH3PrepareLayoutStep()
        (param,) = [p for p in stock.inputs if p.name == "num_frames"]
        assert default_num_frames() == param.default == 124

    @pytest.mark.parametrize(
        "n, latents", [(1, 1), (5, 2), (22, 7), (39, 12), (56, 17)]
    )
    def test_guide_latent_frames(self, n, latents):
        assert guide_latent_frames(n) == latents


def vae_defaults():
    """The H3 video VAE's config defaults, read off its signature - no weights."""
    import inspect

    from diffusers import AutoencoderKLMiniMaxH3

    return {
        name: param.default
        for name, param in inspect.signature(
            AutoencoderKLMiniMaxH3.__init__
        ).parameters.items()
    }


class TestChunkDrift:
    """The chunk sizes are copies of the installed VAE's temporal geometry."""

    def test_frames_per_chunk_is_the_vae_clip_length(self):
        assert GUIDE_FRAMES_PER_CHUNK == vae_defaults()["clip_length"]

    def test_latents_per_chunk_is_the_vae_tokens_per_clip(self):
        # The VAE's own derivation of `tokens_chunk_size`
        config = vae_defaults()
        ratio = math.prod(config["temporal_downsample_factors"])
        assert GUIDE_LATENTS_PER_CHUNK == math.ceil(config["clip_length"] / ratio)

    @pytest.mark.parametrize("chunks", [1, 2, 3, 7])
    def test_guide_latent_frames_is_the_vae_encode_length(self, chunks):
        # `_encode` pads to whole clips and drops `token_drop` trailing latents
        config = vae_defaults()
        ratio = math.prod(config["temporal_downsample_factors"])
        n = config["clip_length"] * (chunks - 1) + 5
        encoded = (
            chunks * math.ceil(config["clip_length"] / ratio) - config["token_drop"]
        )
        assert guide_latent_frames(n) == encoded

    @pytest.mark.parametrize("frame", [0, 17, 34, 119])
    def test_aligned_frames_pass(self, frame):
        assert guide_frame_problem(frame) is None

    @pytest.mark.parametrize("frame", [-17, 1, 16, 18, True, False, 17.0, "17", None])
    def test_other_frames_are_refused(self, frame):
        assert guide_frame_problem(frame)

    def test_a_misaligned_frame_names_its_neighbours(self):
        assert "use 17 or 34" in guide_frame_problem(20)

    def test_ending_exactly_at_the_end_fits(self):
        assert guide_end_problem(102, 22, 124) is None
        assert guide_end_problem(0, 124, 124) is None

    def test_past_the_end_is_refused(self):
        assert "past the end" in guide_end_problem(119, 22, 124)
        assert guide_end_problem(0, 125, 124)


# 2. Rotary origin


class TestPositions:
    @pytest.mark.parametrize("frame", [0, 17, 34])
    def test_origin_is_the_target_time_it_lands_on(self, frame):
        positions = guide_positions(frame, 7)
        assert positions[0, 0] == target_time()[frame // 17 * 5]

    def test_rows_and_spacing(self):
        positions = guide_positions(17, 7)
        assert positions.shape == (7 * ROWS_PER_FRAME, 3)
        times = positions[::ROWS_PER_FRAME, 0]
        target = target_time()
        assert torch.allclose(times - times[0], target[:7] - target[0])
        assert torch.equal(positions[:ROWS_PER_FRAME, 1:], frame_grid())
        assert torch.equal(positions[-ROWS_PER_FRAME:, 1:], frame_grid())
        assert torch.all(positions[:ROWS_PER_FRAME, 0] == positions[0, 0])

    def test_a_guide_at_the_start_matches_the_target_clock(self):
        positions = guide_positions(0, 5)
        assert torch.allclose(positions[::ROWS_PER_FRAME, 0], target_time()[:5])


# 3. Splice


class TestSplice:
    def test_one_frame_guide_is_a_first_keyframe(self):
        plain = stock_layout()
        spliced = splice_guide_rows(plain, TEXT, guide_positions(0, 1), VIDEO_TAG)
        assert_layouts_equal(spliced, stock_layout(("first",)))

    def test_two_guides_over_a_keyframe(self):
        base = stock_layout(("first",))
        a, b = guide_positions(0, 2), guide_positions(17, 7)
        count = a.shape[0] + b.shape[0]
        position_ids, tags, video, audio, text, rows = splice_guide_rows(
            base, TEXT, torch.cat([a, b]), VIDEO_TAG
        )
        assert rows == base[5] + count == ROWS_PER_FRAME + count
        assert position_ids.shape[0] == base[0].shape[0] + count
        start = TEXT + ROWS_PER_FRAME
        assert torch.equal(position_ids[start : start + count], torch.cat([a, b]))
        assert torch.all(tags[start : start + count] == VIDEO_TAG)
        assert torch.equal(audio, base[3] + count)
        assert torch.equal(text, base[4])
        assert audio[0] == start + count
        assert torch.equal(video[: TEXT + rows - TEXT], torch.arange(TEXT, TEXT + rows))
        everything = torch.cat([text, audio, video]).sort().values
        assert torch.equal(everything, torch.arange(position_ids.shape[0]))
        assert torch.equal(tags[audio], torch.full_like(audio, AUDIO_TAG))
        assert torch.all(tags[video] == VIDEO_TAG)

    def test_zero_rows_is_the_same_layout(self):
        base = stock_layout(("first",))
        assert splice_guide_rows(base, TEXT, torch.empty(0, 3), VIDEO_TAG) is base


# 4. Frames


def uint8_frames(n=3, h=4, w=6, channels=3):
    return np.random.default_rng(0).integers(0, 255, (n, h, w, channels), np.uint8)


class TestFramesArray:
    def test_pil_list(self):
        frames = uint8_frames()
        result = guide_frames_array([Image.fromarray(f) for f in frames])
        assert result.shape == (3, 4, 6, 3) and result.dtype == np.uint8
        assert np.array_equal(result, frames)

    def test_uint8_array(self):
        frames = uint8_frames()
        assert np.array_equal(guide_frames_array(frames), frames)

    def test_float_tensor(self):
        result = guide_frames_array(torch.ones(2, 4, 6, 3) * 0.5)
        assert result.dtype == np.uint8 and result.shape == (2, 4, 6, 3)
        assert np.all(result == 128)

    def test_float_values_are_clipped(self):
        result = guide_frames_array(np.full((1, 2, 2, 3), 2.0, np.float32))
        assert np.all(result == 255)

    def test_batch_of_one(self):
        assert guide_frames_array(uint8_frames()[None]).shape == (3, 4, 6, 3)

    def test_rgba_is_dropped_to_rgb(self):
        assert guide_frames_array(uint8_frames(channels=4)).shape == (3, 4, 6, 3)

    def test_audio_video_unwraps(self):
        frames = uint8_frames()
        clip = AudioVideo(frames, torch.zeros(1, 10), 16000, fps=24)
        assert np.array_equal(guide_frames_array(clip), frames)

    def test_one_video_list_unwraps(self):
        frames = uint8_frames()
        assert np.array_equal(guide_frames_array([frames]), frames)

    @pytest.mark.parametrize(
        "value",
        [np.zeros((4, 6, 3), np.uint8), np.zeros((3, 4, 6, 2), np.uint8), [], 5],
    )
    def test_bad_values_raise(self, value):
        with pytest.raises(ValueError):
            guide_frames_array(value)

    def test_an_unloaded_path_raises(self):
        with pytest.raises(ValueError, match="did not load"):
            guide_frames_array(["clip.mp4", "b"])


class TestFit:
    def test_same_size_is_the_same_object(self):
        frames = uint8_frames(2, 8, 8)
        assert fit_guide_frames(frames, 8, 8) is frames

    def test_wide_clip_onto_a_square_canvas(self):
        result = fit_guide_frames(uint8_frames(2, 8, 16), 8, 8)
        assert result.shape == (2, 8, 8, 3) and result.dtype == np.uint8


# 6. insert_guides


def fresh(workflow=None):
    blocks = minimax.MiniMaxH3Blocks()
    if workflow:
        blocks = blocks.get_workflow(workflow)
    pipeline = blocks.init_pipeline()
    insert_audio_hold(pipeline)
    return pipeline


def names(pipeline):
    return [
        (prefix, list(sequence.sub_blocks))
        for prefix, sequence in h3_hold.core_denoise_sequences(pipeline)
    ]


class TestInsert:
    def test_idempotent(self):
        pipeline = fresh()
        assert insert_guides(pipeline) is True
        once = names(pipeline)
        assert insert_guides(pipeline) is True
        assert names(pipeline) == once

    def test_t2va_gets_the_condition_steps(self):
        pipeline = fresh("t2va")
        assert insert_guides(pipeline)
        ((prefix, order),) = names(pipeline)
        condition = order.index(prefix + GUIDE_CONDITION_BLOCK)
        assert order[condition + 1] == prefix + "prepare_latents"
        assert order[condition + 2] == prefix + GUIDE_LATENTS_BLOCK
        sequence = h3_hold.core_denoise_sequences(pipeline)[0][1]
        assert isinstance(
            sequence.sub_blocks[prefix + "prepare_layout"], guide_blocks()[0]
        )

    def test_fl2va_keeps_its_steps(self):
        stock = names(fresh("fl2va"))
        pipeline = fresh("fl2va")
        assert insert_guides(pipeline)
        ((prefix, order),) = names(pipeline)
        # fl2va's own steps, plus only the guide audio block, before the audio
        # hold writes the target rows after the guide's
        assert [n for n in order if n != prefix + GUIDE_AUDIO_BLOCK] == stock[0][1]
        audio = order.index(prefix + GUIDE_AUDIO_BLOCK)
        assert order[audio + 1] == prefix + h3_rules.HOLD_BLOCK
        assert order.index(prefix + "prepare_latents_fl2va") < audio
        sequence = h3_hold.core_denoise_sequences(pipeline)[0][1]
        assert isinstance(
            sequence.sub_blocks[prefix + "prepare_layout"], guide_blocks()[0]
        )

    def test_ref2va_is_untouched(self):
        stock = names(fresh("ref2va"))
        pipeline = fresh("ref2va")
        assert insert_guides(pipeline) is False
        assert names(pipeline) == stock
        assert not takes_guides(pipeline)
        assert "ref2va" not in (guides_refusal(pipeline) or "")
        assert guides_refusal(pipeline)

    def test_whole_graph_swaps_only_the_two_layouts(self):
        pipeline = fresh()
        assert insert_guides(pipeline)
        kinds = [
            type(sequence.sub_blocks[prefix + "prepare_layout"]).__name__
            for prefix, sequence in h3_hold.core_denoise_sequences(pipeline)
        ]
        assert kinds.count("DwH3GuideLayoutStep") == 2
        assert len(kinds) == 3

    def test_guides_is_an_input(self):
        pipeline = fresh()
        insert_guides(pipeline)
        assert "guides" in [p.name for p in pipeline._blocks.inputs]
        assert takes_guides(pipeline)
        assert guides_refusal(pipeline) is None

    def test_a_pipeline_without_the_insert_takes_none(self):
        pipeline = fresh()
        assert not takes_guides(pipeline)
        assert guides_refusal(pipeline)


class TestAnchors:
    def test_installed_diffusers_has_every_anchor(self):
        assert layout_anchor_problem() is None

    def test_a_missing_anchor_blocks_the_insert(self, monkeypatch):
        monkeypatch.setattr(before_denoise, "_frame_position_grid", None)
        monkeypatch.delattr(before_denoise, "_frame_position_grid")
        assert layout_anchor_problem() == "_frame_position_grid"
        pipeline = fresh("t2va")
        before = names(pipeline)
        assert insert_guides(pipeline) is False
        assert names(pipeline) == before
        assert "_frame_position_grid" in guides_refusal(pipeline)

    def test_the_audio_position_helper_is_anchored(self, monkeypatch):
        # guide_audio_positions imports it from before_denoise
        assert "_fill_audio_positions" in LAYOUT_ANCHORS
        monkeypatch.delattr(before_denoise, "_fill_audio_positions")
        assert layout_anchor_problem() == "_fill_audio_positions"

    def test_a_missing_nested_anchor(self, monkeypatch):
        monkeypatch.delattr(Stock, "build_packed_sequence")
        assert (
            layout_anchor_problem()
            == "MiniMaxH3PrepareLayoutStep.build_packed_sequence"
        )


# 7. t2va wrapper steps


def components_for_steps():
    scheduler = SimpleNamespace(
        scale_noise=lambda condition, strength, noise: condition + noise * 0
    )
    return SimpleNamespace(
        _execution_device=torch.device("cpu"),
        patch_size=PATCH,
        scheduler=scheduler,
        keyframe_noise_aug=0.999,
    )


def drive(block, state, components):
    return block(components, state)[1]


class TestWrapperSteps:
    @pytest.mark.parametrize(
        "condition",
        [None, []],
    )
    @pytest.mark.parametrize(
        "which",
        ["condition", "latents"],
    )
    def test_no_condition_latents_is_a_no_op(self, condition, which):
        step = guide_blocks()[{"condition": 1, "latents": 2}[which]]
        latents = torch.randn(12, 3)
        state = PipelineState()
        state.set("latents", latents)
        state.set("num_condition_video_rows", 0)
        if condition is not None:
            state.set("condition_latents", condition)
        state = drive(step(), state, components_for_steps())
        assert state.get("latents") is latents
        assert state.get("condition_rows") is None

    @pytest.mark.parametrize("which", ["condition", "latents"])
    def test_no_condition_latents_draws_nothing_from_the_seed(self, which):
        """#648 C-F302: a run without guides must draw the same noise as stock t2va"""
        step = guide_blocks()[{"condition": 1, "latents": 2}[which]]
        generator = torch.Generator().manual_seed(42)
        before = generator.get_state().clone()
        state = PipelineState()
        state.set("latents", torch.randn(12, 3))
        state.set("num_condition_video_rows", 0)
        state.set("generator", generator)
        drive(step(), state, components_for_steps())
        assert torch.equal(generator.get_state(), before)

    def test_with_a_condition_latent_they_pack_and_prepend(self):
        _, condition_step, latents_step, _ = guide_blocks()
        condition = torch.randn(1, 4, 2, LAT_H, LAT_W)
        rows = 2 * ROWS_PER_FRAME
        latents = torch.randn(12, 16)
        state = PipelineState()
        state.set("latents", latents)
        state.set("num_condition_video_rows", rows)
        state.set("condition_latents", [condition])
        state.set("generator", torch.Generator().manual_seed(0))
        state = drive(condition_step(), state, components_for_steps())
        packed = state.get("condition_rows")
        assert packed.shape[0] == rows
        state = drive(latents_step(), state, components_for_steps())
        assert state.get("latents").shape[0] == rows + 12
        assert torch.equal(state.get("latents")[:rows], packed)
        assert torch.equal(state.get("latents")[rows:], latents)


# 8. The layout block


def layout_components():
    return SimpleNamespace(
        _execution_device=torch.device("cpu"),
        canvas_multiple=32,
        config=SimpleNamespace(canvas_short_edge=768, canvas_max_pixels=768 * 1344),
        vae_frames_per_chunk=17,
        vae_latents_per_chunk=5,
        fps=24,
        min_duration=5,
        max_duration=15,
        vae_spatial_compression_ratio=16,
        patch_size=PATCH,
        audio_channels=CHANNELS,
        audio_tag=AUDIO_TAG,
        video_tag=VIDEO_TAG,
        pixel_mean=0,
        pixel_std=1,
        keyframe_encode_seed=0,
        vae=None,
    )


FRAMES = 124
SIDE = 64
LATENTS = 5 * ((FRAMES - 5) // 17) + 2
STATE_KEYS = [
    "height",
    "width",
    "num_frames",
    "num_latent_frames",
    "latent_height",
    "latent_width",
    "num_audio_latents",
    "position_ids",
    "token_tags",
    "video_indices",
    "audio_indices",
    "text_indices",
    "num_condition_video_rows",
    "num_condition_audio_rows",
]


def layout_state(guides=None, condition_latents=None):
    state = PipelineState()
    state.set("text_token_tags", torch.ones(TEXT, dtype=torch.long))
    state.set("height", SIDE)
    state.set("width", SIDE)
    state.set("num_frames", FRAMES)
    state.set("keyframe_anchors", ())
    if guides is not None:
        state.set("guides", guides)
    if condition_latents is not None:
        state.set("condition_latents", condition_latents)
    return state


def same(a, b):
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    return a == b


class TestLayoutBlock:
    @pytest.mark.parametrize("guides", [None, []])
    def test_no_guides_is_the_stock_layout(self, guides):
        stock = drive(Stock(), layout_state(), layout_components())
        ours = drive(guide_blocks()[0](), layout_state(guides), layout_components())
        for key in STATE_KEYS:
            assert same(stock.get(key), ours.get(key)), key
        assert not ours.get("condition_latents")

    def encoder(self, monkeypatch, seen):
        def encode(vae, pixels, mean, std, seed):
            seen.append(tuple(pixels.shape))
            frames = pixels.shape[2]
            latents = 1 if frames == 1 else (frames - 5) // 17 * 5 + 2
            return torch.zeros(1, 4, latents, SIDE // 16, SIDE // 16)

        monkeypatch.setattr(
            "diffusers.modular_pipelines.minimax_h3.encoders.encode_vae_condition",
            encode,
        )
        return encode

    def make_block(self, monkeypatch, seen):
        # The block binds the encoder when the guide blocks are built
        monkeypatch.setattr(h3_guides, "_GUIDE_BLOCKS", None)
        self.encoder(monkeypatch, seen)
        return h3_guides.guide_blocks()[0]

    def test_one_guide_adds_its_rows(self, monkeypatch):
        seen = []
        block = self.make_block(monkeypatch, seen)
        try:
            video = np.zeros((22, 32, 32, 3), np.uint8)
            state = drive(
                block(),
                layout_state([{"video": video, "frame": 17}]),
                layout_components(),
            )
        finally:
            monkeypatch.setattr(h3_guides, "_GUIDE_BLOCKS", None)
        assert seen == [(1, 3, 22, SIDE, SIDE)]
        rows_per_frame = (SIDE // 16 // 2) ** 2
        guide_rows = 7 * rows_per_frame
        stock = drive(Stock(), layout_state(), layout_components())
        assert state.get("num_condition_video_rows") == guide_rows
        assert len(state.get("condition_latents")) == 1
        assert state.get("condition_latents")[0].shape[2] == 7
        assert (
            state.get("position_ids").shape[0]
            == stock.get("position_ids").shape[0] + guide_rows
        )
        assert torch.equal(
            state.get("audio_indices"), stock.get("audio_indices") + guide_rows
        )
        assert state.get("num_latent_frames") == LATENTS

    def test_keyframe_latents_stay_first(self, monkeypatch):
        seen = []
        block = self.make_block(monkeypatch, seen)
        marker = torch.full((1, 4, 1, SIDE // 16, SIDE // 16), 3.0)
        try:
            state = layout_state(
                [{"video": np.zeros((1, 64, 64, 3), np.uint8), "frame": 0}],
                condition_latents=[marker],
            )
            state.set("keyframe_anchors", ("first",))
            state = drive(block(), state, layout_components())
        finally:
            monkeypatch.setattr(h3_guides, "_GUIDE_BLOCKS", None)
        latents = state.get("condition_latents")
        assert len(latents) == 2 and latents[0] is marker
        assert latents[1].abs().sum() == 0

    def test_past_the_end_raises(self, monkeypatch):
        seen = []
        block = self.make_block(monkeypatch, seen)
        try:
            video = np.zeros((22, 64, 64, 3), np.uint8)
            with pytest.raises(ValueError, match="past the end"):
                drive(
                    block(),
                    layout_state([{"video": video, "frame": 119 // 17 * 17 + 17}]),
                    layout_components(),
                )
        finally:
            monkeypatch.setattr(h3_guides, "_GUIDE_BLOCKS", None)
        assert seen == []


# 8a. A guide's audio (#649)

AUDIO_C = 4


def audio_layout_components():
    return SimpleNamespace(
        **vars(layout_components()),
        audio_sampling_rate=16000,
        audio_latent_channels=AUDIO_C,
    )


class TestGuideAudioMath:
    @pytest.mark.parametrize("frames, latents", [(22, 37), (39, 65), (1, 2), (5, 8)])
    def test_latent_count(self, frames, latents):
        assert guide_audio_latents(frames) == latents

    def test_span_at_frame_zero(self):
        assert guide_audio_span(0, 39) == (0, 0.0, 65)

    def test_span_starts_at_or_before_the_frame(self):
        start, pad, count = guide_audio_span(17, 22)
        assert start == 28  # floor(17 * 40 / 24)
        assert pad == pytest.approx(17 / 24 - 28 / 40)
        assert 0 <= pad < 1 / 40
        # It ends where the clip's last frame does on the target audio grid
        assert start + count == guide_audio_latents(17 + 22)

    @pytest.mark.parametrize("frame", [0, 17, 34, 51, 85])
    @pytest.mark.parametrize("rate", [16000, 48000])
    def test_pre_padding_keeps_the_origin(self, frame, rate):
        waveform = torch.zeros(2, rate)
        waveform[:, 0] = 1.0  # the guide's first sample
        padded = pad_guide_audio(waveform, frame, rate)
        start = frame * 40 // 24
        onset = int(torch.nonzero(padded[0])[0])
        # The first sample sits at the guide frame's own time, measured from
        # the audio latent the span starts at
        assert start * (rate // 40) + onset == round(frame * rate / 24)

    def test_no_padding_on_the_hop(self):
        waveform = torch.ones(2, 100)
        assert pad_guide_audio(waveform, 0, 16000) is waveform

    def test_positions_are_the_target_audio_rows(self):
        stock = stock_layout()
        position_ids, _, _, audio_indices = stock[:4]
        width_grid = before_denoise._frame_position_grid(LAT_H, LAT_W, 2, 2)[1]
        start, count = 2, 3
        positions = guide_audio_positions(TEXT, width_grid, start, count, CHANNELS)
        assert positions.shape == (count * CHANNELS, 3)
        target = position_ids[audio_indices].reshape(CHANNELS, AUDIO_LATENTS, 3)
        assert torch.equal(
            positions.reshape(CHANNELS, count, 3),
            target[:, start : start + count].to(positions.dtype),
        )

    def test_splice_leads_the_audio(self):
        stock = stock_layout()
        position_ids, token_tags, video_indices, audio_indices, text_indices, _ = stock
        width_grid = before_denoise._frame_position_grid(LAT_H, LAT_W, 2, 2)[1]
        count = 3
        positions = guide_audio_positions(TEXT, width_grid, 0, count, CHANNELS)
        spliced = splice_guide_audio_rows(stock[:5], positions, AUDIO_TAG)
        new_ids, new_tags, new_video, new_audio, new_text = spliced
        rows = count * CHANNELS
        assert new_ids.shape[0] == position_ids.shape[0] + rows
        assert new_audio.numel() == audio_indices.numel() + rows
        assert torch.equal(new_audio[rows:], audio_indices + rows)
        assert (new_tags[new_audio] == AUDIO_TAG).all()
        assert (new_tags[new_video] == VIDEO_TAG).all()
        assert torch.equal(new_ids[new_audio[:rows]], positions.to(new_ids.dtype))
        assert torch.equal(new_ids[new_video], position_ids[video_indices])
        assert torch.equal(new_text, text_indices)


class TestGuideAudioWaveform:
    def test_audio_video(self):
        clip_ = AudioVideo(uint8_frames(5), np.ones((2, 100), np.float32), 48000)
        audio, rate = guide_audio_waveform(clip_)
        assert audio.shape == (2, 100) and rate == 48000

    def test_mono_gains_its_channel(self):
        clip_ = AudioVideo(uint8_frames(5), np.ones(100, np.float32), 16000)
        assert guide_audio_waveform([clip_])[0].shape == (1, 100)

    @pytest.mark.parametrize(
        "value",
        [
            AudioVideo(uint8_frames(5), None, None),
            AudioVideo(uint8_frames(5), np.zeros((2, 0), np.float32), 16000),
            uint8_frames(5),
        ],
    )
    def test_no_audio_is_refused(self, value):
        with pytest.raises(ValueError, match="no audio"):
            guide_audio_waveform(value)


class TestGuideAudioLayout(TestLayoutBlock):
    def make_audio_block(self, monkeypatch, seen, encoded):
        def encode_span(encoder, components, waveform, num_latents):
            encoded.append((tuple(waveform.shape), num_latents))
            return torch.full((2, num_latents, AUDIO_C), 5.0)

        monkeypatch.setattr(h3_hold, "encode_audio_span", encode_span)
        return self.make_block(monkeypatch, seen)

    def run_guides(self, monkeypatch, guides):
        seen, encoded = [], []
        block = self.make_audio_block(monkeypatch, seen, encoded)
        try:
            state = drive(block(), layout_state(guides), audio_layout_components())
        finally:
            monkeypatch.setattr(h3_guides, "_GUIDE_BLOCKS", None)
        return state, encoded

    def test_audio_guide_adds_its_audio_rows(self, monkeypatch):
        guide = {
            "video": np.zeros((39, 32, 32, 3), np.uint8),
            "frame": 0,
            "audio": torch.ones(2, 16000 * 39 // 24),
            "sample_rate": 16000,
        }
        state, encoded = self.run_guides(monkeypatch, [guide])
        stock = drive(Stock(), layout_state(), layout_components())
        assert encoded and encoded[0][1] == 65
        assert state.get("num_condition_audio_rows") == 2 * 65
        assert (
            state.get("audio_indices").numel()
            == stock.get("audio_indices").numel() + 2 * 65
        )
        rows = state.get(GUIDE_AUDIO_ROWS)
        assert rows.shape == (2 * 65, AUDIO_C)
        # The guide rows are timed as the target's first 65 audio latents
        audio = state.get("audio_indices")
        ids = state.get("position_ids")
        guide_ids = ids[audio[: 2 * 65]].reshape(2, 65, 3)
        target_ids = ids[audio[2 * 65 :]].reshape(2, -1, 3)
        assert torch.equal(guide_ids, target_ids[:, :65])

    def test_audio_false_adds_none(self, monkeypatch):
        guide = {"video": np.zeros((22, 32, 32, 3), np.uint8), "frame": 17}
        state, encoded = self.run_guides(monkeypatch, [guide])
        assert encoded == []
        assert not state.get("num_condition_audio_rows")
        assert state.get(GUIDE_AUDIO_ROWS) is None


class TestGuideAudioStep:
    def run(self, latents, indices, rows):
        state = PipelineState()
        state.set("audio_latents", latents)
        state.set("audio_indices", indices)
        if rows is not None:
            state.set(GUIDE_AUDIO_ROWS, rows)
        return drive(guide_blocks()[3](), state, components_for_steps())

    def test_prepends_the_rows(self):
        target = torch.zeros(8, AUDIO_C)
        rows = torch.ones(4, AUDIO_C)
        state = self.run(target, torch.arange(12), rows)
        latents = state.get("audio_latents")
        assert latents.shape == (12, AUDIO_C)
        assert torch.equal(latents[:4], rows) and torch.equal(latents[4:], target)
        # Once: run again on its own output it leaves it be
        again = self.run(latents, torch.arange(12), rows)
        assert again.get("audio_latents").shape == (12, AUDIO_C)

    def test_without_rows_does_nothing(self):
        target = torch.zeros(8, AUDIO_C)
        state = self.run(target, torch.arange(8), None)
        assert state.get("audio_latents") is target


# 9. Pipeline._with_guides


def ns(pipeline):
    return SimpleNamespace(name="video", pipeline=pipeline, base_dir=None)


@pytest.fixture(scope="module")
def guided():
    pipeline = fresh("t2va")
    assert insert_guides(pipeline)
    return pipeline


@pytest.fixture
def warnings_seen(monkeypatch):
    seen = []
    monkeypatch.setattr(
        pipeline_module, "emit_warning", lambda message, **data: seen.append(message)
    )
    return seen


def clip(n=5):
    return uint8_frames(n, 4, 4)


class TestWithGuides:
    def test_none_passes_through(self):
        arguments = {"prompt": "x"}
        assert Pipeline._with_guides(ns(object()), arguments) is arguments

    def test_empty_list_drops_the_key(self):
        result = Pipeline._with_guides(ns(object()), {"prompt": "x", "guides": []})
        assert result == {"prompt": "x"}

    def test_a_pipeline_that_cannot_take_guides(self):
        with pytest.raises(ValueError, match="guides"):
            Pipeline._with_guides(
                ns(fresh("ref2va")), {"guides": [{"video": clip(), "frame": 0}]}
            )
        with pytest.raises(ValueError, match="guides"):
            Pipeline._with_guides(
                ns(object()), {"guides": [{"video": clip(), "frame": 0}]}
            )

    def test_references_are_refused(self, guided):
        with pytest.raises(ValueError, match="references"):
            Pipeline._with_guides(
                ns(guided),
                {"guides": [{"video": clip(), "frame": 0}], "references": [object()]},
            )

    def test_too_many(self, guided):
        many = [{"video": clip(), "frame": 0}] * (GUIDE_LIMIT + 1)
        with pytest.raises(ValueError, match="at most 4"):
            Pipeline._with_guides(ns(guided), {"guides": many})

    @pytest.mark.parametrize(
        "guide",
        [
            {"video": clip(), "frame": 0, "audio": "x.wav"},
            {"video": clip(), "frame": 0, "extra": 1},
            {"video": clip()},
            {"frame": 0},
            "clip.mp4",
        ],
    )
    def test_bad_entries(self, guided, guide):
        with pytest.raises(ValueError, match="guides"):
            Pipeline._with_guides(ns(guided), {"guides": [guide]})

    @pytest.mark.parametrize("frame", [5, -17, True, 1.5])
    def test_bad_frames(self, guided, frame):
        with pytest.raises(ValueError, match="frame"):
            Pipeline._with_guides(
                ns(guided), {"guides": [{"video": clip(), "frame": frame}]}
            )

    def test_not_a_list(self, guided):
        with pytest.raises(ValueError, match="list"):
            Pipeline._with_guides(ns(guided), {"guides": {"video": clip(), "frame": 0}})

    def test_aligned_clips_pass_without_a_warning(self, guided, warnings_seen):
        result = Pipeline._with_guides(
            ns(guided),
            {
                "guides": [
                    {"video": clip(5), "frame": 0},
                    {"video": clip(22), "frame": 17},
                ]
            },
        )
        assert [g["video"].shape[0] for g in result["guides"]] == [5, 22]
        assert [g["frame"] for g in result["guides"]] == [0, 17]
        assert warnings_seen == []

    def test_a_30_frame_clip_is_cut_to_22(self, guided, warnings_seen):
        frames = clip(30)
        result = Pipeline._with_guides(
            ns(guided), {"guides": [{"video": frames, "frame": 0}]}
        )
        (cut,) = result["guides"]
        assert cut["video"].shape[0] == 22
        assert np.array_equal(cut["video"], frames[:22])
        assert len(warnings_seen) == 1 and "22" in warnings_seen[0]

    def test_the_arguments_are_not_mutated(self, guided):
        arguments = {"guides": [{"video": clip(3), "frame": 0}]}
        Pipeline._with_guides(ns(guided), arguments)
        assert arguments["guides"][0]["video"].shape[0] == 3


class TestWithGuideAudio:
    def test_audio_true_carries_the_soundtrack(self, guided):
        video = AudioVideo(clip(22), np.ones((2, 1000), np.float32), 48000)
        result = Pipeline._with_guides(
            ns(guided), {"guides": [{"video": video, "frame": 17, "audio": True}]}
        )
        (guide,) = result["guides"]
        assert guide["video"].shape[0] == 22
        assert guide["audio"].shape == (2, 1000)
        assert guide["sample_rate"] == 48000

    @pytest.mark.parametrize("extra", [{}, {"audio": False}])
    def test_audio_false_or_absent_carries_none(self, guided, extra):
        video = AudioVideo(clip(5), np.ones((2, 1000), np.float32), 48000)
        result = Pipeline._with_guides(
            ns(guided), {"guides": [{"video": video, "frame": 0, **extra}]}
        )
        assert result["guides"][0]["audio"] is None

    @pytest.mark.parametrize(
        "video", [AudioVideo(clip(5), None, None), clip(5)], ids=["silent", "frames"]
    )
    def test_audio_true_without_audio_is_refused(self, guided, video):
        with pytest.raises(ValueError, match="no audio"):
            Pipeline._with_guides(
                ns(guided), {"guides": [{"video": video, "frame": 0, "audio": True}]}
            )

    def test_a_path_is_read_with_its_soundtrack(self, guided, monkeypatch):
        read = []

        def load(location, base_dir=None):
            read.append(location)
            return AudioVideo(clip(5), np.ones((2, 10), np.float32), 16000)

        monkeypatch.setattr(pipeline_module, "load_audio_video", load)
        result = Pipeline._with_guides(
            ns(guided), {"guides": [{"video": "clip.mp4", "frame": 0, "audio": True}]}
        )
        assert read == ["clip.mp4"]
        assert result["guides"][0]["sample_rate"] == 16000


# ---------------------------------------------------------------------------
# 10. A file guide's soundtrack survives argument realization (#649 bounce)
# ---------------------------------------------------------------------------


def _write_guide(path, num_frames=5, fps=24, sample_rate=16000, audio=True):
    from diffusers.utils.export_utils import encode_video

    frames = [Image.new("RGB", (16, 16), (i * 40, 0, 0)) for i in range(num_frames)]
    track = torch.full((2, int(num_frames / fps * sample_rate)), 0.25)
    encode_video(
        frames,
        fps=fps,
        output_path=str(path),
        audio=track if audio else None,
        audio_sample_rate=sample_rate if audio else None,
    )
    return path.name


class TestFileGuideRealization:
    """A guide naming a file - what `asset:`/`output:` resolve to - is loaded by
    the `video` key convention before `_with_guides` sees it, so the convention
    itself must keep the soundtrack when the guide holds it."""

    def realize(self, guides, base_dir):
        from dw.arguments import realize_args

        arguments = {"guides": guides}
        # The variables pass, then the steps pass (dw/workflow_run.py)
        realize_args(arguments, str(base_dir), apply_key_conventions=False)
        realize_args(arguments, str(base_dir))
        return arguments

    def test_audio_true_keeps_the_files_soundtrack(self, guided, tmp_path):
        name = _write_guide(tmp_path / "guide.mp4")
        arguments = self.realize([{"video": name, "frame": 0, "audio": True}], tmp_path)
        assert isinstance(arguments["guides"][0]["video"], AudioVideo)

        result = Pipeline._with_guides(ns(guided), arguments)
        guide = result["guides"][0]
        assert guide["video"].shape[0] == 5
        assert guide["sample_rate"] == 16000
        assert guide["audio"].shape[0] == 2 and guide["audio"].numel()

    def test_audio_absent_still_loads_frames_only(self, guided, tmp_path):
        name = _write_guide(tmp_path / "guide.mp4")
        arguments = self.realize([{"video": name, "frame": 0}], tmp_path)
        assert not isinstance(arguments["guides"][0]["video"], AudioVideo)

        guide = Pipeline._with_guides(ns(guided), arguments)["guides"][0]
        assert guide["audio"] is None and guide["video"].shape[0] == 5

    def test_audio_true_on_a_silent_file_is_refused(self, guided, tmp_path):
        name = _write_guide(tmp_path / "silent.mp4", audio=False)
        arguments = self.realize([{"video": name, "frame": 0, "audio": True}], tmp_path)
        with pytest.raises(ValueError, match="no audio"):
            Pipeline._with_guides(ns(guided), arguments)

    def test_a_deferred_reference_is_left_for_later(self, tmp_path):
        arguments = self.realize(
            [{"video": "previous_result:take", "frame": 0, "audio": True}], tmp_path
        )
        assert arguments["guides"][0]["video"] == "previous_result:take"
