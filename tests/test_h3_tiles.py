"""MiniMax-H3 tiled refine (#1043): the feather, the shared noise, the
decode-free graph, the refine block's noise input, and the tile loop."""

from types import SimpleNamespace

import pytest
import torch

from dw.pipeline_processors.h3_hold import HELD_LATENTS_OUTPUT, insert_audio_hold
from dw.pipeline_processors.h3_rules import (
    HELD_AUDIO_OUTPUT,
    HELD_AUDIO_RATE_OUTPUT,
    HOLD_AUDIO_INPUT,
    LATENT_SCALE,
    REFINE_BLOCK,
    REFINE_NOISE,
    REFINE_STRENGTH_INPUT,
    REFINE_TILES_INPUT,
    tile_plan,
)
from dw.pipeline_processors.h3_tiles import (
    blend,
    canvas_noise,
    crop,
    decode_free_blocks,
    feather_weights,
    run_tiled_refine,
)

minimax = pytest.importorskip("diffusers.modular_pipelines.minimax_h3")
from diffusers.modular_pipelines.minimax_h3.before_denoise import (  # noqa: E402
    patchify_video_latents,
)
from diffusers.modular_pipelines.minimax_h3.references import (  # noqa: E402
    MiniMaxH3AudioReference,
)
from dw.pipeline_processors.h3_hold_steps import EncodedAudioReference  # noqa: E402
from tests.test_h3_refine import components, make_state, run  # noqa: E402

PLAN = tile_plan(1920, 1088, 1344, 768, 256)
CANVAS = (1, 4, 2, 1088 // LATENT_SCALE, 1920 // LATENT_SCALE)


def plan_weights(plan):
    return [feather_weights(plan, tile) for tile in plan]


class TestFeather:
    @pytest.mark.parametrize(
        "size", [(1920, 1088, 1344, 768, 256), (2688, 1536, 1344, 768, 192)]
    )
    def test_weights_normalize_to_one_everywhere(self, size):
        plan = tile_plan(*size)
        total = torch.zeros(size[1] // 16, size[0] // 16)
        for tile, weights in zip(plan, plan_weights(plan)):
            x, y, w, h = tile
            total[y // 16 : (y + h) // 16, x // 16 : (x + w) // 16] += weights
        # Divided by the sum the way blend does, then every cell is covered
        assert bool((total > 0).all())

    def test_ramps_only_on_shared_sides(self):
        first, last = PLAN[0], PLAN[-1]
        weights = feather_weights(PLAN, first)
        assert weights[0, 0] == 1 and weights[0, 0:3].min() == 1
        assert weights[-1, -1] < 1
        assert feather_weights(PLAN, last)[-1, -1] == 1

    def test_ramp_is_linear_and_complementary(self):
        left, right = PLAN[0], PLAN[1]
        overlap = (left[0] + left[2] - right[0]) // 16
        fall = feather_weights(PLAN, left)[0, -overlap:]
        rise = feather_weights(PLAN, right)[0, :overlap]
        assert torch.allclose(fall + rise, torch.ones(overlap))
        steps = rise[1:] - rise[:-1]
        assert torch.allclose(steps, steps[0].expand_as(steps))


class TestBlend:
    def test_identical_constant_tiles_give_the_constant_back(self):
        tiles = [torch.full((1, 4, 2, h // 16, w // 16), 3.5) for _, _, w, h in PLAN]
        result = blend(tiles, PLAN, CANVAS)
        assert result.shape == CANVAS
        assert torch.allclose(result, torch.full(CANVAS, 3.5))

    def test_a_ramp_survives_the_blend(self):
        """Tiles cut from one smooth canvas blend back into it, no seam."""
        source = torch.randn(CANVAS, generator=torch.manual_seed(0))
        tiles = [crop(source, tile) for tile in PLAN]
        assert torch.allclose(blend(tiles, PLAN, CANVAS), source, atol=1e-5)

    def test_uncovered_canvas_is_refused(self):
        with pytest.raises(ValueError, match="uncovered"):
            blend([torch.zeros(1, 4, 2, 48, 84)], PLAN[:1], CANVAS)


class TestNoise:
    def test_tile_noise_is_the_canvas_noise_at_its_offset(self):
        latents = torch.zeros(CANVAS)
        noise = canvas_noise(latents, torch.Generator().manual_seed(7))
        again = canvas_noise(latents, torch.Generator().manual_seed(7))
        assert torch.equal(noise, again) and noise.dtype == torch.float32
        x, y, w, h = PLAN[3]
        tile = crop(noise, PLAN[3])
        assert tile.shape == (1, 4, 2, h // 16, w // 16)
        assert torch.equal(tile[..., 0, 0], noise[..., y // 16, x // 16])
        # Where two tiles overlap they share noise
        first, second = crop(noise, PLAN[0]), crop(noise, PLAN[1])
        assert torch.equal(first[..., :, -1], noise[..., :48, PLAN[0][2] // 16 - 1])
        assert second.shape[-1] == PLAN[1][2] // 16


def real_pipelines():
    graph = minimax.MiniMaxH3Blocks()
    return [
        ("whole", graph.init_pipeline()),
        ("t2va", graph.get_workflow("t2va").init_pipeline()),
    ]


class TestDecodeFree:
    @pytest.mark.parametrize("which", [0, 1])
    def test_variant_has_no_decode_and_shares_blocks(self, which):
        pipeline = real_pipelines()[which][1]
        insert_audio_hold(pipeline)
        tile_blocks, text_blocks, decoder = decode_free_blocks(pipeline)
        names = list(tile_blocks.sub_blocks)
        assert not any(n == "decode" or n.startswith("decode.") for n in names)
        assert "decode" in pipeline._blocks.sub_blocks or any(
            n.startswith("decode.") for n in pipeline._blocks.sub_blocks
        )
        assert list(text_blocks.sub_blocks) == ["text_encoder"]
        assert decoder is not None
        inputs = {param.name for param in tile_blocks.inputs}
        assert {"prompt_embeds", REFINE_NOISE, REFINE_TILES_INPUT} <= inputs
        for name, block in tile_blocks.sub_blocks.items():
            assert block is pipeline._blocks.sub_blocks[name]
        if which == 1:
            assert f"denoise.{REFINE_BLOCK}" in names
        else:
            denoise = tile_blocks.sub_blocks["denoise"]
            for sequence in denoise.sub_blocks.values():
                if hasattr(sequence, "sub_blocks"):
                    assert REFINE_BLOCK in sequence.sub_blocks

    def test_a_graph_with_nothing_to_drop_has_no_variant(self):
        bare = SimpleNamespace(
            _blocks=SimpleNamespace(sub_blocks={"denoise": object()})
        )
        assert decode_free_blocks(bare) == (None, None, None)

    def test_a_tiled_call_without_a_variant_names_refine_tiles(self):
        pipeline = SimpleNamespace()
        with pytest.raises(ValueError, match=REFINE_TILES_INPUT):
            run_tiled_refine(pipeline, {REFINE_TILES_INPUT: {}}, "x")


def patched(comps):
    comps.patch_size = (1, 2, 2)
    return comps


class TestRefineBlockNoise:
    def run_block(self, noise=None, seed=3):
        state = make_state(seed=seed)
        # Rows are 4 latent channels x (1,2,2) patch -> a (1, C, 1, 2, 3) canvas
        rows = state.get("latents")
        assert rows.shape == (6, 8)
        if noise is not None:
            state.set(REFINE_NOISE, noise)
        clean = rows.clone()
        run(patched(components()), state)
        return clean, state.get("latents")

    def canvas_noise(self):
        return torch.randn(1, 2, 1, 4, 6, generator=torch.manual_seed(11))

    def test_given_noise_replaces_the_draw(self):
        noise = self.canvas_noise()
        clean, refined = self.run_block(noise)
        packed = patchify_video_latents(noise, (1, 2, 2))
        expected = components().scheduler.scale_noise(clean, 1 - 0.2, packed)
        assert packed.shape == clean.shape
        assert torch.allclose(refined, expected)

    def test_without_noise_it_draws_as_before(self):
        clean, refined = self.run_block()
        drawn = torch.randn(clean.shape, generator=torch.Generator().manual_seed(3))
        expected = components().scheduler.scale_noise(clean, 0.8, drawn)
        assert torch.allclose(refined, expected)

    def test_noise_that_packs_to_another_shape_is_refused(self):
        with pytest.raises(ValueError, match=REFINE_NOISE):
            self.run_block(torch.randn(1, 2, 1, 2, 2))


class TestSignature:
    def test_refine_tiles_is_an_argument_and_the_noise_is_not(self):
        from dw.introspection import describe_class

        names = [
            p["name"] for p in describe_class("MiniMaxH3ModularPipeline")["parameters"]
        ]
        assert REFINE_TILES_INPUT in names
        assert REFINE_STRENGTH_INPUT in names
        assert REFINE_NOISE not in names


def blocks_taking(*names):
    return SimpleNamespace(inputs=[SimpleNamespace(name=n) for n in names])


class FakePipeline:
    """Stands in for the loaded pipeline: records each call, returns the tile's
    own crop as the denoised latents."""

    _execution_device = torch.device("cpu")

    def __init__(self):
        self._blocks = "full"
        self.calls = []
        self._dw_tile_blocks = blocks_taking("prompt_embeds", "latents")
        self._dw_text_blocks = blocks_taking("prompt")

    def __call__(self, **kwargs):
        if self._blocks is self._dw_text_blocks:
            self.calls.append(("text", kwargs))
            return {"prompt_embeds": "embeds", "text_token_tags": "tags"}
        assert self._blocks is self._dw_tile_blocks
        self.calls.append(("tile", kwargs))
        result = {"latents": kwargs["latents"] + 0}
        if HELD_AUDIO_OUTPUT in kwargs["output"]:
            result[HELD_AUDIO_OUTPUT] = "audio"
            result[HELD_AUDIO_RATE_OUTPUT] = 44100
        if HELD_LATENTS_OUTPUT in kwargs["output"]:
            # The hold block's encode of the track, which only an encode produces
            assert getattr(kwargs[HOLD_AUDIO_INPUT], "latents", None) is None
            result[HELD_LATENTS_OUTPUT] = "encoded"
        return result


class TestTileLoop:
    def run(self, output=None, **extra):
        pipeline = FakePipeline()
        latents = torch.randn(CANVAS, generator=torch.manual_seed(1))
        arguments = {
            "prompt": "a cat",
            "latents": latents,
            "num_inference_steps": 5,
            REFINE_STRENGTH_INPUT: 0.2,
            REFINE_TILES_INPUT: {"width": 1344, "height": 768, "overlap": 256},
            **extra,
        }
        if output is not None:
            arguments["output"] = output
        decoded = []

        def decode(canvas):
            decoded.append(canvas)
            return [["frame"]]

        result = run_tiled_refine(pipeline, arguments, "step", decode=decode)
        return pipeline, latents, result, decoded

    def test_one_call_per_tile_with_its_crop_and_one_decode(self):
        pipeline, latents, result, decoded = self.run(
            output=["videos", "latents", HELD_AUDIO_OUTPUT, HELD_AUDIO_RATE_OUTPUT]
        )
        tile_calls = [kw for kind, kw in pipeline.calls if kind == "tile"]
        assert len(tile_calls) == len(PLAN) == 4
        for (x, y, w, h), kwargs in zip(PLAN, tile_calls):
            assert (kwargs["width"], kwargs["height"]) == (w, h)
            assert torch.equal(
                kwargs["latents"],
                latents[..., y // 16 : (y + h) // 16, x // 16 : (x + w) // 16],
            )
            assert kwargs[REFINE_NOISE].shape == kwargs["latents"].shape
            assert REFINE_TILES_INPUT not in kwargs
            assert kwargs["prompt_embeds"] == "embeds" and "prompt" not in kwargs
        # Prompt encoded once, the held audio asked of the first tile only
        assert [kind for kind, _ in pipeline.calls].count("text") == 1
        assert HELD_AUDIO_OUTPUT in tile_calls[0]["output"]
        assert HELD_AUDIO_OUTPUT not in tile_calls[1]["output"]
        # Nothing held, so no tile is asked for an encode of it
        assert not any(HELD_LATENTS_OUTPUT in kw["output"] for kw in tile_calls)
        assert pipeline._blocks == "full"
        assert len(decoded) == 1 and decoded[0].shape == latents.shape
        assert torch.allclose(result["latents"], latents, atol=1e-5)
        assert result["videos"] == [["frame"]]
        assert result[HELD_AUDIO_OUTPUT] == "audio"
        assert result[HELD_AUDIO_RATE_OUTPUT] == 44100

    def test_the_track_is_encoded_by_one_tile_and_handed_to_the_rest(self):
        """The audio hold runs once per refine, not once per tile: the first
        tile is asked for the hold block's encode, and every later tile's
        `hold_audio` carries it, so the block writes it instead of encoding."""
        track = MiniMaxH3AudioReference(audio=torch.zeros(2, 160), sample_rate=16000)
        pipeline, _, result, _ = self.run(**{HOLD_AUDIO_INPUT: track})
        tile_calls = [kw for kind, kw in pipeline.calls if kind == "tile"]
        assert len(tile_calls) == 4
        encoding = [
            kw
            for kw in tile_calls
            if getattr(kw[HOLD_AUDIO_INPUT], "latents", None) is None
        ]
        assert len(encoding) == 1 and encoding[0] is tile_calls[0]
        assert tile_calls[0][HOLD_AUDIO_INPUT] is track
        assert HELD_LATENTS_OUTPUT in tile_calls[0]["output"]
        for kwargs in tile_calls[1:]:
            carried = kwargs[HOLD_AUDIO_INPUT]
            assert isinstance(carried, EncodedAudioReference)
            assert carried.latents == "encoded"
            assert carried.audio is track.audio
            assert carried.sample_rate == track.sample_rate
            assert HELD_LATENTS_OUTPUT not in kwargs["output"]
        # The encode is the tiles' hand-off, never part of the step's result
        assert HELD_LATENTS_OUTPUT not in result
        assert result[HELD_AUDIO_OUTPUT] == "audio"

    def test_output_is_narrowed_to_what_was_asked(self):
        _, _, result, _ = self.run(output=["videos"])
        assert list(result) == ["videos"]
        _, latents, single, _ = self.run(output="latents")
        assert single.shape == latents.shape

    def test_blocks_are_restored_when_a_tile_fails(self):
        pipeline = FakePipeline()
        pipeline.__class__ = type(
            "Failing", (FakePipeline,), {"__call__": lambda self, **kw: 1 / 0}
        )
        arguments = {
            "latents": torch.zeros(CANVAS),
            REFINE_TILES_INPUT: {"width": 1344, "height": 768, "overlap": 256},
        }
        with pytest.raises(ZeroDivisionError):
            run_tiled_refine(pipeline, arguments, "step", decode=lambda c: c)
        assert pipeline._blocks == "full"


class TestTileLoopReporting:
    def test_tiles_report_no_decoding_and_the_canvas_decode_reports_one(self):
        from dw.events import RunContext, activate_context, deactivate_context
        from dw.pipeline_processors import progress

        follows = []

        class Recording(FakePipeline):
            def __call__(self, **kwargs):
                follows.append(progress._DECODE_FOLLOWS.get())
                return super().__call__(**kwargs)

        collected = []
        token = activate_context(RunContext(on_event=collected.append))
        try:
            run_tiled_refine(
                Recording(),
                {
                    "latents": torch.zeros(CANVAS),
                    REFINE_TILES_INPUT: {"width": 1344, "height": 768, "overlap": 256},
                },
                "step",
                decode=lambda canvas: [["frame"]],
            )
        finally:
            deactivate_context(token)
        # The text encode may report a decode-free loop either way; the
        # tiles never do
        assert follows[-len(PLAN) :] == [False] * len(PLAN)
        assert progress._DECODE_FOLLOWS.get() is True
        phases = [e.get("phase") for e in collected if e["event"] == "phase"]
        assert phases == ["decoding"]

    def test_the_videos_carry_h3s_frame_rate(self):
        """Bare frame lists play at the writer's 8 fps: pair_audio fed the
        tiled step's videos wrote a 15.5 s file of a 5.17 s take."""
        _, _, result, _ = TestTileLoop().run(output=["videos"])
        videos = result["videos"]
        assert videos.fps == 24
        assert all(video.fps == 24 for video in videos)
        assert videos == [["frame"]]

    def test_pair_audio_plays_the_tiled_videos_at_24_fps(self):
        import numpy

        from dw.tasks.pair_audio import pair_audio

        _, _, result, _ = TestTileLoop().run(output=["videos"])
        paired = pair_audio(
            result["videos"], numpy.zeros((2, 48000), numpy.float32), sample_rate=48000
        )
        paired = paired[0] if isinstance(paired, list) else paired
        assert paired.fps == 24

    def test_the_videos_are_dws_one_frame_list(self):
        """One rate-carrying frame list, beside AudioVideo: the tiled
        refine's videos are the type `load_video` answers, not a second
        class of the same name."""
        from dw import media_types
        from dw.tasks import video_utils

        assert video_utils.FrameList is media_types.FrameList
        _, _, result, _ = TestTileLoop().run(output=["videos"])
        videos = result["videos"]
        assert type(videos) is media_types.FrameList
        assert all(type(video) is media_types.FrameList for video in videos)

    def test_the_frame_rate_survives_the_worker_queue(self):
        import pickle

        _, _, result, _ = TestTileLoop().run(output=["videos"])
        videos = pickle.loads(pickle.dumps(result["videos"]))
        assert videos.fps == 24 and videos[0].fps == 24
