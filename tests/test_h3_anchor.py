"""MiniMax-H3's latent statistics anchor (#1166, #1173): the per-channel
rescale, the step dw inserts after `after_denoise`, the chain's
`anchor: "segment_1"`, and the pre-run checks of both."""

import numpy as np
import pytest
import torch

from dw.guides import anchor_errors
from dw.pipeline_processors.chain import ChainConfig, run_chain
from dw.pipeline_processors.h3_anchor import (
    channel_statistics,
    insert_latent_anchor,
    match_channel_statistics,
    prepared_anchor,
)
from dw.pipeline_processors.h3_rules import (
    ANCHOR_AFTER,
    ANCHOR_BLOCK,
    LATENT_ANCHOR_INPUT,
    TILE_CONFLICTS,
    chain_anchor_problem,
)
from tests.test_chain import FakePipeline, video_output


def synthetic_latent(channels=4, dtype=torch.float32, seed=0):
    generator = torch.Generator().manual_seed(seed)
    scales = torch.arange(1, channels + 1, dtype=torch.float32).reshape(1, -1, 1, 1, 1)
    values = torch.randn((1, channels, 5, 4, 6), generator=generator) * scales + 3.0
    return values.to(dtype)


class TestMatchChannelStatistics:
    def test_each_channel_ends_at_the_anchors_statistics(self):
        latents = synthetic_latent()
        anchor_mean = torch.tensor([0.5, -1.0, 2.0, 0.0])
        anchor_std = torch.tensor([0.25, 1.5, 0.75, 2.0])

        matched = match_channel_statistics(latents, anchor_mean, anchor_std)
        mean, std = channel_statistics(matched)

        assert torch.allclose(mean.flatten(), anchor_mean, atol=1e-5)
        assert torch.allclose(std.flatten(), anchor_std, atol=1e-5)

    def test_matching_onto_an_encoded_anchors_own_statistics(self):
        anchor = synthetic_latent(seed=1)
        mean, std = channel_statistics(anchor)
        matched = match_channel_statistics(synthetic_latent(), mean, std)
        new_mean, new_std = channel_statistics(matched)
        assert torch.allclose(new_mean, mean, atol=1e-5)
        assert torch.allclose(new_std, std, atol=1e-5)

    def test_keeps_the_latents_dtype_and_shape(self):
        latents = synthetic_latent(dtype=torch.bfloat16)
        matched = match_channel_statistics(latents, torch.zeros(4), torch.ones(4))
        assert matched.dtype == torch.bfloat16
        assert matched.shape == latents.shape

    def test_a_flat_channel_lands_on_the_anchors_mean(self):
        latents = synthetic_latent()
        latents[:, 0] = 7.0
        matched = match_channel_statistics(
            latents, torch.full((4,), 2.0), torch.ones(4)
        )
        assert torch.isfinite(matched).all()
        assert torch.allclose(matched[:, 0], torch.tensor(2.0))

    def test_refuses_a_packed_latent(self):
        with pytest.raises(ValueError, match="video latent"):
            match_channel_statistics(torch.zeros(12, 3), torch.zeros(3), torch.ones(3))

    def test_refuses_a_channel_mismatch(self):
        with pytest.raises(ValueError, match="latent channels"):
            match_channel_statistics(synthetic_latent(), torch.zeros(3), torch.ones(3))


class TestPreparedAnchor:
    def test_frames_are_cut_to_a_whole_latent_length(self):
        frames = np.zeros((9, 4, 6, 3), dtype=np.uint8)
        assert prepared_anchor("anchor", frames, None).shape[0] == 5

    def test_refuses_what_is_not_frames(self):
        with pytest.raises(ValueError, match="anchor"):
            prepared_anchor("anchor", 42, None)


class TestChainAnchor:
    def test_segments_after_the_first_get_segment_ones_frames(self):
        pipeline = FakePipeline(video_output)

        run_chain(pipeline, {"segments": 3, "anchor": "segment_1"}, {"prompt": "p"})

        assert LATENT_ANCHOR_INPUT not in pipeline.calls[0]
        first = pipeline.calls[1][LATENT_ANCHOR_INPUT]
        assert pipeline.calls[2][LATENT_ANCHOR_INPUT] is first
        # Segment 1's colour (index 0), every one of its frames
        assert len(first) == 4
        assert first[1].getpixel((0, 0)) == (0, 100, 150)

    def test_no_anchor_by_default(self):
        pipeline = FakePipeline(video_output)
        run_chain(pipeline, {"segments": 3}, {})
        assert all(LATENT_ANCHOR_INPUT not in call for call in pipeline.calls)

    def test_an_explicit_none(self):
        pipeline = FakePipeline(video_output)
        run_chain(pipeline, {"segments": 2, "anchor": "none"}, {})
        assert all(LATENT_ANCHOR_INPUT not in call for call in pipeline.calls)

    def test_config_refuses_an_unknown_anchor(self):
        with pytest.raises(ValueError, match="none, segment_1"):
            ChainConfig({"segments": 2, "anchor": "segment_2"}, {})

    def test_config_refuses_another_continuity(self):
        with pytest.raises(ValueError, match="last_frame' or 'guide"):
            ChainConfig(
                {"segments": 2, "anchor": "segment_1", "continuity": "last_segment"}, {}
            )

    @pytest.mark.parametrize("continuity", ["last_frame", "guide"])
    def test_rule_accepts_the_anchor_continuities(self, continuity):
        chain = {"anchor": "segment_1", "continuity": continuity}
        assert chain_anchor_problem(chain) is None

    def test_latent_anchor_conflicts_with_tiles(self):
        assert LATENT_ANCHOR_INPUT in TILE_CONFLICTS


def definition(workflow="fl2va", chain=None, pipeline_class=None, **arguments):
    pipeline = {
        "configuration": {"component_type": "ModularPipeline"},
        "from_pretrained_arguments": {
            "model_name": "MiniMaxAI/MiniMax-H3",
            "workflow": workflow,
        },
        "arguments": arguments,
    }
    if pipeline_class:
        pipeline["configuration"] = {"component_type": pipeline_class}
        pipeline["from_pretrained_arguments"] = {"model_name": "some/model"}
    if chain is not None:
        pipeline["chain"] = chain
    return {"steps": [{"name": "video", "pipeline": pipeline}]}


class TestAnchorErrors:
    @pytest.mark.parametrize("workflow", ["t2va", "fl2va"])
    @pytest.mark.parametrize("continuity", ["last_frame", "guide"])
    def test_a_segment_1_chain_on_h3(self, workflow, continuity):
        chain = {"segments": 3, "anchor": "segment_1", "continuity": continuity}
        assert anchor_errors(definition(workflow, chain)) == []

    @pytest.mark.parametrize("workflow", ["t2va", "fl2va"])
    def test_a_latent_anchor_argument_on_h3(self, workflow):
        assert anchor_errors(definition(workflow, latent_anchor="clip.mp4")) == []

    def test_no_anchor_is_clean(self):
        assert anchor_errors(definition(chain={"segments": 2})) == []
        assert anchor_errors(definition(chain={"segments": 2, "anchor": "none"})) == []

    def test_a_variable_anchor_is_left_to_the_run(self):
        chain = {"segments": 2, "anchor": "variable:anchor"}
        assert anchor_errors(definition(chain=chain)) == []

    def test_an_unknown_anchor_names_the_values(self):
        chain = {"segments": 2, "anchor": "first"}
        (error,) = anchor_errors(definition(chain=chain))
        assert error["path"] == "steps[0].pipeline.chain.anchor"
        assert "none, segment_1" in error["message"]

    def test_segment_1_with_another_continuity(self):
        chain = {"segments": 2, "anchor": "segment_1", "continuity": "last_segment"}
        (error,) = anchor_errors(definition(chain=chain))
        assert "last_frame' or 'guide" in error["message"]

    def test_segment_1_off_h3(self):
        chain = {"segments": 2, "anchor": "segment_1"}
        (error,) = anchor_errors(definition(chain=chain, pipeline_class="WanPipeline"))
        assert error["path"] == "steps[0].pipeline.chain.anchor"

    def test_segment_1_on_ref2va(self):
        chain = {"segments": 2, "anchor": "segment_1"}
        (error,) = anchor_errors(definition("ref2va", chain=chain))
        assert "not ref2va" in error["message"]

    def test_latent_anchor_off_h3(self):
        workflow = definition(pipeline_class="WanPipeline", latent_anchor="clip.mp4")
        (error,) = anchor_errors(workflow)
        assert error["path"] == "steps[0].pipeline.arguments.latent_anchor"

    def test_latent_anchor_on_ref2va(self):
        (error,) = anchor_errors(definition("ref2va", latent_anchor="clip.mp4"))
        assert "not ref2va" in error["message"]


minimax = pytest.importorskip("diffusers.modular_pipelines.minimax_h3")
from diffusers.modular_pipelines.modular_pipeline import PipelineState  # noqa: E402
from dw.pipeline_processors import h3_hold  # noqa: E402
from dw.pipeline_processors.h3_anchor_steps import DwH3LatentAnchorStep  # noqa: E402
from dw.pipeline_processors.h3_hold import insert_audio_hold  # noqa: E402


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
    @pytest.mark.parametrize("workflow", ["t2va", "fl2va"])
    def test_follows_after_denoise(self, workflow):
        pipeline = fresh(workflow)
        assert insert_latent_anchor(pipeline) is True
        ((prefix, order),) = names(pipeline)
        after = order.index(prefix + ANCHOR_AFTER)
        assert order[after + 1] == prefix + ANCHOR_BLOCK

    def test_idempotent(self):
        pipeline = fresh()
        insert_latent_anchor(pipeline)
        once = names(pipeline)
        insert_latent_anchor(pipeline)
        assert names(pipeline) == once

    def test_ref2va_gets_no_anchor(self):
        pipeline = fresh("ref2va")
        before = names(pipeline)
        assert insert_latent_anchor(pipeline) is False
        assert names(pipeline) == before

    def test_takes_latent_anchor_as_an_input(self):
        inputs = {param.name for param in DwH3LatentAnchorStep().inputs}
        assert LATENT_ANCHOR_INPUT in inputs


class TestStep:
    def state(self, latents, anchor=None):
        state = PipelineState()
        state.set("latents", latents)
        state.set(LATENT_ANCHOR_INPUT, anchor)
        state.set("height", 32)
        state.set("width", 32)
        return state

    def test_without_an_anchor_the_latents_are_untouched(self):
        latents = synthetic_latent()
        state = self.state(latents)
        _, state = DwH3LatentAnchorStep()(None, state)
        assert state.get("latents") is latents

    def test_with_an_anchor_the_latents_take_its_statistics(self, monkeypatch):
        from dw.pipeline_processors import h3_anchor_steps

        anchor_latent = synthetic_latent(seed=3)
        monkeypatch.setattr(
            h3_anchor_steps,
            "anchor_statistics",
            lambda components, frames, height, width: channel_statistics(anchor_latent),
        )
        frames = np.zeros((5, 32, 32, 3), dtype=np.uint8)
        state = self.state(synthetic_latent(), frames)
        _, state = DwH3LatentAnchorStep()(None, state)
        mean, std = channel_statistics(state.get("latents"))
        anchor_mean, anchor_std = channel_statistics(anchor_latent)
        assert torch.allclose(mean, anchor_mean, atol=1e-5)
        assert torch.allclose(std, anchor_std, atol=1e-5)
