"""The ref2va one-guide layout (#1411): the guide's rows spliced into diffusers'
ref2va packed sequence, held at the condition timestep, the layout step that
encodes them, and the VRAM estimate that counts them."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dw.pipeline_processors import h3_guides
from dw.vram_estimate import required_gb, required_vram_gb

pytest.importorskip("diffusers.modular_pipelines.minimax_h3")
from diffusers.modular_pipelines.minimax_h3 import (  # noqa: E402
    before_denoise,
    references,
)
from diffusers.modular_pipelines.modular_pipeline import PipelineState  # noqa: E402
from dw.pipeline_processors.h3_guide_steps import (  # noqa: E402
    DwH3Ref2VAGuideLayoutStep,
    append_ref2va_guide_rows,
    ref2va_guide_positions,
)

Stock = before_denoise.MiniMaxH3Ref2VAPrepareLayoutStep

TEXT = 10
LATENT_FRAMES = 7
LAT_H, LAT_W = 4, 6
PATCH = (1, 2, 2)
AUDIO_LATENTS = 8
CHANNELS = 2
AUDIO_TAG, VIDEO_TAG = 7, 9
CHANNELS_LATENT = 4
ROWS_PER_FRAME = (LAT_H // PATCH[1]) * (LAT_W // PATCH[2])
TARGET_ROWS = LATENT_FRAMES * ROWS_PER_FRAME
REF_ROWS = 1 * ROWS_PER_FRAME


def reference_latent():
    return torch.zeros(1, CHANNELS_LATENT, 1, LAT_H, LAT_W)


def stock_layout():
    """The real `build_ref2va_packed_sequence` with one image reference:
    text | reference video rows | target audio | target video."""
    image = references.MiniMaxH3ImageReference(image=np.zeros((8, 8, 3), np.uint8))
    return Stock.build_ref2va_packed_sequence(
        torch.ones(TEXT, dtype=torch.long),
        [image],
        [reference_latent()],
        [],
        LATENT_FRAMES,
        LAT_H,
        LAT_W,
        AUDIO_LATENTS,
        PATCH,
        CHANNELS,
        AUDIO_TAG,
        VIDEO_TAG,
    )


def layout_tuple():
    *packed, _ref_video_rows, _ref_audio_rows = stock_layout()
    return (*packed, REF_ROWS)


class TestSplice:
    def test_the_fixture_has_the_stock_shape(self):
        position_ids, tags, video, audio, text, rows = layout_tuple()
        assert position_ids.dtype == torch.float64 and position_ids.shape[1] == 3
        assert rows == REF_ROWS
        assert video.numel() == REF_ROWS + TARGET_ROWS
        assert audio.numel() == AUDIO_LATENTS * CHANNELS
        assert text.numel() == TEXT

    def test_positions_are_the_target_video_rows(self):
        layout = layout_tuple()
        position_ids, _, video, _, _, rows = layout
        guide = ref2va_guide_positions(layout)
        assert guide.shape == (TARGET_ROWS, 3)
        assert torch.equal(guide, position_ids[video[rows:]])
        # a copy: editing it leaves the layout alone
        before = position_ids.clone()
        guide += 1
        assert torch.equal(position_ids, before)

    def test_rows_are_appended_at_the_end(self):
        layout = layout_tuple()
        position_ids, tags, video, audio, text, rows = layout
        end = position_ids.shape[0]
        spliced = append_ref2va_guide_rows(
            layout, ref2va_guide_positions(layout), VIDEO_TAG
        )
        new_ids, new_tags, new_video, new_audio, new_text, new_rows = spliced
        assert new_ids.shape[0] == end + TARGET_ROWS
        # the original rows are untouched, the guide rows copy the target's
        assert torch.equal(new_ids[:end], position_ids)
        assert torch.equal(new_ids[end:], position_ids[video[rows:]])
        assert new_ids.dtype == torch.float64
        assert torch.equal(new_tags[:end], tags)
        assert (new_tags[end:] == VIDEO_TAG).all()
        assert new_tags.dtype == tags.dtype
        assert torch.equal(
            new_video,
            torch.cat(
                [video[:rows], torch.arange(end, end + TARGET_ROWS), video[rows:]]
            ),
        )
        assert torch.equal(new_audio, audio)
        assert torch.equal(new_text, text)
        assert new_rows == rows + TARGET_ROWS

    def test_no_guide_rows_is_the_layout_itself(self):
        layout = layout_tuple()
        assert append_ref2va_guide_rows(layout, torch.zeros(0, 3), VIDEO_TAG) is layout


class TestConditionTimesteps:
    """`MiniMaxH3SetTimestepsStep.build_row_timesteps` pins
    `video_indices[:num_condition_video_rows]`: the guide rows sit in that
    prefix, the target rows after it."""

    def test_guide_rows_get_the_condition_timestep(self):
        layout = layout_tuple()
        spliced = append_ref2va_guide_rows(
            layout, ref2va_guide_positions(layout), VIDEO_TAG
        )
        _, _, video, audio, text, rows = spliced
        sampler, condition = 0.75, 0.0
        timesteps, inverse = (
            before_denoise.MiniMaxH3SetTimestepsStep.build_row_timesteps(
                video, audio, rows, 0, text.numel(), sampler, 0.5, condition, 0.0
            )
        )
        per_row = timesteps[inverse]
        end = layout[0].shape[0]
        guide_rows = torch.arange(end, end + TARGET_ROWS)
        reference_rows = video[:REF_ROWS]
        target_rows = video[REF_ROWS + TARGET_ROWS :]
        assert target_rows.numel() == TARGET_ROWS
        assert (per_row[guide_rows] == condition).all()
        assert (per_row[reference_rows] == condition).all()
        assert (per_row[target_rows] == sampler).all()
        # and the target rows are the stock ones, not the guide's
        assert not set(target_rows.tolist()) & set(guide_rows.tolist())


# The layout step ---------------------------------------------------------


SIDE = 64
NUM_FRAMES = 22


def fake_stock_layout(marker):
    """A stand-in for the stock ref2va layout step: writes what it would."""
    position_ids, tags, video, audio, text, rows = layout_tuple()

    def layout(components, state):
        state.set("height", SIDE)
        state.set("width", SIDE)
        state.set("num_frames", NUM_FRAMES)
        state.set("text_token_tags", torch.ones(TEXT, dtype=torch.long))
        state.set("latent_height", LAT_H)
        state.set("latent_width", LAT_W)
        state.set("num_latent_frames", LATENT_FRAMES)
        state.set("position_ids", position_ids)
        state.set("token_tags", tags)
        state.set("video_indices", video)
        state.set("audio_indices", audio)
        state.set("text_indices", text)
        state.set("num_condition_video_rows", rows)
        state.set("condition_latents", [marker])
        state.set("num_audio_latents", AUDIO_LATENTS)
        return components, state

    return layout


def step_components():
    return SimpleNamespace(
        _execution_device=torch.device("cpu"),
        video_tag=VIDEO_TAG,
        vae=None,
        pixel_mean=0,
        pixel_std=1,
        keyframe_encode_seed=0,
    )


def run_layout(monkeypatch, guides, latents=LATENT_FRAMES):
    seen = []
    marker = reference_latent()

    def encode(vae, pixels, mean, std, seed):
        seen.append(tuple(pixels.shape))
        return torch.ones(1, CHANNELS_LATENT, latents, LAT_H, LAT_W)

    monkeypatch.setattr(
        "diffusers.modular_pipelines.minimax_h3.encoders.encode_vae_condition", encode
    )
    block = DwH3Ref2VAGuideLayoutStep()
    block._layout = fake_stock_layout(marker)
    state = PipelineState()
    if guides is not None:
        state.set("guides", guides)
    _, state = block(step_components(), state)
    return state, marker, seen


class TestLayoutStep:
    def guide(self, frames=NUM_FRAMES, frame=0, **extra):
        return {
            "video": np.zeros((frames, 32, 32, 3), np.uint8),
            "frame": frame,
            **extra,
        }

    @pytest.mark.parametrize("guides", [None, []])
    def test_no_guides_is_the_stock_layout(self, monkeypatch, guides):
        state, marker, seen = run_layout(monkeypatch, guides)
        stock = layout_tuple()
        assert seen == []
        assert torch.equal(state.get("position_ids"), stock[0])
        assert state.get("num_condition_video_rows") == REF_ROWS
        assert state.get("condition_latents") == [marker]

    def test_one_guide_follows_the_references(self, monkeypatch):
        state, marker, seen = run_layout(monkeypatch, [self.guide()])
        stock = layout_tuple()
        assert seen == [(1, 3, NUM_FRAMES, SIDE, SIDE)]
        latents = state.get("condition_latents")
        assert len(latents) == 2 and latents[0] is marker
        assert latents[1].shape[2] == LATENT_FRAMES
        assert state.get("num_condition_video_rows") == REF_ROWS + TARGET_ROWS
        assert state.get("position_ids").shape[0] == stock[0].shape[0] + TARGET_ROWS
        assert state.get("video_indices").numel() == (stock[2].numel() + TARGET_ROWS)
        assert torch.equal(state.get("audio_indices"), stock[3])
        assert torch.equal(state.get("text_indices"), stock[4])

    def test_the_wrong_latent_count_is_refused(self, monkeypatch):
        with pytest.raises(ValueError, match="latent frames"):
            run_layout(monkeypatch, [self.guide()], latents=5)

    def test_refusals(self, monkeypatch):
        with pytest.raises(ValueError, match="exactly one clip"):
            run_layout(monkeypatch, [self.guide(), self.guide()])
        with pytest.raises(ValueError, match="frame 17"):
            run_layout(monkeypatch, [self.guide(frame=17)])
        with pytest.raises(ValueError, match="shorter than"):
            run_layout(monkeypatch, [self.guide(frames=5)])
        with pytest.raises(ValueError, match="past the end"):
            run_layout(monkeypatch, [self.guide(frames=39)])
        with pytest.raises(ValueError, match="'audio' is refused"):
            run_layout(monkeypatch, [self.guide(audio=np.zeros(4))])
        with pytest.raises(ValueError, match="guide chain"):
            run_layout(monkeypatch, [self.guide(latents=torch.zeros(1))])


# The VRAM estimate -----------------------------------------------------


ESTIMATE = {
    "voxel_variables": ["width", "height", "num_frames"],
    "base_gb": 1,
    "bytes_per_voxel": 0,
    "gb_per_reference": 2,
    "bytes_per_guide_voxel": 2**20,
}
VALUES = {"width": 64, "height": 32, "num_frames": 22}


def test_required_gb_counts_a_ref2va_guide():
    without = required_gb(ESTIMATE, VALUES, references=1)
    with_guide = required_gb(ESTIMATE, VALUES, references=1, guides=(22,))
    assert with_guide > without
    assert with_guide - without == pytest.approx(2**20 * 22 * 64 * 32 / 2**30)


def test_the_estimate_path_counts_a_ref2va_steps_guide(monkeypatch):
    import dw.guides

    monkeypatch.setattr(
        dw.guides,
        "resolve_probe_path",
        lambda value, base_dir, what="": "/resolved/" + value,
    )

    def definition(guides):
        arguments = {
            "width": 64,
            "height": 32,
            "num_frames": 22,
            "references": ["asset:r.png"],
        }
        if guides:
            arguments["guides"] = guides
        return {
            "cost": [{"device": "cuda", "name": "Card", "vram_gb": 24, "minutes": 1}],
            "vram_estimate": ESTIMATE,
            "steps": [
                {
                    "name": "s1",
                    "pipeline": {
                        "from_pretrained_arguments": {"workflow": "ref2va"},
                        "arguments": arguments,
                    },
                }
            ],
        }

    def probe(path):
        return {"kind": "video", "frame_count": 22}

    plain = required_vram_gb(definition(None), probe=probe)[0]
    guided = required_vram_gb(
        definition([{"video": "asset:c.mp4", "frame": 0}]), probe=probe
    )[0]
    assert guided - plain == pytest.approx(2**20 * 22 * 64 * 32 / 2**30)


def test_the_module_imports_its_block_through_the_guides_helper():
    assert h3_guides.ref2va_guide_block() is DwH3Ref2VAGuideLayoutStep
