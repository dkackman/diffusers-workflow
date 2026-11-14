"""The experimental latent keystone (Task 9, `keystone_slot: "latent"`): segment
1's frame-0 latent written over every later segment's guide latent 0 for the
whole denoise, with segment 1's frame 0 as the keyframe so the two condition
rows at rotary time 0 agree, and the guide's own latent 0 written back after the
loop, before decode.

The rule (chain_rules), the chain that hands segment 1's latent on (chain), the
guide key (h3_guides `prepared_guide`), the layout that writes it and the block
that restores it (h3_guide_steps), and the insert behind the flag."""

import numpy as np
import pytest
import torch

pytest.importorskip("diffusers.modular_pipelines.minimax_h3")

from diffusers.modular_pipelines.minimax_h3 import before_denoise  # noqa: E402

from dw.guides import anchor_errors  # noqa: E402
from dw.pipeline_processors import h3_guides, h3_rules  # noqa: E402
from dw.pipeline_processors.chain import ChainConfig, run_chain  # noqa: E402
from dw.pipeline_processors.chain_rules import (  # noqa: E402
    CHAIN_KEYSTONE_SLOT,
    CHAIN_LATENT_CARRY,
    chain_keystone_slot_problem,
)
from dw.pipeline_processors.h3_guide_steps import (  # noqa: E402
    DwH3KeystoneRestoreStep,
)
from dw.pipeline_processors.h3_rules import (  # noqa: E402
    KEYSTONE_LATENTS,
    KEYSTONE_RESTORE_BLOCK,
    KEYSTONE_RESTORE_ROWS,
)
from tests import test_h3_guides, test_h3_latent_carry  # noqa: E402
from tests.test_chain import FakePipeline, solid_frame  # noqa: E402
from tests.test_h3_guides import (  # noqa: E402
    PATCH,
    SIDE,
    components_for_steps,
    drive,
    fresh,
    guide_blocks,
    layout_components,
    layout_state,
)
from tests.test_h3_latent_carry import (  # noqa: E402
    LATENT_SIDE,
    LatentOutputs,
    h3_workflow,
    latent_frames,
)

FLAG = h3_rules.REFINE_WINDOWS_ENV_VAR
LayoutBlock = test_h3_guides.TestLayoutBlock
KEYFRAME = (1, 2, 3)


@pytest.fixture
def flag_on(monkeypatch):
    monkeypatch.setenv(FLAG, "1")


@pytest.fixture
def flag_off(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)


def keystone_chain(**overrides):
    return {
        "segments": 3,
        "continuity": "guide",
        "keystone": "segment_1",
        CHAIN_KEYSTONE_SLOT: "latent",
    } | overrides


# 1. The rule


def test_latent_keystone_needs_keystone_segment_1(flag_on):
    chain = keystone_chain(keystone="none")
    problem = chain_keystone_slot_problem(chain)
    assert "keystone 'segment_1'" in problem
    errors = anchor_errors(h3_workflow(chain))
    assert [e["path"] for e in errors] == [
        f"steps[0].pipeline.chain.{CHAIN_KEYSTONE_SLOT}"
    ]
    shape = test_h3_latent_carry.GuideChainShape()
    with pytest.raises(ValueError, match="keystone 'segment_1'"):
        ChainConfig(shape.chain(keystone="none", **{CHAIN_KEYSTONE_SLOT: "latent"}), {})


def test_latent_keystone_needs_the_flag(flag_off):
    problem = chain_keystone_slot_problem(keystone_chain())
    assert FLAG in problem and "latent" in problem


def test_latent_keystone_needs_guide_continuity(flag_on):
    # keystone 'segment_1' refuses last_frame first; an unresolved keystone
    # leaves the slot's own continuity rule to say it
    chain = keystone_chain(keystone="variable:k", continuity="last_frame")
    assert "continuity 'guide'" in chain_keystone_slot_problem(chain)


def test_latent_keystone_passes_and_unresolved_is_left_to_the_run(flag_on):
    assert chain_keystone_slot_problem(keystone_chain()) is None
    assert chain_keystone_slot_problem(keystone_chain(continuity="variable:c")) is None
    assert anchor_errors(h3_workflow(keystone_chain())) == []


# 2. The chain hands segment 1's frame-0 latent on


def render(**chain):
    outputs = LatentOutputs()
    pipeline = FakePipeline(outputs)
    run_chain(
        pipeline,
        outputs.shape.chain(keystone="segment_1", **{CHAIN_KEYSTONE_SLOT: "latent"})
        | chain,
        outputs.shape.arguments(
            image=solid_frame(KEYFRAME), output=["videos", "audio", "sampling_rate"]
        ),
    )
    return pipeline, outputs


def test_segment_1_alone_asks_for_its_latents_without_the_carry(flag_on):
    pipeline, outputs = render()
    assert "latents" in pipeline.calls[0]["output"]
    for call in pipeline.calls[1:]:
        assert "latents" not in call["output"]
        (guide,) = call["guides"]
        # A pixel guide, with segment 1's frame-0 latent over its latent 0
        assert "latents" not in guide
        assert torch.equal(
            guide[KEYSTONE_LATENTS], outputs.kept[0]["latents"][:, :, :1]
        )
        # The keyframe is segment 1's rendered frame 0 (the fake colours it
        # (0, 0, 7)), agreeing with that latent
        assert call["image"].getpixel((0, 0)) == (0, 0, 7)


def test_the_keystone_rides_the_carried_latents_and_owns_its_storage(flag_on):
    pipeline, outputs = render(**{CHAIN_LATENT_CARRY: True})
    count = latent_frames(22)
    for index, call in enumerate(pipeline.calls[1:], start=1):
        (guide,) = call["guides"]
        assert torch.equal(
            guide["latents"], outputs.kept[index - 1]["latents"][:, :, -count:]
        )
        keystone = guide[KEYSTONE_LATENTS]
        assert torch.equal(keystone, outputs.kept[0]["latents"][:, :, :1])
        assert keystone.untyped_storage().nbytes() == keystone.numel() * 4


def test_capstone_combines_with_latent_carry_and_held_audio(flag_on):
    references = pytest.importorskip(
        "diffusers.modular_pipelines.minimax_h3.references"
    )
    track = torch.zeros(2, (124 + 2 * 102) * 100)
    held = references.MiniMaxH3AudioReference(audio=track, sample_rate=2400)
    outputs = LatentOutputs()
    pipeline = FakePipeline(outputs)
    chain = outputs.shape.chain(
        segments=None,
        match_audio=True,
        keystone="segment_1",
        **{
            CHAIN_KEYSTONE_SLOT: "latent",
            CHAIN_LATENT_CARRY: True,
            "experimental_capstone": {"enter_below_sigma": 0.7},
        },
    )
    run_chain(
        pipeline,
        chain,
        outputs.shape.arguments(image=solid_frame(KEYFRAME), hold_audio=held),
    )
    assert len(pipeline.calls) == 3
    for call in pipeline.calls[1:]:
        carry, capstone = call["guides"]
        assert carry["latents"] is not None and "audio_latents" not in carry
        assert carry[KEYSTONE_LATENTS] is not None
        assert capstone["past_end"] is True and capstone["enter_below_sigma"] == 0.7
        assert KEYSTONE_LATENTS not in capstone


# 3. The guide key


def guide_with(keystone):
    return {
        "video": np.zeros((22, 32, 32, 3), np.uint8),
        "frame": 0,
        KEYSTONE_LATENTS: keystone,
    }


def test_prepared_guide_passes_the_keystone_through(flag_on):
    keystone = torch.zeros(1, 4, 1, 2, 2)
    prepared = h3_guides.prepared_guide("guides", 0, guide_with(keystone), None)
    assert prepared[KEYSTONE_LATENTS] is keystone


def test_prepared_guide_refuses_the_keystone_with_the_flag_off(flag_off):
    with pytest.raises(ValueError, match=FLAG):
        h3_guides.prepared_guide(
            "guides", 0, guide_with(torch.zeros(1, 4, 1, 2, 2)), None
        )


@pytest.mark.parametrize(
    "keystone", [torch.zeros(1, 4, 2, 2, 2), torch.zeros(4, 2, 2), "latent"]
)
def test_prepared_guide_refuses_a_keystone_that_is_not_one_latent(flag_on, keystone):
    with pytest.raises(ValueError, match="one latent frame"):
        h3_guides.prepared_guide("guides", 0, guide_with(keystone), None)


# 4. The layout writes it over latent 0, and the restore step writes it back


def latent_guide(frames=22, seed=7):
    generator = torch.Generator().manual_seed(seed)
    shape = (1, 4, latent_frames(frames), LATENT_SIDE, LATENT_SIDE)
    return {
        "video": np.zeros((frames, 32, 32, 3), np.uint8),
        "frame": 0,
        "audio": None,
        "sample_rate": None,
        "latents": torch.randn(shape, generator=generator),
    }


def packed(latent):
    return before_denoise.patchify_video_latents(latent, PATCH)


def test_latent_keystone_rows_written_and_restored(monkeypatch):
    guide = latent_guide()
    carry_latent_0 = guide["latents"][:, :, :1].clone()
    keystone = torch.full((1, 4, 1, LATENT_SIDE, LATENT_SIDE), 9.0)
    guide[KEYSTONE_LATENTS] = keystone
    block = LayoutBlock().make_block(monkeypatch, [])
    state = drive(block(), layout_state([guide]), layout_components())

    # The layout: latent 0 is the keystone, the rest the carry's own
    condition = state.get("condition_latents")[-1]
    assert torch.equal(condition[:, :, :1], keystone)
    assert torch.equal(condition[:, :, 1:], guide["latents"][:, :, 1:])
    # ...and the carried tensor itself is left alone
    assert torch.equal(guide["latents"][:, :, :1], carry_latent_0)

    # Packed in front of the generated rows, as the loop sees them
    _, condition_step, latents_step, _ = guide_blocks()
    rows = state.get("num_condition_video_rows")
    state.set("latents", torch.zeros(5, packed(keystone).shape[1]))
    state.set("generator", torch.Generator().manual_seed(0))
    state = drive(condition_step(), state, components_for_steps())
    state = drive(latents_step(), state, components_for_steps())
    latents = state.get("latents")
    assert latents.shape[0] == rows + 5
    ((start, own),) = state.get(KEYSTONE_RESTORE_ROWS)
    count = own.shape[0]
    assert start == 0  # no keyframe rows: the guide's rows lead
    # During the loop: row 0 of the guide is the keystone's
    assert torch.equal(latents[start : start + count], packed(keystone))

    # After the restore step: it is the carry's own latent 0 again
    state = drive(DwH3KeystoneRestoreStep(), state, components_for_steps())
    latents = state.get("latents")
    assert torch.equal(latents[start : start + count], packed(carry_latent_0))
    assert torch.equal(latents[count:rows], packed(guide["latents"])[count:])


def test_the_restore_rows_sit_after_the_keyframe_rows(monkeypatch):
    guide = latent_guide()
    guide[KEYSTONE_LATENTS] = torch.ones(1, 4, 1, LATENT_SIDE, LATENT_SIDE)
    marker = torch.zeros(1, 4, 1, LATENT_SIDE, LATENT_SIDE)
    state = layout_state([latent_guide(seed=8), guide], condition_latents=[marker])
    state.set("keyframe_anchors", ("first",))
    block = LayoutBlock().make_block(monkeypatch, [])
    state = drive(block(), state, layout_components())
    rows_per_frame = (SIDE // 16 // 2) ** 2
    ((start, own),) = state.get(KEYSTONE_RESTORE_ROWS)
    # The keyframe's one latent, then the first guide's seven
    assert start == rows_per_frame * (1 + latent_frames(22))
    assert own.shape[0] == rows_per_frame


def test_no_keystone_leaves_no_restore_rows_and_the_step_is_a_no_op(monkeypatch):
    block = LayoutBlock().make_block(monkeypatch, [])
    state = drive(block(), layout_state([latent_guide()]), layout_components())
    assert state.get(KEYSTONE_RESTORE_ROWS) is None
    latents = torch.randn(12, 16)
    state.set("latents", latents)
    state = drive(DwH3KeystoneRestoreStep(), state, components_for_steps())
    assert state.get("latents") is latents


def test_a_keystone_off_the_canvas_is_refused(monkeypatch):
    guide = latent_guide()
    guide[KEYSTONE_LATENTS] = torch.zeros(1, 4, 1, 2, 2)
    block = LayoutBlock().make_block(monkeypatch, [])
    with pytest.raises(ValueError, match=KEYSTONE_LATENTS):
        drive(block(), layout_state([guide]), layout_components())


# 5. The restore block's place in the graph


@pytest.mark.parametrize("workflow", ["t2va", "fl2va"])
def test_the_restore_block_goes_in_before_after_denoise(workflow):
    pipeline = fresh(workflow)
    h3_guides.insert_guides(pipeline)
    assert h3_guides.insert_keystone_restore(pipeline) is True
    once = test_h3_guides.names(pipeline)
    assert h3_guides.insert_keystone_restore(pipeline) is True
    assert test_h3_guides.names(pipeline) == once
    ((prefix, order),) = once
    restore = order.index(prefix + KEYSTONE_RESTORE_BLOCK)
    assert order[restore + 1] == prefix + "after_denoise"
    assert order.index(prefix + "denoise") < restore


def test_ref2va_gets_no_restore_block():
    pipeline = fresh("ref2va")
    assert h3_guides.insert_keystone_restore(pipeline) is False
