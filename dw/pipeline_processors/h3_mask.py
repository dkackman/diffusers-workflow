"""MiniMax-H3 regional edit: re-denoising a `source_video` inside a mask while
everything outside it stays the source (#1062).

The refine schedule (dw/pipeline_processors/h3_hold.py) re-denoises a whole
canvas from `refine_strength`. A regional edit is that refine with most rows
held: a `mask_video` or `mask_image` - white regenerates - is binarized at
`MASK_THRESHOLD`, max-pooled onto the transformer's patch-row grid (a cell any
white pixel touches regenerates) and dilated by `mask_grow` cells. Every other
generated row is a kept row, written back after each denoise step to the source
noised to that step's sigma, with the noise the refine block re-noised the
canvas with. At the last step the sigma is 0 and a kept row is the source
latent exactly; the decode still round-trips it through the VAE.

Two blocks go into each of H3's core-denoise sequences, always, as no-ops when
`source_video` is not passed:

- `DwH3SourceLatentsStep`, before `set_timesteps` (after the hold block) - VAE-
  encodes `source_video` over the generated video rows of `latents`, and keeps
  the clean rows (`SOURCE_ROWS`) and which ones the mask keeps (`KEEP_ROWS`);
- `DwH3MaskBlendStep`, a sub-block of the `denoise` loop after `update` -
  writes the kept rows back after every step. Conditioning rows are never
  touched.

The refine block (unchanged otherwise) hands its re-noise on as
`REFINE_ROW_NOISE`. This module puts the blocks in; the mask-to-rows
arithmetic is dw/pipeline_processors/h3_mask_cells.py (re-exported here), and
the blocks are dw/pipeline_processors/h3_mask_steps.py, imported in `blocks()`
only, since it imports diffusers when it loads (#790).
The argument rules shared with validation are
dw/pipeline_processors/h3_rules.py (`mask_problems`, `media_problems`).

The audio pass (#1192) is the same `source_video` with no mask and no
`hold_audio` (`h3_rules.is_audio_pass`): the picture is finished and only the
audio is generated, from the prompt and its `<d>` lines. The source block
encodes the source over the generated rows as for an edit, and two more blocks
go into each sequence beside it:

- `DwH3AudioPassStep`, before `denoise` - widens `num_condition_video_rows`
  over every video row, so `update` steps none of them, and rebuilds the row
  plan with every video row pinned at t = 1.0: the exact clean source latent at
  every step, as a reference row is held. The audio rows keep the stock audio
  schedule from full noise (strength 1 for audio), so they attend to the real
  picture from the first step - which a re-noised video row, the regional
  edit's way, would hide under noise at the steps where the audio takes shape;
- `DwH3ReleaseVideoStep`, before `after_denoise` - puts the count back, so the
  decode finds the video rows where it slices them.

No refine runs: `refine_strength` is refused on an audio pass. The decode is
still a VAE round trip of the kept rows, so the result plays the source's own
frames instead (`HELD_VIDEO_OUTPUT`, swapped in by dw/output_extraction.py the
way `dw_held_audio` is). `listen_scale` 0.5 or 0.25 encodes the kept rows from
an area-resampled copy of the source on a smaller canvas (`listen_frames`,
`h3_rules.listen_canvas`), which is cheaper per step; the audio's length
follows `num_frames`, which the scale leaves alone, and the picture returned
is still the full-size source.
"""

import logging


from .h3_hold import core_denoise_sequences
from .h3_rules import (
    AUDIO_PASS_BEFORE,
    AUDIO_PASS_BLOCK,
    RELEASE_VIDEO_BEFORE,
    RELEASE_VIDEO_BLOCK,
    BLEND_AFTER,
    BLEND_BLOCK,
    REFINE_BEFORE,
    SOURCE_BEFORE,
    SOURCE_BLOCK,
)
from .h3_mask_cells import binarized, keep_rows, regenerate_cells  # noqa: F401 - re-exported

logger = logging.getLogger("dw")


def blocks():
    """(DwH3SourceLatentsStep, DwH3MaskBlendStep) - the classes in
    dw/pipeline_processors/h3_mask_steps.py, imported here on first use because
    that module imports diffusers when it loads (#790)."""
    from . import h3_mask_steps

    return h3_mask_steps.DwH3SourceLatentsStep, h3_mask_steps.DwH3MaskBlendStep


def audio_pass_blocks():
    """(DwH3AudioPassStep, DwH3ReleaseVideoStep), imported on first use as
    `blocks()` imports its pair (#790)."""
    from . import h3_mask_steps

    return h3_mask_steps.DwH3AudioPassStep, h3_mask_steps.DwH3ReleaseVideoStep


def listen_frames(frames, width, height):
    """`(frames, height, width, 3)` uint8 source frames area-resampled to
    `width` x `height` - the copy an audio pass at a `listen_scale` below 1
    encodes. torch is imported here, on first use."""
    import numpy as np
    import torch
    import torch.nn.functional as F

    pixels = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2)
    resized = F.interpolate(pixels.float(), size=(height, width), mode="area")
    rounded = resized.round().clamp(0, 255).to(torch.uint8)
    return rounded.permute(0, 2, 3, 1).contiguous().numpy()


def _insert_audio_pass(sequence, prefix, pass_block, release_block):
    """The audio-pass pair into one sequence, or neither when an anchor is
    missing. Returns whether the sequence has them."""
    sub_blocks = sequence.sub_blocks
    pass_name, release_name = prefix + AUDIO_PASS_BLOCK, prefix + RELEASE_VIDEO_BLOCK
    if pass_name in sub_blocks and release_name in sub_blocks:
        return True
    pass_before = prefix + AUDIO_PASS_BEFORE
    release_before = prefix + RELEASE_VIDEO_BEFORE
    if pass_before not in sub_blocks or release_before not in sub_blocks:
        return False
    names = list(sub_blocks)
    sub_blocks.insert(pass_name, pass_block(), names.index(pass_before))
    names = list(sub_blocks)
    sub_blocks.insert(release_name, release_block(), names.index(release_before))
    return True


def _loop(sequence, prefix):
    """The sequence's denoise loop, or None - the step `refine` anchors to."""
    loop = sequence.sub_blocks.get(prefix + REFINE_BEFORE)
    return loop if hasattr(loop, "sub_blocks") else None


def insert_mask_blocks(pipeline):
    """Put the source and blend blocks into every H3 core-denoise sequence.

    Idempotent - a block already in place is left alone. Returns whether the
    pipeline now takes a regional edit. The two go in as a pair: a sequence
    missing either anchor (a diffusers that renamed `set_timesteps`, the
    `denoise` loop or its `update`) gets neither, with a warning, and a
    pipeline left with none refuses `source_video` at the call.
    """
    source_block, blend_block = blocks()
    pass_block, release_block = audio_pass_blocks()
    inserted = False
    for prefix, sequence in core_denoise_sequences(pipeline):
        sub_blocks = sequence.sub_blocks
        source_name, source_before = prefix + SOURCE_BLOCK, prefix + SOURCE_BEFORE
        loop = _loop(sequence, prefix)
        loop_names = list(loop.sub_blocks) if loop is not None else []
        if source_name in sub_blocks and BLEND_BLOCK in loop_names:
            inserted = True
            _insert_audio_pass(sequence, prefix, pass_block, release_block)
            continue
        if source_before not in sub_blocks or BLEND_AFTER not in loop_names:
            logger.warning(
                f"MiniMax-H3 has a denoise sequence with no '{source_before}' step "
                f"or '{prefix}{REFINE_BEFORE}' loop with an '{BLEND_AFTER}' to "
                f"anchor the regional edit to, so it cannot take a source_video"
            )
            continue
        if source_name not in sub_blocks:
            names = list(sub_blocks)
            sub_blocks.insert(source_name, source_block(), names.index(source_before))
        if BLEND_BLOCK not in loop_names:
            loop.sub_blocks.insert(
                BLEND_BLOCK, blend_block(), loop_names.index(BLEND_AFTER) + 1
            )
        inserted = True
        if not _insert_audio_pass(sequence, prefix, pass_block, release_block):
            logger.warning(
                f"MiniMax-H3 has a denoise sequence with no '{prefix}"
                f"{AUDIO_PASS_BEFORE}'/'{prefix}{RELEASE_VIDEO_BEFORE}' step to "
                f"anchor the audio pass to, so it cannot score a finished picture"
            )
    return inserted


def edits_regions(pipeline):
    """Whether a pipeline's block graph carries the regional edit."""
    for prefix, sequence in core_denoise_sequences(pipeline):
        loop = _loop(sequence, prefix)
        if (
            prefix + SOURCE_BLOCK in sequence.sub_blocks
            and loop is not None
            and BLEND_BLOCK in loop.sub_blocks
        ):
            return True
    return False


def passes_audio(pipeline):
    """Whether a pipeline's block graph carries the audio pass."""
    return any(
        prefix + AUDIO_PASS_BLOCK in sequence.sub_blocks
        and prefix + RELEASE_VIDEO_BLOCK in sequence.sub_blocks
        for prefix, sequence in core_denoise_sequences(pipeline)
    )
