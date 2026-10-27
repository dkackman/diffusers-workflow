"""MiniMax-H3 tiled refine (#1043): `refine_tiles` on a `refine_strength` step.

A refine canvas can be bigger than H3 can denoise in one call - 1920x1088
against a 1344x768 limit. The step therefore refines it as overlapping tiles
(`h3_rules.tile_plan`), each a normal call of the pipeline on a crop of
`latents`, then blends the denoised tiles in latent space and decodes the
canvas once.

Four things make the seams disappear, and each is here:

- The noise is drawn once for the whole canvas from the step's own generator
  and cropped per tile (`canvas_noise`, `crop`), handed to the refine block as
  `dw_refine_noise`. Two tiles start from identical noise where they overlap,
  so what they denoise differs only by what each one can see.
- Each tile runs the graph with every decode block removed
  (`decode_free_blocks`), so what comes back is the denoised latents, never
  pixels: pixels would blend visibly, latents blend before the one decode.
  The prompt is encoded once, by the graph's `text_encoder` alone, and every
  tile is handed the embeddings.
- The blend is a linear feather across each overlap (`feather_weights`), the
  way the VAE's own `_blend` joins its tiles, accumulated and divided by the
  weight sum so it normalizes to 1 everywhere.
- The canvas decodes once, through the loaded pipeline's own video decode
  block - the same colours and VAE a plain call has, whose spatial tiling is
  on by default.

The held soundtrack is the hold block's own output, taken from the first tile:
every tile holds the same track, and none of them decodes audio.

Imports diffusers only inside the functions that need it, like the rest of the
H3 modules, so validation never loads a modular pipeline.
"""

import contextlib

import torch

from ..events import emit_log, emit_phase, get_context
from ..media_types import FrameList
from .h3_rules import (
    HELD_AUDIO_OUTPUT,
    HELD_AUDIO_RATE_OUTPUT,
    H3_FPS,
    LATENT_SCALE,
    REFINE_NOISE,
    REFINE_TILES_INPUT,
    tile_plan,
)
from .progress import decoded_later, reported_blocks

TEXT_BLOCK = "text_encoder"
TEXT_OUTPUTS = ["prompt_embeds", "text_token_tags"]
TILE_BLOCKS = "_dw_tile_blocks"
TEXT_BLOCKS = "_dw_text_blocks"
VIDEO_DECODE = "_dw_video_decode"


def _is_decode(name):
    return name == "decode" or name.startswith("decode.")


def _find_video_decode(block):
    """The MiniMaxH3VideoDecodeStep somewhere under `block`, or None."""
    from diffusers.modular_pipelines.minimax_h3.decoders import (
        MiniMaxH3VideoDecodeStep,
    )

    if isinstance(block, MiniMaxH3VideoDecodeStep):
        return block
    for child in (getattr(block, "sub_blocks", None) or {}).values():
        found = _find_video_decode(child)
        if found is not None:
            return found
    return None


def decode_free_blocks(pipeline):
    """`(tile_blocks, text_blocks, video_decode)` built from a pipeline's own
    graph, sharing its block instances so dw's hold, refine and release blocks
    stay in; (None, None, None) when there is no decode block to drop or no
    video decoder among them.

    `tile_blocks` is the graph with every top-level decode block removed, and
    the text encoder too when it has one - a tile takes `prompt_embeds`
    instead. `text_blocks` is that encoder alone, run once.
    """
    from diffusers.modular_pipelines import SequentialPipelineBlocks

    blocks = getattr(pipeline, "_blocks", None)
    sub_blocks = getattr(blocks, "sub_blocks", None)
    if not hasattr(sub_blocks, "items"):
        return None, None, None
    dropped = {name: block for name, block in sub_blocks.items() if _is_decode(name)}
    decoder = next(
        (found for found in map(_find_video_decode, dropped.values()) if found),
        None,
    )
    if decoder is None:
        return None, None, None
    text = sub_blocks.get(TEXT_BLOCK)
    kept = {
        name: block
        for name, block in sub_blocks.items()
        if name not in dropped and name != TEXT_BLOCK
    }
    tile_blocks = SequentialPipelineBlocks.from_blocks_dict(kept)
    text_blocks = (
        None
        if text is None
        else SequentialPipelineBlocks.from_blocks_dict({TEXT_BLOCK: text})
    )
    return tile_blocks, text_blocks, decoder


def _input_names(blocks):
    return {param.name for param in blocks.inputs}


def attach_tile_blocks(pipeline):
    """Build the tiled-refine variants and keep them on the pipeline (None for
    a graph that cannot run tiled)."""
    tile_blocks, text_blocks, decoder = decode_free_blocks(pipeline)
    setattr(pipeline, TILE_BLOCKS, tile_blocks)
    setattr(pipeline, TEXT_BLOCKS, text_blocks)
    setattr(pipeline, VIDEO_DECODE, decoder)


def _axis_ramps(spans, scale):
    """Per tile along one axis, `(before, after)` overlaps with its neighbours
    in latent cells - the span is `(start, length)` in pixels, sorted by start."""
    ramps = {}
    for index, (start, length) in enumerate(spans):
        before = after = 0
        if index > 0:
            earlier = spans[index - 1]
            before = max(0, earlier[0] + earlier[1] - start) // scale
        if index + 1 < len(spans):
            after = max(0, start + length - spans[index + 1][0]) // scale
        ramps[(start, length)] = (before, after)
    return ramps


def _axis_weights(length, before, after):
    """`length` cell weights: rising over `before` cells, falling over `after`,
    1 between. A ramp of n cells runs (1..n)/(n+1), so no cell is 0 and the
    rise of one tile and the fall of its neighbour sum to 1."""
    weights = torch.ones(length, dtype=torch.float32)
    if before:
        weights[:before] *= (torch.arange(before, dtype=torch.float32) + 1) / (
            before + 1
        )
    if after:
        weights[length - after :] *= torch.arange(after, 0, -1, dtype=torch.float32) / (
            after + 1
        )
    return weights


def feather_weights(plan, tile, scale=LATENT_SCALE):
    """The `(h, w)` latent-cell weights of one tile of `plan` (x, y, w, h in
    pixels): a separable ramp rising linearly over the overlap it shares with a
    neighbour on each side, 1 elsewhere and on the canvas edges."""
    x, y, width, height = tile
    columns = _axis_ramps(sorted({(px, pw) for px, _, pw, _ in plan}), scale)
    rows = _axis_ramps(sorted({(py, ph) for _, py, _, ph in plan}), scale)
    left, right = columns[(x, width)]
    top, bottom = rows[(y, height)]
    return (
        _axis_weights(height // scale, top, bottom)[:, None]
        * _axis_weights(width // scale, left, right)[None, :]
    )


def canvas_noise(latents, generator):
    """Float32 noise shaped like the whole `latents` canvas, drawn once."""
    from diffusers.utils.torch_utils import randn_tensor

    return randn_tensor(
        latents.shape,
        generator=generator,
        device=latents.device,
        dtype=torch.float32,
    )


def crop(canvas, tile, scale=LATENT_SCALE):
    """The `(x, y, w, h)` pixel tile of a `(..., H, W)` latent canvas."""
    x, y, width, height = tile
    return canvas[
        ..., y // scale : (y + height) // scale, x // scale : (x + width) // scale
    ]


def blend(tiles, plan, shape):
    """The latent canvas of `shape` from denoised `tiles` (one tensor per entry
    of `plan`): each weighted by its feather, summed, divided by the weights'
    sum. Float32 on the CPU, whatever the tiles came in on.

    Raises:
        ValueError: If a pixel of the canvas is in no tile
    """
    total = torch.zeros(shape, dtype=torch.float32)
    weight_sum = torch.zeros(shape[-2:], dtype=torch.float32)
    for tile, latents in zip(plan, tiles):
        weights = feather_weights(plan, tile)
        x, y, width, height = tile
        rows = slice(y // LATENT_SCALE, (y + height) // LATENT_SCALE)
        columns = slice(x // LATENT_SCALE, (x + width) // LATENT_SCALE)
        total[..., rows, columns] += latents.detach().float().cpu() * weights
        weight_sum[rows, columns] += weights
    if bool((weight_sum <= 0).any()):
        raise ValueError("refine_tiles leaves part of the canvas uncovered")
    return total / weight_sum


@contextlib.contextmanager
def swapped_blocks(pipeline, blocks):
    """Run `pipeline` on `blocks` for the duration, restoring its own."""
    original = pipeline._blocks
    pipeline._blocks = blocks
    try:
        yield
    finally:
        pipeline._blocks = original


def decode_canvas(pipeline, canvas):
    """Frames of a latent canvas, decoded once by the pipeline's own video
    decode block, as the one-video list a plain call's `videos` is."""
    from diffusers.modular_pipelines.modular_pipeline import PipelineState

    state = PipelineState()
    state.set(
        "latents",
        canvas.to(device=pipeline._execution_device, dtype=torch.float32),
    )
    state.set("output_type", "pil")
    getattr(pipeline, VIDEO_DECODE)(pipeline, state)
    return state.get("videos")


def run_tiled_refine(pipeline, arguments, label, decode=None):
    """Refine `arguments["latents"]` as the tiles of `arguments["refine_tiles"]`
    and return what a plain call returns for the step's `output`: a dict of
    `videos`, `latents` (the blended canvas, 5D on the CPU) and the held audio
    keys, narrowed to the names `output` lists (all of them when it lists none,
    a bare value when it is one name).

    Args:
        pipeline: The loaded H3 modular pipeline
        arguments: The step's call arguments, `refine_tiles` and `latents` among them
        label: What the progress narration calls the step
        decode: `decode(canvas) -> videos`, the pipeline's own decode when None

    Raises:
        ValueError: If the pipeline's graph cannot drop its decode blocks
    """
    tile_blocks = getattr(pipeline, TILE_BLOCKS, None)
    if tile_blocks is None:
        raise ValueError(
            f"{REFINE_TILES_INPUT} needs a MiniMax-H3 graph with a decode "
            f"block to leave out, and {type(pipeline).__name__} has none"
        )
    settings = arguments[REFINE_TILES_INPUT]
    arguments = {k: v for k, v in arguments.items() if k != REFINE_TILES_INPUT}
    latents = arguments["latents"]
    canvas_width = latents.shape[-1] * LATENT_SCALE
    canvas_height = latents.shape[-2] * LATENT_SCALE
    plan = tile_plan(
        canvas_width,
        canvas_height,
        settings["width"],
        settings["height"],
        settings["overlap"],
    )
    requested = arguments.pop("output", None)
    noise = canvas_noise(latents, arguments.get("generator"))

    text_blocks = getattr(pipeline, TEXT_BLOCKS, None)
    if text_blocks is not None:
        takes = _input_names(text_blocks)
        with swapped_blocks(pipeline, text_blocks):
            encoded = pipeline(
                **{k: v for k, v in arguments.items() if k in takes},
                output=TEXT_OUTPUTS,
            )
        # What the tiles take instead of the prompt itself
        kept = _input_names(tile_blocks)
        arguments = {k: v for k, v in arguments.items() if k not in takes or k in kept}
        arguments.update(encoded)

    context = get_context()
    tiles, held = [], {}
    with swapped_blocks(pipeline, tile_blocks), decoded_later():
        for index, tile in enumerate(plan):
            context.check_cancelled()
            x, y, width, height = tile
            emit_log(
                f"Refining tile {index + 1}/{len(plan)} ({x},{y} {width}x{height})",
                tile=index + 1,
                tiles=len(plan),
                x=x,
                y=y,
                width=width,
                height=height,
            )
            names = ["latents"]
            if index == 0:
                names += [HELD_AUDIO_OUTPUT, HELD_AUDIO_RATE_OUTPUT]
            with reported_blocks(pipeline, f"{label} tile {index + 1}/{len(plan)}"):
                result = pipeline(
                    **{
                        **arguments,
                        "width": width,
                        "height": height,
                        "latents": crop(latents, tile),
                        REFINE_NOISE: crop(noise, tile),
                        "output": names,
                    }
                )
            tiles.append(result["latents"])
            if index == 0:
                held = {k: result.get(k) for k in names[1:]}

    canvas = blend(tiles, plan, tuple(latents.shape))
    emit_phase("decoding")
    videos = (decode or (lambda value: decode_canvas(pipeline, value)))(canvas)
    if isinstance(videos, list):
        # Bare frame lists carry no rate, and pair_audio or the writer would
        # play them at 8 fps: say they are H3's 24
        videos = FrameList(
            (
                FrameList(video, H3_FPS) if isinstance(video, list) else video
                for video in videos
            ),
            H3_FPS,
        )
    outputs = {"videos": videos, "latents": canvas, **held}
    if requested is None:
        return outputs
    if isinstance(requested, str):
        return outputs.get(requested)
    return {name: outputs[name] for name in requested if name in outputs}
