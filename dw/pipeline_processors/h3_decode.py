"""H3's long decode: the video latents a piece of the take at a time
(`streamed_decode`, `DwH3StreamedVideoDecodeStep`), split out of
`h3_windows.py`, which swaps the block in and re-exports these names.

Imports diffusers when it loads, so it is imported only by `h3_windows`.
"""

import torch
from diffusers.modular_pipelines.minimax_h3.decoders import MiniMaxH3VideoDecodeStep

# 5. The long decode: a piece of the take at a time

# VAE clips per decode piece: 8 clips are 136 frames, about 0.85 GB of float32
# pixels at 960x544 and 1.7 GB at 1344x768
DECODE_PIECE_CLIPS = 8


def streamed_decode(vae, z, finish, piece_clips=DECODE_PIECE_CLIPS):
    """Decode H3 video latents `z` a piece at a time, handing each run of
    finished pixel frames to `finish` in order. The runs, concatenated in
    time, equal `vae.decode(z)` exactly.

    The stock decode keeps every decoded clip on the device and concatenates
    them, then the decode block converts the whole take to float32: about
    9 + 9 + 18 GB for 120 s at 960x544, past a 24 GB card. A take that long
    exists only through the windowed refine.

    Each piece is `vae.decode` of `piece_clips` whole clips' latents plus the
    `token_overlap` latents the last clip also reads, so `decode` decodes
    exactly those clips: the same latents the whole decode gives each clip,
    and through `decode`, so on-demand placement and offload hooks still run.
    `decode` leaves a piece's last clip's overlap frames raw at its end, and
    the whole decode cross-fades them into the next clip's first frames. That
    tail is held back and blended into the next piece with the VAE's own
    `_blend`, which is how `_decode` joins two clips. The last piece runs to
    the end of `z`, so `decode` pads and trims the end as the whole decode
    does: the padding depends only on the latent count modulo the clip, and
    every piece starts on a clip.

    A VAE with `token_drop` 0 has no overlap to join: it decodes whole.

    Raises:
        ValueError: If `piece_clips` is not a whole number of 1 or more
    """
    if isinstance(piece_clips, bool) or not isinstance(piece_clips, int):
        raise ValueError(f"piece_clips is a whole number, got {piece_clips!r}")
    if piece_clips < 1:
        raise ValueError(f"piece_clips is 1 or more, got {piece_clips}")
    clip = vae.tokens_chunk_size
    overlap = vae.token_overlap
    if overlap < 1:
        # No `token_drop`: clips do not overlap, so there is no tail to hold
        # back and the stock whole decode is already the right one
        finish(vae.decode(z, return_dict=False)[0])
        return
    # Frames each clip contributes before the next clip's cross-fade
    clip_frames = clip * vae.temporal_compression_ratio - vae.frame_pre_padding
    total = z.shape[2]
    piece = piece_clips * clip
    tail = None
    start = 0
    while True:
        # A piece is not the last while one more clip reads past it: then the
        # whole decode blends this piece's tail into that clip
        last = start + piece + overlap >= total
        stop = total if last else start + piece + overlap
        pixels = vae.decode(z[:, :, start:stop], return_dict=False)[0]
        if tail is not None:
            pixels = vae._blend(tail, pixels, vae.frame_overlap, dim=-3)
        if last:
            finish(pixels)
            return
        kept = piece_clips * clip_frames
        tail = pixels[:, :, kept:]
        finish(pixels[:, :, :kept])
        del pixels
        start += piece


class DwH3StreamedVideoDecodeStep(MiniMaxH3VideoDecodeStep):
    """H3's video decode, a piece of the take at a time (`streamed_decode`).

    The frames are the stock block's exactly. Each piece is denormalized,
    clamped and post-processed to the output type as it is decoded, so the
    device holds one piece's pixels rather than the whole take's. A `pt`
    output goes through the stock block."""

    piece_clips = DECODE_PIECE_CLIPS

    @property
    def description(self):
        return (
            "dw: decodes the video latents a piece of the take at a time, so a long take's pixels never sit on "
            "the device whole - the frames are the stock decode's. "
            + super().description
        )

    @torch.no_grad()
    def __call__(self, components, state):
        block_state = self.get_block_state(state)
        output_type = block_state.output_type
        if output_type not in ("pil", "np"):
            return super().__call__(components, state)
        device = components._execution_device
        config = components.vae.config
        latents_mean = torch.tensor(config.latents_mean, device=device).view(
            1, -1, 1, 1, 1
        )
        latents_std = torch.tensor(config.latents_std, device=device).view(
            1, -1, 1, 1, 1
        )
        latents = block_state.latents * latents_std + latents_mean
        pixel_mean = torch.tensor(components.pixel_mean, device=device).view(
            1, -1, 1, 1, 1
        )
        pixel_std = torch.tensor(components.pixel_std, device=device).view(
            1, -1, 1, 1, 1
        )
        pieces = []

        def finish(video):
            video = (video.float() * pixel_std + pixel_mean).clamp(0, 1)
            pieces.append(
                components.video_processor.postprocess_video(
                    video, output_type=output_type
                )
            )

        streamed_decode(components.vae, latents, finish, self.piece_clips)
        if output_type == "np":
            import numpy

            block_state.videos = numpy.concatenate(pieces, axis=1)
        else:
            block_state.videos = [
                [frame for piece in pieces for frame in piece[batch]]
                for batch in range(latents.shape[0])
            ]
        self.set_block_state(state, block_state)
        return components, state


def _video_decode_slots(block):
    """[(holder, key)] for every MiniMaxH3VideoDecodeStep under `block`."""
    slots = []
    for key, child in (getattr(block, "sub_blocks", None) or {}).items():
        if isinstance(child, MiniMaxH3VideoDecodeStep):
            slots.append((block.sub_blocks, key))
        else:
            slots.extend(_video_decode_slots(child))
    return slots


def video_decode_steps(pipeline):
    """The video decode blocks a pipeline runs."""
    return [
        holder[key]
        for holder, key in _video_decode_slots(getattr(pipeline, "_blocks", None))
    ]
