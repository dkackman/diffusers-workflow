"""LTX2RefinePipeline: LTX2Pipeline that refines a clip at its own size.

`LTX2Pipeline` takes `latents` but never pixels, and the only diffusers route
from a clip to latents, `LTX2LatentUpsamplePipeline`, always runs its 2x
upsampler. This subclass adds one argument, `video`: the clip is VAE-encoded
at `width` x `height` and handed to the parent as `latents`, so the parent's
own `noise_scale` blend and `sigmas` schedule refine it in place.

The encode is the upsample pipeline's, minus the upsampler: the video
processor's preprocess, then `vae.encode`, with the result left unnormalized.
That is deliberate - `LTX2Pipeline.prepare_latents` normalizes 5-D latents
itself, exactly as it does for the upsampler's raw output in refine-clip, so
normalizing here would do it twice and start the refine off-distribution.

The class holds no schedule and no model constant: the strength is the
workflow's `noise_scale` and `sigmas`.
"""

import inspect
import re
import textwrap

import torch
from diffusers import LTX2Pipeline

from ..media_types import AudioVideo


def retrieve_latents(encoder_output, generator=None):
    """The sampled latents of a VAE encode - the upsample pipeline's own
    `retrieve_latents` with its default sample mode."""
    if hasattr(encoder_output, "latent_dist"):
        return encoder_output.latent_dist.sample(generator)
    if hasattr(encoder_output, "latents"):
        return encoder_output.latents
    raise AttributeError("Could not access latents of provided encoder_output")


_PARENT_SIGNATURE = inspect.signature(LTX2Pipeline.__call__)

VIDEO_DOC = """    video (`list[PIL.Image.Image]`, `np.ndarray`, `torch.Tensor` or a dw video, *optional*):
        A clip to refine at its own size; a dw video (an `AudioVideo`, e.g. `fit_to_model`'s) is taken as its
        frames. It is resized to `width` x `height`, VAE-encoded and passed to
        `LTX2Pipeline` as `latents`, which renoises it at `noise_scale` and denoises it over `sigmas`.
        `num_frames` defaults to the clip's length floored to the 8 * n + 1 grid; an explicit `num_frames` must
        be on that grid and no longer than the clip, which is trimmed to it. Cannot be combined with `latents`.
"""


def _call_doc():
    """The parent's call docstring with `video` added at the head of Args.

    `video` is indented to match the parent's own entries, read off its
    docstring: a wrapped parent can leave its Args block indented past what
    `inspect.cleandoc` strips, and a `video` at a different indent would fold
    every entry after it into its description.
    """
    doc = LTX2Pipeline.__call__.__doc__ or "Args:\n"
    head, args, rest = doc.partition("Args:\n")
    entry = re.match(r"[ \t]*", rest).group()
    video = textwrap.indent(textwrap.dedent(VIDEO_DOC), entry)
    return head + args + video + rest


class LTX2RefinePipeline(LTX2Pipeline):
    """LTX2Pipeline with a `video` argument: refine a clip at its own size,
    its VAE-encoded latents renoised at `noise_scale` over `sigmas`."""

    def encode_video(self, video, width, height, num_frames=None, generator=None):
        """The clip's VAE latents [1, C, F, H, W] at `width` x `height`,
        unnormalized - the parent normalizes 5-D latents itself.

        `num_frames` None takes the clip's length floored to the VAE's
        8 * n + 1 grid. An explicit one must be on that grid and no longer
        than the clip; the clip is trimmed to it. A dw `AudioVideo` is
        taken as its frames: it has no length of its own to slice.
        """
        if isinstance(video, AudioVideo):
            video = video.frames
        ratio = self.vae_temporal_compression_ratio
        available = len(video)
        if num_frames is None:
            num_frames = (available - 1) // ratio * ratio + 1
            if available < 1 + ratio:
                raise ValueError(
                    f"`video` has {available} frames; LTX2RefinePipeline needs at least {1 + ratio}."
                )
        elif num_frames % ratio != 1:
            raise ValueError(
                f"`num_frames` must be {ratio} * n + 1 for the video VAE, but is {num_frames}."
            )
        elif num_frames > available:
            raise ValueError(
                f"`num_frames` is {num_frames} but `video` has only {available} frames; "
                "trim or loop the clip first, or lower `num_frames`."
            )
        pixels = self.video_processor.preprocess_video(
            video[:num_frames], height=height, width=width
        )
        pixels = pixels.to(device=self._execution_device, dtype=self.vae.dtype)
        latents = retrieve_latents(self.vae.encode(pixels), generator)
        return latents.to(torch.float32)

    @torch.no_grad()
    def __call__(self, *args, video=None, **kwargs):
        if video is None:
            return super().__call__(*args, **kwargs)

        bound = _PARENT_SIGNATURE.bind(self, *args, **kwargs)
        bound.apply_defaults()
        call = dict(bound.arguments)
        call.pop("self")
        if call["latents"] is not None:
            raise ValueError(
                "Pass either `video` or `latents` to LTX2RefinePipeline, not both: "
                "`video` is encoded into `latents`."
            )

        generator = call["generator"]
        if isinstance(generator, list):
            generator = generator[0]
        latents = self.encode_video(
            video,
            width=call["width"],
            height=call["height"],
            num_frames=call["num_frames"],
            generator=generator,
        )
        prompt, prompt_embeds = call["prompt"], call["prompt_embeds"]
        if isinstance(prompt, list):
            prompts = len(prompt)
        elif prompt is None and prompt_embeds is not None:
            prompts = prompt_embeds.shape[0]
        else:
            prompts = 1
        call["latents"] = latents.repeat(
            prompts * call["num_videos_per_prompt"], 1, 1, 1, 1
        )
        call["num_frames"] = (
            latents.shape[2] - 1
        ) * self.vae_temporal_compression_ratio + 1
        return super().__call__(**call)


# The wrapper's own *args/**kwargs would hide the parent's arguments from
# get_pipeline_signature and from validation's unknown-argument check, so it
# advertises the parent's signature with `video` added.
LTX2RefinePipeline.__call__.__signature__ = _PARENT_SIGNATURE.replace(
    parameters=[
        *_PARENT_SIGNATURE.parameters.values(),
        inspect.Parameter("video", inspect.Parameter.KEYWORD_ONLY, default=None),
    ]
)
LTX2RefinePipeline.__call__.__doc__ = _call_doc()
