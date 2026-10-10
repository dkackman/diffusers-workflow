"""The model-backed task commands' handlers.

Split from `dw/tasks/task.py` (#790), which had grown past the module-size
ceiling with every handler in one file. Each of these loads a model - an
upscaler, a face restorer, a captioner, a speech or text model, the H3 latent
upscaler and VAE, a voice or stem separator - so each registers with
`consumes_device=True` and asks `task.device_for(arguments)` which
accelerator to run on.

Their implementations are imported inside the handlers: at module scope the
transformers/model imports add seconds to every startup for workflows that
never run these tasks. `dw/tasks/task.py` imports this module, which is what
registers its commands (`docs/TASKS.md` *Adding a task*).
"""

import logging

from .image_ops import per_frame
from .registry import register_command
from ..task_domains import POSITIVE, UNIT, script_lines_errors

logger = logging.getLogger("dw")


def _voices_argument_errors(arguments):
    """attribute_voices' static check, imported on use: its module loads torch."""
    from .voice_attribution import voices_argument_errors

    return voices_argument_errors(arguments)


@register_command(
    "attribute_voices",
    implementation="dw.tasks.voice_attribution.attribute_voices",
    consumes_device=True,
    returns="json",
    domains={"window_seconds": POSITIVE, "min_reference_seconds": POSITIVE},
    static_check=_voices_argument_errors,
)
def _handle_attribute_voices(task, arguments, previous_pipelines):
    """Say which reference voice sings each line of a song, by timbre"""
    from .voice_attribution import attribute_voices

    return attribute_voices(device=task.device_for(arguments), **arguments)


@register_command(
    "check_script",
    implementation="dw.tasks.script_check.check_script",
    consumes_device=True,
    returns="json",
    domains={"similarity": UNIT},
    static_check=script_lines_errors,
)
def _handle_check_script(task, arguments, previous_pipelines):
    """Check that a take speaks its script, line by line"""
    from .script_check import check_script

    return check_script(device=task.device_for(arguments), **arguments)


@register_command(
    "separate_stems",
    implementation="dw.tasks.voice_attribution.separate_stems",
    consumes_device=True,
)
def _handle_separate_stems(task, arguments, previous_pipelines):
    """Split a mix into vocals, drums, bass and other stems, one audio result each"""
    from .voice_attribution import separate_stems

    return separate_stems(device=task.device_for(arguments), **arguments)


@register_command(
    "upscale", implementation="dw.tasks.upscale.upscale_image", consumes_device=True
)
def _handle_upscale(task, arguments, previous_pipelines):
    """Upscale an image using a spandrel-compatible super-resolution model"""
    logger.debug("Upscaling image")
    image = arguments.pop("image")
    model_name = arguments.pop("model_name")
    from .upscale import upscale_image

    device = task.device_for(arguments)
    return per_frame(
        image,
        lambda frame: upscale_image(frame, model_name, device=device, **arguments),
    )


@register_command(
    "diffusion_upscale",
    implementation="dw.tasks.diffusion_upscale.diffusion_upscale",
    consumes_device=True,
)
def _handle_diffusion_upscale(task, arguments, previous_pipelines):
    """Upscale an image using a diffusion-based upscale pipeline"""
    logger.debug("Diffusion upscaling image")
    image = arguments.pop("image")
    from .diffusion_upscale import diffusion_upscale

    device = task.device_for(arguments)
    return per_frame(
        image, lambda frame: diffusion_upscale(frame, device=device, **arguments)
    )


@register_command(
    "restore_faces",
    implementation="dw.tasks.restore_faces.restore_faces",
    consumes_device=True,
)
def _handle_restore_faces(task, arguments, previous_pipelines):
    """Restore faces in an image using a spandrel-compatible face restoration model"""
    logger.debug("Restoring faces")
    image = arguments.pop("image")
    model_name = arguments.pop("model_name")
    from .restore_faces import restore_faces
    from .video_utils import is_video

    if is_video(image) and arguments.get("upsample_img") is not None:
        raise ValueError(
            "restore_faces: 'upsample_img' is a single background image and "
            "cannot be used with a video input (it would be pasted under "
            "every frame). Remove 'upsample_img' or pass a single image."
        )

    device = task.device_for(arguments)
    return per_frame(
        image,
        lambda frame: restore_faces(frame, model_name, device=device, **arguments),
    )


@register_command(
    "segment", implementation="dw.tasks.segment.segment_image", consumes_device=True
)
def _handle_segment(task, arguments, previous_pipelines):
    """Segment objects in an image using text prompt"""
    logger.debug("Segmenting image")
    image = arguments.pop("image")
    prompt = arguments.pop("prompt")
    from .segment import segment_image

    device = task.device_for(arguments)
    return per_frame(
        image, lambda frame: segment_image(frame, prompt, device=device, **arguments)
    )


@register_command(
    "interpolate_frames",
    implementation="dw.tasks.interpolate_frames.interpolate_frames",
    consumes_device=True,
)
def _handle_interpolate_frames(task, arguments, previous_pipelines):
    """Interpolate video frames to increase frame rate"""
    logger.debug("Interpolating frames")
    video = arguments.pop("video")
    from .interpolate_frames import interpolate_frames

    return interpolate_frames(video, device=task.device_for(arguments), **arguments)


@register_command(
    "upscale_h3_latents",
    implementation="dw.tasks.h3_latent_upscale.upscale_h3_latents",
    consumes_device=True,
    summary=(
        "Resize MiniMax-H3 video latents to a larger canvas (e.g. a 960x544 "
        "take to 1344x768) without denoising again; decode_h3_latents turns "
        "the result into frames. The whole promoting workflow is "
        "get_guide('workflows', section='upscale_h3_latents')."
    ),
    parameter_descriptions={
        "latents": (
            "H3 video latents, (batch, 24, frames, height, width) - a MiniMax-H3 "
            "pipeline step's 'latents' output, e.g. 'previous_result:base.latents' "
            "from a step whose output lists 'latents'."
        ),
        "width": (
            "Target width in pixels: a multiple of 16, 1x to 4x the latents' "
            "own width, within H3's 1344x768 canvas."
        ),
        "height": (
            "Target height in pixels: a multiple of 16, 1x to 4x the latents' "
            "own height, within H3's 1344x768 canvas."
        ),
        "model_name": (
            "Hugging Face repo id holding the upscaler, never a path (default "
            "LBH-123-AI/Minimax_h3_latent_Upscaler, read at a pinned revision)."
        ),
        "weight_name": (
            "The safetensors file within model_name, a bare file name - no "
            "'/' (default: the bf16 v1 checkpoint; in the default repo it is "
            "read from the v1 checkpoint folder)."
        ),
    },
    domains={"width": POSITIVE, "height": POSITIVE},
)
def _handle_upscale_h3_latents(task, arguments, previous_pipelines):
    """Resize MiniMax-H3 video latents to a larger canvas"""
    logger.debug("Upscaling H3 latents")
    latents = arguments.pop("latents")
    from .h3_latent_upscale import upscale_h3_latents

    return upscale_h3_latents(latents, device=task.device_for(arguments), **arguments)


@register_command(
    "decode_h3_latents",
    implementation="dw.tasks.h3_latent_upscale.decode_h3_latents",
    consumes_device=True,
    summary=(
        "Decode MiniMax-H3 video latents into frames with the H3 video VAE, "
        "through diffusers' own H3 decode block. Returns video only, at 24 fps; "
        "pair_audio puts a soundtrack back under it. The whole promoting "
        "workflow is get_guide('workflows', section='upscale_h3_latents')."
    ),
    parameter_descriptions={
        "latents": (
            "H3 video latents, (batch, 24, frames, height, width) - a pipeline "
            "step's 'latents' output or an upscale_h3_latents result."
        ),
        "model_name": (
            "The MiniMax-H3 repo whose 'vae' subfolder decodes (default "
            "MiniMaxAI/MiniMax-H3)."
        ),
    },
)
def _handle_decode_h3_latents(task, arguments, previous_pipelines):
    """Decode MiniMax-H3 video latents into frames"""
    logger.debug("Decoding H3 latents")
    latents = arguments.pop("latents")
    from .h3_latent_upscale import decode_h3_latents

    return decode_h3_latents(latents, device=task.device_for(arguments), **arguments)


@register_command(
    "image_to_text",
    implementation="dw.tasks.image_to_text.image_to_text",
    consumes_device=True,
)
def _handle_image_to_text(task, arguments, previous_pipelines):
    """Generate text caption from an image"""
    logger.debug("Captioning image")
    image = arguments.pop("image")
    from .image_to_text import image_to_text

    return image_to_text(image, device=task.device_for(arguments), **arguments)


@register_command(
    "judge",
    implementation="dw.tasks.judge.judge",
    consumes_device=True,
    returns="scalar",
)
def _handle_judge(task, arguments, previous_pipelines):
    """Score an image against a rubric with a vision-language model"""
    logger.debug("Judging")
    image = arguments.pop("image")
    from .judge import judge

    return judge(image, device=task.device_for(arguments), **arguments)


@register_command(
    "transcribe_audio",
    implementation="dw.tasks.audio_transcription.transcribe_audio",
    consumes_device=True,
)
def _handle_transcribe_audio(task, arguments, previous_pipelines):
    """Transcribe spoken audio to text"""
    logger.debug("Transcribing audio")
    audio = arguments.pop("audio")
    from .audio_transcription import transcribe_audio

    return transcribe_audio(audio, device=task.device_for(arguments), **arguments)


@register_command(
    "text_generation",
    implementation="dw.tasks.text_generation.generate_text",
    consumes_device=True,
)
def _handle_text_generation(task, arguments, previous_pipelines):
    """Generate text from a prompt using a local LLM"""
    logger.debug("Generating text")
    prompt = arguments.pop("prompt")
    from .text_generation import generate_text

    return generate_text(prompt, device=task.device_for(arguments), **arguments)


@register_command(
    "generate_speech",
    implementation="dw.tasks.speech_generation.generate_speech",
    consumes_device=True,
)
def _handle_speech_generation(task, arguments, previous_pipelines):
    """Speak a line of text with a local text-to-speech model"""
    logger.debug("Generating speech")
    if ("text" in arguments) == ("messages" in arguments):
        raise ValueError(
            "generate_speech needs exactly one of 'text' (the line to speak) "
            "or 'messages' (chat-templated input for a model such as VibeVoice)"
        )
    text = arguments.pop("text", None)
    from .speech_generation import generate_speech

    return generate_speech(
        text,
        device=task.device_for(arguments),
        seed=task.seed_for(arguments),
        **arguments,
    )
