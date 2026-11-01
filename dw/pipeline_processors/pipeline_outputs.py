"""What a pipeline's output needs after the call: the audio sample rate the
pipeline's components imply, attached so the result writer knows the rate of
an `audios` output; the safety-checker warning when every image came back
blank; and the diagnosis of a CRF error that means a video was passed where
an image step expected a picture.

Split from dw/pipeline_processors/pipeline.py, which imports from here
(never the other way) and calls these around `_run_once`.
"""

import logging

from ..events import emit_warning
from ..media_types import AudioVideo

logger = logging.getLogger("dw")


def _denoiser(pipeline):
    """The module whose forward is one denoise pass, or None."""
    from .components import get_cache_transformer

    denoiser = get_cache_transformer(pipeline)
    return denoiser if denoiser is not None else getattr(pipeline, "unet", None)


def warn_if_safety_checker_blanked(output):
    """Say so when a safety checker replaced a generated image with a black one.

    Stable Diffusion 1.5's checker false-positives readily, and it returns a
    solid black image rather than an error. Run to run that reads as the seed
    having no effect - the same result every time - so the reason belongs
    where the identical images do: the run's warnings, which is the only
    place a consumer of a `succeeded` job would ever see it.

    Args:
        output: The pipeline output, which may carry nsfw_content_detected
    """
    flags = getattr(output, "nsfw_content_detected", None)
    if not flags:
        return

    blanked = sum(1 for flag in flags if flag)
    if blanked:
        # emit_warning rather than logger.warning: a blanked image is a
        # succeeded job whose file is solid black, and a consumer over the
        # API or MCP sees the job's warnings list and nothing else - the log
        # line never reaches the one party that cannot tell the picture apart
        # from a rendered one (#133)
        emit_warning(
            f"The safety checker blanked {blanked} of {len(flags)} generated "
            "images - they are solid black, and no seed will change that. "
            "Pass 'safety_checker': null in from_pretrained_arguments to "
            "load the pipeline without it.",
            kind="safety_checker_blanked",
            blanked=blanked,
            images=len(flags),
        )


# Where an audio pipeline's components record the rate they generate at, in
# the order they are tried. LTX-2's vocoder names it output_sampling_rate,
# AudioLDM2's vocoder and StableAudio's VAE name it sampling_rate, and
# Kandinsky 6's MMAudioVAE names it sample_rate (its vocoder carries none).
# A vocoder's rate comes first: LTX-2's audio VAE works below it
_SAMPLE_RATE_SOURCES = (
    ("vocoder", "output_sampling_rate"),
    ("vocoder", "sampling_rate"),
    ("audio_vae", "sample_rate"),
    ("vae", "sampling_rate"),
)


def _component_sample_rate(pipeline, has_audios):
    for component_name, attribute in _SAMPLE_RATE_SOURCES:
        if component_name == "vae" and not has_audios:
            # A video VAE's config is not where an audio rate lives - only an
            # audio-only pipeline's VAE (StableAudio) reports one there
            continue
        config = getattr(getattr(pipeline, component_name, None), "config", None)
        sample_rate = getattr(config, attribute, None)
        if sample_rate is not None:
            return sample_rate
    return None


def attach_audio_sample_rate(pipeline, output):
    """Record the generating component's sample rate on an output that carries audio.

    Pipelines that generate audio - with their video (LTX-2, on `.audio`) or by
    itself (AudioLDM2, StableAudio, on `.audios`) - return the waveform without
    its sample rate; only the vocoder or VAE that produced it knows that. Saving
    needs the rate, so it travels with the output, and get_artifact_list wraps
    `.audios` items in an AudioTrack that carries it.

    Args:
        pipeline: The pipeline that produced the output
        output: The pipeline output
    """
    has_audio = getattr(output, "audio", None) is not None
    has_audios = getattr(output, "audios", None) is not None
    if not (has_audio or has_audios):
        return

    sample_rate = _component_sample_rate(pipeline, has_audios)
    if sample_rate is None:
        logger.warning(
            "Pipeline generated audio but no component reports its sample rate - "
            "set 'sample_rate' (audio results) or 'audio_sample_rate' (muxed video) "
            "in the step result to save it correctly"
        )
        return

    logger.debug(f"Generated audio has a sample rate of {sample_rate}Hz")
    output.audio_sample_rate = sample_rate


def _diagnose_image_crf_error(error, arguments):
    """Rewrite diffusers' `image_crf` re-compression error into one that
    names the actual mismatch, when it is one: an argument the pipeline
    expects as a still image (`PIL.Image.Image`) instead holds an
    `AudioVideo` - a video where an image was required. Diffusers' own
    message suggests `image_crf=0`, which does not fix anything, since the
    argument still receives a video; the caller needs to derive a still
    first. Returns None when the error is something else, so the caller
    re-raises unchanged.
    """
    if not isinstance(error, (ValueError, TypeError)):
        return None
    if "image_crf" not in str(error) or "re-compression requires" not in str(error):
        return None

    offending = [
        name for name, value in arguments.items() if isinstance(value, AudioVideo)
    ]
    if not offending:
        return None

    names = ", ".join(sorted(offending))
    return ValueError(
        f"Argument(s) {names} expected an image but received a video. Derive a "
        f"still from it first - `get_last_frame` or `video_frames` - and pass "
        f"that instead."
    )
