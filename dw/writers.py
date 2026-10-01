"""The byte-level writes behind `Result`: where a file goes, how pixels,
frames and waveforms are shaped for the encoder, and the image metadata pair.

None of this knows about a step or a result definition - it is what
`Result.save_artifact` calls once it has decided what to write. The video
writes (`export_to_video`, `encode_video`) stay in `Result`, where their
patch targets are looked up.
"""

import json
import logging
import os

import numpy
import soundfile
import torch

from .events import emit_warning
from .security import MAX_DECODE_PIXELS, validate_output_path

logger = logging.getLogger("dw")


def _artifact_size(artifact):
    """How much there is to write, said the way the thing itself counts -
    frames for a video, samples for a waveform. Best effort: it is narration
    beside a file name, so anything it cannot measure it does not mention."""
    try:
        frames = getattr(artifact, "frames", None)
        if frames is not None:
            return f"{len(frames)} frames"
        if hasattr(artifact, "__len__") and not isinstance(artifact, (str, bytes)):
            return f"{len(artifact)} frames"
    except Exception:
        pass
    return ""


def _file_size_mb(path):
    try:
        return os.path.getsize(path) / (1024 * 1024)
    except OSError:
        return 0.0


# Image formats that carry an alpha channel; every other image content type
# takes the picture flattened over white
ALPHA_CONTENT_TYPES = {"image/png", "image/webp", "image/gif", "image/tiff"}


def flatten_alpha_for(image, content_type, file_name):
    """The image as the content type can carry it: unchanged when it has no
    alpha channel or the format keeps one, otherwise composited over white
    and said out loud.

    Pillow refuses to write mode RGBA as JPEG, and it refused after the whole
    generation had run - Qwen-Image-2.1 decodes an alpha channel natively,
    and every image template in the catalog asked for image/jpeg. A warning
    rather than an error, because the run did what was asked and the file
    is usable; the warning names the format that would have kept the alpha.
    """
    mode = getattr(image, "mode", None)
    if mode not in ("RGBA", "LA", "PA") or content_type in ALPHA_CONTENT_TYPES:
        return image
    from PIL import Image as PILImage

    rgba = image.convert("RGBA")
    flattened = PILImage.new("RGB", rgba.size, (255, 255, 255))
    flattened.paste(rgba, mask=rgba.getchannel("A"))
    if getattr(image, "info", None):
        flattened.info.update(image.info)
    emit_warning(
        f"The image written to {file_name} had an alpha channel that "
        f"{content_type} cannot carry - it was flattened over white. Set "
        f"the step's result 'content_type' to 'image/png' to keep the "
        f"transparency.",
        kind="alpha_discarded",
        file=file_name,
        content_type=content_type,
    )
    return flattened


def frames_for_encoding(frames):
    """Generated frames in the form `encode_video` encodes without first
    inspecting them.

    A pipeline that returns `output_type="np"` hands back float frames in
    [0, 1], and diffusers' `encode_video` establishes that range with three
    full-size temporaries - `np.zeros_like`, `np.ones_like` and the bool
    mask - before converting. On a 121-frame 960x544 clip that is ~3 GB of
    allocation and 16 s of wall clock on an idle box, against 2.4 s for the
    encode itself, and it is the bulk of a 'saving' phase that ran for 53 s
    with nothing else in it (#97, measured on lem 2026-09-14).

    Converting here is a pass and a half and hands back a torch tensor,
    which `encode_video` takes as given - so the check never runs. Frames
    outside [0, 1] are left exactly as they were: that is the branch where
    diffusers warns and treats them as pixel values already, and it is not
    a path any pipeline here produces or that this can be tested against.

    The source array is never written to - a later step may still read this
    result through a `previous_result:` reference, and the step cache
    retains it.
    """
    if not isinstance(frames, numpy.ndarray) or frames.size == 0:
        return frames
    if not numpy.issubdtype(frames.dtype, numpy.floating):
        return frames
    if float(frames.min()) < 0.0 or float(frames.max()) > 1.0:
        return frames
    denormalized = numpy.empty(frames.shape, dtype=numpy.uint8)
    # Frame by frame: the whole-array form allocates another copy the size
    # of the video, which is the cost this exists to avoid
    for index in range(frames.shape[0]):
        denormalized[index] = numpy.round(frames[index] * 255.0)
    return torch.from_numpy(denormalized)


def output_file_path(output_dir, file_name):
    """The path a result file is written to, confined to the output directory.

    Every part of the name is workflow-supplied - the workflow id, the step
    name, a result's file_base_name, and the keys of a dict artifact - and
    they are concatenated into a file name. Without this a name carrying a
    path separator would write outside the output directory, so the joined
    path goes through the same validator every other path in the engine does.

    A rerun that would otherwise produce a name already on disk gets a
    '-2', '-3', ... counter instead of silently overwriting it - the same
    guarantee ComfyUI's SaveImage node makes by scanning its output
    directory before every write.
    """
    candidate = validate_output_path(os.path.join(output_dir, file_name), output_dir)
    return _dedupe_existing_path(candidate)


def _dedupe_existing_path(path):
    """Append an incrementing counter before the extension until `path` is free."""
    if not os.path.exists(path):
        return path

    base, ext = os.path.splitext(path)
    counter = 2
    while True:
        candidate = f"{base}-{counter}{ext}"
        if not os.path.exists(candidate):
            return candidate
        counter += 1


# Audio is written in chunks of this many frames - see write_audio
AUDIO_WRITE_CHUNK_FRAMES = 1 << 20


def write_audio(output_path, waveform, sample_rate, **write_arguments):
    """Write a single waveform to disk.

    soundfile.write() hands the whole waveform to libsndfile in one call, whose vorbis
    encoder segfaults past 2**21 frames - about 48 seconds of 44.1kHz audio. Writing in
    chunks avoids that and bounds the encoder's working set for long audio.

    Args:
        output_path: Path of the file to write
        waveform: Numpy array shaped (samples,) or (samples, channels)
        sample_rate: Sample rate to record in the file
        write_arguments: Container and encoding arguments for soundfile
    """
    channels = 1 if waveform.ndim == 1 else waveform.shape[1]

    with soundfile.SoundFile(
        output_path,
        "w",
        samplerate=sample_rate,
        channels=channels,
        **write_arguments,
    ) as audio_file:
        for start in range(0, len(waveform), AUDIO_WRITE_CHUNK_FRAMES):
            audio_file.write(waveform[start : start + AUDIO_WRITE_CHUNK_FRAMES])


def normalize_audio(artifact):
    """Convert an audio artifact into waveforms soundfile can write.

    Pipelines return audio as torch tensors or numpy arrays, channels first and
    optionally batched. soundfile wants samples first, one waveform at a time.

    Args:
        artifact: Audio waveform(s) as a torch tensor or numpy array, or anything
            carrying one as '.audio' - an AudioTrack, or a video whose soundtrack
            is what is being saved

    Returns:
        List of numpy arrays shaped (samples,) or (samples, channels)
    """
    if hasattr(artifact, "audio"):
        if artifact.audio is None:
            raise ValueError(
                f"Cannot save a {type(artifact).__name__} as audio - it carries no audio track"
            )
        artifact = artifact.audio

    # Torch tensors may be on the GPU and in a dtype numpy does not understand
    if hasattr(artifact, "detach"):
        artifact = artifact.detach().float().cpu().numpy()

    waveform = numpy.asarray(artifact)

    if waveform.ndim == 1:
        return [waveform]

    if waveform.ndim == 2:
        # Channels first - a waveform always has far more samples than channels
        if waveform.shape[0] < waveform.shape[1]:
            waveform = waveform.T
        return [waveform]

    if waveform.ndim == 3:
        # (batch, channels, samples) - one waveform per batch item
        return [item.T for item in waveform]

    raise ValueError(f"Cannot save audio with shape {waveform.shape}")


def as_audio_track(audio):
    """Convert a generated waveform into the tensor encode_video expects.

    encode_video wants a float torch tensor on the CPU shaped (channels, samples) -
    pipelines hand back bfloat16 tensors that are still on the GPU, or numpy arrays.

    A mono track is duplicated into two channels, because the mp4 audio stream
    takes nothing else: diffusers' _write_audio refuses any other channel count
    with a raw tensor shape, and a mono voice track paired onto a picture is the
    ordinary case, not an edge one (#106). Duplicating one channel is lossless,
    but it is a change to what was handed in, so it warns.

    Args:
        audio: Waveform as a torch tensor or numpy array

    Returns:
        Float CPU torch tensor holding the waveform, with at least 2 channels
    """
    if not isinstance(audio, torch.Tensor):
        audio = torch.from_numpy(numpy.asarray(audio))

    audio = audio.detach().float().cpu()
    return _as_stereo(audio)


def _as_stereo(audio):
    """Duplicate a mono waveform into two channels, leaving anything else alone.

    Orientation is read the way encode_video reads it: (samples,) and (1, samples)
    are mono, and so is (samples, 1) - a lone channel laid out samples-first.
    A 2-channel waveform in either orientation is already what the encoder takes
    and is handed back untouched, so a short stereo track is never mistaken for
    many channels of one sample.
    """
    if audio.ndim == 2:
        if audio.shape[0] == 2 or audio.shape[1] == 2:
            return audio
        if audio.shape[0] == 1:
            audio = audio[0]
        elif audio.shape[1] == 1:
            audio = audio[:, 0]
        else:
            return audio
    elif audio.ndim != 1:
        return audio

    emit_warning(
        f"Duplicating a mono audio track of {audio.shape[0]} samples into two "
        f"channels - an mp4 audio stream takes stereo and nothing else"
    )
    return audio.unsqueeze(0).repeat(2, 1)


def read_embedded_metadata(path):
    """The generation metadata a saved image carries, or None.

    The read-side mirror of embed_image_metadata: the 'parameters' PNG
    text chunk, or the EXIF UserComment for JPEG/WebP. Returns the parsed
    dict, or None when the file has no metadata this writer produced.
    """
    try:
        from PIL import Image

        with Image.open(path) as image:
            # image.text would load() the whole image to reach chunks after
            # IDAT; the writer puts its chunk before IDAT, where info holds it
            if image.width * image.height > MAX_DECODE_PIXELS:
                return None
            text = image.info.get("parameters")
            if text is None and "exif" in getattr(image, "info", {}):
                import piexif
                import piexif.helper

                exif = piexif.load(image.info["exif"])
                comment = exif.get("Exif", {}).get(piexif.ExifIFD.UserComment)
                if comment:
                    text = piexif.helper.UserComment.load(comment)
        if text is None:
            return None
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except Exception as e:
        logger.debug(f"No readable metadata in {path}: {e}")
        return None


def embed_image_metadata(image, output_path, content_type, metadata):
    """Save an image with embedded generation metadata.

    Args:
        image: PIL Image to save
        output_path: Path to save the image to
        content_type: MIME type of the image
        metadata: The generation parameters to embed
    """
    metadata_json = json.dumps(metadata, default=str)

    if content_type == "image/png":
        from PIL.PngImagePlugin import PngInfo

        png_info = PngInfo()
        png_info.add_text("parameters", metadata_json)
        image.save(output_path, pnginfo=png_info)
        logger.debug(f"Embedded PNG metadata in {output_path}")
    elif content_type in ("image/jpeg", "image/webp"):
        try:
            import piexif
            import piexif.helper

            exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}}
            if hasattr(image, "info") and "exif" in image.info:
                exif_dict = piexif.load(image.info["exif"])
            exif_dict["Exif"][piexif.ExifIFD.UserComment] = (
                piexif.helper.UserComment.dump(metadata_json)
            )
            exif_bytes = piexif.dump(exif_dict)
            image.save(output_path, exif=exif_bytes)
            logger.debug(f"Embedded EXIF metadata in {output_path}")
        except ImportError:
            logger.warning(
                "piexif not installed - saving without metadata. "
                "Install with: pip install piexif"
            )
            image.save(output_path)
    else:
        image.save(output_path)
