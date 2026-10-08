import logging

from .. import resolve_device
from ..events import emit_log
from ..media_types import AudioVideo
from .qr_code import get_qrcode_image
from .image_utils import process_image
from .video_utils import process_video
from .gather import gather_images, gather_inputs, gather_videos
from .format_messages import (
    format_chat_message,
    batch_decode_post_process,
    get_dict_value,
)

# The registry lives in its own module so a task module can register its own
# handler (`beats`) without a cycle through this one; re-exported here under
# the names the rest of dw and the tests read
from .registry import _COMMAND_INFO, _COMMAND_REGISTRY, register_command  # noqa: F401
from ..task_domains import (
    AT_LEAST_ONE,
    CHANNEL_LEVEL,
    CLOSED_UNIT,
    FIT_MODES,
    INGREDIENTS_FITS,
    INGREDIENTS_LAYOUTS,
    JOIN_WINDOWS_CURVES,
    NON_NEGATIVE,
    NON_POSITIVE,
    POSITIVE,
    SEED,
    UNIT,
    face_track_errors,
    fit_to_model_errors,
    ingredients_grid_errors,
    join_windows_errors,
    lut_errors,
    paste_face_track_errors,
    script_lines_errors,
    slice_audio_errors,
    window_video_errors,
)
from . import beats  # noqa: F401 - registers analyze_beats
from . import cuts  # noqa: F401 - registers plan_cuts
from . import trim  # noqa: F401 - registers trim_video

# The model-backed handlers (upscale, restore_faces, segment, interpolate_frames,
# image_to_text, text_generation, diffusion_upscale) are imported inside their
# handlers - at module scope their transformers/model imports add seconds to
# every startup for workflows that never run those tasks

logger = logging.getLogger("dw")


def task_command_info(command_name):
    """Where a task command's argument schema lives: a dict with 'kind'
    ('command', 'image_processor' or 'video_processor'), 'implementation'
    (dotted path or None for free-form), 'provided', and 'returns'
    ('artifact', the default, or 'scalar' for a bare-number command like
    `judge` - missing entirely for an image/video processor, which is
    always artifact-shaped). Raises ValueError for a name that is not a
    task command at all."""
    info = _COMMAND_INFO.get(command_name)
    if info is not None:
        return info
    if command_name in _VIDEO_PROCESSOR_INFO:
        return _VIDEO_PROCESSOR_INFO[command_name]
    from .image_utils import available_processors

    if command_name in available_processors():
        return {"kind": "image_processor", "implementation": None, "provided": ()}
    raise ValueError(f"Unknown task command: '{command_name}'")


# Command handler functions
@register_command("qr_code", implementation="dw.tasks.qr_code.get_qrcode_image")
def _handle_qr_code(task, arguments, previous_pipelines):
    """Generate QR code image"""
    logger.debug("Generating QR code")
    return get_qrcode_image(**arguments)


@register_command("gather_images", implementation="dw.tasks.gather.gather_images")
def _handle_gather_images(task, arguments, previous_pipelines):
    """Gather multiple images"""
    logger.debug("Gathering images")
    return gather_images(**arguments)


@register_command("gather_videos", implementation="dw.tasks.gather.gather_videos")
def _handle_gather_videos(task, arguments, previous_pipelines):
    """Gather multiple videos"""
    logger.debug("Gathering videos")
    return gather_videos(**arguments)


# gather_inputs passes its whole dict through unchanged - free-form by design
@register_command("gather_inputs")
def _handle_gather_inputs(task, arguments, previous_pipelines):
    """Gather inputs from various sources"""
    logger.debug("Gathering inputs")
    return gather_inputs(arguments)


@register_command(
    "concat_videos",
    implementation="dw.tasks.concat_videos.concat_videos",
    domains={
        "trim_frames": NON_NEGATIVE,
        "crossfade_ms": NON_NEGATIVE,
        "audio_bleed_ms": NON_NEGATIVE,
        "seam_fade_ms": NON_NEGATIVE,
        "fps": POSITIVE,
        "sample_rate": POSITIVE,
    },
)
def _handle_concat_videos(task, arguments, previous_pipelines):
    """Concatenate videos - and the audio generated with them - into one"""
    logger.debug("Concatenating videos")
    from .concat_videos import concat_videos

    return concat_videos(**arguments)


@register_command(
    "dissolve_videos",
    implementation="dw.tasks.dissolve_videos.dissolve_videos",
    domains={
        "dissolve_frames": NON_NEGATIVE,
        "fade_in_frames": NON_NEGATIVE,
        "fade_out_frames": NON_NEGATIVE,
        "fps": POSITIVE,
    },
)
def _handle_dissolve_videos(task, arguments, previous_pipelines):
    """Join videos with cross-dissolves, fading the whole from and to a colour"""
    logger.debug("Dissolving videos")
    from .dissolve_videos import dissolve_videos

    return dissolve_videos(**arguments)


@register_command(
    "join_into_song",
    implementation="dw.tasks.join_into_song.join_into_song",
    domains={
        "cue_seconds": NON_NEGATIVE,
        "duck_delay_ms": NON_NEGATIVE,
        "duck_db": NON_POSITIVE,
        "duck_ramp_ms": NON_NEGATIVE,
        "fps": POSITIVE,
    },
)
def _handle_join_into_song(task, arguments, previous_pipelines):
    """Join dialogue shots and song shots into one video over the unbroken song"""
    logger.debug("Joining dialogue into a song")
    from .join_into_song import join_into_song

    return join_into_song(**arguments)


@register_command(
    "fade_audio",
    implementation="dw.tasks.audio_utils.fade_audio",
    domains={
        "fade_in_ms": NON_NEGATIVE,
        "fade_out_ms": NON_NEGATIVE,
        "sample_rate": POSITIVE,
    },
)
def _handle_fade_audio(task, arguments, previous_pipelines):
    """Fade an audio track in from silence and out to it"""
    logger.debug("Fading audio")
    from .audio_utils import fade_audio

    return fade_audio(**arguments)


@register_command(
    "normalize_audio",
    implementation="dw.tasks.audio_dynamics.normalize_audio",
    domains={"sample_rate": POSITIVE, "target_lufs": NON_POSITIVE},
)
def _handle_normalize_audio(task, arguments, previous_pipelines):
    """Scale an audio track so its peak sits at a given level"""
    logger.debug("Normalizing audio")
    from .audio_dynamics import normalize_audio

    return normalize_audio(**arguments)


@register_command(
    "slice_audio",
    implementation="dw.tasks.audio_utils.slice_audio",
    domains={
        "start_seconds": NON_NEGATIVE,
        "duration_seconds": POSITIVE,
        "start_frame": NON_NEGATIVE,
        "lead_frames": SEED,
        "num_frames": POSITIVE,
        "fps": POSITIVE,
        "sample_rate": POSITIVE,
    },
    static_check=slice_audio_errors,
)
def _handle_slice_audio(task, arguments, previous_pipelines):
    """Cut a time- or frame-aligned slice out of an audio track"""
    logger.debug("Slicing audio")
    from .audio_utils import slice_audio

    return slice_audio(**arguments)


@register_command(
    "gain_audio",
    implementation="dw.tasks.audio_utils.gain_audio",
    domains={
        "start_seconds": NON_NEGATIVE,
        "duration_seconds": POSITIVE,
        "start_frame": NON_NEGATIVE,
        "num_frames": POSITIVE,
        "fps": POSITIVE,
        "sample_rate": POSITIVE,
    },
)
def _handle_gain_audio(task, arguments, previous_pipelines):
    """Apply a gain to a time- or frame-aligned region of an audio track"""
    logger.debug("Gaining audio region")
    from .audio_utils import gain_audio

    return gain_audio(**arguments)


@register_command(
    "resample_audio",
    implementation="dw.tasks.audio_utils.resample_audio",
    domains={"target_sample_rate": POSITIVE, "sample_rate": POSITIVE},
)
def _handle_resample_audio(task, arguments, previous_pipelines):
    """Resample an audio track to a different sample rate"""
    logger.debug("Resampling audio")
    from .audio_utils import resample_audio

    return resample_audio(**arguments)


@register_command("video_frames", implementation="dw.tasks.video_utils.frames_as_array")
def _handle_video_frames(task, arguments, previous_pipelines):
    """The frames of a generated video, as one array a later step can condition on"""
    logger.debug("Extracting video frames")
    from .video_utils import frames_as_array

    return frames_as_array(**arguments)


@register_command(
    "loop_frames",
    implementation="dw.tasks.video_utils.loop_frames",
    domains={"num_frames": POSITIVE},
)
def _handle_loop_frames(task, arguments, previous_pipelines):
    """Repeat a still or a short clip into a run of a given length"""
    logger.debug("Looping frames")
    from .video_utils import loop_frames

    return loop_frames(**arguments)


@register_command(
    "window_video",
    implementation="dw.tasks.windows.window_video",
    domains={
        "index": NON_NEGATIVE,
        "num_frames": POSITIVE,
        "overlap": NON_NEGATIVE,
        "fps": POSITIVE,
    },
    static_check=window_video_errors,
)
def _handle_window_video(task, arguments, previous_pipelines):
    """Cut one overlapping, fixed-length window out of a long video"""
    logger.debug("Cutting a video window")
    from .windows import window_video

    return window_video(**arguments)


@register_command(
    "fit_to_model",
    implementation="dw.tasks.fit.fit_to_model",
    summary=(
        "Fit a video into a model's working size and frame count - letterbox, "
        "stretch or crop - with a record restore_to_source reads to undo it."
    ),
    parameter_descriptions={
        "video": (
            "The source - an earlier step's video, or an asset:/output: "
            "reference. Its fps is kept; its soundtrack is not."
        ),
        "width": "The model's working width in pixels.",
        "height": "The model's working height in pixels.",
        "num_frames": (
            "The model's frame count. A longer source is cut to its first "
            "num_frames frames; a shorter one holds its last frame."
        ),
        "mode": (
            "letterbox (scale to fit, centred on black), stretch (resize to "
            "fill exactly) or crop (scale to fill, centre-crop)."
        ),
        "downscale": (
            "Fit into width/downscale x height/downscale instead, default 1 - "
            "2 when width/height are a 2x model's output size. Both must be "
            "divisible by it; the fit record's model size is the divided one."
        ),
    },
    domains={
        "width": POSITIVE,
        "height": POSITIVE,
        "num_frames": POSITIVE,
        "downscale": POSITIVE,
    },
    choices={"mode": FIT_MODES},
    static_check=fit_to_model_errors,
)
def _handle_fit_to_model(task, arguments, previous_pipelines):
    """Fit a video into a model's working size and frame count"""
    logger.debug("Fitting a video to the model")
    from .fit import fit_to_model

    return fit_to_model(**arguments)


@register_command(
    "restore_to_source",
    implementation="dw.tasks.fit.restore_to_source",
    summary=(
        "Put a model's output for a fit_to_model video back at the source's "
        "size (times the model's scale, inferred) and frame count."
    ),
    parameter_descriptions={
        "video": (
            "The model's output for the fitted video, at the fitted size or a "
            "uniform multiple of it - 2x restores to twice the source."
        ),
        "fit": (
            "The record fit_to_model returned - previous_result:<step>.fit, or "
            "its saved .json. A crop fit restores only the part it kept, at "
            "the source's pixel density."
        ),
    },
)
def _handle_restore_to_source(task, arguments, previous_pipelines):
    """Put a fitted video back at its source's size and length"""
    logger.debug("Restoring a video to its source")
    from .fit import restore_to_source

    return restore_to_source(**arguments)


@register_command(
    "join_windows",
    implementation="dw.tasks.windows.join_windows",
    domains={"num_frames": POSITIVE, "overlap": NON_NEGATIVE, "fps": POSITIVE},
    choices={"curve": JOIN_WINDOWS_CURVES},
    static_check=join_windows_errors,
    media_arguments=("source",),
)
def _handle_join_windows(task, arguments, previous_pipelines):
    """Blend processed overlapping windows back into one video the source's length"""
    logger.debug("Joining video windows")
    from .windows import join_windows

    return join_windows(**arguments)


@register_command(
    "frame_grid",
    implementation="dw.tasks.video_utils.frame_grid",
    domains={"count": POSITIVE, "columns": POSITIVE, "tile_width": POSITIVE},
)
def _handle_frame_grid(task, arguments, previous_pipelines):
    """Tile evenly sampled frames of a video into one contact-sheet image"""
    logger.debug("Building frame grid")
    from .video_utils import frame_grid

    return frame_grid(**arguments)


@register_command(
    "ingredients_grid",
    implementation="dw.tasks.image_utils.ingredients_grid",
    domains={
        "width": POSITIVE,
        "height": POSITIVE,
        "gap": NON_NEGATIVE,
        "max_images": POSITIVE,
    },
    choices={"layout": INGREDIENTS_LAYOUTS, "fit": INGREDIENTS_FITS},
    static_check=ingredients_grid_errors,
)
def _handle_ingredients_grid(task, arguments, previous_pipelines):
    """Lay individual images out as one reference sheet"""
    logger.debug("Building ingredients grid")
    from .image_utils import ingredients_grid

    return ingredients_grid(**arguments)


@register_command(
    "pair_audio",
    implementation="dw.tasks.pair_audio.pair_audio",
    domains={"sample_rate": POSITIVE},
)
def _handle_pair_audio(task, arguments, previous_pipelines):
    """Pair a video's frames with an audio track generated beside them"""
    logger.debug("Pairing audio with video")
    from .pair_audio import pair_audio

    return pair_audio(**arguments)


@register_command(
    "crossfade_audio",
    implementation="dw.tasks.audio_utils.crossfade_audio",
    domains={"crossfade_ms": NON_NEGATIVE, "sample_rate": POSITIVE},
)
def _handle_crossfade_audio(task, arguments, previous_pipelines):
    """Join audio tracks with an equal-power crossfade"""
    logger.debug("Crossfading audio")
    from .audio_utils import crossfade_audio

    return crossfade_audio(**arguments)


@register_command(
    "loop_audio",
    implementation="dw.tasks.audio_utils.loop_audio",
    domains={
        "duration_seconds": POSITIVE,
        "target_frames": POSITIVE,
        "fps": POSITIVE,
        "crossfade_ms": NON_NEGATIVE,
        "sample_rate": POSITIVE,
    },
)
def _handle_loop_audio(task, arguments, previous_pipelines):
    """Loop a short recording into a bed of a given length"""
    logger.debug("Looping audio")
    from .audio_utils import loop_audio

    return loop_audio(**arguments)


@register_command(
    "find_loop_bed",
    implementation="dw.tasks.loop_bed.find_loop_bed",
    returns="json",
    domains={
        "start_seconds": NON_NEGATIVE,
        "end_seconds": POSITIVE,
        "min_seconds": POSITIVE,
        "max_seconds": POSITIVE,
        "max_bin_dbfs": NON_POSITIVE,
        "max_mean_dbfs": NON_POSITIVE,
        "max_spike_db": NON_NEGATIVE,
        "crossfade_ms": NON_NEGATIVE,
        "loop_seconds": POSITIVE,
        "target_bed_dbfs": NON_POSITIVE,
        "max_candidates": POSITIVE,
        "fps": POSITIVE,
    },
)
def _handle_find_loop_bed(task, arguments, previous_pipelines):
    """Rank the quiet windows of a recording worth looping into a room-tone bed"""
    logger.debug("Searching for a loop bed")
    from .loop_bed import find_loop_bed

    return find_loop_bed(**arguments)


@register_command(
    "stabilize_video", implementation="dw.tasks.stabilize.stabilize_video"
)
def _handle_stabilize_video(task, arguments, previous_pipelines):
    """Remove a generated clip's accumulated framing drift"""
    logger.debug("Stabilizing video")
    from .stabilize import stabilize_video

    return stabilize_video(**arguments)


@register_command(
    "crop_face_track",
    implementation="dw.tasks.face_track.crop_face_track",
    consumes_device=True,
    summary=(
        "Follow one face through a clip and crop a steady square around it, "
        "with a per-frame record of the box and a face-detail strength."
    ),
    parameter_descriptions={
        "clip": (
            "Video to track - a file path, an asset:/output: reference, or an "
            "earlier step's video. A clip that records its shots resets the "
            "track at each boundary."
        ),
        "crop_size": "Side of every crop in pixels; a multiple of 32.",
        "padding": (
            "Space added around the face on each side, as a fraction of its "
            "size (0 to 3)."
        ),
        "gate_full": (
            "Face width over frame width at or below which a frame's strength is 1."
        ),
        "gate_zero": (
            "Face width over frame width at or above which a frame's strength "
            "is 0; must exceed gate_full."
        ),
        "min_confidence": "Detector score below which a detection is ignored.",
        "detector_repo": "Hugging Face repo holding the YuNet face detector.",
        "detector_file": "The .onnx file in detector_repo.",
    },
    domains={
        "crop_size": POSITIVE,
        "padding": NON_NEGATIVE,
        "gate_full": POSITIVE,
        "gate_zero": POSITIVE,
        "min_confidence": POSITIVE,
    },
    static_check=face_track_errors,
)
def _handle_crop_face_track(task, arguments, previous_pipelines):
    """Crop a steady square around the one face a clip follows"""
    logger.debug("Tracking a face")
    from .face_track import crop_face_track

    return crop_face_track(device=task.device_for(arguments), **arguments)


@register_command(
    "paste_face_track",
    implementation="dw.tasks.face_track.paste_face_track",
    summary=(
        "Blend repaired face crops back into the clip crop_face_track tracked, "
        "feathered and scaled by each frame's strength, keeping its audio."
    ),
    parameter_descriptions={
        "clip": (
            "The source video crop_face_track tracked - a file path, an "
            "asset:/output: reference, or an earlier step's video. Its audio, "
            "frame rate and shots are kept."
        ),
        "repaired": (
            "The crops after a face-detail pass, still padded to the 8n+1 "
            "count crop_face_track produced."
        ),
        "track": (
            "The track record crop_face_track returned - "
            "previous_result:<step>.track, or its saved .json."
        ),
        "feather": "Fraction of the paste's radius that fades out (0 to 1).",
        "color_match": (
            "Match each crop's mean colour to the source inside the mask "
            "before blending."
        ),
    },
    domains={"feather": NON_NEGATIVE},
    static_check=paste_face_track_errors,
)
def _handle_paste_face_track(task, arguments, previous_pipelines):
    """Blend repaired face crops back into the clip they were cut from"""
    logger.debug("Pasting a face track")
    from .face_track import paste_face_track

    return paste_face_track(**arguments)


@register_command(
    "mix_audio",
    implementation="dw.tasks.audio_utils.mix_audio",
    domains={"gains": NON_NEGATIVE, "sample_rate": POSITIVE},
)
def _handle_mix_audio(task, arguments, previous_pipelines):
    """Layer audio tracks on top of one another, rather than end to end"""
    logger.debug("Mixing audio")
    from .audio_utils import mix_audio

    return mix_audio(**arguments)


@register_command(
    "compress_audio",
    implementation="dw.tasks.audio_dynamics.compress_audio",
    domains={
        "ratio": POSITIVE,
        "attack_ms": NON_NEGATIVE,
        "release_ms": NON_NEGATIVE,
        "sample_rate": POSITIVE,
    },
)
def _handle_compress_audio(task, arguments, previous_pipelines):
    """Shape a track's dynamics with a compressor, limiter or gate"""
    logger.debug("Compressing audio")
    from .audio_dynamics import compress_audio

    return compress_audio(**arguments)


@register_command(
    "filter_audio",
    implementation="dw.tasks.audio_dynamics.filter_audio",
    domains={"cutoff_hz": POSITIVE, "sample_rate": POSITIVE},
)
def _handle_filter_audio(task, arguments, previous_pipelines):
    """Run a track through a single lowpass/highpass/bandpass/notch filter"""
    logger.debug("Filtering audio")
    from .audio_dynamics import filter_audio

    return filter_audio(**arguments)


@register_command(
    "analyze_audio",
    implementation="dw.tasks.audio_dynamics.analyze_audio",
    domains={"sample_rate": POSITIVE},
)
def _handle_analyze_audio(task, arguments, previous_pipelines):
    """Measure a track's levels and spectral balance without changing it"""
    logger.debug("Analyzing audio")
    from .audio_dynamics import analyze_audio

    return analyze_audio(**arguments)


@register_command(
    "analyze_shots",
    implementation="dw.tasks.assess.analyze_shots",
    returns="json",
    assessment=True,
)
def _handle_analyze_shots(task, arguments, previous_pipelines):
    """Measure each shot of a cut's soundtrack and how far apart they sit"""
    from .assess import analyze_shots

    return analyze_shots(**arguments)


@register_command(
    "analyze_seams",
    implementation="dw.tasks.assess.analyze_seams",
    returns="json",
    assessment=True,
)
def _handle_analyze_seams(task, arguments, previous_pipelines):
    """Measure every seam of a cut - level step, hole, click, frame jump"""
    from .assess import analyze_seams

    return analyze_seams(**arguments)


@register_command(
    "analyze_sync_drift",
    implementation="dw.tasks.assess.analyze_sync_drift",
    returns="json",
    assessment=True,
)
def _handle_analyze_sync_drift(task, arguments, previous_pipelines):
    """Measure how far a cut's soundtrack sits from its picture"""
    from .assess import analyze_sync_drift

    return analyze_sync_drift(**arguments)


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


@register_command("compose_text", implementation="dw.tasks.compose_text.compose_text")
def _handle_compose_text(task, arguments, previous_pipelines):
    """Join parts written once into one block of text"""
    logger.debug("Composing text")
    from .compose_text import compose_text

    return compose_text(**arguments)


@register_command("select", implementation="dw.tasks.select.select")
def _handle_select(task, arguments, previous_pipelines):
    """Reduce a list of candidates to one by a deterministic rule"""
    logger.debug("Selecting")
    from .select import select

    return select(**arguments)


@register_command(
    "format_chat_message", implementation="dw.tasks.format_messages.format_chat_message"
)
def _handle_format_chat_message(task, arguments, previous_pipelines):
    """Format chat message for LLM input"""
    logger.debug("Formatting chat message")
    return format_chat_message(**arguments)


@register_command(
    "get_dict_value", implementation="dw.tasks.format_messages.get_dict_value"
)
def _handle_get_dict_value(task, arguments, previous_pipelines):
    """Extract value from dictionary"""
    logger.debug("Getting dictionary value")
    return get_dict_value(**arguments)


def _per_frame(image, process):
    """Run an image command over a video, frame by frame.

    A video bound to an image argument - an AudioVideo from a generation,
    concat or dissolve step, or a frame array from video_frames - is processed
    one frame at a time and comes back as one video artifact, its soundtrack
    carried through untouched. A single image is processed as itself.
    """
    from ..shots import carried_shots
    from .video_utils import frames_as_pil_list, is_video

    if not is_video(image):
        return process(image)
    frames = [process(frame) for frame in frames_as_pil_list(image)]
    audio = getattr(image, "audio", None)
    sample_rate = getattr(image, "sample_rate", None)
    # One frame out per frame in, so the shot boundaries carry through too
    return AudioVideo(
        frames,
        audio,
        sample_rate,
        fps=getattr(image, "fps", None),
        shots=carried_shots(image),
    )


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
    return _per_frame(
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
    return _per_frame(
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
    return _per_frame(
        image,
        lambda frame: restore_faces(frame, model_name, device=device, **arguments),
    )


def _load_media(media):
    """An image-or-video argument as something _per_frame takes.

    A string is a file path (an asset:/output: reference already resolved):
    a video extension is read with its audio, anything else as an image. A
    value that is not a string - a PIL Image, an AudioVideo, a frame list -
    is already loaded and comes back as itself. Shared by the finishing
    commands that take `media` (#603).
    """
    if not isinstance(media, str):
        return media
    import os

    from ..security import ALLOWED_VIDEO_EXTENSIONS
    from .video_utils import load_audio_video

    if os.path.splitext(media)[1].lower() in ALLOWED_VIDEO_EXTENSIONS:
        return load_audio_video(media)
    from ..argument_media import fetch_image

    return fetch_image(media)


@register_command(
    "grade",
    implementation="dw.tasks.grade.grade_image",
    summary=(
        "Adjust exposure, contrast, tonal range (highlights, shadows, whites, "
        "blacks), clarity, white balance, saturation, fade and vignette of "
        "an image or a video."
    ),
    parameter_descriptions={
        "media": (
            "Image or video to grade. An image is a PIL Image; a video is a "
            "file path or an asset:/output: reference, read with its audio "
            "and graded frame by frame, keeping its frame rate and audio "
            "unchanged."
        ),
    },
    domains={
        "contrast": NON_NEGATIVE,
        "saturation": NON_NEGATIVE,
        "temperature": CLOSED_UNIT,
        "tint": CLOSED_UNIT,
        "highlights": CLOSED_UNIT,
        "shadows": CLOSED_UNIT,
        "whites": CLOSED_UNIT,
        "blacks": CLOSED_UNIT,
        "clarity": CLOSED_UNIT,
        "vignette": CLOSED_UNIT,
        "fade": UNIT,
    },
    media_arguments=("media",),
)
def _handle_grade(task, arguments, previous_pipelines):
    """Adjust the exposure, tone, white balance and colour of an image or a video"""
    logger.debug("Grading media")
    media = _load_media(arguments.pop("media"))
    import inspect

    from ..task_domains import check_arguments
    from .grade import grade_image

    # A value from a variable or an earlier step never met the static pass
    check_arguments("grade", **arguments)
    # Read off the signature, so a parameter added there cannot drop out of
    # the log (#603)
    defaults = {
        name: parameter.default
        for name, parameter in inspect.signature(grade_image).parameters.items()
        if parameter.default is not inspect.Parameter.empty
    }
    applied = {
        name: arguments.get(name, default)
        for name, default in defaults.items()
        if arguments.get(name, default) != default
    }
    emit_log(
        f"grade: applied {applied}"
        if applied
        else "grade: no adjustment (all identity)",
        command="grade",
        **applied,
    )
    return _per_frame(media, lambda frame: grade_image(frame, **arguments))


_MEDIA_DESCRIPTION = (
    "Image or video. An image is a PIL Image; a video is a file path or an "
    "asset:/output: reference, read with its audio and processed frame by "
    "frame, keeping its frame count, frame rate and audio unchanged."
)


@register_command(
    "sharpen",
    implementation="dw.tasks.finish.sharpen_image",
    summary="Sharpen an image or a video with an unsharp mask.",
    parameter_descriptions={"media": _MEDIA_DESCRIPTION},
    domains={"amount": NON_NEGATIVE, "radius": POSITIVE, "threshold": CHANNEL_LEVEL},
    media_arguments=("media",),
)
def _handle_sharpen(task, arguments, previous_pipelines):
    """Sharpen an image or a video with an unsharp mask"""
    logger.debug("Sharpening media")
    media = _load_media(arguments.pop("media"))
    from ..task_domains import check_arguments
    from .finish import sharpen_image

    # A value from a variable or an earlier step never met the static pass
    check_arguments("sharpen", **arguments)
    return _per_frame(media, lambda frame: sharpen_image(frame, **arguments))


@register_command(
    "film_grain",
    implementation="dw.tasks.finish.film_grain",
    summary=(
        "Add seeded film grain to an image or a video, strongest in the "
        "midtones; every video frame gets different grain."
    ),
    parameter_descriptions={
        "media": _MEDIA_DESCRIPTION,
        "seed": (
            "The grain's seed. Defaults to the workflow's or step's seed, "
            "which a run always has (a random one when the workflow names "
            "none, recorded in the manifest), so a rerun reproduces the grain."
        ),
    },
    domains={"amount": UNIT, "size": AT_LEAST_ONE, "chroma": UNIT, "seed": SEED},
    media_arguments=("media",),
)
def _handle_film_grain(task, arguments, previous_pipelines):
    """Add film grain to an image or a video, reproducibly from a seed"""
    logger.debug("Adding film grain")
    media = _load_media(arguments.pop("media"))
    seed = task.seed_for(arguments)
    from ..task_domains import check_arguments
    from .finish import film_grain

    check_arguments("film_grain", seed=seed, **arguments)
    emit_log(f"film_grain: seed {seed}", command="film_grain", seed=seed)
    return film_grain(media, seed=seed, **arguments)


@register_command(
    "apply_lut",
    implementation="dw.tasks.lut.apply_lut",
    summary=(
        "Colour an image or a video through a 3D lookup table - from a .cube "
        "file or built from a palette - blended with the original by strength."
    ),
    parameter_descriptions={"media": _MEDIA_DESCRIPTION},
    domains={"strength": UNIT},
    static_check=lut_errors,
    media_arguments=(
        "media",
        "lut",
    ),
)
def _handle_apply_lut(task, arguments, previous_pipelines):
    """Apply a .cube or palette 3D lookup table to an image or a video"""
    logger.debug("Applying a LUT")
    media = _load_media(arguments.pop("media"))
    from ..task_domains import check_arguments
    from .lut import apply_lut, lookup_for

    lut = arguments.pop("lut", None)
    palette = arguments.pop("palette", None)
    # A value from a variable or an earlier step never met the static pass
    check_arguments("apply_lut", **arguments)
    # Read or built once, not once per video frame
    lookup = lookup_for(lut, palette)
    return _per_frame(media, lambda frame: apply_lut(frame, lookup, **arguments))


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
    return _per_frame(
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


@register_command(
    "extract_sections", implementation="dw.tasks.text_sections.extract_sections"
)
def _handle_extract_sections(task, arguments, previous_pipelines):
    """Reduce generated text to a known set of labelled sections"""
    logger.debug("Extracting sections")
    from .text_sections import extract_sections

    return extract_sections(**arguments)


@register_command(
    "batch_decode_post_process",
    implementation="dw.tasks.format_messages.batch_decode_post_process",
    provided=("processor",),
)
def _handle_batch_decode(task, arguments, previous_pipelines):
    """Batch decode post-processing with pipeline reference"""
    logger.debug("Performing batch decode post-processing")
    pipeline_reference = task.task_definition["pipeline_reference"]
    if pipeline_reference not in previous_pipelines:
        raise KeyError(
            f"Pipeline reference '{pipeline_reference}' not found in previous pipelines. "
            f"Available pipelines: {list(previous_pipelines.keys())}"
        )
    processor = previous_pipelines[pipeline_reference].pipeline
    return batch_decode_post_process(processor, **arguments)


def _handle_image_processing(task, arguments, previous_pipelines):
    """Handle image processing commands"""
    logger.debug("Processing image")
    device = task.device_for(arguments)
    return _per_frame(
        arguments.pop("image"),
        lambda frame: process_image(frame, task.command, device, arguments),
    )


def _handle_video_processing(task, arguments, previous_pipelines):
    """Handle video processing commands"""
    logger.debug("Processing video")
    device = task.device_for(arguments)
    return process_video(
        arguments.pop("video"),
        task.command,
        device,
        arguments,
    )


# Command names process_video (video_utils.py) accepts, with the function
# whose signature carries their arguments. video_utils dispatches via a plain
# if-chain, so keep this in sync with the branches in process_video().
# get_first/last_frame pin frame_index themselves, so it is 'provided'; they
# share get_frame's implementation and so would share its generic docstring
# summary too (#366) - 'summary' overrides that per command.
_VIDEO_PROCESSOR_INFO = {
    "get_frame": {
        "kind": "video_processor",
        "implementation": "dw.tasks.video_utils.get_frame",
        "provided": (),
    },
    "get_first_frame": {
        "kind": "video_processor",
        "implementation": "dw.tasks.video_utils.get_frame",
        "provided": ("frame_index",),
        "summary": "The first frame of a video, as a PIL image.",
    },
    "get_last_frame": {
        "kind": "video_processor",
        "implementation": "dw.tasks.video_utils.get_frame",
        "provided": ("frame_index",),
        "summary": "The last frame of a video, as a PIL image.",
    },
}
_VIDEO_PROCESSOR_COMMANDS = sorted(_VIDEO_PROCESSOR_INFO)


class Task:
    """
    Represents a task that can be executed as part of a workflow.
    Tasks are atomic operations like image processing, data gathering, or message formatting.
    """

    def __init__(self, task_definition, device, seed=None):
        """
        Initialize task with its configuration and device settings.

        Args:
            task_definition: Dictionary containing task configuration and parameters
            device: Device to run task on (e.g., 'cuda', 'mps', 'cpu')
            seed: The workflow/step-resolved seed, when one was set - None for
                an unseeded run. Only a handler that calls seed_for(arguments)
                consumes it; most tasks run no generator and ignore it
        """
        self.task_definition = task_definition
        self.device = device
        self.seed = seed
        logger.debug(f"Initialized task: {self.name} for device: {device}")

    @property
    def name(self):
        """Get task name from command property"""
        return self.command

    def device_for(self, arguments):
        """Get the device this task runs on, consuming any override in its arguments.

        A task can pin itself to a device - a captioning model on the CPU while the GPU
        holds a pipeline, for instance. The argument is removed either way so it does
        not reach the command as a duplicate.

        Args:
            arguments: Arguments for this run of the task

        Returns:
            Device identifier the task should run on
        """
        return resolve_device(arguments.pop("device", self.device))

    def seed_for(self, arguments):
        """Get the seed this task run should use, consuming any override in
        its arguments.

        A task step reproducible the way a pipeline step is: the
        workflow/step-resolved seed by default, an explicit `seed` in the
        step's own arguments taking precedence - and the argument is removed
        either way so it does not reach the command as a duplicate. None
        means no seed was ever set anywhere, so the task should run exactly
        as it always did - unseeded and non-reproducible.

        Args:
            arguments: Arguments for this run of the task
        """
        return arguments.pop("seed", self.seed)

    @property
    def argument_template(self):
        """
        Get argument template for this task.

        Returns:
            Dictionary of arguments from inputs or arguments section
        """
        # A task will either be an input array or a dictionary of arguments
        if "inputs" in self.task_definition:
            logger.debug("Using inputs as argument template")
            return self.task_definition["inputs"]

        logger.debug("Using arguments as argument template")
        return self.task_definition["arguments"]

    @property
    def command(self):
        """Get command name or 'unknown' if not specified"""
        return self.task_definition.get("command", "unknown")

    def run(self, arguments, previous_pipelines={}):
        """
        Execute the task with given arguments using the command registry.

        Args:
            arguments: Dictionary of arguments for task execution
            previous_pipelines: Dictionary of previously created pipelines

        Returns:
            Task output based on command type

        Raises:
            ValueError: If command is unknown
            KeyError: If required arguments or pipeline references are missing
        """
        logger.debug(f"Running task: {self.command}")
        logger.debug(f"Task arguments: {arguments}")

        try:
            # Cooperative cancellation reaches task steps too - without this
            # a cancel during a long task waits for the whole task to finish
            from ..events import emit_phase, get_context

            get_context().check_cancelled()
            # A task reports nothing of its own - a captioning model loading
            # and decoding is otherwise indistinguishable from a hang
            emit_phase("task", detail=self.command)

            # Look up command in registry
            if self.command in _COMMAND_REGISTRY:
                handler = _COMMAND_REGISTRY[self.command]
                return handler(self, arguments, previous_pipelines)

            # Not a registered command - check whether it names an image or
            # video processor instead. Imported lazily here to preserve
            # image_utils' lazy-import discipline for callers that never
            # touch image processing.
            from .image_utils import available_processors

            if self.command in available_processors():
                return _handle_image_processing(self, arguments, previous_pipelines)

            if self.command in _VIDEO_PROCESSOR_COMMANDS:
                return _handle_video_processing(self, arguments, previous_pipelines)

            # Unknown command - not in the registry, and not a known image or
            # video processor name either
            error_msg = (
                f"Unknown task command: '{self.command}'. "
                f"Registered commands: {sorted(_COMMAND_REGISTRY.keys())}. "
                f"Image processors: {available_processors()}. "
                f"Video processors: {_VIDEO_PROCESSOR_COMMANDS}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        except KeyError as e:
            # Missing required arguments or pipeline references
            logger.error(
                f"Missing required data for task {self.command}: {e}", exc_info=True
            )
            raise
        except (ValueError, TypeError) as e:
            # Invalid arguments or type mismatches
            logger.error(
                f"Invalid arguments for task {self.command}: {e}", exc_info=True
            )
            raise
        except (OSError, IOError) as e:
            # File operations, resource loading errors
            logger.error(f"I/O error in task {self.command}: {e}", exc_info=True)
            raise
        except Exception as e:
            # Catch-all for unexpected errors
            logger.error(
                f"Unexpected error ({type(e).__name__}) executing task {self.command}: {e}",
                exc_info=True,
            )
            raise
