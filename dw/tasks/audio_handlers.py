"""The audio task commands' handlers, and the checks of a finished cut.

Split from `dw/tasks/task.py` (#790), which had grown past the module-size
ceiling with every handler in one file. These are the commands that shape or
measure a soundtrack - fade, gain, slice, mix, dynamics, loop beds - and the
assessment commands that measure a cut's audio against itself and its picture
(#485). None loads a model, so none consumes a device.

Each handler imports its implementation on use, as task.py's did: an audio
module pulls in numpy/scipy, a cost a workflow that never touches audio
should not pay at startup. `dw/tasks/task.py` imports this module, which is
what registers its commands (`docs/TASKS.md` *Adding a task*).
"""

import logging

from .registry import register_command
from ..task_domains import (
    FINITE,
    NON_NEGATIVE,
    NON_POSITIVE,
    POSITIVE,
    SEED,
    slice_audio_errors,
)

logger = logging.getLogger("dw")


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
    domains={
        "peak_dbfs": FINITE,
        "sample_rate": POSITIVE,
        "target_lufs": NON_POSITIVE,
    },
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
    whole_numbers=("start_frame", "num_frames", "lead_frames"),
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
        "gain_db": FINITE,
        "sample_rate": POSITIVE,
    },
    whole_numbers=("start_frame", "num_frames"),
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
    whole_numbers=("target_sample_rate",),
)
def _handle_resample_audio(task, arguments, previous_pipelines):
    """Resample an audio track to a different sample rate"""
    logger.debug("Resampling audio")
    from .audio_utils import resample_audio

    return resample_audio(**arguments)


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
    whole_numbers=("target_frames",),
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
    whole_numbers=("max_candidates",),
)
def _handle_find_loop_bed(task, arguments, previous_pipelines):
    """Rank the quiet windows of a recording worth looping into a room-tone bed"""
    logger.debug("Searching for a loop bed")
    from .loop_bed import find_loop_bed

    return find_loop_bed(**arguments)


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
        "threshold_dbfs": FINITE,
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
    domains={"cutoff_hz": POSITIVE, "q": FINITE, "sample_rate": POSITIVE},
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
