"""Join videos with cross-dissolves, and fade the whole from and to a colour.

`concat_videos` cuts: every seam is a hard edit, which is right for a scene
built from shots that each carry their own sound. A lyrical piece - a nature
film, a montage cut to a score - wants its shots to melt into one another
instead. Each seam here overlaps the tail of one video with the head of the
next by `dissolve_frames`, blending linearly across the overlap, so the result
is shorter than the plain sum by one overlap per seam; the soundtrack, when
every video carries one, is crossfaded over exactly the same span so it stays
in step with the picture. A `fade_in_frames` / `fade_out_frames` pair opens and
closes the piece on `fade_color`, which is what a film does instead of
starting on a full frame.
"""

import logging

import numpy
from PIL import Image

from ..events import emit_warning
from ..result import AudioVideo
from ..shots import shot_record
from .audio_utils import (
    as_channels_samples,
    crossfade_concat,
    match_levels as match_track_levels,
    resample_waveform,
    warn_on_level_spread,
)
from .concat_videos import video_names
from .video_utils import check_same_frame_size, frames_as_array, load_audio_video

logger = logging.getLogger("dw")


def dissolve_videos(
    videos,
    dissolve_frames=12,
    fade_in_frames=0,
    fade_out_frames=0,
    fade_color=(0, 0, 0),
    fps=None,
    match_levels=None,
    match_levels_dbfs=None,
    sample_rate=None,
):
    """Task command: join videos with cross-dissolves at every seam.

    Args:
        videos: The videos to join, in order - frame lists, frame arrays,
            AudioVideos, or the path or URL of a video file. Give each video
            its own entry, as with concat_videos. Soundtracks at different
            sample rates are not a constraint - see `sample_rate` below
        dissolve_frames: Frames of overlap at each seam. 0 is a hard cut
        fade_in_frames: Frames over which the first video rises out of
            `fade_color`
        fade_out_frames: Frames over which the last video sinks into it
        fade_color: The RGB colour the fades come from and go to
        fps: Frame rate of the videos - required to crossfade audio at a
            dissolve, and ignored when no video carries any
        match_levels: Even the shots' loudness out before joining - "rms"
            for perceived level, "peak" for the loudest sample. Off by
            default; see concat_videos, which has the same pair. Left off, a
            spread wide enough to hear is logged as a warning
        match_levels_dbfs: The level match_levels moves every shot to -
            defaults to -1 dBFS for "peak" and -20 dBFS for "rms". A shot
            that would clip at the target is held at -0.5 dBFS peak instead,
            reported as a match_levels_held warning, with a per-shot log
            event naming the hold
        sample_rate: The rate the joined soundtrack is at. Shots that come
            from different sources routinely carry different rates, and
            unlike a level jump that difference has no editorial meaning, so
            by default the highest rate among the inputs is chosen and the
            rest are resampled up to it, with a warning naming which - the
            same conversion concat_videos does (#108, #287). Give this to
            pin the target instead

    Returns:
        One AudioVideo; its audio is None unless every input carries a track

    Raises:
        ValueError: If a video is too short to carry its share of the overlaps
    """
    if not isinstance(videos, list) or not videos:
        raise ValueError("dissolve_videos needs a non-empty list of videos")
    if dissolve_frames < 0 or fade_in_frames < 0 or fade_out_frames < 0:
        raise ValueError("dissolve_videos frame counts cannot be negative")

    loaded = [load_audio_video(v) if isinstance(v, str) else v for v in videos]
    clips = [frames_as_array(v).astype(numpy.float32) for v in loaded]
    check_same_frame_size(clips, "dissolve_videos")

    for index, clip in enumerate(clips):
        seams = (index > 0) + (index < len(clips) - 1)
        if len(clip) < seams * dissolve_frames:
            raise ValueError(
                f"video {index} has {len(clip)} frames, too few for its "
                f"{seams} dissolve(s) of {dissolve_frames} frames"
            )

    joined = clips[0]
    # Where each clip's first frame landed - the start of its dissolve
    frame_starts = [0]
    for clip in clips[1:]:
        frame_starts.append(len(joined) - dissolve_frames)
        joined = _dissolve_join(joined, clip, dissolve_frames)

    if fade_in_frames or fade_out_frames:
        color = numpy.asarray(fade_color, dtype=numpy.float32)
        total = len(joined)
        fade_in = min(fade_in_frames, total)
        fade_out = min(fade_out_frames, total - fade_in)
        if fade_in:
            weights = _ramp(fade_in, ascending=True)
            joined[:fade_in] = _blend(color, joined[:fade_in], weights)
        if fade_out:
            weights = _ramp(fade_out, ascending=False)
            joined[total - fade_out :] = _blend(
                color, joined[total - fade_out :], weights
            )

    frames = [Image.fromarray(frame) for frame in joined.round().astype(numpy.uint8)]
    sample_starts = []
    audio, sample_rate = _dissolve_audio(
        loaded,
        dissolve_frames,
        fps,
        match_levels,
        match_levels_dbfs,
        sample_rate,
        sample_starts,
    )
    shots = _dissolve_shots(
        video_names(videos),
        frame_starts,
        len(frames),
        sample_starts,
        audio,
        dissolve_frames,
    )
    logger.info(
        f"Dissolved {len(clips)} videos into {len(frames)} frames "
        f"({dissolve_frames}-frame seams)"
    )
    written_fps = fps or next(
        (v.fps for v in loaded if getattr(v, "fps", None)),
        None,
    )
    return AudioVideo(frames, audio, sample_rate, fps=written_fps, shots=shots)


def _dissolve_shots(
    names, frame_starts, total_frames, sample_starts, audio, dissolve_frames
):
    """One shot per video, partitioning the dissolved picture and track.

    A dissolve belongs to the shot coming in: each shot runs from where its
    dissolve opens to where the next one's does, so the counts add up to the
    file's and `overlap_frames` says how much of its head is blended.
    """
    frame_ends = frame_starts[1:] + [total_frames]
    if audio is None:
        sample_starts = [None] * len(frame_starts)
        sample_ends = sample_starts
    else:
        sample_ends = sample_starts[1:] + [audio.shape[1]]
    shots = []
    for index, name in enumerate(names):
        start_sample = sample_starts[index]
        shots.append(
            shot_record(
                name,
                frame_starts[index],
                frame_ends[index] - frame_starts[index],
                start_sample,
                None if start_sample is None else sample_ends[index] - start_sample,
            )
        )
        if index and dissolve_frames:
            shots[-1]["overlap_frames"] = dissolve_frames
    return shots


def _dissolve_join(previous, following, overlap):
    """Overlap the tail of `previous` with the head of `following`."""
    if overlap == 0:
        return numpy.concatenate([previous, following])
    weights = _ramp(overlap, ascending=True)
    blended = _blend(previous[-overlap:], following[:overlap], weights)
    return numpy.concatenate([previous[:-overlap], blended, following[overlap:]])


def _ramp(count, ascending):
    """Blend weights that never sit on 0 or 1, so no frame is a bare copy of
    either side - the seam's first frame already carries some of the incoming
    picture and its last still carries some of the outgoing one."""
    weights = (numpy.arange(count, dtype=numpy.float32) + 1) / (count + 1)
    return weights if ascending else 1 - weights


def _blend(from_frames, to_frames, weights):
    weights = weights.reshape(-1, 1, 1, 1)
    return from_frames * (1 - weights) + to_frames * weights


def _dissolve_audio(
    videos,
    dissolve_frames,
    fps,
    match_levels=None,
    match_levels_dbfs=None,
    sample_rate=None,
    starts=None,
):
    """Crossfade every video's track over the seams' own span.

    `starts` is filled with where each track begins in the joined one.
    """
    tracks = [v for v in videos if isinstance(v, AudioVideo) and v.audio is not None]
    if len(tracks) != len(videos):
        if tracks:
            logger.warning(
                "dissolve_videos: some videos carry no audio - the result is silent"
            )
        return None, None
    if fps is None and dissolve_frames:
        raise ValueError("dissolve_videos needs 'fps' to crossfade audio at a dissolve")

    # Shots assembled from different sources disagree on rate routinely, and
    # the disagreement carries no editorial meaning - so it is converted
    # rather than refused, which is what made an agent invent a
    # resample_audio step by hand for concat_videos before #108 (#287)
    names = video_names(videos)
    track_names = [
        name
        for name, video in zip(names, videos)
        if isinstance(video, AudioVideo) and video.audio is not None
    ]
    rates = {v.sample_rate for v in tracks}
    sample_rate = sample_rate or max(rates)
    waveforms = [as_channels_samples(v.audio) for v in tracks]
    if len(rates) != 1 or any(v.sample_rate != sample_rate for v in tracks):
        per_track = {name: v.sample_rate for name, v in zip(track_names, tracks)}
        emit_warning(
            "dissolve_videos: videos carry audio at different sample rates ("
            + ", ".join(f"{name}: {rate} Hz" for name, rate in per_track.items())
            + f") - resampling them all to {sample_rate} Hz. Pass "
            "'sample_rate' to pin a different target, or resample ahead of "
            "this step with the 'resample_audio' task.",
            kind="sample_rate_mismatch",
            command="dissolve_videos",
            sample_rate=sample_rate,
            sample_rates=per_track,
        )
        waveforms = [
            (
                waveform
                if video.sample_rate == sample_rate
                else resample_waveform(waveform, video.sample_rate, sample_rate)
            )
            for video, waveform in zip(tracks, waveforms)
        ]
    crossfade_ms = dissolve_frames / fps * 1000 if dissolve_frames else 0
    if match_levels:
        waveforms = match_track_levels(
            waveforms, match_levels, match_levels_dbfs, "dissolve_videos"
        )
    else:
        warn_on_level_spread(waveforms, "dissolve_videos")
    return (
        crossfade_concat(waveforms, sample_rate, crossfade_ms, starts),
        sample_rate,
    )
