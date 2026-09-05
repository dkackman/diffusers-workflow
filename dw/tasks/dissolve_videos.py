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

from ..result import AudioVideo
from .audio_utils import as_channels_samples, crossfade_concat
from .video_utils import frames_as_array, load_audio_video

logger = logging.getLogger("dw")


def dissolve_videos(
    videos,
    dissolve_frames=12,
    fade_in_frames=0,
    fade_out_frames=0,
    fade_color=(0, 0, 0),
    fps=None,
):
    """Task command: join videos with cross-dissolves at every seam.

    Args:
        videos: The videos to join, in order - frame lists, frame arrays,
            AudioVideos, or the path or URL of a video file. Give each video
            its own entry, as with concat_videos
        dissolve_frames: Frames of overlap at each seam. 0 is a hard cut
        fade_in_frames: Frames over which the first video rises out of
            `fade_color`
        fade_out_frames: Frames over which the last video sinks into it
        fade_color: The RGB colour the fades come from and go to
        fps: Frame rate of the videos - required to crossfade audio at a
            dissolve, and ignored when no video carries any

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

    for index, clip in enumerate(clips):
        seams = (index > 0) + (index < len(clips) - 1)
        if len(clip) < seams * dissolve_frames:
            raise ValueError(
                f"video {index} has {len(clip)} frames, too few for its "
                f"{seams} dissolve(s) of {dissolve_frames} frames"
            )

    joined = clips[0]
    for clip in clips[1:]:
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
    audio, sample_rate = _dissolve_audio(loaded, dissolve_frames, fps)
    logger.info(
        f"Dissolved {len(clips)} videos into {len(frames)} frames "
        f"({dissolve_frames}-frame seams)"
    )
    return AudioVideo(frames, audio, sample_rate)


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


def _dissolve_audio(videos, dissolve_frames, fps):
    """Crossfade every video's track over the seams' own span."""
    tracks = [v for v in videos if isinstance(v, AudioVideo) and v.audio is not None]
    if len(tracks) != len(videos):
        if tracks:
            logger.warning(
                "dissolve_videos: some videos carry no audio - the result is silent"
            )
        return None, None
    if fps is None and dissolve_frames:
        raise ValueError("dissolve_videos needs 'fps' to crossfade audio at a dissolve")

    rates = {v.sample_rate for v in tracks}
    if len(rates) != 1:
        raise ValueError(f"dissolve_videos needs one sample rate, got {sorted(rates)}")
    sample_rate = rates.pop()
    crossfade_ms = dissolve_frames / fps * 1000 if dissolve_frames else 0
    waveforms = [as_channels_samples(v.audio) for v in tracks]
    return crossfade_concat(waveforms, sample_rate, crossfade_ms), sample_rate
