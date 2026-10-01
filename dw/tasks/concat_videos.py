"""Concatenate videos - and the audio generated with them - into one video.

The standalone counterpart of what a chained pipeline step does internally:
frames are joined end to end with an optional head trim on every video after
the first, and audio tracks are joined at each seam with an equal-power
crossfade drawn from the trimmed-off material, so video and audio stay in
sync. Cuts have no trimmed material to fade with, so they can instead let the
outgoing tail ring on across the seam - see `audio_bleed_ms`.
"""

import logging

import numpy

from ..events import emit_log, emit_warning
from ..media_types import AudioVideo
from ..shots import measured_num_samples, nested_shots, shot_record, trimmed_shots
from ..dsp import as_channels_samples
from ..task_domains import frames_to_samples
from ..dsp import resample_waveform
from .joins import (
    bleed_join,
    equal_power_crossfade_join,
    fit_audio_to_frames,
    match_levels as match_track_levels,
    video_names,
    warn_on_level_spread,
)
from .video_utils import (
    check_same_frame_size,
    frames_as_pil_list,
    is_video_location,
    load_audio_video,
)

logger = logging.getLogger("dw")


def concat_videos(
    videos,
    trim_frames=0,
    crossfade_ms=75,
    audio_bleed_ms=0,
    audio_bleed_gain_db=0,
    seam_fade_ms=None,
    fps=None,
    match_levels=None,
    match_levels_dbfs=None,
    sample_rate=None,
):
    """Concatenate a list of videos into a single AudioVideo.

    Args:
        videos: The videos to join, in order - frame lists, frame arrays,
            AudioVideos (from previous_result references), the path or URL
            of a video file, or a {"location": ...} dict wrapping either -
            the same idiom get_last_frame(video=...) and a pipeline's
            'image' argument accept (#510) - each read with the audio muxed
            into it. Give
            each video its own entry: one previous_result reference naming a
            step that produced several videos fans this step out over them,
            one concatenation per video, rather than joining them. Every
            video must be the same frame size - unlike a sample-rate
            mismatch, this task does not resize one for you, so a
            statically-resolvable (asset:/output:/literal path) size
            disagreement is refused at validate; one only known at run time
            still fails there (#504). To fit the odd video: `video_frames`
            to get its frames, `resize_rescale` to the target size
            (`resize_center_crop` squares the frame first and then stretches
            it, distorting a non-square target), then `pair_audio(fit="video")`
            to put its soundtrack back before passing it here (#551)
        trim_frames: Frames dropped from the head of every video after the
            first - the trim used when each video was generated from the
            previous one's last frame
        crossfade_ms: Equal-power crossfade at each audio seam, drawn from
            the trimmed material - so it has no effect when trim_frames is 0,
            which is where every cut-based workflow sits. A hard cut's seam is
            shaped by audio_bleed_ms or seam_fade_ms instead
        audio_bleed_ms: How long the outgoing video's tail rings on over the head
            of the next one, at seams with nothing trimmed to crossfade. For
            cut-based workflows, where every shot is generated independently and
            a running laugh track would otherwise butt-join into silence.
            0 (the default) leaves the seam as a plain declicked join
        audio_bleed_gain_db: Gain applied to the bled tail before it is added,
            in dB. 0 (the default) is unchanged, full-scale, matching the
            outgoing material exactly; negative ducks a tail that would
            otherwise push the seam over 0 dBFS, or that reads as too present
            against the incoming shot. Has no effect when audio_bleed_ms is 0
        seam_fade_ms: Fade applied on each side of a seam that gets neither a
            crossfade nor a bleed. Defaults to the few milliseconds that keep a
            butt-join from clicking; raise it to a hundred or so for a graceful
            hard cut on tonal material, which a bleed would only stutter. It is
            the wrong tool for a continuous bed such as a laugh track or room
            tone: a fade only deepens the hole a bleed is there to cover
        fps: Frame rate of the videos - required to join audio when
            trimming, and the rate the joined file is written at unless
            the step's result.fps overrides it
        match_levels: Even the shots' loudness out before joining -
            "rms" matches perceived level (the measurement
            get_gallery_metadata reports as mean_dbfs), "peak" matches the
            loudest sample. Off by default. Shots generated independently
            land 10 dB apart routinely, and that jump is the one seam
            artifact none of the fade controls can hide, because it is not
            at the seam but either side of it. Left off, a spread wide
            enough to hear is logged as a warning
        match_levels_dbfs: The level match_levels moves every shot to -
            defaults to -1 dBFS for "peak" and -20 dBFS for "rms". A shot
            that would clip at the target is held at -0.5 dBFS peak instead,
            reported as a match_levels_held warning, with a per-shot log
            event naming the hold
        sample_rate: The rate the joined soundtrack is at. Shots that come
            from different sources routinely carry different rates - a 24 kHz
            voice clip paired onto a 32 kHz generation - and unlike a level
            jump that difference has no editorial meaning, so by default the
            highest rate among the inputs is chosen and the rest are
            resampled up to it, with a warning naming which. Give this to pin
            the target instead (#108)

    Returns:
        One AudioVideo; its audio is None when no input video carries any
    """
    if not isinstance(videos, list) or not videos:
        raise ValueError("concat_videos needs a non-empty list of videos")

    if len(videos) == 1 and audio_bleed_ms:
        # The bleed acts on the seam between two inputs; a single input - even
        # an earlier join whose inner seams are recorded in its shots - has
        # none, so the request would otherwise vanish without a trace (#565)
        emit_warning(
            f"concat_videos: 'audio_bleed_ms' ({audio_bleed_ms}) has no effect "
            f"with one input - the bleed acts on seams between inputs, and a "
            f"joined input's own inner seams are not reworked.",
            kind="bleed_no_seam",
            command="concat_videos",
        )

    # Named before they are loaded: a path is the only thing that names
    # itself, and the load below replaces it with what it holds
    names = video_names(videos)
    # A shot an earlier run already wrote is loaded here rather than by
    # gather_videos, which reads frames only and would join it silent
    videos = [load_audio_video(v) if is_video_location(v) else v for v in videos]
    clips = [frames_as_pil_list(v) for v in videos]
    check_same_frame_size(clips, "concat_videos")

    # Every track up front rather than one at a time: levels are matched
    # across the whole set, so the last shot's loudness has to be known
    # before the first one is scaled
    waveforms = [
        (
            as_channels_samples(video.audio)
            if isinstance(video, AudioVideo) and video.audio is not None
            else None
        )
        for video in videos
    ]
    # One rate before anything is joined. Shots assembled from different
    # sources disagree routinely, and the disagreement carries no meaning -
    # so it is converted rather than refused, which is what made an agent
    # invent a resample_audio step by hand (#108)
    rates = [
        video.sample_rate
        for video, waveform in zip(videos, waveforms)
        if waveform is not None and video.sample_rate
    ]
    sample_rate = sample_rate or (max(rates) if rates else None)
    if rates and len(set(rates)) == 1 and rates[0] != sample_rate:
        # The inputs agree and the caller pinned another rate: converting
        # to what was asked for is not a decision made on its behalf (#453)
        emit_log(
            f"concat_videos: resampling every track from {rates[0]} Hz to the "
            f"requested {sample_rate} Hz",
            command="concat_videos",
            sample_rate=sample_rate,
        )
    elif rates and any(rate != sample_rate for rate in rates):
        # emit_warning rather than logger.warning, for the reason the level
        # spread below is emitted: resampling every track is an audio
        # decision made on the caller's behalf, and a caller reading the job
        # over the API or MCP sees the warnings list and nothing else - the
        # conversion landing silently is worse than the loud failure it
        # replaced (#108)
        per_video = {
            name: video.sample_rate
            for name, video in zip(names, videos)
            if isinstance(video, AudioVideo) and video.audio is not None
        }
        emit_warning(
            "concat_videos: videos carry audio at different sample rates ("
            + ", ".join(f"{name}: {rate} Hz" for name, rate in per_video.items())
            + f") - resampling them all to {sample_rate} Hz. Pass "
            "'sample_rate' to pin a different target, or resample ahead of "
            "this step with the 'resample_audio' task.",
            kind="sample_rate_mismatch",
            command="concat_videos",
            sample_rate=sample_rate,
            sample_rates=per_video,
        )
    waveforms = [
        (
            waveform
            if waveform is None
            or not video.sample_rate
            or video.sample_rate == sample_rate
            else resample_waveform(waveform, video.sample_rate, sample_rate)
        )
        for video, waveform in zip(videos, waveforms)
    ]

    if match_levels:
        waveforms = match_track_levels(waveforms, match_levels, match_levels_dbfs)
    else:
        warn_on_level_spread(waveforms)

    frames = []
    audio = None
    audio_native_rate = None
    # Where each video landed, measured on the joined picture and track as
    # they grow - never derived from the frame count, so a track that runs
    # long shows up here as the samples it actually took (#378)
    shots = []

    # Whether the joined track carries anything at all - decides whether a
    # silent input among these gets silence of its own length or is skipped
    # outright, matching join_into_song's rule that a shot with no track is
    # silence for its length rather than a gap that shifts everything after
    # it early (#513, #553)
    has_audio = any(waveform is not None for waveform in waveforms)
    silence_channels = (
        next(waveform.shape[0] for waveform in waveforms if waveform is not None)
        if has_audio
        else None
    )

    for index, (video, clip) in enumerate(zip(videos, clips)):
        head_trim = trim_frames if index > 0 else 0
        start_frame = len(frames)
        start_sample = audio.shape[1] if audio is not None else 0
        frames.extend(clip[head_trim:])
        inner = getattr(video, "shots", None)
        if inner:
            video_shots = nested_shots(
                trimmed_shots(inner, head_trim),
                start_frame,
                start_sample if has_audio else None,
                getattr(video, "sample_rate", None),
                sample_rate,
            )
        else:
            video_shots = [
                shot_record(
                    names[index], start_frame, len(frames) - start_frame, start_sample
                )
            ]
        if index and video_shots:
            # Every seam this step draws is a cut it chose to make, unlike a
            # chain's inner segments (continuity is expected there) - marking
            # it lets analyze_seams tell the two apart (#466)
            video_shots[0]["hard_cut"] = True
        # Which input this shot came from - named_shots (dw/shots.py) uses
        # it to place a step's override name on the right shot once an
        # earlier input has nested more than one of its own (#432)
        for shot in video_shots:
            shot["source_index"] = index
        shots.extend(video_shots)

        if not has_audio:
            continue

        if waveforms[index] is None:
            if fps is None:
                raise ValueError(
                    "concat_videos needs 'fps' to fill silence for a video "
                    "with no audio track of its own"
                )
            waveform = numpy.zeros(
                (silence_channels, frames_to_samples(len(clip), fps, sample_rate)),
                dtype=numpy.float32,
            )
            emit_log(
                f"concat_videos: {names[index]} carries no audio - filled with silence",
                command="concat_videos",
                video=names[index],
            )
        else:
            waveform = waveforms[index]
            input_fps = fps or getattr(video, "fps", None)
            if input_fps:
                # A per-input shortfall against its own frame grid propagates
                # into the join and compounds across every further join that
                # takes this one's own output as an input (#435/#553's
                # remedy for the aggregate track, extended here per input -
                # #562). fit_audio_to_frames pads a short track and warns
                # once the gap is a frame or more; it deliberately leaves an
                # *over*-length track alone (#378 - a shot keeps the samples
                # it actually took), so that direction is warned here
                # instead, since it is exactly the drift analyze_sync_drift's
                # single-seam threshold misses
                wanted_samples = frames_to_samples(len(clip), input_fps, sample_rate)
                overrun_samples = waveform.shape[1] - wanted_samples
                if overrun_samples > sample_rate / input_fps:
                    emit_warning(
                        f"concat_videos: {names[index]}'s audio is "
                        f"{overrun_samples} sample(s) longer than its own "
                        f"{len(clip)}-frame length before the join - drift "
                        f"like this compounds at the seam",
                        command="concat_videos",
                        kind="audio_frame_drift",
                        video=names[index],
                        drift_samples=int(overrun_samples),
                    )
                waveform = fit_audio_to_frames(
                    waveform, sample_rate, len(clip), input_fps, "concat_videos"
                )

        if audio is None:
            audio = waveform
            audio_native_rate = getattr(video, "sample_rate", None)
            continue

        if head_trim > 0 and fps is None:
            raise ValueError(
                "concat_videos needs 'fps' to trim audio in step with the frames"
            )

        trim_samples = (
            frames_to_samples(head_trim, fps, sample_rate) if head_trim else 0
        )
        if trim_samples == 0 and audio_bleed_ms:
            audio = bleed_join(
                audio,
                waveform,
                sample_rate,
                audio_bleed_ms,
                seam_fade_ms,
                audio_bleed_gain_db,
                native_sample_rate=audio_native_rate,
                seam=index,
                between=f"{names[index - 1]} -> {names[index]}",
            )
        else:
            audio = equal_power_crossfade_join(
                audio,
                waveform[:, :trim_samples],
                waveform[:, trim_samples:],
                sample_rate,
                crossfade_ms,
                seam_fade_ms,
                seam=index,
            )
        audio_native_rate = getattr(video, "sample_rate", None)

    # The rate the caller declared, else the rate the first input carries -
    # either beats the result's 8 fps default (#84)
    written_fps = fps or next(
        (v.fps for v in videos if getattr(v, "fps", None)),
        None,
    )
    # Reconciled against the frame grid before shots are measured (#435), so
    # an input already short of its own grid does not carry its shortfall
    # into this join's shot map and compound in a later one
    audio = fit_audio_to_frames(
        audio, sample_rate, len(frames), written_fps, "concat_videos"
    )

    # A seam's crossfade leaves the samples before it where they were, so a
    # shot's track is everything up to where the next measured one began
    measured_num_samples(shots, _length(audio) if audio is not None else None)

    logger.debug(f"Concatenated {len(videos)} videos into {len(frames)} frames")
    return AudioVideo(frames, audio, sample_rate, fps=written_fps, shots=shots)


def _length(audio):
    """How many samples a joined track holds, 0 for none."""
    return 0 if audio is None else audio.shape[1]
