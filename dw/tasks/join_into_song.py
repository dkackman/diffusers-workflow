"""Join a spoken scene into a song: dialogue shots, then shots sung to one track.

A musical number breaks out of dialogue - the last spoken line plays, the
song enters under it, and the sung shots play over the *unbroken* song
rather than over the separate slices each was generated against. Putting
that together from the audio tasks needs a song offset of `dialogue length -
cue`, and a workflow cannot do arithmetic: a hand-computed literal goes stale
the moment one dialogue shot is regenerated at another length (#486). This
task measures the joined dialogue at run time and places the song from it.

The timeline, in samples at the song's rate:

- the dialogue shots are joined end to end, each track fitted to its own
  frames, so the joined dialogue ends exactly where the first song shot's
  frame 0 is - `D`
- the song is placed at `D - cue_seconds * sr`, so song time `cue_seconds`
  lands on that frame, and runs to the end of the picture
- the dialogue ducks by `duck_db` from `duck_delay_ms` after the song enters,
  over a linear `duck_ramp_ms` ramp
- the song shots' own audio is discarded

No final level is chosen here: what a deliverable sits at is the workflow's
to decide, with `normalize_audio` after this step, and the headroom warnings
at save apply as they do to any track.
"""

import logging

import numpy

from ..events import emit_log, emit_warning
from ..loudness import MIN_LUFS_SECONDS, integrated_lufs
from ..result import AudioVideo
from ..shots import measured_num_samples, shot_record
from ..task_domains import check_arguments
from .audio_utils import (
    as_channels_samples,
    frames_to_samples,
    load_audio,
    resample_waveform,
    slice_samples,
)
from .concat_videos import video_names
from .video_utils import (
    check_same_frame_size,
    frames_as_pil_list,
    is_video_location,
    load_audio_video,
)

logger = logging.getLogger("dw")

COMMAND = "join_into_song"


def join_into_song(
    dialogue,
    song_shots,
    song,
    cue_seconds=0,
    dialogue_target_lufs=None,
    duck_delay_ms=0,
    duck_db=-12,
    duck_ramp_ms=250,
    fps=None,
):
    """Join dialogue shots and song shots into one video over the unbroken song.

    Args:
        dialogue: The spoken shots, in order - each keeps its own audio. The
            same entries concat_videos' `videos` takes: previous results,
            paths of files an earlier run wrote, or {"location": ...} dicts.
            A shot with no track is filled with silence for its length
        song_shots: The sung shots, in order. Their audio is discarded - they
            play over `song`
        song: The unbroken track the song shots were sliced from - an audio
            result, a video or audio file's path, or anything carrying
            '.audio' and '.sample_rate'. Its rate is the output's
        cue_seconds: The song time that lands on the first song shot's
            frame 0 - the start of the slice that shot was generated against.
            0 starts the song exactly at the cut; above 0 the song enters
            that long before it, under the last spoken line. Longer than the
            dialogue is refused: the song would have to start before the film
        dialogue_target_lufs: Integrated loudness (BS.1770) each dialogue
            shot is gained to, with one static gain per shot. Omitted, the
            shots keep their own levels. A shot too short (under 400 ms) or
            too quiet to measure is left as it is, with a warning
        duck_delay_ms: How long after the song enters the dialogue starts to
            duck. With it at or past the dialogue's end, nothing ducks
        duck_db: How far the dialogue ducks, in dB - 0 or below
        duck_ramp_ms: The length of the linear ramp into the duck
        fps: The rate the videos play at, when they do not carry one - a
            pipeline's frames carry none, a file brings its own. The joined
            file is written at it unless the step's result.fps overrides it

    Returns:
        One AudioVideo: the dialogue's frames then the song shots', over the
        mix, with one shot record per input
    """
    check_arguments(
        COMMAND,
        cue_seconds=cue_seconds,
        duck_delay_ms=duck_delay_ms,
        duck_db=duck_db,
        duck_ramp_ms=duck_ramp_ms,
        fps=fps,
    )
    # A number handed in through an untyped variable arrives as a string
    cue_seconds = float(cue_seconds)
    duck_delay_ms, duck_db = float(duck_delay_ms), float(duck_db)
    duck_ramp_ms = float(duck_ramp_ms)
    if dialogue_target_lufs is not None:
        dialogue_target_lufs = float(dialogue_target_lufs)
    if fps is not None:
        fps = float(fps)
    for name, value in (("dialogue", dialogue), ("song_shots", song_shots)):
        if not isinstance(value, list) or not value:
            raise ValueError(
                f"{COMMAND} needs a non-empty list of videos for '{name}' - a "
                "join into a song has spoken shots before it and sung shots "
                "after it"
            )

    names = video_names(dialogue + song_shots)
    dialogue = [_loaded(video) for video in dialogue]
    song_shots = [_loaded(video) for video in song_shots]
    videos = dialogue + song_shots
    clips = [frames_as_pil_list(video) for video in videos]
    check_same_frame_size(clips, COMMAND)
    fps = _one_fps(videos, names, fps)

    song_waveform, sample_rate = _song_track(song)
    channels = song_waveform.shape[0]

    # The dialogue's track, shot by shot. Each shot's boundary is placed on
    # the frame grid from the running frame count, so rounding never
    # accumulates and D is the sample the first song frame sits at
    frames = []
    shots = []
    pieces = []
    for index, (video, clip) in enumerate(zip(dialogue, clips)):
        start_frame = len(frames)
        frames.extend(clip)
        start = frames_to_samples(start_frame, fps, sample_rate)
        end = frames_to_samples(len(frames), fps, sample_rate)
        piece = _dialogue_track(
            video, names[index], end - start, fps, sample_rate, channels
        )
        if dialogue_target_lufs is not None:
            piece = _match_loudness(
                piece, sample_rate, dialogue_target_lufs, names[index]
            )
        pieces.append(piece)
        shots.append(
            _shot(names[index], index, start_frame, len(frames) - start_frame, start)
        )
    dialogue_track = numpy.concatenate(pieces, axis=1)
    dialogue_samples = dialogue_track.shape[1]

    for offset, clip in enumerate(clips[len(dialogue) :]):
        index = len(dialogue) + offset
        start_frame = len(frames)
        frames.extend(clip)
        shots.append(
            _shot(
                names[index],
                index,
                start_frame,
                len(frames) - start_frame,
                frames_to_samples(start_frame, fps, sample_rate),
            )
        )
    total_samples = frames_to_samples(len(frames), fps, sample_rate)

    cue_samples = int(round(float(cue_seconds) * sample_rate))
    if cue_samples > dialogue_samples:
        raise ValueError(
            f"{COMMAND}: cue_seconds={cue_seconds:g} is longer than the "
            f"{dialogue_samples / sample_rate:.2f} s of dialogue, so the song "
            "would have to start before the film does. Pass a cue_seconds no "
            "longer than the dialogue - it is the song time the first song "
            "shot's frame 0 lands on, the start of the slice it was "
            "generated against"
        )
    song_entry = dialogue_samples - cue_samples

    mix = numpy.zeros((channels, total_samples), dtype=numpy.float32)
    mix[:, :dialogue_samples] += _ducked(
        dialogue_track, song_entry, sample_rate, duck_delay_ms, duck_db, duck_ramp_ms
    )
    mix[:, song_entry:] += _placed_song(
        song_waveform, total_samples - song_entry, sample_rate
    )

    measured_num_samples(shots, mix.shape[1])
    emit_log(
        f"{COMMAND}: song enters at {song_entry / sample_rate:.3f} s, "
        f"{cue_seconds:g} s before the first song shot at "
        f"{dialogue_samples / sample_rate:.3f} s",
        command=COMMAND,
        song_entry_sample=song_entry,
        dialogue_samples=dialogue_samples,
        sample_rate=sample_rate,
    )
    logger.debug(
        f"Joined {len(dialogue)} dialogue and {len(song_shots)} song shots "
        f"into {len(frames)} frames"
    )
    return AudioVideo(frames, mix, sample_rate, fps=fps, shots=shots)


def _loaded(video):
    """A video entry as something with frames, loading a path with its audio."""
    return load_audio_video(video) if is_video_location(video) else video


def _one_fps(videos, names, fps):
    """The rate the join plays at: the one given, else the one the inputs agree on.

    Frames are joined one for one, so a song shot at another rate would play
    at the wrong speed against the song - refused, not resampled.
    """
    carried = [
        (name, getattr(video, "fps", None)) for name, video in zip(names, videos)
    ]
    rates = {float(rate) for _name, rate in carried if rate}
    if len(rates) > 1:
        raise ValueError(
            f"{COMMAND} needs every video at one frame rate, got "
            + ", ".join(
                f"{name}: {float(rate):g} fps" for name, rate in carried if rate
            )
            + ". Frames are joined one for one, so a shot at another rate would "
            "play at the wrong speed against the song"
        )
    if fps is None:
        fps = next((rate for _name, rate in carried if rate), None)
    if fps is None:
        raise ValueError(
            f"{COMMAND} needs 'fps' - none of the videos carries a frame rate "
            "of its own, and the song is placed from the dialogue's length in "
            "time"
        )
    return fps


def _song_track(song):
    """The song as a (channels, samples) waveform and its rate."""
    if is_video_location(song):
        location = song["location"] if isinstance(song, dict) else song
        waveform, rate = load_audio(location)
    else:
        waveform = getattr(song, "audio", song)
        rate = getattr(song, "sample_rate", None)
        if waveform is None:
            raise ValueError(
                f"{COMMAND} needs a song track - the one given carries no audio"
            )
        waveform = as_channels_samples(waveform)
    if not rate:
        raise ValueError(
            f"{COMMAND} needs the song's sample rate - the track it was given "
            "does not carry one"
        )
    return waveform, int(rate)


def _dialogue_track(video, name, length, fps, sample_rate, channels):
    """One dialogue shot's track at the song's rate, fitted to its own frames.

    Fitted rather than measured as-is: the song is placed from where the
    dialogue ends, so a track running past its frames would push the song
    off the first song shot's picture by the overrun. A shot with no track
    is silence for its length - skipping it would land every later shot's
    audio early.
    """
    audio = getattr(video, "audio", None)
    if audio is None:
        emit_log(
            f"{COMMAND}: {name} carries no audio - filled with silence",
            command=COMMAND,
            shot=name,
        )
        return numpy.zeros((channels, length), dtype=numpy.float32)
    waveform = as_channels_samples(audio)
    rate = getattr(video, "sample_rate", None) or sample_rate
    if rate != sample_rate:
        emit_log(
            f"{COMMAND}: resampling {name} from {rate} Hz to the song's "
            f"{sample_rate} Hz",
            command=COMMAND,
            shot=name,
            sample_rate=sample_rate,
        )
        waveform = resample_waveform(waveform, rate, sample_rate)
    waveform = _with_channels(waveform, channels)
    difference = waveform.shape[1] - length
    if abs(difference) >= sample_rate / float(fps):
        emit_warning(
            f"{COMMAND}: {name}'s track is {waveform.shape[1] / sample_rate:.3f} s "
            f"and its frames are {length / sample_rate:.3f} s - "
            + ("trimmed" if difference > 0 else "padded with silence")
            + " to its frames, so the song lands on the first song shot's "
            "frame 0.",
            kind="dialogue_fitted_to_frames",
            command=COMMAND,
            shot=name,
            difference_samples=int(difference),
        )
    return slice_samples(waveform, 0, length)


def _with_channels(waveform, channels):
    """The waveform with the song's channel count - mono spread, extra dropped."""
    have = waveform.shape[0]
    if have == channels:
        return waveform
    if have == 1:
        return numpy.repeat(waveform, channels, axis=0)
    if channels == 1:
        return waveform.mean(axis=0, keepdims=True)
    return waveform[:channels]


def _match_loudness(piece, sample_rate, target_lufs, name):
    """One dialogue shot gained by one static amount to target_lufs."""
    measured = integrated_lufs(piece.T, sample_rate)
    if measured is None:
        seconds = piece.shape[1] / float(sample_rate)
        reason = (
            f"it is {seconds:.2f} s, under the {MIN_LUFS_SECONDS * 1000:.0f} ms "
            "loudness measurement needs"
            if seconds < MIN_LUFS_SECONDS
            else "its loudness could not be measured (silent or near it)"
        )
        emit_warning(
            f"{COMMAND}: {name} was left at its own level rather than matched "
            f"to {target_lufs:g} LUFS - {reason}.",
            kind="dialogue_unmatched",
            command=COMMAND,
            shot=name,
            target_lufs=target_lufs,
        )
        return piece
    gain_db = float(target_lufs) - measured
    emit_log(
        f"{COMMAND}: {name} {measured:.1f} LUFS, gained {gain_db:+.1f} dB",
        command=COMMAND,
        shot=name,
        measured_lufs=round(measured, 1),
        gain_db=round(gain_db, 2),
    )
    return (piece * (10.0 ** (gain_db / 20.0))).astype(numpy.float32)


def _ducked(track, song_entry, sample_rate, delay_ms, duck_db, ramp_ms):
    """The dialogue with its duck applied: unity, a linear ramp, then held."""
    length = track.shape[1]
    start = song_entry + int(round(float(delay_ms) * sample_rate / 1000.0))
    if start >= length or duck_db == 0:
        return track
    ramp = int(round(float(ramp_ms) * sample_rate / 1000.0))
    floor = 10.0 ** (float(duck_db) / 20.0)
    envelope = numpy.ones(length, dtype=numpy.float32)
    if ramp:
        # Linear in amplitude from unity at `start` to the floor at
        # `start + ramp`, the ramp the plan names rather than a step
        steps = numpy.arange(ramp, dtype=numpy.float32) / ramp
        section = 1.0 + (floor - 1.0) * steps
        end = min(length, start + ramp)
        envelope[start:end] = section[: end - start]
        envelope[end:] = floor
    else:
        envelope[start:] = floor
    return track * envelope


def _placed_song(song, length, sample_rate):
    """The song from its start, `length` samples of it - padded when short."""
    have = song.shape[1]
    if have < length:
        short = (length - have) / float(sample_rate)
        emit_warning(
            f"{COMMAND}: the song is {have / sample_rate:.2f} s and the picture "
            f"needs {length / sample_rate:.2f} s of it from where it enters - "
            f"padded with {short:.2f} s of silence, so the end of the cut has "
            "no song under it. A longer song, or fewer song shots, covers it.",
            kind="song_short",
            command=COMMAND,
            song_seconds=have / float(sample_rate),
            needed_seconds=length / float(sample_rate),
        )
    return slice_samples(song, 0, length)


def _shot(name, index, start_frame, num_frames, start_sample):
    """A shot record tagged with the input it came from, for named_shots."""
    shot = shot_record(name, start_frame, num_frames, start_sample)
    if index:
        # Every seam here is a cut this step draws, as concat_videos marks it
        shot["hard_cut"] = True
    shot["source_index"] = index
    return shot
