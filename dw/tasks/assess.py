"""Assessment probes: measure a finished cut and say where to look (#387).

Three read-only task commands - `analyze_shots`, `analyze_seams` and
`analyze_sync_drift` - each take a video (a path, or the AudioVideo an
earlier step returned) and answer a JSON-safe dict: every measurement it
took, the `findings` its rules raised (`dw/assessment_rules.py`), the names
of the rules it read the measurements against and where the shot
boundaries came from. Nothing acts on a finding; they are places to look.

The reader streams. A cut is minutes of full-resolution picture, and every
one of these measurements needs only a 64x36 grey thumbnail of each frame
and the soundtrack, so `read_media` decodes the file once and keeps exactly
that - never a frame list, and never `load_audio`'s 0.25 s fit of the track
to the picture, which would move the very sample count `analyze_sync_drift`
is measuring. The soundtrack is trimmed to the audio stream's own duration,
which the container already reports net of the encoder's priming, so a
lossy codec's padding is not read as drift.

Shot boundaries come, in order, from the `shots` argument, the AudioVideo's
own `shots`, the run manifest beside a file (`dw.runs.shots_beside`), and
otherwise the whole file is one shot. `shots_source` says which. A shot
record may carry `hard_cut: true` - a cut meant as a cut - and the frame
jump rule does not fire at the seam that shot opens.
"""

import logging
import math

import numpy

from ..assessment_rules import HOLE_VOICED_DBFS, crosses, finding, rules_for
from ..events import emit_warning

logger = logging.getLogger("dw")

THUMB_WIDTH = 64
THUMB_HEIGHT = 36

# Audio windows around a seam, in seconds. The edge windows say whether the
# join itself is voiced (the hole guard) and what the balance does across it;
# they are not the level step, which is shot against shot (`_shot_rms`): a
# shot's own last and first quarter-second differ by whatever the take does
# there - 20 dB on a line that trails off and opens on a breath - and read
# as a step at every seam of a cut made of one clip (#387's bounce).
LEVEL_WINDOW = 0.25
FLOOR_WINDOW = 0.02
CLICK_WINDOW = 0.002
CLICK_NEIGHBOUR_WINDOW = 0.01

# The inter-frame difference a shot is expected to show, below which its
# own motion is not a baseline: a static shot's 90th percentile is ~0, and
# dividing a seam's change by it would call any change at all a jump.
# Grey levels on the 0-255 scale.
TYPICAL_DELTA_FLOOR = 2.0
TYPICAL_DELTA_PERCENTILE = 90

# A ratio against a zero neighbour has no size; report it capped
CLICK_CAP_DB = 120.0

_SILENCE = 1e-10


class Media:
    """What the probes read from a video: thumbnails, soundtrack, timing.

    thumbs: (frames, THUMB_HEIGHT, THUMB_WIDTH) uint8 grey, or None
    audio: (channels, samples) float32, or None
    video_seconds / audio_seconds: each stream's own duration, or None
    """

    def __init__(
        self,
        thumbs,
        audio,
        sample_rate,
        fps,
        video_seconds=None,
        audio_seconds=None,
        shots=None,
    ):
        self.thumbs = thumbs
        self.audio = audio
        self.sample_rate = sample_rate
        self.fps = fps
        self.video_seconds = video_seconds
        self.audio_seconds = audio_seconds
        self.shots = shots

    @property
    def frame_count(self):
        return 0 if self.thumbs is None else int(self.thumbs.shape[0])


def _stream_seconds(stream):
    if stream is None or stream.duration is None or stream.time_base is None:
        return None
    return float(stream.duration * stream.time_base)


def _as_float_samples(samples):
    if samples.dtype.kind == "u":
        iinfo = numpy.iinfo(samples.dtype)
        half = (iinfo.max + 1) / 2
        return (samples.astype(numpy.float32) - half) / half
    if samples.dtype.kind == "i":
        return samples.astype(numpy.float32) / numpy.iinfo(samples.dtype).max
    return samples.astype(numpy.float32)


def read_media(path):
    """Stream a video file once into a Media.

    Each decoded frame is reduced to a grey thumbnail as it arrives and then
    dropped, so memory holds thumbnails, not pictures. The soundtrack is
    kept whole (the probes cut windows out of it anywhere) and trimmed to
    the audio stream's reported duration.
    """
    import av

    from ..media_info import _as_frame_samples

    thumbs = []
    chunks = []
    with av.open(path) as container:
        video = container.streams.video[0] if container.streams.video else None
        audio = container.streams.audio[0] if container.streams.audio else None
        if video is None and audio is None:
            raise ValueError(f"{path} has neither a video nor an audio stream")
        fps = float(video.average_rate) if video and video.average_rate else None
        video_seconds = _stream_seconds(video)
        audio_seconds = _stream_seconds(audio)
        channels = int(audio.channels) if audio is not None else 0
        streams = [s for s in (video, audio) if s is not None]
        for frame in container.decode(*streams):
            if isinstance(frame, av.VideoFrame):
                thumbs.append(
                    frame.reformat(
                        width=THUMB_WIDTH, height=THUMB_HEIGHT, format="gray"
                    ).to_ndarray()[:THUMB_HEIGHT, :THUMB_WIDTH]
                )
            elif isinstance(frame, av.AudioFrame):
                samples = _as_float_samples(frame.to_ndarray())
                chunks.append(_as_frame_samples(samples, channels))
        sample_rate = int(audio.rate) if audio is not None else None

    waveform = None
    if chunks:
        waveform = numpy.ascontiguousarray(numpy.concatenate(chunks, axis=0).T)
        if audio_seconds is not None:
            waveform = waveform[:, : int(round(audio_seconds * sample_rate))]
    if video_seconds is None and fps and thumbs:
        video_seconds = len(thumbs) / fps
    return Media(
        numpy.stack(thumbs) if thumbs else None,
        waveform,
        sample_rate,
        fps,
        video_seconds,
        audio_seconds,
    )


def _thumb(frame):
    """One in-memory frame (PIL image, or an HxWxC array or tensor, uint8 or
    float in [0, 1]) as a grey thumbnail."""
    from PIL import Image

    if not isinstance(frame, Image.Image):
        array = frame
        if hasattr(array, "detach"):
            array = array.detach().cpu().numpy()
        array = numpy.asarray(array)
        if array.dtype.kind == "f":
            array = numpy.clip(array * 255.0, 0, 255)
        array = array.astype(numpy.uint8)
        if array.ndim == 3 and array.shape[-1] == 1:
            array = array[..., 0]
        frame = Image.fromarray(array)
    return numpy.asarray(
        frame.convert("L").resize((THUMB_WIDTH, THUMB_HEIGHT), Image.BILINEAR),
        dtype=numpy.uint8,
    )


def _in_memory_thumbs(frames):
    """Thumbnails of an AudioVideo's frames, one frame at a time: a list, an
    (N, H, W, C) array, or a SegmentedFrames replaying (N, H, W, 3) chunks."""
    from ..pipeline_processors.chain import SegmentedFrames

    thumbs = []
    if isinstance(frames, SegmentedFrames):
        for chunk in frames:
            for frame in chunk:
                thumbs.append(_thumb(frame))
    else:
        for frame in frames:
            thumbs.append(_thumb(frame))
    return numpy.stack(thumbs) if thumbs else None


def _file_path(video):
    """The validated file a probe's 'video' names, or None for an in-memory
    clip: a path, or the VideoFileReference an asset:/output: reference or a
    literal path is realized to (dw/arguments.py, #387)."""
    from ..locations import validate_media_path
    from .video_utils import VideoFileReference

    if isinstance(video, VideoFileReference):
        video = video.path
    if isinstance(video, str):
        return validate_media_path(video, None, "a video to assess")
    return None


def media_from(video):
    """A Media from a path, a VideoFileReference or an in-memory AudioVideo."""
    path = _file_path(video)
    if path is not None:
        return read_media(path)
    if hasattr(video, "frames") or hasattr(video, "audio"):
        from .audio_utils import as_channels_samples

        audio = getattr(video, "audio", None)
        waveform = None if audio is None else as_channels_samples(audio)
        thumbs = (
            _in_memory_thumbs(video.frames)
            if getattr(video, "frames", None) is not None
            else None
        )
        sample_rate = getattr(video, "sample_rate", None)
        fps = getattr(video, "fps", None)
        media = Media(
            thumbs,
            waveform,
            sample_rate,
            fps,
            video_seconds=(thumbs.shape[0] / fps)
            if thumbs is not None and fps
            else None,
            audio_seconds=(
                waveform.shape[1] / sample_rate
                if waveform is not None and sample_rate
                else None
            ),
            shots=getattr(video, "shots", None),
        )
        return media
    raise ValueError(
        "a probe takes a stored video (asset:, output: or a path) or the "
        f"video an earlier step returned, not {type(video).__name__} - a URL "
        "downloads as bare frames with no soundtrack to measure"
    )


def resolve_shots(video, media, shots=None):
    """The shot records to measure against, and where they came from."""
    if shots:
        return [dict(shot) for shot in shots], "argument"
    if media.shots:
        return [dict(shot) for shot in media.shots], "artifact"
    path = _file_path(video)
    if path is not None:
        from ..runs import shots_beside

        recorded = shots_beside(path)
        if recorded:
            return [dict(shot) for shot in recorded], "manifest"
    return None, "none"


def _whole_file_shot(media):
    samples = media.audio.shape[1] if media.audio is not None else None
    return {
        "name": "whole",
        "start_frame": 0,
        "num_frames": media.frame_count,
        "start_sample": 0 if samples is not None else None,
        "num_samples": samples,
    }


def _sample_span(shot, media):
    """A shot's (start, count, source) on the soundtrack: recorded, else
    derived from its frames."""
    start = shot.get("start_sample")
    count = shot.get("num_samples")
    if start is not None and count is not None:
        return int(start), int(count), "recorded"
    if media.fps and media.sample_rate:
        scale = media.sample_rate / media.fps
        return (
            int(round(shot.get("start_frame", 0) * scale)),
            int(round(shot.get("num_frames", 0) * scale)),
            "derived",
        )
    return None, None, None


def _db(value):
    return None if value is None or value <= _SILENCE else 20.0 * math.log10(value)


def _rms(window):
    if window is None or window.size == 0:
        return None
    return math.sqrt(float(numpy.mean(numpy.square(window, dtype=numpy.float64))))


def _peak(window):
    if window is None or window.size == 0:
        return None
    return float(numpy.max(numpy.abs(window)))


def _clip(media, start, end):
    total = media.audio.shape[1]
    start = max(0, min(total, int(start)))
    end = max(start, min(total, int(end)))
    return media.audio[:, start:end]


def _round(value, places=2):
    return None if value is None else round(float(value), places)


def _findings(probe, record, at, skip=()):
    found = []
    for rule in rules_for(probe):
        if rule["name"] in skip or rule["field"] not in record:
            continue
        if crosses(rule, record[rule["field"]]):
            found.append(finding(rule, record[rule["field"]], at))
    return found


def _span_overrun(media, shot):
    """How far past the file `shot`'s frames or samples reach, or None when
    it fits. `_clip` silently clamps an out-of-range window to the file
    (#425): a shot record with `start_frame + num_frames` past the video's
    real length, or explicit `start_sample + num_samples` past the
    soundtrack's, is measured over a window shorter than the caller asked
    for and nothing says so unless this is checked first."""
    over_frames = None
    if media.thumbs is not None:
        start_frame = shot.get("start_frame", 0)
        num_frames = shot.get("num_frames")
        if isinstance(start_frame, (int, float)) and not isinstance(start_frame, bool):
            if isinstance(num_frames, (int, float)) and not isinstance(
                num_frames, bool
            ):
                end = int(start_frame) + int(num_frames)
                if end > media.frame_count:
                    over_frames = end - media.frame_count
    over_samples = None
    if media.audio is not None:
        start_sample, num_samples = shot.get("start_sample"), shot.get("num_samples")
        if isinstance(start_sample, (int, float)) and not isinstance(
            start_sample, bool
        ):
            if isinstance(num_samples, (int, float)) and not isinstance(
                num_samples, bool
            ):
                total = media.audio.shape[1]
                end = int(start_sample) + int(num_samples)
                if end > total:
                    over_samples = end - total
    if over_frames is None and over_samples is None:
        return None
    return {"frames": over_frames, "samples": over_samples}


def _shot_span_findings(probe, records, media):
    """Findings (and a run warning) for every shot record whose declared
    span reaches past the file - clipped silently otherwise (#425)."""
    findings = []
    overrun_names = []
    for shot in records or []:
        overrun = _span_overrun(media, shot)
        if overrun is None:
            continue
        detail = (
            f"{overrun['frames']} frame(s)"
            if overrun["frames"] is not None
            else f"{overrun['samples']} sample(s)"
        )
        findings.append(
            {
                "rule": "shot_span_overrun",
                "severity": "warning",
                "at": {"shot": shot.get("name")},
                "value": overrun,
                "threshold": 0,
                "says": f"shot {shot.get('name')!r} reaches {detail} past the file's end",
            }
        )
        overrun_names.append(shot.get("name"))
    if overrun_names:
        emit_warning(
            f"{probe}: shot record(s) {', '.join(str(name) for name in overrun_names)} "
            "reach past the file's end and were silently clipped to it",
            kind="shot_span_overrun",
            probe=probe,
            shots=overrun_names,
        )
    return findings


def _answer(probe, measurements, findings, shots_source, shot_dependent=()):
    """A probe's answer, with `rules_applied` cut down to the rules that
    actually ran. `shot_dependent` names (or `True` for all of the probe's
    rules) the ones that only mean anything measured shot against shot; with
    no shot boundaries at all (`shots_source == "none"`) those are reported
    as `rules_skipped` instead of `rules_applied`, and a run warning says why
    - without this a shotless file (an asset kept before #393, an upload, a
    cut joined outside dw) read as a clean pass with nothing measured (#394).
    """
    names = [rule["name"] for rule in rules_for(probe)]
    dependent = set(names) if shot_dependent is True else set(shot_dependent)
    applied, skipped = names, []
    if shots_source == "none" and dependent:
        applied = [name for name in names if name not in dependent]
        skipped = [
            {"rule": name, "reason": "no shot boundaries"}
            for name in names
            if name in dependent
        ]
        emit_warning(
            f"{probe} found no shot boundaries for this file, so "
            f"{', '.join(sorted(dependent))} could not be measured - pass "
            "shots= to supply them",
            kind="no_shot_boundaries",
            probe=probe,
        )
    return {
        **measurements,
        "findings": findings,
        "rules_applied": applied,
        "rules_skipped": skipped,
        "shots_source": shots_source,
    }


def analyze_shots(video, shots=None):
    """Task command: each shot's level and spectral balance, and how far
    apart the shots sit.

    Args:
        video: A video file's path, or the video an earlier step returned
        shots: Shot records to measure by, overriding any the video carries

    Returns:
        {shots: [{name, start_frame, num_frames, peak_dbfs, rms_dbfs, crest_db, low_dbfs, mid_dbfs,
        high_dbfs, samples}], rms_range_db, findings, rules_applied,
        rules_skipped, shots_source}
    """
    media = media_from(video)
    return shots_answer(media, *resolve_shots(video, media, shots))


def shots_answer(media, records, source):
    """`analyze_shots` over an already-read Media and resolved shots."""
    from .audio_utils import _spectral_balance

    if media.audio is None:
        return _answer(
            "analyze_shots",
            {"shots": [], "rms_range_db": None, "has_audio": False},
            _shot_span_findings("analyze_shots", records, media),
            source,
            shot_dependent={"shot_level_spread"},
        )
    records = records or [_whole_file_shot(media)]
    overrun_findings = _shot_span_findings("analyze_shots", records, media)

    measured = []
    for shot in records:
        start, count, samples_source = _sample_span(shot, media)
        window = _clip(media, start, start + count) if start is not None else None
        peak = _db(_peak(window))
        rms = _db(_rms(window))
        balance = (
            _spectral_balance(window, media.sample_rate)
            if window is not None and window.size
            else {"low_dbfs": None, "mid_dbfs": None, "high_dbfs": None}
        )
        measured.append(
            {
                "name": shot.get("name"),
                "start_frame": shot.get("start_frame"),
                "num_frames": shot.get("num_frames"),
                "peak_dbfs": _round(peak),
                "rms_dbfs": _round(rms),
                "crest_db": _round(None if peak is None or rms is None else peak - rms),
                **{key: _round(value) for key, value in balance.items()},
                "samples": samples_source,
            }
        )

    voiced = [shot for shot in measured if shot["rms_dbfs"] is not None]
    rms_range = None
    at = None
    if len(voiced) > 1:
        loudest = max(voiced, key=lambda shot: shot["rms_dbfs"])
        quietest = min(voiced, key=lambda shot: shot["rms_dbfs"])
        rms_range = _round(loudest["rms_dbfs"] - quietest["rms_dbfs"])
        at = {"between": [loudest["name"], quietest["name"]]}
    elif voiced:
        rms_range = 0.0
    answer = {"shots": measured, "rms_range_db": rms_range, "has_audio": True}
    return _answer(
        "analyze_shots",
        answer,
        overrun_findings + (_findings("analyze_shots", answer, at) if at else []),
        source,
        shot_dependent={"shot_level_spread"},
    )


def _band_shares(window, sample_rate):
    from .audio_utils import _spectral_balance

    if window is None or window.size == 0:
        return None
    bands = _spectral_balance(window, sample_rate)
    energies = {
        key: (10.0 ** (value / 10.0) if value is not None else 0.0)
        for key, value in bands.items()
    }
    total = sum(energies.values())
    if total <= 0.0:
        return None
    return {key: value / total for key, value in energies.items()}


def _shot_rms(media, shot):
    """A shot's RMS level over its whole sample span, in dBFS, or None."""
    start, count, _source = _sample_span(shot, media)
    if start is None:
        return None
    return _db(_rms(_clip(media, start, start + count)))


def _seam_audio(media, before_end, after_start, previous_rms, next_rms):
    """Audio measurements at a seam: the edge windows end at `before_end`
    and open at `after_start` (the same sample at a cut, either side of the
    fade at a dissolve), and the join is what lies between them - or the
    FLOOR_WINDOW centred on the cut. The level step is between the two
    shots' own levels, `previous_rms` and `next_rms`."""
    rate = media.sample_rate
    level = int(round(LEVEL_WINDOW * rate))
    before = _clip(media, before_end - level, before_end)
    after = _clip(media, after_start, after_start + level)
    before_rms = _db(_rms(before))
    after_rms = _db(_rms(after))

    centre = (before_end + after_start) // 2
    half_floor = max(1, int(round(FLOOR_WINDOW * rate / 2)))
    join = (
        _clip(media, before_end, after_start)
        if after_start - before_end > 2 * half_floor
        else _clip(media, centre - half_floor, centre + half_floor)
    )
    floor = _db(_rms(join))

    half_click = max(1, int(round(CLICK_WINDOW * rate / 2)))
    neighbour = int(round(CLICK_NEIGHBOUR_WINDOW * rate))
    click_peak = _peak(_clip(media, centre - half_click, centre + half_click))
    neighbour_peak = max(
        _peak(_clip(media, centre - half_click - neighbour, centre - half_click))
        or 0.0,
        _peak(_clip(media, centre + half_click, centre + half_click + neighbour))
        or 0.0,
    )
    click = None
    if click_peak is not None and click_peak > _SILENCE:
        click = (
            CLICK_CAP_DB
            if neighbour_peak <= _SILENCE
            else min(CLICK_CAP_DB, 20.0 * math.log10(click_peak / neighbour_peak))
        )

    shares_before = _band_shares(before, rate)
    shares_after = _band_shares(after, rate)
    spectral_shift = (
        sum(abs(shares_after[key] - shares_before[key]) for key in shares_before) / 2.0
        if shares_before and shares_after
        else None
    )
    return {
        "before_rms_dbfs": _round(before_rms),
        "after_rms_dbfs": _round(after_rms),
        "before_shot_rms_dbfs": _round(previous_rms),
        "after_shot_rms_dbfs": _round(next_rms),
        "level_step_db": _round(
            None
            if previous_rms is None or next_rms is None
            else abs(next_rms - previous_rms)
        ),
        "floor_dbfs": _round(floor),
        "click_db": _round(click),
        "spectral_shift": _round(spectral_shift, 3),
    }


def _typical_delta(thumbs, start, end):
    """The TYPICAL_DELTA_PERCENTILE of frame-to-frame change inside
    thumbs[start:end], or None for fewer than two frames."""
    span = thumbs[max(0, start) : max(0, end)]
    if span.shape[0] < 2:
        return None
    deltas = numpy.abs(numpy.diff(span.astype(numpy.int16), axis=0)).mean(axis=(1, 2))
    return float(numpy.percentile(deltas, TYPICAL_DELTA_PERCENTILE))


def _seam_video(media, previous, shot, fade):
    """Picture measurements at the seam `shot` opens: the largest single-frame
    change across it (at a dissolve, across the whole fade - each step of a
    fade is small, which is what a dissolve is), against the larger of the
    two shots' own typical change, floored."""
    thumbs = media.thumbs
    start = int(shot.get("start_frame", 0))
    first = max(0, start - 1)
    last = min(thumbs.shape[0] - 1, start + max(0, fade - 1) if fade else start)
    if last <= first:
        return {"frame_delta": None, "typical_delta": None, "jump_ratio": None}
    across = numpy.abs(
        numpy.diff(thumbs[first : last + 1].astype(numpy.int16), axis=0)
    ).mean(axis=(1, 2))
    frame_delta = float(across.max())
    typical = [
        _typical_delta(
            thumbs,
            int(previous.get("start_frame", 0))
            + int(previous.get("overlap_frames") or 0),
            start,
        ),
        _typical_delta(thumbs, start + fade, start + int(shot.get("num_frames", 0))),
    ]
    typical = max(
        [TYPICAL_DELTA_FLOOR] + [value for value in typical if value is not None]
    )
    return {
        "frame_delta": _round(frame_delta),
        "typical_delta": _round(typical),
        "jump_ratio": _round(frame_delta / typical),
    }


def analyze_seams(video, shots=None):
    """Task command: measure every seam between shots, audio and picture.

    Args:
        video: A video file's path, or the video an earlier step returned
        shots: Shot records to measure by, overriding any the video carries.
            A shot marked `hard_cut: true` opens a seam meant as a cut

    Returns:
        {seams: [{seam, between, seconds, kind, level_step_db,
        before_shot_rms_dbfs, after_shot_rms_dbfs, floor_dbfs, click_db,
        spectral_shift, before_rms_dbfs, after_rms_dbfs,
        frame_delta, typical_delta, jump_ratio}], findings, rules_applied,
        rules_skipped, shots_source}
    """
    media = media_from(video)
    return seams_answer(media, *resolve_shots(video, media, shots))


def seams_answer(media, records, source):
    """`analyze_seams` over an already-read Media and resolved shots."""
    if not records or len(records) < 2:
        return _answer(
            "analyze_seams",
            {"seams": []},
            _shot_span_findings("analyze_seams", records, media),
            source,
            shot_dependent=True,
        )

    seams = []
    findings = _shot_span_findings("analyze_seams", records, media)
    for index in range(1, len(records)):
        previous, shot = records[index - 1], records[index]
        fade = int(shot.get("overlap_frames") or 0)
        start_frame = int(shot.get("start_frame", 0))
        seam_frame = start_frame + fade / 2.0
        seconds = seam_frame / media.fps if media.fps else None
        record = {
            "seam": index,
            "between": [previous.get("name"), shot.get("name")],
            "seconds": _round(seconds, 3),
            "kind": "dissolve" if fade else "cut",
            "hard_cut": bool(shot.get("hard_cut")),
        }
        skip = set()
        if media.audio is not None and media.sample_rate:
            start, _count, _source = _sample_span(shot, media)
            if start is not None:
                fade_samples = (
                    int(round(fade / media.fps * media.sample_rate))
                    if fade and media.fps
                    else 0
                )
                record.update(
                    _seam_audio(
                        media,
                        start,
                        start + fade_samples,
                        _shot_rms(media, previous),
                        _shot_rms(media, shot),
                    )
                )
                if (
                    record["before_rms_dbfs"] is None
                    or record["after_rms_dbfs"] is None
                    or record["before_rms_dbfs"] <= HOLE_VOICED_DBFS
                    or record["after_rms_dbfs"] <= HOLE_VOICED_DBFS
                ):
                    skip.add("seam_hole")
        else:
            skip.add("seam_hole")
        if media.thumbs is not None:
            record.update(_seam_video(media, previous, shot, fade))
        if record["hard_cut"]:
            skip.add("seam_frame_jump")
        seams.append(record)
        findings.extend(
            _findings(
                "analyze_seams",
                record,
                {
                    "seam": index,
                    "between": record["between"],
                    "seconds": record["seconds"],
                },
                skip,
            )
        )
    return _answer(
        "analyze_seams", {"seams": seams}, findings, source, shot_dependent=True
    )


def analyze_sync_drift(video, shots=None):
    """Task command: how far the soundtrack sits from the picture, shot by
    shot and over the whole file.

    Args:
        video: A video file's path, or the video an earlier step returned
        shots: Shot records to measure by, overriding any the video carries

    Returns:
        {shots: [{name, start_offset_ms, end_offset_ms}], max_offset_ms,
        video_seconds, audio_seconds, length_delta_ms, findings,
        rules_applied, rules_skipped, shots_source}
    """
    media = media_from(video)
    return sync_drift_answer(media, *resolve_shots(video, media, shots))


def sync_drift_answer(media, records, source):
    """`analyze_sync_drift` over an already-read Media and resolved shots."""
    measured = []
    findings = _shot_span_findings("analyze_sync_drift", records, media)
    rate = media.sample_rate
    fps = media.fps
    for shot in records or []:
        start, count = shot.get("start_sample"), shot.get("num_samples")
        if start is None or count is None or not rate or not fps:
            continue
        start_frame = int(shot.get("start_frame", 0))
        end_frame = start_frame + int(shot.get("num_frames", 0))
        record = {
            "name": shot.get("name"),
            "start_offset_ms": _round((int(start) / rate - start_frame / fps) * 1000.0),
            "end_offset_ms": _round(
                ((int(start) + int(count)) / rate - end_frame / fps) * 1000.0
            ),
        }
        measured.append(record)
        findings.extend(
            _findings(
                "analyze_sync_drift",
                record,
                {
                    "shot": record["name"],
                    "seconds": _round(end_frame / fps, 3),
                },
            )
        )

    length_delta = None
    if media.audio_seconds is not None and media.video_seconds is not None:
        length_delta = _round((media.audio_seconds - media.video_seconds) * 1000.0)
    answer = {
        "shots": measured,
        "max_offset_ms": (
            max((record["end_offset_ms"] for record in measured), key=abs)
            if measured
            else None
        ),
        "video_seconds": _round(media.video_seconds, 4),
        "audio_seconds": _round(media.audio_seconds, 4),
        "length_delta_ms": length_delta,
    }
    findings.extend(
        _findings(
            "analyze_sync_drift",
            {"length_delta_ms": length_delta},
            {"file": True},
        )
    )
    return _answer("analyze_sync_drift", answer, findings, source)


__all__ = [
    "Media",
    "analyze_seams",
    "analyze_shots",
    "analyze_sync_drift",
    "media_from",
    "read_media",
    "resolve_shots",
    "seams_answer",
    "shots_answer",
    "sync_drift_answer",
]
