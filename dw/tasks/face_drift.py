"""analyze_face_drift: does a recurring character still look like their cast
portrait (#1146)?

Given a clip and a reference face, YuNet finds the faces in a few frames of
each shot, SFace embeds them, and the cosine match against the reference is
reported per shot: the median and the minimum, over the sampled frames that
had a face, of the best match among the faces in that frame. The answer has
the shape of the assessment probes (`dw/tasks/assess.py`) - every
measurement, `findings`, `rules_applied`, `shots_source` - and its findings
are places to look, not verdicts. It is not an assessment probe itself
(`assessment=True` is for probes that read a finished cut's soundtrack and
seams) and its two rules live here, as `measure_sync`'s do.

The reference is one face. With none the call is refused; with several the
largest is used and `notes` says so.

Models: YuNet through `face_track.face_detector_model` (the one loader) and
SFace the same way, both at a pinned revision and through the task model
cache. OpenCV works in BGR; the pictures here are RGB until they reach it.
"""

import logging

import numpy as np
from PIL import Image

from ..assessment_rules import finding, sort_findings
from ..task_domains import real_number
from ..task_problems import face_drift_problems
from .assess import media_from, resolve_shots
from .face_track import (
    DEFAULT_DETECTOR_FILE,
    DEFAULT_DETECTOR_REPO,
    face_detector_model,
    required_weights as detector_weights,
)
from .weights import file_entries

logger = logging.getLogger("dw")

COMMAND = "analyze_face_drift"

# OpenCV's own upload of SFace, read at a pinned revision so a later push to
# the repo cannot change what an existing workflow measures
SFACE_REPO = "opencv/face_recognition_sface"
SFACE_FILE = "face_recognition_sface_2021dec.onnx"
SFACE_REVISION = "3d7082438a6e4551e840c9b2bb60b71e8da4b524"

# cv2.FaceRecognizerSF_FR_COSINE
COSINE = 0

# OpenCV's published cosine threshold for SFace: at or above it two faces
# are the same person
DEFAULT_THRESHOLD = 0.363
DEFAULT_SAMPLES = 8
# The detector score below which a face is ignored, as crop_face_track's
DETECTOR_CONFIDENCE = 0.6
PLACES = 3

# What a finding's `at` carries of its shot's measurement
AT_KEYS = ("shot", "index", "start_frame", "num_frames", "start", "end")

RULES = {
    "face_drift": {
        "name": "face_drift",
        "severity": "warn",
        "says": (
            "the faces in this shot sit this close to the reference, below the"
            " threshold - the character may not look like their portrait here"
        ),
    },
    "no_face": {
        "name": "no_face",
        "severity": "info",
        "threshold": 1,
        "says": "no face was found in the frames sampled from this shot",
    },
}


def required_weights(arguments):
    """SFace and YuNet's weight files, for the plan (dw/tasks/weights.py)."""
    return file_entries(SFACE_REPO, SFACE_FILE, SFACE_REVISION) + detector_weights({})


def _models(device=None):
    """(detect, recognizer): `detect(bgr)` is every face YuNet finds as a full
    row (box, landmarks, score), `recognizer` an SFace FaceRecognizerSF."""
    import cv2

    from .model_cache import cached_model

    model = face_detector_model(
        DEFAULT_DETECTOR_REPO, DEFAULT_DETECTOR_FILE, DETECTOR_CONFIDENCE, device
    )

    def load():
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id=SFACE_REPO, filename=SFACE_FILE, revision=SFACE_REVISION
        )
        return cv2.FaceRecognizerSF.create(path, "")

    recognizer = cached_model(
        ("analyze_face_drift", SFACE_REPO, SFACE_FILE, SFACE_REVISION), load
    )

    def detect(bgr):
        height, width = bgr.shape[:2]
        model.setInputSize((width, height))
        _, faces = model.detect(bgr)
        return [] if faces is None else list(faces)

    return detect, recognizer


def _embeddings(detect, recognizer, picture):
    """[(area, embedding)] for each face in an RGB picture, largest first."""
    bgr = np.ascontiguousarray(np.asarray(picture.convert("RGB"))[..., ::-1])
    faces = []
    for row in detect(bgr):
        area = float(row[2]) * float(row[3])
        faces.append((area, recognizer.feature(recognizer.alignCrop(bgr, row))))
    return sorted(faces, key=lambda face: -face[0])


def _reference_picture(reference):
    """The reference as a PIL image: an image, an array, or a stored path."""
    if reference is None:
        raise ValueError(f"{COMMAND} needs 'reference' - a picture of the face to hold")
    if isinstance(reference, Image.Image):
        return reference
    if isinstance(reference, np.ndarray):
        return Image.fromarray(reference)
    if isinstance(reference, str):
        from ..locations import validate_media_path

        path = validate_media_path(reference, None, "the reference picture")
        with Image.open(path) as opened:
            return opened.convert("RGB")
    raise ValueError(
        f"{COMMAND}: 'reference' must be an image or the path of one, "
        f"not {type(reference).__name__}"
    )


def _reference_embedding(detect, recognizer, reference, notes):
    faces = _embeddings(detect, recognizer, _reference_picture(reference))
    if not faces:
        raise ValueError(
            f"{COMMAND}: no face found in 'reference' - give a picture with the "
            "character's face in it"
        )
    if len(faces) > 1:
        notes.append(
            f"reference holds {len(faces)} faces; the largest was used. Crop "
            "the portrait to the one face to be sure of the match"
        )
    return faces[0][1]


def _span(start, count, total):
    """A shot's (first, last) frame numbers, clipped to the clip; last is exclusive."""
    first = max(0, min(int(start), total))
    return first, max(first, min(int(start) + int(count), total))


def _sample_indexes(start, count, total, samples):
    """Up to `samples` evenly spaced frame numbers of a shot, inside the clip."""
    first, last = _span(start, count, total)
    span = last - first
    if span <= 0:
        return []
    if span <= samples:
        return list(range(first, last))
    return sorted({first + (span * i) // samples for i in range(samples)})


def _stored_path(video):
    """The validated file a 'video' names, or None for an in-memory clip."""
    from ..locations import validate_media_path
    from .video_utils import VideoFileReference

    if isinstance(video, VideoFileReference):
        video = video.path
    if isinstance(video, str):
        return validate_media_path(video, None, "a video to measure")
    return None


def _picture_at(video, index):
    """One frame as a PIL image, seeking in a stored file rather than decoding it."""
    path = _stored_path(video)
    if path is not None:
        from ..media_frames import frames_at

        return frames_at(path, [f"frame:{index}"])[0]["image"]
    from .video_utils import extract_frame

    return extract_frame(video, index)


def _best_match(recognizer, embedding, faces):
    return max(float(recognizer.match(embedding, face, COSINE)) for _, face in faces)


def _seconds(frame, fps):
    return round(frame / float(fps), PLACES) if fps else None


def _where(index, name, first, last, fps):
    """Where a shot sits in the clip: its frames and, given an fps, its times
    in seconds - two shots with the same name stay apart by these (#1165)."""
    return {
        "shot": name,
        "index": index,
        "start_frame": first,
        "num_frames": last - first,
        "start": _seconds(first, fps),
        "end": _seconds(last, fps),
    }


def _shot_measurement(where, matches, sampled, faces_found):
    return {
        **where,
        "frames_sampled": sampled,
        "faces_found": faces_found,
        "similarity_median": round(float(np.median(matches)), PLACES)
        if matches
        else None,
        "similarity_min": round(float(min(matches)), PLACES) if matches else None,
    }


def _measure_shot(video, index, shot, media, models, embedding, samples):
    detect, recognizer = models
    start = int(shot.get("start_frame", 0))
    count = shot.get("num_frames")
    if count is None:
        count = media.frame_count - start
    first, last = _span(start, count, media.frame_count)
    where = _where(index, shot.get("name"), first, last, media.fps)
    indexes = _sample_indexes(start, count, media.frame_count, samples)
    pictures = (_picture_at(video, frame) for frame in indexes)
    matches = _matches(detect, recognizer, embedding, pictures)
    return _shot_measurement(where, matches, len(indexes), len(matches))


def _matches(detect, recognizer, embedding, pictures):
    """The best match against `embedding` of each picture that holds a face."""
    matches = []
    for picture in pictures:
        faces = _embeddings(detect, recognizer, picture)
        if faces:
            matches.append(_best_match(recognizer, embedding, faces))
    return matches


def face_similarities(frames, reference, samples, device=None):
    """The SFace similarity to `reference` of up to `samples` evenly spaced
    frames of an in-memory clip, one per sampled frame that holds a face (the
    best match among its faces) - `analyze_face_drift`'s scoring, for a caller
    holding frames rather than a file (the chain check's `face_drop`).

    `frames` are PIL images or RGB arrays; `reference` an image, an array or
    a stored path, whose largest face is held. A reference with no face is
    refused (ValueError); a clip with no face answers []."""
    detect, recognizer = _models(device)
    embedding = _reference_embedding(detect, recognizer, reference, [])
    total = len(frames)
    pictures = (
        _reference_picture(frames[index])
        for index in _sample_indexes(0, total, total, int(samples))
    )
    return _matches(detect, recognizer, embedding, pictures)


def _findings(measurements, threshold):
    found = []
    for record in measurements:
        at = {key: record[key] for key in AT_KEYS}
        if record["faces_found"] == 0:
            found.append(finding(RULES["no_face"], 0, at))
        elif record["similarity_median"] < threshold:
            rule = {**RULES["face_drift"], "threshold": threshold}
            found.append(finding(rule, record["similarity_median"], at))
    return sort_findings(found)


def _checked_settings(threshold, samples_per_shot):
    """(threshold, samples) as numbers, or ValueError in the static check's
    words: `face_drift_problems` is the one rule for both (#692)."""
    threshold = real_number(threshold, "threshold", COMMAND, required=True)
    samples = real_number(samples_per_shot, "samples_per_shot", COMMAND, required=True)
    problems = face_drift_problems(threshold, samples)
    if problems:
        raise ValueError(problems[0][1])
    return float(threshold), int(samples)


def analyze_face_drift(
    video,
    reference,
    shots=None,
    threshold=DEFAULT_THRESHOLD,
    samples_per_shot=DEFAULT_SAMPLES,
    device=None,
):
    """Task command: how close the faces in each shot are to a reference face.

    Args:
        video: A stored video (asset:, output: or a path) or the video an
            earlier step returned
        reference: The face to hold - an image or the path of one. With
            several faces the largest is used; with none the call is refused
        shots: Shot records to measure, else the video's own, else the run
            manifest beside the file, else the whole file is one shot
        threshold: Median cosine similarity below which a shot is a
            `face_drift` finding (-1 to 1); 0.363 is OpenCV's SFace threshold
        samples_per_shot: Frames read from each shot, evenly spaced (1 to 64)
        device: Where the models run

    Returns:
        {measurements: [{shot, index, start_frame, num_frames, start, end,
        frames_sampled, faces_found, similarity_median, similarity_min}],
        findings, rules_applied, shots_source, notes}. `start`/`end` are
        seconds (None when the clip has no fps); each finding's `at` carries
        the shot's name, index, frames and times
    """
    threshold, samples = _checked_settings(threshold, samples_per_shot)
    models = _models(device)
    notes = []
    embedding = _reference_embedding(*models, reference, notes)

    media = media_from(video)
    records, source = resolve_shots(video, media, shots)
    if not records:
        records = [{"name": "whole", "start_frame": 0, "num_frames": media.frame_count}]
        notes.append(
            "no shot boundaries found; the whole file was measured as one shot"
        )
    measurements = [
        _measure_shot(video, index, shot, media, models, embedding, samples)
        for index, shot in enumerate(records)
    ]
    return {
        "measurements": measurements,
        "findings": _findings(measurements, threshold),
        "rules_applied": list(RULES),
        "shots_source": source,
        "notes": notes,
    }
