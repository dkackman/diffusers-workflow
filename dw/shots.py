"""Shot boundaries a joined video carries: where each input landed in it.

A step that joins shots - `concat_videos`, `dissolve_videos`, a chained
pipeline - knows exactly where every seam fell, in frames and in samples, and
used to throw that away: a consumer checking a cut had to re-derive the seams
from arguments, and a shot whose track ran 267 samples long drifted the rest
of the cut with nothing saying where (#378). The join now records one entry
per shot on the `AudioVideo` it returns (`AudioVideo.shots`), the step's
manifest entry carries them, and `get_gallery_metadata` reads them back.

A shot is a dict:

- `name` - which input it was: `shot@<key>` when the step named a `for_each`
  member, else the path it was given, else `video N` / `segment N`
- `start_frame`, `num_frames` - its place on the joined picture. The shots
  partition the frames: the counts add up to the file's frame count
- `start_sample`, `num_samples` - its place on the joined track, *measured*
  from the waveform the join built rather than derived from the frame
  numbers, so an overrun shows up as a count that disagrees with the frames'.
  None when the video has no track, or when the track is one the join did
  not build shot by shot (a chain's `match_audio`)
- `overlap_frames` - a dissolve's head: the frames at its start that are
  blended with the shot before it

Every other `AudioVideo` constructor either carries the list (same frames),
rescales it (`interpolate_frames`), re-measures the sample side for a new
track (`pair_audio`), or builds a video with no shots at all.
`tests/test_shots.py` fails on a constructor site nobody decided for.
"""

import copy

from .arguments import PREVIOUS_RESULT_PREFIX

# The step names a for_each member as `<group>@<entry>`; only a member of the
# group conventionally called `shot` names a shot
SHOT_REFERENCE_PREFIX = f"{PREVIOUS_RESULT_PREFIX}shot@"


def shot_record(
    name, start_frame, num_frames, start_sample=None, num_samples=None, **extra
):
    """One shot's entry, in the key order the manifest shows."""
    record = {
        "name": name,
        "start_frame": int(start_frame),
        "num_frames": int(num_frames),
        "start_sample": None if start_sample is None else int(start_sample),
        "num_samples": None if num_samples is None else int(num_samples),
    }
    record.update(extra)
    return record


def carried_shots(source):
    """The shots of a video whose frames a step kept one for one, copied."""
    shots = getattr(source, "shots", None)
    return copy.deepcopy(shots) if shots else None


def without_samples(shots):
    """The shots with their sample side cleared - a track that is gone."""
    return [{**shot, "start_sample": None, "num_samples": None} for shot in shots]


def rescaled_shots(shots, multiplier):
    """The shots of a video whose frames were multiplied by interpolation.

    N frames become (N - 1) * multiplier + 1: every frame but the last gains
    multiplier - 1 frames after it. A shot starting at frame s now starts at
    s * multiplier, and the last shot keeps the one frame nothing follows.
    Interpolation drops the track, so the sample side goes with it.
    """
    if not shots:
        return None
    total = sum(shot["num_frames"] for shot in shots)
    rescaled = []
    for shot in shots:
        start = shot["start_frame"] * multiplier
        end = shot["start_frame"] + shot["num_frames"]
        new_end = (end - 1) * multiplier + 1 if end == total else end * multiplier
        entry = {
            **shot,
            "start_frame": start,
            "num_frames": new_end - start,
            "start_sample": None,
            "num_samples": None,
        }
        if shot.get("overlap_frames"):
            entry["overlap_frames"] = shot["overlap_frames"] * multiplier
        rescaled.append(entry)
    return rescaled


def remeasured_shots(shots, fps, sample_rate, total_samples):
    """The shots of a video laid over a new track by pair_audio.

    The frame side is unchanged. The new track was not built shot by shot, so
    each shot's samples are the stretch of the track its frames play over: a
    shot starts at start_frame / fps seconds, and the last one runs to the end
    of the track as written - with `fit: "video"` that is the fitted length.
    Without a frame rate there is no way to place a frame on the track, so the
    sample side is cleared rather than guessed.
    """
    if not shots:
        return None
    if not fps or not sample_rate or total_samples is None:
        return without_samples(shots)
    starts = [
        min(int(round(shot["start_frame"] / fps * sample_rate)), total_samples)
        for shot in shots
    ]
    ends = starts[1:] + [total_samples]
    return [
        {**shot, "start_sample": start, "num_samples": max(end - start, 0)}
        for shot, start, end in zip(shots, starts, ends)
    ]


def shot_reference_names(references):
    """`shot@<key>` for each entry of a step's list naming a shot, else None.

    A step's `videos` argument is written as a list of `previous_result:`
    references; by the time the task runs `gather:` has expanded into exactly
    such a list, so the entry at position i names the video the join put at
    position i. Only a reference to a `shot@` member names a shot - anything
    else keeps the name the join gave it.
    """
    if not isinstance(references, list):
        return None
    names = []
    for reference in references:
        if isinstance(reference, str) and reference.startswith(SHOT_REFERENCE_PREFIX):
            # `previous_result:shot@x.field` names the member, not the field
            member = reference[len(PREVIOUS_RESULT_PREFIX) :]
            names.append(member.split(".", 1)[0])
        else:
            names.append(None)
    return names


def named_shots(shots, names):
    """The shots with each positional one renamed where the step named it."""
    if not shots or not names or len(names) != len(shots):
        return shots
    return [
        {**shot, "name": name} if name else shot for shot, name in zip(shots, names)
    ]


def step_shots(saved_shots, saved_files, references=None):
    """The `shots` a step's manifest entry and step_end carry, or None.

    `saved_shots` maps each file the step wrote to the shots its video
    carried (`Result.saved_shots`). A step that wrote one file lists that
    file's shots; one that wrote several marks each shot with the `file` it
    belongs to, so no shot is ever read against the wrong file. `references`
    is the step's `videos` argument as written, which names the shots.
    """
    if not saved_shots:
        return None
    names = shot_reference_names(references)
    files = [path for path in saved_files or [] if path in saved_shots]
    if len(files) == 1 and len(saved_files) == 1:
        return named_shots(copy.deepcopy(saved_shots[files[0]]), names)
    return [
        {**shot, "file": path}
        for path in files
        for shot in named_shots(copy.deepcopy(saved_shots[path]), names)
    ]


def shots_for_file(shots, path, step_files):
    """The shots of one file out of a manifest entry's `shots`, or None.

    `path` and `step_files` are as the manifest records them, so a
    run-relative path matches a run-relative `file`.
    """
    if not shots:
        return None
    if any("file" in shot for shot in shots):
        own = [
            {key: value for key, value in shot.items() if key != "file"}
            for shot in shots
            if shot.get("file") == path
        ]
        return own or None
    return shots if list(step_files or []) == [path] else None
