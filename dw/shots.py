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


def trimmed_shots(shots, head_trim):
    """The shots of a video after dropping `head_trim` frames off its start.

    concat_videos trims the head of every video after the first before
    joining it. A shot entirely inside the trim never reaches the joined
    picture and is dropped; one straddling the cut survives, clipped to what
    is left and re-based to start at 0, so a later frame offset places it
    correctly. The crossfade drawn from the trimmed material makes the
    surviving samples' position in the joined track unmeasurable, so the
    sample side is cleared regardless of rate.
    """
    if not head_trim:
        return shots
    clipped = []
    for shot in shots:
        end = shot["start_frame"] + shot["num_frames"]
        if end <= head_trim:
            continue
        start = max(shot["start_frame"], head_trim)
        clipped.append(
            {
                **shot,
                "start_frame": start - head_trim,
                "num_frames": end - start,
                "start_sample": None,
                "num_samples": None,
            }
        )
    return clipped


def nested_shots(
    shots, frame_offset, sample_offset, native_rate, target_rate, fps=None
):
    """An input's own shots, offset onto where the whole input landed in a join.

    Frames are exact: a join only ever adds frames before an input, never
    inside it, so `start_frame + frame_offset` is where each inner shot now
    sits. Samples are only ever offset when the join measured where the
    input's own track landed (`sample_offset`) and both rates are known -
    resampling a partial waveform inside the crossfaded region is not a
    measurement, so trimmed_shots already clears those before this runs.
    Otherwise the sample side is cleared, same as without_samples.

    A caller that knows the joined track's frame rate (`fps`) can have the
    sample side *derived* from each shot's new frame position
    (frames_to_samples) instead of rescaled from its own already-rounded
    `start_sample` - the same choice #401 made for the top-level seam
    position, because rescaling a stored value compounds whatever rounding
    an earlier join already did, drifting a sample or two off what a later
    pair_audio would measure for the same boundary (#405). concat_videos
    does not pass `fps` here: its sample_offset is a measurement of the
    real, unevenly-spaced crossfades it drew, not a multiple of a frame
    rate, so deriving from frame position would disagree with the track it
    actually built.
    """
    rescale = (
        target_rate / native_rate
        if sample_offset is not None and native_rate and target_rate
        else None
    )
    derive = fps and target_rate and sample_offset is not None
    offset = []
    for shot in shots:
        entry = {**shot, "start_frame": shot["start_frame"] + frame_offset}
        start_sample = shot.get("start_sample")
        if derive:
            entry["start_sample"] = round(entry["start_frame"] / fps * target_rate)
        elif rescale is not None and start_sample is not None:
            entry["start_sample"] = sample_offset + round(start_sample * rescale)
        else:
            entry["start_sample"] = None
            entry["num_samples"] = None
        offset.append(entry)
    return offset


def measured_num_samples(shots, total_samples):
    """Fill each shot's `num_samples` from where the next measured one starts.

    A shot's track runs up to wherever the next shot with a known
    `start_sample` begins, or to the end of the joined track for the last
    one - shared by concat_videos and dissolve_videos so nesting an input's
    shots (which can leave some entries with no `start_sample`) is handled
    the same way in both.
    """
    for index, shot in enumerate(shots):
        if total_samples is None or shot["start_sample"] is None:
            shot["num_samples"] = None
            if total_samples is None:
                shot["start_sample"] = None
            continue
        end = total_samples
        for following in shots[index + 1 :]:
            if following["start_sample"] is not None:
                end = following["start_sample"]
                break
        shot["num_samples"] = end - shot["start_sample"]
    return shots


def shot_reference_names(references):
    """A name per entry of a step's list naming a shot, else None.

    A step's `videos` argument is written as a list of `previous_result:`
    references; by the time the task runs `gather:` has expanded into exactly
    such a list, so the entry at position i names the video the join put at
    position i. A reference to a `shot@` member keeps that name (the
    for_each entry, not the field read off it); any other `previous_result:`
    reference is named after the step it points at, so a shot generated by an
    ordinary step (a `pair_audio`, a chain) is traceable in the manifest and
    in a probe finding the same way (#396). Anything else (an `asset:` path,
    a literal video) keeps the name the join gave it.
    """
    if not isinstance(references, list):
        return None
    names = []
    for reference in references:
        if isinstance(reference, str) and reference.startswith(SHOT_REFERENCE_PREFIX):
            # `previous_result:shot@x.field` names the member, not the field
            member = reference[len(PREVIOUS_RESULT_PREFIX) :]
            names.append(member.split(".", 1)[0])
        elif isinstance(reference, str) and reference.startswith(
            PREVIOUS_RESULT_PREFIX
        ):
            # `previous_result:step.field` names the step, not the field
            step = reference[len(PREVIOUS_RESULT_PREFIX) :]
            names.append(step.split(".", 1)[0])
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


def _rename_in_place(shots, names):
    """Write the step-named `name`s back onto the shot dicts themselves.

    `Result.save` stores `saved_shots[path]` as the artifact's own `.shots`
    list, not a copy (`self._artifacts_for` / `getattr(artifact, "shots")`),
    and that same artifact is what a later `previous_result:` step reads
    (`results[step.name]` holds it directly). Renaming only the manifest's
    deep copy left that artifact carrying the join's `video N` fallback, so
    a probe reading `previous_result:cut` still saw the unnamed shot even
    after the manifest was fixed (#396 follow-up). Mutating here reaches
    both.
    """
    for shot, named in zip(shots, named_shots(shots, names)):
        if named is not shot:
            shot["name"] = named["name"]


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
    for path in files:
        _rename_in_place(saved_shots[path], names)
    if len(files) == 1 and len(saved_files) == 1:
        return copy.deepcopy(saved_shots[files[0]])
    return [
        {**shot, "file": path}
        for path in files
        for shot in copy.deepcopy(saved_shots[path])
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
