# GYRE - build log

A record of what was made for this piece and what was changed in the engine to
make it, so both can be reviewed independently of the film.

## The piece

**GYRE** - one shape, nine worlds. A spiral holds the exact centre of a square
frame and turns clockwise for the whole film. Every 5.17 seconds the material of
it is replaced on a hard cut, moving outward through scale: an iris, a
fingerprint, a fern fiddlehead, a nautilus, a dew-strung orb web, a whirlpool, a
hurricane, a galaxy - and back to an eye, with the galaxy reflected in it.

The cut is the whole idea. The eye tracks a shape it believes it is still
watching while the universe underneath it is swapped out. That only works if
every spiral sits at the same point on screen at the same size, which is what
the registration pass below exists to guarantee.

## Passes

| File | What it does |
| ---- | ------------ |
| `GyreStills.json` | Z-Image Turbo paints the nine source stills at 1024x1024, and MiniMax-Music3 writes two candidate scores |
| `GyreFrames.json` | Registration: `recenter_crop` re-frames each still to 704x704 around its own measured spiral centre, at a per-still scale, so all nine agree |
| `GyreFilm.json` | MiniMax-H3 (fl2va) animates each registered still into a 124-frame shot, `concat_videos` cuts them together, `pair_audio` lays the unbroken score over the edit |

## Changes to the engine

### `recenter_crop` image processor - `dw/tasks/image_utils.py`

New. Takes a square window `crop` of the shorter side across, centred on
`(center_x, center_y)` in normalised 0-1 coordinates, and resizes it to
`width` x `height`. The window may run off the edge of the source - a feature
near a border is the case that needs moving furthest - and `fill` decides what
lies outside: `"edge"` replicates the border pixels, anything else is read as a
PIL colour.

Registered in the `_PROCESSORS` table beside `crop_square`, so it is available
as a task command with no further plumbing.

Why it was needed: a generative model will not put a subject on an exact pixel
just because the prompt asks it to, and across nine unrelated prompts it never
puts them in the same place twice. Measuring each spiral's centre once and
re-framing around it is both more reliable and far cheaper than re-rolling
stills until they happen to line up. It generalises past this film to any
sequence that has to cut or dissolve in register.

### `mix_audio` task command - `dw/tasks/audio_utils.py`

New, registered in `dw/tasks/task.py`. Layers audio tracks on top of one
another and returns the weighted sum; `crossfade_audio` already joined tracks
end to end, but nothing in the engine put one *under* another. Shorter tracks
are padded with silence to the longest. It does not rescale the result, because
quietening a mix is a decision about how it should sound - follow it with
`normalize_audio`.

Why it was needed: H3 generates a soundtrack for each world alongside the
picture, and the existing pattern (`concat_videos` then `pair_audio`) throws all
of it away in favour of the score. For this film that is the wrong trade - the
sound of the whirlpool, the hurricane and the surf changing at every cut while
the music runs unbroken *is* the idea, in the ear instead of the eye. GYRE mixes
the nine worlds at 0.55 under the score at 1.0 and normalizes to -1 dBFS.

### `stabilize_video` task command - `dw/tasks/stabilize.py`

New module, registered in `dw/tasks/task.py`. Measures the translation between
consecutive frames by phase correlation, accumulates it into the clip's drift,
and shifts every frame back so the framing at the end is the framing at the
start. `smooth=0` locks to the first frame; a window in frames instead removes
only the wander faster than that window, so a slow deliberate camera move
survives. The result is cropped to the rectangle every frame still covers and
resized back to the original size, so no exposed edges appear.

Why it was needed, and this one is the real find: **H3 drifts.** Asked for a
locked camera on a pinned keyframe, the test shot still slid 38px down a 704px
frame over 124 frames - 5% of the picture, invisible within the shot and glaring
the moment two shots are cut together, because the subject snaps back to centre
at every cut. That would have quietly wrecked the whole premise of this film
after an hour of GPU time. Phase correlation takes the measured drift from 38px
to 2px.

Note on the sign convention: the peak of the cross-power spectrum sits at the
*negative* of the displacement. `_pair_shift` negates it so every caller can
read `(dx, dy)` as "how far the picture moved", which is worth knowing because
getting it backwards doubles the drift instead of removing it, and looks
plausible enough in a single number to miss.

## Things tried and abandoned

- **Automatic spiral-centre detection.** A centre-of-rotational-symmetry finder
  (score each candidate centre by how well the image matches a rotated copy of
  itself about that point). It found the flat, blurred backgrounds instead of
  the spirals, because a featureless patch correlates perfectly with any
  rotation of itself. Weighting the score by local contrast helped and did not
  fix it. Registration was measured by eye instead, and the utility was not
  shipped - an unreliable one is worse than none. The prototype is in the
  session scratchpad, not the repo.
- **A continuous clockwise camera roll across all nine shots**, so the whole
  film would be one unbroken turn. H3 ignored `Roll Clockwise with small
  amplitude at slow speed` entirely - the test shot has no roll in it. Doing it
  in post would mean zooming ~1.41x to keep the corners covered, which throws
  away a third of the resolution for a decorative flourish. Dropped in favour of
  fixing the drift, which the concept actually depends on.
- **`fill="edge"` when a crop window runs off the source.** Replicated border
  pixels leave visible streaks against texture. `reflect` and `symmetric` were
  added to `recenter_crop` for it; the nautilus uses `symmetric` and the
  mirrored volcanic sand is undetectable.

## What the first full run taught

Three things only showed up once the whole thing existed, and all three are now
fixed in the workflows.

**The score was passed as `{"location": ...}`.** The engine resolves that into a
bare waveform, which carries no sample rate, so `slice_audio` refused it - after
all nine shots had been generated. Every expensive step writes its shot out, so
nothing was lost, but it is the reason `GyreAssemble.json` exists: the edit half
of the film, run over the shots already on disk. Re-cutting a different mix
balance now costs a minute instead of an hour.

**`stabilize_video`'s argument could not be called `video`.** `dw/arguments.py`
loads any argument named `video` or `*_video` itself, with diffusers'
`load_video`, which returns bare frames and drops the soundtrack. So a path
passed under that name arrived stripped of its audio *and* as a list, which the
previous-result machinery then fanned out one iteration per frame - nine shots
of 124 frames asked the engine for 124^9 combinations. The argument is `clip`
for that reason, which is why `concat_videos` can take paths too: `videos`
escapes the same magic by being plural.

**The world sound was inaudible.** At `world_gain` 0.55 against the score's 1.0,
H3's own soundtrack sat 22 dB under the music - present in the file, absent to
the ear, which defeats the whole reason `mix_audio` was written. The measurement
worth keeping is the method: project the mix onto the score, and whatever is
left over is the world. `world_gain` 1.8 puts it 13 dB down, which is a bed you
hear change at every cut without it competing with the music.

## Shot 4 took three attempts

The nautilus is the only shot that had to be re-generated, and it is a good
illustration of what H3 will and will not hold. The first prompt asked for a
sheet of seawater to slide in across the sand and flood the chambers; by
mid-shot the black volcanic sand had dissolved into a grey wash with circular
blob artifacts and the shell was covered in black speckles. The second held the
scene still but still asked for water, and H3 rendered it as a grey smear
sweeping through the frame. The third removed water from the shot entirely and
made light the only thing that moves - iridescence travelling around the spiral,
wet sand grains glittering and going dark - and it is clean end to end.

The lesson generalises: H3 is reliable when the frame's *contents* stay put and
something about them changes, and unreliable when it has to move new material
through a scene it was given as a still. `GyreReshoot.json` is kept for this -
re-generating one shot without disturbing the other eight.

## The dissolve variant

[`GyreDissolve.json`](GyreDissolve.json) replaces `concat_videos` with
`dissolve_videos`, softening every cut into a cross-dissolve.

It turns out to be a different piece rather than a softer one. Because all nine
spirals sit on the same centre at the same size, an overlap does not read as a
fade between two pictures - at the midpoint of the seam the two subjects are
concentric, so the pupil sits inside the fingerprint's whorl and the iris fibres
read as ridges. The shape holds still and its *material* transforms through the
seam. Registration bought this for free; it would look like an ordinary mix
between any two unregistered shots.

The arithmetic has to be passed in, since JSON cannot compute it:
`dissolve_videos` consumes one overlap per seam, so
`total_frames = 1116 - 8 * dissolve_frames`, and
`start_frame = round((50.05 - total_frames / 24) * 24)` keeps the piece ending
on the score's final chord. Rendered at 6 frames (0.25s, 44.50s total) and 12
frames (0.50s, 42.50s total).

## A per-step seed does not isolate the RNG stream

Worth writing down, because it looks like it should. `dw/workflow_schema.json`
lets a step carry its own `seed` ("Default seed for the entire step"), so the
three stills workflows looked consolidatable into one that names, per step, the
seed its chosen image was drawn on. Tried it: only `still_1_iris` came back
byte-identical. The other eight differed by a mean of 11 to 109 levels per
pixel - different pictures, not drifted ones. A step's result still depends on
its position in the run, not on its seed alone.

That is why `GyreStills.json`, `GyreStillsFix.json` and `GyreStillsFix2.json`
are all kept rather than merged. Together they reproduce the nine source stills
exactly, and they have to: `GyreFrames.json`'s crop centres are hand-measured on
those exact images, so regenerating the stills differently would leave its
numbers pointing at nothing while still validating and still running.

## Cleanup

Intermediates removed (124 MB): the stills and re-shoot runs, the four
registration passes, the single-shot probe, the two discarded nautilus takes,
and the three superseded cuts of the film. Everything the pipeline needs was
first copied into `assets/` - nine source stills, nine registered frames, nine
generated shots, both scores - and `GyreAssemble.json` and `GyreDissolve.json`
were re-run afterwards to confirm they still rebuild both films byte-identically
from those assets alone.

`assets/` was itself removed from the repository afterwards - 30 MB across the
nine stills, the nine frames, the nine shots and both scores, none of which the
engine needs to be in version control. The workflows still address those paths,
so the directory has to be restored from the archived copy before passes 2 and
3, `GyreAssemble.json`, `GyreDissolve.json` or `GyreReshoot.json` will run.
Re-running the workflows gets close but does not get back: the re-shoot takes
and the choice between the two scores were both hand-picked and neither
selection was recorded. The rejected instrumental score is in the archive too,
although no workflow reads it.

`GyreTest.json`, the single-shot probe, was deleted as redundant:
`GyreReshoot.json` does the same job and is needed anyway.

The pytest files added with the three new utilities - `tests/test_stabilize.py`,
`tests/test_mix_audio.py`, `tests/test_recenter_crop.py` - are kept. They cover
shipped engine code rather than this film, and the phase-correlation sign
convention in particular is the kind of thing that fails silently.
