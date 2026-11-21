# Face track subject selection (#1321)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out session,
2026-11-14). Don filed the idea on 2026-11-13 from a README read of
[ComfyUI-H3-FaceRefine](https://github.com/Carasibana/ComfyUI-H3-FaceRefine)
(MIT). Its `H3 Face Track + Crop` node picks the subject once per shot and does
not jump to another face. Plan v1 is the plan comment on #1321. It was approved
at its defaults on 2026-11-13 (ruled by a lead session under Don's delegation)
and built as one stage, #1348. The plan comment still holds the full design.
This file records what shipped and what stayed cut.

## The problem

`crop_face_track` (`dw/tasks/face_track.py`) kept one track. It started on the
most confident YuNet detection, which is arbitrary when two faces are visible.
After `MAX_GAP` misses it restarted on the most confident face anywhere in the
frame, so a lost subject could become another person. `paste_face_track` then
blended the repair onto the wrong face at full strength, and a caller had no
argument to say which person was meant. Two-person H3 shots are routine (#484).
No field report showed a wrong-person repair. The defect was found in the code.

## Verdict: build smaller

- **Built:** `select` by size or position, and a lost state that never jumps to
  another face.
- **Cut: `select: "reference"` (identity picture via SFace).** SFace's
  `alignCrop` needs YuNet's landmarks, which `_detector` and `detect_tiled`
  drop. Reusing `face_drift`'s loader means moving it to a leaf module (with the
  cache key `tests/test_face_drift.py` pins), adding SFace to `required_weights`,
  and adding a new media argument to the security docs and `test_locations`.
  That is about $4–5 for one stage. *What would bring it back:* a real take where
  two faces of similar size and position defeat every geometric rule (two
  people side by side swapping places, say).
- **Cut: Gaussian (non-causal) smoothing of centre and size.** It would change
  every frame's strength (strength comes from the smoothed width) and break the
  EMA and hold tests in `tests/test_face_track.py`, and nobody has judged
  whether it looks better. *What would bring it back:* a face-repair output
  where the crop visibly lags or breathes on a moving head. That would make it
  a `taste` A/B on that take.
- **Cut: per-shot `select`.** Shot records accept extra keys, but nothing would
  validate them. *What would bring it back:* a cut clip where one rule can't
  serve every shot.

Non-goals that stayed out: a manual picker UI, a body-detector fallback, anime
detectors, and any MCP, REST or workflow-syntax surface.

## Plan history

- **v1 (approved 2026-11-13).** Q1 [scope] build smaller as above; Q2 [default]
  `select` defaults to `largest`, with `confident` restoring the old rule, and
  the release note says so; Q3 [default] lost frames keep the source pixels
  instead of restarting on another face; Q4 [none] one `select` per clip; Q5
  [none] both face-repair templates expose `select`.

## What was built

### Stage A (#1348): `select` and no jump after a lost track

Merge `50598c64` on `develop` (commits 7ea5afeb engine, 0235e716 templates,
docs and tests, 9ee4e5cd TASKS.md paragraph split).

- **`select`** on `crop_face_track`: `largest` (default; ties go to the more
  confident face), `centre`, `left`, `right`, `confident`. It is applied only
  when a track starts: frame 0, each recorded shot start, each content cut.
  Registered with `choices`, refused at validate time by `face_track_errors`
  (deferred references skipped) and at run time by `face_track_problems`. A
  refusal names the value and the five valid ones.
- **Lost state.** After `MAX_GAP` misses, frames get `box` null, strength 0 and
  `state` `"lost"`, so `paste_face_track` keeps the source pixels. A lost
  track is picked up only (1) by a face within `REACQUIRE_DISTANCE` (3 face
  widths) of the last box, or (2) by the single face of a segment that has
  never had more than one face. Otherwise it stays lost until the next shot
  start or cut, where `select` applies again.
- **Track record:** `select`, and `lost` as inclusive `{start, end}` ranges
  (`[]` when nothing was lost). `resets` keeps its `{frame, reason: "lost"}`
  entry. The log line counts lost frames.
- **Templates:** `templates/minimax/face-repair` and `templates/ltx2/face-repair`
  gain a `select` variable, default `largest`, passed to the crop step.
- **Docs:** `docs/TASKS.md` (argument, tracking, record fields, refusals, the
  behaviour change), the module docstring, the two template READMEs, and an
  `ATTRIBUTIONS.md` entry for H3-FaceRefine (concepts only, no code adapted).

**Bounces: 0.** The first deploy didn't restart the server, because a job was
running on lem and the deploy wasn't forced. A later lead session handed the
stage to the tester once the `dw-serve` unit served the build.

Cases verified: C-F824 to C-F834, all over MCP on CPU-built clips (stacked
`ingredients_grid` portrait panels), with no GPU generation.

## Behaviour change (owed to the release note)

On a clip with more than one face, the default start face changes from the most
confident to the `largest`. `select: "confident"` restores the old rule.
Frames after a lost track keep the source pixels instead of being repaired on
whichever face the tracker restarted on. Single-face clips are unchanged
except where a subject was lost and no pick-up rule applies.

## Cost and bounces

| Stage | Bounces | Cost |
|---|---|---|
| #1348 | 0 (one deploy waited on a running job) | Not recorded on the stage. The plan estimated $3–5 |

## Deferred and open

- **`reference`, Gaussian smoothing, per-shot `select`:** cut, triggers above.
- **#1358** (found during the verify): a crop step with
  `result.content_type: "application/json"` validates clean, but the run fails
  at save with `Object of type AudioVideo is not JSON serializable`. Not a
  `select` finding.
- Cosmetic, from the verify: a JSON `true` given as `select` is echoed as
  `True` in the refusal.
- Related: `face-repair-complete.md` cut multi-face tracking from v1. This
  feature still tracks one face. It only makes the choice of which face
  explicit.
