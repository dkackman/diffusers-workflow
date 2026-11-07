# Camera moves classified as static or moving (#1200)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out,
2026-11-07). Plan v2 approved by Don on 2026-11-07 for stage A only; one
stage, #1255, verified the same day by reading the plugin files.

## The idea

#1200 proposed taking about 12 camera moves from a community gallery of 40
(wangweihong/comfyui-cinematic-gallery, which has no licence, so we took only
the concepts), writing our own phrasing for each in the H3 and LTX-2.5
caption formats, and probing them on lem (2 seeds × 2 families). It also
asked for every move to be classified as `static` or `moving`. The long-form
playbook (#1199) picks its H3 `shot` preset from that split, and the counter
bench had found that `carry_pick`/`look_hold` on `last_frame` froze a
moving-camera arc.

## What the design found

- **The vocabulary had 13 move ids, not 6.** Six were measured by the
  Higgsfield move trial (#1148): `static`, `dolly-in`, `dolly-out`,
  `truck-left`, `crane-up`, `orbit`. Seven were ids without a measurement:
  `truck-right`, `crane-down`, `tracking`, `whip-pan`, `push-in`, `handheld`,
  `crash-zoom`. Several of the issue's candidates were already ids.
- **Several candidates are not moves.** Worm's-eye and overhead are angles,
  rack focus is a focus effect, and low arc is a variant of `orbit`.
- **A static/moving rule already existed as a stand-in** in
  `long-form/SKILL.md` ("anything but `static` is moving, until #1200") and
  `levers.md`. It matters for H3 only: shot presets exist only in
  `templates/minimax/chained-segments.json`, and LTX chains keep
  `last_frame`.
- **The freeze came from the bench's S2 arc** (`bench/records/counters.md`,
  *Eye notes (S2)*), not from a vocabulary `orbit` clip. No dolly, truck or
  crane chain was benched.
- **Cost was overstated.** At the move trial's ~2.2 min per clip, the
  issue's 48 clips come to about 2 GPU-hours, not 6-8.

## Decisions (Don, 2026-11-07)

- **Q1 [scope]: new move ids?** No.
- **Q2 [spend]: lem time for the probe?** No. Stage A only. The probe
  (stage B) was filed as its own parked idea, **#1260**, with the plan's
  stage B text as its body.
- **Q3 [none]: classification.** Only `static` is `static`; `push-in` and
  `handheld` are `moving`. As written.

## What was built

**Stage A, #1255.** Commit `fac9c10d`, merged to `develop` as `b8b40adc`.
Plugin-only, so nothing was deployed.

- **`class` column** in `plugins/dw/skills/cinematography/references/vocabulary.md`,
  *`move`*: the table is `| id | class | Means | Note |`, and the id stays
  the first backticked cell. `static` means the camera's position and
  orientation don't change across the take. `moving` is every other id. The
  cost asymmetry decides the borderline moves: a moving shot under
  `shot: static` gets its pose pinned (the measured freeze), while a
  near-static shot under `shot: moving` only loses length.
- **Per-move H3 note.** Every `moving` row reads "H3: never `last_frame`
  with `look_hold`/`carry_pick` (measured on the arc only)", citing the
  bench record.
- **Long-form pointer.** `long-form/SKILL.md` §3 and `references/levers.md`
  point at the column in place of the "#1200" placeholder: `move: static`
  takes `shot: static` (or `static_long`), and every `moving` id takes
  `shot: moving`. `SKILL.md` is 10,595 B, under the 12 KiB cap.
- **Tests** in `tests/test_plugin_skills.py`:
  `test_cinematography_every_move_has_a_static_or_moving_class`,
  `test_cinematography_every_moving_id_carries_the_h3_last_frame_rule`,
  `test_the_move_class_comes_from_the_vocabulary`.
- **Deviation:** one sentence beyond the plan explains that `crash-zoom` is
  `moving` even though the camera body stays put, because the view changes
  across the take. Without it, a strict reading of the definition would put
  it in `static`.

Unchanged: the compile tables, `script-to-video/references/shot-document.md`
(it still names ids only, with no copied list), the engine, the schema, the
templates and the `shot` enum.

## Verification and bounces

| Stage | Bounces | Verified by |
|---|---|---|
| A, #1255 | 0 | C-F724 to C-F727, by reading the skill files as a consumer, plus a validate-only `shot: "moving"` check on `templates/minimax/chained-segments` |

Session cost isn't recorded here: the stage comments name no `usage:`
figures.

## Deferred

- **The measurement probe** (#1260, parked with Don as a `spend` decision).
  It covers the 7 unmeasured moves on two seeds and a second seed on the
  measured 6, on H3 and LTX-2.5: 40 clips, about 1.5 GPU-hours. Until it
  runs, the seven ids stay hints.
- **Benching counters on the borderline moves** (`push-in`, `handheld`)
  belongs to #1225 (a drift counter for moving shots).
- **New move ids** from the community list: declined (Q1).
