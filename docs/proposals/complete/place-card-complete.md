# Place cards: one still per recurring set, as an H3 Ref2VA reference (#1577)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out,
2026-11-22). Plan v1 approved on 2026-11-22 by the plan-review delegate under
Don's 2026-11-21 delegation (every question at its default) as "build
smaller": a probe, then skill text. Two stages, #1581 (probe, lem) and #1582
(skill text, mini-ai and lem), both verified. The probe record is
`bench/records/place-card.md` (PRELIMINARY); the skill text is
`plugins/dw/skills/minimax-h3/references/cuts.md`, *A place card holds the
set*. This doc is the design record around them.

## The idea

WanGP Showrunner (deepbeepmeep/WanGP at f4cfc12, v20; WanGP Community
License 2.0, so concepts only, no code) treats a recurring location as a
cast member with role "place": one wide master of the set, no cast, no text,
passed as a reference into every shot set there, close shots and no-people
shots included. A reverse angle gets its own card made by image editing, and
a changed state of the set is a "look".

## Decisions

- **Verdict: build smaller.** No field report records a set lost across
  shots; the demand was measured on the bench (#1398's B0 reverse lost 3 of
  4 set markers with the portrait only). The skills already carried an
  unprobed "keep a wide master too". WanGP's "every shot" rule was treated as
  a hypothesis, because every extra H3 picture has a measured identity cost.
- **Q1 (spend):** ~40 min of lem for the arms. **Q2 (scope):** build now,
  not defer alongside #1323. **Q3 (taste):** the marker table stands as
  PRELIMINARY; Don's viewing of R1, R2 and N1 lifts it later, not blocking.
- **Cut:** a planner that adds places by itself (dw has no planner; the
  skills are the planner, so it is a line of skill text); LTX-2.5 (already
  puts the location on its reference sheet); Kandinsky-6 (first frame only);
  a reference-count curve beyond the templates' two people plus a card.

## What the probe found (PRELIMINARY)

Workspace `probe-1577`, 1344x768. The H3 arms use seed 1398 (B0's real
seed, so R1 matches its control); the Klein stills use seed 42. The card is
`asset:places/1577-study.jpg` (common library), Klein job `5d47505f9762`.

| Arm | Job | Set markers (lamp, shelf, window; coat vs E only) | Face sim. | Reading |
|---|---|---|---|---|
| R0 (#1398 B0) | `fdba91e57259` | shelf only | 0.842 (re-check) | the control: words alone lose the set |
| R1 portrait + card, reverse | `a80f0b1ec000` | lamp, shelf, window; coat lost | 0.673 | a real reverse, not pulled to K's framing; face cost about the window's |
| R2 portrait + card, close-up | `1b58fcff664b` | lamp, shelf, window; coat lost | 0.804 | not pulled wide |
| R4 two portraits + card, 175 frames | `c2af95984127` | all three; coat lost | 0.378, unusable (two faces) | fit 24 GB, no OOM; staged side by side; a portrait prop leaked early |
| N1 card only, no people | `16f0a5e071f9` | lamp, shelf, window | - | holds the set; slight push-in |
| N0 same prompt, t2va | `6a555749ffde` | shelf only | - | words alone lose it |
| K-night (a look) | `e5ba385f1e51` | same layout, night window | - | works as a look |
| K-rev (image-edit reverse) | `15077fa2496a` | - | - | job succeeded, image failed: Klein returned K's own view |

R3 (reverse with K-rev) was not run: R1 was not pulled to K's framing, and
K-rev failed anyway. The card holds what it shows and nothing else (the
coat, absent from the empty card, stayed wrong everywhere).

## What was built

- **#1581 (stage A):** `bench/workflows/place-card-{shot,two,insert}.json`
  (also saved in `probe-1577`), `bench/records/place-card.md`, and an
  `EXEMPT` entry for the record in `tests/test_plugin_intents.py`.
- **#1582 (stage B):** `minimax-h3/references/cuts.md` gains *A place card
  holds the set*: make it on Klein `templates/flux3/text-to-image`, keep it
  with `keep_output(..., asset_name="places/<set>", shared=True)`, pass it as
  a `MiniMaxH3ImageReference` after the people so `<Picture N>` numbering
  doesn't move, and name it "the room of `<Picture K>`". People who belong
  to the set (an audience, a crowd, a jury) stay in it. Try the card first at
  a cut, then #1398's 22-frame window. Close-ups may carry it (check the
  face); a no-people shot may take it alone in ref2va; a new look is a new
  card; Klein's edit does not make a reverse card. `script-to-video` step 3
  and `series-episodes` step 0 cast a place card beside the people, replacing
  the unprobed "keep a wide master too"; `series-episodes`' shared assets
  name place cards; `storyboard` says the board's location image is the card;
  `intents.md` row 21 points to it. Pins in `tests/test_plugin_skills.py`
  (`TestPlaceCard`) and `tests/test_plugin_intents.py`; plugin version bump.
- #1323 (storyboard sheet as a Ref2VA picture) got a comment on what the
  still route found.

## Deferred

- **Don's viewing** of R1, R2 and N1, which lifts PRELIMINARY (Q3).
- **Reverse cards.** Klein `image-edit` did not produce one; a reverse
  card needs another route (a second text-to-image of the reverse view is
  untested). *Comes back on* a reverse shot a forward card pulls.
- **Two-person staging with a card** (R4 stood the pair side by side) and
  the portrait prop leak: one run, not separated from the card's effect.
- **LTX-2.5, H3-Image and Kandinsky-6** place-card arms: *come back on* a
  field report of a lost set on that family.

## Cost and bounces

- **#1581:** about 61 lem GPU-minutes against the ~40 planned (R4 alone
  took 23 min). One bounce: `status:needs-info`, because the record lives in
  the repo and the tester needed its text quoted on the issue. No failure.
- **#1582:** two bounces. (1) C-F1278: the card definition dropped the
  plan's audience clause ("people who belong in the set stay in it"). (2)
  C-F1281 (2): `validate_workflow` refused R4's shape (175 frames x 3
  references) on the 3090, because #900's gate held projections to the
  card's reported 23.6 GB while #479's calibration put the ran point at
  23.83. Filed #1586, which recalibrated every Ref2VA template's `base_gb`
  to 15.7 (175x3 projects 23.53); the stage then only corrected the quoted
  projection in `cuts.md`.
- Session dollar costs were not reported on the stages, so none are given.
