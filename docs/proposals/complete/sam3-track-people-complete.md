# SAM 3 as a `track_people` backend (#1397)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out session,
2026-11-18). Don filed the idea after #1375 accepted a limit: `track_people`
detects only on each cut's first frame, so a person who enters later, or comes
in through a dissolve the cut detector doesn't split, gets no mask (C-F893's
fixture, `ep65-shot-kf-priya-hal`). SAM 3 (`facebook/sam3`) segments from a
text prompt in one model and tracks through a clip. The plan (v2) is the plan
comment on #1397, and it holds the full sweep. This file records what was built
and measured.

## Verdict: build smaller, in two stages

An opt-in `backend: "sam3"` arm on `track_people` (stage A), then a bench of it
against Grounding-DINO + SAM2 on both hosts (stage B). No default change, no
`segment` change, and no template exposes `backend`. Whether SAM 3 replaces the
default was left to a plan v3 on the bench's evidence.

**Bench result:** keep Grounding-DINO as the default and SAM 3 as an option.
No plan v3 is proposed, so nothing more is filed under #1397.

## Plan history

- **v1 → v2 (ruled 2026-11-16 21:16).** No scope change.
  - Q1 [scope], default: bench SAM 3 only. Periodic re-detection on today's
    stack (DINO every N frames, new boxes added as SAM2 objects) stays the
    unbuilt fallback.
  - Q2 [trust], accepted: the SAM License (facebookresearch/sam3 `LICENSE`,
    2025-11-19) on an opt-in mask path, recorded in `docs/ATTRIBUTIONS.md` with
    its use restrictions and the fact that dw does not redistribute the weights.
  - Q3 [none], default: late entrants are numbered after first-frame people, in
    order of appearance, and `people` gains `first_frame`.
  - Q4 [spend], approved subject to a download hold on both hosts.
  - Q5 [default], otherwise: keep `transformers>=5.19.0`, import the Sam3
    classes lazily, refuse `backend="sam3"` on an older transformers naming 5.21.
  - Text corrections: the demand line read the newest 60 of the last 200 jobs;
    there was no "stage C".
- **Rulings during the build:**
  - 2026-11-17 23:23: the `facebook/sam3` download hold lifted on mini-ai.
  - 2026-11-17 (on #1405, 03:28 UTC 11-18): **the transformers floor moves to
    `>=5.21` after all**, as a `pyproject.toml` change deployed by
    `scripts/deploy.sh`, not a hand install on lem. This reverses Q5's "no floor
    raise". Don also released the `facebook/sam3` download on lem, for that repo
    only. The ruling asks for a speech-task smoke on both hosts after the deploy.

## What was built

### Stage A (#1404): `track_people(backend="sam3")`

- `dw/tasks/person_track.py`: `backend` (`grounding-dino` default, `sam3`) and
  `sam3_model_name` (default `facebook/sam3`). The per-cut bodies are split into
  `_dino_cut` and `_sam3_cut`, and `track_people` stays under the 150-line
  function ratchet. The sam3 path opens a `Sam3VideoInferenceSession` per cut,
  takes the prompt as is, and keeps `_exclusive_masks`. People get
  `first_frame`. The no-person error names the backend.
- `backend` is checked at validate time through a `static_check`
  (`track_people_errors` → `task_domains.choice_errors`), so `"bogus"` and `""`
  are refused naming both values.
- `required_weights` on the sam3 path declares only `facebook/sam3`, so a
  validate quotes it alone and gated (6.4 GB). `dw/hub_cache.py` was fixed so a
  configs-only cache entry still counts as a download.
- Per-backend default threshold (`DEFAULT_THRESHOLDS`): DINO 0.3, SAM 3 0.5.
- Docs: `docs/TASKS.md` "The sam3 backend", the `docs/ARCHITECTURE.md` row,
  and the `facebook/sam3` row in `docs/ATTRIBUTIONS.md`.
- Tests: a fake SAM 3 session in `tests/test_person_track.py` (late entrant,
  numbering, `max_people`, exclusivity, the error), the transformers refusal,
  and `required_weights` and plan rows for both backends.
- Verified on mini-ai (mps): C-F961–C-F968 all pass. On C-F963 the sam3 run
  gave the grey-jacket man entering through the dissolve his own mask
  (`first_frame` 10, score 0.626), and the DINO run still missed him.

### Stage B (#1405): the bench

- `bench/workflows/track-people-backend-ab.json`, its arms file, the
  structural test `tests/test_bench_track_people_ab.py`, and the record section
  "track_people: Grounding-DINO against SAM 3" in
  `bench/records/performer-replacement.md`.
- `pyproject.toml` floor raised to `transformers>=5.21` (`de6fa1a3`, per Don's
  ruling). lem now runs transformers 5.23.0, installed by the deploy.
- 12 cells (ep65 dissolve, ep62 pair, ep49 × two arms × two hosts), each with
  a job id. Verified on both hosts: C-F969 and C-F970 pass.

## What the bench found

- **The hosts agree cell for cell.** Scores are within 0.03, first frames
  within one frame.
- **Late entrants: only SAM 3 sees them.** ep65's entrant got a mask on both
  hosts (score 0.626 mps, 0.644 cuda). DINO missed him on both.
- **Elsewhere the backends agree.** On ep62 and ep49 they find the same people
  with boxes within a few pixels.
- **SAM 3 also finds false entrants.** On ep49 it found two background objects
  (a cabinet edge and a photo on the fridge) at 0.50–0.55, just above its 0.5
  floor. A floor of 0.55 would drop them and keep ep65's real entrant, but three
  moments in two clips are too few to move a default.
- **Cost.** SAM 3 takes about 2.5x DINO's wall time on cuda and about 3x on
  mps. On cuda it reserves about 3 GB more at peak (8.3–8.7 GB against
  5.2–5.5 GB). Both fit a 24 GB card.

## Deferred, and why

- **Changing the default, exposing `backend` in `replace-performer-cuts`, and
  SAM 3 behind `segment`:** not proposed. SAM 3 wins only on mid-cut entrants,
  costs 2.5–3x, and adds near-floor false entrants that need checking on the
  contact sheet.
- **A raised sam3 threshold (0.55):** too little evidence. It is a candidate if a
  field report shows false entrants in use.
- **Periodic re-detection on DINO + SAM2:** stays the unbuilt fallback (Q1).
- **The lazy-import refusal:** it now fires only when transformers is below the
  install floor, since the floor is 5.21. The refusal and its test stay.

## Owed

- **A speech-task smoke on mini-ai after its next deploy.** Don's floor ruling
  asks for it on both hosts. lem's passed (`ed13adda2dc7` speech,
  `b07b81924ff3` Audio8). The stage B session could reach lem only.

## Bounces per stage

- **Stage A (#1404):** two. An architecture review bounced first, because
  `track_people` grew to 202 lines against the 150-line ratchet. Then a verify
  bounced on C-F961/C-F962: `backend` was not refused at validate, and a
  configs-only cache entry hid the sam3 download. A later verify pass found
  nothing wrong but was held by the download hold. The pass after the hold
  lifted verified the stage.
- **Stage B (#1405):** one. The cuda half was bounced because lem's six cells
  hadn't run: lem had transformers 5.20.2 and no `facebook/sam3`. The lead
  handed it to Don, who ruled the floor bump and the download. The hand-off
  gate also caught one ruff format before the first hand-off.

## Cost

The stage comments carry no `usage:` figures, so cost is not recorded here.
GPU time: 12 bench runs. lem's six took about 29 min (DINO 86–244 s, SAM 3
219–635 s), and mini-ai's three sam3 cells took about 31 min (320–917 s) plus
its DINO cells. That is over the plan's 30–45 min estimate, which was not
measured on these hosts. Downloads: `facebook/sam3` once per host
(6.9 GB on lem).
