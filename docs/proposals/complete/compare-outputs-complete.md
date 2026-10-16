# `compare_outputs`: aligned grids and diff metrics across runs (#871)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out session,
2026-10-14). The idea came from the H3 quality matrix on lem (workspace
`h3-quality`). Every judgement there needed local tooling: ffmpeg contact
sheets, `tools/facegrid2.py` face crops and `ffmpeg psnr` against a baseline.
A text-only MCP client can do none of that. Plan v1 was designed on #871.
Don's review made v2, which says metrics always compare the same pixels and
moves the plugin pointer out of the full `SKILL.md`. Don approved v2. It was
built as one stage, #889. The plan comment on #871 stays the full design;
this is the record of what shipped.

## The problem

Judging "did this knob change the picture, and how much" across runs took
N `get_output_frames` calls, aligned by eye, and nothing in dw computed PSNR
or SSIM. The numeric half could not be done from a client without ffmpeg.

## Verdict: build smaller

Cut from the proposal, with what would bring each back:
- **`crops: "faces"`** (stage B, deferred and not filed). YuNet is already in
  `dw/tasks/face_track.py`, so it needs no new dependency. *Trigger:* a field
  report showing per-output coordinates are still the bottleneck. *To settle
  before filing:* where the YuNet weights are downloaded. `face_track`
  downloads them in the worker; stage B would otherwise fetch them in the
  server process.
- **`same_staging`.** dw reports findings, not verdicts. The matrix's bands
  are in the guide as a reading aid instead: ~44 dB is rerun noise, 23–27 dB
  is changed numerics, ~19 dB is a re-staged shot. *Trigger:* two or more
  comparison sessions read the bands the same way and want it as a field.
- **`layout`.** A sheet and a grid are the same output with more or fewer
  columns.
- **Saving the grid.** dw has no run-less output. The grid comes back
  inline, like `get_output_frames`. *Trigger:* someone asks to keep one, as a
  `keep_as` that writes to the asset library.

Non-goals stayed out: `latest`/`vN` expansion in names, audio comparison,
and resizing outputs to make different sizes comparable.

## Plan history

- **v1:** per-output crops computed metrics whenever the boxes were the same
  size, and a pointer line went in `minimax-h3/SKILL.md`.
- **v2 (approved):** Don's review settled that **metrics always compare the
  same pixels**. One shared crop is both what is shown and what is measured.
  Per-output crops are display only, and metrics then cover the full frame.
  The plugin pointer moved to `minimax-h3/references/quality-and-speed.md`,
  because `SKILL.md` was at 12,282 of 12,288 bytes. The route became a sync
  `def` so decode and SSIM run in FastAPI's threadpool. Q1: a new tool. Q2:
  per-output crops, with the rule above. Q3: raise `SURFACE_BUDGET` with a
  dated ledger entry.

## What was built

### Stage A (#889): `compare_outputs` and `POST /api/gallery/compare`

Commits f15ca771 (the route and `dw/media_compare.py`), 9c3c23a2 (the MCP
tool and tests), 74700668 (docs and the OpenAPI snapshot), 93d06dbe (map
rows and the single owners for the `at` and crop parses) and 99f04d67
(negative moments refused, `max` in the reply).
- **Engine:** `dw/media_compare.py` owns `compare`, `psnr`, `ssim` (601
  luma, 11×11 Gaussian window, σ 1.5, numpy + `cv2.GaussianBlur`, no
  `skimage`), `mean_abs_diff`, `check_request` and `crop_box`. The grid
  reuses `dw/media_frames.py`'s compositor. It runs in the server process,
  needs no GPU and queues nothing.
- **Reply:** one grid image, with rows for outputs in the given order
  (labelled, `(baseline)` marked) and columns for moments (each cell stamped
  `frame N`). It also has a text summary per non-baseline output, per column
  and as mean/min/max, with `identical_columns` counted apart so ∞ doesn't
  skew the mean. It names the `region` measured.
- **Nulls with a reason, never a resize:** `size differs from baseline: WxH
  vs WxH`. A moment past an output's end gives a blank cell and `past end (N
  frames)`. `fps_differs` flags clips compared by frame number at different
  rates.
- **Shared owners:** `media_frames.moment_index` owns the `at` grammar and
  `media_frames.parse_crop` owns the `[x, y, w, h]` parse. They are used by
  both `get_output_frames` and `compare_outputs`, and each tool keeps its own
  bounds policy. `docs/ARCHITECTURE.md` *Media and DSP* has an **Output
  comparison** row and a **Frame moments and crops** row.
- **Surface:** the MCP tool count went from 64 to 65. `SURFACE_BUDGET` is
  14,990, with a dated ledger entry for the measured 14,989.25.
- **Docs:** `docs/MCP.md` (a tool row), `docs/SERVER.md` (the route),
  `docs/WORKFLOW_GUIDE.md` *Assessing a run's output* → "Comparing outputs"
  (the metrics, the region rule and the dB bands as a reading aid), and
  `quality-and-speed.md` (one line).
- **Tests:** engine tests in `tests/test_media_compare.py`, route tests on
  real PyAV-encoded clips and PNGs in `tests/test_compare_route.py`, and MCP
  pins and summary tests in `tests/test_mcp_server.py`.

**Deviations from the plan:**
1. **A crop must lie inside every output's frame, or it is a 400.**
   `get_output_frames` clamps instead. Clamping per output would measure a
   different region in each row.
2. **The summary gives mean, min and max**, more than the plan's "mean and
   min". `max` was dropped from the MCP text at first and restored in
   99f04d67.
3. **A grid that can't shrink without going under the 64 px per-cell floor
   comes back unshrunk**, with `downscaled_to` null and a note that it is
   over budget. It does not claim a shrink that didn't happen.
4. **Negative moments are refused** (`'frame:-1' is negative - moments count
   from the start of the clip`). `get_output_frames` still counts them back
   from the end. That makes two bounds-policy differences from frames'
   grammar: this one and past-end. It came from acceptance case C-F463,
   through `moment_index(from_end=False)`.

## Verification and bounces

| Stage | Bounces | What bounced |
|---|---|---|
| #889 | 2 | Architecture review: there was no map row for `dw/media_compare.py`, and it had second owners of the `at` grammar and the crop parse (fixed 93d06dbe). Verify on mini-ai: C-F463 call 9, where `frame:-1` and `-1.0` wrapped to the end of the clip instead of being refused (fixed 99f04d67). |

- #889 was verified on mps (mini-ai, develop @ 5ff05cf5) against C-F454 to
  C-F466 and SE-F049. All 14 cases pass. Nothing in the stage is
  CUDA-specific.
- Cost is not recorded: the stage comments name no `usage:` figures.

## Found along the way

- **The fixture literal in C-F454/C-F459:** `qr_code` squares its output, so
  S is 640×640, not 640×384. The tester filed a suite amendment on harnest.
- **Cosmetic, not changed:** in a one-column grid (stills), row labels are
  clipped at the cell width.
- **Not a server issue:** C-F466's tool-count check is vacuous over MCP,
  because no tool reply states the count.

## Follow-up

#861, #875, #862 and #865 each end in "is variant X the same as baseline?".
They can now answer that over MCP with `compare_outputs` instead of local
ffmpeg.
