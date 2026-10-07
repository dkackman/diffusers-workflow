# Finishing tasks: tonal `grade`, `sharpen`, `film_grain`, `apply_lut` (#603)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out session,
2026-10-07). Designed on #603 from a comparative review of
vrgamegirl19/comfyui-vrgamedevgirl (source-available, incompatible with
Apache-2.0: **concept only, nothing copied**). Plan v1 was approved on
2026-10-05 with Q1 = build smaller and defaults for Q2-Q4; plan v2 recorded
those answers. Built as stages #633 (A), #634 (B), #635 (C), #636 (D) and
#637 (E).

## The problem

`grade` (#349) was dw's only finishing control: exposure, contrast,
saturation, temperature and tint. There was no tonal-range control, no LUT,
no grain and no sharpening, so a generated clip couldn't get the standard
finishing pass inside a workflow, and a series had no way to share one look.

## Verdict and what was decided

**Build, smaller.** The demand signal was Don's approval; no field report
asked for these pieces.
- Q1: no `make_lut` task, no `.cube` writer, no `CubeLut` output, no
  `text/x-cube` mapping, no `lut` listing kind, no UI. `apply_lut` takes a
  `palette` directly instead.
- Q2: the `.cube` parser is strict (domain 0..1, values in [0,1]).
- Q3: film grain differs on every frame; no `static` flag.
- Q4: stage E (the series-episodes look step) is included.

Corrections the plan found against the issue: `match_levels` is an audio
argument on the joins, not a task, so the look is a new optional per-shot
step; `asset:`/`output:` references already arrive validated, so only a
literal `lut` path needed a new gate; `upload_asset` refused `.cube`; and
`grade`'s "applied" log used a hand-kept defaults dict.

## What was built

Every piece runs on CPU, per frame, on images and videos (fps and audio
kept), with alpha passed through and every parameter defaulting to
identity. `docs/TASKS.md` has a section for each task.

### Stage A (#633): tonal controls in `grade`

- Seven parameters in `dw/tasks/grade.py`: `highlights`, `shadows`,
  `whites`, `blacks`, `clarity`, `vignette` (-1..1) and `fade` (0..1).
  Highlights and shadows are luma-masked so the other half is bit-exact;
  clarity is luma minus a Gaussian blur (σ = 2% of the shorter side),
  masked to midtones.
- Order: exposure, contrast, whites/blacks, highlights/shadows, clarity,
  temperature/tint, saturation, fade, vignette.
- The `UNIT` domain kind (closed [0,1]) in `dw/task_domains.py`.
- The "applied" log's defaults are derived from the signature.
- The blur is `scipy.ndimage.gaussian_filter` (after the architecture
  bounce; the first build hand-rolled it).

### Stage B (#634): `sharpen` and `film_grain`

- `dw/tasks/finish.py`. `sharpen(media, amount=1.0, radius=2.0,
  threshold=0)` is Pillow's `ImageFilter.UnsharpMask` (after the
  architecture bounce). Contract: `amount` is applied in whole percent and
  `threshold` in whole levels; `radius` is Pillow's blur radius.
- `film_grain(media, amount=0.1, size=1.0, chroma=0.0, seed=None)`: noise at
  1/size resolution upsampled bilinearly, luma-to-chroma mix, midtone
  weighting. `seed` goes through `Task.seed_for`; one `default_rng` per call
  is consumed frame by frame, so frames differ and a run reproduces.
- The `SEED` domain kind (a whole number ≥ 0), so `validate_workflow`
  refuses `"abc"`, `1.5` or `-1` (after the tester's C-F262 bounce), with
  the resolved seed re-checked at run time.

### Stage C (#635): `apply_lut` from a `.cube`, and `.cube` upload

- `dw/tasks/lut.py`: a strict parser (16 MiB, UTF-8, `TITLE`/`LUT_3D_SIZE`/
  `DOMAIN_MIN`/`DOMAIN_MAX`/comments/data only, size 2..65, exactly size³
  finite rows in [0,1], red-fastest). Refusals name the file's base name and
  line, never an absolute path.
- The lookup is Pillow's `ImageFilter.Color3DLUT`, built once per step
  (after the architecture bounce; the first build hand-rolled trilinear
  interpolation). The `strength` blend is numpy.
- `lut` and the finishing tasks' `media` are in `TASK_MEDIA_ARGUMENTS`
  (`dw/locations.py`), so a literal path outside the roots, a `..` segment
  or a non-http(s) URL refuses at validate. `LOCAL_ONLY_TASK_ARGUMENTS`
  refuses an http(s) `lut`: it is never fetched (after the tester's SE-F044
  bounce; the first build refused only at run time).
- `.cube` is in both upload allowlists, server and MCP, kept equal by
  `tests/test_mcp_twins.py`.

### Stage D (#636): `apply_lut` takes a `palette`

- `palette`: 2 to 16 `#rrggbb` colours, dark to light. Exactly one of `lut`
  or `palette`. The rules are `palette_problem` and `lut_source_problem` in
  `dw/task_domains.py`, refused at validate and re-checked at run time; the
  architecture map's "Task argument domains" row names them (the
  architecture bounce).
- `palette_table` builds a 33³ table in memory: input luminance picks a
  position along the palette gradient, and the output keeps the input's
  luminance with the gradient's hue and chroma. It goes through the same
  `Color3DLUT` and `strength` blend as a `.cube`. Nothing is written to disk.

### Stage E (#637): series-episodes look step

- `plugins/dw/skills/series-episodes/SKILL.md` gains "Optional look, before
  the recut" (not a beat; the five beats stay five): one shared `palette` →
  `apply_lut` per shot, or an uploaded `.cube` as `lut`, optionally followed
  by `film_grain`. The full recipe and a validating fragment are in
  `references/look.md`.

## Deferred, and why

- **`make_lut` and a `.cube` file output** (Q1). Comes back only if someone
  needs to export a LUT outside dw.
- **The plugin version bump stage E's plan asked for.** The plugin version is
  the engine's and only the release script bumps both
  (`test_the_plugin_version_is_the_engine_version`), so it waits for the next
  release, which is Don's.
- **The 16 MiB cap and non-UTF-8 refusals over MCP.** `upload_asset(content=)`
  caps at 4 MB, so these are covered by pytest only (`tests/test_lut.py`).
- **Alpha pass-through over MCP.** No fixture has alpha; unit tests cover it.
- Non-goals unchanged: other LUT formats, 1D LUTs, GPU paths, temporal grain,
  shipped presets.

## Cost and bounces

The plan estimated about $15.5 across the five stages. The stage comments
record no `usage:` figures, so actual cost is not recorded here.

| stage | architecture bounces | tester bounces | cause |
|---|---|---|---|
| A #633 | 1 | 0 | hand-rolled blur → scipy |
| B #634 | 1 | 1 | hand-rolled unsharp mask → Pillow; non-integer `seed` passed validate (C-F262) |
| C #635 | 1 | 1 | hand-rolled trilinear lookup → Pillow; literal `lut` paths refused only at run time (SE-F044) |
| D #636 | 1 | 0 | architecture map row not updated |
| E #637 | 0 | 0 | — |

Three of the four architecture bounces were build-vs-buy: the plan said
"numpy/PIL" without naming the Pillow and scipy primitives that already
existed. A plan touching image processing should name them.

## Acceptance cases

C-F251–C-F269 (`regression-suite-complete.md`) and SE-F044
(`regression-suite-security.md`) in harnest. One case-text note from stage
C's verify: a content upload returns `asset:uploads/<name>`, so cases must
use the `uploads/` spelling.
