# Temporal flicker probe; `temporal_denoise` task not built (#1747)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out session,
2026-11-26). Don filed the idea on 2026-11-26: a utility task that removes
frame-to-frame shimmer from a finished clip (H3 grain crawl, per-frame decode
exposure wobble, upscaler shimmer) with flow-compensated temporal filtering in
YCbCr, after `H3TemporalStabilizer` in Alex240z/ComfyUI-H3-ReTime (MIT, read at
`ce5fe07`; nothing from it was run). Plan v2 is the plan comment on #1747. This
file records what was measured, why the task was not built, and what would bring
it back.

## Verdict: build smaller, probe first; the probe said no

The evidence for the problem was thin: no `field-report` names shimmer, grain
crawl or flicker in a finished clip, and the nearest record (#1196, H3 flicker
via TST attention, benched under #1199) closed with no measured effect. A task
name is costly to take back once a saved workflow names it, so Don approved the
delegate's order (2026-11-26, "approve with delegate recommendations"): a CPU
bench probe first, and the task only if the numbers showed a gain **and** Don
saw less shimmer with no smear.

Neither gate passed, and Don ruled "record it, no build" (#1749, 2026-11-26).
The feature's whole deliverable is the bench script and its record.

## Plan history

- **v1 (design, 2026-11-26).** Build smaller: one mode (the flow-compensated
  temporal IIR), `flow_bilateral`, `hqdn3d` and `spatial_strength` cut, no
  template wiring. Stage A the task, stage B the measurement.
- **v2 (approved 2026-11-26).** The delegate passed it to Don (scope, taste) with
  "probe before building". v2 swapped the order: stage A the probe, stage B the
  task, gated on stage A.
  - Q1 [scope]: one mode, no `spatial_strength`, no template wiring. Ruled yes.
  - Q2 [taste]: Don looks at a before/after pair; his look gates the task.
  - Q3 [none]: `temporal_denoise`, media argument `clip` (a `video` argument is
    auto-loaded as bare frames and drops audio, `dw/tasks/stabilize.py`), own
    module. Never exercised.
  - Q4 [none]: reimplemented, credited under "Ideas reimplemented here".
- **Corrections to the idea found by the sweep:** the sketch's static-region
  frame-difference metric doesn't exist on MCP (`assess.py`'s `_frame_deltas`
  uses 64x36 thumbnails, too coarse for grain), so the measures live in the
  bench. `opencv-python-headless` is already core and has DIS flow and `remap`,
  so no dependency was needed.

## What was built

### Stage A (#1749): the probe (bench only)

Feature commit `494f44e1`, merged to `develop` as `8122365a`; record fix
`068439e9`, merged as `25a60f45`.

- **`bench/temporal_flicker.py`:** CPU only. Float YCbCr; DIS flow
  (`PRESET_FAST`) at half scale, neighbour warped with `cv2.remap`;
  forward-backward consistency mask at `occlusion_tolerance`; forward and
  backward IIR passes averaged; separate luma and chroma strengths. Two
  measures: flicker (mean abs dY on a static mask, input flow under 0.25 px) and
  moving gradient energy (Sobel, input flow over 1 px), plus a diagnostic
  outside the gate (strong moving edges after a σ 1.5 blur). Gate fixed at a
  flicker ratio ≤ 0.9 **and** a gradient ratio ≥ 0.95.
- **`tests/test_bench_temporal_flicker.py`:** synthetic static grain (flicker
  ratio 0.45), a translating square (gradient kept at ≥ 0.97, no ghost trail),
  identity at zero strength, a one-frame clip, and the gain needing both
  measures.
- **`docs/ATTRIBUTIONS.md`:** an "Ideas reimplemented here" line for
  `H3TemporalStabilizer`. No code was copied.
- **Record:** `bench/records/temporal-flicker.md` and `.json`.
- **Pair:** `asset:temporal-flicker-1749/{before,after,pair}.mp4` in the `bench`
  workspace on mini-ai.

**Result on the only H3 video on mini-ai** (the fox tracking shot, 124 frames,
960x544, 24 fps), at luma 0.5, chroma 0.5, tolerance 1.5 px:

| measure | before | after | after/before |
|---|---|---|---|
| flicker, static mask | 2.7478 | 2.1713 | 0.790 |
| moving gradient energy | 8718.33 | 6458.62 | **0.741** |
| strong moving edges (diagnostic) | 23970.04 | 20673.83 | 0.863 |

The filter took 2.5 s wall, 3.8 s CPU, so CPU time is not a risk. A rerun
matched every printed digit. The tester's independent MCP check agreed in
direction: frame-to-frame difference 23.06 → 21.32 on a crop, and the fox's
fine texture band (`texture_bands`) at 0.61-0.78×, softer with no doubled edge.

**Don's look (Q2):** at best marginally better (he leaned toward the after
clip), close and hard to judge. Not a clear visual win.

**Caveat on the clip.** It was not picked for grain crawl; it was the only
choice. A tracking shot leaves 0.08% of pixels static and moves 98.9% by more
than 1 px, so the flicker number rests on few pixels and nearly every pixel goes
through the warp: the filter's hardest case.

### Stage B (#1750): the `temporal_denoise` task — not built

Closed `not planned` on 2026-11-26 under Don's ruling. Nothing under `dw/`,
`dw_mcp/`, the plugin or `docs/TASKS.md` changed. Its acceptance cases
(C-F1458 to C-F1465) were written `pending: #1750` and are left for the curator
to retire.

## Cost and bounces

| Stage | Bounces | Cost |
|---|---|---|
| #1749 | 1 (C-F1455: the hand-off lacked the rerun, absolute values, frame size and region definitions; text-only fix) | Session cost not recorded on the stage; the plan estimated $2-3. CPU only, no GPU time |
| #1750 | — | Not built |

## Deferred and open

- **The task comes back as its own feature** if a probe on a better clip passes
  both gates. The obvious next probes, from Don's ruling:
  - a **locked-off H3 shot** with grain on a still background (none is on
    mini-ai today), which tests the filter's best case;
  - a **lower strength** (luma/chroma under 0.5), or a tighter
    `occlusion_tolerance`, to see whether flicker can fall without the moving
    edges softening.
  - Either needs only a rerun of `bench/temporal_flicker.py`; the stage B design
    in plan v2 (arguments, domains, memory bound, tests, docs) still stands.
- **A `field-report` naming shimmer in a finished clip** would also reopen it:
  the value question was never answered by a user.
- **The cut modes** (`flow_bilateral`, `hqdn3d`, `spatial_strength`) come back
  only with a clip where the flow mode leaves visible grain that a spatial pass
  fixes.
- Template or skill-chain wiring needs Don's verdict that it should run by
  default.
