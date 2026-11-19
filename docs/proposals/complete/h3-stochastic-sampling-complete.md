# H3 stochastic sampling probe: turbo video and H3-Image (#1438)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out session,
2026-11-19). Don filed the idea on 2026-11-18 from
PedroMarinhoDev/ComfyUI-H3-Image-EasyFix (Apache-2.0). Its Re-Noise Sampler swaps
the Euler step for an `lcm`-style update, `x = (1 - s1) * x0 + s1 * noise`, to
remove grainy fine texture on MiniMax-H3-Image. `MiniMaxH3Scheduler` already has
that update on every step as `stochastic_sampling` (EasyFix's `lcm` plan). The
question was whether it cleans H3 turbo video (lightx2v 4-step LoRA), slows
texture creep on a chain, or removes speckle from H3-Image. Plan v1 is the plan
comment on #1438. Don approved it at its defaults on 2026-11-18, and it was built
as one bench-only stage, #1459. The full measurements are in
`bench/records/h3-stochastic-sampling.md`.

## Verdict: build (bench only), result: drop every arm

- The flag takes, and turbo video renders clean with it, not broken.
- The re-noise is **unseeded**. Two flagged runs at seed 42 differ as much as
  flag-off differs from flag-on (PSNR 12.44 against 12.74 dB). Two flag-off runs
  are bit-identical, even across cards.
- On a 3-segment chain the flag does not reduce creep: x2.15 against x1.82 on the
  whole frame, x1.83 against x1.10 on the centre box. Mean seam |L| is unchanged
  (1.73 against 1.74).
- H3-Image shows no speckle at 2048² or at 16 MP (4096², which fit the 3090 at a
  62 GB host peak), so the image arms had nothing to fix and did not run.
- No template variable and no documented option. Any later adoption needs a
  seeded re-noise path first.

## Plan history

- **v1 (approved 2026-11-18, at its defaults).**
  - Q1 [none]: rebuild the scheduler with `configuration` + `from_config_args`,
    with the control going through the same rebuild with the flag false.
  - Q2 [none]: treat flagged arms as unseeded, and run each flagged setting twice.
  - Q3 [none]: run the chain pair only if V1 is clean or better, and the 16 MP still
    only if it fits.
  - Q4 [spend]: up to about 55 min of lem GPU, inside a 60 min cap.
- **Corrections to the issue found by the sweep:**
  - A bare `"stochastic_sampling": true` in a `scheduler` block is silently
    ignored. `load_and_configure_scheduler` reads only `configuration`,
    `from_config_args` and `shift`, and the schema doesn't refuse extra keys. As
    filed, the flagged arm would have run plain.
  - Neither H3 denoise loop passes a `generator` to `scheduler.step`, so the
    re-noise draws from the global RNG, which dw never seeds. The build confirmed
    this.

## What was built

### Stage A (#1459): stochastic-sampling bench (bench only)

Merge `4947fa5d` on `develop`: 91165b37 (workflows and arms) and cbac5323 (the
record). No engine, template, MCP, REST, schema, plugin or user-doc change, and no
deploy.

- `bench/workflows/probe-1438-video-with-audio.json`,
  `probe-1438-chained-segments.json` (3 segments, seed 42) and
  `probe-1438-h3-image.json`. Each rebuilds the scheduler as `MiniMaxH3Scheduler`
  with `from_config_args` `{stochastic_sampling, euler_update_form}` from declared
  variables, plus `shift` on video. `audio_scheduler` stays shift-only. Variables
  do reach `from_config_args`, so no per-arm fixed copies were needed.
- `bench/arms/probe-1438.json`: workspace `bench`, one note per arm.
- **Checkpoint forms:** MiniMax-H3 declares no `euler_update_form`, so it runs the
  class default `blend` and there was no V0b arm. MiniMax-H3-Image declares
  `velocity`.
- **Proof that the flag took:** the fail-fast job c811b290cab1 (flag with
  `velocity`) was refused in `__init__` with the blend-only error. V1 reports
  `load_cached: false` after a flag-only change. `get_job_workflow` shows the
  rebuilt block. The `Loading scheduler` line is only in lem's server journal,
  not in `get_job_events` (see below).
- **GPU:** about 23.7 min of job wall across ten jobs, inside the 60 min cap.

**Deviations:** no V0b (the checkpoint is already `blend`); a flag-off repeat,
V0b-extra, was added as the seeding control; the repeat arms ran under a
server-saved `-b` workflow id to miss the step cache; I1/I2/I2b did not run
because their gate was unmet; runs went through MCP `run_workflow`, not
`bench/driver.py`.

Cases verified: M-F428 to M-F433, over MCP. M-F429 passed under the amendment
filed as dkackman/harnest#248, which accepts the fail-fast refusal,
`load_cached: false` and the realized workflow in place of the load line in job
events (advisor ruling, 2026-11-19).

## Cost and bounces

| Stage | Bounces | Cost |
|---|---|---|
| #1459 | 1 (the record wasn't quoted where the tester could read it, and the `Loading scheduler` line isn't in `get_job_events`) | Session cost not recorded on the stage; the plan estimated $3–5. GPU: about 23.7 min of job wall on lem, both cards |

## Deferred and open

- **A seeded re-noise path** (pass the step's generator to `scheduler.step`,
  upstream in diffusers or in a dw wrapper). Not needed now, since no arm won. It
  is the precondition for revisiting the flag with several draws per arm.
- **Stray scheduler keys pass validation silently.** Proposed in plan v1, kind
  `default`, not ruled and not filed.
- **The scheduler-load line is missing from job events:** #1487 (`idea`).
- **The per-step mixed plan (`SSRRRS`)** needs a per-step switch the scheduler
  doesn't have. The every-step flag showed no gain, so there is no follow-up.
- **The H3-Image arms** stay in the arms file for a later run on a scene that
  does speckle.
- EasyFix's Steering node was out of scope.
