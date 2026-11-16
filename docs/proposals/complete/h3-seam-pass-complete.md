# The windowed refine seam pass, out of the flag (#1230)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out session,
2026-11-15). Found by the long-form bench (#1199); record
`bench/records/structural.md` *Step 4*. Plan v1 (2026-11-07) was parked under the
post-#1199 no-new-scope default; Don unparked it on 2026-11-15, plan v2 refreshed
it against `develop` dc38380f, and Don approved it at the defaults to Q1-Q3 the
same day. Built as three stages: #1376 (A), #1377 (B), #1378 (C).

## The problem

The one seam pass that measurably wins seam jump on every chain shot type benched
(S1 static, S2 arc, S3 dialogue; look too on S2; nothing worse past the floor; no
shimmer or lost motion at 120 s by Don's eye) was a temporal windowed refine over a
finished take, reachable only through the server env var
`DW_EXPERIMENTAL_REFINE_WINDOWS`. That var also gated nine unproven spikes, the
argument carried an `experimental_` name, nothing checked it at validate time, the
refine decoded the take once more than it needed, and no template or playbook
entry offered it.

**It is a seam pass, not a burn fix.** The record says the refine smooths seams
and does not undo the burn (`structural.md`, texture drift inside the floor). Every
doc and template line written here says so.

## Verdict and what was decided

**Build smaller:** no separate playbook stage (the line went in with the template)
and no burn work.
- Q1 [scope]: `experimental_refine_windows` renamed `refine_windows`, beside
  `refine_tiles`. The old name is refused, naming the new one.
- Q2 [default]: every key optional, defaults `{frames: 42, overlap: 12, blend:
  pyramid, positions: rebased}` (the 120 s bench setting); the template sets
  `refine_strength` 0.3 as a variable.
- Q3 [none]: the windowed denoiser, the long-refine layout step and the streamed
  decode go on every H3 load and run the stock path when the argument is absent.

The flag keeps gating SelfLift, the capstone, latent carry, keystone latents and
slot `latent`, `anchor_every`, the sigma gate, keystone restore and TST, under the
same env var name (`h3_tst.py` reads it directly). Only the windows left it.

## What was built

- **Stage A (#1376): `refine_windows`, defaulted, validated, out of the flag.**
  The rename; the defaults; the unconditional swap in `components.py` (keystone
  restore and SelfLift stay inside the flag's `if`); `windows_refusal` without its
  flag branch; a validate-time `windows_errors` check (`dw/guides.py`, modelled on
  `selflift_errors`) raising the run-time refusals for free: no `refine_strength`,
  a `CONFLICTS` input, `refine_tiles`, a bad overlap or size, an unknown key, a
  rebased stride off the 5-latent grid, a `source_video` over 15 s (naming
  `encode_h3_latents`), and the old name. The bench workflow key, stale comments,
  the `WORKFLOW_GUIDE.md` section (no longer experimental) and the
  `ARCHITECTURE.md` H3 row followed. Tests pin each gated spike as still refused
  with the flag off.
- **Stage B (#1377): the refine skips its own decode when `videos` isn't asked
  for.** `h3_tiles.skips_decode` / `run_without_decode`, a branch in
  `_call_pipeline` after the `refine_tiles` one: the decode blocks are trimmed to
  the outputs the step asks for, so `decode.audio` still runs and `decode.video`
  doesn't. On the 26 s fixture the refine's `decode_s` went from 98.5 s to 2.0 s
  and the run from 1560 s (with `videos`) to 1310 s (without).
- **Stage C (#1378): `templates/minimax/h3-seam-pass` and the playbook line.**
  encode → windowed refine (`refine_strength` 0.3, `refine_windows: {}`,
  `hold_audio: variable:source`) → decode (`num_frames` from encode's
  `source_frames`) → `pair_audio` (the source's soundtrack, `fit: video`). A
  required `source`, a flat cost row on the `repair-segment` pattern, a
  validate-time check that `source` is a video (`encode_h3_latents_errors`), and
  `COMPACT_BUDGET` raised. The long-form skill names it in `SKILL.md` *Per seam*
  (paid for by tightening, within the 12 288-byte limit), `references/levers.md`
  *The seam pass* moves it out of "do not offer", and
  `references/quality-and-speed.md` carries its cost (about 36 GPU-min per minute
  of take, about 16 GiB peak). On the verify run over the 26 s chain every seam
  came out lower than the source's on both `analyze_seams` measures (jump_ratio
  0.59-0.90 against 0.87-1.18).

## Bounces per stage

- **A: 1.** M-F355 found the gated spikes accepted at validate. Not a build miss:
  lem's `.env` still set `DW_EXPERIMENTAL_REFINE_WINDOWS=1` from the narrated-take
  bench. Don ruled the flag off lem; with it unset all 14 cases passed.
- **B: 1.** The first build skipped the decode only when the step asked for no
  decoded media at all, so an output keeping `audio` still decoded video. The
  re-ship made the trigger "an output that leaves `videos` out", as the plan said.
- **C: 0 tester bounces; 1 architecture-review bounce** (a second owner of a
  reference-prefix rule in `dw/validation.py`; the required-null `hold_audio` rule
  moved to one owner in `dw/hold_audio.py`). All five cases passed first verify.

Per-stage cost: the stage comments name no `usage:` figures, so it is left out.

## Deferred, and why

- **The burn.** The pass doesn't move it; not this feature's.
- **The float32 streamed decode's cost**, about half the pass (27 of 71.5 min at
  120 s). Measuring a bf16 or tiled decode is its own idea.
- **The ordering line with #1325 seam restore.** Whichever lands second writes
  it: restore inside the chain first, this pass when a finished take still shows
  jumps. #1325 had not landed when this closed, so it falls to #1325.
- **#1188 hybrid windows** keeps building on `h3_windows`; not folded in.
- Noticed on verify, not filed: validate warns "sets no 'seed'" though `seed` is
  `variable:seed`, and the output's `shots` metadata is null, so `analyze_seams`
  needed the boundaries passed in.
