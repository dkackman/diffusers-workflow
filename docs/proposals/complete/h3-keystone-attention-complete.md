# MiniMax-H3 keystone attention probe (#1391)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out session,
2026-11-19). Don filed the idea on 2026-11-16 from roycho87/geiru_context_suite
1.1.x's Keystone Attention (GPL-3.0, concepts only, no code read) and the notes
behind it, roycho87/NerdNotes tests 19 and 20 (CC BY-SA 4.0). Those credit the idea
to ckinpdx's Carry Attention (ComfyUI-MMH3Tools). On a chained H3 clip with a
keystone, the probe re-splits the slice of attention that the keystone keys and the
carried-guide keys share, moving it toward the keystone on the late blocks. Plan v4
is the plan comment on #1391, and it still holds the full sweep and the math. This
file records what was built, what was measured, and what stayed cut. The bench
record is `bench/records/narrated.md`, in the two *Keystone attention* sections.

## Verdict: build smaller, a one-call probe

- **An H3 fl2va call argument, not a chain key.** `experimental_keystone_attention`
  sits behind `DW_EXPERIMENTAL_REFINE_WINDOWS=1`. In that call, "keystone" means the
  `image` keyframe rows and "carried" means the `guides` rows.
- **The bench chain was built by hand.** Every segment's `image` is segment 1's
  frame 0, which is where `keystone_slot: first` puts it.
- **Cut:** the `keystone: segment_1` chain key, and the `latent`, `{frame}` and
  `last` slots.
- **Our drift isn't theirs.** The no-keystone narrated control `aed61997052a` held
  luma (-0.5) but rose in saturation (+8.8) and face sharpness (+105). So the open
  question was whether re-splitting the attention touches face sharpness.

## Plan history

- **v1 (approved 2026-11-16, at defaults).**
  - Q1 [scope]: a call argument, not a chain key.
  - Q2 [spend]: a gated two-arm bench.
  - Q3 [none]: diffusers' block indices are taken as ComfyUI's.
  - Q4 [none]: refused alongside TST, sparse attention, a capstone or gated guide
    rows.
- **v2 (decompose).** Folded in the rulings and Don's lem scheduling: queue behind
  #1385 and #1382, one H3 job at a time.
- **Don's 2026-11-16 amendment (Q6, from roycho87's comment on #1391,
  issuecomment-6922271175).** Stage B opens with a numbering check (a brightened
  keystone) and a face-sharpness group sweep (a sharpened keystone).
- **v3 (stage B build, approval withdrawn).** `seam-restore-control.json` runs sparse
  on every segment, and stage A refuses sparse. Q5 [loosen]: both arms run dense,
  and `aed61997052a` becomes a reference column that differs in two things.
- **v4 (decompose).** Recorded Q5 at its default and folded in Q6, which v2 and v3
  had missed.
- **Rulings during stage B (2026-11-18 and 19), each recorded on #1394:**
  - run on lem's current diffusers, `e84bda1c`, not the control's `43e30028`;
  - step 1 only, then back to Don, after lem went down during a dense run;
  - no step 2: run the gate on roycho87's blocks;
  - after viewing: gate failed, stop.

## What was built

### Stage A (#1393): the re-split processor (server)

Merges on `develop`: 8e5d7338 (b22d38e1, 218a2db1), 14b5a468 (378de5cf, the
architecture-review fix), and ef128323 (1c099e94, the M-F379 fix).

- **`dw/pipeline_processors/h3_keystone_attention.py`.** It takes the argument
  (`balance` 20 and `blocks` `31-33,36-44,47` by default; N is read off the block
  list) and runs the validate-time refusals.
- **Row publication.** The guide layout step (`h3_guide_steps.py`) publishes the
  keyframe and guide row counts.
- **Install and math.** It installs on a subset of blocks per call, and the re-split
  is exact. One LSE-returning main pass is made, plus one pass over the keystone keys
  only and one over the carried keys only. Then pK' = S·b·pK/(b·pK+pC), and
  O' = O + ΔpK·O_K + ΔpC·O_C.
- **The LSE op order follows torch's SDPA priority,** so `balance: 1` reproduces the
  run without the argument.
- **Tests.** Unit tests (`tests/test_h3_keystone_attention.py`) check it against a
  dense softmax reference, and test the parser and each refusal. The spikes are gated
  at validate time.
- **Docs.** The `docs/ARCHITECTURE.md` row "H3 experimental attention processors (TST,
  keystone attention)", one experimental line in `docs/WORKFLOW_GUIDE.md`, and the
  `docs/ATTRIBUTIONS.md` entries: NerdNotes tests 19-20, geiru's Keystone Attention,
  and ckinpdx's Carry Attention.

**Bounces (3):**
1. The architecture review bounced it: the new module had no map row.
2. The first verify couldn't run, because lem served without the flag. `.env` was
   then set and the server restarted. That is not counted against the build.
3. The tester failed M-F379: `balance: 1` differed from no argument by a MAD of 2.17.
   The cause was a bf16 kernel mismatch, not the math: the main pass tried cuDNN
   first, while H3's own processor dispatches SDPA, which runs flash first on
   torch 2.14. It was fixed by following torch's SDPA priority order.

It verified on 2026-11-18 against develop @ 940c51d0.

### Stage B (#1394): numbering check, then the early gate (bench only)

Merge on `develop`: 3af05299 (ef169c02, f9eda0f1, ce786481, f502ab25).

- **Bench files.** `bench/workflows/keystone-attention-control.json` and
  `keystone-attention.json`, with their `*-early.json` gate cuts; the probes
  `keystone-probe-seg01.json`, `keystone-probe.json` and `keystone-probe-noarg.json`;
  the measure `bench/keystone_luma.py`; and the arm file
  `bench/arms/keystone-attention.json`.
- **Tests.** `tests/test_bench_keystone_attention.py` checks that the control is
  `seam-restore-control` with exactly two changes, that the arm is the control plus
  the argument, and that each early job is cut after seg02.
- **Attributions.** A `docs/ATTRIBUTIONS.md` line credits roycho87's comment and
  NerdNotes test 19 for the block-finding method.
- **No bounces.** The stage was re-planned once before it was built (v3).

## What was measured

All runs used diffusers `0.44.0.dev0` @ `e84bda1c`, workspace `bench`, dense
attention, one H3 job at a time.

**Step 1, the numbering check: it matches.** Each run renders segment 2 alone, at
seed 2, with segment 1's 22-frame tail as the guide and `balance` 20. The figure is
the mean luma of the new frames.

| Job | Keystone | `blocks` | vs normal |
|---|---|---|---|
| `53e01c397101` | normal | none | - |
| `4444aa27a7dc` | bright ×1.3 | none | +0.13% |
| `b686b8a0a40b` | bright | `0-29` | +0.57% |
| `ad1e332586f1` | bright | `30-39` | **+8.39%** |

That is roycho87's shape (+4.4% / +4.1% / +13.8%), so diffusers' indices 0-49 are
ComfyUI's `diffusion_model.blocks[i]`, and Q3's assumption now holds as measured.
Our plain keystone carries much less on its own (+0.13% against his +4.4%). On this
chain, the 22-frame carried guide seems to outweigh the single keyframe row.

**Step 2 (the sharpness sweep) did not run.** It was cut by Don's ruling.

**Early gate (seg01-02):** control `2817a04202ae` on cuda:0, attention
`29923da18475` on cuda:1. The two arms differ only in the argument on seg02.

| Measure | Control | Attention |
|---|---|---|
| seg02 face-sharpness rise (seam + in-segment) | -16.9 | -7.7 |
| seg02 face sharpness, first 1 s → last 1 s | 103.6 → 92.0 | 127.1 → 101.3 |
| seam-1 `jump_ratio` | 1.09 | 1.19 (1.09x) |
| SFace, seg02 last 1 s | .907 | .894 |

- **It fails on its letter.** The attention arm's "rise" is higher than the control's,
  and the gate asked for no worse. The seam passes.
- **Neither arm creeps sharper.** Both fall through seg02. The gate's direction came
  from #1325 and can't tell "creeps sharper" apart from "keeps more of the keystone's
  detail".
- **The op acts.** seg02 opens nearer the keystone's sharpness, and pose and framing
  shift (PSNR 17.7 dB against the control). But the face crops show no clear texture
  or identity win.
- **The LSE op costs about +5% per pass.** That is an estimate across two cards,
  corrected by the identical seg01, because `run_workflow` has no device pin.
- **Dense runs about 2x sparse per pass on this chain,** not the +22-28% `speed.md`
  recorded for short clips. seg01 took 71 s per pass against 34 s, and seg02 86-100 s
  against 47 s.

**Don's verdict (2026-11-19): gate failed, stop.** He watched both takes and saw no
clear win.

Stage cost isn't recorded, because the stage comments don't name `usage:` figures.
lem time: about 0.4 h for segment 1, 2.2 h for step 1, and 1.9 h for the gate.

## Cut, and what brings each back

- **Step 2, the face-sharpness group sweep.** Comes back only if Don wants to go
  after face sharpness through attention again. It would run narrow late groups,
  about 2.5 h dense.
- **The full arms (6 segments).** Come back after a step 2 pick that wins on a gate.
  At the dense per-pass figures above, both arms together are about 5-6 h.
- **The chain key (`keystone: segment_1` + attention) and the other slots.** Come back
  only on a viewing win. `first` and `last` would be wired first, because their
  keystone sits in the keyframe rows.
- **A sparse-aware re-split.** Any positive result would be dense-only, while the
  shipped templates run sparse (Q5). That is a new stage, proposed only after a
  positive result.
- **The moving-shot arm.** Comes back only if arm 1 helps. The keystone's pull-back
  to segment 1's view is the risk there.

## What stays

`experimental_keystone_attention` stays experimental and behind the flag, with no
template or default change. It is the attention seam a revival starts from, and its
block numbering has now been checked on lem.
