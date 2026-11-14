# MiniMax-H3 precision burn probe; tensor parallelism deferred (#1319)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out session,
2026-11-14). Don filed the idea on 2026-11-13 from the diffusers 0.43.0 release
notes: diffusers #14609 adds tensor parallelism (TP) for the H3 transformer. The
idea was to shard H3 across lem's two RTX 3090s, run it at a higher precision
than int8, and use that to test whether quantization error adds to the facial
burn that builds over a narrated chain. Plan v1 is the plan comment on #1319. It
was approved at its defaults on 2026-11-13, ruled by a lead session under Don's
delegation, and built as one stage, #1346. The plan comment still holds the full
sweep. This file records what was measured and what stayed cut.

## Verdict: build smaller

**TP can't run H3 above int8 on lem.** The transformer is about 33B parameters,
about 66 GB in bf16, or about 33 GB per rank at TP=2. A reviewer on #14609 reports
`tp_degree=2 OOMs at ~31 GiB/rank`. TP also rules out block offload, because each
shard has to stay on its card. At int8 a shard might fit, but all it would buy is
less offload churn, and only if SDNQ layers survive DTensor sharding, which is
untested.

**The precision question doesn't need TP.** A bf16 transformer under the
existing block group offload needs host RAM, not VRAM. So the feature became one
bench probe on one card, on diffusers 0.42.1, with no wait on #1315.

## Plan history

- **v1 (approved 2026-11-13).**
  - Q1 [scope]: cut TP and defer it under triggers (a) and (b) below.
  - Q2 [spend]: one lem night for a gate segment and a 12-segment arm, with a
    45-minute-per-segment stop.
  - Q3 [taste]: Don judges the face strip. This doesn't gate the stage.
  - Q4 [none]: a sibling bench file rather than a `weights_dtype` split.
  - Condition from the ruling: run overnight with nothing else on either card.
- **Correction to the idea found by the sweep:** `narrated-take.json` feeds
  `variable:weights_dtype` to both the transformer and the text encoder, so the
  arm had to change the transformer alone.

## What was built

### Stage 1 (#1346): bf16-transformer burn probe (bench only)

Merge `84bb0d05` on `develop` (commits eda971ec and 2780cbe4 for the workflow,
ca83da67 for `bench/face_windows.py`, 5016b2cf for the record), plus a
record-only follow-up, f58b4ce7.

- **`bench/workflows/narrated-take-bf16.json`:** `narrated-take.json` with the
  transformer's SDNQ config removed. The transformer loads in bf16 under the same
  block-level group offload. The text encoder and VAE stay int8. `host_ram_gb` is
  110, set from the gate. The server can't reach `bench/` by path, so both jobs
  were submitted inline, and `get_job_workflow` holds what ran.
- **`bench/face_windows.py`:** per CP1 window, the median Laplacian variance of
  the face crop, its mean HSV saturation, and the SFace cosine against the face at
  0.5 s. It also writes the face strip. Face-crop sharpness and saturation were
  controller scratch until then. They are now in the repo.
- **Gate `db33868da007`:** stop_after 1, seed 1, other card idle. 951 s wall,
  worker peak RSS 108,325 MB (mostly the mmap'd bf16 checkpoint), committed peak
  37,124 MB, peak reserved VRAM 22,142 MB. No stop condition was hit.
- **Arm `af390360c19b`:** 12 segments, seed 1, `tier: best`, shot dialogue, no
  counters. 223.7 min, against the int8 control `d5584b91816d` at 233.0 min. The
  arm reused the gate's cached pipeline, so its time has no load in it. bf16 did
  not cost the 5–8 h the plan estimated.
- **Record:** `bench/records/narrated.md`, section "bf16 transformer, seed 1".
- **Strip:** `asset:bench/1346-bf16-face-strip.jpg` in workspace
  `constitution-walk-3`, int8 row on top, faces at 0.5, 40, 60 and 85 s. It was
  copied in by `scp`, because `upload_asset` couldn't read the local file.

**Result: precision is not a burn lever at int8→bf16.** Growth from each arm's
own segment 1:

| measure | int8 | bf16 |
|---|---|---|
| face_drop at segment 12 | 0.207 | 0.225 |
| face-crop sharpness growth | 9.0x | 8.7x |
| saturation growth | +18.7 | +18.9 |
| SFace similarity growth | -.18 | -.22 |

The result isn't marginal, so no second seed was asked for. `tiers.md` still
stands: int4→int8 removes most of the S1 burn. The step above int8 doesn't help.

**Bounces: 0.** Two build sessions ended while the arm was still running, and
the driver parked the stage for making no progress. A lead session handed it back
once the arm had finished. The jobs ran on lem's hand-started `595cee1f` server
(#1334). The workflow is bench-only, so the result doesn't depend on that.

Cases verified: C-F814 to C-F818, over MCP, with no GPU jobs. C-F814 validated
the realized workflow inline, because the server can't resolve a `bench/` path.

## Cost and bounces

| Stage | Bounces | Cost |
|---|---|---|
| #1346 | 0 (one no-progress park while the arm ran) | Session cost not recorded on the stage; the plan estimated $3–5. GPU: 15.9 min gate + 223.7 min arm on cuda:0, other card idle |

## Deferred and open

- **Tensor parallelism: deferred, not declined.** It comes back as its own
  feature if either trigger holds:
  - (a) precision is shown to be a burn lever **and** a card configuration exists
    where a higher-precision shard fits (40 GB or more per card, or an
    fp8/int8-plus-unquantized-modulation scheme that fits 2×24 GB without
    offload). **This probe shows (a) is not met.**
  - (b) a measurement shows block-offload transfer is a large share of int8 wall
    time on lem, so TP at int8 would buy real speed. That needs a PCIe-versus-NVLink
    check on lem first. Not measured.
  - What TP would touch is the sweep in plan v1. Every layer assumes one job per
    card: the device pin, worker slot, dispatch, host-RAM gate, VRAM ceiling,
    observed cost, job record and its REST/MCP shape, and about 40 tests.
- **No "ship higher precision" feature follows** unless Don's Q3 viewing of the
  strip disagrees with the measures. Q3 is still with him.
- The burn itself stays open. Other levers are #1313 and #1322.
- `validate_workflow` can't reach `bench/` workflows by path or name, so bench
  probes are validated inline.
