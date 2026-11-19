# MiniMax-H3 fl2va/ref2va hybrid transformer probe (#1441)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out session,
2026-11-19). Don filed the idea on 2026-11-18 from
Adudeguyman/ComfyUI-Fantastic-MiniMaxH3-PromptBuilder's `REFMODS.md`, which
recommends `smhfacct/Minimax-H3-fl2va-ref2va-hybrid-models` over MiniMax's
ref2va weights. The claim was reference conditioning at fl2va quality. Plan v1
is the plan comment on #1441. Don approved it at its defaults on 2026-11-18,
and it was built as one stage, #1463. The plan comment still holds the full Hub
sweep (provenance, licence, format). This file records what was measured and
what stayed cut.

## Verdict: build smaller, a bench probe only

- **The Hub files were not pulled.** There are four ComfyUI single-file int8
  merges, 21 GB each, built from unnamed "pruned int8-convrot" bases. diffusers'
  `from_single_file` converter for `MiniMaxH3Transformer3DModel` refuses a
  pruned checkpoint and has no int8 dequant, so loading them would have needed
  a converter that dw would own.
- **The recipe is a tensor graft, so we built it ourselves** from the two bf16
  transformers lem already caches (`MiniMaxAI/MiniMax-H3` `transformer` and
  `transformer_ref`). Without quantization noise, this tests the claim more
  cleanly than the int8 files would.
- **The issue's main prize doesn't follow from a checkpoint.** References in a
  guide chain are refused because a `references` step runs diffusers' Ref2VA
  block graph, which has no anchor or chain steps. The fl2va turbo LoRA on a
  reference shot is refused by partition name, not by weights. Neither is
  unlocked by a hybrid in `transformer_ref`.
- **"MiniMax admitted faults"** is unsourced. The model card's "known
  training-quality issue" has no link.

## Plan history

- **v1 (approved 2026-11-18).**
  - Q1 [spend]: run stage A, b20-49 only, with the scratch deleted after the
    record.
  - Q2 [taste]: SFace sets the floor (within 0.02 of stock), and Don's look
    decides quality.
  - Q3 [trust]: no Hub download.
  - Condition from the ruling: keep the `df` stop check on lem's model disk.

## What was built

### Stage A (#1463): graft b20-49 locally, bench one reference shot (bench only)

Merges on `develop`: 6fcd3d17 (`feat/1441-A-h3-hybrid`, 03bce6a4 + 3d7c42ae),
then record-only follow-ups 8e0bf7a3 and c3d3365c (36e2f0a0).

- **`bench/h3_hybrid_graft.py`:** reads both parents' shards per tensor,
  lazily, asserts identical names and shapes, takes ref2va's `adaln_proj` for
  the chosen blocks and fl2va everywhere else, and writes ref2va's
  `config.json`. It prints a JSON tensor report.
- **`tests/test_bench_h3_hybrid_graft.py`:** covers block-range selection,
  whole-number block matching, ref-only tensors being taken, fl-only and shape
  mismatches being refused, and a two-shard safetensors round trip.
- **`bench/workflows/probe-1441-hybrid-ref.json`** and
  **`probe-1441-stock-ref.json`:** the two arms differ only in
  `load_components.pretrained_model_name_or_path.transformer_ref`. The server
  can't reach `bench/` by path, so both arms were submitted inline.
- **`bench/records/h3-hybrid.md`:** method, disk, tensor report, jobs, wall
  times, SFace, and a verdict line left blank for Don.

### What was measured

- **Parents.** The two parents' 638 tensor names and shapes are identical. The
  card's claim holds: 60 tensors (`transformer_blocks.{20..49}.adaln_proj.linear.
  {weight,bias}`) came from ref2va, and there are no ref-only tensors. The two
  parents' `config.json` files are byte-identical.
- **Size.** Each parent and the graft are 66.3 GB in bf16. The plan's estimate
  of ~40 GB was low.
- **Renders** (workspace `bench`, seed 1441, `cast/estela.jpg`, 124 frames at
  960x544, 9 steps, ref2va 8-step turbo, `{int4}`):

| Arm | Job | Wall | Load | SFace median / min |
|---|---|---|---|---|
| stock ref2va | `7cd603a1922f` | 247.0 s | 5.3 s (cached) | 0.855 / 0.759 |
| hybrid b20-49 | `d5a6c03f9ee3` | 308.6 s | 48.8 s (uncached quantize) | 0.788 / 0.561 |

  The scoring jobs were `d7589d81e015` (stock) and `3a77473b74fa` (hybrid),
  using `analyze_face_drift` with `samples_per_shot` 16. The tester's
  independent re-score reproduced the hybrid's figures exactly.
- **Result.** The hybrid misses the SFace floor by 0.067, against 0.02
  allowed. b25-49 was not run: the plan ran it only if b20-49 showed a quality
  gain, and that call is Don's. The record's verdict line is still blank for
  him.
- **Unchanged.** An FL2V LoRA on the hybrid step is still refused by the same
  rule as on stock. No catalog entry was added, `reference-to-video`'s defaults
  are unchanged, and nothing was downloaded. The scratch graft was deleted.

### Bounces

Stage A was bounced three times before it verified on the fourth pass, all on
the record and not the code:
1. M-F445: the record wasn't quoted (disk, tensor report).
2. M-F444 and M-F445: the scoring settings and wall times weren't quoted.
3. M-F445: the `df` reading came after the graft. The plan's first step had
   been missed. The graft was re-run with the `df` taken first, at 11:51 UTC
   (scratch disk 671.8 GB free, model disk 137.6 GB free, against a bar of
   132.6 GB). A `torch.equal` check showed the rerun was tensor-identical to
   the rendered graft, so the render was not repeated.

Two engine bugs surfaced during verify and were fixed as their own issues:
- #1488: a local component path still listed its repo in
  `plan.downloads_required`.
- #1489: a second LoRA with no `weight_name` slipped past the partition rule.

Stage cost isn't recorded, because the stage comments don't name `usage:`
figures.

## Cut, and what brings each back

- **Hub download (smhfacct int8 files).** Comes back only if a local graft wins
  and Don wants an int8 variant. A converter would then get its own plan.
- **Other block ranges (b25-49, b30-49).** Comes back if Don's look at the two
  renders sees a quality gain. The rebuild is `--first 25` on the graft script,
  about a minute.
- **Catalog entry or template.** Comes back if a hybrid holds the SFace floor
  and wins Don's look. It would be a `workflows/models/` config of
  `templates/minimax/reference-to-video` with the per-component override.
- **The fl2va turbo LoRA on a hybrid.** Comes back on the same win, as a
  checkpoint-aware LoRA partition filed as its own feature.
- **References in a guide chain.** Comes back as its own feature, a new
  dw-owned block sequence. The weights don't unlock it. Related: #980
  (FastH3 Omni, parked).
