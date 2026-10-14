# MiniMax H3: the base-model quality path and measured speed levers

Part of the `minimax-h3` skill; its *Hard rules* say when to read this.

Measured on a CUDA box (RTX 3090 24 GB, `sage_hub`), 1344x768, 4 references
(2 portraits, 2 voice clips), 124 frames, seed 42, #863. **These are CUDA
numbers**: MPS has no sage backend and runs several times slower, so take the
ordering, not the minutes. Face score is realism at native-resolution crops,
judged from frames by a model (5 = cinema-grade photographic).

## Quality ladder

Base model = `lora_model_name: null`, shift video 12 / audio 3. Turbo = the
shift the LoRA names.

| Rung | Canvas | Weights | LoRA | Steps | Attn | Min | Face /5 |
|---|---|---|---|---|---|---|---|
| L0 | 960x544 | int4 | ref2v 4-step v0.1 | 5 | sage_hub | 3.5 | 2 |
| L1 | 1344x768 | int4 | ref2v 4-step v0.1 | 5 | sage_hub | 5.4 | 2.5 |
| L7 | 1344x768 | int8 | ref2v 4-step v0.1 | 5 | sage_hub | 7.2 | 3 |
| L3 | 1344x768 | int8 | ref2v 8-step v1.0 768p | 9 | sage_hub | 11.2 | 3 |
| L4 | 1344x768 | int8 | ref2v 8-step v1.0 768p | 9 | native | 14.5 | 3 |
| B4 | 960x544 | int4 | none | 12 | sage_hub | 6.6 | 3 |
| B1 | 960x544 | int4 | none | 30 | sage_hub | 15.1 | 3 |
| B5 | 1344x768 | int4 | none | 12 | sage_hub | 11.8 | 3.5 |
| B6 | 1344x768 | int4 | none | 8 | sage_hub | 8.5 | 1.5 (duplicated a character) |
| B3 | 1344x768 | int8 | none | 12 | sage_hub | 12.9 | 3.5 |
| L5 | 1344x768 | int8 | none | 30 | sage_hub | 29.5 | 4 |
| L6 | 1344x768 | int8 | none | 30 | native | 44.2 | 4 |
| P1 | 1344x768 | int8 | none | 30 | sage_hub + `first_block` 0.05 | 23.1 | 4 |
| C1 / C2 (close-up) | 1344x768 | int8 | none | 12 / 30 | sage_hub | 13.9 / 22.8 | 4 / 4 |

- **Faces**: turbo LoRAs plus int4 cause the smeary faces, not resolution. The
  base model at int8 fixes it: `lora_model_name: null`, 12-30 steps, shift
  12/3. A close-up is already 4/5 at 12 steps.
- **Composition** (who stands where): attention never moves it; quantization
  doesn't on a turbo LoRA (L1 = L7) but does on the base model (int4 vs
  int8); a different LoRA family re-stages; base step count doesn't.
- **Steps**: the base floor is 12. 8 steps duplicated a character (B6).
- **Two-character shots at 960x544 smear** (~60 px faces): render them at
  1344x768 from the start, not at 544 and upscaled.
- **Draft then final**: a 12-step draft (~13 min) of the wide, then the
  30-step final with `first_block` 0.05 (~23 min). Draft to final carries
  blocking but not always props.
- **One int8 job at a time**: host RAM.
- **Frame ceiling** at 1344x768 with 4 references is 124; 141 was refused
  (23.9 GB > 23.6). Write to 124 or drop a reference.

## Prompting the base model

- It invents an extra person unless every subject is placed explicitly
  ("Howie at frame left, Pat at the right end of the desk"). A head-count
  sentence ("exactly two people") alone failed three times.
- Name the set dressing (chair, desk items) in every shot of a sequence, or
  each shot draws its own set: a chair changed between two close-ups.

## What does not help

- **Realism People LoRA**: its header says `base_model: minimax-h3-fl2va`,
  `training: text_to_video`, rank 32. It is inert on `reference-to-video`
  (Ref2VA) despite `r2v` in the filename. It works on FL2VA/I2VA (4/5) but
  those lose the voice and identity references.
- **LTX-2.5 refine-in-place on an H3 shot** is not a face fix: cleaner edges,
  waxy skin, identity drift (2.5/5, 2/5).

## Speed levers

Per phase, from the job event log (minutes unless noted):

| Rung | Change | Denoise | s/pass | Total | vs baseline |
|---|---|---|---|---|---|
| B3 | 12 steps | 10.24 | 55.8 | 12.89 | |
| P2 | + `first_block` 0.05 | 10.31 | 56.3 | 12.79 | 0% |
| P4 | + `text_kv` | 10.24 | 55.9 | 12.69 | ~0% |
| P3 | + compile `repeated_blocks` | 14.02 | 288.6 first, 55.2 rest | 16.58 | +29% |
| L5 | 30 steps | 26.97 | 55.8 | 29.49 | |
| P1 | + `first_block` 0.05 | 20.61 | 42.2 avg | 23.06 | **-22%** |
| P5 | + `taylorseer` interval 3 order 1 | | | OOM at 2 min | |

Denoise is 80-91% of a job, ~56 s/pass with int8 weights streamed by
block-level group offload: compute-bound, not PCIe-bound.

- **`sage_hub`**: -23% at turbo 8-step, -33% at base 30, visually lossless.
  Advise it first where the box has it: `attention_backend=sage_hub`, an
  argument on every H3 template (default `null`, the H3 step only). It needs
  CUDA, the `kernels` package and `DIFFUSERS_TRUST_REMOTE_KERNELS=true`; leave
  it `null` on mps or cpu, where it passes validate and fails at run time.
- **`first_block` 0.05**: -22% at 30 base steps with identical staging and
  faces (PSNR 27.5 dB against L5). It skips nothing at 12 steps or on a
  turbo schedule: use it at 30 base steps only.
- **`torch.compile`**: +289 s on the first pass, ~1% per pass after. A net
  loss for one job; it breaks even only across ~16 shots at one shape in one
  warm worker.
- **`text_kv`**: no measurable effect on Ref2VA.
- **`taylorseer`**: OOM at this canvas with 4 references.

Open: `first_block` 0.08-0.10 at 30 steps (speed against face cost);
`first_block` at 20 steps against plain 30; whether P1 holds on a close-up and
a second seed; MagCache needs a calibration run (no H3 preset).
