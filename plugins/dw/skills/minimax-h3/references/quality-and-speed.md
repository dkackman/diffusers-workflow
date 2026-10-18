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
| L1s / L7s | 1344x768 | int4 / int8 | ref2v 4-step v0.1 | 5 | sparse | 5.2 / 6.5 | = L1 / L7 (Don, 3 seeds) |
| B3s | 1344x768 | int8 | none | 12 | sparse | 12.1 | = B3 (Don, 3 seeds) |
| L5s | 1344x768 | int8 | none | 30 | sparse | 26.6 | = L5 (Don, 3 seeds) |

- **Faces**: turbo LoRAs plus int4 cause the smeary faces, not resolution. The
  base model at int8 fixes it: `lora_model_name: null`, 12-30 steps, shift
  12/3. A close-up is already 4/5 at 12 steps.
- **Composition** (who stands where): `sage_hub` against `native` never moves
  it. `sparse` against either keeps the shot but shifts details (a figure's
  height, props), so a draft and its final use the same backend. Quantization
  doesn't move it on a turbo LoRA (L1 = L7) but does on the base model (int4
  vs int8); a different LoRA family re-stages; base step count doesn't.
- **Steps**: the base floor is 12. 8 steps duplicated a character (B6).
- **Two-character shots at 960x544 smear** (~60 px faces): render them at
  1344x768 from the start, not at 544 and upscaled.
- **Draft then final**: pick the path by target and render a cheap draft
  before the final. See *Ramp* below.
- **Submit freely**: the server holds a second heavy job, queued with
  `waiting_for`, until host RAM frees.
- **Frame ceiling** at 1344x768 with 4 references is 124; 141 was refused
  (23.9 GB > 23.6). Write to 124 or drop a reference.

## Ramp

Spend GPU time on quality in two tiers: a draft to approve the shot, then a
final of the same shot. Pick the path by the target **before the first
render**, because a draft only carries to a final on its own path.

### 1. Pick the path

Both tiers use the same template (`templates/minimax/reference-to-video`, or
`templates/minimax/dialogue-short` for a cut). `attention_backend=sage_hub`
is an argument on every H3 template, `dialogue-short` included. Set it at
both tiers on CUDA: it never moves composition.

The H3 templates default to `attention_backend=sparse` with top-k routing
(`sparse_topk` 0.15), which shifts details against sage (see *Composition*).
The sparse rows below were timed at `sparse_tau` 0.5, 6-12% faster than
`sage_hub` with equal faces and staging on 3 seeds (#931). Top-k is faster
again at equal or better faces (#932): L7 ~6.0 min, L5 ~24.4 min. Its
draft-to-final carry is not separately measured; the draft and the final
share the setting either way. Draft and final always share one backend: pass
the backend the row names at both tiers.

**`dialogue-short` at 1344x768** (every rung below is that canvas): a shot
with 4 references must be 124 frames (*Frame ceiling* above). The default
`shots` list's `tag` (`shots[4]`) is 141 frames with 4 references, so
`validate_workflow` refuses it at `variables.shots[4]`. Pass your own
`shots` argument at both tiers: `get_workflow` the template, copy
`variables.shots`, set `shots[4].num_frames` to 124 (or drop a reference
from it), and pass the list unchanged at draft and final. A cut you write
follows the same rule: every 4-reference shot at 124. Check the trimmed
shot's line still fits 5.17 s.

| Target | Draft | Final | Only these change |
|---|---|---|---|
| **TV** (sitcom, social, previs) | **L1s**, ~5.2 min | **L7s**, ~6.5 min | `weights_dtype` `{int4}` to `{int8}` |
| **Cinema** (realistic faces), sage | **B3**, ~12.9 min | **P1**, ~23.1 min (or L5, ~29.5) | `num_inference_steps` 12 to 30; P1 adds `cache` |
| **Cinema**, sparse | **B3s**, ~12.1 min | **L5s**, ~26.6 min | `num_inference_steps` 12 to 30 |
| **Close-up** | **C1**, ~13.9 min | ship the draft | nothing |

- **TV.** Both rungs are 1344x768 on the 4-step Ref2VA v0.1 LoRA:
  `lora_model_name` `lightx2v/Minimax-h3-Turbo`, `lora_weight_name`
  `minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors`,
  `num_inference_steps` 5, `video_shift` 12, `audio_shift` 3. L1 is
  `weights_dtype` `{int4}`, L7 is `{int8}`; `attention_backend=sparse` (the
  default) at both tiers. Under sparse, L1s -> L7s kept staging on 3 of 3
  seeds and matched sage; the TV faces smear a little under either backend
  (the turbo LoRA, not attention; #931). **L0** (the same at 960x544,
  ~3.5 min) is a throwaway sketch only. Its staging did carry to L7 (below),
  but it smeared a face at both new seeds, so a face can't be approved on it,
  and two-character shots at 960x544 smear. Render those at 1344x768 from
  the start. The L7 final (3/5) is **not confirmed by ear or in playback**.
- **Cinema.** The base model: `lora_model_name` `null`, `weights_dtype`
  `{int8}`, 1344x768, `video_shift` 12, `audio_shift` 3. B3 is 12 steps, L5 is
  30. On `dialogue-short`, add the 124-frame `shots` above at both tiers. **P1** is L5 plus the template's `cache` argument
  `{"type": "first_block", "threshold": 0.05}` (-22%, the same staging and
  faces). Both tiers warn `lora_disabled`. That is expected: the warning asks
  for the base schedule, and these values are it. Never draft at B4 (960x544
  int4 base) for a 1344x768 final: its staging held on only 2 of 4 seeds.
  The sparse pair (B3s -> L5s) is the same arguments with
  `attention_backend=sparse`: skin equal to sage and staging carried on 3 of 3
  seeds (#931). The sage pair with P1 is still the fastest cinema final;
  sparse with the `first_block` cache is not measured, so don't combine them.
- **Close-ups**: C1 is B3's arguments on a close-up shot. It scores 4/5 at
  12 steps (C1 = C2), so the draft is the deliverable.

### 2. What carries from draft to final

The composition rules under *Quality ladder* hold: sage never moves
staging, sparse shifts details (so it never changes between tiers), and
neither does base step count; a different LoRA family re-stages; so does int4
against int8 on the base model. On top of those:

- keep the seed, canvas, prompt and references identical between the tiers;
- blocking carries, but props and set dressing sometimes don't (a jacket on
  a chair in C1 only).

Measured (#861) on seeds 42, 7 and 1234, judged by Don: B3 -> P1, C1 -> C2
and L0 -> L7 each held staging on **3 of 3** seeds, and B3s -> L5s under
sparse too (#931). L1 = L7 (the TV pair
itself) is measured on **1 seed** (42). B4 -> B5 held on only 2 of 4 seeds
(42, 7, 1234, 99). At seed 1234, B5 (int4) staged like B3 (int8): on the
base model the canvas moved staging more than the weights did. That is one
seed against the int4/int8 rule, which stands until it's measured further.

### 3. What to approve at each tier

- **Draft:** blocking, line delivery, timing and head count. Re-roll the
  seed or the prompt here, and only here.
- **Final:** faces, skin and hair. On these paths a final that re-stages
  usually has an argument the table doesn't allow: diff its arguments
  against the draft's before suspecting the seed.

### 4. How to judge

- Face crops at native resolution, at fixed frames
  (`get_output_frames`), the same frames at both tiers.
- A contact sheet (`get_output_frames(count=12)`) for staging.
- `check-script` for the lines.
- A 1-5 face score, recorded on the run with `set_note`.

Claude can't judge audio by ear (voice timbre, lip-sync feel, room sound)
or motion (smear, flicker) in playback. Say so in the note, and ask a
human for those.

### 5. Cost

Quote `validate_workflow`'s `plan.estimate` for each tier, draft plus
final. It scales by steps, canvas and frames (#866), but not by
`weights_dtype`, `attention_backend`, `cache` or a null LoRA. When one of
those moves it sets `low_confidence` and names the variable in `differs`.
Every rung above moves one. Quote that rung's minutes from the ladder instead
(the *Attn* column says which backend they were measured with), and say
which figure you are quoting.

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

- **`sage_hub`**: -23% at turbo 8-step, -33% at base 30 against `native`,
  visually lossless. `attention_backend` is an argument on every H3 template
  (the H3 step only); the default is `sparse`, 6-12% faster than `sage_hub`
  on the ramp rungs and runnable without extra packages. `sage_hub` needs
  CUDA, the `kernels` package and `DIFFUSERS_TRUST_REMOTE_KERNELS=true`; on
  mps or cpu it passes validate and fails at run time, while `sparse` there
  falls back to dense with a warning.
- **`first_block` 0.05**: -22% at 30 base steps with identical staging and
  faces (PSNR 27.5 dB against L5). It skips nothing at 12 steps or on a
  turbo schedule, yet still perturbs the result: use it at ~20 or more base
  steps only, never with the turbo LoRAs. It is the `cache` argument on every H3
  template, `{"type": "first_block", "threshold": 0.05}`; `null` (the default)
  is off.
- **`torch.compile`**: +289 s on the first pass, ~1% per pass after. A net
  loss for one job; it breaks even only across ~16 shots at one shape in one
  warm worker.
- **`text_kv`**: no measurable effect on Ref2VA.
- **`taylorseer`**: OOM at this canvas with 4 references.
- **Did a speed setting change the output?** Run it with and without the setting (same seed) and pass both to `compare_outputs`; read the PSNR against the bands in the workflow guide's *Assessing a run's output* (about 44 dB is rerun noise, 23-27 dB is changed numerics with the same shot, about 19 dB is a re-staged shot).

Open: `first_block` 0.08-0.10 at 30 steps (speed against face cost);
`first_block` at 20 steps against plain 30; whether P1 holds on a close-up and
a second seed; MagCache needs a calibration run (no H3 preset).
