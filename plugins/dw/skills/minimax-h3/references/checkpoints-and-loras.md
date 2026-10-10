# MiniMax H3: checkpoints and LoRAs

Part of the `minimax-h3` skill; its *Hard rules* say when to read this.

A checkpoint comes with a canvas, a sigma shift and an alpha and they move
together - change one, change all. `video_shift`, `audio_shift` and
`lora_alpha` are variables everywhere, so a swap is arguments, not a file.

Four combinations are tested, plus one addition below, nothing else:
**default** - 4-step FL2VA v1.2 (`minimax_h3_fl2v_turbo_4step_v1.2_768p_bf16.safetensors`),
shift **6**/3, `num_inference_steps=5`, alpha unset, on every T2VA/FL2VA
template at 960x544 and 1344x768 (since 2026-10-10; trained at 768p,
measured at 544p on one prompt). 544p FL2VA 8-step, 960x544, shift 12/3,
9 steps - the slower alternative; 768p FL2VA 8-step, 1344x768, shift 6/3,
9 steps; 768p Ref2VA 8-step, shift 12/3, 9 steps - every `ref2va` template
(no tested 4-step Ref2VA file). Swap file, shift and steps together. The
544p 4-step v0.1 file renders noise. v1.2 records alpha 8; v1.0/v1.1 record
128 - never state an alpha the header doesn't.

**Realism People** (#585) as a second `loras` entry on a t2va
step (`fal/MiniMax-H3-Realism-People-LoRA`,
`h3-realism-people-t2v-i2v-r2v.safetensors`, scale 0.7, own `adapter_name`;
a saved copy): validate warns the name fits neither partition, so never on
a `ref2va` template.

Acc PDD, HyperFlow and drozbay FastH3 need their own loaders, not `loras`:
don't try them. Any other LoRA (a style, a motion fix): `list_loras` first -
each entry names its partition, trigger and scale; `trial` is unproven,
`rejected` says why it fails.
