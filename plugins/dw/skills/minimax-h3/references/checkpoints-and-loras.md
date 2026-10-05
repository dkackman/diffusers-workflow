# MiniMax H3: checkpoints and LoRAs

Part of the `minimax-h3` skill; its *Hard rules* say when to read this.

A checkpoint comes with a canvas, a sigma shift and an alpha and they move
together - change one, change all. `video_shift`, `audio_shift` and
`lora_alpha` are variables everywhere, so a swap is arguments, not a file.

Three combinations are tested, plus two additions below, nothing else:
544p FL2VA turbo, 960x544, shift 12/3, alpha unset - default; 768p FL2VA
turbo, 1344x768, shift **6**/3, alpha unset - `video-with-audio-768p`; 768p
Ref2VA turbo, shift 12/3, alpha unset - every `ref2va` template. The two 768p
LoRAs differ in shift; do not generalise.

Two 768p additions (#585, one prompt): a **4-step draft** on
`video-with-audio-768p` - `lora_weight_name=minimax_h3_fl2v_turbo_4step_v1.2_768p_bf16.safetensors`,
`num_inference_steps=5`, shift 6/3, alpha unset; ~25% faster, one cold run;
render at 8. **Realism People** as a second `loras` entry on its t2va
step (`fal/MiniMax-H3-Realism-People-LoRA`,
`h3-realism-people-t2v-i2v-r2v.safetensors`, scale 0.7, own `adapter_name`;
a saved copy): validate warns the name fits neither partition, so never on
a `ref2va` template.

Acc PDD, HyperFlow and drozbay FastH3 need their own loaders, not `loras`:
don't try them. Any other LoRA (a style, a motion fix): `list_loras` first -
each entry names its partition, trigger and scale; `trial` is unproven,
`rejected` says why it fails.
