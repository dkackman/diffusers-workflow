# FLUX 3 Image Klein workflows

Text-to-image, image editing and multi-reference generation with Black Forest Labs'
[FLUX 3 Image Klein](https://huggingface.co/black-forest-labs/flux-3-image-klein): a
5.6B distilled flow map that draws in 16 steps with no guidance scale and no negative
prompt. It pairs the FLUX.2 VAE with a Qwen3-VL-4B text encoder. The weights are gated
under the FLUX Kommunity License: accept it on the model page and run `hf auth login`
before the first download.

Settings come from the Hugging Face card, BFL's
[flux-3-image-klein](https://github.com/black-forest-labs/flux-3-image-klein)
repository, [docs.bfl.ml](https://docs.bfl.ml) and the diffusers pipeline source
(`diffusers.modular_pipelines.flux3`, merged on 2026-10-15 and not yet in a release).
These templates need diffusers installed from git. Audited against those sources on
2026-10-15 ([the audit](../../../docs/proposals/audits/2026-10-15-flux-3-klein-audit.md)).

Klein was trained on long captions written by BFL's prompt upsampler, and BFL says not
to use it without one: a short prompt turns into a generic picture. The two upsampler
builtins (`flux3_klein_upsample` and `flux3_klein_upsample_edit`) run BFL's
`flux-3-image-klein-pup` for you, and the system prompts they give it are BFL's own,
read from the pup repository at a pinned commit (see below). To write a caption by hand,
read BFL's [prompting guide](https://docs.bfl.ml/guides/prompting_summary). It is
written for the API model, which upsamples on the server side, so treat it as a
description of the target caption and not as advice for a short prompt.

An agent driving this family from Claude Code has the `flux3-klein` skill of the
[dw plugin](../../../plugins/dw/README.md), which chooses between these templates and
states the family's rules.

Read them in this order and each introduces one new idea on top of the last.

## The templates

| Example | What it introduces |
| ------- | ------------------ |
| [text-to-image.json](text-to-image.json) | The baseline. The prompt is already a full upsampled caption (BFL's man-walking-a-dog example), 16 steps, no guidance. Width and height are multiples of 16, up to 4096, and the pipeline only complains after the load, so the workflow declares the rule up front. 24 s on an RTX 3090 at 1024x1024 (0.8 s/step), peaking at 22.1GB reserved |
| [upsample-then-generate.json](upsample-then-generate.json) | A short request becomes a caption first: the pup upsampler writes it, `release_models` frees the upsampler (about 18GB), and Klein draws from the result. The caption is saved as an intermediate, so it can be read, edited or reused as the prompt of text-to-image |
| [image-edit.json](image-edit.json) | One reference and an instruction. The upsampler sees the same image as the pipeline. With no width or height the output is 1 MP at the reference's aspect ratio, on the 16-pixel grid |
| [multi-reference.json](multi-reference.json) | Several references, in order. A caption written by hand names them `<ref_image_0>`, `<ref_image_1>` and so on (zero-based). The first reference sets the aspect ratio. BFL trained on up to 10; the pipeline does not enforce that, so more than 10 run but are outside what the model learned. Two references ran in 45 s on an RTX 3090 (2.6 s/step, references add tokens), also peaking at 22.1GB reserved |
| [layout-boxes.json](layout-boxes.json) | Placement by box. Elements are tagged `<id>` in the prose, and a JSON table gives each id a `[top, left, bottom, right]` box on a 0 to 1000 grid. Boxes are hints, not masks: a small boxed element may not appear, and elements can run past their box |

## dw craft, not vendor guidance

Three things in these templates are this repository's choices. BFL does not say them.

**Every component is placed explicitly.** Each of the transformer, text encoder and VAE
carries a `device`. The modular pipeline's execution device is its first model
component's device, so a component left unplaced stays on the CPU, and the pipeline then
encodes the prompt and draws its noise there. The failures looked like `index is on cpu`
in the text encoder, and `Cannot generate a cpu tensor from a generator of type cuda`.
`residency: on_demand` on the text encoder breaks for the same reason, because the
encoder step moves `input_ids` to the execution device before the wrapper moves the
weights. The weights are about 19GB in bf16, so with everything resident a 24GB card
fits 1024x1024 with under 2GB to spare. A larger output, or several references, wants
the VAE on demand or a quantized transformer, and neither has been measured yet.

**The upsampler is released before Klein loads.** The pup (Qwen3.5-9B, Apache-2.0,
ungated) is about 18GB and Klein about 19GB, which do not fit a 24GB card together. The
upsample step sets `release_models`, and the generation step loads after it.

**The upsampler's system prompts are read from the Hub at a pinned commit.** BFL's
prompts are in the pup repository's `system_prompts.json` and are still changing, so the
builtins name the file, the key and a full 40-character revision through the
`system_prompt.from_hub` form of `text_generation`, and never carry a copy. A template
runs the same text until someone re-pins it. The builtins also turn the chat template's
thinking mode off, which BFL's usage assumes.

## Not here yet

These follow-ups are dated 2026-10-15 and are in the audit's reading order:
- The base (non-distilled) checkpoint with classifier-free guidance, which needs the modular `guider` to be reachable from a workflow.
- Upsampled layout boxes from a short request (BFL's `t2i_bbox`), which need BFL's box planning and reconciling code.
- LoRA, a typography template, the fp8 and NVFP4 checkpoints (fused only on Ada and Blackwell cards), and a measurement on Apple Silicon.
- Re-pinning the system prompts when diffusers PR #15058, which carries them, merges.
