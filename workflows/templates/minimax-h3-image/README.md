# MiniMax-H3-Image workflows

Text-to-image and reference-driven editing with MiniMax's
[MiniMax-H3-Image](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image): an image model
distilled to a fixed six-step sampler with its guidance baked in, conditioned by the same
Qwen3-VL text encoder as MiniMax-H3. It takes no step count, no guidance scale and no
negative prompt, and these templates declare none.

Settings come from the Hugging Face card, MiniMax's
[MiniMax-H3-Image](https://github.com/MiniMax-AI/MiniMax-H3-Image) repository, the
[diffusers documentation](https://huggingface.co/docs/diffusers/main/en/api/pipelines/minimax_h3_image)
and the diffusers pipeline source (`diffusers.modular_pipelines.minimax_h3_image`). The
weights are under the MiniMax-H3-Image Community License Agreement
([LICENSE](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/blob/main/LICENSE)): read it
before use, since its territory and commercial terms are narrower than an open licence's.
Audited against those sources on 2026-11-17
([the audit](../../../docs/proposals/audits/2026-11-17-minimax-h3-image-audit.md)).
The timings are from `bench/records/h3-image.md`, on an RTX 3090 with 128 GB of host RAM. They are priced on the slower of lem's two 3090s, which sits on a PCIe x4 link; this offload is bus-bound, so a card on an x16 link runs it in about half the time. A price covers a run that has just loaded its models, less the load itself: the first text encode after a load is about 10 s slower than a fully warm one, and the job's start-up and save add about 10 s more.

Read them in this order and each introduces one new idea on top of the last.

## The templates

| Example | What it introduces |
| ------- | ------------------ |
| [text-to-image.json](text-to-image.json) | The baseline. A detailed English description of the whole picture, used verbatim and truncated at 4096 tokens. The canvas is a multiple of 32 on both sides, declared up front because the pipeline only refuses one after the load. `size` picks 2048x2048 (the default) or one of the vendor's recommended sizes of about 16 MP. Priced at 2.3 min at 2048x2048 on cuda:0 (2.0 min fully warm) |
| [multi-reference.json](multi-reference.json) | One or more references in `images`, named in the prompt one-based in list order: `<Picture 1>`, `<Picture 2>` and so on. FLUX 3 Klein names its references zero-based (`<ref_image_0>`), so a Klein prompt does not carry over. A single-image edit is a list of one. The vendor states up to 16 references; three have been timed here. Priced at 2.7 min at 2048x2048 on cuda:0: 2.68 min measured after a load, less the load (2.2 min fully warm) |

## Hard rules

- **Host RAM, not VRAM, is the cost.** Warm, the recipe holds about 64 GB of host RAM
  (`host_ram_gb: 64`) and under 6 GB of VRAM (3.7 to 5.5 GiB reserved, measured).
- **A cold first run needs about 120 GB of host RAM.** Quantizing the weights on a host
  without the quantized cache peaked at 117 GB RSS (116,760 MB measured), and took
  38.2 min instead of 2.0.
- **No speedup below 2048x2048.** A denoise pass is bound by streaming the offloaded
  transformer's weights, not by compute: a true 1024x1024 took 15.1 s a pass against
  2048x2048's 15.2 s, and 2.5 min in all against 2.0. By default the pipeline also
  generates any canvas under about 4 MP at about 4 MP (its minimum generation area) and
  resizes back. The templates do not expose that minimum.
- **These templates are CUDA only.** The recipe quantizes on the GPU and offloads with
  CUDA streams. On Apple Silicon see *On mps* below.
- **The first download is about 64 GiB** (63.93 GiB measured, the text encoder aside,
  which comes from MiniMax-H3's repository and is shared with it).
- **No prompt enhancer.** MiniMax's prompt expander is a separate 27B model for
  text-to-image only, and its diffusers blocks run remote code; these templates do not
  use it. Write the full description by hand: task first, then layout, then details and
  style.

## On mps

On a 64 GB Mac these templates are refused at validate: `host_ram_gb: 64` plus the gate's
margin is about 70 GB, more than the host has, and on unified memory group offload frees
nothing. The still runs there instead from pre-quantized int8 folders made on a CUDA host,
through `bench/workflows/h3-image-mps-t2i.json`. That workflow loads one model at a time:
an `encode` step with only the text encoder, released, then a `generate` step with the
transformer and the VAE, fed `prompt_embeds` by `previous_result:`. Its folder paths are
host-specific, so it stays in `bench` and is never a catalog template.

Measured on mini-ai (M5 Pro, 64 GB, job `b99e189e1059`): one 2048x2048 still in 3.2 min
with the model load, denoise 77.9 s for the 6 passes, peak 34.2 GB of MPS memory, and a
host peak of about 60 GB RSS during the `generate` load (59.9 GB on `b99e189e1059`,
60.2 GB on `608885eca475`), leaving about 20 GB of the 64 GB. Nothing else host-heavy
should run beside it. dw switches off the transformer folder's saved quantized matmul on
mps (#1744) and says so in a job warning. The edit with references
(`bench/workflows/h3-image-mps-edit3.json`) has no mps measurement. The
record is `bench/records/h3-image.md`, *mini-ai (mps)*.

## dw craft, not vendor guidance

The recipe is this repository's, measured on a 24 GB card. MiniMax's own routes are
bf16 on an 80 GB card or the hosted API.

**The text encoder comes from MiniMax-H3.** It is the same Qwen3-VL encoder, so
`load_components` loads it from `MiniMaxAI/MiniMax-H3` at the commit the H3 templates
pin (`0919fb68`). A host that runs H3 already holds its download and its int8 quantized
copy, and the two families share them.

**The encoder is truncated to 51 layers.** The pipeline conditions on `hidden_states[50]`,
so the language model's later layers and its `lm_head` are never used: `remove_modules`
drops the head and `truncate_layers` keeps 51 layers, which cuts the encoder's host RAM
and its pass time. The encoder runs under leaf-level group offload.

**The offload layout.** The transformer and the text encoder are int8 SDNQ, quantized on
the GPU and returned to the CPU. The transformer runs under block-level group offload,
two blocks per group with CUDA streams; the VAE is loaded onto the GPU on demand.

**Changing the quantizer arguments repeats the cold quantize.** The quantized cache is
keyed on the `quantization_config` arguments, `weights_dtype` among them. Any change, even
to `modules_to_not_convert`, misses the cache and pays the cold pass and its 120 GB host
RAM peak again. The arguments here match the bench workflows exactly, so a host that ran
the bench hits its cache.
