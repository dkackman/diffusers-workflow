# Kandinsky 6.0 workflows

Text- and image-to-video with a generated soundtrack (44.1 kHz, lip-sync included),
using Kandinsky Lab's [Kandinsky 6.0](https://huggingface.co/collections/kandinskylab/kandinsky-60-diffusers)
(MIT). The family is a 3B Lite and a 29B Pro, each as a distilled, a main and a
pretrain checkpoint, plus a separate tiled super-resolution pipeline. Every
checkpoint is trained on five-second clips: 121 frames at 24 fps, natively 480x864.

Settings come from the model cards, Kandinsky Lab's
[kandinsky-6](https://github.com/kandinskylab/kandinsky-6) repository and the
diffusers pipeline source (`diffusers.pipelines.kandinsky6`, merged to diffusers
`main` on 2026-10-06 and not yet in a release). These templates need diffusers
installed from git. Audited against those sources on 2026-10-06
([the audit](../../../docs/proposals/audits/2026-10-06-kandinsky-6-audit.md)).

The prompt format is the vendor's own: the prompt beautifier's system prompts in the
vendor repository (`comfyui/kandinsky6/beautifier_prompts/t2av_system.txt` and
`i2av_system.txt`) describe what the model was trained on. That is one paragraph of
180-300 words, followed by exactly one `<AUDCAP>…<ENDAUDCAP>` audio sentence, with
any speech in `<S>…<E>`. Diffusers carries a shorter version of the same
instructions as `_T2VA_EXPANSION_INSTRUCTION` and `_I2VA_EXPANSION_INSTRUCTION`,
which the pipeline runs on its own text encoder when `expand_prompts` is true. The
stored prompts under `prompts/kandinsky6/` are written to the vendor's format.

Read them in this order and each introduces one new idea on top of the last.

## The basics

| Example | What it introduces |
| ------- | ------------------ |
| [text-to-video.json](text-to-video.json) | The baseline: text to video plus soundtrack on the 3B Lite-distill checkpoint. Ten PiFlow steps at guidance 1.0, and both are part of the checkpoint: the scheduler refuses custom sigmas. Width and height are set explicitly, because the pipeline's defaults (512x768 at guidance 5.0) are not the model's. The repo is pinned to a revision because the vendor has re-keyed the weights as diffusers' module names changed |
| [image-to-video.json](image-to-video.json) | A supplied still becomes the first frame, and the prompt describes what happens next. The pipeline resizes and center-crops the still to `width` x `height`, so a portrait still wants `480x864` |

## Apple Silicon: parked

These templates do not run on MPS yet (measured on a 64GB Mac, torch 2.14.1,
2026-10-06). The ten denoising steps are fine there (about 54 s each), but the
HunyuanVideo VAE's decode is not. Tiled or not, MPS keeps about 2 GiB of driver
memory per decode tile, and neither `torch.mps.empty_cache()` nor
`PYTORCH_MPS_LOW_WATERMARK_RATIO` frees it. Live tensors stay under 1 GiB, yet
3 latent frames (9 video frames) hold about 30 GiB in bf16 or fp16, and about
58 GiB in fp32. The full 31 latent frames get the worker killed. The audio
decode is about 1 GiB. A CPU decode of the same 3 latent frames had not finished
after 10 minutes.

## Not here yet

These follow-ups are dated 2026-10-06 and are in the audit's reading order:
- Prompt expansion (`expand_prompts`) and a speech/lip-sync example.
- Lite at 50 steps with guidance 5.0.
- Super-resolution: `Kandinsky6SRPipeline` needs compiled flex attention, so it is CUDA-only until it has been measured elsewhere.
- Pro-distill and Pro: the transformer is 60 GB in bf16.
- Sizing an image-to-video clip from the still's own aspect ratio, which the vendor's tools do.
