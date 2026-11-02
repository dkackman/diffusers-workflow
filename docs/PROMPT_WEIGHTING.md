# Prompt Weighting

A1111-style prompt weighting sets the emphasis of individual words or phrases, and lifts the 77-token CLIP limit on the prompt (`dw/prompt_weighting.py`, after sd_embed). Flux-family pipelines only.

## Enabling

Set `prompt_weighting` to `true` in the pipeline configuration:

```json
{
    "pipeline": {
        "configuration": {
            "component_type": "FluxPipeline",
            "prompt_weighting": true
        },
        "from_pretrained_arguments": {
            "model_name": "black-forest-labs/FLUX.1-schnell",
            "torch_dtype": "torch.bfloat16"
        },
        "arguments": {
            "prompt": "a (photorealistic:1.4) portrait with (bright red hair:1.3) and [freckles]"
        }
    }
}
```

## Syntax

| Syntax | Effect | Example |
| ------ | ------ | ------- |
| `(word:1.5)` | Set weight to 1.5 | `(beautiful:1.5) landscape` |
| `(word)` | Multiply weight by 1.1 | `(beautiful) landscape` |
| `((word))` | Multiply by 1.1 twice (1.21) | `((beautiful)) landscape` |
| `[word]` | Reduce weight (divide by 1.1) | `forest with [clouds]` |
| `[[word]]` | Reduce twice (0.826) | `forest with [[clouds]]` |
| `\(` `\)` | Literal parentheses | `\(actual parens\)` |

Weights are multiplicative when nested: `(((word:1.3)))` = 1.3 x 1.1 x 1.1 = 1.573.

## How It Works

Before the pipeline call, a `prompt` string containing `(` or `[` is parsed into weighted tokens, run through the CLIP and T5 encoders with the weights applied, and replaced by `prompt_embeds` and `pooled_prompt_embeds`. A prompt with neither character passes through as a plain string, and so does a `prompt` that is not a string.

- `prompt_2` (Flux's T5 prompt, defaulting to `prompt`) is consumed and weighted separately when present.
- diffusers rejects a string `negative_prompt` beside `prompt_embeds`, so a `negative_prompt` is dropped once weighting applies, with only a debug log line.

## Supported Pipelines

`FluxPipeline`, `FluxImg2ImgPipeline`, `FluxInpaintPipeline` and `FluxControlNetPipeline` by name, and any other pipeline whose class name starts with `Flux` (`FluxKontextPipeline`, `FluxFillPipeline`, a subclass) that carries the same CLIP + T5 stack: `tokenizer`, `tokenizer_2`, `text_encoder`, `text_encoder_2`. On any other pipeline the prompt stays a plain string and a warning is logged.

## Requirements

- The pipeline's text encoders must be loaded, not set to `null`.
- Mutually exclusive with `remote_text_encoder`, which wins if both are set.

## Example

```bash
python -m dw.run workflows/templates/prompt-weighting.json \
    prompt="a (cinematic:1.5) shot of a (dragon:1.3) breathing [smoke] over a (medieval:0.8) castle"
```

See [prompt-weighting.json](../workflows/templates/prompt-weighting.json) (FLUX.1 schnell, 4 steps).
