# Prompt Weighting

Enable A1111-style prompt weighting to control emphasis on individual words or phrases. Also supports prompts longer than the standard 77-token CLIP limit.

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

When enabled, prompts containing weighting syntax are intercepted before pipeline execution:

1. The prompt string is parsed for weight tokens
2. Tokens are run through the pipeline's text encoders with per-token weights applied
3. The resulting embedding tensors replace the `prompt` string argument
4. The pipeline receives `prompt_embeds` and `pooled_prompt_embeds` instead

Prompts without any weighting syntax (`(`, `[`) pass through unchanged as plain strings.

## Supported Pipelines

Currently supports Flux-based pipelines:

- FluxPipeline
- FluxImg2ImgPipeline
- FluxInpaintPipeline
- FluxControlNetPipeline

## Requirements

- The pipeline's text encoders must be loaded (not set to `null`)
- Cannot be used with `remote_text_encoder` (mutually exclusive)

## Example

```bash
python -m dw.run examples/FluxSchnellWeighted.json \
    prompt="a (cinematic:1.5) shot of a (dragon:1.3) breathing [smoke] over a (medieval:0.8) castle"
```

See [FluxSchnellWeighted.json](../examples/FluxSchnellWeighted.json).
