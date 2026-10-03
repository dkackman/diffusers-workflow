# LoRA Support

LoRA (Low-Rank Adaptation) models apply lightweight style or subject modifications to a base model. Add one or more LoRAs to any pipeline step.

## Basic Usage

```json
{
    "pipeline": {
        "configuration": { "component_type": "FluxPipeline" },
        "from_pretrained_arguments": {
            "model_name": "black-forest-labs/FLUX.1-dev",
            "torch_dtype": "torch.bfloat16"
        },
        "loras": [
            {
                "model_name": "XLabs-AI/flux-RealismLora"
            }
        ],
        "arguments": {
            "prompt": "a photorealistic landscape"
        }
    }
}
```

## LoRA Properties

```json
"loras": [
    {
        "model_name": "user/lora-repo",
        "weight_name": "specific_weights.safetensors",
        "subfolder": "lora_subfolder",
        "adapter_name": "my_adapter",
        "scale": 0.8
    }
]
```

| Property | Required | Description |
| -------- | -------- | ----------- |
| `model_name` | Yes | HuggingFace Hub repo ID |
| `weight_name` | No | Specific weight file in the repo |
| `subfolder` | No | Subfolder within the repo |
| `adapter_name` | No | Named identifier for the adapter. Defaults to the LoRA's position in the list (`"0"`, `"1"`, ...) if omitted |
| `scale` | No | Blend strength (default: 1.0). Lower = less effect |

Any other property (e.g. `revision`) is forwarded as-is to the underlying `load_lora_weights()` call.

## Multiple LoRAs

Stack multiple LoRAs. They are blended via weighted adapter composition:

```json
"loras": [
    {
        "model_name": "XLabs-AI/flux-RealismLora",
        "adapter_name": "realism",
        "scale": 0.7
    },
    {
        "model_name": "user/style-lora",
        "adapter_name": "style",
        "scale": 0.5
    }
]
```

## LoRA with Quantization

LoRAs work with quantized models:

```json
{
    "pipeline": {
        "transformer": {
            "configuration": { "component_type": "SD3Transformer2DModel" },
            "quantization_config": {
                "configuration": { "config_type": "BitsAndBytesConfig" },
                "arguments": { "load_in_4bit": true, "bnb_4bit_quant_type": "{nf4}" }
            },
            "from_pretrained_arguments": {
                "model_name": "stabilityai/stable-diffusion-3.5-large",
                "subfolder": "transformer",
                "torch_dtype": "torch.bfloat16"
            }
        },
        "configuration": { "component_type": "StableDiffusion3Pipeline" },
        "from_pretrained_arguments": {
            "model_name": "stabilityai/stable-diffusion-3.5-large",
            "torch_dtype": "torch.bfloat16"
        },
        "loras": [
            {
                "model_name": "crystalwizard/cubic-abstract-1",
                "weight_name": "cubic-abstract-lora.safetensors"
            }
        ],
        "arguments": { "prompt": "cubart a leaf" }
    }
}
```

## Variable LoRA

Make the LoRA configurable via workflow variables:

```json
{
    "variables": {
        "lora": "XLabs-AI/flux-RealismLora"
    },
    "steps": [{
        "pipeline": {
            "loras": [{ "model_name": "variable:lora" }],
            "arguments": { "prompt": "variable:prompt" }
        }
    }]
}
```

```bash
python -m dw.run workflow.json lora="other-user/other-lora"
```

## Examples

- [lora.json](../workflows/templates/lora.json) — Flux with realism LoRA and variables
- [lora.json](../workflows/templates/lora.json) — SD 3.5 with yarn art style LoRA
- [lora.json](../workflows/templates/lora.json) — A LoRA adapter on FLUX.1 dev, with the SD 3.5 variant in its description

## LoRA catalog

The catalog records LoRAs that were tried on a base model: `proven`, `trial`,
or `rejected` with the reason. It is a library like the prompt library - the
server's own `loras/` at the workspace root (writable, shared by every
workspace), then the `loras/` an examples tree brings (read-only). One JSON
file per LoRA, under a family folder: `loras/qwen-image/voxel-style.json`.

```json
{
  "model_name": "fal/MiniMax-H3-Realism-People-LoRA",
  "weight_name": "h3-realism-people-t2v-i2v-r2v.safetensors",
  "revision": "<sha>",
  "base_models": ["MiniMaxAI/MiniMax-H3"],
  "workflow": "t2va",
  "description": "Realistic people and faces",
  "use_when": "People, crowds or faces should look photoreal",
  "scale": { "default": 0.7, "range": [0.7, 0.7] },
  "status": "proven",
  "evidence": [{ "issue": 585, "note": "Best arm of the 2026-10-03 eval" }]
}
```

`model_name`, `weight_name`, `revision` and `scale.default` drop straight into
a step's `loras` entry. `base_models` holds exact repo ids, and matching is
exact - an adapter on the wrong base usually loads and is quietly worse.
`workflow` constrains a MiniMax-H3 entry to one partition (`t2va`, `fl2va`,
`ref2va`). `proven` and `rejected` entries need `evidence`; a rejected
entry's first note is why. The schema is `GET /api/lora-schema`.

Promotion: a trial that worked is saved with `save_lora` (or
`PUT /api/loras/{name}`) as `proven`, its job id in `evidence`.

## Finding LoRAs on the Hub

`recommend_loras(model, query)` (`GET /api/loras/recommend`) is the only
place dw searches the Hub, and only when called. It returns the catalog's
entries for the model first, ranked against the query, then Hub adapters
whose card declares that exact base (`base_model:adapter:<repo>`), most
downloaded first. Nothing is downloaded; the weight file's header is read to
check its layout. Hub rows are candidates to trial, never recommendations,
and carry `warnings`:

| Warning | Meaning |
| --- | --- |
| `will_not_load` | Full-weight `.diff` keys diffusers' converters refuse |
| `unknown_format` | The tensor names match no known LoRA layout |
| `header_unreadable` | The header range read failed; format unchecked |
| `multiple_weights` | Several `.safetensors`; pick one from `weights` |
| `gated` | Needs the server's HF token to have accepted the gate |
| `no_license` | The card declares none |
| `stale` | Last changed before its base was - trained on an older revision |

A repo holding only pickle `.bin` weights is never offered. A repo the
catalog marks `rejected` comes back as `rejected` with the reason. When the
Hub is unreachable the catalog rows still come back, with `hub_error`.

Limits: the search runs the typed query plus at most 4 of its words;
`query` is capped at 200 characters and `limit` is 1-25. One Hub search runs
at a time per server, and a concurrent call gets catalog rows plus
`hub_error` saying a search is already running. An empty `hub` with
`hub_error` set means the search failed, not that no adapters exist.
