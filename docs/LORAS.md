# LoRA Support

A LoRA (Low-Rank Adaptation) adapter applies a style, subject or step-distillation change to a base model. A pipeline step takes a `loras` list, a sibling of `configuration`, on any pipeline diffusers can load adapters into (`load_lora_weights()`; `dw/pipeline_processors/adapters.py`).

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
        "scale": 0.8,
        "alpha": 16
    }
]
```

| Property | Required | Description |
| -------- | -------- | ----------- |
| `model_name` | Yes | Hugging Face Hub repo id. `null` switches the entry off: it is not loaded and the step runs without it, with a `lora_disabled` warning at validate and run. A template's `loras` list is fixed JSON, so this is how a caller drops its adapter (MiniMax-H3: pass `lora_model_name=null`) |
| `weight_name` | No | Specific weight file in the repo |
| `subfolder` | No | Subfolder within the repo |
| `adapter_name` | No | The adapter's name in `set_adapters()`. Defaults to the entry's position in the list (`"0"`, `"1"`, ...), counting switched-off entries |
| `scale` | No | Adapter weight, default 1.0. Lower = less effect. Takes a `variable:` reference |
| `alpha` | No | Network alpha, overriding the checkpoint's. peft scales an adapter by `scale * alpha / rank`, taking alpha from the file (a per-module `.alpha` tensor, else a `__metadata__` alpha, else the rank). Positive; state it only where the file records the wrong figure, after reading its header. Fails if no loaded layer carries the adapter. The run logs the effective `alpha / rank` |

Any other property (e.g. `revision`) is passed to `load_lora_weights()` unchanged.

`scale` and `alpha` are set in place on the loaded adapter, so a server run that changes
only them reuses the warm pipeline; changing `model_name` or `weight_name` reloads.

A step-distillation (turbo) LoRA sets the step count and sigma shift the step runs at.
Switching one off, raise both to the base model's schedule, or the step runs undertrained.

## Multiple LoRAs

Several entries load as separate adapters, each at its own `scale`, active together:

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

A LoRA loads onto a quantized component as onto any other:

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

A `variable:` reference makes the adapter an argument:

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

- [lora.json](../workflows/templates/lora.json): FLUX.1 dev with `XLabs-AI/flux-RealismLora` as the `lora` variable; its description names an SD 3.5 variant (`linoyts/yart_art_sd3-5_lora`)
- [lora-styles.json](../workflows/templates/lora-styles.json): one prompt through ten FLUX.1 in-context LoRA styles, one image per adapter

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
`workflow` constrains a MiniMax-H3 entry to a partition (`t2va`, `fl2va`,
`ref2va`) - a string, or a list of them for an adapter that covers several
(the turbo keyframe files are `["t2va", "fl2va"]`). `proven` and `rejected` entries need
`evidence`; a rejected entry's first note is why. The schema is `GET /api/lora-schema`
(`dw/lora_catalog_schema.json`).

The server's validate enforces that `workflow` (#870): a step whose workflow
sits in the other partition is refused when the entry is `proven`, and only
warned when it is `trial` or `rejected`. Without a covering entry the file
name decides (`ref2v` / `fl2v`), as in `dw/adapter_compatibility.py`.

Promotion: a trial that worked is saved with `save_lora` (or
`PUT /api/loras/{name}`) as `proven`, its job id in `evidence`. Promoting
it is also what makes its `workflow` refuse a mismatched step.

## Finding LoRAs on the Hub

`recommend_loras(model, query)` (`GET /api/loras/recommend`) is the only
place dw searches the Hub, and only when called. It returns the catalog's
entries for the model first, ranked against the query, then Hub adapters
whose card declares that exact base (`base_model:adapter:<repo>`), most
downloaded first. Nothing is downloaded: for a single-weight repo the weight
file's header is read to check its layout. Hub rows are candidates to trial, never
recommendations, and carry `warnings`:

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
catalog marks `rejected` comes back in `hub` as `rejected` with the reason
only when the Hub search turns it up; rejected entries are left out of
`catalog` (`list_loras` still shows them). When the
Hub is unreachable the catalog rows still come back, with `hub_error`.

Limits: the search runs the typed query plus at most 4 of its words;
`query` is capped at 200 characters and `limit` is 1-25. One Hub search runs
at a time per server, and a concurrent call gets catalog rows plus
`hub_error` saying a search is already running. An empty `hub` with
`hub_error` set means the search failed, not that no adapters exist.
