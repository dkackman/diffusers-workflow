# Inference Acceleration

Speed up generation by caching intermediate computations and skipping redundant transformer steps. Two systems are available: diffusers built-in caching and TeaCache.

## Diffusers Built-in Cache

Applied at pipeline load time via the `cache` configuration. Hooks auto-reset between runs.

### FirstBlockCache

Simplest and broadest support. Compares first-block residuals to decide whether to skip remaining blocks.

```json
"configuration": {
    "component_type": "FluxPipeline",
    "cache": {
        "type": "first_block",
        "threshold": 0.05
    }
}
```

Higher threshold = more speedup, more quality loss. Start with `0.05` and increase to taste.

**Example:** [FluxDevFirstBlockCache.json](../examples/FluxDevFirstBlockCache.json)

### MagCache

Magnitude-based caching with error accumulation. Requires `num_inference_steps` to match the pipeline arguments:

```json
"cache": {
    "type": "mag",
    "threshold": 0.06,
    "num_inference_steps": 28,
    "max_skip_steps": 3,
    "retention_ratio": 0.2
}
```

| Property | Default | Description |
| -------- | ------- | ----------- |
| `threshold` | 0.06 | Accumulated error threshold for skipping |
| `num_inference_steps` | required | Must match pipeline arguments |
| `max_skip_steps` | 3 | Max consecutive steps to skip |
| `retention_ratio` | 0.2 | Fraction of initial steps where skipping is disabled |

### TaylorSeerCache

Taylor series approximation of cached outputs:

```json
"cache": {
    "type": "taylorseer",
    "cache_interval": 5,
    "max_order": 1
}
```

| Property | Default | Description |
| -------- | ------- | ----------- |
| `cache_interval` | 5 | Full computation every N steps |
| `max_order` | 1 | Taylor series order (higher = better approximation, more memory) |

### FasterCache

Experimental, video-oriented. Uses FFT frequency decomposition:

```json
"cache": {
    "type": "faster"
}
```

Best for video models like CogVideoX. No additional parameters needed for basic use.

## TeaCache

Training-free acceleration that monkey-patches the transformer's forward function. Uses polynomial-rescaled L1 distance to determine when to skip computation.

```json
"configuration": {
    "component_type": "FluxPipeline",
    "teacache": {
        "rel_l1_thresh": 0.6
    }
}
```

TeaCache requires `num_inference_steps` in the pipeline arguments — it needs to know the total step count.

### Configuration

| Property | Description |
| -------- | ----------- |
| `rel_l1_thresh` | Cache threshold. Model-specific defaults apply if omitted. |
| `coefficients` | Array of 5 polynomial coefficients. Override model defaults. |
| `variant` | Explicit model variant for multi-variant architectures. |

### Supported Models

Model coefficients and defaults are stored in [teacache_models.json](../dw/teacache_models.json). Currently implemented with a custom forward function:

- **Flux** (FluxTransformer2DModel) — thresholds: 0.25 (~1.5x), 0.4 (~1.8x), 0.6 (~2.0x), 0.8 (~2.25x)

Registry includes coefficients for Mochi, LTX-Video, CogVideoX, HunyuanVideo, Wan2.1, and Lumina2 (forward functions pending).

### Variants

Some models have multiple variants with different coefficients:

```json
"teacache": {
    "rel_l1_thresh": 0.2,
    "variant": "cogvideox_2b"
}
```

**Example:** [FluxDevTeaCache.json](../examples/FluxDevTeaCache.json)

## Cache vs TeaCache

| | Diffusers Cache | TeaCache |
| --- | --- | --- |
| Setup | Built into diffusers | Custom forward functions |
| Model support | Any transformer with CacheMixin | Requires per-model implementation |
| Maintenance | Maintained by HuggingFace | Maintained in this project |
| Configuration | Set once at load time | Applied per-execution via context manager |
| Approach | Various algorithms (block, magnitude, Taylor) | Polynomial-rescaled L1 distance |

They are **mutually exclusive** — use one or the other, not both.

For most cases, start with `first_block` cache. Use TeaCache when you need fine-tuned control over Flux acceleration thresholds.
