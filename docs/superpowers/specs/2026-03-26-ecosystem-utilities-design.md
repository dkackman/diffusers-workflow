# Ecosystem Utilities Design Spec

Four new capabilities from the stable diffusion ecosystem: Marigold depth/normals examples, GroundingDINO+SAM2 segmentation task, RIFE frame interpolation task, and image metadata embedding.

## 1. Marigold Depth & Normals (Example Workflows Only)

Marigold ships as native diffusers pipelines (`MarigoldDepthPipeline`, `MarigoldNormalsPipeline`). No code changes needed — just example workflows.

### Files to Create

- `examples/MarigoldDepth.json`
- `examples/MarigoldNormals.json`

### MarigoldDepth.json

Two-step workflow:
1. `gather_images` — load an input image (URL or local path via variable)
2. Pipeline step using `MarigoldDepthPipeline` with `prs-eth/marigold-depth-lcm-v1-0` — outputs a depth map as a PIL Image

The depth output chains naturally into ControlNet steps via `previous_result:`.

### MarigoldNormals.json

Same structure using `MarigoldNormalsPipeline` with `prs-eth/marigold-normals-lcm-v1-0`.

### Dependencies

None beyond existing diffusers install.

---

## 2. GroundingDINO + SAM2 Segmentation Task

A new `segment` task command that takes an image + text prompt and returns a binary mask image.

### Files to Create/Modify

- `dw/tasks/segment.py` — core implementation
- `dw/tasks/task.py` — import and register command
- `examples/Segment.json` — basic segmentation example
- `examples/SegmentAndInpaint.json` — segment then inpaint with FluxFill

### Task Registration

```python
# In task.py
from .segment import segment_image

@register_command("segment")
def _handle_segment(task, arguments, previous_pipelines):
    """Segment objects in an image using text prompt"""
    logger.debug("Segmenting image")
    image = arguments.pop("image")
    prompt = arguments.pop("prompt")
    return segment_image(image, prompt, device=task.device, **arguments)
```

### Task Arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `image` | PIL Image | yes | — | Input image |
| `prompt` | string | yes | — | Text description of object(s) to segment |
| `model_name` | string | no | `"IDEA-Research/grounding-dino-base"` | GroundingDINO model |
| `sam_model_name` | string | no | `"facebook/sam2-hiera-large"` | SAM2 model |
| `threshold` | float | no | `0.3` | Detection confidence threshold |
| `invert` | bool | no | `false` | Invert the output mask |

### Returns

PIL Image — binary mask (white = detected object, black = background). When multiple objects match the prompt, masks are combined via union.

### Implementation: `dw/tasks/segment.py`

```python
def segment_image(image, prompt, device="cpu", **kwargs):
    """Segment objects matching text prompt, returning a binary mask."""
```

Flow:
1. Load GroundingDINO via `transformers.AutoProcessor` + `AutoModelForZeroShotObjectDetection`
2. Run detection with text prompt, filter by `threshold`
3. Extract bounding boxes
4. Load SAM2 via `transformers.AutoProcessor` + `AutoModelForMaskGeneration` (or `SamModel` + `SamProcessor`)
5. Load SAM2 via `Sam2Model.from_pretrained()` + `Sam2Processor.from_pretrained()` from `transformers`
6. Feed image + bounding boxes to SAM2
7. Combine all output masks (union) into single binary mask
8. Optionally invert
9. Return as PIL Image (mode "L", 0/255 values)

Lazy imports for `transformers` model classes (`AutoModelForZeroShotObjectDetection`, `AutoProcessor`, `Sam2Model`, `Sam2Processor`) — raises `ImportError` with install instructions if missing.

### Dependencies

- `transformers` (already likely installed)
- `torch` (already installed)

### Example: Segment.json

```json
{
    "id": "Segment",
    "variables": {
        "image_url": "https://example.com/photo.jpg",
        "prompt": "dog"
    },
    "steps": [
        {
            "name": "load_image",
            "task": {
                "command": "gather_images",
                "arguments": {
                    "urls": ["variable:image_url"]
                }
            },
            "result": {
                "content_type": "image/png",
                "save": false
            }
        },
        {
            "name": "segment",
            "task": {
                "command": "segment",
                "arguments": {
                    "image": "previous_result:load_image",
                    "prompt": "variable:prompt"
                }
            },
            "result": {
                "content_type": "image/png"
            }
        }
    ]
}
```

### Example: SegmentAndInpaint.json

Three-step workflow:
1. `gather_images` — load input image
2. `segment` — generate mask from text prompt
3. `FluxFillPipeline` step — inpaint using mask from step 2 and a new prompt

---

## 3. Frame Interpolation Task (RIFE)

A new `interpolate_frames` task that takes video frames and returns interpolated frames at higher frame count.

### Files to Create/Modify

- `dw/tasks/interpolate_frames.py` — core implementation
- `dw/tasks/task.py` — import and register command
- `examples/InterpolateFrames.json` — video gen + interpolation example

### Task Registration

```python
# In task.py
from .interpolate_frames import interpolate_frames

@register_command("interpolate_frames")
def _handle_interpolate_frames(task, arguments, previous_pipelines):
    """Interpolate video frames to increase frame rate"""
    logger.debug("Interpolating frames")
    video = arguments.pop("video")
    return interpolate_frames(video, device=task.device, **arguments)
```

### Task Arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `video` | list[PIL Image] | yes | — | Input video frames |
| `multiplier` | int | no | `2` | Frame count multiplier (2, 4, or 8) |
| `model_name` | string | no | see below | RIFE model weights (HF repo or local path) |

### Returns

List of PIL Images — the interpolated frame sequence. The caller should set `fps` in the result config to `original_fps * multiplier` to maintain correct playback speed.

### Implementation: `dw/tasks/interpolate_frames.py`

```python
def interpolate_frames(video, device="cpu", **kwargs):
    """Interpolate between video frames using RIFE."""
```

Flow:
1. Validate `multiplier` is 2, 4, or 8
2. Load RIFE model (PyTorch implementation for device compatibility)
3. For each consecutive frame pair, generate intermediate frame(s)
4. For 4x: run two passes of 2x interpolation
5. For 8x: run three passes of 2x interpolation
6. Return combined frame list

Uses a vendored PyTorch RIFE IFNet architecture with weights downloaded from HuggingFace Hub at first use. This ensures broad device compatibility across CUDA, MPS, and CPU without requiring ncnn or Vulkan. The IFNet model files are small (~30MB) and cached locally after first download. Lazy imports.

### Dependencies

- `torch` (already installed)
- RIFE model weights downloaded from HuggingFace Hub at first use

### Example: InterpolateFrames.json

```json
{
    "id": "InterpolateFrames",
    "variables": {
        "prompt": "A cat walking across a sunlit room",
        "multiplier": 2
    },
    "steps": [
        {
            "name": "generate_video",
            "pipeline": {
                "configuration": {
                    "offload": "sequential",
                    "component_type": "MochiPipeline",
                    "vae": { "enable_tiling": true, "enable_slicing": true }
                },
                "from_pretrained_arguments": {
                    "model_name": "genmo/mochi-1-preview",
                    "variant": "bf16",
                    "torch_dtype": "torch.bfloat16"
                },
                "arguments": {
                    "prompt": "variable:prompt",
                    "num_frames": 85
                }
            },
            "result": {
                "content_type": "video/mp4",
                "save": false,
                "fps": 30
            }
        },
        {
            "name": "interpolate",
            "task": {
                "command": "interpolate_frames",
                "arguments": {
                    "video": "previous_result:generate_video",
                    "multiplier": "variable:multiplier"
                }
            },
            "result": {
                "content_type": "video/mp4",
                "fps": 60
            }
        }
    ]
}
```

---

## 4. Image Metadata Embedding

Opt-in embedding of generation parameters into saved image files via the `result` config.

### Files to Modify

- `dw/workflow_schema.json` — add `embed_metadata` property to result definition
- `dw/result.py` — embed metadata when saving images
- `dw/step.py` — pass step metadata to Result when `embed_metadata` is true
- `examples/MetadataEmbed.json` — example workflow

### Schema Change

Add to the `result` definition properties in `workflow_schema.json`:

```json
"embed_metadata": {
    "description": "Whether to embed generation parameters as metadata in saved images (PNG info chunks or EXIF)",
    "type": "boolean",
    "default": false
}
```

### What Gets Embedded

A JSON object stored as a text chunk containing:
- `workflow_id` — the workflow's ID
- `step_name` — the step that produced the image
- `model_name` — from `from_pretrained_arguments` (if pipeline step)
- `arguments` — the step's inference arguments (prompt, negative_prompt, guidance_scale, num_inference_steps, seed, etc.)
- `task_command` — if a task step, the command name

### Implementation Changes

**`step.py`:** When `embed_metadata` is true in the step's result definition, collect metadata and pass it to `Result` via a new `set_metadata(metadata_dict)` method. Metadata is collected from:

- `step_definition["name"]` — the step name
- The workflow ID (passed through from `workflow.py`)
- For pipeline steps: `from_pretrained_arguments.model_name`, and the inference `arguments` dict (prompt, negative_prompt, guidance_scale, num_inference_steps, seed, width, height, etc.)
- For task steps: `task.command` and the task `arguments` dict

**`result.py`:**
- Add `self.metadata = None` to `Result.__init__`
- Add `set_metadata(self, metadata)` method
- In `save_artifact`, when `content_type` starts with `image/` and `self.metadata` is not None:
  - **PNG:** Use `PIL.PngImagePlugin.PngInfo` to add a `parameters` text chunk containing the JSON metadata. Pass `pnginfo` to `image.save()`.
  - **JPEG/WebP:** Use `piexif` (lazy import, optional) to write metadata as EXIF UserComment. If `piexif` is not installed, log a warning and save without metadata.

### Format

PNG text chunk key: `parameters`
Value: JSON string of the metadata dict.

This is compatible with tools that read A1111-style PNG info, though the structure is JSON rather than A1111's custom text format.

### Example: MetadataEmbed.json

```json
{
    "id": "MetadataEmbed",
    "variables": {
        "prompt": "A serene mountain landscape at golden hour",
        "steps": 25
    },
    "steps": [
        {
            "name": "generate",
            "pipeline": {
                "configuration": {
                    "offload": "model",
                    "component_type": "FluxPipeline"
                },
                "from_pretrained_arguments": {
                    "model_name": "black-forest-labs/FLUX.1-dev",
                    "torch_dtype": "torch.bfloat16"
                },
                "arguments": {
                    "prompt": "variable:prompt",
                    "num_inference_steps": "variable:steps",
                    "guidance_scale": 3.5
                }
            },
            "result": {
                "content_type": "image/png",
                "embed_metadata": true
            }
        }
    ]
}
```

### Dependencies

- `piexif` — optional, only needed for JPEG/WebP metadata. PNG works with PIL alone.

---

## Summary of Changes

| Feature | New Files | Modified Files | New Dependencies |
|---------|-----------|----------------|------------------|
| Marigold examples | 2 example JSONs | none | none |
| Segment task | `dw/tasks/segment.py`, 2 example JSONs | `dw/tasks/task.py` | `transformers` (likely already present) |
| Frame interpolation | `dw/tasks/interpolate_frames.py`, 1 example JSON | `dw/tasks/task.py` | RIFE model weights (auto-download) |
| Metadata embedding | 1 example JSON | `dw/result.py`, `dw/step.py`, `dw/workflow_schema.json` | `piexif` (optional, JPEG only) |
