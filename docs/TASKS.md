# Task Commands

Tasks are utility operations that run outside of pipeline inference. Use them for image preprocessing, data gathering, and other non-model operations.

```json
{
    "name": "step_name",
    "task": {
        "command": "command_name",
        "arguments": { ... }
    },
    "result": { "content_type": "image/jpeg" }
}
```

## Image Processing

### ControlNet Preprocessors

Generate control images for ControlNet pipelines:

| Command | Description |
| ------- | ----------- |
| `canny` | Canny edge detection |
| `canny_cv` | OpenCV Canny (alternative) |
| `depth` | Depth estimation (DPT) |
| `midas` | Monocular depth (MiDaS) |
| `zoe` | Zoe depth estimation |
| `zoe_depth` | Zoe depth with colorization |
| `leres` | Relative depth (LeReS) |
| `normal_bae` | Surface normal estimation |
| `openpose` | Pose estimation |
| `dw_pose` | DW pose estimation |
| `mlsd` | Line segment detection |
| `lineart` | Line art extraction |
| `lineart_standard` | Standard line art |
| `hed` | HED edge detection |
| `scribble` | Scribble-style edges |
| `pidi` | Boundary detection |
| `shuffle` | Content-preserving shuffle |
| `teed` | TEED edge detection |
| `anyline` | Anyline edge detection |
| `sam` | Segment Anything |
| `segmentation` | Semantic segmentation |
| `depth_estimator` | Depth hint generation |
| `depth_estimator_tensor` | Depth hint as tensor |

All accept an `image` argument with processing parameters:

```json
{
    "task": {
        "command": "canny",
        "arguments": {
            "image": {
                "location": "https://example.com/photo.jpg",
                "low_threshold": 50,
                "high_threshold": 200,
                "detect_resolution": 1024,
                "image_resolution": 1024
            }
        }
    }
}
```

### Image Manipulation

| Command | Description | Extra Arguments |
| ------- | ----------- | --------------- |
| `remove_background` | Remove image background | |
| `resize_center_crop` | Resize with center crop | `width`, `height` |
| `resize_resample` | Resample to nearest 64px multiple | |
| `resize_rescale` | Resize to exact dimensions | `width`, `height` |
| `resize_bucket` | Snap to closest model-native aspect ratio | `resolution`, `ratios`, `alignment` |
| `crop_square` | Center crop to square | |
| `add_border_and_mask` | Add border with alpha mask | |
| `add_border_and_mask_with_size` | Border with specific dimensions | `width`, `height` |
| `get_image_size` | Return `{width, height}` dict | |

### Aspect Ratio Bucketing

The `resize_bucket` command snaps an image to the closest model-native aspect ratio, then resizes with 64-pixel alignment. This avoids distortion and ensures the model generates at a resolution it was trained on.

```json
{
    "task": {
        "command": "resize_bucket",
        "arguments": {
            "image": "previous_result:input_image",
            "resolution": 1024
        }
    },
    "result": { "content_type": "image/png" }
}
```

| Argument | Required | Description |
| -------- | -------- | ----------- |
| `resolution` | No | Target short-side size in pixels (default: 1024) |
| `ratios` | No | Custom list of `[w, h]` ratio pairs (default: standard SDXL/Flux ratios) |
| `alignment` | No | Round dimensions to this multiple (default: 64) |

**Default ratios:** 1:1, 4:3, 3:4, 3:2, 2:3, 16:9, 9:16, 21:9, 9:21

For example, a 1600x900 photo (16:9) at resolution 1024 becomes 1792x1024. A 800x600 photo (4:3) becomes 1344x1024.

## Video Processing

| Command | Description | Extra Arguments |
| ------- | ----------- | --------------- |
| `get_first_frame` | Extract first video frame | |
| `get_last_frame` | Extract last video frame | |
| `get_frame` | Extract frame at index | `frame_index` |

## Data Gathering

### gather_images

Load images from URLs and/or file glob patterns:

```json
{
    "task": {
        "command": "gather_images",
        "arguments": {
            "urls": ["https://example.com/a.jpg", "https://example.com/b.jpg"],
            "glob": "./images/*.jpg"
        }
    }
}
```

Returns a list of images that can be referenced by later steps with `previous_result:`.

### gather_videos

Same as `gather_images` but for video files.

### gather_inputs

Pass through arguments directly. Useful for organizing data flow.

## Image Upscaling

Upscale images using spandrel-compatible super-resolution models (ESRGAN, SwinIR, HAT, DAT, and 40+ other architectures). Models are auto-detected from weight files.

```json
{
    "task": {
        "command": "upscale",
        "arguments": {
            "image": "previous_result:generate",
            "model_name": "Kim2091/UltraSharp",
            "filename": "4x-UltraSharp.pth"
        }
    }
}
```

| Argument | Required | Description |
| -------- | -------- | ----------- |
| `image` | Yes | PIL Image or `previous_result:` reference |
| `model_name` | Yes | HuggingFace repo ID or local file path |
| `filename` | No | Specific weight file in a HF repo (auto-detected if only one) |
| `tile_size` | No | Tile size for large images (default: 512) |
| `tile_overlap` | No | Overlap between tiles in pixels (default: 32) |

Large images are automatically tiled to avoid GPU memory issues. Models can be loaded from HuggingFace Hub repos or local `.pth`/`.safetensors` files.

**Example:** [SpandrelUpscale.json](../examples/SpandrelUpscale.json) — Generate at 512px, then 4x upscale to 2048px.

## Face Restoration

Restore and enhance faces in images using spandrel-compatible face restoration models (GFPGAN, CodeFormer, RestoreFormer). Uses facexlib for face detection and alignment, then runs each detected face through the restoration model.

```json
{
    "task": {
        "command": "restore_faces",
        "arguments": {
            "image": "previous_result:generate",
            "model_name": "leonelhs/gfpgan",
            "filename": "GFPGANv1.4.pth"
        }
    }
}
```

| Argument | Required | Description |
| -------- | -------- | ----------- |
| `image` | Yes | PIL Image or `previous_result:` reference |
| `model_name` | Yes | HuggingFace repo ID or local file path |
| `filename` | No | Specific weight file in a HF repo (auto-detected if only one) |
| `upscale_factor` | No | Background upscale factor (default: 1, no upscaling) |
| `face_size` | No | Cropped face size in pixels (default: 512) |
| `use_parse` | No | Use face parsing for better blending (default: true) |
| `only_center_face` | No | Only restore the largest/center face (default: false) |
| `detection_resize` | No | Resize shorter side for detection speed (default: 640) |
| `eye_dist_threshold` | No | Skip faces with eye distance below this (default: 5) |
| `upsample_img` | No | Pre-upscaled background image (e.g., from a prior upscale step) |

Models are loaded via spandrel, so any `.pth`/`.safetensors` face restoration weights work. CodeFormer requires `pip install spandrel-extra-arches` (non-commercial license).

**Example:** [FaceRestore.json](../examples/FaceRestore.json) — Generate a portrait, then restore faces with GFPGAN v1.4.

### Combining with Upscaling

You can chain upscaling and face restoration. Generate first, upscale the background, then paste restored faces onto the upscaled image:

```json
{
    "steps": [
        {
            "name": "generate",
            "pipeline": { "..." : "..." },
            "result": { "content_type": "image/jpeg" }
        },
        {
            "name": "upscale",
            "task": {
                "command": "upscale",
                "arguments": {
                    "image": "previous_result:generate",
                    "model_name": "Kim2091/UltraSharp",
                    "filename": "4x-UltraSharp.pth"
                }
            },
            "result": { "content_type": "image/jpeg" }
        },
        {
            "name": "restore",
            "task": {
                "command": "restore_faces",
                "arguments": {
                    "image": "previous_result:generate",
                    "model_name": "leonelhs/gfpgan",
                    "filename": "GFPGANv1.4.pth",
                    "upscale_factor": 4,
                    "upsample_img": "previous_result:upscale"
                }
            },
            "result": { "content_type": "image/jpeg" }
        }
    ]
}
```

This gives the best results: the super-resolution model handles background detail while the face model handles facial features, composited together at the upscaled resolution.

## Object Segmentation

Detect and segment objects using text prompts via GroundingDINO + SAM2. Returns a binary mask image suitable for inpainting workflows.

```json
{
    "task": {
        "command": "segment",
        "arguments": {
            "image": "previous_result:input_image",
            "prompt": "dog"
        }
    },
    "result": { "content_type": "image/png" }
}
```

| Argument | Required | Description |
| -------- | -------- | ----------- |
| `image` | Yes | PIL Image or `previous_result:` reference |
| `prompt` | Yes | Text description of object(s) to detect (e.g., "dog", "red car") |
| `model_name` | No | GroundingDINO model ID (default: `IDEA-Research/grounding-dino-base`) |
| `sam_model_name` | No | SAM2 model ID (default: `facebook/sam2-hiera-large`) |
| `threshold` | No | Detection confidence threshold (default: 0.3) |
| `invert` | No | Invert the output mask (default: false) |

Returns a grayscale PIL Image (mode "L") — white (255) for detected objects, black (0) for background. Use with inpainting pipelines like FluxFillPipeline.

**Examples:**

- [Segment.json](../examples/Segment.json) — Segment an object from an image
- [SegmentAndInpaint.json](../examples/SegmentAndInpaint.json) — Segment, then inpaint the masked region

## Image Captioning

Generate text captions from images using HuggingFace image-to-text models (BLIP, BLIP-2, ViT-GPT2, GIT, etc.).

```json
{
    "task": {
        "command": "image_to_text",
        "arguments": {
            "image": "previous_result:input_image"
        }
    },
    "result": { "content_type": "text/plain" }
}
```

| Argument | Required | Description |
| -------- | -------- | ----------- |
| `image` | Yes | PIL Image or `previous_result:` reference |
| `model_name` | No | HuggingFace model ID (default: `Salesforce/blip-image-captioning-base`) |
| `prompt` | No | Text prompt for conditional captioning (supported by BLIP-2, etc.) |
| `max_new_tokens` | No | Maximum tokens to generate (default: 50) |

Returns a caption string. Save as `text/plain` for `.txt` output, or pass to a downstream step via `previous_result:` as a prompt for image generation.

For Florence-2's advanced task-token captioning (detailed captions, object detection, OCR), use the built-in `describe_image` workflow instead:

```json
{
    "name": "caption",
    "workflow": {
        "path": "builtin:describe_image.json",
        "arguments": { "image": "previous_result:input_image" }
    },
    "result": { "content_type": "text/plain" }
}
```

**Examples:**

- [ImageToText.json](../examples/ImageToText.json) — Basic BLIP captioning, saves as `.txt`
- [ImageToTextBlip2.json](../examples/ImageToTextBlip2.json) — BLIP-2 with conditional prompt
- [CaptionToImage.json](../examples/CaptionToImage.json) — Caption an image, then regenerate with Flux

## Text Generation / Prompt Expansion

Generate or expand text using a local language model. Useful for expanding short prompts into detailed image generation prompts, rewriting text, or other text-to-text tasks.

```json
{
    "task": {
        "command": "text_generation",
        "arguments": {
            "prompt": "a cat on a windowsill",
            "system_prompt": "You are a helpful AI assistant that creates detailed prompts for text to image generative AI. When supplied input generate only the prompt, no other text."
        }
    },
    "result": { "content_type": "text/plain" }
}
```

| Argument | Required | Description |
| -------- | -------- | ----------- |
| `prompt` | Yes | The user message or short prompt to expand/transform |
| `system_prompt` | No | System instruction for the model (e.g., "expand this into a detailed image prompt") |
| `model_name` | No | HuggingFace model ID (default: `Qwen/Qwen2.5-1.5B-Instruct`) |
| `max_new_tokens` | No | Maximum tokens to generate (default: 500) |

Returns a text string. Save as `text/plain` for `.txt` output, or pass to a downstream step via `previous_result:` as a prompt for image generation.

Any HuggingFace chat model works — Qwen2.5, Llama 3.2, Phi-3.5, etc. The default (Qwen2.5-1.5B-Instruct) is small enough to run alongside diffusion models.

There is also a built-in `augment_prompt` workflow (`builtin:augment_prompt.json`) that does the same thing using a 3-step pipeline approach with Phi-3.5-mini. The `text_generation` task is the simpler single-step alternative.

**Examples:**

- [ExpandPrompt.json](../examples/ExpandPrompt.json) — Expand a short prompt and save as `.txt`
- [ExpandAndGenerate.json](../examples/ExpandAndGenerate.json) — Expand prompt, then generate with Flux

## Frame Interpolation

Increase video frame rate using RIFE (Real-Time Intermediate Flow Estimation). Takes a list of video frames and inserts intermediate frames between each pair.

```json
{
    "task": {
        "command": "interpolate_frames",
        "arguments": {
            "video": "previous_result:generate_video",
            "multiplier": 2
        }
    },
    "result": { "content_type": "video/mp4", "fps": 60 }
}
```

| Argument | Required | Description |
| -------- | -------- | ----------- |
| `video` | Yes | List of PIL Images (video frames) or `previous_result:` reference |
| `multiplier` | No | Frame count multiplier: 2, 4, or 8 (default: 2) |
| `model_name` | No | HuggingFace repo with RIFE weights (default: `styler00dollar/RIFE-v4.6`) |

Uses vendored IFNet v4.6 architecture. Weights are downloaded from HuggingFace Hub on first use.

**Example:** [InterpolateFrames.json](../examples/InterpolateFrames.json) — Generate video with Mochi, then 2x interpolate from 30fps to 60fps.

## Metadata Embedding

Embed generation parameters in saved images. Enable by setting `embed_metadata: true` in a step's result configuration:

```json
{
    "result": {
        "content_type": "image/png",
        "embed_metadata": true
    }
}
```

| Format | Storage | Notes |
| ------ | ------- | ----- |
| PNG | Text chunk (`parameters` key) | Always available |
| JPEG/WebP | EXIF UserComment | Requires `pip install piexif` |

Metadata includes step name, model name, and generation arguments (prompt, steps, guidance scale, etc.) as JSON.

**Example:** [MetadataEmbed.json](../examples/MetadataEmbed.json) — Generate with Flux and embed parameters in PNG.

## QR Code Generation

```json
{
    "task": {
        "command": "qr_code",
        "arguments": {
            "qr_code_contents": "https://example.com"
        }
    }
}
```

## Multi-Step Example

Canny edge detection followed by ControlNet generation:

```json
{
    "steps": [
        {
            "name": "edges",
            "task": {
                "command": "canny",
                "arguments": {
                    "image": {
                        "location": "photo.jpg",
                        "low_threshold": 50,
                        "high_threshold": 200
                    }
                }
            },
            "result": { "content_type": "image/jpeg" }
        },
        {
            "name": "generate",
            "pipeline": {
                "configuration": {
                    "component_type": "FluxControlPipeline",
                    "offload": "sequential"
                },
                "from_pretrained_arguments": {
                    "model_name": "black-forest-labs/FLUX.1-Canny-dev",
                    "torch_dtype": "torch.bfloat16"
                },
                "arguments": {
                    "control_image": "previous_result:edges",
                    "prompt": "a watercolor painting",
                    "num_inference_steps": 50
                }
            },
            "result": { "content_type": "image/jpeg" }
        }
    ]
}
```

## Examples

- [FluxCanny.json](../examples/FluxCanny.json) — Canny edge ControlNet
- [FluxDepth.json](../examples/FluxDepth.json) — Depth-guided generation
- [qr_code.json](../examples/qr_code.json) — QR code with artistic ControlNet
- [upscale.json](../examples/upscale.json) — Gather, resize, and diffusion upscale
- [SpandrelUpscale.json](../examples/SpandrelUpscale.json) — Generate + spandrel 4x upscale
- [FaceRestore.json](../examples/FaceRestore.json) — Generate portrait + GFPGAN face restoration
- [Segment.json](../examples/Segment.json) — Text-prompted object segmentation
- [SegmentAndInpaint.json](../examples/SegmentAndInpaint.json) — Segment + inpaint
- [ImageToText.json](../examples/ImageToText.json) — BLIP image captioning
- [ImageToTextBlip2.json](../examples/ImageToTextBlip2.json) — BLIP-2 conditional captioning
- [CaptionToImage.json](../examples/CaptionToImage.json) — Caption then regenerate
- [InterpolateFrames.json](../examples/InterpolateFrames.json) — RIFE frame interpolation
- [MetadataEmbed.json](../examples/MetadataEmbed.json) — Embed generation parameters in PNG
- [ExpandPrompt.json](../examples/ExpandPrompt.json) — LLM prompt expansion
- [ExpandAndGenerate.json](../examples/ExpandAndGenerate.json) — Expand prompt + generate image
