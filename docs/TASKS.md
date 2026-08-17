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
| `strip_exif` | Remove all EXIF/metadata from image | |
| `add_watermark` | Add visible text watermark | `text`, `position`, `opacity`, `font_size`, `color`, `margin` |
| `get_image_size` | Return `{width, height}` dict | |

### EXIF Stripping

Remove all EXIF metadata, GPS coordinates, camera info, and timestamps from images for privacy-safe preprocessing:

```json
{
    "task": {
        "command": "strip_exif",
        "arguments": {
            "image": "previous_result:input_image"
        }
    },
    "result": { "content_type": "image/png" }
}
```

Returns a clean copy with pixel data only — no embedded metadata. Useful as a first step when processing user-uploaded images.

### Watermark Embedding

Add a visible text watermark to images for responsible AI compliance:

```json
{
    "task": {
        "command": "add_watermark",
        "arguments": {
            "image": "previous_result:generate",
            "text": "AI Generated",
            "position": "bottom-right",
            "opacity": 128
        }
    },
    "result": { "content_type": "image/png" }
}
```

| Argument | Required | Description |
| -------- | -------- | ----------- |
| `text` | No | Watermark text (default: "AI Generated") |
| `position` | No | "bottom-right", "bottom-left", "top-right", "top-left", or "center" (default: "bottom-right") |
| `opacity` | No | Text opacity 0-255 (default: 128) |
| `font_size` | No | Font size in pixels, 0 = auto-scale ~3% of image height (default: 0) |
| `color` | No | RGB array for text color (default: white) |
| `margin` | No | Pixel margin from edges (default: 10) |

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

The frame commands accept videos in any shape a result carries them: PIL frame
lists, numpy or torch frame arrays, and audio+video pairs (LTX-2, MiniMax H3).
The extracted frame is always a PIL image.

### concat_videos

Concatenate videos - and the audio generated with them - into one video. The
standalone counterpart of a chained pipeline step's stitching (see "Chained
video generation" in the workflow guide):

```json
{
    "task": {
        "command": "concat_videos",
        "arguments": {
            "videos": "previous_result:gather",
            "trim_frames": 1,
            "crossfade_ms": 75,
            "fps": 24
        }
    },
    "result": { "content_type": "video/mp4", "fps": 24 }
}
```

| Argument | Required | Description |
| -------- | -------- | ----------- |
| `videos` | Yes | The videos to join, in order (from `gather_videos` or `previous_result`) |
| `trim_frames` | No | Frames dropped from the head of every video after the first (default: 0) |
| `crossfade_ms` | No | Equal-power crossfade at each audio seam (default: 75) |
| `fps` | No | Frame rate of the videos - required to join audio when trimming |

### slice_audio

Cut a slice out of an audio track, addressed in seconds or in video frames.
Slices reaching past the end of the track are zero-padded:

```json
{
    "task": {
        "command": "slice_audio",
        "arguments": {
            "audio": "./soundtrack.wav",
            "start_frame": 124,
            "num_frames": 124,
            "fps": 24
        }
    },
    "result": { "content_type": "audio/wav", "sample_rate": 44100 }
}
```

| Argument | Required | Description |
| -------- | -------- | ----------- |
| `audio` | Yes | Path or URL of an audio file, or a waveform from a previous step |
| `start_seconds` / `duration_seconds` | One pair | The slice in seconds |
| `start_frame` / `num_frames` / `fps` | One pair | The slice in video frames |
| `sample_rate` | With a waveform | Sample rate of a directly passed waveform (files carry their own) |

### crossfade_audio

Join audio tracks with an equal-power crossfade. Each seam overlaps the two
tracks by the fade window:

```json
{
    "task": {
        "command": "crossfade_audio",
        "arguments": {
            "audios": "previous_result:slices",
            "crossfade_ms": 75,
            "sample_rate": 44100
        }
    },
    "result": { "content_type": "audio/wav", "sample_rate": 44100 }
}
```

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

**Example:** [SpandrelUpscale.json](../examples/archive/SpandrelUpscale.json) — Generate at 512px, then 4x upscale to 2048px.

## Diffusion Upscaling

Upscale images using Stable Diffusion upscale pipelines. Text-guided upscaling with better detail recovery than traditional super-resolution, especially for faces and textures.

Two modes are available:
- **x4** (default): `StableDiffusionUpscalePipeline` — 4x upscale via `stabilityai/stable-diffusion-x4-upscaler`
- **x2**: `StableDiffusionLatentUpscalePipeline` — 2x upscale via `stabilityai/sd-x2-latent-upscaler`

```json
{
    "task": {
        "command": "diffusion_upscale",
        "arguments": {
            "image": "previous_result:generate",
            "prompt": "high quality, detailed",
            "negative_prompt": "blurry, low quality, artifacts",
            "mode": "x4"
        }
    }
}
```

| Argument | Required | Description |
| -------- | -------- | ----------- |
| `image` | Yes | PIL Image or `previous_result:` reference |
| `prompt` | No | Text guidance for upscaling (default: "") |
| `negative_prompt` | No | Negative text guidance (default: none) |
| `mode` | No | `"x4"` or `"x2"` (default: `"x4"`) |
| `model_name` | No | Override the default model for the selected mode |
| `num_inference_steps` | No | Denoising steps (default: 25) |
| `guidance_scale` | No | Classifier-free guidance scale (default: 9.0) |
| `noise_level` | No | Noise level for x4 mode (default: 20, ignored for x2) |

**Examples:**
- [DiffusionUpscaleX4.json](../examples/archive/DiffusionUpscaleX4.json) — Generate at 512px, then 4x diffusion upscale to 2048px.
- [DiffusionUpscaleX2.json](../examples/archive/DiffusionUpscaleX2.json) — Generate at 512px, then 2x latent upscale to 1024px.

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

**Example:** [FaceRestore.json](../examples/tasks/FaceRestore.json) — Generate a portrait, then restore faces with GFPGAN v1.4.

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

- [Segment.json](../examples/archive/Segment.json) — Segment an object from an image
- [SegmentAndInpaint.json](../examples/archive/SegmentAndInpaint.json) — Segment, then inpaint the masked region

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

- [ImageToText.json](../examples/tasks/ImageToText.json) — Basic BLIP captioning, saves as `.txt`
- [ImageToTextBlip2.json](../examples/tasks/ImageToTextBlip2.json) — BLIP-2 with conditional prompt
- [CaptionToImage.json](../examples/tasks/CaptionToImage.json) — Caption an image, then regenerate with Flux

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

- [ExpandPrompt.json](../examples/tasks/ExpandPrompt.json) — Expand a short prompt and save as `.txt`
- [ExpandAndGenerate.json](../examples/tasks/ExpandAndGenerate.json) — Expand prompt, then generate with Flux

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
| `model_name` | No | HuggingFace repo with RIFE v4.13 weights (default: `imaginairy/rife-interpolation`) |
| `filename` | No | Weights filename within the repo (default: `rife-flownet-4.13.2.safetensors`) |

Uses vendored IFNet v4.13 architecture. Weights are downloaded from HuggingFace Hub on first use.

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

**Example:** [MetadataEmbed.json](../examples/tasks/MetadataEmbed.json) — Generate with Flux and embed parameters in PNG.

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

| Argument | Required | Description |
| -------- | -------- | ----------- |
| `qr_code_contents` | Yes | Data to encode (URL, text, etc.) |
| `height` | No | Used with `width` to derive output resolution (default: 768) |
| `width` | No | Used with `height` to derive output resolution (default: 768) |

The QR code is generated then resampled to `max(height, width)`, aligned to the nearest 64px multiple.

**Example:** [qr_code.json](../examples/archive/qr_code.json) — QR code with artistic ControlNet

## Chat/Dict Plumbing

These small tasks glue together multi-step pipelines that mix raw `transformers` components with task steps — used internally by the builtin `augment_prompt` and `describe_image` workflows, but usable directly in any workflow.

### format_chat_message

Build a `text_inputs` chat message list from a system and user message, in the shape a `transformers.pipeline` text-generation call expects:

```json
{
    "task": {
        "command": "format_chat_message",
        "arguments": {
            "system_prompt": "You are a helpful assistant.",
            "user_message": "variable:prompt"
        }
    }
}
```

| Argument | Required | Description |
| -------- | -------- | ----------- |
| `system_prompt` | Yes | System instruction |
| `user_message` | Yes | User message content |

Returns `{"text_inputs": [{"role": "system", ...}, {"role": "user", ...}]}`. Pass the result to a `transformers.pipeline` step's `text_inputs` argument via `previous_result:`.

### get_dict_value

Extract a single value from a dictionary result (e.g., a `transformers` pipeline's output) for use in a later step:

```json
{
    "task": {
        "command": "get_dict_value",
        "arguments": {
            "dict": "previous_result:augment_prompt",
            "key": "generated_text"
        }
    }
}
```

| Argument | Required | Description |
| -------- | -------- | ----------- |
| `dict` | Yes | Dictionary (or `previous_result:` reference) to read from |
| `key` | Yes | Key to extract |

Returns the value at `key`, or `None` if the key is absent.

### batch_decode_post_process

Decode generated token IDs and run model-specific post-processing (e.g., Florence-2's task-token parsing), using the processor from an earlier pipeline step:

```json
{
    "task": {
        "command": "batch_decode_post_process",
        "pipeline_reference": "describe_image_processor",
        "arguments": {
            "generated_ids": "previous_result:describe_image_model.generated_ids",
            "task": "<DETAILED_CAPTION>"
        }
    }
}
```

| Argument | Required | Description |
| -------- | -------- | ----------- |
| `pipeline_reference` | Yes | Name of an earlier pipeline step whose processor to reuse (sibling of `command`/`arguments`, not inside `arguments`) |
| `generated_ids` | Yes | Token IDs to decode (e.g., a model step's `generated_ids` output) |
| `task` | Yes | Task token to post-process for (e.g., `<DETAILED_CAPTION>`) |

Calls `processor.batch_decode(...)` then `processor.post_process_generation(..., task=task)` and returns `parsed_answer[task]`. See the builtin `describe_image` workflow for the full Florence-2 pattern.

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

- [FluxCanny.json](../examples/flux/FluxCanny.json) — Canny edge ControlNet
- [FluxDepth.json](../examples/flux/FluxDepth.json) — Depth-guided generation
- [qr_code.json](../examples/archive/qr_code.json) — QR code with artistic ControlNet
- [upscale.json](../examples/archive/upscale.json) — Gather, resize, and diffusion upscale
- [SpandrelUpscale.json](../examples/archive/SpandrelUpscale.json) — Generate + spandrel 4x upscale
- [FaceRestore.json](../examples/tasks/FaceRestore.json) — Generate portrait + GFPGAN face restoration
- [Segment.json](../examples/archive/Segment.json) — Text-prompted object segmentation
- [SegmentAndInpaint.json](../examples/archive/SegmentAndInpaint.json) — Segment + inpaint
- [ImageToText.json](../examples/tasks/ImageToText.json) — BLIP image captioning
- [ImageToTextBlip2.json](../examples/tasks/ImageToTextBlip2.json) — BLIP-2 conditional captioning
- [CaptionToImage.json](../examples/tasks/CaptionToImage.json) — Caption then regenerate
- [InterpolateFrames.json](../examples/InterpolateFrames.json) — RIFE frame interpolation
- [MetadataEmbed.json](../examples/tasks/MetadataEmbed.json) — Embed generation parameters in PNG
- [ExpandPrompt.json](../examples/tasks/ExpandPrompt.json) — LLM prompt expansion
- [ExpandAndGenerate.json](../examples/tasks/ExpandAndGenerate.json) — Expand prompt + generate image
