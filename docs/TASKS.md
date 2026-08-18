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

### video_frames

The frames of a generated video, as one `(frames, height, width, channels)`
uint8 array. That is the shape an argument taking frames rather than a video
wants - LTX-2's keyframe conditions, which are mapped from 0-255 - and it is one
artifact where a list of frames would become one artifact per frame and multiply
the step that consumed it:

```json
{
    "name": "opening_frames",
    "task": {
        "command": "video_frames",
        "arguments": { "video": "previous_result:opening" }
    },
    "result": { "content_type": "video/mp4", "save": false, "fps": 24 }
}
```

| Argument | Required | Description |
| -------- | -------- | ----------- |
| `video` | Yes | The video - a frame list, a frame array or tensor, or an audio+video pair |

An argument that goes through diffusers' video processor instead - LTX-2's
IC-LoRA references - wants the `[0, 1]` frames the pipeline returned rather than
this array; hand those over with `previous_result:step.frames`.

**Example:** [LTX2Extend.json](../examples/LTX2Extend.json)

### pair_audio

Pair a video with an audio track, so the two are saved as one muxed file. A
pipeline that generates its own soundtrack returns the pair together; anything
working on the frames alone - a latent upsampler, an interpolator, an upscaler -
returns frames without it, and this puts it back:

```json
{
    "task": {
        "command": "pair_audio",
        "arguments": {
            "video": "previous_result:upscale",
            "audio": "previous_result:base"
        }
    },
    "result": { "content_type": "video/mp4", "fps": 24 }
}
```

| Argument | Required | Description |
| -------- | -------- | ----------- |
| `video` | Yes | The frames - a frame list, a frame array or tensor, or an audio+video pair whose own soundtrack is replaced |
| `audio` | Yes | The soundtrack - a waveform, or the earlier step whose video carried one, which brings its sample rate along |
| `sample_rate` | No | Sample rate of the waveform. Required unless `audio` carries one; given here it wins |

**Example:** [LTX2TwoStage.json](../examples/LTX2TwoStage.json)

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

**Example:** [SpandrelUpscale.json](../examples/tasks/SpandrelUpscale.json) — Generate at 512px, then 4x upscale to 2048px.

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
- [DiffusionUpscale.json](../examples/tasks/DiffusionUpscale.json) — Generate at 512px, then upscale. `mode` selects which: `x4` (the default) reaches 2048px, `x2` reaches 1024px through the latent upscaler.

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

- [Segment.json](../examples/tasks/Segment.json) — Segment an object from an image
- [SegmentAndInpaint.json](../examples/tasks/SegmentAndInpaint.json) — Segment, then inpaint the masked region

## Image Captioning

Generate text captions from images using a vision-language model.

Transformers 5 removed the dedicated `image-to-text` pipeline this task used to build, along with the BLIP/ViT-GPT2/GIT captioning models that ran on it. Captioning now goes through the same `image-text-to-text` pipeline as any other VLM, so `model_name` needs a vision-language model (SmolVLM, Qwen2.5-VL, LLaVA, etc.) and `prompt` is a question put to the model rather than a text fragment to continue.

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
| `image` | Yes | PIL Image, URL/path, or `previous_result:` reference |
| `model_name` | No | HuggingFace vision-language model ID (default: `HuggingFaceTB/SmolVLM-256M-Instruct`) |
| `prompt` | No | What to ask about the image (default: `Describe this image.`) — ask a narrower question for a narrower caption |
| `system_prompt` | No | System instruction for the model |
| `max_new_tokens` | No | Maximum tokens to generate (default: 50) |

The default model is deliberately tiny, matching the footprint of the old captioning default; it produces short, plain captions. Point `model_name` at something larger for detail.

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

- [ImageToText.json](../examples/tasks/ImageToText.json) — Basic captioning with the default model, saves as `.txt`
- [ImageToTextVLM.json](../examples/tasks/ImageToTextVLM.json) — Larger VLM answering a specific question
- [CaptionToImage.json](../examples/tasks/CaptionToImage.json) — Caption an image, then regenerate with Flux

## Extracting Sections

Reduce generated text to a known set of labelled sections, dropping anything else:

```json
{
    "task": {
        "command": "extract_sections",
        "arguments": {
            "text": "previous_result:expand",
            "sections": ["integrated_multimodal_description", "overall_soundscape", "non_diegetic_music"]
        }
    },
    "result": { "content_type": "text/plain" }
}
```

| Argument | Required | Description |
| -------- | -------- | ----------- |
| `text` | Yes | The generated text, usually a `previous_result:` reference |
| `sections` | Yes | Section labels to keep, in the order they should appear |
| `keep_preamble` | No | Keep any text before the first label (default: `true`) |

A section runs from its `label:` to the end of that paragraph, so a blank line ends one and a single newline does not — a field holding one line per item stays intact. Repeats are dropped, missing sections are skipped, and text with no recognised label is returned unchanged.

This exists because a model asked for a rigid format usually produces it and then keeps going — restating the description, appending a summary, or looping until it runs out of tokens. Prompting against that is unreliable, and at small model sizes adding rules to an already long specification can make adherence worse. Trailing text is not free either: a prompt is conditioning, and a pipeline that does not truncate spends memory and attention on whatever arrives. Keeping the fields that were asked for is deterministic where prompting is not.

The built-in `h3_context_ir` workflow applies this to its own output, so a workflow delegating to it receives only the fields MiniMax H3 expects.

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
| `model_name` | No | HuggingFace model ID (default: `Qwen/Qwen2.5-1.5B-Instruct`, or `HuggingFaceTB/SmolVLM-256M-Instruct` when an image is supplied) |
| `image` | No | PIL Image, URL/path, or `previous_result:` reference — see below |
| `repetition_penalty` | No | Vision path only (default: 1.15) — see below |
| `generate_kwargs` | No | Anything else to pass to the model's `generate()` — `no_repeat_ngram_size`, `top_p`, `min_new_tokens`. Merged last, so it overrides the settings above |
| `max_new_tokens` | No | Maximum tokens to generate (default: 500) |

### Writing a prompt from a picture

Supplying `image` switches the task to a vision-language model, so the generated text describes what is actually in the picture instead of what the prompt guesses is there. `model_name` must then name a VLM — a text-only model cannot be loaded as one.

```json
{
    "task": {
        "command": "text_generation",
        "arguments": {
            "prompt": "Write a video prompt that starts from this picture.",
            "image": "previous_result:input_image",
            "model_name": "Qwen/Qwen3-VL-4B-Instruct"
        }
    },
    "result": { "content_type": "text/plain" }
}
```

This matters most ahead of an image-conditioned generation step. Those pipelines pin the supplied picture as the first frame, so a prompt written without seeing it will describe a scene the keyframe contradicts and the two conditionings pull against each other. Pass the same image to both and the prompt agrees with the frame it opens on.

A vision model is large enough to be worth releasing before the generation model loads — see `release_models` in the workflow guide.

Generation stays greedy so a workflow reproduces, but greedy decoding against a long, rigid format specification makes these models loop — emitting a complete answer and then repeating its closing sections until the token budget runs out. The vision path applies a `repetition_penalty` of 1.15 to stop that. Measured on Qwen3-VL against the MiniMax H3 prompt spec, 1.05 still looped through the whole budget while 1.15 ended on its own at a length matching the format's own guidance. Raise it if a model still repeats itself, or set `1.0` to disable.

A penalty reins the looping in but does not guarantee the model stops where the format ends; for that, trim the output with `extract_sections` below.

There is a limit to what a small model will follow. Against the MiniMax H3 spec, neither Qwen3-VL-4B nor 8B produces the `<d>[Language]...</d>` dialogue tag or the `(S1)` speaker ids, whether the idea implies speech or supplies the line verbatim; the 8B is worse on layout, capitalising its section labels. Showing a complete worked example does produce them - by copying the example word for word, which is useless - and a placeholder skeleton does not produce them at all. The visual description these models write is grounded and usable; the dialogue markup is not. Write prompts by hand where a subject has to speak.

For anything the arguments above do not cover, `generate_kwargs` goes straight to `generate()`:

```json
"arguments": {
    "prompt": "a cat on a windowsill",
    "generate_kwargs": { "no_repeat_ngram_size": 25 }
}
```

It is merged after everything else, so it can override `repetition_penalty` and the sampling settings as well as add to them.

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
- [SpandrelUpscale.json](../examples/tasks/SpandrelUpscale.json) — Generate + spandrel 4x upscale
- [FaceRestore.json](../examples/tasks/FaceRestore.json) — Generate portrait + GFPGAN face restoration
- [Segment.json](../examples/tasks/Segment.json) — Text-prompted object segmentation
- [SegmentAndInpaint.json](../examples/tasks/SegmentAndInpaint.json) — Segment + inpaint
- [ImageToText.json](../examples/tasks/ImageToText.json) — BLIP image captioning
- [ImageToTextVLM.json](../examples/tasks/ImageToTextVLM.json) — VLM captioning with a specific question
- [CaptionToImage.json](../examples/tasks/CaptionToImage.json) — Caption then regenerate
- [InterpolateFrames.json](../examples/InterpolateFrames.json) — RIFE frame interpolation
- [MetadataEmbed.json](../examples/tasks/MetadataEmbed.json) — Embed generation parameters in PNG
- [ExpandPrompt.json](../examples/tasks/ExpandPrompt.json) — LLM prompt expansion
- [ExpandAndGenerate.json](../examples/tasks/ExpandAndGenerate.json) — Expand prompt + generate image
