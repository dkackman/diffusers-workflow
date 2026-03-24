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
| `crop_square` | Center crop to square | |
| `add_border_and_mask` | Add border with alpha mask | |
| `add_border_and_mask_with_size` | Border with specific dimensions | `width`, `height` |
| `get_image_size` | Return `{width, height}` dict | |

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
