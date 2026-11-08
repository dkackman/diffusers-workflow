# Task Commands

Tasks are utility operations that run outside of a pipeline step. Use them for preprocessing, assembly, data gathering, and analysis - most are plain code, and some run a small helper model (an upscaler, a face detector, a stem separator).

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

Every task's schema lists a `"device"` argument (`cpu`, `cuda:1`, ...) that pins
where a helper model runs, for example off the accelerator a loaded pipeline is
using. A task that runs no model ignores it.

`GET /api/tasks/{command}` on the [server](SERVER.md) (and `get_task` over MCP)
returns each command's arguments, read from its implementation's real
signature. The web editor builds task forms from it, and validation flags an
unknown task argument as it flags a pipeline one.

**Numeric arguments.** Domains a signature cannot carry are declared in `dw/task_domains.py`, and
validation reports a value outside one as an error at its JSON path: a count of
frames or seconds and a sample or frame rate must be above zero, an offset zero
or above. The commands refuse the same values at run time, which catches a value
that arrived from a `variable:` or an earlier step.

Every command and validation read numbers the same way (`whole_number` and
`real_number` in `dw/task_domains.py`). A number may be a JSON number or a
numeric string, since a `variable:` from the command line is text: `3`, `3.0`,
`"3"` and `"3.0"` are the whole number 3, and `"23.976"` is a frame rate. A
whole-number argument (a frame count, an index, a pixel size, `resample_audio`'s
`target_sample_rate`) refuses a fractional value such as `3.5`, naming the
argument and saying "whole number"; it is never truncated. Every numeric
argument refuses `true`/`false`, non-numeric text and an infinite or NaN value.
Validation and the run refuse with the same sentence. `Task.run` coerces every
declared numeric argument before the command sees it (`coerce_arguments`), so
`exposure: "0.5"` grades exactly as `0.5` does.

## Adding a task

A task is a handler registered with `@register_command` (`dw/tasks/registry.py`).
The function named by `implementation` is the argument schema: its signature is
what `get_task` reports and validation checks. Every validate-time rule is
declared on the same decorator, so a task cannot be half-registered (#692):

```python
from .registry import register_command
from ..task_domains import NON_NEGATIVE, POSITIVE
from ..task_problems import my_task_errors


def my_task(source, num_frames, start_frame=0, mode="fast"): ...


@register_command(
    "my_task",
    implementation="dw.tasks.my_task.my_task",
    domains={"num_frames": POSITIVE, "start_frame": NON_NEGATIVE},
    whole_numbers=("num_frames", "start_frame"),
    choices={"mode": ("fast", "best")},
    static_check=my_task_errors,
    media_arguments=("source",),
)
def _handle_my_task(task, arguments, previous_pipelines):
    return my_task(**arguments)
```

- `domains` - each numeric argument's domain (`POSITIVE`, `NON_NEGATIVE`, ...
  in `dw/task_domains.py`), refused at validation at
  `steps[i].task.arguments.<name>` and at run time.
- `whole_numbers` - the numeric arguments that take only a whole number; the
  rest of `domains` take any real number.
- `choices` - each argument's literal choices; `get_task` lists them.
- `static_check` - a function of the step's `arguments` returning
  `(argument, message)` pairs for rules across arguments, named `*_errors`, in
  `dw/task_problems.py` (the media-timing rules for window, slice and fit are in
  `dw/task_domains.py`), registered to exactly one command. One that needs the
  task's own parser lives in the task's module behind an import on use, as
  `attribute_voices`' does.
- `media_arguments` - an argument that reads a file under a name that does not
  say so (`source`, `media`, `lut`, `clip`, `track`). Validation confines it as
  it does `image` or `*_video`; an argument not listed is not path-checked.

`TASK_ARGUMENT_DOMAINS`, `TASK_ARGUMENT_CHOICES`, `TASK_STATIC_CHECKS`,
`TASK_WHOLE_NUMBER_ARGUMENTS` and `TASK_MEDIA_ARGUMENTS` are read-only views of
these declarations; `tests/test_task_registry_rules.py` pins them to the
registry and to each command's signature.

A new task is:

1. a module in `dw/tasks/` holding the function and its decorator;
2. one import line for that module in `dw/tasks/task.py` - the registry is
   filled by importing `dw.tasks.task`, so a module it does not import
   registers nothing;
3. a section in this file.

A task that loads Hub weights also declares
`register_command(required_weights="dw.tasks.<module>.required_weights")`. The
hook returns `[{"repo": ...}]` for the step's arguments (empty when the repo is
not known yet, e.g. a `previous_result:` `model_name`) and resolves each repo
with `argument_or_default` from `dw/tasks/weights.py`, the call the loader
makes, so the declaration and the load name the same repo. A validate's
`downloads_required` lists those repos; `tests/test_task_weights.py` pins it.

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
| `lineart` | Coarse line art extraction |
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

A validate's `downloads_required` lists the repo each detector fetches when it is not cached: `depth` `Intel/dpt-hybrid-midas`; `depth_estimator` and `depth_estimator_tensor` `Intel/dpt-large`; the pretrained `controlnet_aux` detectors (`mlsd`, `normal_bae`, `lineart`, `openpose`, `hed`, `scribble`, `pidi`, `midas`, `zoe`, `leres`) `lllyasviel/Annotators`; `teed` `fal-ai/teed`; `anyline` `TheMistoAI/MistoLine`; `sam` `ybelkada/segment-anything`; `segmentation` `openmmlab/upernet-convnext-small`. `canny`, `shuffle` and `lineart_standard` fetch nothing. `zoe_depth` and `dw_pose` fetch outside the Hugging Face cache and are not listed.

Each takes an `image`. The detector's own parameters go beside it in `arguments`, not inside the `image` dict, whose keys other than `location` are ignored:

```json
{
    "task": {
        "command": "canny",
        "arguments": {
            "image": { "location": "https://example.com/photo.jpg" },
            "low_threshold": 50,
            "high_threshold": 200,
            "detect_resolution": 1024,
            "image_resolution": 1024
        }
    }
}
```

### Image Manipulation

| Command | Description | Extra Arguments |
| ------- | ----------- | --------------- |
| `remove_background` | Remove image background | |
| `resize_center_crop` | Crop to a centered square, then stretch to width x height - distorts a non-square target | `width`, `height` |
| `resize_resample` | Scale the short side to `resolution` (default 1024), each side rounded to a multiple of 64 | `resolution` |
| `resize_rescale` | Resize to exact dimensions | `width`, `height` |
| `resize_bucket` | Snap to closest model-native aspect ratio | `resolution`, `ratios`, `alignment` |
| `crop_square` | Center crop to square | |
| `recenter_crop` | Re-frame around a chosen point at a chosen scale, so a series of images registers on one feature; the window may run off the source | `center_x`, `center_y`, `crop`, `width`, `height`, `fill` |
| `add_border_and_mask` | Add a black border, as fractions of each side, and its mask; returns `{bordered_image, mask}` | `zoom_all`, `zoom_left`, `zoom_right`, `zoom_up`, `zoom_down`, `overlap` |
| `add_border_and_mask_with_size` | Fit inside `width` x `height`, bordered to that size, with its mask | `width`, `height`, `overlap` |
| `strip_exif` | Remove all EXIF/metadata from image | |
| `add_watermark` | Add visible text watermark | `text`, `position`, `opacity`, `font_size`, `color`, `margin` |
| `get_image_size` | Return `{width, height}` dict | |

A validate's `downloads_required` lists `briaai/RMBG-2.0` for `remove_background` when it is not cached.

### EXIF Stripping

`strip_exif` returns a copy with pixel data only: no EXIF, GPS, camera or timestamp metadata.

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

### Watermark Embedding

`add_watermark` draws a visible text watermark:

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

`resize_bucket` snaps an image to the closest model-native aspect ratio, then resizes with `alignment`-pixel rounding.

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

### ingredients_grid

Lay separate images (a character, a prop, a location) out on one canvas as a
reference sheet: [`frame_grid`](#frame_grid)'s counterpart for separate images,
and the still `ltx2/reference-sheet` conditions on. Pure PIL.

```json
{
    "name": "sheet",
    "task": {
        "command": "ingredients_grid",
        "arguments": {
            "images": ["asset:hero.png", "asset:sword.png", "asset:castle.png"],
            "width": 768,
            "height": 448
        }
    },
    "result": { "content_type": "image/png" }
}
```

| Argument | Default | Description |
| -------- | ------- | ----------- |
| `images` | required | Images in reading order: references, paths or `{"location": ...}` dicts; `gather:` of a `for_each` step splices in |
| `width`, `height` | 768, 448 | Canvas size in pixels |
| `layout` | `auto` | `rows`, `panels` or `auto` |
| `fit` | `contain` | `contain` scales each image to fit whole and pads with `background`; `cover` fills its cell and crops the overflow |
| `gap` | 8 | Pixels between cells, and kept clear of the canvas edge |
| `background` | `white` | Canvas colour, a name or `#hex` |
| `max_images` | 12 | More images than this is refused, never silently dropped |

`rows` keeps each image's aspect ratio and justifies each row to the canvas
width, searching the in-order row splits for the least wasted canvas, overflow
and row-height spread. `panels` gives every image an equal grid cell, with a
short last row centred. `auto` uses `rows` for up to 3 images and `panels` for
more. Transparency is flattened onto white. Returns one RGB image of exactly
`width` x `height`.

To build the sheet inside `ltx2/reference-sheet`, add this step first and set
`static_sheet`'s `video` to `previous_result:sheet` in place of the
`reference_sheet` asset, keeping `width` and `height` the clip's own:

```json
{
    "name": "sheet",
    "task": {
        "command": "ingredients_grid",
        "arguments": {
            "images": ["asset:hero.png", "asset:sword.png", "asset:castle.png"],
            "width": "variable:width",
            "height": "variable:height"
        }
    },
    "result": { "content_type": "image/png", "save": false }
}
```

### grade

The four finishing commands (`grade`, `sharpen`, `film_grain`, `apply_lut`)
share these rules. They run on CPU, on an image or a video. Their input is named
`media`, and a file path or `asset:`/`output:` reference to a video is read with
its audio. A video is processed one frame at a time and keeps its frame count,
frame rate and audio. An alpha channel passes through untouched. A value outside
an argument's range is refused by `validate_workflow`, naming the argument and
the range, and again at run time when it arrives from a `variable:` or an
earlier step; `get_task(<command>)` reports each range.

`grade` adjusts exposure, contrast, tonal range, local contrast, white balance,
colour and a matte or vignette look.

```json
{
    "task": {
        "command": "grade",
        "arguments": {
            "media": "previous_result:generate_video",
            "highlights": -0.3,
            "shadows": 0.2,
            "clarity": 0.25,
            "fade": 0.15,
            "vignette": 0.3
        }
    },
    "result": { "content_type": "video/mp4" }
}
```

Every argument is optional and defaults to identity, so a step with none
returns the input pixels. The step's job events carry one log naming every
argument not at its identity value, or "no adjustment (all identity)".

| Argument | Range | Identity | Effect |
| -------- | ----- | -------- | ------ |
| `exposure` | any | 0.0 | Stops to brighten (positive) or darken (negative): a multiply by 2^exposure |
| `contrast` | 0 or above | 1.0 | Multiplier around mid grey; below 1 flattens |
| `whites` | -1.0..1.0 | 0.0 | Moves the white point: the top of the curve up or down, black held |
| `blacks` | -1.0..1.0 | 0.0 | Moves the black point: the bottom of the curve up (lift) or down (crush), white held |
| `highlights` | -1.0..1.0 | 0.0 | Lifts or pulls down tones above mid grey, by a smooth luma mask; tones at and below mid grey are untouched |
| `shadows` | -1.0..1.0 | 0.0 | The same for tones below mid grey; tones at and above mid grey are untouched |
| `clarity` | -1.0..1.0 | 0.0 | Adds (positive) or removes (negative) medium-detail local contrast in the midtones |
| `temperature` | -1.0..1.0 | 0.0 | Warmer (toward red) or cooler (toward blue) |
| `tint` | -1.0..1.0 | 0.0 | Toward magenta (positive) or green (negative) |
| `saturation` | 0 or above | 1.0 | Multiplier around each pixel's luma; 0 is greyscale |
| `fade` | 0.0..1.0 | 0.0 | Lifts the black floor and flattens the shadows toward it, white kept: a matte look |
| `vignette` | -1.0..1.0 | 0.0 | Darkens (positive) or lightens (negative) toward the corners; the centre is untouched |

**Order of operations** (fixed, whatever order the arguments are written in):
exposure, contrast, whites/blacks, highlights/shadows, clarity,
temperature/tint, saturation, fade, vignette.

### sharpen

`sharpen` is an unsharp mask: the difference between the image and its
Gaussian blur (the detail) is added back, scaled by `amount`. The
[finishing rules](#grade) apply.

```json
{
    "task": {
        "command": "sharpen",
        "arguments": {
            "media": "previous_result:generate_video",
            "amount": 0.8,
            "radius": 1.5,
            "threshold": 3
        }
    },
    "result": { "content_type": "video/mp4" }
}
```

| Argument | Range | Default | Effect |
| -------- | ----- | ------- | ------ |
| `amount` | 0 or above | 1.0 | How much of the detail is added back; 0 is identity. Applied in whole percent, so it is rounded to 0.01 |
| `radius` | above zero | 2.0 | The Gaussian blur's radius in pixels: the scale of the detail that is sharpened |
| `threshold` | 0..255 | 0 | The smallest difference, in channel levels, between a pixel and its blur that gets sharpened; smaller differences are left alone. A whole number of levels: a fractional value is rounded |

### film_grain

`film_grain` adds seeded Gaussian grain, strongest in the midtones and weaker
toward black and white. The [finishing rules](#grade) apply.

```json
{
    "task": {
        "command": "film_grain",
        "arguments": {
            "media": "previous_result:generate_video",
            "amount": 0.15,
            "size": 1.5,
            "chroma": 0.2
        }
    },
    "result": { "content_type": "video/mp4" }
}
```

| Argument | Range | Default | Effect |
| -------- | ----- | ------- | ------ |
| `amount` | 0.0..1.0 | 0.1 | Grain strength; 0 is identity |
| `size` | 1 or above | 1.0 | Grain size in pixels: the noise is generated at 1/size resolution and upsampled |
| `chroma` | 0.0..1.0 | 0.0 | 0 puts the same grain on every channel (brightness only, hue unchanged); 1 draws independent grain per channel; values between mix the two |
| `seed` | a whole number, 0 or above | the workflow's or step's seed | Seeds the grain |

**Seeding:** one generator per step, consumed frame by frame, so each frame
gets different grain and the same seed reproduces the output byte for byte. A
run always has a seed (drawn at random when the workflow names none, and
recorded in the manifest), so rerunning a job reproduces its grain. The step's
job events log the seed used.

### apply_lut

`apply_lut` colours through a 3D lookup table (LUT), read from a `.cube` file
or built from a `palette`, with trilinear interpolation, blended with the
original by `strength`. The [finishing rules](#grade) apply.

```json
{
    "task": {
        "command": "apply_lut",
        "arguments": {
            "media": "previous_result:generate_video",
            "lut": "asset:looks/teal-orange.cube",
            "strength": 0.7
        }
    },
    "result": { "content_type": "video/mp4" }
}
```

| Argument | Range | Default | Effect |
| -------- | ----- | ------- | ------ |
| `lut` | a `.cube` file | none | The LUT: an `asset:` reference (upload one with `upload_asset`, or `POST /api/uploads`), an `output:` reference, or a path inside the workflow's directory, the asset libraries or the output root |
| `palette` | 2..16 `#rrggbb` colours | none | Instead of a `.cube`, a look built from colours ordered dark to light (below) |
| `strength` | 0.0..1.0 | 1.0 | 0 returns the original, 1 the full LUT result; values between blend the two |

Exactly one of `lut` or `palette` is given; both or neither refuses, naming
the two.

**A palette.** `palette` is a list of 2 to 16 `#rrggbb` colours, ordered dark
to light, spaced evenly along the luminance axis: the first colour sits at
black, the last at white. Each pixel keeps its own luminance (Rec. 709 luma)
and takes its hue and chroma from the palette at that luminance, interpolated
between neighbouring colours, so shadows lean toward the first colour and
highlights toward the last. Near black and white the chroma shrinks to stay in
range. The 33³ table is built in memory and applied as a `.cube` is. The same
palette always gives the same result, so one palette passed as a `variable:` to
every shot of a series gives them one look:

```json
{
    "variables": { "look": ["#102030", "#e0c090"] },
    "steps": [
        {
            "name": "look",
            "task": {
                "command": "apply_lut",
                "arguments": {
                    "media": "previous_result:generate_video",
                    "palette": "variable:look",
                    "strength": 0.6
                }
            },
            "result": { "content_type": "video/mp4" }
        }
    ]
}
```

A palette that is not a list, a colour that is not `#rrggbb`, or fewer than 2
or more than 16 colours refuses, naming the bad entry.

**Where the file may be.** A literal path is held to the same roots as any
media argument and must end in `.cube`; anything else is refused, naming only
what the workflow wrote. The file is parsed once per step.

**The `.cube` the parser takes** is the 3D subset of the format, read strictly.
Anything else refuses the step, naming the file and the line:

- at most 16 MiB of UTF-8 text;
- only `TITLE`, `LUT_3D_SIZE`, `DOMAIN_MIN`, `DOMAIN_MAX`, `#` comment lines,
  blank lines and data rows. `LUT_1D_SIZE` (a 1D LUT) and every other keyword
  refuse;
- every keyword before the data, and `LUT_3D_SIZE` exactly once, 2..65;
- a domain of exactly `0 0 0` to `1 1 1` (the default when the file names
  none);
- exactly size³ data rows of three finite numbers, each in 0..1, listed red
  fastest, then green, then blue, as the format specifies.

A file with values slightly outside 0..1 or a wider domain, as some grading
applications write, is refused rather than clipped.

### A finishing chain

The finishing commands chain through `previous_result:`, each taking the last
one's output as its `media`. Grade first, sharpen, and add grain last so it is
not itself sharpened:

```json
{
    "steps": [
        {
            "name": "graded",
            "task": {
                "command": "grade",
                "arguments": { "media": "previous_result:generate_video", "fade": 0.15, "vignette": 0.3 }
            },
            "result": { "content_type": "video/mp4" }
        },
        {
            "name": "sharpened",
            "task": {
                "command": "sharpen",
                "arguments": { "media": "previous_result:graded", "amount": 0.6 }
            },
            "result": { "content_type": "video/mp4" }
        },
        {
            "name": "grainy",
            "task": {
                "command": "film_grain",
                "arguments": { "media": "previous_result:sharpened", "amount": 0.12, "size": 1.5 }
            },
            "result": { "content_type": "video/mp4" }
        }
    ]
}
```

## Video Processing

| Command | Description | Extra Arguments |
| ------- | ----------- | --------------- |
| `get_first_frame` | Extract first video frame | |
| `get_last_frame` | Extract last video frame | |
| `get_frame` | Extract frame at index | `frame_index` |
| `stills_to_video` | Hold storyboard stills for a duration each, as one silent clip | `images`, `durations`, `fps` - see [stills_to_video](#stills_to_video) |
| `window_video` | One overlapping, fixed-length window of a long video, with its audio | `index`, `num_frames`, `overlap`, `fps` - see [window_video](#window_video) |
| `join_windows` | Blend processed windows back into one video the source's length, with the source's audio | `videos`, `source`, `num_frames`, `overlap`, `curve`, `fps` - see [join_windows](#join_windows) |
| `fit_to_model` | Fit a video into a model's working size and exact frame count; returns `{video, fit}` | `width`, `height`, `num_frames`, `mode` (`letterbox`, `stretch`, `crop`), `downscale` - see [fit_to_model and restore_to_source](#fit_to_model-and-restore_to_source) |
| `restore_to_source` | Put a model's output back at its source's size and length, from the `fit` record | `fit`, `fps` (omitted: the input's rate; `"source"`: the source's; or a number) - see [fit_to_model and restore_to_source](#fit_to_model-and-restore_to_source) |

The frame commands accept videos in any shape a result carries them: PIL frame
lists, numpy or torch frame arrays, and audio+video pairs (LTX-2, MiniMax H3).
The extracted frame is always a PIL image.

### stills_to_video

Hold each still on screen for a set time, as one clip: an animatic of a
storyboard. The result joins and scores like any shot.

```json
{
    "name": "animatic",
    "task": {
        "command": "stills_to_video",
        "arguments": {
            "images": [
                "previous_result:frame_1",
                "previous_result:frame_2",
                "previous_result:frame_3"
            ],
            "durations": [1, 2, 0.5],
            "fps": 24
        }
    },
    "result": { "content_type": "video/mp4", "save": true }
}
```

| Argument | Required | Description |
| -------- | -------- | ----------- |
| `images` | Yes | The stills in order - images, arrays, or `{"media_type": "image", "location": "asset:x.png"}`. All the same size |
| `durations` | Yes | Seconds per still: one number for all, or one per still |
| `fps` | No | Frame rate, default 24, at most 120 |

Each still gets `max(1, round(duration * fps))` frames. The clip carries a
silent 48 kHz mono track of its own length (`assemble-and-score` refuses a video
with no track), and its `shots` record where each still starts.

Refused: stills of different sizes (naming both); a non-positive duration; an
`fps` out of range; a duration count that does not match a literal image list;
a total over 600 seconds or 36000 frames; and, before allocating, a clip over
the 4 GiB in-memory frame budget (frames x width x height x 3 uint8 bytes),
naming the highest `fps` that fits, smaller images, or a split joined with
`concat_videos`. All are caught by `validate_workflow` when the values are
literal; the budget needs every still to be a file (`asset:`, `output:`, a
path) whose header gives its size, so stills from an earlier step are
checked when the run starts.

### concat_videos

Concatenate videos and their audio into one video: the standalone counterpart
of a chained pipeline step's stitching ("Chained video generation" in the
workflow guide).

```json
{
    "task": {
        "command": "concat_videos",
        "arguments": {
            "videos": ["previous_result:shot_1", "previous_result:shot_2"],
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
| `videos` | Yes | The videos to join, in order: `previous_result` references, or the path or URL of a video file (read with its muxed audio), each optionally wrapped in `{"location": ...}`. All must be the same frame size; see [join input rules](#join-input-rules) |
| `trim_frames` | No | Frames dropped from the head of every video after the first (default: 0) |
| `crossfade_ms` | No | Equal-power crossfade at each audio seam, drawn from the trimmed material - no effect when `trim_frames` is 0, and validation warns when one is written there (default: 75) |
| `audio_bleed_ms` | No | How long the outgoing video's tail rings on over the head of the next one, at seams with nothing trimmed to crossfade (default: 0, off) |
| `audio_bleed_gain_db` | No | Gain applied to the bled tail before it is added, in dB - negative ducks a tail that would otherwise push the seam over 0 dBFS (default: 0, unchanged) |
| `seam_fade_ms` | No | Fade on each side of a seam that gets neither a crossfade nor a bleed - for tonal material, not for a continuous bed (default: 3, just enough not to click) |
| `fps` | No | Frame rate of the videos - required to join audio when trimming, and the rate the joined file is written at unless `result.fps` overrides it. An input carrying a different rate is re-timed, with an `fps_overrides_input` warning |
| `match_levels` | No | Even the shots' loudness out before joining - `"rms"` for perceived level (the measurement `get_gallery_metadata` reports as `mean_dbfs`), `"peak"` for the loudest sample. Off by default |
| `match_levels_dbfs` | No | The level `match_levels` moves every shot to (default: -1 dBFS for `peak`, -20 dBFS for `rms`). A shot that would clip at the target is held at -3 dBFS peak instead for `rms` (room for the AAC mux; -0.5 for `peak`), reported as a `match_levels_held` warning with a per-shot log event |
| `sample_rate` | No | The joined soundtrack's rate. Default: the highest among the inputs, the rest resampled up with a warning (#108) |
| `continuous_joins` | No | `true` when the joins continue one take rather than cut between shots, for example pieces of one chain take joined back up. Each seam is recorded without `hard_cut`, so `analyze_seams` measures its frame jump as it measures a chain's own seams, and the audio keeps the default few-ms declick (`seam_fade_ms: 0` butt-joins it). Refused beside `seam_fade_ms` above 0 or `audio_bleed_ms` (default: `false`, every seam a cut) |

Shots an earlier run wrote are joined by path or URL without regenerating them.
A file's track, and a `previous_result:` shot's, is fitted to its frames'
duration, so codec padding does not drift the sound off the picture across
seams:

```json
{
    "task": {
        "command": "concat_videos",
        "arguments": {
            "videos": [
                "/path/to/outputs/shot_01.mp4",
                "/path/to/outputs/shot_02.mp4",
                "previous_result:shot_03_rerendered"
            ],
            "trim_frames": 0,
            "fps": 24
        }
    },
    "result": { "content_type": "video/mp4", "fps": 24 }
}
```

Give each video its own entry. One `previous_result` reference naming a step
that produced several videos fans the step out, one concatenation per video.

**Seams.** `trim_frames` suits a *chain*: the trimmed head covers the same time
as the outgoing tail, so `crossfade_ms` blends the two. A *cut* has nothing to
crossfade, and generated shots tend to open on near-silence. `audio_bleed_ms`
lays a decaying copy of the outgoing tail over the incoming head, shortening
neither side. A shot's head is silent longer than the picture suggests: on a
five-shot H3 sitcom cut, 700 ms left a 44 dB hole, 1800 ms 32 dB, and 2500 ms
little better, so the `dialogue-short` template defaults `audio_bleed_ms` to
1800:

```json
{
    "task": {
        "command": "concat_videos",
        "arguments": {
            "videos": ["previous_result:shot_1", "previous_result:shot_2"],
            "trim_frames": 0,
            "audio_bleed_ms": 1800,
            "fps": 24
        }
    },
    "result": { "content_type": "video/mp4", "fps": 24 }
}
```

A bleed copies ambience; on anything tonal (music, a half-spoken word) the
copy reads as a stutter, and `bleed_join` warns when the outgoing tail looks
tonal rather than noise-like. For a tonal ending, fade the seam with
`seam_fade_ms` (a hundred or so ms) and accept the cut. On a continuous bed (a
laugh track, room tone) a longer fade only deepens the hole (54-59 dB with a
250-500 ms fade and no bleed). `audio_bleed_ms` wins where both are set and
there is material to bleed. A bleed covers the gap but cannot fill it; the
complete fix is a continuous bed under the whole cut: `slice_audio` a few
seconds of tone, [`loop_audio`](#loop_audio) it to the cut's length,
`mix_audio` it under and `pair_audio` it back onto the picture.

#### Join input rules

These apply to `concat_videos` and `dissolve_videos` (and frame size and rate
to `join_into_song`).

- **Frame size.** Every video must be the same size; nothing is resized. A size
  disagreement between `asset:`/`output:`/literal-path entries (wrapped in
  `{"location": ...}` or not) is refused at validate; one involving a
  `previous_result:` fails at run time (#504, #518). To fit the odd video:
  `video_frames`, then `resize_rescale` to the target size (`resize_center_crop`
  squares then stretches, distorting a non-square target), then
  `pair_audio(fit="video")` to put its soundtrack back (#551).
- **Levels.** Independent shots land far apart (two shots of one scene measured
  `peak_dbfs` -2.65 and -12.56), a jump either side of the seam that no fade can
  hide. `match_levels` scales each input before the join: `"rms"` for perceived
  level, `"peak"` for material with big transients. Each input is levelled as
  one unit; an input that is itself a join keeps its inner steps, and one wider
  than 3 dB is an `inner_level_step` warning. Left off, a spread of 6 dB or more
  is a job warning, unless a later step such as `pair_audio` replaces the track.
  A shot at or below -40 dBFS, or one needing 20 dB or more of gain, is reported
  as `match_levels_near_silent`.
- **Sample rate.** Inputs at different rates are resampled up to the highest,
  with a warning, unless `sample_rate` pins the target.
- **Joined track.** The soundtrack is fitted to the joined frames. A track short
  of the frame grid is padded with silence: a frame or more warns
  `joined_audio_padded_to_frames`, less is only logged. The AAC encode can trim
  a further 16-32 samples, logged, or warned as `joined_audio_short_after_mux`
  if it reaches a frame. The shot records are re-measured against the file as
  written, so `media.shots` stays accurate.

### export_timeline

A joined piece as an editable timeline: an OpenTimelineIO `.otioz` bundle that
DaVinci Resolve (18.5+) imports with one clip per shot. The bundle is a zip of
`version.txt` (`1.0.0`), `content.otio` and a `media/` folder; each clip's
`target_url` is `media/<file>`, percent-encoded. The OTIO JSON is written directly, so there is
no `opentimelineio` dependency. Rates are the exact rational the engine writes
(23.976 is 24000/1001, stored by OTIO as a float) with integral frame values.

Checked by hand in Resolve 21 (2026-11-06): File > Import > Timeline takes the
`.otioz`, or its unzipped `content.otio` beside `media/`, and gives one clip per
shot at the recorded rate. The Mac App Store build of Resolve is sandboxed and
reads only folders it has been granted. Copy the file into
`~/Movies/DaVinci Resolve/` (or another folder added under Media Storage) first.
From elsewhere, such as `~/Desktop`, an `.otioz` import silently does nothing and
an `.otio` reports "The clip was not found".

```json
{
    "task": {
        "command": "export_timeline",
        "arguments": {
            "video": "output:<identity>/latest/final/cut.mp4",
            "name": "episode-1"
        }
    },
    "result": { "content_type": "application/x-otioz" }
}
```

| Argument | Meaning |
| -------- | ------- |
| `video` | The joined video, a stored file (`asset:`, `output:` or a path). A clip still in memory is refused: the bundle carries the file |
| `sources` | One file per shot, in order. Each clip then points at its own file over that file's whole range, at that file's own frame rate, and the bundle holds those files, not `video`. A count other than one per shot is refused. Default: every clip points into `video` over its shot's span, so the bundle holds one media file |
| `shots` | Shot records; else the video's own from the run manifest, else the whole file is one clip |
| `name` | The timeline's name |

By default there is one video track and, when the file has a soundtrack, one
audio track. Where the shots leave frames uncovered the track carries a gap, so
clips and gaps add up to the video's length; shots that overlap are refused,
since a track lays its clips end to end, and so is a shot that runs past the
video's last frame. The concept and schema come from
leonardofrito/Directors-Cut (MIT); see [ATTRIBUTIONS.md](ATTRIBUTIONS.md).

### dissolve_videos

Join videos with a cross-dissolve at every seam, and fade the whole piece in
from and out to a colour: a montage cut to a score, where `concat_videos` cuts.
The [join input rules](#join-input-rules) apply.

```json
{
    "task": {
        "command": "dissolve_videos",
        "arguments": {
            "videos": ["previous_result:shot_1", "previous_result:shot_2"],
            "dissolve_frames": 12,
            "fade_in_frames": 12,
            "fade_out_frames": 24,
            "fps": 24
        }
    },
    "result": { "content_type": "video/mp4", "fps": 24 }
}
```

| Argument | Required | Description |
| -------- | -------- | ----------- |
| `videos` | Yes | The videos to join, in order, in the shapes `concat_videos` takes. All the same frame size |
| `dissolve_frames` | No | Frames of overlap at each seam, blended linearly (default: 12). 0 is a hard cut |
| `fade_in_frames` | No | Frames over which the first video rises out of `fade_color` (default: 0) |
| `fade_out_frames` | No | Frames over which the last video sinks into it (default: 0) |
| `fade_color` | No | The RGB colour the fades come from and go to (default: black) |
| `fps` | No | Frame rate of the videos - required to crossfade audio at a dissolve, and the rate the dissolved file is written at unless `result.fps` overrides it |
| `match_levels` | No | Even the shots' loudness out before joining - `"rms"` or `"peak"`, as with [`concat_videos`](#concat_videos). Off by default |
| `match_levels_dbfs` | No | As with `concat_videos` |
| `sample_rate` | No | The joined soundtrack's rate. Default: the highest among the inputs, the rest resampled up with a warning |

Every seam shortens the result by one overlap, so eight 124-frame shots joined
with 12-frame dissolves run 908 frames, not 992 - size a soundtrack slice to
the joined length, not the sum. When every input carries audio, the tracks are
crossfaded over exactly the seam's span so they stay in step with the picture;
when any input is silent the result is, and `pair_audio` puts a score under it.

**Example:** [dissolve-between-shots.json](../workflows/templates/dissolve-between-shots.json)

### join_into_song

Join a spoken scene into a musical number: dialogue shots keep their own
audio, and the sung shots after them play over the *unbroken* song rather than
the slices each was generated against. The song enters at `dialogue length -
cue_seconds`, measured at run time, so regenerating a dialogue shot at another
length cannot leave a stale offset:

```json
{
    "task": {
        "command": "join_into_song",
        "arguments": {
            "dialogue": ["previous_result:line_a", "previous_result:line_b"],
            "song_shots": "gather:sung",
            "song": "asset:song.mp3",
            "cue_seconds": 1.5
        }
    },
    "result": { "content_type": "video/mp4" }
}
```

| Argument | Required | Description |
| -------- | -------- | ----------- |
| `dialogue` | Yes | The spoken shots, in order, each keeping its own audio: a non-empty list in the shapes `concat_videos`' `videos` takes |
| `song_shots` | Yes | The sung shots, in order, in the same shapes. Their own audio is discarded |
| `song` | Yes | The unbroken track the song shots were sliced from - an audio result, a video or audio file's path, or anything carrying `.audio` and `.sample_rate`. Its rate is the output's |
| `cue_seconds` | No | The song time that lands on the first song shot's frame 0 (the start of the slice it was generated against). `0` (the default) starts the song at the cut; above `0` it enters that long before, under the last spoken line. Must be ≥ 0 and no longer than the dialogue |
| `dialogue_target_lufs` | No | Integrated loudness (BS.1770) each dialogue shot is gained to, with one static gain per shot. Omitted (the default), the shots keep their own levels. A shot too short (under 400 ms) or too quiet to measure is left at its own level, with a `dialogue_unmatched` warning |
| `duck_delay_ms` | No | How long after the song enters the dialogue starts to duck (default: `0`). At or past the dialogue's end, nothing ducks. Must be ≥ 0 |
| `duck_db` | No | How far the dialogue ducks, in dB (default: `-12`). Must be ≤ 0 |
| `duck_ramp_ms` | No | The length of the linear ramp into the duck (default: `250`). Must be ≥ 0 |
| `fps` | No | The rate the videos play at, needed only when none carries one (a pipeline's frames carry none, a file brings its own). Must be > 0 |

Each dialogue shot's track is fitted to its own frames (trimmed or padded,
warning `dialogue_fitted_to_frames` past a frame's difference); a shot with no
track is filled with silence for its length. The joined dialogue ends at sample
`D`, the first song shot's frame 0, and the song is placed at
`D - cue_seconds * sample_rate`, running to the end of the picture; a song
shorter than that warns `song_short` and is padded with silence. The dialogue
ducks by `duck_db` from `duck_delay_ms` after the song enters, over a linear
`duck_ramp_ms` ramp.

Frames are joined one for one: videos of different sizes or rates are refused,
as is a join where no video carries a rate and `fps` is not given, or an `fps`
that contradicts the videos' rate. A step that wrote its video with
`result.fps` hands that rate on.

There is no final normalization; follow with
[`normalize_audio`](#normalize_audio) (the templates normalize to -3 dBFS peak).

The result is one video: the dialogue's frames then the song shots', over the
mix, with a shot record per input (`shot@<key>`, named from the references as
`for_each` names a member), its samples measured off the built waveform.

### stabilize_video

Remove a generated clip's accumulated framing drift, the slow wander a video
model adds over a shot meant to hold still:

```json
{
    "task": {
        "command": "stabilize_video",
        "arguments": {
            "clip": "variable:shot_1",
            "smooth": 0
        }
    }
}
```

| Argument | Required | Description |
| -------- | -------- | ----------- |
| `clip` | Yes | The video - a frame list, a frame array or tensor, an audio+video pair, or the path or URL of a video file, read with its audio, so a shot an earlier run wrote can be steadied without regenerating it |
| `smooth` | No | `0` (the default) or `1` locks the framing to the first frame, for a shot generated from a pinned keyframe. A window of 2 or more frames removes only the wander faster than that window, so a slow deliberate camera move survives |

The argument is `clip`, not `video`, because the engine loads an argument named
`video` itself as bare frames, dropping the soundtrack. Frames are shifted back,
cropped to the region every frame covers and resized to the original size; the
soundtrack passes through. `smooth: 0` on a shot with a deliberate camera move
fights the move, silently. Run it only on a shot that drifts, not on every shot
before a cut.

### trim_video

Keep one span of a video's frames, and its audio over the same span:

```json
{
    "task": {
        "command": "trim_video",
        "arguments": {
            "video": "previous_result:shot",
            "start_frame": 12,
            "num_frames": 72
        }
    }
}
```

| Argument | Required | Description |
| -------- | -------- | ----------- |
| `video` | Yes | The clip. A path, `asset:` or `output:` reference is read with its audio; an earlier step's video is used as is. A URL is fetched as frames only |
| `start_frame` | Yes | The first frame kept, from `0` |
| `num_frames` | Yes | How many frames are kept: `1` or more |
| `fps` | No | The clip's frame rate. Defaults to the rate the clip carries; needed only to cut its audio, which is refused without one |

Frames `[start_frame, start_frame + num_frames)` are kept. The audio is cut to
the same span, each end `frames_to_samples` of a frame index rounded on its
own, so consecutive trims tile the track exactly. A clip with audio comes back
as an audio+video pair; a bare frame list as one silent clip at `fps`, so a
step saves it as one video rather than one per frame. Carried shots are clipped
to the span and re-based to `0`, their sample side cleared; a clip with none
comes back as one shot named after its file (else `video 1`). A span past the clip's end is refused, naming the clip's frame count (never
shortened or padded), as are a `num_frames` of `0` or less and a negative
`start_frame`; `validate_workflow` catches these on literal values.

### chain_repair_cut

The pieces that re-render one segment of a finished chain in place (#1169),
for [templates/minimax/repair-segment](../workflows/templates/minimax/repair-segment.json).
A chain run with `save_segments` and `keep_segments` leaves each segment as a
playable mp4 beside the take, and names each file on the take's shots
(`segment_file`, in the run manifest). This command finds the files from the
take, and cuts the guides a repair render of segment `k` needs and where the
repair sits on the take:

```json
{
    "task": {
        "command": "chain_repair_cut",
        "arguments": {
            "chain_take": "output:minimax/MiniMaxH3I2VChained/latest/final/<the take>.mp4",
            "segment": 3,
            "num_frames": 175,
            "head_frames": 22,
            "tail_frames": 39
        }
    }
}
```

| Argument | Required | Description |
| -------- | -------- | ----------- |
| `chain_take` | One of the two | The chain's final video. Its shots (the run manifest beside it) give each segment's place on the take and name its segment file |
| `chain_output` | One of the two | One of the chain's segment files. Alone, the take is found through the run manifest that names it. Beside a take whose shots name no files, the others are found by name beside it, and refused when they disagree in rate or size (two runs mixed in one directory) or leave a gap; `chain_repair_splice` still needs the take's shots |
| `segment` | Yes | The segment to re-render, 1-based. Only 2 to N-1: the first has no segment before it and the last none after it, and are refused |
| `num_frames` | Yes | The repair render's length, on the 17n + 5 grid: at least `head_frames` + segment k's frames + `tail_frames`, rounded up onto the grid. A smaller one is refused naming the smallest; a larger one reaches further into segment k+1 |
| `head_frames` | No | Segment k-1's last frames, held at the render's frame 0: a whole-latent length, 17m + 5 from 22 (default `22`) |
| `tail_frames` | No | Segment k+1's frames held as a guide ending at the render's last frame, a whole-latent length (default `39`, so the tail hold outnumbers the head) |
| `width`, `height` | No | The render's canvas; when given, refused unless it is the segment files' own size |

A `chain_take` alone whose segment files cannot be found is refused with the
fix for its case: a take run without `keep_segments` (re-run the chain with
`keep_segments` and `save_segments`), a take that is already a repair, which
leaves no segment files (repair the original chain take), and a take copied
away from its run, with no manifest beside it (pass `chain_output` as well).

It returns a record for later steps (`previous_result:cut.<field>`):
`head_guide` and `tail_guide` (clips with their audio, cut from the segment
files, for two `guides` entries with `"audio": true`), `keyframe` (the head
guide's first frame, the fl2va keyframe a guide chain gives its own
segments), `tail_frame` (where the tail guide is placed: `num_frames -
tail_frames`, always on 17j), `keep_start` (the render frame the kept span
starts at), `chain_take`, and `start_frame`/`end_frame` (the take's frames
the span replaces), with `overrun`, `segment_frames`, `segment_files` and
`total_frames` (the take's). The neighbours are counted from their decode.

The frames between the two guides number 12 more than a multiple of 17
(17n + 5 less two 17m + 5 guides), and an interior segment of a 17n + 5 chain
with a 22- or 39-frame carry is a multiple of 17, so the new frames replace
segment k and the first `overrun` frames of segment k+1 (12 at the smallest
render), and the take resumes on segment k+1's frame `overrun`.

### chain_repair_splice

Puts a repair render's span into the take in place (#1169): the second half
of `chain_repair_cut`.

| Argument | Required | Description |
| -------- | -------- | ----------- |
| `chain_take` | Yes | The take, `previous_result:cut.chain_take` |
| `video` | Yes | The repair render |
| `segment` | Yes | The segment repaired, 1-based |
| `start_frame`, `end_frame` | Yes | The take's frames the span replaces, from the cut |
| `keep_start` | Yes | The render frame the span starts at, from the cut |

Apart from frames `[start_frame, end_frame)` the result is the take: its
frames, its frame count (a `duration_seconds` take, shorter than its segment
files, stays its own length) and its track, sample for sample, so every other
seam keeps its crossfade or declick. A `match_audio` take keeps its own track
whole under the repaired picture. Otherwise the span's audio is the render's
own, at the take's rate, blended in from the take's track and back into it
inside the span, over the window the chain blended those seams with (the
shot's `crossfade_ms`, else the few-ms declick). Its shots are the take's,
each keeping its records (a chain seam's `crossfade_ms` and `trim_frames`),
with the span as one `segment k (repaired)` shot and the next shot starting
where the take resumes. Those two record their seams as a chain seam is: the
head frames trimmed (the render's head guide; the take's own head of the
resumed shot) and the blend applied. None of them is `hard_cut`, so
`analyze_seams` measures the repaired seams and reads every other seam as the
take's. No shot names a `segment_file`, so a repaired take cannot be repaired
again: re-render the chain, or repair the original take.

The take must be the chain's own output with its shots (the run manifest's
shot list): they tell a `match_audio` track from the chain's own and give each
seam's blend. A take without them, or whose shots do not place `segment` at
`start_frame`, is refused rather than guessed at.

### window_video

Cut one overlapping, fixed-length window out of a long video, so a
video-to-video model that reads one bucket of frames (121 for LTX) can work
through a longer source a window at a time:

```json
{
    "name": "window",
    "for_each": [
        {"name": "w0", "index": 0},
        {"name": "w1", "index": 1},
        {"name": "w2", "index": 2}
    ],
    "task": {
        "command": "window_video",
        "arguments": {
            "video": "asset:long-take.mp4",
            "index": "item:index",
            "num_frames": 121,
            "overlap": 16
        }
    }
}
```

| Argument | Required | Description |
| -------- | -------- | ----------- |
| `video` | Yes | The long source. A path, `asset:` or `output:` reference is read with its audio; an earlier step's video is used as it is. A URL is fetched as frames only, so its window is silent |
| `index` | Yes | Which window, from `0` |
| `num_frames` | Yes | Frames per window - the model's bucket, e.g. an 8n+1 LTX length |
| `overlap` | Yes | Frames each window shares with the one before it; `0` or more, and below `num_frames` |
| `fps` | No | The source's frame rate. Defaults to the rate its file was read at; needed only to cut its audio |

With `stride = num_frames - overlap`, window `i` covers source frames
`i*stride - overlap` up to (not including) `i*stride + stride`. The first
window starts `overlap` frames before the source and repeats its first frame
there; the last may run past the source's end and repeats its last frame.
Every window is exactly `num_frames` frames, float32 in `[0, 1]` like
`loop_frames`' output. A source of `N` frames takes `ceil(N / stride)`
windows, indexes `0` to `(N - 1) // stride`.

The window's audio is the source's samples for its real frames, cut on frame
boundaries (`frames_to_samples`, #401); the repeated frames carry silence.

Memory (#695): a file source (a path, `asset:` or `output:`) is never decoded
whole; the window reads only the frames it covers plus the soundtrack.

One step makes one window, so drive it with `for_each` over `{name, index}`
entries as above. Refused: an `overlap` not below `num_frames`, a negative
`overlap` or `index` (caught by `validate_workflow` on literal values), and, at
run time, an `index` whose stride would start at or past the source's end
(`index * stride >= source frames`), naming the frame count and the last valid
index.

### join_windows

The other half of `window_video`: blend the processed windows back into one
video the source's length. Give it the windows in order, the source they were
cut from, and the same `num_frames` and `overlap`:

```json
[
    {
        "name": "window",
        "for_each": "variable:windows",
        "task": {
            "command": "window_video",
            "arguments": {
                "video": "variable:source_video",
                "index": "item:index",
                "num_frames": 121,
                "overlap": 16
            }
        }
    },
    {
        "name": "joined",
        "task": {
            "command": "join_windows",
            "arguments": {
                "videos": "gather:window",
                "source": "variable:source_video",
                "num_frames": 121,
                "overlap": 16
            }
        }
    }
]
```

with `"windows": [{"name": "w0", "index": 0}, {"name": "w1", "index": 1}, ...]`.
A step that processes each window goes between the two, and `videos` gathers
that step instead.

| Argument | Required | Description |
| -------- | -------- | ----------- |
| `videos` | Yes | The processed windows, in window order: `gather:<step>` over the step that processed them |
| `source` | Yes | The video the windows were cut from, as for `window_video`'s `video`. Its frame count sets the plan and its track is the output's |
| `num_frames` | Yes | Frames per window, as given to `window_video` |
| `overlap` | Yes | Frames each window shares with the one before, as given to `window_video` |
| `curve` | No | The blend's shape across a seam: `cosine` (the default, `(1 - cos πt) / 2`), `smoothstep` (`3t² - 2t³`) or `linear` (`t`) |
| `fps` | No | The joined video's frame rate: omitted or `"source"` for the rate the source's file was read at, or a number above zero for that rate. No frames are added or dropped; the source's audio is put back over the frames at this rate |

The result has exactly the source's frame count, each source frame once:
window 0's real frames, then each later window's first `overlap` frames
blended over the previous window's last `overlap` frames, and the last
window's pad dropped. The incoming weight is `w(t)` at `t = (k + 1) /
(overlap + 1)` for seam frame `k` - the open ramp `dissolve_videos` uses, so
no seam frame is a bare copy of either side. The windows may be at a
different size from the source (a 2x upscale); the output is at the windows'
size.

Memory (#695): the output is built as uint8, with only a seam's `overlap`
frames blended in float32. A local file or `asset:`/`output:` `source` is read
for its frame count and soundtrack only; a URL source is decoded whole.

The window count must be exactly `ceil(source_frames / (num_frames -
overlap))` (`window_count` in `dw/task_domains.py`; `window_video`'s last valid
`index` is one below it). A different count is refused naming both numbers and
the entries to add or drop: by `validate_workflow` when the count is knowable
from the document (`source` an `asset:`/`output:` reference or literal path,
probed header-only; `num_frames` and `overlap` literal; `videos` a list or a
`gather:` over a `for_each` step), otherwise at run time
(`dw/window_count_errors.py` applies the task's `window_count_problem`). Also
refused: a window whose frame count is not `num_frames` (named by position),
windows of different sizes, an unknown `curve`, an `overlap` not below
`num_frames`, and an `fps` that is neither `"source"` nor above zero.

The output's audio is the source's own track (none when it has none); the
windows' audio is discarded. Its shot records are one per window over the
frames it owns, with `overlap_frames` on every seam and cumulative sample
spans (#401), so `assess_output` reads the seams as dissolves.

### fit_to_model and restore_to_source

A video-to-video model works at its own size and frame count, not the
source's. `fit_to_model` fits the source into the model's `width` x `height`
and exactly `num_frames`, the pipeline runs on the fitted video, and
`restore_to_source` puts the output back at the source's size and length:

```json
[
    {
        "name": "fit",
        "task": {
            "command": "fit_to_model",
            "arguments": {
                "video": "asset:clip.mp4",
                "width": 768,
                "height": 512,
                "num_frames": 121,
                "mode": "letterbox"
            }
        }
    },
    {"name": "upscale", "pipeline": {"...": "reads previous_result:fit.video"}},
    {
        "name": "restore",
        "task": {
            "command": "restore_to_source",
            "arguments": {
                "video": "previous_result:upscale",
                "fit": "previous_result:fit.fit"
            }
        }
    }
]
```

#### Reframe to another aspect

To turn a 16:9 clip into 9:16, 4:5 or 1:1:

- **Crop.** `fit_to_model` with `mode: "crop"` and the target `width`/`height`
  scales each frame to fill and centre-crops it, at full resolution with no
  model, losing the sides. Set `num_frames` to the clip's own length to keep
  every frame. Do not follow it with `restore_to_source`, which undoes the fit.

  ```json
  {
      "name": "reframe",
      "task": {
          "command": "fit_to_model",
          "arguments": {
              "video": "asset:clip.mp4",
              "width": 432,
              "height": 768,
              "num_frames": 121,
              "mode": "crop"
          }
      }
  }
  ```

- **Re-render natively.** A generated clip is usually better re-made at the
  portrait size. Any H3 template takes any 32-multiple `width`/`height` (for
  example 544x960), and that keeps the audio.

Generative outpaint of video (filling the new sides) is not supported; see #1066.

`mode` is `letterbox` (default: scale to fit, centred on black), `stretch`
(resize to fill exactly) or `crop` (scale to fill, centre-crop). The frame
count is exact: a longer source is cut to its first `num_frames` frames and a
shorter one holds its last frame. `downscale` (default 1) divides `width` and
`height`, which must both be divisible by it: a 2x upscaler fits its reference
with `downscale: 2` and passes the full `width`/`height` to the pipeline
(`templates/ltx2/upscale-clip`).

The step returns `{video, fit}` (`previous_result:<step>.video` and `.fit`).
`video` is a float32 `(frames, height, width, 3)` array in 0-1 carrying its
`fps`, ready for a pipeline's `video` or a reference condition's `frames`. The
`fit` record holds `mode`, `source_width`, `source_height`, `source_frames`,
`model_width`, `model_height`, `model_frames`, `content_box` (where the source
sits in the model frame), `source_box` (the part of the source kept, all of it
unless `mode` is `crop`), each box `{x, y, w, h}`, and `source_fps` (`null`
when the source carries none).

`restore_to_source` takes the pipeline's output and that record. The scale is
the output's size over the fitted size and must be the same across and down,
so a 2x upscaler restores to twice the source. Letterbox bars are cropped off
and frames are trimmed to the source's count (a shorter output is returned as
is). `crop` returns the kept region at the source's pixel density times the
scale: 640x480 fitted into 512x288 restores to 640x360. A saved video grows an
odd side by one pixel, since the encoder needs even sides.

Its `fps` sets the restored clip's rate without adding or dropping a frame:
omitted, the input's rate (the pipeline's); `"source"`, the record's
`source_fps` (else the input's); or a positive number. Anything else is
refused, at validation for a literal. `templates/ltx2/alpha-matte` passes
`"source"`, so a 24 fps model run on an 8 fps clip is written back at 8.

Neither task carries a soundtrack; pair the source's back with `pair_audio`.

### crop_face_track

Follow one face through a clip and cut a steady square around it, for a
face-detail pass that needs the face large and still:

```json
{
    "name": "face_crops",
    "task": {
        "command": "crop_face_track",
        "arguments": {
            "clip": "asset:interview.mp4",
            "crop_size": 512
        }
    }
}
```

| Argument | Required | Description |
| -------- | -------- | ----------- |
| `clip` | Yes | The video - frames, an audio+video pair, or the path or URL of a video file. Named `clip` so the engine hands it over as read, with the shots it records |
| `crop_size` | No | Side of every crop in pixels, a multiple of `multiple`. Default `512` |
| `padding` | No | Space added around the face on each side, as a fraction of its size, `0` to `3`. Default `0.6` |
| `gate_full` | No | Face width over frame width at or below which strength is 1. Default `0.06` |
| `gate_zero` | No | Face width over frame width at or above which strength is 0; must exceed `gate_full`. Default `0.12` |
| `min_confidence` | No | Detections scoring below this are ignored; below `1`. Default `0.6` |
| `modulus` | No | The crop count is padded to `modulus * n + remainder` frames - the frame grid of the model the crops feed. At least `1`. Default `8` |
| `remainder` | No | The padded count's remainder, `0` up to `modulus - 1`. Default `1` |
| `multiple` | No | `crop_size` must be a multiple of this - the model's frame-size step. At least `1`. Default `32` |
| `min_frames` | No | Refuse the clip when its padded crop count is below this - the shortest clip the model the crops feed takes. A whole number, at least `1`. No check when unset |
| `max_frames` | No | Refuse the clip when its padded crop count is above this - the longest the model takes. A whole number, at least `min_frames`. No check when unset |
| `fps` | No | Refuse the clip unless its frame rate is this, within `0.01` (23.976 passes for `23.98`, not for `24`). Above `0`. No check when unset |
| `detector_repo` | No | Hub repo holding the YuNet weights. Default `opencv/face_detection_yunet`, read at a pinned revision; any other must be a Hub repo id |
| `detector_file` | No | The detector file in that repo, a bare `.onnx` name. Default `face_detection_yunet_2023mar.onnx` |

A validate's `downloads_required` lists the detector file as a file entry when it is not cached, pinned at its revision when `detector_repo` is the default.

`modulus`, `remainder` and `multiple` follow `plan_cuts`: the workflow declares
the model's grid. The defaults (8, 1, 32) are LTX's 8n+1 frames and 32-pixel
steps; MiniMax H3's is `modulus: 17, remainder: 5, multiple: 32`. The
`face-repair` template tunes the gate for LTX-2.5 at `padding` 1.5, `gate_full`
0.03 and `gate_zero` 0.06: a medium face already renders cleanly, and the wider
context keeps the refine from re-inventing hair and clothing.

**Example:** [face-repair.json](../workflows/templates/ltx2/face-repair.json) — find, refine and paste back a wide shot's far face.

Detection is OpenCV's YuNet on the full frame and on four overlapping enlarged
tiles, merged by non-maximum suppression, so a face a few dozen pixels wide is
still seen. One track is kept, smoothed with an exponential moving average;
across a short miss the last box is held at a decaying strength. The track
restarts at each shot boundary the clip records, or, with none recorded, where
the picture jumps (a sharp drop in its HSV colour histogram, or a spike in
frame difference). A padded square around the smoothed box is cut from every
frame, resized to `crop_size`, and the crops are padded to
`modulus * n + remainder` frames with mirrored warm-up and cool-down frames.

Each frame's strength (how much a face-detail pass should change it) is 1 when
face width over frame width is at or below `gate_full`, 0 at or above
`gate_zero`, linear between, times the hold decay on missed frames.

`min_frames`, `max_frames` and `fps` refuse a clip the downstream model cannot
take, before the detector loads, naming the clip's frame count and its padded
count (`the clip's 346 frames pad to 362 crops on the 17n+5 grid, above
max_frames 345`). With `fps` set, a clip with no frame rate is refused.

It returns `{crops, track}`. `crops` is a video at the clip's frame rate; a
soundtrack is padded with `pad_before` and `pad_after` frames of silence to
exactly `crop_frames / fps` seconds, so `hold_audio:
previous_result:<step>.crops` lines up frame for frame. A soundtrack on a clip
with no frame rate is dropped with a `crop_audio_dropped` warning. `track` is a
JSON record saved as its own `.json` file, read as
`previous_result:<step>.track`:

| Field | Description |
| ----- | ----------- |
| `source` | `{width, height, frames, fps}` of the input |
| `crop_size`, `padding`, `gate_full`, `gate_zero`, `min_confidence` | The parameters used |
| `detector` | `{repo, file}` |
| `pad_before`, `pad_after` | Mirrored frames added to the start and end |
| `crop_frames` | Frames in `crops`, including the padding |
| `face_found` | Whether any frame had a face |
| `message` | Present only when no face was found |
| `frames` | One entry per source frame: `box` (the smoothed `[x, y, w, h]`, or null), `crop` (the `[x, y, side, side]` square cut), `strength`, and `state` (`tracked`, `held` or `none`) |
| `resets` | `[{frame, reason, detail}]` where the track restarted; `reason` is `shot`, `cut` or `lost`; a `cut`'s `detail` is `{histogram_correlation, frame_change}` |

A clip with no face still succeeds: every strength is 0, the crops are the
centre square, and a `no_face_found` warning is raised. `validate_workflow`
refuses, before anything downloads: a `crop_size` that is not a multiple of
`multiple`; a `modulus` below 1; a `remainder` below 0 or not below `modulus`;
a `gate_full` or `gate_zero` above 1, or a `gate_zero` at or below `gate_full`;
a `min_confidence` at or above 1; a `padding` out of range; a `detector_repo`
that is not a repo id; a `detector_file` that is not a `.onnx` name; a
`min_frames`, `max_frames` or `fps` at or below 0; and a `max_frames` below
`min_frames`.

### paste_face_track

Put the crops of a face-detail pass back into the clip they were cut from,
using the `track` record `crop_face_track` returned:

```json
{
    "name": "pasted",
    "task": {
        "command": "paste_face_track",
        "arguments": {
            "clip": "asset:interview.mp4",
            "repaired": "previous_result:face_repair",
            "track": "previous_result:face_crops.track"
        }
    }
}
```

| Argument | Required | Description |
| -------- | -------- | ----------- |
| `clip` | Yes | The video `crop_face_track` tracked - frames, an audio+video pair, or the path or URL of a video file |
| `repaired` | Yes | The crops after a face-detail pass, still padded to the count `crop_face_track` produced; any frame size |
| `track` | Yes | The `track` record `crop_face_track` returned (`previous_result:<step>.track`), or the path of its saved `.json`, such as an `output:` reference |
| `feather` | No | Fraction of the paste's radius that fades out, `0` to `1`. Default `0.3` |
| `color_match` | No | Shift each crop's mean colour onto the source's inside the mask before blending. Default `true` |
| `fps` | No | The output's frame rate: omitted or `"source"` for the clip's own, or a number above zero for that rate. No frames are added or dropped |

The pad frames are dropped: the crop at `pad_before + i` goes back on source
frame `i`, resized to its square and blended with a radial feathered mask
(the inscribed circle, its outer `feather` fraction fading on a smoothstep)
times that frame's strength. A frame with strength 0 passes through exactly.

It returns the source clip with its audio and shots, frame for frame, at the
clip's rate unless `fps` sets another. Refused: a track whose frame count or
size does not match the clip, a `repaired` with fewer frames than the track's
`crop_frames`, and a `track` that is not a `crop_face_track` record; at
validation, a `feather` past `1` and an `fps` that is neither `"source"` nor
above zero.

**Example:** [face-repair.json](../workflows/templates/ltx2/face-repair.json) — the repaired crops pasted back over the source, its soundtrack kept.

### video_frames

The frames of a generated video, as one `(frames, height, width, channels)`
uint8 array: the shape LTX-2's keyframe conditions want (mapped from 0-255), and
one artifact where a list of frames would fan the consuming step out per frame:

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

**Example:** [extend-clip.json](../workflows/templates/ltx2/extend-clip.json)

### loop_frames

Repeat a still, or lap a clip from its first frame, to exactly `num_frames`
frames, with no crossfade: the video counterpart of `loop_audio`, for a
conditioning input shorter than its model reads (LTX-2.5's Ingredients IC-LoRA
reference wants a static video of at least 121 frames).

| Argument | Required | Description |
| -------- | -------- | ----------- |
| `video` | Yes | Frames, or a still. A still on disk is passed as `{"media_type": "image", "location": "asset:x.png"}`, since a bare path under `video` is read as a video file; a still from an earlier step as `previous_result:` |
| `num_frames` | Yes | Frames to return, a whole number of at least 1 |

Returns a `(num_frames, height, width, channels)` float32 array in `[0, 1]`
with no fps or audio: the shape `LTX2ReferenceCondition.frames` takes, not a
keyframe condition, which wants uint8 (use `video_frames`). Refuses an empty
input. No weights.

**Example:** [reference-sheet.json](../workflows/templates/ltx2/reference-sheet.json)

### frame_grid

A contact sheet of a clip: `count` evenly spaced frames, first and last
included, tiled into one PIL image. No weights.

| Argument | Required | Description |
| -------- | -------- | ----------- |
| `video` | Yes | Frames in any shape a result carries |
| `count` | No | Frames to sample, clamped to the clip's length. Default `12` |
| `columns` | No | Tiles per row. Default: a wide-biased grid, the last row left-justified |
| `tile_width` | No | Tile width in pixels; height follows the aspect ratio. Default `320` |
| `label` | No | Burn each tile's timestamp (or frame index, when the clip carries no fps) into its corner. Default `true` |

`count`, `columns` and `tile_width` must be whole numbers of at least 1, and
`label` a boolean. An empty clip is refused.

### pair_audio

Pair a video with an audio track, saved as one muxed file: the way to put a
soundtrack back after a step that works on frames alone (an upsampler, an
interpolator, an upscaler):

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
| `video` | Yes | The frames: a frame list, array or tensor, or an audio+video pair whose soundtrack is replaced. Their own rate is carried through (a file's, or a pipeline's that reports one, #1056); frames that carry none are fitted and written at 8 fps unless `fps` is given |
| `audio` | Yes | An [audio input](#slice_audio). A mono track is duplicated into two channels on save, with a warning, since an mp4 audio stream takes stereo only |
| `sample_rate` | No | Rate of a bare waveform; given, it wins |
| `fps` | No | The rate the frames play at, needed only when they carry none. `fit` and the length check measure against it. `result.fps` is not a substitute: it is applied after the fit, so alone on rate-less frames it writes a 24 fps picture under a track fitted to 8 fps (`video_fps_defaulted`, #1056) |
| `fit` | No | `"video"` cuts or pads the track with silence to the frames' length, warning `audio_padded_to_video` / `audio_trimmed_to_video`. Unset (the default), the track is used as is and a length mismatch warns `audio_video_length_mismatch`. Any other value is refused at run time |

`fit` is exact for the waveform handed to the encoder. The AAC mux can trim or
pad the written track by a further few dozen samples (#428), which is logged;
on a video with recorded `shots` it warns `joined_audio_short_after_mux` only
if it reaches a frame. `get_gallery_metadata`'s `media.shots` and
`assess_output`'s `sync_length` measure the written file, so trust them for
the track's actual length.

When the video carries recorded `shots` (from `concat_videos`,
`dissolve_videos` or a chain step), `pair_audio` remeasures their sample fields
against the new track: every shot but the last starts at
`round(start_frame / fps * sample_rate)`, and the last runs to the written
track's end, so its `num_samples` can sit a few dozen samples off the frame
count's. That is the encoder's trim, not a mismatch; a real length mismatch is
the thresholded warning above.

**Example:** [assemble-and-score.json](../workflows/templates/assemble-and-score.json)

### slice_audio

Cut a slice out of an audio track, addressed in seconds or in video frames.
Either half of a pair may be left out: an omitted start begins at the head, an
omitted duration runs to the end. A slice past the end of the track is
zero-padded to the length asked for, and more than a few milliseconds of
padding warns `slice_past_end`; to fill a cut longer than the recording, build
a bed with [`loop_audio`](#loop_audio) and slice that. `past_end: "stop"` cuts
at the end instead, keeping a short source at its own length: a joined piece
whose sections may come back short (`templates/minimax/music-long`) wants
that, and [`crossfade_audio`](#crossfade_audio) then warns `joined_short` with
the real and the asked total. A `"stop"` slice is a section kept whole up to
its length, so one that drops a tail still playing (its RMS within 10 dB of
the kept part's) warns `slice_dropped_loud_tail` with the seconds dropped and
both levels, however long the tail; a slice that cuts a source a little short
of its end (under 5% of the slice and under 10 s, any non-silent tail) warns
`slice_trimmed_tail` whatever `past_end` is. `end_snap_seconds` moves a cut
that lands inside the source back to its quietest point in that many seconds
before the end - in a song, a phrase gap or a bar break.

**Audio inputs.** In every audio task, `audio` (and each entry of `audios`)
takes the path or URL of an audio file, or of a video file whose soundtrack is
taken; a waveform from a previous step; or an earlier step's video generated
with a soundtrack. Files and videos bring their sample rate; `sample_rate` is
needed only for a bare waveform, and given, it wins.

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
| `audio` | Yes | An [audio input](#slice_audio) |
| `start_seconds` / `duration_seconds` | One pair | The slice in seconds; either may be omitted. `start_seconds` with `num_frames` + `fps` and no duration takes its length from the frames |
| `start_frame` / `num_frames` / `fps` | One pair | The slice in video frames; `fps` is required, start and count may be omitted |
| `lead_frames` | No | Frame form only, default 0: extra audio before the cut. The slice starts at `start_frame - lead_frames` and still runs `num_frames`, so `start_frame` 48 with `lead_frames` 12 at 24 fps is audio from frame 36 for `num_frames` frames. A whole number, 0 or above; a nonzero value is refused with the seconds form, and refused when `start_frame - lead_frames` is below zero (by `validate_workflow` for literals, at run time otherwise) |
| `sample_rate` | With a waveform | Rate of a bare waveform |
| `past_end` | No | `"pad"` (default) zero-pads a slice past the end of the source to the length asked; `"stop"` cuts it at the end, so a short source keeps its own length (logged), and the track carries the length asked for to `fade_audio` and `crossfade_audio`. A slice starting at or past the end is refused with `"stop"` |
| `end_snap_seconds` | No | Off by default (0 is off too); zero or above. When the slice's end cuts into the source, the end moves back to the centre of the quietest 0.2 s window (lowest RMS) within this many seconds before it, never before the slice's start; an even level keeps the exact cut. The slice comes out that much shorter (logged), and the track carries the asked length and the snap on to `fade_audio` and `crossfade_audio`, which counts it in the joined total but not as a track that ended early. An end the source does not reach has nothing to snap |

### gain_audio

Apply a gain, in decibels, to a region of an audio track, addressed as in
`slice_audio`; the rest passes through. Unlike `slice_audio`, a region past the
end of the track is clipped to it, not zero-padded:

```json
{
    "task": {
        "command": "gain_audio",
        "arguments": {
            "audio": "./dialogue.wav",
            "gain_db": -12,
            "start_frame": 124,
            "num_frames": 48,
            "fps": 24
        }
    },
    "result": { "content_type": "audio/wav", "sample_rate": 44100 }
}
```

| Argument | Required | Description |
| -------- | -------- | ----------- |
| `audio` | Yes | An [audio input](#slice_audio) |
| `gain_db` | Yes | Gain to apply within the region, in decibels - negative ducks it, positive boosts it |
| `start_seconds` / `duration_seconds` | No | The region in seconds; either may be omitted |
| `start_frame` / `num_frames` / `fps` | No | The region in video frames; `fps` is required if either is given, start and count may be omitted |
| `sample_rate` | With a waveform | Rate of a bare waveform |

With no region argument the gain applies to the whole track (#395). A start
with no duration runs to the end of the track.

### crossfade_audio

Join audio tracks with an equal-power crossfade. Each seam overlaps the two
tracks by the fade window. When a track was sliced with `past_end: "stop"` and
kept shorter than asked (through a `fade_audio` too), the join warns
`joined_short`: the joined length against the length the tracks asked for,
and each short track's kept and asked seconds. A track `slice_audio`'s
`end_snap_seconds` cut at a quiet point is counted in the totals but named
apart, under `snapped_tracks`; when no track ended early, that shortfall is
logged rather than warned, since the snap was asked for:

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

### fade_audio

Fade a track in from silence and out to it, on the equal-power cosine the seam
joins use:

```json
{
    "task": {
        "command": "fade_audio",
        "arguments": {
            "audio": "previous_result:soundtrack",
            "fade_in_ms": 500,
            "fade_out_ms": 2500,
            "sample_rate": 44100
        }
    }
}
```

| Argument | Required | Description |
| -------- | -------- | ----------- |
| `audio` | Yes | An [audio input](#slice_audio) |
| `fade_in_ms` | No | Length of the fade in, from the head of the track (default: 0) |
| `fade_out_ms` | No | Length of the fade out, to the tail of the track (default: 0) |
| `sample_rate` | With a waveform | Rate of a bare waveform |

**Example:** [audio-trim-fade.json](../workflows/templates/audio-trim-fade.json) — slice a generated track to length, then fade the cut into an ending.

### normalize_audio

Scale a track so its loudest sample, or its integrated loudness, sits at a
level. Without `limit` only the gain changes:

```json
{
    "task": {
        "command": "normalize_audio",
        "arguments": {
            "audio": "previous_result:faded",
            "peak_dbfs": -1.0,
            "sample_rate": 44100
        }
    }
}
```

| Argument | Required | Description |
| -------- | -------- | ----------- |
| `audio` | Yes | An [audio input](#slice_audio) |
| `peak_dbfs` | No | The level the loudest sample is moved to, in dB below full scale (default: -1.0). 0 is full scale. With `limit`, a true-peak (4x oversampled, BS.1770) ceiling the output never crosses |
| `target_lufs` | No | An integrated loudness (BS.1770) to scale the track to instead, in LUFS (0 or below). Without `limit` the gain is capped so the peak stays under `peak_dbfs`, and `target_lufs_capped` warns when that cap wins. A track shorter than 400 ms (or silent throughout) cannot be measured, warns `target_lufs_unmeasurable`, and falls back to `peak_dbfs` |
| `limit` | No | Reach `target_lufs` past a transient instead of letting it cap the gain: a true-peak look-ahead limiter (5 ms look-ahead, 20 ms hold, 150 ms release, linked across channels) holds `peak_dbfs`, and the gain is searched for until the limited track lands within 0.1 LU of `target_lufs` - limiting takes some loudness back, more on dense material. The limiter itself never adds gain, and it stops at 12 dB of reduction - past that the gain stops too, and `target_lufs_capped` warns with `limited: true` and `shortfall_lu` (as it does for any track left more than 0.1 LU short). More than 6 dB warns `limiter_heavy` (pumping can be audible). The step's log carries `constraint: "limiter"`, `gain_db`, `max_gain_reduction_db`, `limited_fraction` (0 when the limiter was not needed), `output_true_peak_dbfs` and `output_lufs` On an input that carries shots, `limiter_uneven_across_shots` names a shot the limiter works much harder on than the rest (default: false) |
| `sample_rate` | With a waveform | Rate of a bare waveform |

A silent track is returned unchanged.

To level dialogue with a laugh or a shout on it, use `limit`, so one transient
does not set the level of every line around it:

```json
{"arguments": {"audio": "previous_result:dialogue", "target_lufs": -16, "peak_dbfs": -1.0, "limit": true}}
```

**Example:** [dissolve-between-shots.json](../workflows/templates/dissolve-between-shots.json)

**Headroom and clipping warnings.** Saving audio or a video with a muxed
soundtrack checks the written level against two thresholds, reported in
`get_job`'s warnings and readable back afterward as `media.peak_dbfs` from
`get_gallery_metadata`:

- `audio_no_headroom`: a plain audio file's waveform peaks at or above
  -0.5 dBFS before encoding.
- `audio_clipped`: the written file, decoded back, measures at or above
  0.0 dBFS.

A video mux reports only `audio_clipped`, so a soundtrack decoding back between
-0.5 and 0.0 dBFS warns nothing; read `media.peak_dbfs` to judge one.

### mix_audio

Layer tracks on top of one another (`crossfade_audio` puts them one after
another), such as a score under a film's own sound:

```json
{
    "task": {
        "command": "mix_audio",
        "arguments": {
            "audios": ["previous_result:soundtrack", "previous_result:world"],
            "gains": [0.5, 1.0],
            "sample_rate": 44100
        }
    }
}
```

| Argument | Required | Description |
| -------- | -------- | ----------- |
| `audios` | Yes | The tracks to layer, each an [audio input](#slice_audio) |
| `gains` | No | One plain multiplier per track, in the same order - not decibels. Defaults to unity on every track |
| `sample_rate` | With a raw waveform | Sample rate of the waveforms. Required unless every track brings its own; given here it wins |

Tracks of different lengths are padded with silence to the longest. The sum is
not rescaled; follow with `normalize_audio` if it peaks past full scale.

**Example:** [dissolve-between-shots.json](../workflows/templates/dissolve-between-shots.json) — a
generated score mixed under the shots' own audio.

### loop_audio

Make a bed of a given length out of a short recording: room tone laid under a
whole cut, the complete fix for the hole at a seam:

```json
{
    "task": {
        "command": "loop_audio",
        "arguments": {
            "audio": "previous_result:room_tone",
            "target_frames": 620,
            "fps": 24,
            "crossfade_ms": 250
        }
    }
}
```

| Argument | Required | Description |
| -------- | -------- | ----------- |
| `audio` | Yes | Path or URL of an audio or video file, a video generated with a soundtrack (which brings its sample rate along), or a waveform |
| `duration_seconds` | One of | How long the bed should be, in seconds |
| `target_frames` / `fps` | One of | How long the bed should be, in video frames — how a bed is matched to a cut exactly |
| `crossfade_ms` | No | Crossfade at each loop point, clamped to the material available (default 250) |
| `sample_rate` | With a waveform | Sample rate of a waveform passed directly; given for a file or a video it overrides the rate they carry |

Laps are joined with an equal-power crossfade, so the loop point does not
click, but a transient in the source recurs once per lap as a level pulse (9.3
dB measured on one source); pick a source with even level, or find one with
[`find_loop_bed`](#find_loop_bed). Only the last lap is trimmed, so the bed
lands exactly on the requested length.

The bed is laid under the cut with `mix_audio` and attached to the picture with
`pair_audio`:

```json
[
{ "name": "bed",   "task": { "command": "loop_audio",
                             "arguments": { "audio": "previous_result:room_tone",
                                            "target_frames": 620, "fps": 24 } } },
{ "name": "mixed", "task": { "command": "mix_audio",
                             "arguments": { "audios": ["previous_result:episode",
                                                       "previous_result:bed"],
                                            "gains": [1.0, 0.25] } } },
{ "name": "cut",   "task": { "command": "pair_audio",
                             "arguments": { "video": "previous_result:episode",
                                            "audio": "previous_result:mixed" } },
  "result": { "content_type": "video/mp4", "fps": 24 } }
]
```

To pick the source, run [`find_loop_bed`](#find_loop_bed) over the range that
should hold room tone, `slice_audio` its top candidate's
`start_seconds`/`duration_seconds`, loop that, and mix it in at the candidate's
`gain`.

### find_loop_bed

Pick the window of a recording worth looping into the room-tone bed above,
measured as it will sound looped. A level check alone misses faint speech
(audible once it repeats every lap), lap-rate modulation, and 1 ms ticks that
50 ms RMS cannot see.

```json
{
    "task": {
        "command": "find_loop_bed",
        "arguments": {
            "audio": "output:episode/latest/final/cut.mp4",
            "start_seconds": 30.0,
            "end_seconds": 90.0
        }
    },
    "result": { "content_type": "application/json" }
}
```

| Argument | Required | Description |
| -------- | -------- | ----------- |
| `audio` | Yes | Path, `asset:`/`output:` reference of an audio or video file (a video is read audio-only - frames are never decoded), or a track or video from `previous_result:` |
| `start_seconds` / `end_seconds` | No | The range to search; the whole file if omitted |
| `min_seconds` / `max_seconds` | No | Shortest/longest window tried (default `0.5` / `2.0`) |
| `max_bin_dbfs` | No | Every 50 ms bin of a window must be at or below this (default `-55`) |
| `max_mean_dbfs` | No | A window's mean (RMS) level must be at or below this (default `-60`) |
| `max_spike_db` | No | Most a window's largest 1 ms peak may sit above its median 1 ms peak (default `12`) |
| `crossfade_ms` | No | The crossfade candidates are looped with - `loop_audio`'s own default, so what is measured is what `loop_audio` will make (default `250`) |
| `loop_seconds` | No | Length of the looped result that is measured (default `10.0`) |
| `target_bed_dbfs` | No | The level each candidate's `gain` is computed to reach (default `-60`) |
| `max_candidates` | No | How many ranked candidates to return (default `5`) |
| `shots` | No | Shot boundaries, the assessment probes' shape: `[{name, start_frame, num_frames}]`, optionally with `start_sample`/`num_samples`; overrides the shots a video carries or its run records |
| `fps` | No | Frame rate the shots' frames count at, for a source with none of its own (an audio file); a video's own rate otherwise |

Every window on a 50 ms grid, from `min_seconds` to `max_seconds` long, is
judged against four rules in order and counted in `rejected` under the first
one it fails, so `rejected` is a tally of the whole grid:

- `too_loud` — a 50 ms bin above `max_bin_dbfs`, or a mean above `max_mean_dbfs`.
- `silent` — digital silence: a mean of zero, or a median 1 ms peak of zero
  (more than half the window is exact zeros, whatever sits in the rest).
- `spike` — its largest 1 ms peak more than `max_spike_db` above its median
  1 ms peak. The largest peak is also looked for in the 5 ms just outside
  each end, unless that neighbouring bin is itself too loud, so a window
  ending on a click is thrown out rather than putting the click's onset at
  the loop's seam.
- `tonal` — the flatness/harmonicity test `bleed_join` uses, taken over every
  0.1 s and every 0.2 s block inside the window (one per 50 ms step; no
  longer than `min_seconds`). A window is tonal when any block in it is, so a
  faint syllable is not diluted by the pauses around it. Flatness is measured
  over the band the source occupies, so an upsampled source's empty top band
  does not read as tonal (#198). A candidate's `flatness`
  and `harmonicity` are the readings of its blocks closest to failing
  (lowest flatness, highest harmonicity).

On a cut, a bed must come from inside one shot. Shots resolve in the
probes' order - the `shots` argument (`shots_source: "argument"`), else the
shots a video from an earlier step carries (`"artifact"`), else the ones the
run manifest beside the file records (`"manifest"`, which is why
`output:` the joined file finds them unasked), else none (`null`, the search
above unchanged). With shots, a window that crosses a boundary, or lies where
no shot covers, is counted under `rejected.shot_boundary` before the four
rules (so theirs count only in-shot windows), each candidate names its
`shot` (`shot@<name>` from a manifest), and `source.shots` lists each shot's
`{name, start_seconds, end_seconds}` as placed in the soundtrack. A shot with
both frames and recorded samples is placed at their overlap - the picture's
cut and the audio's can sit a few samples apart. A shot placed by frames on
a source with no frame rate is refused; pass `fps`.

The survivors are thinned so no two overlap, steadiest source first, to a
pool of 200 (`LOOPED_POOL`), or `max_candidates` when that is larger. Each one
in the pool is then
looped with `loop_audio`'s own crossfade to `loop_seconds` and measured:
`ripple_db` (the 5-95% spread of the looped 50 ms bins), `envelope_peak_db`/
`envelope_peak_hz` (the strongest level wobble) and `lap_component_db` (the
wobble at the lap rate). Candidates are ranked by looped `ripple_db`, lowest
first; one whose `envelope_peak_db` is above -15 dB carries the
`lap_modulation` warning.

```json
{
    "source": { "duration_seconds": 620.4, "sample_rate": 44100, "searched": [30.0, 90.0], "shots_source": null },
    "criteria": { "min_seconds": 0.5, "max_seconds": 2.0, "target_bed_dbfs": -60.0 },
    "candidates": [
        {
            "rank": 1,
            "start_seconds": 41.28,
            "duration_seconds": 1.35,
            "end_seconds": 42.63,
            "shot": null,
            "mean_dbfs": -63.1,
            "max_bin_dbfs": -57.4,
            "spike_db": 4.2,
            "flatness": 0.61,
            "harmonicity": 0.08,
            "looped": {
                "ripple_db": 1.1,
                "envelope_peak_db": -24.0,
                "envelope_peak_hz": 0.4,
                "lap_hz": 0.096,
                "lap_component_db": -26.0
            },
            "gain_db": 3.1,
            "gain": 1.43,
            "warnings": []
        }
    ],
    "rejected": { "too_loud": 812, "silent": 0, "spike": 14, "tonal": 203 },
    "findings": []
}
```

`start_seconds`/`duration_seconds` are `slice_audio`'s arguments and `gain`
is `mix_audio`'s multiplier for reaching `target_bed_dbfs`; the task builds
nothing. Finding nothing is not an error: `candidates` is `[]`, `rejected`
counts the windows each rule threw out, and one `no_loop_bed` finding says
which rule to relax.

It is not an assessment probe (it is not in `list_tasks`' `assessment` list).
The result must be saved as `application/json`; read it over MCP with
`get_output_text`.

Out-of-range literals (`min_seconds`/`max_seconds`/`loop_seconds` at or below
zero, `max_candidates` at or below zero, a negative `crossfade_ms`, and the
like) are refused at validation. `end_seconds` at or before `start_seconds`,
`min_seconds` above `max_seconds`, a range past the end of the file, and a
source shorter than `min_seconds` fail at run time — they depend on the file.

### resample_audio

Convert a track to a different sample rate, such as the rate a pipeline that
conditions on audio wants (MiniMax H3's audio VAE):

```json
{
    "task": {
        "command": "resample_audio",
        "arguments": {
            "audio": "previous_result:edit",
            "target_sample_rate": 44100
        }
    }
}
```

| Argument | Required | Description |
| -------- | -------- | ----------- |
| `audio` | Yes | An [audio input](#slice_audio) |
| `target_sample_rate` | Yes | The rate to convert to, a whole number above zero |
| `sample_rate` | With a waveform | Rate of a bare waveform; given for a file or video, it overrides the rate they carry |

A track already at the target rate is returned untouched. The conversion is
PyAV's. Every audio task returns the waveform with its rate, so tasks chain
through `previous_result:` without restating it; a `result.sample_rate`
decides what is written to disk.

**Example:** [assemble-and-score.json](../workflows/templates/assemble-and-score.json)

### compress_audio

Shape a track's dynamics with an envelope follower: a compressor, limiter or
gate, chosen by `mode`:

```json
{
    "task": {
        "command": "compress_audio",
        "arguments": {
            "audio": "previous_result:mixed",
            "threshold_dbfs": -18.0,
            "ratio": 4.0,
            "attack_ms": 10,
            "release_ms": 100,
            "sample_rate": 44100
        }
    }
}
```

| Argument | Required | Description |
| -------- | -------- | ----------- |
| `audio` | Yes | An [audio input](#slice_audio) |
| `threshold_dbfs` | Yes | The level the envelope is measured against, in dB below full scale. Cannot be above 0 |
| `ratio` | No | How hard the reduction is above the threshold, in `compress`/`gate` mode (default: 4.0). Ignored in `limit` mode, which always holds the signal at the threshold |
| `attack_ms` | No | How fast the envelope rises to a louder signal (default: 10.0). 0 means instantly |
| `release_ms` | No | How fast the envelope falls back after a louder signal ends (default: 100.0). 0 means instantly |
| `mode` | No | `compress` (turn down what's above the threshold), `limit` (hold the signal at the threshold), or `gate` (turn down what's below the threshold) (default: `compress`) |
| `sample_rate` | With a waveform | Rate of a bare waveform |

A silent track is returned unchanged.

`mode: "limit"` is a sample-peak limiter with no look-ahead: the envelope
reacts to a transient as it arrives, so the transient's leading edge and any
inter-sample peak get through. To hold a true-peak ceiling while reaching a
loudness target, use `normalize_audio` with `limit: true`.

### filter_audio

Run a track through one biquad filter stage:

```json
{
    "task": {
        "command": "filter_audio",
        "arguments": {
            "audio": "previous_result:world",
            "cutoff_hz": 120,
            "kind": "highpass",
            "sample_rate": 44100
        }
    }
}
```

| Argument | Required | Description |
| -------- | -------- | ----------- |
| `audio` | Yes | An [audio input](#slice_audio) |
| `cutoff_hz` | Yes | The filter's corner frequency. Must be below the Nyquist frequency (half the sample rate) |
| `kind` | No | `lowpass`, `highpass`, `bandpass`, or `notch` (default: `lowpass`) |
| `q` | No | The filter's resonance/bandwidth (default: 0.707, a Butterworth response) |
| `sample_rate` | With a waveform | Rate of a bare waveform |

A silent track is returned unchanged.

### analyze_audio

Measure a track's peak and RMS level, crest factor and a rough low/mid/high
spectral balance, to tune a `compress_audio` or `filter_audio` step against:

```json
{
    "task": {
        "command": "analyze_audio",
        "arguments": {
            "audio": "previous_result:mixed",
            "sample_rate": 44100
        }
    }
}
```

| Argument | Required | Description |
| -------- | -------- | ----------- |
| `audio` | Yes | An [audio input](#slice_audio) |
| `sample_rate` | With a waveform | Rate of a bare waveform |

Returns a dict, not a track: `peak_dbfs`, `rms_dbfs`, `crest_factor_db`,
`low_dbfs` (20-250 Hz), `mid_dbfs` (250-4000 Hz), `high_dbfs` (4000-20000 Hz).
Each band is its share of the track's power on the same scale as `rms_dbfs`
(their powers sum to the part of it between 20 Hz and 20 kHz), so the loudest
band sits near `rms_dbfs` and compares with `compress_audio`'s
`threshold_dbfs`. A silent track, or a band with no content at the track's
sample rate, reads as `null` rather than `-inf`.

### measure_sync

Measure how far a shot's audio sits from the source slice it should match. `analyze_sync_drift` counts samples against frames, so it cannot see audio whose samples line up but whose content is shifted - a lip-sync model that re-sings the reference late. This cross-correlates the onset envelopes of the two. One call measures one shot; the result is JSON and nothing is built:

```json
{
    "name": "sync",
    "task": {
        "command": "measure_sync",
        "arguments": {
            "audio": "previous_result:shot",
            "reference": "asset:song-slice.wav",
            "max_lag_seconds": 3
        }
    },
    "result": { "content_type": "application/json" }
}
```

| Argument | Required | Description |
| -------- | -------- | ----------- |
| `audio` | Yes | The shot's audio, an [audio input](#slice_audio) |
| `reference` | Yes | The slice of the source it should match, in the same forms. A shot with no slice has nothing to measure and is refused |
| `max_lag_seconds` | No | The largest lag searched, either way (default `2`). A lag at this edge may be larger still, and warns |
| `threshold_seconds` | No | A lag longer than this, either way, is an `audio_out_of_sync` finding (default `0.1`), unless the correlation is under `min_confidence` |
| `min_confidence` | No | A correlation under this is a `sync_low_confidence` finding and the lag is not to be trusted (default `0.3`) |
| `sample_rate` | With a waveform | Rate of a bare waveform |

Returns `{lag_seconds, confidence, max_lag_seconds, audio_seconds, reference_seconds, findings, warnings}`. **`lag_seconds` is positive when the audio is late** - its events fall that many seconds after the reference's - and negative when early. `confidence` is the normalised correlation at that lag, 0 to 1. A silent or flat audio or reference gives `lag_seconds: null` and a `sync_unmeasurable` finding. Each finding is `{rule, severity, at, value, threshold, says}`. It is not part of `assess_output`.

### analyze_beats

Find where a song's beats fall, to cut picture to them (a Music 3 song carries
no tempo map). The beats are tracked through an onset envelope; the result is
JSON:

```json
{
    "name": "beats",
    "task": {
        "command": "analyze_beats",
        "arguments": {
            "audio": "previous_result:song",
            "anchors": [0.52, 31.9]
        }
    },
    "result": { "content_type": "application/json" }
}
```

| Argument | Required | Description |
| -------- | -------- | ----------- |
| `audio` | Yes | An [audio input](#slice_audio) |
| `sample_rate` | With a waveform | Rate of a bare waveform, a whole number |
| `tempo_bpm` | No | The tempo, when known. With exactly one anchor it lays an exact grid and nothing is detected; otherwise it steers detection toward that tempo |
| `anchors` | No | Where beats are known to fall, ascending: a list of times in seconds, or of `{beat_index, seconds}` marks placing detected beat `beat_index` (from 0) at `seconds`. One kind per list |
| `min_bpm` / `max_bpm` | No | The tempo range searched (default `60` / `200`); `min_bpm` must be below `max_bpm` |

Returns `{bpm, beats, downbeat_phase, method, calibration, duration_seconds,
warnings}`: `beats` ascend, in seconds. `method` is `onset` (tracked from the
onsets), `rms_peaks` or `grid`. `downbeat_phase` is which of the first four
beats starts a bar, `null` when none stands out. `calibration` is
`{offset_s, drift, anchors_used}`, the anchors' shift at the first mark and
how much it changes by the last. `warnings` repeats what was emitted.

Anchors correct the detection three ways:

- Two or more warp the detected beats piecewise-linearly through them, so a
  detection that starts late or drifts lands on the marks.
- One alone shifts every beat by the same amount.
- One plus `tempo_bpm` lays an exact grid through it, with no detection:
  `method` is `grid`.

A bare-seconds anchor snaps to the nearest detected beat, so it corrects a
drift of under half a beat; where the detection is further off than that, use
`{beat_index, seconds}` marks, which name the beat instead of guessing it.

A track with no clear, regular onsets (a pad, a swell, a noise bed) falls back
to the peaks of its loudness (`method` `rms_peaks`), with a warning. A faint
pulse that still tracks is kept, with a warning to check a few beats and pass
`anchors` or `tempo_bpm` if they are off. The tempo is the track's strongest
period, folded into the range (a half-time pulse of 50 BPM reports 100). A
silent track, or one whose loudest 50 ms is under -40 dBFS, returns empty
`beats` and a warning; raise a real song's level first (`gain_audio`).

Detected without anchors, `bpm` is within `min_bpm`-`max_bpm`, or `null`: a
pulse outside the range is folded by octaves into it (85.7 BPM searched at
140-200 reports 171.4); where no octave fits, the beats keep to a tempo in the
range and a warning names the pulse; `rms_peaks` peaks with no octave in range
give `null`. With anchors, `bpm` is the warped beats' median tempo, and with
`method: "grid"` it is `tempo_bpm` as given. The beats run to the song's ends.

Refused at `validate_workflow`: `min_bpm` at or above `max_bpm`, and malformed
anchors (mixed kinds, out of order, a non-whole `beat_index`). Refused at run
time: an anchor past the song's end; a `beat_index` past the detected count,
once detection has run; and, for a grid, an anchor that puts beat 0 before
0 s. The result must be `application/json`.

### analyze_phrases

Find when a song's sung phrases start and stop, from voiced energy alone (no
model, no download). It works where Whisper cannot hear the words: a
non-English verse, a hallucinated transcript, wordless vocals. Feed it the
vocal stem where there is one; the result is JSON:

```json
{
    "name": "phrases",
    "task": {
        "command": "analyze_phrases",
        "arguments": {
            "audio": "previous_result:stems.vocals"
        }
    },
    "result": { "content_type": "application/json" }
}
```

| Argument | Required | Description |
| -------- | -------- | ----------- |
| `audio` | Yes | The song or its vocal stem, an [audio input](#slice_audio) (`previous_result:<separate_stems step>.vocals`) |
| `min_gap_seconds` | No | A silence shorter than this between voiced runs is bridged into one phrase (default `0.35`). Raise it to merge phrases split by breaths, lower it to split lines sung close together. At least 0 |
| `min_phrase_seconds` | No | A phrase shorter than this, after bridging, is dropped as a click or breath (default `0.3`). Above 0 |
| `floor_db` | No | The level, in dBFS of a 20 ms frame, a frame must reach to count as voiced. Unset, it is the track's 95th-percentile frame less 35 dB, never below -60 dBFS (the floor `attribute_voices` uses). For a fused legato or reverberant verse, raising it (e.g. `-30`) splits lines where `min_gap_seconds` cannot; check the result by ear, since a raised floor can match the counts on wrong boundaries. At or below 0 |
| `centre` | No | `true` (the default) weights a stereo file to its centre, where a lead vocal is mixed; `false` uses the band energy of the downmix. A real boolean - `"false"` is refused |
| `sample_rate` | With a waveform | Rate of a bare waveform, a whole number |

Returns `{phrases: [{start, end}], duration_seconds, method, warnings}`, in
seconds rounded to 0.01; the phrases ascend and do not overlap.

- **`method: "centre"`** (stereo, `centre: true`): mid = (L+R)/2 and side =
  (L-R)/2 are both bandpassed to 250-3500 Hz, and a 20 ms frame's power is
  max(0, P_mid - P_side), smoothed over three frames. What is panned away from
  the centre cancels, so a hard-panned guitar or noise bed does not read as
  singing - and a vocal panned hard to one side does not either: use
  `centre: false` for it.
- **`method: "band"`**: the band energy of the downmix, for a mono file (which
  also warns), for `centre: false`, and for a stereo file whose channels are
  the same (it has no side to cancel, so no warning).

A frame is voiced at or above the floor; gaps shorter than `min_gap_seconds`
are bridged, then runs shorter than `min_phrase_seconds` are dropped.

Warnings: no phrases found (a silent track, or one whose loudest frame is
under -50 dBFS in the voice band, unless `floor_db` is given); and one phrase
covering more than 90% of the track, whose remedy is `separate_stems`'
`vocals` on a mix, or a raised `floor_db` on a stem. `warnings` repeats what
was emitted.

Refused at validation and at run time: a missing `audio`, a negative
`min_gap_seconds`, a `min_phrase_seconds` at or below 0, a `floor_db` above 0,
and a `centre` that is not `true` or `false`. The result must be
`application/json`.

### plan_cuts

Plan a music video's cuts from a song's lyrics and beats. `transcribe_audio`
(with timestamps) says when each line is sung and `analyze_beats` says where
the beats fall; this turns the two into shots, each a start frame and a
length, that tile the song exactly. A transcript chunk `transcribe_audio`
marked `suspect` `no_words`, `repetition`, `stock_phrase` or `silent` is left
out of the lines; one marked `long` is still used. Each run of consecutive
chunks with the same mark adds one warning naming its span, chunk count and
(clipped) text. Lines squeezed to no time join the next scene, with one warning
per run naming the first three. The result is JSON:

```json
{
    "name": "plan",
    "task": {
        "command": "plan_cuts",
        "arguments": {
            "transcript": "previous_result:transcribe",
            "beats": "previous_result:beats",
            "duration_s": "previous_result:beats.duration_seconds",
            "lyrics": "variable:lyrics",
            "segment_by": "line",
            "fps": 24,
            "min_scene_s": 1.5,
            "max_scene_s": 10,
            "vocal_tail_s": 0.5,
            "snap_to_beats": true
        }
    },
    "result": { "content_type": "application/json" }
}
```

| Argument | Required | Description |
| -------- | -------- | ----------- |
| `transcript` | Yes | `transcribe_audio`'s result with `timestamps` set: a `{text, chunks}` dict of `{start, end, text}` chunks in seconds (a null `end` runs to the next chunk's start, or the song's end). Or an [`analyze_phrases`](#analyze_phrases) result (recognised by its `phrases` key), each phrase one line, with no suspect filtering or word alignment; see [phrase transcripts](#plan_cuts) below |
| `lyrics` | No | The song's own lyrics, one line per line with a blank line or a lone section tag (`[Chorus]`) between stanzas, or a list of lines. The shots carry these lines verbatim and in order; the transcript only times them (an `analyze_phrases` transcript: one line per phrase, see `transcript`). `null` or `""` means none: the transcript's text is used (a phrase result has none: `lyric` is `null`) |
| `beats` | For `segment_by: "beat"` | `analyze_beats`' result, or a list of beat times in seconds |
| `segment_by` | No | `line` (default), `stanza` or `beat` |
| `fps` | No | Frames per second the shots are counted in (default `24`) |
| `duration_s` | No | The song's length. Default: `analyze_beats`' `duration_seconds` when `beats` is that whole dict, else a phrase result's `duration_seconds`, else the transcript's last end (with a warning). `"beats": "previous_result:beats"` passes only the result's `beats` list (the engine takes the key an argument is named for), so pass `"duration_s": "previous_result:beats.duration_seconds"` too |
| `min_scene_s` | No | The shortest a shot may be (default `1.0`) |
| `max_scene_s` | No | The longest a shot may be (default none) |
| `vocal_tail_s` | No | Seconds a sung shot holds past its last word (default `0`) |
| `include_instrumental_gaps` | No | Whether a long silence becomes its own instrumental shot (default `true`) |
| `min_gap_seconds` | No | The shortest silence that becomes its own shot (default `2.0`) |
| `snap_to_beats` | No | Whether every cut moves to its nearest beat (default `false`) |
| `modulus` / `remainder` | No | The model's render grid: a shot renders `modulus * n + remainder` frames (H3: `17`, `5`). With none, a shot renders exactly its cut |
| `min_frames` / `max_frames` | No | The fewest and most frames a shot may render (H3: `124`, `345`) |
| `lead_s` | No | Seconds a shot starts before its cut, as a run-up the model renders and `trim_video` drops. Each shot's `lead_frames` is `min(round(lead_s * fps), start_frame)`, so the first shot has none (default `0`) |

Returns `{shots, bpm, fps, duration_s, total_frames, render_frames,
warnings}`. `shots` tile the song in order, each `{name, start_frame,
num_frames, cut_frames, lead_frames, lyric, kind}`: `start_frame` and
`cut_frames` are the cut's start and length on the timeline (the `cut_frames`
sum to `round(duration_s * fps)`), `lead_frames` the run-up before the cut,
and `num_frames` the length to render: `max(min_frames, lead_frames +
cut_frames)` raised to the grid, so the render starts at `start_frame -
lead_frames` and its cut is frames `[lead_frames, lead_frames + cut_frames)`.
With no grid arguments and no `lead_s`, `num_frames` is `cut_frames` and `lead_frames` is `0`.
`render_frames` is the sum of the `num_frames` - what the render costs, which
is more than `total_frames` when shots lead or round up. `lyric` the lines sung in it joined by newlines or `null`, and `kind`
`"vocal"` or `"instrumental"`. It is a dict, not a list, so it stays one
artifact. Its `result` may only be `application/json`.

**Phrase transcripts.** Without `lyrics`, a phrase is a line with no lyric
(`lyric: null`, kind vocal), and all three `segment_by` values work (`stanza`
groups phrases a `min_gap_seconds` silence apart). With `lyrics` of as many
lines as phrases they map one to one, and a warning names any line whose phrase
is under 1/4 or over 4x the median phrase length. With a different count, a
warning names both counts and the knob (raise `analyze_phrases`'
`min_gap_seconds` for more phrases than lines, lower it for fewer, or raise its
`floor_db` for fused lines), and the lines are spread evenly from the first
phrase's start to the last phrase's end. A long phrase split into pieces gives
every piece its whole lyric.

How the plan is made:

- With `lyrics`, those lines are the lines, and the transcript only lends them
  timings: they are aligned to it word by word, since Whisper mishears sung
  words and splits lines where it likes. A line never heard is placed between
  its neighbours with a warning, as a shot of its own: when its neighbours
  touch, it takes `min_scene_s` (at least half a second) from them, never
  more than half of either. Blank lines and section tags are stanza
  breaks, not sung lines. Without `lyrics`, each transcript chunk is a line.
- `segment_by` makes a shot per line, per stanza (the lyrics' blank-line
  groups, or lines without a `min_gap_seconds` silence between them) or per
  few beats (a cut on the first beat at least `min_scene_s` after the last).
- The silence between lines goes to the shot before it, or - when it lasts
  `min_gap_seconds` and `include_instrumental_gaps` is on - becomes its own
  instrumental shot, as can the silence before the first line and after the
  last.
- A shot over `max_scene_s` splits evenly, each cut on its nearest beat when
  there are beats, and a sung shot split this way is warned about by its
  lyric. Given `lyrics`, each piece holds only the words heard inside it
  (the lyric's words aligned to the transcript's); a piece where no word is
  heard has no lyric and is warned about. When no word of the lyrics was
  heard, or with no `lyrics`, every piece carries the whole lyric. One under `min_scene_s` merges into its
  shorter neighbour.
  A shot still outside the range is warned about by name.
- A shot whose render would pass `max_frames` is split, on a beat when
  there is one, and warned about; its lyric is divided as above.
- `snap_to_beats` moves every cut to its nearest beat; with no beats it warns
  and leaves the cuts where the lines put them.
- Every boundary is rounded once, from its absolute time, so the frame counts
  sum to the song's and no rounding drifts.

Refused, at `validate_workflow` where the value is a literal and at run start
otherwise: a bare-string `transcript` (the message names `timestamps` /
`return_timestamps`), `segment_by` not one of the three, `segment_by: "beat"`
without `beats`, `min_scene_s` above `max_scene_s`, and `fps`, `duration_s`,
`max_scene_s` at or below zero or `min_scene_s`, `vocal_tail_s`,
`min_gap_seconds` or `lead_s` below zero, `modulus`, `min_frames` or
`max_frames` at or below zero, `min_frames` above `max_frames`, a `remainder`
outside `0` to below `modulus` (or one with no `modulus`), and a `max_frames`
below the smallest grid length at or above `min_frames`. Also refused at run
time: a song whose length can't be known (no `duration_s`, no `beats` result,
and no transcript to end on), `lyrics` with no sung lines, and a `lead_s` whose
frames leave no room under `max_frames` for a shot's own frames.
[music-video-cuts.json](../workflows/templates/minimax/music-video-cuts.json)
chains `separate_stems`, `transcribe_audio` (of the vocal stem), `analyze_beats`
(of the full mix) and `plan_cuts`, passing H3's grid (`17`, `5`, `124`, `345`)
and a `lead_s` variable (default `0.5`). When Whisper can't time the words,
swap the `transcribe` step for `analyze_phrases` on
`previous_result:stems.vocals` and pass `lyrics`, one line per sung phrase. Its
shots are what
[music-video.json](../workflows/templates/minimax/music-video.json) renders:
the agent writes a prompt per shot, drops `lyric` and `kind`, and passes the
list as `shots`.

## Assessment Probes

Three read-only commands measure a finished cut and say where to look:
`analyze_shots`, `analyze_seams`, `analyze_sync_drift`. A probe is a command
registered with `assessment=True` (`register_command`, `dw/tasks/registry.py`;
the probes register in `dw/tasks/audio_handlers.py`); other JSON-answering
tasks such as [`attribute_voices`](#voice-attribution) and
[`check_script`](#script-check) are not probes. Each takes a video (a stored
file - `asset:`, `output:` or a path - or the video an earlier step returned;
not a URL, whose download has no soundtrack) and answers one JSON document:
every measurement, plus `findings` (measurements that crossed a rule in the
table below), `rules_applied`, `rules_skipped` (rules that could not run, with
a reason) and `shots_source`. A probe streams the file (a 64x36 grey thumbnail
per frame plus the soundtrack), so it runs on a cut of any length.

Findings are places to look, not verdicts: nothing in the engine acts on one
and no run fails for one.

A probe's `result` must save as JSON:

```json
{
    "task": {
        "command": "analyze_seams",
        "arguments": {
            "video": "output:<identity>/latest/final/cut.mp4"
        }
    },
    "result": { "content_type": "application/json" }
}
```

Any other `content_type` (or none) fails validation.

Shot boundaries come, in order: the step's own `shots` argument, the shots
carried by the video an earlier step returned, the run manifest beside the
file, and otherwise the whole file is treated as one shot. `shots_source`
reports which: `argument`, `artifact`, `manifest`, or `none`. With `none`, the
shot-against-shot rules (`shot_level_spread`, and all of `analyze_seams`') go
to `rules_skipped` with `reason: "no shot boundaries"`, and the run warns
`no_shot_boundaries`.

### analyze_shots

Each shot's level and spectral balance, and how far apart the shots sit:

| Field | Meaning |
| ----- | ------- |
| `shots[].name` | The shot's name |
| `shots[].start_frame` / `num_frames` | The shot's frame range, as the shot record gave it |
| `shots[].peak_dbfs` | Peak level within the shot |
| `shots[].rms_dbfs` | RMS level within the shot |
| `shots[].crest_db` | `peak_dbfs` minus `rms_dbfs` |
| `shots[].low_dbfs` / `mid_dbfs` / `high_dbfs` | Spectral balance (20-250 Hz / 250-4000 Hz / 4000-20000 Hz), on the same scale as `rms_dbfs` |
| `shots[].samples` | Whether the shot's sample span was `recorded` (carried by the shot record) or `derived` (scaled from its frames) |
| `shots[].dead_air_seconds` | The longest run of 50ms windows inside the shot at or below the `DEAD_AIR_FLOOR_DBFS` threshold (-65 dBFS) |
| `shots[].dead_air_at` | Where that run starts, in seconds into the file |
| `shots[].dead_air_floor_dbfs` | The quietest 50ms window measured inside that run - not the threshold. Null when the run is pure digital silence, and null when there is no run at all |
| `rms_range_db` | The spread between the loudest and quietest voiced shot |
| `has_audio` | Whether the file carries a soundtrack at all |

### analyze_seams

Every seam between shots, audio and picture:

| Field | Meaning |
| ----- | ------- |
| `seams[].seam` | The seam's index (1-based) |
| `seams[].between` | `[previous shot name, next shot name]` |
| `seams[].seconds` | Where the seam sits in the file |
| `seams[].kind` | `cut` or `dissolve` (a dissolve has `overlap_frames`) |
| `seams[].hard_cut` | Whether the incoming shot is marked `hard_cut: true`; `concat_videos` sets it on a shot opening a seam it butt-joined (a plain cut, or a cut with a `seam_fade_ms` fade). False on a bled seam |
| `seams[].audio_bleed_ms` | The bleed the seam got (clamped to the material); absent where none ran. The picture is still a cut, so `seam_frame_jump` stays quiet |
| `seams[].seam_fade_ms` / `crossfade_ms` | Copied from the incoming shot record when present; such a seam skips `seam_hole` |
| `seams[].before_shot_rms_dbfs` / `after_shot_rms_dbfs` | RMS level of the whole shot either side of the seam |
| `seams[].level_step_db` | The absolute difference between those two shot levels. Shot against shot, not the audio at the seam's edges: a take's own tail and head can sit 20 dB apart, which is not a step the cut made |
| `seams[].before_rms_dbfs` / `after_rms_dbfs` | RMS level of the 0.25 s either side of the seam - what `seam_hole`'s both-sides-voiced guard reads |
| `seams[].floor_dbfs` | RMS level of the join itself (the fade, or a short window centred on a cut) |
| `seams[].click_db` | How far a spike at the join peaks above its immediate neighbours |
| `seams[].spectral_shift` | How much the low/mid/high balance shifts across the seam (0-1) |
| `seams[].frame_delta` | The largest single-frame picture change across the seam |
| `seams[].typical_delta` | The larger shot's own typical frame-to-frame change, floored |
| `seams[].jump_ratio` | `frame_delta` divided by `typical_delta` |
| `seams[].saturation_step` | Mean saturation (per-pixel max(RGB) - min(RGB)) of the 4 frames after the seam minus the 4 before it (before a dissolve's fade), clipped to the shots. Signed: positive means the incoming shot is more saturated |
| `seams[].lightness_step` | The same for mean Rec. 709 luma. Signed: negative means the incoming shot is darker |
| `seams[].contrast_step` | The same for the luma standard deviation. Signed |
| `seams[].look_ratio` | The largest of the three steps' sizes divided by that measure's own local rate: the 90th percentile of its change from one 4-frame mean to the next inside either shot, floored at 1.0. A step inside a shot that is itself changing that way (a sunrise) scores low |
| `seams[].look_ratio_measure` | Which measure gave `look_ratio`: `saturation`, `lightness` or `contrast` |
| `seams[].detail_ratio` | Mean high-pass detail (luma against its 3x3 blur) after the seam divided by before it: below 1 means the incoming shot is softer. Null when the before window has almost no detail (below 0.5) |

The six look fields are measures, not rules: none raises a finding, and they
appear in neither `rules_applied` nor `rules_skipped`. They are on the 0-255
scale of a 128x72 colour copy of each frame, and the numbers are not
perceptual - a `look_ratio` of 10 is not a calibrated "visible". They are null
when the file has no picture.

### analyze_sync_drift

How far the soundtrack sits from the picture, shot by shot and over the
whole file. Only shots whose records carry `start_sample`/`num_samples` are
measured; with none, `shots` is empty, `max_offset_ms` is null, and only
`sync_length` can fire:

| Field | Meaning |
| ----- | ------- |
| `shots[].name` | The shot's name |
| `shots[].start_offset_ms` | How far the audio sits from the picture at the shot's start |
| `shots[].end_offset_ms` | How far the audio sits from the picture at the shot's end |
| `max_offset_ms` | The largest `end_offset_ms` across all shots, by magnitude |
| `video_seconds` / `audio_seconds` | Each stream's own duration |
| `length_delta_ms` | `audio_seconds` minus `video_seconds` |

### analyze_face_drift

Whether a recurring character still looks like their cast portrait. It is not
a probe: it measures faces, and its two rules live in the command rather than
in the table below. It returns `measurements`, `findings`, `rules_applied`,
`shots_source` and `notes` (no `rules_skipped`). Needs the SFace and YuNet
weights from `opencv/face_recognition_sface` and `opencv/face_detection_yunet`,
which `validate_workflow` lists under `downloads_required`.

```json
{
    "task": {
        "command": "analyze_face_drift",
        "arguments": {
            "video": "output:<identity>/latest/final/cut.mp4",
            "reference": "asset:cast/ada.png"
        }
    },
    "result": { "content_type": "application/json" }
}
```

| Argument | Meaning |
| -------- | ------- |
| `video` | A stored video, or the video an earlier step returned |
| `reference` | The face to hold. With several faces the largest is used and `notes` says so; with none the step is refused |
| `shots` | Shot records, as the probes take them; else the video's own, else the run manifest, else the whole file is one shot (and `notes` says so) |
| `threshold` | Median cosine similarity below which a shot raises `face_drift`, from -1 to 1. Default 0.363, OpenCV's published SFace threshold |
| `samples_per_shot` | Frames read per shot, evenly spaced, 1 to 64. Default 8 |

Each shot is a `measurements[]` entry: `shot`, `index` (its place in the
shot list), `start_frame`, `num_frames`, `start` and `end` (seconds; null
when the clip has no fps), `frames_sampled`, `faces_found` (sampled frames with a face), `similarity_median` and
`similarity_min` - over those frames, the best cosine match among the faces
in each frame. Both are null when no sampled frame had a face. Each
finding's `at` carries the shot's `shot`, `index`, `start_frame`,
`num_frames`, `start` and `end`, so two shots with one name stay apart. Findings:
`face_drift` (warn) when the median is below `threshold`, and `no_face`
(info) for a shot where no sampled frame had a face. Like the probes, they
are places to look, not verdicts: a turned head or a mask also lowers the
score, so look at the shot before regenerating it.

### Rules

Each rule names the probe and field it reads, how the value is compared to
its threshold, and the severity of a crossing:

| Rule | Probe | Field | Threshold | Severity |
| ---- | ----- | ----- | --------- | -------- |
| `shot_level_spread` | `analyze_shots` | `rms_range_db` | >= 6.0 dB | warn |
| `seam_level_step` | `analyze_seams` | `level_step_db` | > 3.0 dB | warn |
| `seam_click` | `analyze_seams` | `click_db` | > 12.0 dB | warn |
| `seam_hole` | `analyze_seams` | `floor_dbfs` | < -50.0 dBFS | warn |
| `seam_frame_jump` | `analyze_seams` | `jump_ratio` | > 25.0 | info |
| `shot_dead_air` | `analyze_shots` | `dead_air_seconds` | > 0.4 s | warn |
| `sync_drift` | `analyze_sync_drift` | `end_offset_ms` | > 40.0 ms (magnitude) | warn |
| `sync_length` | `analyze_sync_drift` | `length_delta_ms` | > 40.0 ms (magnitude) | warn |

Guards beyond the threshold:

- `seam_hole` fires only while both sides of the seam are voiced above -30 dBFS
  (a quiet join between quiet shots is a pause, not a hole), and is skipped at
  a seam carrying `seam_fade_ms` or `crossfade_ms` (listed in `rules_skipped`
  with those seams).
- `seam_frame_jump` is skipped at a seam whose incoming shot is `hard_cut: true`
  or that got an `audio_bleed_ms` bleed.
- `shot_dead_air` is skipped inside a shot whose own RMS is at or below
  -30 dBFS, and for a shot record marked `silent_by_design: true` (#1162). The
  remedy for a gap it reports is a room-tone bed: `slice_audio` ->
  `loop_audio` -> `mix_audio`.

A `shots` record reaching past the file's own length is a separate finding,
`shot_span_overrun` (severity `warn`), on all three probes; the record is
clipped to the file before the rules above run.
`validate_workflow` reports the same mistake ahead of the run when the
video's length is already knowable (a `shots` argument against an
`asset:`/literal video); a `previous_result:`/`output:` video not yet
written is left to the finding.

`list_tasks` names the probes in their own `assessment` list, alongside
`commands`: exactly the commands declared `assessment=True`.

## Stem separation

Split a mix into its htdemucs stems, `vocals`, `drums`, `bass` and `other`,
each its own audio result. It is the separator `attribute_voices` runs (one
shared loader and cache). Transcribing the `vocals` stem aligns sung lyrics
better than the full mix, and `drums` + `bass` + `other` is an instrumental
bed.

```json
[
{
    "name": "stems",
    "task": {
        "command": "separate_stems",
        "arguments": { "audio": "asset:song.mp3" }
    },
    "result": { "content_type": "audio/wav", "file_base_name": "stems" }
},
{
    "name": "lyrics",
    "task": {
        "command": "transcribe_audio",
        "arguments": { "audio": "previous_result:stems.vocals" }
    },
    "result": { "content_type": "application/json" }
}
]
```

| Argument | Required | Description |
| -------- | -------- | ----------- |
| `audio` | Yes | An [audio input](#slice_audio) |
| `sample_rate` | With a waveform | Rate of a bare waveform |

The result saves one file per stem, `<file_base_name>-vocals`, `-drums`, `-bass`
and `-other`, and a later step reads one as `previous_result:<step>.vocals`.
Stems are 44.1 kHz stereo, whatever the source was, and sum back to the mix
(approximately). The model is fixed (`htdemucs`, fp32) and needs the `demucs`
package; its weights download on first use through demucs' own downloader,
outside the Hugging Face cache, so `downloads_required` does not list them. On
MPS a failed separation falls back to the CPU and warns
(`separation_cpu_fallback`).

**Dialogue and the rest.** On a dialogue shot with a score under it, `vocals`
is the dialogue and `drums` + `bass` + `other`, summed with `mix_audio`, the
rest. On H3 shots under a Music 3 score the `vocals` stem matched the dry shot
within 0.1 dB RMS, with the score's leak about 30 dB under
(record: `docs/proposals/audits/2026-11-dialogue-stems.md` in the repo). The rest is not a clean
score: a shot's SFX and foley land in `drums` and `other`, so keep the score
file when the goal is to replace the music. A speechless shot gives a
near-silent `vocals` stem and an invented transcript, so judge "no dialogue"
by the stem's level.

**Stacked vocals.** A chorus is often recorded as layered stereo doubles - every
Music 3 chorus is - and htdemucs returns all of them in the one `vocals` stem.
When the stem's left and right channels sing different lines over a sung span,
the step warns `stacked_vocals` and names the spans in m:ss (up to 8, then "and
N more"):

```
separate_stems: the vocals stem holds stacked (layered stereo) voices over
1:19–1:34, 1:59–2:09; treat those spans as more than one voice before
processing them as one
```

Pitch or formant work on a stacked span distorts it, and an `attribute_voices`
window or a lip-sync shot over one has two singers in it: pick a span outside
the flagged ones, or plan a two-shot. The stems are unchanged. The check is a 1 s
window test on the stereo stem: the window must be sung (half its frames voiced,
and within 14 dB of the stem's level), with an L/R correlation under 0.45 and a
side/mid energy ratio over 0.4. Windows merge across a one-window gap, and
spans under 2 s are dropped. A mono source, an instrumental and a single doubled
word never flag it. The thresholds are the `STACK_*` constants in
`dw/tasks/voice_attribution.py`.

## Voice attribution

Which reference voice sings each line of a song, by timbre (pitch cannot
tell a tenor from a mezzo): the section-to-singer map that staging lip-sync
shots needs and a generation model does not return.

`attribute_voices` separates the vocal stem (htdemucs), reduces every line and
every voice's reference to its voiced frames, embeds each with speechbrain's
ECAPA speaker encoder, scores every line against every voice by cosine, and
rolls lines up into named windows by the voiced seconds they overlap. It
decides nothing: the argmax is always reported with its margin and voiced
time, so a weak answer shows as weak. A validate's `downloads_required` lists
`speechbrain/spkrec-ecapa-voxceleb` only; htdemucs downloads outside the
Hugging Face cache.

```json
{
    "task": {
        "command": "attribute_voices",
        "arguments": {
            "audio": "asset:song.mp3",
            "voices": {
                "lena": [{"start_seconds": 4.0, "duration_seconds": 6.0}],
                "marcus": [{"start_seconds": 32.0, "duration_seconds": 6.0}]
            },
            "windows": [
                {"name": "shot-1", "start": 0.0, "end": 12.0},
                {"name": "shot-2", "start": 12.0, "end": 24.0},
                {"name": "shot-3", "start": 24.0, "end": 40.0}
            ]
        }
    },
    "result": { "content_type": "application/json" }
}
```

| Argument | Required | Description |
| -------- | -------- | ----------- |
| `audio` | Yes | The song, an [audio input](#slice_audio) |
| `voices` | Yes | Each voice's name mapped to its reference: a list of `{start_seconds, duration_seconds}` (or `{start, end}`) spans into `audio` (a single span may be given bare), or a separate clip as a path, `asset:` or `previous_result:` audio. At least 2 voices; names follow the variable-name pattern; each voice's reference must total at least `min_reference_seconds` |
| `lines` | No | Spans to attribute, `{start, end, text?}` (a `transcribe_audio` transcript, or its `chunks` list, drops in directly) or `{start_seconds, duration_seconds, text?}`. A zero-length line (`start` = `end`, as Whisper's word timestamps sometimes give) comes back with `voice: null` rather than refusing the call. Omitted: the song is cut into fixed windows of `window_seconds` |
| `windows` | No | Named spans to roll lines up into, `{name, start, end}`. Omitted: mirrors the fixed windows when `lines` is also omitted, otherwise none |
| `window_seconds` | No | Length of the fixed windows used without `lines` (default `2.0`) |
| `min_reference_seconds` | No | Least total reference length per voice; a shorter one is refused by name (default `3.0`) |
| `separate` | No | Isolate the vocal stem with htdemucs before embedding (default `true`); `false` for audio that is already a dry vocal |

A literal `voices` is checked at validation as well as on the step: fewer
than two voices, a bad name, a malformed span, or a span-list reference under
`min_reference_seconds` is refused at `steps[i].task.arguments.voices`, and a
bare path as a voice meets the same location policy as `audio`. What needs
the song itself - a span past its end, a clip's voiced length - is the
step's to refuse.

The result: `voices` (the names), `separated`, `duration_seconds`, `lines[]`
(`start`, `end`, `text`, `scores`, `voice`, `margin`, `voiced_seconds`,
`uncertain`, `reason`), `windows[]` (`name`, `start`, `end`, `voiced_seconds`,
`share`, `voice`, `uncertain`, `reason`), `reference_similarity` (pairwise
cosine between the voices' references), `voiced_floor_dbfs` (the floor this
song's stem was read against), `warnings` and the `thresholds` compared
against.

The voiced floor is relative to the stem, since a quiet sung verse can sit
20 dB or more under a loud chorus.

| Threshold | Value | What crossing it does |
| --------- | ----- | ---------------------- |
| `voiced_level_percentile` | 95 | The stem's level is this percentile of its 20 ms frames' rms |
| `voiced_floor_below_level_db` | 35.0 dB | A frame at or above the stem's level less this is voiced |
| `voiced_floor_min_dbfs` | -60.0 dBFS | The floor never drops below this, however quiet the stem |
| `min_voiced_seconds` | 0.5 s | A line or window under this much voiced time has no voice - `voice: null`, `uncertain: true` |
| `uncertain_margin` | 0.05 | A line whose best score beats the runner-up by less than this is `uncertain` - the argmax is still reported |
| `uncertain_share_margin` | 0.2 | A window whose leading voice's share beats the runner-up's by less than this is `uncertain` - a duet line, or a window straddling a hand-over |
| `piece_seconds` | 2.0 s | A line longer than this is also scored in pieces of this length |
| `mixed_line_share` | 0.25 | A line whose confidently attributed pieces give a second voice at least this share of their voiced time is `uncertain`, and its `reason` names each voice's seconds - one embedding of two singers can name either of them with a wide margin |
| `voices_too_similar` | 0.8 | Two references scoring above this against each other make every line between them a weak answer whatever its scores say |

A `voices_too_similar` pair is reported in `reference_similarity` and as a job
warning; pick better-separated reference spans.

Both models are fixed (no model-name argument) and run in fp32. Weights
download on first use (htdemucs from `dl.fbaipublicfiles.com` into the torch
hub cache, ECAPA from speechbrain) and are cached between calls. On MPS a
failed htdemucs pass falls back to the CPU and warns
(`separation_cpu_fallback`).

**Example:** [attribute-lines.json](../workflows/templates/attribute-lines.json) — Which singer sings each line of a song, and when: `transcribe_audio` with timestamps, its transcript handed whole to `attribute_voices` as `lines`.

### Checking the lip-sync target

A sung multi-shot piece can put the right song on the wrong mouth: the shot
lip-syncs whoever the prompt or reference made most prominent. This check lines
each sung line up with the picture and looks at whose mouth is open on it.

1. Run `attribute-lines` on the song with each singer's reference spans as
   `voices`. Every line comes back with numeric `start` and `end` seconds into
   the song, its `voice`, `margin` and `uncertain` flag.
2. Map each line's song times onto the cut's timeline (the rules below).
3. For every line with a non-null `voice`, take two moments: `start + 0.3 s`,
   where the mouth has opened on the line's first syllable, and the line's
   midpoint.
4. Make one `get_output_frames` call per face: `at` the moments, `crop` the
   face's region, `hear: 1.0`. At most 32 moments per call, so a long song
   takes more than one call per face.
5. On each line, the singer `voice` names should be the face with an open
   mouth. Another face singing it is the wrong lip-sync target; both mouths
   open on a solo line is a shot that lip-syncs everyone.
6. Skip the lines that are `uncertain` or have a null `voice`, and name them
   in the report - a skipped line is unchecked, not passed.

**Song time to cut time.** `attribute-lines` answers in seconds into the song;
the frames are read from the cut. The offset depends on the assembly:

- `minimax/music-video`: the template re-slices the song at each entry's
  `start_frame` for `cut_frames`, joins those pieces with `crossfade_audio` and
  lays them under the edit with `pair_audio` at `fit: video`. Song time is cut
  time only when the entries start at frame 0, leave no gaps and are in song
  order; otherwise map each line through its entry's `start_frame`.
- `join_into_song`: add `start_frame / fps − cue_seconds` to every line time,
  where `start_frame` is the first sung shot's, from the step's shots output.
  That shot's frame 0 is where song time `cue_seconds` lands.
- Any other assembly: transcribe the final cut's own track instead (weaker
  attribution, since ducked dialogue is in the mix too).

**Where it misleads:**

- A duet line (two singers in one Whisper segment) comes back `uncertain`, its
  `reason` giving each voice's seconds. `timestamps: "word"` attributes word by
  word, with noisier scores.
- When accompaniment drowns the words, Whisper can return one hallucinated
  segment over most of a song; re-run with `timestamps: "word"`, or transcribe
  the `vocals` stem of [`separate_stems`](#stem-separation).
- `start + 0.3 s` is a heuristic (a singer may open late), so trust it together
  with the midpoint frame, not alone.

## Data Gathering

### gather_images

Load images from URLs and/or a file glob:

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

A `urls` entry may also be an `asset:` or `output:` reference (or a local path):
it loads from disk under the same read policy as a `glob` match, so a template
whose `urls` is `["variable:image_url"]` takes an uploaded image as well as a URL.

### gather_videos

As `gather_images` (`glob`, `urls`), for video files. Each video is one
artifact holding its frames and muxed audio. To *join* videos on disk, give
their paths to `concat_videos` directly: a `previous_result` reference to a
gather step fans the consuming step out over the videos.

### gather_inputs

Returns its whole `arguments` dict unchanged; it takes any keys. Useful for
organizing data flow.

### select

Pick one candidate out of many by a fixed rule, with no model. With `judge`
it makes best-of-N:

```json
{
    "name": "pick",
    "task": {
        "command": "select",
        "arguments": {
            "candidates": ["gather:still"],
            "scores": ["gather:judge"],
            "rule": "argmax"
        }
    },
    "result": { "content_type": "image/jpeg" }
}
```

| Argument | Required | Description |
| -------- | -------- | ----------- |
| `candidates` | Yes | The values to choose among, in order |
| `scores` | Yes | One number (or numeric string) per candidate |
| `rule` | Yes | `argmax`, `argmin`, `first_above` (first score at or above `threshold`), `first_below` (at or below) or `index` |
| `threshold` | For `first_above` / `first_below` | The score to compare against |
| `index` | For `index` | A whole-number position into `candidates` |

The step's result is the winning value; its position and score are recorded in
the manifest and the step's end event. Refused, at validation where literal and
at run time: candidates and scores of different lengths, a non-numeric score,
an unknown rule, a missing `threshold` or `index`, an `index` out of range, and
(at run time) no candidate passing a threshold rule.

**Example:** [best-of-n-to-video.json](../workflows/templates/best-of-n-to-video.json)

## Videos in image tasks

Every image command - the upscalers, face restoration, segmentation, object
removal and the image processors - takes a video where it takes an image: an `AudioVideo` from
a generation, `concat_videos` or `dissolve_videos` step, or a frame array from
`video_frames`. The command runs over the frames one at a time and returns one
video artifact, its soundtrack carried through untouched, so a generated clip
can be upscaled without losing what was generated alongside it:

```json
{
    "task": {
        "command": "upscale",
        "arguments": {
            "image": "previous_result:generate_video",
            "model_name": "Kim2091/UltraSharp"
        }
    },
    "result": { "content_type": "video/mp4", "fps": 24 }
}
```

Captioning (`image_to_text`) is the exception - describe a frame, taken with
`get_first_frame`, rather than a video.

A file path or `asset:`/`output:` reference to a video is not accepted here
(`grade`, whose input is `media`, is the exception), even though the route above takes a video from a step - route it through
`video_frames` first: an image command's `image` argument must be an
`AudioVideo`, a frame array, or a still image.

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
| `image` | Yes | PIL Image or `previous_result:` reference - a video runs frame by frame, see [Videos in image tasks](#videos-in-image-tasks) |
| `model_name` | Yes | HuggingFace repo ID or local file path |
| `filename` | No | Specific weight file in a HF repo (auto-detected if only one) |
| `tile_size` | No | Tile size (default: 512) |
| `tile_overlap` | No | Overlap between tiles in pixels (default: 32) |

A validate's `downloads_required` lists the `model_name` / `filename` file when it is not cached, or the whole repo when no `filename` is given (such a repo must hold exactly one weight file); a `previous_result:` or local-path model gives none.

Images over 512x512 pixels are tiled. Models load from Hugging Face Hub repos (`.pth`, `.safetensors`, `.pt`, `.ckpt`) or local files.

**Example:** [upscale-spandrel.json](../workflows/templates/upscale-spandrel.json) — upscale an image you already have 4x; the input is a path or URL.

## Diffusion Upscaling

Text-guided upscaling with Stable Diffusion upscale pipelines, with better detail recovery than super-resolution on faces and textures.

Two modes are available:
- **x4** (default): `StableDiffusionUpscalePipeline` — 4x upscale via `stabilityai/stable-diffusion-x4-upscaler`
- **x2**: `StableDiffusionLatentUpscalePipeline` — 2x upscale via `stabilityai/sd-x2-latent-upscaler`

A validate's `downloads_required` lists the repo for the `mode` given; `model_name` overrides it.

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
| `image` | Yes | PIL Image or `previous_result:` reference - a video runs frame by frame, see [Videos in image tasks](#videos-in-image-tasks) |
| `prompt` | No | Text guidance for upscaling (default: "") |
| `negative_prompt` | No | Negative text guidance (default: none) |
| `mode` | No | `"x4"` or `"x2"` (default: `"x4"`) |
| `model_name` | No | Override the default model for the selected mode |
| `num_inference_steps` | No | Denoising steps (default: 25) |
| `guidance_scale` | No | Classifier-free guidance scale (default: 9.0) |
| `noise_level` | No | Noise level for x4 mode (default: 20, ignored for x2) |

**Example:** [upscale-diffusion.json](../workflows/templates/upscale-diffusion.json) — prompt-guided upscale of an image you already have; `mode` selects `x4` (the default) or `x2`.

## H3 Latent Upscaling

Two tasks promote a MiniMax-H3 take in latent space; the workflow guide's
[Promoting an H3 take](WORKFLOW_GUIDE.md#promoting-an-h3-take-to-768p-in-latent-space-upscale_h3_latents-and-decode_h3_latents)
section has the full recipe. A third, `encode_h3_latents`, goes the other way
for the experimental windowed refine of a long take
([Refining a long take in time windows](WORKFLOW_GUIDE.md#experimental-refining-a-long-take-in-time-windows)).

### upscale_h3_latents

Resize H3 video latents `(B, 24, T, H, W)` to a larger plate without denoising
again, as the input to a tiled refine. Upscale-only output above 768p is soft.

| Argument | Required | Description |
| -------- | -------- | ----------- |
| `latents` | Yes | H3 latents, e.g. `previous_result:base.latents` |
| `width` / `height` | Yes | The target size: multiples of 16, 1x to 4x the latents' own size per axis, at most 1920x1088 (or 1088x1920 portrait) |
| `model_name` | No | Hub repo id, not a path (default `LBH-123-AI/Minimax_h3_latent_Upscaler`, at a pinned revision). `Alissonerdx/Minimax-H3-ComfyUI` with `h3_upscaler_lms_v0.1.safetensors` is a sharpness-tuned alternative |
| `weight_name` | No | A bare `.safetensors` name (default `minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors`, read from the default repo's `minimax_h3_latent_upscaler_3d_conv_v1/` folder) |

Returns latents `(B, 24, T, height/16, width/16)` in the input's dtype and
device (bf16 on GPU, fp32 on CPU). A size that breaks the rules above is
refused, at validation for literal values; so is an input that is not H3
latents. `downloads_required` lists the upscaler file.

### decode_h3_latents

Decode H3 latents to frames through diffusers' `MiniMaxH3VideoDecodeStep` (VAE
in fp32), so the colours match the pipeline's own decode. Takes `latents` (a
pipeline step's `latents` or an `upscale_h3_latents` result), `model_name`
(default `MiniMaxAI/MiniMax-H3`, its `vae` subfolder) and `num_frames` (keep
only the first that many frames: an `encode_h3_latents` step's `source_frames`,
so a padded take comes back at its own length). Returns one video of the
first batch item at 24 fps with no audio; `pair_audio` puts a soundtrack back.
Latents too short to decode to any frame (one or two latent frames) are
refused, and so is a `num_frames` above what the latents decode to.
It declares no `required_weights`: the VAE is the one the H3 pipeline step
already cached.

### encode_h3_latents

Experimental. Encode a finished take into one H3 video latent, one 17-frame VAE
clip at a time, so a take too long to encode whole (a `source_video` refine
encodes it as one float32 tensor) still reaches a refine. The result is
bit-identical to the whole-take encode.

| Argument | Required | Description |
| -------- | -------- | ----------- |
| `source` | Yes | The take: an `output:`/`asset:` video file, read a clip's range at a time, or an earlier step's video. 24 fps, both sides a multiple of 32 |
| `model_name` | No | The MiniMax-H3 repo whose `vae` subfolder encodes (default `MiniMaxAI/MiniMax-H3`) |

Returns `latents` (the refine's `latents` argument), `num_frames` (the take
padded with its last frame up to the next 17m + 5, the refine's `num_frames`)
and `source_frames` (the take's own count, for `decode_h3_latents`'
`num_frames`). Refused: a take that is not 24 fps, a side off the 32 grid, an
empty take, and one over 30 minutes. A file header that claims far more frames
than its duration holds is refused; one that claims a few more (a
variable-frame-rate take) is encoded at the duration's count, with a warning.

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
| `image` | Yes | PIL Image or `previous_result:` reference - a video runs frame by frame, see [Videos in image tasks](#videos-in-image-tasks) |
| `model_name` | Yes | HuggingFace repo ID or local file path |
| `filename` | No | Specific weight file in a HF repo (auto-detected if only one) |
| `upscale_factor` | No | Background upscale factor (default: 1, no upscaling) |
| `face_size` | No | Cropped face size in pixels (default: 512) |
| `use_parse` | No | Use face parsing for better blending (default: true) |
| `only_center_face` | No | Only restore the largest/center face (default: false) |
| `detection_resize` | No | Resize shorter side for detection speed (default: 640) |
| `eye_dist_threshold` | No | Skip faces with eye distance below this (default: 5) |
| `upsample_img` | No | Pre-upscaled background image (e.g., from a prior upscale step). Single-image input only: refused with a video |

A validate's `downloads_required` lists the `model_name` / `filename` file when it is not cached (the whole repo when no `filename` is given), and facexlib's two weights as URL entries while they are missing from facexlib's weights directory: `detection_Resnet50_Final.pth` always and `parsing_parsenet.pth` unless `use_parse` is false. Those still download outside the Hugging Face cache.

Models are loaded via spandrel, so any `.pth`/`.safetensors` face restoration weights work. CodeFormer requires `pip install spandrel-extra-arches` (non-commercial license). The detector and parser are loaded once per settings/device and reused across frames; `release_models` drops them with the restorer.

**Example:** [restore-faces.json](../workflows/templates/restore-faces.json) — Generate a portrait, then restore faces with GFPGAN v1.4.

### Combining with Upscaling

Upscale the background, then paste restored faces onto the upscaled image:

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

## Object Segmentation

Detect and segment objects using text prompts via GroundingDINO + SAM2. Returns a binary mask image suitable for inpainting workflows.

A validate's `downloads_required` lists `IDEA-Research/grounding-dino-base` and `facebook/sam2-hiera-large`; `model_name` and `sam_model_name` override them.

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
| `image` | Yes | PIL Image or `previous_result:` reference - a video runs frame by frame, see [Videos in image tasks](#videos-in-image-tasks) |
| `prompt` | Yes | Text description of object(s) to detect (e.g., "dog", "red car") |
| `model_name` | No | GroundingDINO model ID (default: `IDEA-Research/grounding-dino-base`) |
| `sam_model_name` | No | SAM2 model ID (default: `facebook/sam2-hiera-large`) |
| `threshold` | No | Detection confidence threshold (default: 0.3) |
| `invert` | No | Invert the output mask (default: false) |

Returns a grayscale PIL Image (mode "L"): white (255) for detected objects, black (0) for background. Of SAM2's three proposals per detected box, segment keeps the one SAM2 rates best, usually the whole object. To erase what it found, pass the mask to [`remove_object`](#object-removal), which chains with no `invert`. To paint something new in its place, pass it to a prompted inpainter such as FluxFillPipeline.

**Examples:**

- [segment.json](../workflows/templates/segment.json) — Segment an object from an image
- [segment-and-remove.json](../workflows/templates/segment-and-remove.json) — Segment, then erase the object with `remove_object` (no prompt)
- [segment-and-inpaint.json](../workflows/templates/segment-and-inpaint.json) — Segment, then repaint the masked region with a prompt

## Object Removal

Erase what a mask covers and fill the hole from the surrounding texture, with no prompt (#956). `remove_object` runs [LaMa](https://github.com/advimman/lama) (big-lama) through spandrel: no diffusion, so a warm run takes well under a second on a GPU, and it runs on CUDA, MPS and CPU.

```json
{
    "task": {
        "command": "remove_object",
        "arguments": {
            "image": "asset:photo.png",
            "mask_image": "previous_result:mask",
            "dilate": 8
        }
    },
    "result": { "content_type": "image/png" }
}
```

| Argument | Required | Description |
| -------- | -------- | ----------- |
| `image` | Yes | PIL Image or reference - a video runs frame by frame, see [Videos in image tasks](#videos-in-image-tasks) |
| `mask_image` | Yes | White where to erase, black where to keep |
| `dilate` | No | Pixels to grow the mask by before the fill, a whole number 0 or above (default: 8) |
| `model_name` | No | Hugging Face repo id of the weights, never a local path (default: `michaelgold/big-lama`, at a pinned revision) |
| `weight_name` | No | The `.safetensors` file in that repo (default: `big-lama.safetensors`) |

**The mask.** White means fill, the polarity [`segment`](#object-segmentation) returns, so `segment` → `remove_object` chains with no `invert`. The mask is converted to grayscale and binarized at 128 (at or above is fill). A mask of a different size from the image is resized to it (nearest). An all-black mask, as `segment` returns when it finds nothing, gives the image back unchanged with a warning in the job's events, and loads no model.

**`dilate`.** LaMa fills cleanly only when the mask overshoots the object's edge, and a `segment` mask is tight to it. The default 8 px suits a segmented object. Raise it (16 to 24) for a soft edge, a shadow or a glow around the object. Use 0 only for a hand-drawn mask that already has a margin.

**The output** is the input's size. Only the region inside the dilated mask comes from the model: every pixel outside it is the input's own, unchanged.

**Video.** Frames (e.g. from `video_frames`) are cleaned one at a time with the same still mask, and come back at the same count and fps, with the audio kept. There is no temporal model, so a fill can flicker from frame to frame, and the mask cannot follow a moving object.

**Limits.** LaMa suits small to medium objects: a cup, a sign, a watermark, a passer-by in the distance. A large region, such as a person filling a third of the frame, comes back smeared. For that, a prompted inpainter whose prompt names the background is the remaining route, and it may paint a patch that does not match (#946).

**Weights.** `michaelgold/big-lama`, file `big-lama.safetensors` (about 200 MB), read at revision `98ab7cface89a594b381f32fc7a9f2cae3f91ccc` for the default pair (another `model_name` or `weight_name` is read at `main`); `downloads_required` lists it as a file entry while it is not cached. `biali/big-lama` holds a byte-identical copy and is not tried automatically. LaMa (Suvorov et al.) is licensed **Apache-2.0**, whatever the Hub card says. The task reads safetensors only: a `weight_name` that is not `.safetensors` (the common `big-lama.pt` is TorchScript) and a `model_name` that is a local path are refused at validation and at run time, as are weights that do not load as an inpainting model. It runs in fp32 on every device.

**Examples:**

- [remove-object.json](../workflows/templates/remove-object.json) — Erase what a supplied mask covers
- [segment-and-remove.json](../workflows/templates/segment-and-remove.json) — Find an object by name, then erase it

## Image Captioning

Caption an image with a vision-language model, through transformers'
`image-text-to-text` pipeline: `model_name` must be a VLM (SmolVLM, Qwen2.5-VL,
LLaVA, ...) and `prompt` is a question put to it. BLIP/ViT-GPT2/GIT captioners
do not load (transformers 5 removed them).

A validate's `downloads_required` lists `HuggingFaceTB/SmolVLM-256M-Instruct` for `image_to_text` and `judge`; `model_name` overrides it.

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

The default model is tiny and writes short, plain captions; point `model_name` at something larger for detail. Returns a caption string: save as `text/plain`, or pass on via `previous_result:` as a prompt.

For a detailed caption, hand the image to [`text_generation`](#text-generation--prompt-expansion) with a question and a larger vision-language model; that is what [describe-and-regenerate.json](../workflows/templates/describe-and-regenerate.json) does ahead of its prompt expansion.

**Examples:**

- [image-to-text.json](../workflows/templates/image-to-text.json) — captioning with the default model, saved as `.txt`; set `model_name` to a larger VLM and ask a question
- [describe-and-regenerate.json](../workflows/templates/describe-and-regenerate.json) — Describe an image, expand the caption, then regenerate it

### judge

Score an image against a rubric with a vision-language model, for `select` to
pick from. Returns a bare number, so the step takes no `result` block.

| Argument | Required | Description |
| -------- | -------- | ----------- |
| `image` | Yes | An image (or a list, scored as one prompt) |
| `rubric` | Yes | The question or criterion |
| `scale` | Yes | `[low, high]`; the prompt asks for one number in that range |
| `model_name` | No | A VLM (default `HuggingFaceTB/SmolVLM-256M-Instruct`) |
| `max_new_tokens` | No | Default `20` |

Decoding is greedy, so a score reproduces. The first number in the reply is
the score, not clamped to `scale`; a reply with no number is refused ("could
not parse a score"). **Example:** [best-of-n-to-video.json](../workflows/templates/best-of-n-to-video.json)

## Composing Text

Assemble one block of text out of parts written once, such as a character
bible repeated in every shot's prompt. The engine has no string interpolation
(see the no-interpolation rule in [the workflow guide](WORKFLOW_GUIDE.md)): a
part is a *whole* value, and `compose_text` joins parts in order.

```json
[
{
    "name": "shot_1_prompt",
    "task": {
        "command": "compose_text",
        "arguments": {
            "parts": [
                "variable:character_a_bible",
                "variable:character_a_voice",
                "variable:shot_1_action"
            ],
            "separator": "\n\n"
        }
    }
},
{
    "name": "shot_1",
    "pipeline": { "arguments": { "prompt": "previous_result:shot_1_prompt" } }
}
]
```

| Argument | Required | Description |
| -------- | -------- | ----------- |
| `parts` | Yes | The parts to join, in order — each a whole value, usually a `variable:`, `prompt:` or `previous_result:` reference. Numbers are written out; `null` is dropped, so an optional part can be a variable left null |
| `separator` | No | What goes between the parts (default: a blank line, the paragraph break the prompt formats use) |
| `skip_empty` | No | Drop parts that are null or blank (default `true`). With it off, an empty part still contributes its separator |

Refused: a `parts` that is not a list; a part that is neither text nor a
number (the reference resolved to something other than text); and a list left
with nothing once empty parts are dropped. The parts are positional; there is
no named form.

## Splitting Text into Takes

`split_takes` cuts a long text into takes a speech model can say in one go
(`dw/tasks/text_takes.py`). Audio8-TTS holds about 95 s of reference, text and
speech together, but a take over about 40 s of speech loops, skips or stops
early (`bench/records/audio.md`), so a long narration is spoken take by take
and joined (`templates/narration-long`).

```json
{
    "name": "split",
    "task": {
        "command": "split_takes",
        "arguments": { "text": "variable:text", "max_seconds": 37, "words_per_second": 3.0 }
    },
    "result": { "content_type": "application/json" }
}
```

| Argument | Required | Description |
| -------- | -------- | ----------- |
| `text` | Yes | The text to speak |
| `max_seconds` | No | The longest a take may run (default 37: at 3.0 words a second, 111 words, the longest voice-clone take measured word-complete) |
| `words_per_second` | No | The rate that turns the limit into a word count (default 3.0, the voice-clone voice's measured rate; written dialogue measured 2.5 to 4 words a second as filling a clip) |

It answers `{takes, count, words}`. A take ends at a sentence boundary; a
sentence that alone is over the limit is cut at its last comma before the
limit, else (no comma) at the word boundary nearest the limit, and the run warns.
A take already 60% of the way to the limit ends at a paragraph break (a blank
line), so the join's pause falls where the text has one. The takes rejoin to the text, modulo whitespace.
The engine expands `for_each` before a run, so the list cannot feed a
`for_each` in the same workflow. `templates/narration-long` takes the plain
text, runs this task, and hands the takes to `templates/narration-from-takes`
as a child workflow, which expands its own `for_each`. Refused: empty text, a limit that is not
above zero.

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

A label counts only at the start of a line, in any capitalisation, and is
written back in the declared spelling. A section runs from its `label:` to the
end of that paragraph, so a blank line ends one and a single newline does not.
Repeats are dropped, missing sections are skipped, and text with no recognised
label is returned trimmed but otherwise as it is. Use it on a model that
produces a rigid format and then keeps going.

The built-in `h3_context_ir` workflow applies this to its own output, so a workflow delegating to it receives only the fields MiniMax H3 expects.

## Text Generation / Prompt Expansion

Generate or expand text with a local language model, such as a short prompt expanded into a detailed one.

A validate's `downloads_required` lists `Qwen/Qwen2.5-1.5B-Instruct`, or `HuggingFaceTB/SmolVLM-256M-Instruct` when an `image` is given; `model_name` overrides it. A `system_prompt.from_hub` file is not listed.

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
| `generate_kwargs` | No | Anything else to pass to the model's `generate()` — `no_repeat_ngram_size`, `top_p`, `min_new_tokens`, `stop_strings`. Merged last, so it overrides the settings above. Filtered like `chat_template_kwargs` — see below |
| `max_new_tokens` | No | Maximum tokens to generate (default: 500) |
| `chat_template_kwargs` | No | Extra keyword arguments for the model's chat template (for a template that takes switches such as a thinking toggle). Values must be scalars (boolean, number, string or null), and the generation call's own keys (`images`, `tools`, `max_new_tokens`, `tokenizer_encode_kwargs`, ...) are refused |
| `revision` | No | Model revision (branch, tag or commit) to load `model_name` at |

`image` may also be a list of images, each sent as its own picture in the one user message. A list is one prompt over all the images, in order, and returns one answer — not one answer per image. The same holds for `judge` and `image_to_text`, which hand their `image` to this task. An empty list, or one holding only nulls, is no image.

`system_prompt` may name a file in a Hugging Face repo instead of carrying the text,
for a vendor that ships its prompt instructions there:
`{"from_hub": {"repo": "owner/name", "file": "system_prompts.json", "revision": "<40-hex commit>", "key": "t2i"}}`.
The revision must be a full commit, so the text can't move under the workflow. A `.json`
file takes `key` to pick one string; any other file is read whole. Only public repos are
reachable (the download never uses the server's Hugging Face token), and a file over 1 MiB
is refused. The Hub is asked, without credentials, whether the file is public before every
use, so `from_hub` needs network access even when the file is already cached.

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

Ahead of an image-conditioned generation step, which pins the picture as the
first frame, pass the same image to both so the prompt agrees with the frame.
A vision model is worth releasing before the generation model loads
(`release_models` in the workflow guide).

Generation is greedy so a workflow reproduces. Greedy decoding against a long,
rigid format makes these models loop, so the vision path applies a
`repetition_penalty` of 1.15 (on Qwen3-VL against the MiniMax H3 spec, 1.05
still looped; 1.15 stopped on its own). Raise it if a model still repeats
itself, or set `1.0` to disable; trim the output with
[`extract_sections`](#extracting-sections) to keep only the fields asked for.
Qwen3-VL-4B and 8B do not produce the H3 spec's `<d>[Language]...</d>` dialogue
tag or `(S1)` speaker ids; their visual description is usable, but write
prompts by hand where a subject has to speak.

`generate_kwargs` goes straight to `generate()`, merged last, so it overrides
`repetition_penalty` and the sampling settings:

```json
{
    "arguments": {
        "prompt": "a cat on a windowsill",
        "generate_kwargs": { "no_repeat_ngram_size": 25 }
    }
}
```

It is checked before the model loads, as `chat_template_kwargs` is. Values must be scalars or lists of scalars (`stop_strings` is a list), and the call's own keys are refused (`max_new_tokens`, `tokenizer_encode_kwargs`, `prefix`, `handle_long_generation` and the rest of the pipeline's named arguments). `custom_generate` and `trust_remote_code` need trusted workflows (`--trust-workflows`): with them, transformers downloads a repo's `custom_generate/generate.py` and runs it in place of its own decoding loop.

**Examples:**

- [expand-prompt.json](../workflows/templates/expand-prompt.json) — Expand a short prompt and save as `.txt`; add a pipeline step to generate from it

## Speech Generation

Speak a line of text with a local text-to-speech model. A validate's `downloads_required` lists `suno/bark-small` (`model_name` overrides it), plus `speechbrain/spkrec-xvect-voxceleb` when `speaker_embedding` is given. Any transformers text-to-speech checkpoint can be named as `model_name`. Three transformers families are wired for their voice controls, and Audio8-TTS has its own vendored branch:

| Family | Checkpoint | Voice |
|--------|-----------|-------|
| Bark | `suno/bark-small` (the command's default) | Distinct speakers by `voice_preset` |
| MMS / VITS | `facebook/mms-tts-eng`, and `mms-tts-<code>` per language (the template's default) | One voice, a quarter Bark's size; a multi-speaker VITS takes `speaker_id` in `forward_params` |
| SpeechT5 | `microsoft/speecht5_tts` | The voice of a reference clip, given as `speaker_embedding` |
| Audio8-TTS | `Edge0/Audio8-TTS-Realistic-0.6b` | A cloned voice from `reference_audio` + `reference_text`, or its default voice; 44.1 kHz mono, many languages from one checkpoint |

The result is a waveform carrying the rate its model generated at, so it composes with [`slice_audio`](#slice_audio), [`fade_audio`](#fade_audio) and [`pair_audio`](#pair_audio) directly ([`concat_videos`](#concat_videos) and `dissolve_videos` join videos — pair the track onto a video first).

```json
{
    "task": {
        "command": "generate_speech",
        "arguments": {
            "text": "The way ahead is longer still.",
            "voice_preset": "v2/en_speaker_6"
        }
    },
    "result": { "content_type": "audio/wav" }
}
```

| Argument | Required | Description |
| -------- | -------- | ----------- |
| `text` | One of `text`/`messages` | The line to speak |
| `messages` | One of `text`/`messages` | Chat input for a model that takes a conversation (a list of `{"role": ..., "content": ...}` dicts), passed through as the pipeline's `text_inputs` so the model's chat template applies. A model with no chat template (Bark) raises |
| `model_name` | No | HuggingFace model ID (default: `suno/bark-small`) |
| `voice_preset` | No | The speaker, for a model with presets (`v2/en_speaker_0` through `v2/en_speaker_9` for Bark). A preprocessing argument: passed through `forward_params` it would be dropped. A single-voice model with no processor (`facebook/mms-tts-eng`) refuses it |
| `speaker_embedding` | No | A reference audio file (typically `asset:`) whose voice a SpeechT5 model speaks in, reduced to an x-vector with speechbrain's `spkrec-xvect-voxceleb` and passed as `speaker_embeddings`. Any other model refuses it |
| `reference_audio` | No | Audio8-TTS only: a clip of the voice to clone (typically an `asset:` reference), up to 20 s; 5-10 s of clean speech is plenty. Needs `reference_text` |
| `reference_text` | No | Audio8-TTS only: the words spoken in `reference_audio`, verbatim. Needs `reference_audio` |
| `forward_params` | No | Passed to the model's forward/generate call; a multi-speaker VITS takes its `speaker_id` int here |
| `generate_kwargs` | No | Ad-hoc generation settings for a generative model — `temperature`, `do_sample` |

The command defaults to Bark for its distinct speakers; [`generate-speech.json`](../workflows/templates/generate-speech.json) defaults to `facebook/mms-tts-eng`, a quarter the size, for a single voice.

The result needs no `sample_rate`: the track carries its model's rate. A declared `result.sample_rate` still wins, and one that does not match plays the speech at the wrong speed and pitch; the run succeeds with a `rate_override_mismatch` warning.

### Cloning a voice (Audio8-TTS)

`model_name: "Edge0/Audio8-TTS-Realistic-0.6b"` takes its own branch. The model is vendored (`dw/vendored/audio8_tts_model.py`), so no code loads from its repo; its files (`config.json`, `generation_config.json`, `model.safetensors`, `tokenizer.json`, the codec `codec.pth`) are fetched at one pinned revision, and `downloads_required` lists all five. The result is a 44.1 kHz mono track.

With `reference_audio` and `reference_text` together it speaks `text` in the reference's voice; with neither, in its default voice. The language follows `text`. Every refusal fires before the model loads:

- `reference_audio` without `reference_text` (an empty one counts as missing), or the reverse;
- either one on any other model;
- `speaker_embedding`, `voice_preset` or `messages` on Audio8;
- a reference longer than 20 s;
- text that, with the reference, won't fit the model's 2,048-position context. It is refused, never truncated: split the line;
- a `generate_kwargs` / `forward_params` key outside `temperature`, `top_p`, `top_k`, `do_sample`, `max_new_tokens`. `max_new_tokens` defaults to the room left in the context.

The fit check is an estimate; a line that passes it but runs out of room is refused after generation rather than returned cut off. The same seed, text and reference give the same audio.

[`voice-clone.json`](../workflows/templates/voice-clone.json) is the template. **Clone only a voice you have the speaker's consent to use.**

### Generating a voice to condition on

Generated speech is best used as a voice *timbre reference*, not as the track
a mouth follows. A `MiniMaxH3AudioReference` is audio H3 reads, not plays: a
few seconds of a voice fix timbre, pitch and delivery while H3 still generates
the line, and a mouth follows a whole track given only as a reference loosely,
sometimes shifted by whole beats (#795). To keep a track exact, hold it with
`hold_audio` (workflow guide, "H3: generating to a held soundtrack");
`music-video` holds each slice by default (#808). Build the reference with
`from_previous_result`, and the clip's sample rate comes with it:

```json
{
    "references": [
        {
            "reference_type": "diffusers.modular_pipelines.minimax_h3.MiniMaxH3AudioReference",
            "from_previous_result": "voice"
        }
    ]
}
```

The same preset referenced in every shot of a scene makes a character's voice
a conditioning signal rather than a prose description. Narration over shots
nothing lip-syncs to is muxed with [`pair_audio`](#pair_audio). Set
`release_models` on the speech step before a video model loads.

**Examples:**

- [generate-speech.json](../workflows/templates/generate-speech.json) — Speak a line and save it as a `.wav`
- [voice-clone.json](../workflows/templates/voice-clone.json) — Speak a line in the voice of a reference clip (Audio8-TTS)
- [voice-timbre-reference.json](../workflows/templates/minimax/voice-timbre-reference.json) — Generate a voice, then condition H3's `<Audio 1>` on it

## Sound Effects Generation

`generate_sound_effects` makes sound effects with MMAudio (large 44.1 kHz v2) from a video clip, a text prompt, or both, as a 44.1 kHz mono track. A validate's `downloads_required` lists the pinned weights: `hkchengrex/MMAudio` (the network `weights/mmaudio_large_44k_v2.pth` and Synchformer `ext_weights/synchformer_state_dict.pth`), the `audio_vae/` and `vocoder/` config and safetensors of `kandinskylab/Kandinsky-6.0-Lite-distill-5s-Diffusers`, and the weights and tokenizer of `apple/DFN5B-CLIP-ViT-H-14-384`. MMAudio's own `v1-44.pth` and `bigvgan_generator.pt` are never downloaded.

```json
{
    "task": {
        "command": "generate_sound_effects",
        "arguments": {
            "video": "asset:shot.mp4",
            "prompt": "footsteps on gravel, a door creaking",
            "negative_prompt": "music, speech"
        }
    },
    "result": { "content_type": "audio/wav" }
}
```

| Argument | Required | Description |
| -------- | -------- | ----------- |
| `video` | One of `video`/`prompt` | The clip the effects follow (typically an `asset:` or `previous_result:` reference) |
| `prompt` | One of `video`/`prompt` | What should be heard |
| `negative_prompt` | No | What to steer away from, such as `music, speech` |
| `duration` | No | Seconds. Defaults to the clip's length (capped at 20 s), or 8 s text-only; a given value must be at least 0.64 s (16 sync frames at 25 fps), at most 20 s and no longer than the clip |
| `steps` | No | Sampler steps (default: 25) |
| `guidance_scale` | No | Classifier-free guidance (default: 4.5) |

Video-only is the weaker mode; give a `prompt` naming the on-screen sound sources too. A `duration` outside 4-12 s (the model trained around 8 s) warns, as does noise-like output (spectral flatness of 0.3 or more with a centroid of 8 kHz or more).

Every refusal fires before anything loads: neither `video` nor `prompt`; a `duration` under 0.64 s, over 20 s or longer than the clip; an unknown argument; and `trust_remote_code`, `custom_pipeline` or `custom_generate` (the trust gate).

The output runs 7-25 dB hot. Route it through `match_levels` before mixing it with dialogue or music.

**Licences.** The MMAudio weights (network, Synchformer and the VAE tensors) are CC-BY-NC 4.0: non-commercial use only. The DFN5B CLIP encoder is under the Apple Machine Learning Research Model License Agreement: research purposes only. See [ATTRIBUTIONS.md](ATTRIBUTIONS.md).

**Examples:**

- [video-to-sound-effects.json](../workflows/templates/video-to-sound-effects.json) — Effects that follow a clip's picture
- [text-to-sound-effect.json](../workflows/templates/text-to-sound-effect.json) — An effect from a text prompt alone

## Speech Transcription

A validate's `downloads_required` lists `openai/whisper-base`; `model_name` overrides it. `check_script` uses the same model.

Transcribe spoken audio to text with a local Whisper-class model.

```json
{
    "task": {
        "command": "transcribe_audio",
        "arguments": {
            "audio": "previous_result:speak"
        }
    },
    "result": { "content_type": "text/plain" }
}
```

| Argument | Required | Description |
| -------- | -------- | ----------- |
| `audio` | Yes | An [audio input](#slice_audio) |
| `sample_rate` | No | Rate of a bare waveform |
| `model_name` | No | HuggingFace model ID of a Whisper-class ASR model (default: `openai/whisper-base`) |
| `timestamps` | No | `"segment"` or `"word"` to get chunk timings instead of plain text (see below) |

Audio is downmixed to mono and resampled to 16 kHz for the model; the source is untouched. By default the result is plain text, read with MCP's `get_output_text`.

Set `timestamps` to `"segment"` or `"word"` to get chunk timings instead. The result becomes `{"text": ..., "chunks": [{"start": ..., "end": ..., "text": ...}, ...]}`, so the step's `result.content_type` must be `"application/json"`, read with MCP's `get_output_text`. Every `start` and `end` is a number of seconds: a chunk the clip ends inside - a song cut mid-line, which Whisper leaves open-ended - ends at the clip's duration, so the transcript drops straight into [`attribute_voices`](#voice-attribution)' `lines`. `"word"` can give a zero-length chunk (`start` = `end`) on a short or clipped word; `attribute_voices` reads it as a line with no voice.

A chunk that is probably not speech carries one more field, `suspect` (#822), and is never removed. The reasons are tried in this order, and the first that holds wins:

- `"no_words"`: its text has no word at all (`—`, `♪`, `...`).
- `"repetition"`: it holds a word of a repetition loop (`I'm sorry, I'm sorry, ...`), found by the detector and `REPEAT_RUN_MIN`/`REPEAT_MAX_PERIOD` that [`check_script`](#script-check) uses, however loud; or its text and overlapping span repeat an earlier chunk's (#1138).
- `"silent"`: its loudest window is at or below `GUARD_FLOOR_DBFS`, where Whisper invents text; or the whole clip's mean level is at or below `NEAR_SILENT_CLIP_DBFS` (-40 dBFS), as in a voiceless vocals stem (#1135).
- `"stock_phrase"`: its whole text is one of Whisper's stock outro lines ("Thank you for watching", "Please subscribe") (#1136).
- `"long"`: a `"word"` chunk lasting more than `LONG_WORD_SECONDS` (2 s), an instrumental folded into the word. A `"segment"` chunk is never `long`.

Transcribe a song's vocal stem ([`separate_stems`](#stem-separation)) so an instrumental intro is not counted as speech; the marks catch what Whisper still invents over the stem. [`plan_cuts`](#plan_cuts) leaves `no_words`, `repetition`, `stock_phrase` and `silent` chunks out of the lines.

To compare a take against the lines it was meant to speak, use [`check_script`](#script-check) rather than reading the transcript by eye.

**Example:** [transcribe-audio.json](../workflows/templates/transcribe-audio.json) — Transcribe an audio file to text.

## Script check

A validate's `downloads_required` lists `openai/whisper-base` for its transcription; `model_name` overrides it.

Whether a dialogue take speaks its script: a dropped word, an H3 tag read aloud, a last word cut off by the end of the file.

It transcribes the take with [`transcribe_audio`](#speech-transcription) word timestamps (same code and model cache). A heard word at or below the dead-air floor is discarded as unheard, as is every word of a repetition loop (`Pre-pre-pre-...`). H3 markup is stripped from each expected line, the heard words are aligned to the expected words in order (`difflib`), and each line scores the similarity of its expected and heard words. Findings are places to look, not verdicts. It is not an assessment probe; it answers a JSON document.

```json
{
    "task": {
        "command": "check_script",
        "arguments": {
            "audio": "asset:take.wav",
            "lines": ["<d>[English] Where were you?</d>", "I was at the station."]
        }
    },
    "result": { "content_type": "application/json" }
}
```

The step's `result.content_type` must be `"application/json"`; read the answer with MCP's `get_output_text`.

| Argument | Required | Description |
| -------- | -------- | ----------- |
| `audio` | Yes | The take, an [audio input](#slice_audio) |
| `lines` | Yes | The expected lines in order: a list of strings or `{text, shot}` objects, `shot` naming the take's shot the line belongs in; a shot name the map holds more than once (a cut joined from episodes sharing names) also needs `start_frame` on the line, `{text, shot, start_frame}`, and findings call it `name#start_frame`. Markup is stripped (below). `[]` means no speech is expected |
| `shots` | No | The take's shot map, `[{name, start_frame, num_frames, start_sample, num_samples}]`. Default: the take's own `.shots` (an earlier step's video), else the run manifest or `keep_output` sidecar beside an `asset:`/`output:` file - resolved as the assessment probes resolve them (`assess.resolve_shots`). The argument wins over both |
| `similarity` | No | Least similarity, 0..1, of a line's heard words to its expected words before it is a `line_mismatch` (default `0.85`) |
| `model_name` | No | HuggingFace ID of a Whisper-class ASR model (default `openai/whisper-base`) |
| `sample_rate` | No | Rate of a bare waveform |

Markup stripped from an expected line: `<d>[Language] ...</d>`, `<scenetrans>`, `<cutoff>`, `[unclear]`, speaker IDs `(S1)` and `(S1,S2)`, and any other `<tag>` or `[tag]`. A plain parenthetical stays dialogue. The words of the stripped tokens (`cutoff`, `unclear`, `english`, `s1`) are what `tag_spoken` listens for; a tag word heard is reported there and left out of the line's similarity, so it is never a `line_mismatch` too.

| Rule | Fires when | `value` / `threshold` | `at` |
| ---- | ---------- | --------------------- | ---- |
| `line_mismatch` | A line's similarity is below `similarity`; a dropped line comes back with `heard: ""` | The line's similarity / `similarity` | `{line, seconds, word}` - the line's first heard word, or where it should have been |
| `tag_spoken` | A word from the line's stripped markup was heard, and is not also a word of the line's dialogue | The word heard / none | `{line, seconds, word}` |
| `transcript_loop` | A repetition loop (`REPEAT_RUN_MIN` periods of up to `REPEAT_MAX_PERIOD` words) made only of words the script holds - Whisper stuck on a line it heard, so speech after it can be missing from `heard` and lines read 0.0 without being wrong (#961). The loop's first occurrence is kept as heard when it has real duration and speech level; the rest is under `discarded`, one entry per loop (`count`, `start`..`end`). A line heard nowhere, at or after the loop, reads `heard: null`, `similarity: null`, `not_transcribed: "transcript_loop"` and raises no `line_mismatch`. When the shots are known the cut is then transcribed again one shot at a time and the findings and `lines` come from that pass (`transcription_pass: "per_shot"`, each line `pass: "per_shot"`, #963) | Chunks in the loop / `REPEAT_RUN_MIN` | `{line: null, seconds, end}` |
| `line_clipped_at_end` | A line's last heard word ends inside the final `CLIP_TAIL_SECONDS` of its shot (when the line has one) or of the file, and that tail is still above `GUARD_FLOOR_DBFS` and within `CLIP_TAIL_BELOW_WORD_DB` of the last word's own loudest window (a quieter tail is a score bed or room tone, not the word - #953) | The tail's level in dBFS / `GUARD_FLOOR_DBFS` | `{line, seconds, word, shot}` - `shot` when the line has one. A line nothing was heard for has no last word: when a `speech_not_transcribed` block credited to it runs to its shot's end and that tail is voiced, it fires at the block's end with no `word` (#1115); otherwise the line is listed in `rules_skipped` with its `line` index |
| `speech_where_silent` | `lines` is `[]` and words were heard above the floor and outside a repetition loop | Number of words heard / 0 | `{line: null, seconds, word}` - the first; `shot` when shots are known |
| `speech_in_silent_shot` | Shots are known, at least one line names a shot, and words were heard above the floor and outside a repetition loop in a shot no line names. A word is in the shot its midpoint falls in; one a line in another shot matched (a word straddling the cut) is that line's, not counted | Number of words heard in the shot / 0 | `{line: null, seconds, word, shot}` - the shot's first counted word, `seconds` no earlier than the shot's start |
| `unscripted_speech` | Shots are known and a line names the shot; words were heard inside it that align to no line (a line spoken twice, an ad-lib) | Number of unmatched words in the shot / 0 | `{line: null, seconds, end, word, shot, repeats_line}` - the first such word; `repeats_line` is true when those words repeat or contain one of the shot's lines |
| `speech_not_transcribed` | Between two words of one matched line, or between one line's last word and the next line's first (#960; the finding sits on the next line, and a gap holding heard-but-unmatched words is `unscripted_speech`'s), there is a gap of `UNHEARD_MIN_SECONDS` or more, or one of its words lasts `UNHEARD_MIN_SECONDS` + 2 `UNHEARD_MARGIN_SECONDS` or more, and the interior (trimmed by `UNHEARD_MARGIN_SECONDS` each side) holds a window above `UNHEARD_FLOOR_DBFS`: audible speech the ASR left out of its text, which the line's timing absorbed (#923). A block that repeats itself sample for sample (`BED_CORRELATION` 0.98 at a lag of 0.3 s or more, 0.5 s compared) is a looped bed from `loop_audio`, not speech, and is no finding (#1118) | The gap's length in seconds / `UNHEARD_FLOOR_DBFS` | `{line, seconds, end, shot}` - `shot` when the line has one |

Every finding is severity `warn`; `at.shot` is set whenever the shot is known. A rule that does not apply to the call is listed in `rules_skipped` with its reason: `speech_in_silent_shot` and `unscripted_speech` are skipped when the shots carry neither a sample span nor a frame span the take's frame rate can place, or when no line names a shot (no shot is known to be meant silent). When no shots are known at all (no `shots`, none carried by the take, no manifest beside the file) and the lines name at most one shot, the whole file is checked as one implicit shot: `unscripted_speech` runs (a line spoken twice, an ad-lib) and `speech_in_silent_shot` is skipped; `line_clipped_at_end` looks at the file's end only. Lines naming two or more shots with none placed skip both. With shots known there is no minimum word count - one unmatched word inside a named shot fires it.

The thresholds are module constants in `dw/tasks/script_findings.py` (the repetition guard's `REPEAT_*` in `dw/script_lines.py`; `GUARD_FLOOR_DBFS` and `GUARD_WINDOW_SECONDS` in `dw/tasks/script_take.py`), re-exported by `dw/tasks/script_check.py`. They are not in `RULES` (`dw/assessment_rules.py`), whose rules must fire on a file that holds no script. The answer echoes them under `thresholds`.

| Constant | Value | What crossing it does |
| -------- | ----- | --------------------- |
| `DEFAULT_SIMILARITY` | 0.85 | A line below this similarity is a `line_mismatch` (the `similarity` argument overrides it) |
| `GUARD_FLOOR_DBFS` | -65 dBFS (= `DEAD_AIR_FLOOR_DBFS`, `shot_dead_air`'s floor) | A heard word whose loudest window is at or below it is discarded as unheard; also the level a clipped tail must exceed |
| `UNHEARD_MIN_SECONDS` / `UNHEARD_MARGIN_SECONDS` / `UNHEARD_FLOOR_DBFS` | 1.0 s / 0.25 s / -40 dBFS | A gap in a line's words or between two lines (or a stretched word) this long, with a window above the floor in its margin-trimmed interior, is a `speech_not_transcribed` |
| `BED_CORRELATION` | 0.98 | An audible block matching itself this closely at a lag of 0.3 s or more is a looped bed, not untranscribed speech |
| `GUARD_WINDOW_SECONDS` | 0.05 s (= `DEAD_AIR_WINDOW`, `shot_dead_air`'s window) | The window a heard word is measured in - its loudest one, so a span overhanging a pause does not average a real word away |
| `CLIP_TAIL_SECONDS` | 0.25 s | The final stretch of the file a line's last word must end in to be `line_clipped_at_end` |
| `CLIP_TAIL_BELOW_WORD_DB` | 15 dB | A tail more than this far under the last word's loudest window is not the word still sounding (a bed under a finished line), so not `line_clipped_at_end` |
| `REPEAT_RUN_MIN` | 6 | Heard words repeating the same unit this many times over, back to back, are a repetition loop: every chunk holding one is discarded (`reason: "repetition"`), however loud |
| `REPEAT_MAX_PERIOD` | 6 words | The longest repeating unit a loop is looked for in - `pre pre pre` has period 1, `thank you thank you` period 2 |

The result: `findings`, `lines[]` (`expected`, `heard`, `similarity`, `start`, `end`, `shot`; with the shot known, `start` and `end` are kept inside it, and the ASR's own times, when they fall outside - a word stretched across the seam over speech Whisper dropped - are kept as `asr_start` / `asr_end`, #1115), `discarded[]` (`text`, `start`, `end`, `level_dbfs`, `reason` - `below_floor`, `repeated_span` (a chunk repeating the previous one's exact span) or `repetition`, `count` - a repetition loop is one entry spanning its chunks), `unmatched[]` (heard words aligned to no line), `transcript`, `model_name`, `rules_applied`, `rules_skipped[]` (`rule`, `reason`, and `line` for a per-line skip), `thresholds`, and `transcription_pass` (`whole_cut` or `per_shot`; with `per_shot`, a `transcription_note`). A line's `shot` is the shot it names, else the shot its heard words overlap most (null when none is known). `shots` is the map as placed - `[{name, start, end}]` in seconds on the take, or null when none is known - and `shots_source` says where it came from: `argument`, `artifact`, `manifest` or `none`.

How the alignment reads a take:

- A dropped line comes back with `heard: ""` and a `line_mismatch` at where it should have been.
- An added line's words go to `unmatched` and lower no line's similarity.
- A line spoken out of order reads as a mismatch on that line.
- Whisper mishears names and numbers (`2` for `two`), so listen before re-rolling a take on a mismatch.
- A quiet line can be discarded by the guard. Discarded words are reported under `discarded`, not hidden.

A malformed `lines` - not a list, an entry without text, an unknown key in a line object - is refused at validation, at `steps[i].task.arguments.lines`, as is a `similarity` outside 0..1. A line naming a shot the map lacks is refused, listing the map's shots, and so is a line naming a shot the map holds twice; a literal `shots` and literal lines are checked against each other at validation, and a `shots` or `lines` that is a reference is checked when the step runs.

**Example:** [check-script.json](../workflows/templates/check-script.json) — check a take against its script and save the answer as JSON. Its defaults are `openai/whisper-large-v3-turbo` at `similarity` 0.6, not the task's `openai/whisper-base` at 0.85: turbo scored correct lines 0.71 to 1.00 and wrong or missing ones 0.00, where base scored correct lines as low as 0.43 on mishearings (#609). Turbo writes numbers as words (`four fifteen`), so a script with digits costs similarity.

## Frame Interpolation

Raise a video's frame rate with RIFE, inserting frames between each pair. The result is one video without a soundtrack; [`pair_audio`](#pair_audio) puts the original track back. [interpolate-frames.json](../workflows/templates/interpolate-frames.json) is the template.

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
| `video` | Yes | The frames - a frame list, a frame array, or an audio+video pair from a concat or dissolve step (its audio is dropped) - usually a `previous_result:` reference |
| `multiplier` | No | Frame count multiplier: 2, 4, or 8 (default: 2) |
| `model_name` | No | HuggingFace repo with RIFE v4.13 weights (default: `imaginairy/rife-interpolation`) |
| `filename` | No | Weights filename within the repo (default: `rife-flownet-4.13.2.safetensors`) |

The IFNet v4.13 architecture is vendored; the weights download on first use. A validate's `downloads_required` lists the `model_name` / `filename` file when it is not cached.

**Example:** [interpolate-frames.json](../workflows/templates/interpolate-frames.json) — Generate video with Mochi, then 2x interpolate from 30fps to 60fps.

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
| JPEG/WebP | EXIF UserComment | Through piexif, a core dependency |

The metadata is JSON: the step name, the whole workflow and its seed, and the step's model name (a pipeline step) or `task_command` (a task step) with its arguments.

**Example:** [embed-metadata.json](../workflows/templates/embed-metadata.json) — Generate with Flux and embed parameters in PNG.

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

**Example:** [qr-code.json](../workflows/templates/qr-code.json) — QR code with artistic ControlNet

## Chat/Dict Plumbing

Glue for workflows that mix raw `transformers` components with task steps, where `text_generation` does not cover the case.

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

Calls `processor.batch_decode(...)` then `processor.post_process_generation(..., task=task)` and returns `parsed_answer[task]`.

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
                    "image": { "location": "asset:photo.jpg" },
                    "low_threshold": 50,
                    "high_threshold": 200
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

- [controlnet.json](../workflows/templates/controlnet.json) — Canny edge ControlNet (swap the task for another preprocessor)
- [qr-code.json](../workflows/templates/qr-code.json) — QR code with artistic ControlNet
- [upscale-spandrel.json](../workflows/templates/upscale-spandrel.json) — Spandrel 4x upscale of an existing image
- [restore-faces.json](../workflows/templates/restore-faces.json) — Generate portrait + GFPGAN face restoration
- [segment.json](../workflows/templates/segment.json) — Text-prompted object segmentation
- [segment-and-inpaint.json](../workflows/templates/segment-and-inpaint.json) — Segment + inpaint
- [image-to-text.json](../workflows/templates/image-to-text.json) — image captioning with the SmolVLM default
- [describe-and-regenerate.json](../workflows/templates/describe-and-regenerate.json) — Describe, expand, then regenerate
- [interpolate-frames.json](../workflows/templates/interpolate-frames.json) — RIFE frame interpolation
- [embed-metadata.json](../workflows/templates/embed-metadata.json) — Embed generation parameters in PNG
- [expand-prompt.json](../workflows/templates/expand-prompt.json) — LLM prompt expansion
- [upscale-diffusion.json](../workflows/templates/upscale-diffusion.json) — Diffusion upscale of an existing image
- [audio-trim-fade.json](../workflows/templates/audio-trim-fade.json) — Trim a generated track and fade its tail
- [best-of-n-to-video.json](../workflows/templates/best-of-n-to-video.json) — `judge` and `select` pick the best of several stills
- [relevel-clip.json](../workflows/templates/relevel-clip.json) — Fix a clipping soundtrack on a rendered clip without re-rendering
- [generate-speech.json](../workflows/templates/generate-speech.json) — Speak a line with a local text-to-speech model
- [split-takes.json](../workflows/templates/split-takes.json) — `split_takes` alone: see or edit a long text's takes before speaking them
- [narration-long.json](../workflows/templates/narration-long.json) — Split a long text into takes, speak each in one cloned voice and join them
- [voice-timbre-reference.json](../workflows/templates/minimax/voice-timbre-reference.json) — Generate a voice and condition H3's `<Audio 1>` on it
- [dissolve-between-shots.json](../workflows/templates/dissolve-between-shots.json) — Dissolve between supplied shots and mix a score under their own audio
- [face-repair.json](../workflows/templates/ltx2/face-repair.json) — Track, refine and paste back a wide shot's small face
- [minimax/face-repair.json](../workflows/templates/minimax/face-repair.json) — The same on an H3 take, its soundtrack held so the mouth stays on the speech
