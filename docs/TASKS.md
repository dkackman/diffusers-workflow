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

Any task that runs a model accepts a `"device"` argument to pin where it runs -
useful for keeping a helper model (a captioner, an upscaler) off the accelerator a
loaded pipeline is using, or on a second one. `"device"` is listed on every task's
schema, since it is always safe to pass: a task that runs no model - `slice_audio`,
`compose_text`, and the like - just ignores it.

Task argument schemas are discoverable: `GET /api/tasks/{command}` on the
[server](SERVER.md) returns each command's arguments read from its registered
implementation's real signature, the web editor builds task forms from them,
and workflow validation flags task-argument typos the same way it flags
pipeline ones.

A signature carries no domain, though, so the numbers whose domain is not a
judgement call are declared separately (`dw/task_domains.py`) and validation
reports one outside it as an error at its JSON path: a count of frames or
seconds to cut, and a sample rate or frame rate, have to be above zero, and an
offset to start at zero or above. Those are refused rather than interpreted -
`num_frames: -10` used to answer with the track minus its last ten frames and
`target_sample_rate: 0` with the original samples under a 44100 Hz header, both
reported as clean successes. The commands refuse the same values at run time,
which is what catches one that arrived from a `variable:` or an earlier step
rather than being written in the file.

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
| `recenter_crop` | Re-frame around a chosen point at a chosen scale, so a series of images registers on one feature; the window may run off the source | `center_x`, `center_y`, `crop`, `width`, `height`, `fill` |
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
| `videos` | Yes | The videos to join, in order - `previous_result` references, or the path or URL of a video file an earlier run wrote, which is read with the audio muxed into it |
| `trim_frames` | No | Frames dropped from the head of every video after the first (default: 0) |
| `crossfade_ms` | No | Equal-power crossfade at each audio seam, drawn from the trimmed material - no effect when `trim_frames` is 0, and validation warns when one is written there (default: 75) |
| `audio_bleed_ms` | No | How long the outgoing video's tail rings on over the head of the next one, at seams with nothing trimmed to crossfade (default: 0, off) |
| `audio_bleed_gain_db` | No | Gain applied to the bled tail before it is added, in dB - negative ducks a tail that would otherwise push the seam over 0 dBFS (default: 0, unchanged) |
| `seam_fade_ms` | No | Fade on each side of a seam that gets neither a crossfade nor a bleed - for tonal material, not for a continuous bed (default: 3, just enough not to click) |
| `fps` | No | Frame rate of the videos - required to join audio when trimming, and the rate the joined file is written at unless `result.fps` overrides it |
| `match_levels` | No | Even the shots' loudness out before joining - `"rms"` for perceived level (the measurement `get_gallery_metadata` reports as `mean_dbfs`), `"peak"` for the loudest sample. Off by default |
| `match_levels_dbfs` | No | The level `match_levels` moves every shot to (default: -1 dBFS for `peak`, -20 dBFS for `rms`). A shot that would clip at the target is held at -0.5 dBFS peak instead, reported as a `match_levels_held` warning with a per-shot log event |

A video may also be named by path or URL, which is how shots an earlier run
already wrote are joined without regenerating them - the file is read with the
audio muxed into it, and its track is fitted to the frames' own duration so the
codec's block padding does not walk the sound off the picture over a dozen
seams. A shot generated in memory through a `previous_result:` chain gets the
same fit, applied where the file is written rather than where it is decoded,
so per-shot drift does not accumulate across a cut the way it once did:

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
that produced several videos does not hand them all over at once - it fans the
step out over them, one concatenation per video, which is what makes the list
form above the way to join a run's shots.

`trim_frames` and `audio_bleed_ms` address opposite situations. A *chain* carries
its keyframe forward, so the trimmed head is material that covers the same stretch
of time as the outgoing tail and the two can be crossfaded. A *cut* generates each
shot independently, so there is nothing to fade with - and generated shots tend to
open on near-silence and end mid-sound, leaving a butt-join that drops a running
laugh track or a ringing room into a hole. `audio_bleed_ms` fills it the way an
audience carries across a picture cut: a decaying copy of the outgoing tail is laid
over the incoming head, added to whatever is already there, shortening neither side.
Reach for more than the gap looks like it needs: a shot's head is silent for
longer than the picture suggests, and the bleed has to outlast it. Measured on
a five-shot H3 sitcom cut, 700 ms still left a 44 dB hole at the worst seam;
1800 ms brought it to 32 dB and 2500 ms gained almost nothing more, so the
`dialogue-short` template defaults to 1800 and exposes it as `audio_bleed_ms`:

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

A bleed works because it copies ambience, which has no pitch and no attacks to
give the copy away. It is the wrong tool for anything tonal - a copied musical
phrase or half-spoken word reads as a stutter whichever direction it runs.
`bleed_join` checks the outgoing tail's spectral flatness and warns when it
looks tonal or speech-like rather than noise-like, so this failure mode
surfaces in the job's `warnings` list instead of only in the mix. When a
shot ends on something tonal, either give the cut a continuous bed with
`slice_audio` + `loop_audio` + `mix_audio` + `pair_audio`, which leaves no seam
to treat at all, or fade the
seam gracefully with `seam_fade_ms` (a hundred or so milliseconds) and accept the
cut. That advice inverts on a continuous bed - a laugh track, room tone - where a
longer fade only digs the hole deeper (the same sitcom cut measured 54-59 dB
holes with a 250-500 ms fade and no bleed). `audio_bleed_ms` wins where both are
set and there is material to bleed. A bleed covers the gap but cannot fill it:
the silence is inside the incoming shot's own head, and the only complete fix is
a continuous bed under the whole cut: `slice_audio` a few seconds of tone out of
a shot, [`loop_audio`](#loop_audio) it to the length of the cut, `mix_audio` it
under the episode and `pair_audio` it back onto the picture.

Levels are the other thing a cut has to reconcile, and no fade control can
touch it. Shots generated independently land wherever the model put them - two
shots of one scene, same template, same cast, measured `peak_dbfs` -2.65 and
-12.56 - and each reads as fine on its own, because a shot is only wrong
*relative to what it is cut against*. Butt-joined, that is a 10 dB drop at the
cut, and it is not an artifact *at* the seam that a fade could smooth: it is
either side of it. `match_levels` scales each track before the join -
`"rms"` matches perceived level, which is usually what "make these sound the
same" means, and `"peak"` matches the loudest sample, which is the safer
choice on material with big transients. A shot whose gain would clip at the
target is held just below full scale and the log says so. Left off - the
default, so nothing existing changes - a spread of 6 dB or more across the
tracks being joined is reported as a warning rather than passing in silence:
on the job's `warnings` and as a `warning` event in its stream, not only in
the server's log, since the caller who can act on it is the one who asked for
the run. `dissolve_videos` takes the same pair.

### dissolve_videos

Join videos with a cross-dissolve at every seam, and fade the whole piece in
from and out to a colour. Where `concat_videos` cuts - right for shots that
each carry their own sound - this melts one shot into the next, which is what a
montage cut to a score wants:

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
| `videos` | Yes | The videos to join, in order - `previous_result` references, or the path or URL of a video file an earlier run wrote, one entry per video as with `concat_videos` |
| `dissolve_frames` | No | Frames of overlap at each seam, blended linearly (default: 12). 0 is a hard cut |
| `fade_in_frames` | No | Frames over which the first video rises out of `fade_color` (default: 0) |
| `fade_out_frames` | No | Frames over which the last video sinks into it (default: 0) |
| `fade_color` | No | The RGB colour the fades come from and go to (default: black) |
| `fps` | No | Frame rate of the videos - required to crossfade audio at a dissolve, and the rate the dissolved file is written at unless `result.fps` overrides it |
| `match_levels` | No | Even the shots' loudness out before joining - `"rms"` or `"peak"`, as with [`concat_videos`](#concat_videos). Off by default |
| `match_levels_dbfs` | No | The level `match_levels` moves every shot to (default: -1 dBFS for `peak`, -20 dBFS for `rms`). A shot that would clip at the target is held at -0.5 dBFS peak instead, reported as a `match_levels_held` warning with a per-shot log event |

Every seam shortens the result by one overlap, so eight 124-frame shots joined
with 12-frame dissolves run 908 frames, not 992 - size a soundtrack slice to
the joined length, not the sum. When every input carries audio, the tracks are
crossfaded over exactly the seam's span so they stay in step with the picture;
when any input is silent the result is, and `pair_audio` puts a score under it.

**Example:** [dissolve-between-shots.json](../workflows/templates/dissolve-between-shots.json)

### stabilize_video

Remove a generated clip's accumulated framing drift - the slow wander a video
model adds over a shot that was meant to hold still:

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
| `smooth` | No | `0` (the default) locks the framing to the first frame, which is what a shot generated from a pinned keyframe wants. A window in frames instead removes only the wander faster than that window, so a slow deliberate camera move survives and the drift around it does not |

The argument is `clip`, not `video`, on purpose: the engine loads an argument
named `video` itself, as bare frames, which would strip the soundtrack off
before the task ever saw it. Frames are shifted back and the result is cropped
to the region every frame covers, then resized to the original size; a
soundtrack passes through untouched.

It is a stabilization pass, not a format pass. `smooth: 0` on a shot with a
deliberate camera move fights the move - every frame is shifted back toward
the first, cropped and rescaled - and nothing downstream will notice, since
the duration, size and sample rate all survive. Run it on a shot that drifts,
as its own step; do not run it on every shot before a cut, which is what the
assembly templates once did and what made their output visibly wider than
the source. The join tasks refuse shots of different sizes, so no
normalization step is needed before them.

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

**Example:** [extend-clip.json](../workflows/templates/ltx2/extend-clip.json)

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
| `video` | Yes | The frames - a frame list, a frame array or tensor, or an audio+video pair whose own soundtrack is replaced; their own rate is carried through to the output, so `result.fps` is only needed to override it (frames that carry none are written at 8 fps) |
| `audio` | Yes | The soundtrack - a waveform, the earlier step whose video carried one, or the path or URL of an audio or video file; the last two bring their sample rate along. A mono track is fine: an mp4 audio stream takes stereo and nothing else, so saving duplicates the one channel into two and warns that it did |
| `sample_rate` | No | Sample rate of the waveform. Required unless `audio` carries one; given here it wins |

**Example:** [assemble-and-score.json](../workflows/templates/assemble-and-score.json)

### slice_audio

Cut a slice out of an audio track, addressed in seconds or in video frames.
Slices reaching past the end of the track are zero-padded — asking for more
than the source holds returns a track of the length you asked for whose tail is
digital silence, not a shorter track and not an error. Anything past a few
milliseconds of that padding is reported as a `slice_past_end` warning on the
job, because a score laid under a longer cut goes silent for the rest of the
film without anything else saying so; to fill a cut longer than the recording,
build a bed with [`loop_audio`](#loop_audio) first and slice that. Either half
of a pair may be left out - an omitted start begins at the head of the track, an omitted
duration runs to the end of it - so a workflow that trims only when it is given
a length still passes the whole track along:

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
| `audio` | Yes | Path or URL of an audio file (or of a video file, whose soundtrack is taken), a waveform from a previous step, or an earlier step's video generated with a soundtrack (which brings its sample rate along) |
| `start_seconds` / `duration_seconds` | One pair | The slice in seconds; either may be omitted |
| `start_frame` / `num_frames` / `fps` | One pair | The slice in video frames; `fps` is required, start and count may be omitted |
| `sample_rate` | With a waveform | Sample rate of a directly passed waveform (files carry their own) |

### gain_audio

Apply a gain, in decibels, to a region of an audio track - the rest of the
track passes through unchanged. The region is addressed the same way
`slice_audio`'s is, in seconds or in video frames, so ducking a scene under
another (lowering a dialogue track between two timestamps) is one step
instead of the `slice_audio` → `gain` (a whole-track `normalize_audio` on the
slice) → `mix_audio` → `rejoin` → `pair_audio` chain that used to be the only
way to gain part of a track rather than all of it. Unlike `slice_audio`, a
region reaching past the end of the track is clipped to it rather than
zero-padded - there is no silence there to gain, only the end of the real
material:

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
| `audio` | Yes | Path or URL of an audio file (or of a video file, whose soundtrack is taken), a waveform from a previous step, or an earlier step's video generated with a soundtrack (which brings its sample rate along) |
| `gain_db` | Yes | Gain to apply within the region, in decibels - negative ducks it, positive boosts it |
| `start_seconds` / `duration_seconds` | One pair | The region in seconds; either may be omitted |
| `start_frame` / `num_frames` / `fps` | One pair | The region in video frames; `fps` is required, start and count may be omitted |
| `sample_rate` | With a waveform | Sample rate of a directly passed waveform (files carry their own) |

One pair is required — there is no separate "whole track" mode — but the
whole track is still one step: give just `start_seconds: 0` and leave
`duration_seconds` unset (or `start_frame: 0` + `fps` and leave `num_frames`
unset), which runs to the end of the track without needing to already know
how long that is.

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

### fade_audio

Fade a track in from silence and out to it. A slice cut out of the middle of a
piece ends on whatever was sounding at the cut; a fade turns that into an
ending. The curve is the equal-power cosine the seam joins use:

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
| `audio` | Yes | Path or URL of an audio file (or of a video file, whose soundtrack is taken), a waveform from a previous step, or an earlier step's video generated with a soundtrack (which brings its sample rate along) |
| `fade_in_ms` | No | Length of the fade in, from the head of the track (default: 0) |
| `fade_out_ms` | No | Length of the fade out, to the tail of the track (default: 0) |
| `sample_rate` | With a waveform | Sample rate of a directly passed waveform (files carry their own) |

**Example:** [audio-trim-fade.json](../workflows/templates/audio-trim-fade.json) — slice a generated track to length, then fade the cut into an ending.

### normalize_audio

Scale a track so its loudest sample sits at a level. Generated music comes out
wherever the model happened to land - a quiet take needs lifting before it sits
under a picture, a hot one needs headroom before the encoder. Only the gain
changes, so the dynamics survive:

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
| `audio` | Yes | Path or URL of an audio file (or of a video file, whose soundtrack is taken), a waveform from a previous step, or an earlier step's video generated with a soundtrack (which brings its sample rate along) |
| `peak_dbfs` | No | The level the loudest sample is moved to, in dB below full scale (default: -1.0). 0 is full scale |
| `sample_rate` | With a waveform | Sample rate of a directly passed waveform (files carry their own) |

A silent track is returned unchanged.

**Example:** [dissolve-between-shots.json](../workflows/templates/dissolve-between-shots.json)

### mix_audio

Layer tracks on top of one another. `crossfade_audio` puts tracks one after
another; this puts them on top of each other - a score laid under a film's own
sound, where the music runs unbroken while the world underneath it is replaced
at every cut:

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
| `audios` | Yes | The tracks to layer - waveforms, audio or video file paths, or videos generated with a soundtrack |
| `gains` | No | One plain multiplier per track, in the same order - not decibels. Defaults to unity on every track |
| `sample_rate` | With a raw waveform | Sample rate of the waveforms. Required unless every track brings its own; given here it wins |

Tracks of different lengths are padded with silence to the longest, so a score
shorter than the picture leaves the tail dry rather than cutting the picture
down to fit. Summing can push peaks past full scale and the sum is *not*
rescaled - follow it with `normalize_audio` to bring the peak back down.

**Example:** [dissolve-between-shots.json](../workflows/templates/dissolve-between-shots.json) — a
generated score mixed under the shots' own audio.

### loop_audio

Make a bed of a given length out of a short recording — the room tone laid
under a whole cut, which is the only complete fix for the hole at a seam. Each
shot in a cut carries its own room and nothing runs underneath the join;
a continuous bed does, the way a location's room tone is laid under a dialogue
scene so the edits stop being audible:

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

Laps are joined with an equal-power crossfade rather than butted together, so
the loop point itself is not a click. That only smooths the seam: a transient
in the source (a hit, a swell) still recurs once per lap at full strength, so
the loop still reads as a level pulse at the lap rate — measured at 9.3 dB on
a source with one such transient. Pick a source with even internal level to
avoid the pulse; the crossfade does not remove it. The source is used whole
every lap and only the last one is trimmed, so the bed lands exactly on the
requested length; a source longer than the request is trimmed to it.

The bed is laid under the cut with `mix_audio` and attached to the picture with
`pair_audio`:

```json
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
```

Where the bed itself comes from is the open question: a few seconds of a
generated shot's own ambience, cut out with `slice_audio` from a stretch with
nothing tonal in it, is the material that matches — the room the shots were
generated in.

### resample_audio

Convert a track to a different sample rate. A pipeline that conditions on audio
wants it at its own rate (MiniMax H3 at its audio VAE's), and resampling a
supplied recording once, up front, feeds it what it already wants:

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
| `audio` | Yes | Path or URL of an audio or video file, a video generated with a soundtrack (which brings its sample rate along), or a waveform |
| `target_sample_rate` | Yes | The rate to convert to |
| `sample_rate` | With a waveform | Sample rate of a waveform passed directly; given for a file or a video it overrides the rate they carry |

A track already at the target rate is returned untouched. The conversion is
PyAV's, which dw already needs for video - no torchaudio dependency.

Every audio task returns the waveform *and* the rate it is at, so one chains
into the next without the rate being restated: a `resample_audio` fed
`previous_result:` from a `slice_audio` takes the source rate from the slice. A
`sample_rate` given on the step still wins, and one declared on the step's
`result` still decides what is written to disk.

**Example:** [assemble-and-score.json](../workflows/templates/assemble-and-score.json)

### compress_audio

Shape a track's dynamics with an envelope-follower - a compressor, a limiter
and a gate are the same algorithm with different knob settings, so one task
covers all three through `mode`:

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
| `audio` | Yes | Path or URL of an audio file (or of a video file, whose soundtrack is taken), a waveform from a previous step, or an earlier step's video generated with a soundtrack (which brings its sample rate along) |
| `threshold_dbfs` | Yes | The level the envelope is measured against, in dB below full scale. Cannot be above 0 |
| `ratio` | No | How hard the reduction is above the threshold, in `compress`/`gate` mode (default: 4.0). Ignored in `limit` mode, which always holds the signal at the threshold |
| `attack_ms` | No | How fast the envelope rises to a louder signal (default: 10.0). 0 means instantly |
| `release_ms` | No | How fast the envelope falls back after a louder signal ends (default: 100.0). 0 means instantly |
| `mode` | No | `compress` (turn down what's above the threshold), `limit` (hold the signal at the threshold), or `gate` (turn down what's below the threshold) (default: `compress`) |
| `sample_rate` | With a waveform | Sample rate of a directly passed waveform (files carry their own) |

A silent track is returned unchanged.

### filter_audio

Run a track through a single biquad filter stage - trimming the frequencies a
mix doesn't need, or carving out room for another element:

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
| `audio` | Yes | Path or URL of an audio file (or of a video file, whose soundtrack is taken), a waveform from a previous step, or an earlier step's video generated with a soundtrack (which brings its sample rate along) |
| `cutoff_hz` | Yes | The filter's corner frequency. Must be below the Nyquist frequency (half the sample rate) |
| `kind` | No | `lowpass`, `highpass`, `bandpass`, or `notch` (default: `lowpass`) |
| `q` | No | The filter's resonance/bandwidth (default: 0.707, a Butterworth response) |
| `sample_rate` | With a waveform | Sample rate of a directly passed waveform (files carry their own) |

A silent track is returned unchanged.

### analyze_audio

Measure a track without changing it - peak and RMS level, crest factor, and a
rough low/mid/high spectral balance, the numbers a `compress_audio` or
`filter_audio` step downstream is tuned against rather than guessed at:

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
| `audio` | Yes | Path or URL of an audio file (or of a video file, whose soundtrack is taken), a waveform from a previous step, or an earlier step's video generated with a soundtrack (which brings its sample rate along) |
| `sample_rate` | With a waveform | Sample rate of a directly passed waveform (files carry their own) |

Returns a dict, not a track: `peak_dbfs`, `rms_dbfs`, `crest_factor_db`,
`low_dbfs` (20-250 Hz), `mid_dbfs` (250-4000 Hz), `high_dbfs` (4000-20000 Hz).
The three bands are each a share of the track's total power on the same
scale as `rms_dbfs` (their powers sum to it), so the loudest band sits near
`rms_dbfs` rather than tens of dB under it - comparable to `compress_audio`'s
`threshold_dbfs`. A silent track, or a band with no content at the track's
sample rate, reads as `null` rather than `-inf`.

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

Same as `gather_images` but for video files. Each video comes back as one
artifact holding its frames and whatever audio was muxed alongside them, so a
step referencing this one iterates over videos rather than over frames.

To *join* videos that are already on disk, give their paths to `concat_videos`
directly rather than gathering them first: a `previous_result` reference to a
gather step fans the consuming step out over the gathered videos instead of
handing it all of them at once.

### gather_inputs

Pass through arguments directly. Useful for organizing data flow.

## Videos in image tasks

Every image command - the upscalers, face restoration, segmentation and the
image processors - takes a video where it takes an image: an `AudioVideo` from
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
| `tile_size` | No | Tile size for large images (default: 512) |
| `tile_overlap` | No | Overlap between tiles in pixels (default: 32) |

Large images are automatically tiled to avoid GPU memory issues. Models can be loaded from HuggingFace Hub repos or local `.pth`/`.safetensors` files.

**Examples:**
- [upscale-spandrel.json](../workflows/templates/upscale-spandrel.json) — Upscale any existing image 4x.
- [upscale-spandrel.json](../workflows/templates/upscale-spandrel.json) — Upscale an image you already have; there is no generation step, so the input is a path or URL.

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
| `image` | Yes | PIL Image or `previous_result:` reference - a video runs frame by frame, see [Videos in image tasks](#videos-in-image-tasks) |
| `prompt` | No | Text guidance for upscaling (default: "") |
| `negative_prompt` | No | Negative text guidance (default: none) |
| `mode` | No | `"x4"` or `"x2"` (default: `"x4"`) |
| `model_name` | No | Override the default model for the selected mode |
| `num_inference_steps` | No | Denoising steps (default: 25) |
| `guidance_scale` | No | Classifier-free guidance scale (default: 9.0) |
| `noise_level` | No | Noise level for x4 mode (default: 20, ignored for x2) |

**Examples:**
- [upscale-diffusion.json](../workflows/templates/upscale-diffusion.json) — Upscale any existing image. `mode` selects which: `x4` (the default) reaches 2048px, `x2` reaches 1024px through the latent upscaler.
- [upscale-diffusion.json](../workflows/templates/upscale-diffusion.json) — Prompt-guided upscale of an image you already have, with no generation step.

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
| `upsample_img` | No | Pre-upscaled background image (e.g., from a prior upscale step) |

Models are loaded via spandrel, so any `.pth`/`.safetensors` face restoration weights work. CodeFormer requires `pip install spandrel-extra-arches` (non-commercial license).

**Example:** [restore-faces.json](../workflows/templates/restore-faces.json) — Generate a portrait, then restore faces with GFPGAN v1.4.

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
| `image` | Yes | PIL Image or `previous_result:` reference - a video runs frame by frame, see [Videos in image tasks](#videos-in-image-tasks) |
| `prompt` | Yes | Text description of object(s) to detect (e.g., "dog", "red car") |
| `model_name` | No | GroundingDINO model ID (default: `IDEA-Research/grounding-dino-base`) |
| `sam_model_name` | No | SAM2 model ID (default: `facebook/sam2-hiera-large`) |
| `threshold` | No | Detection confidence threshold (default: 0.3) |
| `invert` | No | Invert the output mask (default: false) |

Returns a grayscale PIL Image (mode "L") — white (255) for detected objects, black (0) for background. Use with inpainting pipelines like FluxFillPipeline.

**Examples:**

- [segment.json](../workflows/templates/segment.json) — Segment an object from an image
- [segment-and-inpaint.json](../workflows/templates/segment-and-inpaint.json) — Segment, then inpaint the masked region

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

For a detailed caption, hand the image to [`text_generation`](#text-generation) with a question and a larger vision-language model; that is what [describe-and-regenerate.json](../workflows/templates/describe-and-regenerate.json) does ahead of its prompt expansion.

**Examples:**

- [image-to-text.json](../workflows/templates/image-to-text.json) — Basic captioning with the default model, saves as `.txt`
- [image-to-text.json](../workflows/templates/image-to-text.json) — Larger VLM answering a specific question
- [describe-and-regenerate.json](../workflows/templates/describe-and-regenerate.json) — Describe an image, expand the caption, then regenerate it

## Composing Text

Assemble one block of text out of parts written once. A multi-shot workflow
says the same things about its characters in every shot — who they are, what
they are wearing, what their voice sounds like — and the engine deliberately
has no string interpolation to splice them in with (see the no-interpolation
rule in [the workflow guide](WORKFLOW_GUIDE.md)). Composition is the way
round it: a part is a *whole* value, and `compose_text` joins parts in order.

```json
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
```

| Argument | Required | Description |
| -------- | -------- | ----------- |
| `parts` | Yes | The parts to join, in order — each a whole value, usually a `variable:`, `prompt:` or `previous_result:` reference. Numbers are written out; `null` is dropped, so an optional part can be a variable left null |
| `separator` | No | What goes between the parts (default: a blank line, the paragraph break the prompt formats use) |
| `skip_empty` | No | Drop parts that are null or blank (default `true`). With it off, an empty part still contributes its separator |

A part that is neither text nor a number is an error, not a coercion: it means
the reference in that position resolved to something other than the text meant.

The parts are positional. A named form (`"{bible} says {line}"`) would be the
interpolation the engine does not have, one layer down — so a character bible
is a variable named by every shot that needs it, and a voice string written
once is checked by being the same value rather than by being compared.

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

- [expand-prompt.json](../workflows/templates/expand-prompt.json) — Expand a short prompt and save as `.txt`
- [expand-prompt.json](../workflows/templates/expand-prompt.json) — Expand prompt, then generate with Flux

## Speech Generation

Speak a line of text with a local text-to-speech model. The result is a waveform carrying the rate its model generated at, so it composes with [`slice_audio`](#slice_audio), [`fade_audio`](#fade_audio) and [`pair_audio`](#pair_audio) directly ([`concat_videos`](#concat_videos) and `dissolve_videos` join videos — pair the track onto a video first).

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
| `text` | Yes | The line to speak |
| `model_name` | No | HuggingFace model ID (default: `suno/bark-small`) |
| `voice_preset` | No | The speaker, for a model with presets — `v2/en_speaker_0` through `v2/en_speaker_9` for Bark. A model with no processor (a single-voice model such as `facebook/mms-tts-eng`) refuses a `voice_preset` with an error rather than ignoring it |
| `forward_params` | No | Passed to the model's forward/generate call |
| `generate_kwargs` | No | Ad-hoc generation settings for a generative model — `temperature`, `do_sample` |

The default is Bark because its voice presets give distinct speakers, which is what two characters in a scene need; `facebook/mms-tts-eng` is a quarter the size and a good override where one voice will do. `voice_preset` is a preprocessing argument — it selects the speaker before generation rather than parameterizing it — so naming it here is what makes it reach the processor. Passed through `forward_params` it would be dropped and every character would sound the same.

The result needs no `sample_rate`. A generated track carries the rate its model produced it at, and that beats the 44100 default; declaring one still wins over both, for a track whose rate was reported wrong. Every TTS model runs at a different rate, so a declared rate that does not match plays the speech at the wrong speed and pitch without ever failing.

### Generating a voice to condition on

The role this earns its place in is voice *timbre reference*, not the track a mouth follows. MiniMax H3 lip-syncs well when it generates the speech itself and poorly when it must follow supplied audio, so its `MiniMaxH3AudioReference` takes a few seconds of a voice to fix timbre, pitch and delivery while H3 still generates the line. Build the reference with `from_previous_result` and the clip's own sample rate comes across with it:

```json
"references": [
    {
        "reference_type": "diffusers.modular_pipelines.minimax_h3.MiniMaxH3AudioReference",
        "from_previous_result": "voice"
    }
]
```

Referencing the same preset in every shot of a scene makes a character's voice a conditioning signal rather than a prose description that has to land identically a dozen times. The other honest uses are a voice that must be matched — a specific delivery H3 will not produce from description alone — and narration over shots where nothing has to lip-sync to it, muxed with [`pair_audio`](#pair_audio).

A speech model is worth releasing before a video model loads — set `release_models` on the step, as in the example below.

**Examples:**

- [generate-speech.json](../workflows/templates/generate-speech.json) — Speak a line and save it as a `.wav`
- [voice-timbre-reference.json](../workflows/templates/minimax/voice-timbre-reference.json) — Generate a voice, then condition H3's `<Audio 1>` on it

## Speech Transcription

Transcribe spoken audio to text with a local Whisper-class model. The word-correctness of a TTS deliverable — a dropped line, a mid-sentence truncation — can only be inferred from duration and timing arithmetic without this; `transcribe_audio` checks it directly against the text the deliverable was supposed to speak.

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
| `audio` | Yes | Path or URL of an audio file (or of a video file, whose soundtrack is taken), a video with a soundtrack, or a waveform — usually a `previous_result:` reference |
| `sample_rate` | No | Sample rate of a waveform passed directly |
| `model_name` | No | HuggingFace model ID of a Whisper-class ASR model (default: `openai/whisper-base`) |

Multi-channel audio is downmixed to mono and resampled to 16 kHz before transcription, since that is what a Whisper-class model is trained on; the source audio itself is untouched. The result is plain text, read with MCP's `get_output_text`.

**Example:** [transcribe-audio.json](../workflows/templates/transcribe-audio.json) — Transcribe an audio file to text.

## Frame Interpolation

Increase video frame rate using RIFE (Real-Time Intermediate Flow Estimation). Takes a video and inserts intermediate frames between each pair. The result is one video artifact without a soundtrack - the frame count changed, so [`pair_audio`](#pair_audio) is how the original track comes back. [interpolate-frames.json](../workflows/templates/interpolate-frames.json) shows the interpolation itself.

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

Uses vendored IFNet v4.13 architecture. Weights are downloaded from HuggingFace Hub on first use.

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
| JPEG/WebP | EXIF UserComment | Requires `pip install piexif` |

Metadata includes step name, model name, and generation arguments (prompt, steps, guidance scale, etc.) as JSON.

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

These small tasks glue together multi-step pipelines that mix raw `transformers` components with task steps — for the cases `text_generation` does not cover.

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

- [controlnet.json](../workflows/templates/controlnet.json) — Canny edge ControlNet
- [controlnet.json](../workflows/templates/controlnet.json) — Depth-guided generation
- [qr-code.json](../workflows/templates/qr-code.json) — QR code with artistic ControlNet
- [upscale-spandrel.json](../workflows/templates/upscale-spandrel.json) — Spandrel 4x upscale of an existing image
- [restore-faces.json](../workflows/templates/restore-faces.json) — Generate portrait + GFPGAN face restoration
- [segment.json](../workflows/templates/segment.json) — Text-prompted object segmentation
- [segment-and-inpaint.json](../workflows/templates/segment-and-inpaint.json) — Segment + inpaint
- [image-to-text.json](../workflows/templates/image-to-text.json) — image captioning with the SmolVLM default
- [image-to-text.json](../workflows/templates/image-to-text.json) — VLM captioning with a specific question
- [describe-and-regenerate.json](../workflows/templates/describe-and-regenerate.json) — Describe, expand, then regenerate
- [interpolate-frames.json](../workflows/templates/interpolate-frames.json) — RIFE frame interpolation
- [embed-metadata.json](../workflows/templates/embed-metadata.json) — Embed generation parameters in PNG
- [expand-prompt.json](../workflows/templates/expand-prompt.json) — LLM prompt expansion
- [expand-prompt.json](../workflows/templates/expand-prompt.json) — Expand prompt + generate image
- [upscale-spandrel.json](../workflows/templates/upscale-spandrel.json) — Spandrel upscale of an existing image
- [upscale-diffusion.json](../workflows/templates/upscale-diffusion.json) — Diffusion upscale of an existing image
- [audio-trim-fade.json](../workflows/templates/audio-trim-fade.json) — Trim a generated track and fade its tail
- [generate-speech.json](../workflows/templates/generate-speech.json) — Speak a line with a local text-to-speech model
- [voice-timbre-reference.json](../workflows/templates/minimax/voice-timbre-reference.json) — Generate a voice and condition H3's `<Audio 1>` on it
- [dissolve-between-shots.json](../workflows/templates/dissolve-between-shots.json) — Dissolve between supplied shots and mix a score under their own audio
