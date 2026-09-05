# MiniMax workflows

Joint video-and-audio generation with [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3)
and music generation with [MiniMax-Music3](https://huggingface.co/MiniMaxAI/MiniMax-Music3),
fitted onto a single 24GB consumer GPU. Every example here runs on an RTX 3090;
the memory configuration they share is explained in
[docs/RECIPES_24GB.md](../../docs/RECIPES_24GB.md), and the H3 prompt format each
one writes is what the [built-in enhancer](MiniMaxH3EnhancePrompt.json) produces.

Read them in this order and each introduces one new idea on top of the last.

## The basics

| Example | What it introduces |
| ------- | ------------------ |
| [MiniMaxMusic.json](MiniMaxMusic.json) | The minimal modular pipeline: a `components_manager` owns device placement, and the output is audio, not video |
| [MiniMaxH3.json](MiniMaxH3.json) | The baseline text-to-video-audio run: per-component SDNQ quantization, mixed offload, the turbo LoRA, and muxing video + audio into one file |

A note on length: H3 accepts any `num_frames` of the form `17n + 5` between 124
and 345 - at its fixed 24 fps, that is 5.17 to 14.4 seconds **in a single clip**.
The examples default to 124 frames because that is the fast iteration loop;
passing `num_frames=345` on the command line is all it takes to run the model to
its full native length, and it fits in the same 24GB configuration. Write the
prompt for the length you are generating - its `[Shot N] At 00:0S.000` anchors
should span the full duration, since a prompt scripted for five seconds
conditions a five-second story regardless of the frame count. Reach for chains
and cuts when you need to go past 14 seconds.

## Conditioning on frames

| Example | What it introduces |
| ------- | ------------------ |
| [MiniMaxH3I2V.json](MiniMaxH3I2V.json) | A supplied image pins the first frame (the `fl2va` workflow given only an `image`) |
| [MiniMaxH3FL2VA.json](MiniMaxH3FL2VA.json) | Pinning both ends - `image` and `last_image` - so the model interpolates between two fixed states |
| [MiniMaxH3L2V.json](MiniMaxH3L2V.json) | Pinning the end alone - the model invents the approach to a picture you already have |

## Writing the prompt with a model

| Example | What it introduces |
| ------- | ------------------ |
| [MiniMaxH3EnhancePrompt.json](MiniMaxH3EnhancePrompt.json) | A `workflow` step runs the built-in enhancer, and the pipeline draws its prompt from `previous_result` |
| [MiniMaxH3I2VEnhancePrompt.json](MiniMaxH3I2VEnhancePrompt.json) | Showing the enhancer the same picture the pipeline gets, so prompt and keyframe agree |

## Conditioning on identity

| Example | What it introduces |
| ------- | ------------------ |
| [MiniMaxH3Ref2VA.json](MiniMaxH3Ref2VA.json) | The `references` list: an image fixes a subject's appearance, an audio clip fixes their voice |
| [MiniMaxH3Ref2VAVideo.json](MiniMaxH3Ref2VAVideo.json) | A video reference contributes framing, lighting and camera rather than appearance |
| [MiniMaxH3Ref2VAGeneratedSubject.json](MiniMaxH3Ref2VAGeneratedSubject.json) | Drawing the subject with Z-Image first and referencing it with `from_previous_result` |
| [MiniMaxH3Storyboard.json](MiniMaxH3Storyboard.json) | Several images in one request: a first frame plus storyboard anchors for later shots, so one generation cuts between three boards under an unbroken score |

## Going long: chains

A `chain` block runs the pipeline once per segment and stitches the output. Chains
are the tool for a single unbroken take longer than 14 seconds - but every segment
conditions on the previous segment's output, so expect drift to accumulate: fine
detail sharpens into noise, identity wanders, and each carry compounds the last.
Three things push back on it: reference the original subject picture in every
segment, use `last_segment` continuity rather than a single carried frame, and use
the longest segments your memory allows - drift accumulates per seam, so half the
segments means half the compounding.

| Example | What it introduces |
| ------- | ------------------ |
| [MiniMaxH3I2VChained.json](MiniMaxH3I2VChained.json) | The simplest chain: fixed segment count, `last_frame` continuity, trimmed seams, crossfaded audio |
| [MiniMaxH3Ref2VAChained.json](MiniMaxH3Ref2VAChained.json) | `match_audio`: a supplied track decides the length, and the final video is muxed against the original, seamless track |
| [MiniMaxH3Ref2VAChainedVideo.json](MiniMaxH3Ref2VAChainedVideo.json) | `last_segment` continuity: the previous segment's tail rides along as a video reference, carrying motion and voice across the seam |
| [MiniMaxH3Ref2VAChainedAligned.json](MiniMaxH3Ref2VAChainedAligned.json) | Everything together for a long soundtrack-driven take, with per-segment prompts and crash-safe `save_segments` |

## Going long: cuts (digital shorts)

Chained generation fights drift; a scene cut erases it. Each shot in these
examples is generated fresh from the same portraits, so the last shot is exactly
as clean as the first and the piece can run as long as the script does - which is
how television gets away with it too. This is the pattern for digital shorts:
write shots, not takes.

| Example | What it introduces |
| ------- | ------------------ |
| [MiniMaxH3SitcomShort.json](MiniMaxH3SitcomShort.json) | A five-shot sitcom scene: Z-Image draws the cast, `pipeline_reference` reruns one loaded model per shot, `concat_videos` splices the episode |
| [MiniMaxH3MusicVideo.json](MiniMaxH3MusicVideo.json) | A music video cut to a generated song: `slice_audio` deals frame-exact pieces to lip-synced shots, and `pair_audio` lays the unbroken track over the finished edit |
