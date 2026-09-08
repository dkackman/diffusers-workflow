# MiniMax workflows

Joint video-and-audio generation with [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3)
and music generation with [MiniMax-Music3](https://huggingface.co/MiniMaxAI/MiniMax-Music3),
fitted onto a single 24GB consumer GPU. Every example here runs on an RTX 3090;
the memory configuration they share is explained in
[docs/RECIPES_24GB.md](../../../docs/RECIPES_24GB.md).

The prompt format is MiniMax's own. Two guides on the model card define it -
`docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md` for text- and frame-conditioned
generation and `docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md` for reference-conditioned -
and MiniMax publishes them as an agent skill, `skills/h3-prompt-writing`, in the
[MiniMax-H3 GitHub repository](https://github.com/MiniMax-AI/MiniMax-H3). The
[built-in enhancer](enhance-prompt.json) writes the same format from those guides;
for hand-written prompts, read them rather than reverse-engineering the examples.
Audited against those sources on 2026-09-07
([the audit](../../../docs/proposals/audits/2026-09-07-minimax-h3-audit.md)).
Music3 has the same kind of authority: the caption format is MiniMax's
`skills/music-caption-rewriter` in the
[MiniMax-Music3 GitHub repository](https://github.com/MiniMax-AI/MiniMax-Music3)
(a genre router, 18 family indexes and 1,000 example captions) and the tag
vocabulary is on its model card; audited 2026-09-08
([the audit](../../../docs/proposals/audits/2026-09-08-minimax-music3-audit.md)).

An agent driving this family from Claude Code has the `minimax-h3` and
`minimax-music3` skills of the [dw plugin](../../../plugins/dw/README.md), which
choose among these templates and point at the guides.

Read them in this order and each introduces one new idea on top of the last.

## The basics

| Example | What it introduces |
| ------- | ------------------ |
| [music.json](music.json) | The minimal modular pipeline: a `components_manager` owns device placement, and the output is audio, not video |
| [video-with-audio.json](video-with-audio.json) | The baseline text-to-video-audio run: per-component SDNQ quantization, mixed offload, the turbo LoRA, and muxing video + audio into one file |

A note on `audio_duration`: Music3 reads it as a ceiling rather than a target.
The language model stops when the song ends, so the track is usually shorter
than the ceiling, and a ceiling set to the length the song *should* be leaves
no room for the outro before the hard stop. Ask for more time than the song
needs and trim the tail afterwards -
[templates/audio-trim-fade.json](../audio-trim-fade.json) slices a generated track
to length and fades the cut into an ending. Runtime follows the length actually
generated, not the ceiling, so the margin is free. The output is 44.1 kHz stereo,
the vocoder's native rate; the model card's 32 kHz is what MiniMax's reference
server resamples to.

A note on length: H3 accepts any `num_frames` of the form `17n + 5` between 124
and 345 - at its fixed 24 fps, that is 5.17 to 14.4 seconds **in a single clip**.
The examples default to 124 frames because that is the fast iteration loop;
passing `num_frames=345` on the command line is all it takes to run the model to
its full native length, and it fits in the same 24GB configuration. Write the
prompt for the length you are generating - its `[Shot N] At 00:0S.000` anchors
should span the full duration, since a prompt scripted for five seconds
conditions a five-second story regardless of the frame count. Reach for chains
and cuts when you need to go past 14 seconds.

A note on the canvas: H3 draws on a 768-pixel short edge (960x544 in these
examples is the speed choice, coupled to the 544p turbo LoRA and its nine steps -
change one and change the others), dimensions are multiples of 32, and aspect ratios
run from 1:4 to 4:1. The 5-second floor is diffusers' constraint; the model card and
the hosted API accept 4. Output audio is 32 kHz stereo.

## Conditioning on frames

| Example | What it introduces |
| ------- | ------------------ |
| [image-to-video.json](image-to-video.json) | A supplied image pins the first frame (the `fl2va` workflow given only an `image`) |
| [first-and-last-frame.json](first-and-last-frame.json) | Pinning both ends - `image` and `last_image` - so the model interpolates between two fixed states |
| [last-frame-only.json](last-frame-only.json) | Pinning the end alone - the model invents the approach to a picture you already have |

## Writing the prompt with a model

| Example | What it introduces |
| ------- | ------------------ |
| [enhance-prompt.json](enhance-prompt.json) | A `workflow` step runs the built-in enhancer, and the pipeline draws its prompt from `previous_result` |
| [enhance-prompt-with-image.json](enhance-prompt-with-image.json) | Showing the enhancer the same picture the pipeline gets, so prompt and keyframe agree |

## Conditioning on identity

| Example | What it introduces |
| ------- | ------------------ |
| [reference-to-video.json](reference-to-video.json) | The `references` list: an image fixes a subject's appearance, an audio clip fixes their voice |
| [voice-timbre-reference.json](voice-timbre-reference.json) | A `generate_speech` step speaks a throwaway line with a chosen Bark preset, and that clip goes in as H3's `<Audio 1>` to fix timbre, pitch and delivery while H3 still generates the dialogue; reuse the same preset across shots for a consistent voice |
| [composable-references.json](composable-references.json) | A video reference contributes framing, lighting and camera rather than appearance |
| [generated-subject-reference.json](generated-subject-reference.json) | Drawing the subject with Z-Image first and referencing it with `from_previous_result` |
| [storyboard.json](storyboard.json) | Several images in one request: a first frame plus storyboard anchors for later shots, so one generation cuts between three boards under an unbroken score |

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
| [chained-segments.json](chained-segments.json) | The simplest chain: fixed segment count, `last_frame` continuity, trimmed seams, crossfaded audio |
| [chain-matched-to-audio.json](chain-matched-to-audio.json) | `match_audio`: a supplied track decides the length, and the final video is muxed against the original, seamless track |
| [chain-video-continuity.json](chain-video-continuity.json) | `last_segment` continuity: the previous segment's tail rides along as a video reference, carrying motion and voice across the seam |
| [chain-matched-and-aligned.json](chain-matched-and-aligned.json) | Everything together for a long soundtrack-driven take, with per-segment prompts and crash-safe `save_segments` |

## Going long: cuts (digital shorts)

Chained generation fights drift; a scene cut erases it. Each shot in these
examples is generated fresh from the same portraits, so the last shot is exactly
as clean as the first and the piece can run as long as the script does - which is
how television gets away with it too. This is the pattern for digital shorts:
write shots, not takes. What a cut gives up is audio: H3 carries nothing between
generations except what is passed as a reference, so each shot generates its own
sound and a score would restart at every cut. Write `non_diegetic_music: N/A` in
every shot and lay one track under the finished edit, as `music-video.json` does
with `pair_audio` and [assemble-and-score.json](../assemble-and-score.json) shows
on its own; a character who speaks in several shots keeps one voice by passing the
same clip as an audio reference in each shot, as
[voice-timbre-reference.json](voice-timbre-reference.json) does.

| Example | What it introduces |
| ------- | ------------------ |
| [dialogue-short.json](dialogue-short.json) | A five-shot sitcom scene: Z-Image draws the cast, `pipeline_reference` reruns one loaded model per shot, `concat_videos` splices the episode |
| [music-video.json](music-video.json) | A music video cut to a generated song: `slice_audio` deals frame-exact pieces to lip-synced shots, and `pair_audio` lays the unbroken track over the finished edit |
