---
name: minimax-h3
description: Use when a dw MCP server is connected and the user wants MiniMax H3 video - a clip with its own sound, dialogue or a speaking character, a music video, a multi-shot short with cuts, a longer take, or a subject or voice kept consistent from a reference. Picks the template for the shape, states the rules that bite, quotes cost, and points at MiniMax's own prompt guides.
---

# MiniMax H3 on a dw server

H3 generates video and audio together: speech with lip sync, ambient sound,
score. Every template here fits a 24 GB card. This skill chooses the template
and the arguments; the prompt format is MiniMax's and comes from their text,
not from here.

## Before anything

1. `get_server_info`: the device (H3 templates are CUDA; the quantized
   configurations do not run on mps) and which workspace this session is in.
2. `list_workflows(shape="shot")`, `list_workflows(shape="sequence")` and
   `list_workflows(shape="audio")`: the family's templates by their current
   names, with `summary`, `traits` and `cost`. Trust the listing over the
   names quoted below.
3. `get_workflow` on the one chosen, for its variables and their defaults.

## Which shape is the request

- **One clip, up to 14.4 seconds, from text**: `templates/minimax/video-with-audio`.
  From a one-line idea: `templates/minimax/enhance-prompt` writes the prompt
  with the built-in Context-IR enhancer first.
- **Pinned to a picture**: first frame `templates/minimax/image-to-video`;
  first and last `templates/minimax/first-and-last-frame`; last only
  `templates/minimax/last-frame-only`; a one-line idea plus a picture
  `templates/minimax/enhance-prompt-with-image`.
- **A subject that must look the same**: `templates/minimax/reference-to-video`
  (an image fixes appearance, an audio clip fixes voice);
  `templates/minimax/composable-references` adds a video reference for framing
  and camera; `templates/minimax/generated-subject-reference` draws the subject
  with Z-Image first and references it in the same workflow;
  `templates/minimax/voice-timbre-reference` fixes a voice from a Bark-spoken line.
- **Several boards in one generation, one unbroken score**:
  `templates/minimax/storyboard` - H3 cuts between the boards inside a single
  generation, which no concat of separate clips can match for continuous audio.
- **Longer than 14.4 seconds as one take**: a chain. `templates/minimax/chained-segments`
  (last-frame continuity), `templates/minimax/chain-video-continuity` (the
  previous segment's tail rides along as a video reference - motion, camera and
  voice carry across the seam), `templates/minimax/chain-matched-to-audio` (a
  supplied track sets the length and is muxed back seamless),
  `templates/minimax/chain-matched-and-aligned` (all of it, per-segment prompts).
  Drift compounds per seam: reference the subject picture in every segment,
  prefer `last_segment` continuity, and use the longest segments memory allows.
- **A piece with cuts**: fresh shots from shared portraits, then a concat.
  `templates/minimax/dialogue-short` (Z-Image draws the cast, one loaded model
  per shot, `concat_videos` splices) and `templates/minimax/music-video`
  (shots cut to a generated song, lip-synced slices). A cut erases drift; the
  last shot is as clean as the first. Write shots, not takes.
- **Music alone**: `templates/minimax/music` (Music3).

If none fits, compose from `list_tasks` before authoring a new workflow, and
read the `workflows` guide's authoring section first.

## Hard rules

- `num_frames` is `17n + 5`, from 124 to 345, at a fixed 24 fps: 5.2 to 14.4
  seconds in one clip. Most templates default to 124 for fast iteration (storyboard uses 192); `num_frames=345`
  is the full length and fits the same 24 GB configuration. The 5-second floor
  is diffusers'; the model card says 4.
- Canvas: a 768-pixel short edge, at most 768x1344 pixels, dimensions in multiples of 32,
  aspect from 1:4 to 4:1. Output audio is 32 kHz stereo.
- The text- and frame-conditioned templates render at 960x544 with the 544p
  turbo LoRA in nine steps, and those three go together - change one, change
  all three. The six reference-conditioned templates carry no LoRA and run 20
  steps, because the turbo LoRA is distilled against the base transformer and
  they load the reference one; a reference-conditioned run is about twice the
  time of a turbo one at the same length.
- H3 is guidance-distilled: no `guidance_scale`, no negative prompt. Say what is
  there, never what is not.
- Ref2VA limits: at most 9 images, 3 videos, 3 audio clips, 12 files; audio can
  never be the only reference. References are labelled in the order passed.
- Music3 reads `audio_duration` as a ceiling, not a target: ask for more than
  the song needs and trim with `templates/audio-trim-fade`. Cap: six minutes.
- Write the prompt for the length being generated: shot timestamps should span
  the duration, or a five-second script conditions a five-second story
  whatever the frame count.

## Prompts

H3 wants Context-IR, MiniMax's own format. Do not invent it and do not
paraphrase it from examples:

1. If the `h3-prompt-writing` skill is installed (MiniMax ships it in
   https://github.com/MiniMax-AI/MiniMax-H3 under `skills/`), use it.
2. Else read the guides on the model card:
   https://huggingface.co/MiniMaxAI/MiniMax-H3/raw/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md
   for text- and frame-conditioned generation, and
   https://huggingface.co/MiniMaxAI/MiniMax-H3/raw/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md
   for reference-conditioned generation.
3. Else run `templates/minimax/enhance-prompt` (or `-with-image`), whose
   built-in enhancer writes the format from those guides. Its `idea` is
   framed as `Task: T2VA. Duration: 5.17 seconds. Idea: ...`.

Whichever route: repeat a speaker's voice description verbatim across shots,
and when a reference picture should fix identity but not framing, say so in
the prompt itself - in a reference-conditioned request, in the lines that
define the subject and state what each reference keeps - or every shot
inherits the portrait's composition.

## Run and judge

1. `validate_workflow` first - free, and it catches arguments the pipeline
   does not accept.
2. Quote the listing's `cost` (warm minutes on the card it was measured on;
   a first load is longer). When the listing declares none, say so and give the
   shape of the spend instead: a 124-frame turbo clip is a few minutes on a
   24 GB card, the full 345 frames about three times that, a
   reference-conditioned clip about twice a turbo one, and a chain multiplies
   by its segment count. Get the user's go-ahead before `run_workflow` with
   `acknowledged_cost=true`.
3. `wait_for_job`, then `get_job` for the manifest. A cancelled H3 job runs
   on to its next step boundary, minutes on this model.
4. You cannot watch a video: no tool returns a frame from one. Hand the user
   the gallery `url` (`list_gallery`, or the manifest's file name) and ask them
   to look, and check what you can yourself - `get_job` for the manifest and
   its warnings, `get_gallery_metadata` for duration, size and whether an audio
   stream is present. `get_output_image` works only on image steps, which in
   this family are the Z-Image portraits and boards of
   `templates/minimax/dialogue-short`, `templates/minimax/storyboard`,
   `templates/minimax/generated-subject-reference` and
   `templates/minimax/music-video`. Ask the user to look for the family's
   failure modes: a character that changes between shots (reference the same
   portraits in every shot), a reference portrait imposing its framing on every
   shot, a storyboard skipped, drift sharpening into noise late in a chain.

## Sources

MiniMax-H3 model card and prompt guides (huggingface.co/MiniMaxAI/MiniMax-H3),
the `h3-prompt-writing` skill (github.com/MiniMax-AI/MiniMax-H3), the diffusers
MiniMax-H3 modular pipeline, the `lightx2v/Minimax-h3-Turbo` LoRA notes. Read
2026-09-07; the audit is `docs/proposals/audits/2026-09-07-minimax-h3-audit.md`.
