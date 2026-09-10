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
  It is one beat with fixed cut points, not a building block: four of them
  concatenated give twelve equal-length shots and a cast redrawn four times.
  Past one beat with a recurring cast, use the cuts pattern below.
- **Longer than 14.4 seconds**: decide first whether the seam is a cut or a
  continuation. Chain when the same action or line of speech has to cross the
  seam; cut when the scene changes, and treat each cut as its own generation.
  Six distinct scenes are a cuts piece, not a chain.
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
  last shot is as clean as the first. Write shots, not takes. Each shot
  generates its own audio, so write `non_diegetic_music: N/A` in every shot
  and lay one score under the concat afterwards: `templates/minimax/music`
  writes the track and `templates/assemble-and-score` shows the `pair_audio`
  step that mixes it under the world sound (it takes three shots; for more,
  author the concat and score steps the same way). A character who speaks in
  several shots keeps one voice by passing the same clip as an audio
  reference in each (the `voice-timbre-reference` pattern); a repeated voice
  description alone drifts. Each shot's `num_frames` is its own, so pace the
  cut - a trailer builds by varying shot length. The reference carries
  delivery as well as timbre: a flat read gives a flat performance. Bark's
  presets are conversational; for a narrator with gravitas, `upload_asset` a
  recorded read in that register and reference the same file in every shot.
- **Music alone**: `templates/minimax/music` (Music3); the `minimax-music3` skill.

If none fits, compose from `list_tasks` before authoring a new workflow, and
read the `workflows` guide's authoring section first.

## Hard rules

- `num_frames` is `17n + 5`, from 124 to 345, at a fixed 24 fps: 5.17 to 14.4
  seconds in one clip. Most templates default to 124 for fast iteration (storyboard uses 192); `num_frames=345`
  is the full length and fits the same 24 GB configuration. The 5-second floor
  is diffusers'; the model card says 4.
- Canvas: a 768-pixel short edge, at most 768x1344 pixels, dimensions in multiples of 32,
  aspect from 1:4 to 4:1. Output audio is 32 kHz stereo.
- The text- and frame-conditioned templates render at 960x544 with the 544p
  turbo LoRA in nine steps, and those three go together - change one, change
  all three. Six templates that condition on references alone
  (`reference-to-video`, `composable-references`, `voice-timbre-reference`,
  `generated-subject-reference`, `chain-matched-to-audio`,
  `chain-video-continuity`) carry no LoRA and run 20 steps, because the turbo
  LoRA is distilled against the base transformer and they load the reference
  one; such a run is about twice the time of a turbo one at the same length.
  `storyboard`, `dialogue-short`, `music-video` and
  `chain-matched-and-aligned` pass references *and* keep the turbo LoRA at
  nine steps; say nine for those, not 20.
- Nothing carries between generations except what is passed as a reference:
  no latent memory and no extension mode, in the checkpoint, the hosted API or
  diffusers. Identity rides on a picture, voice on an audio clip, motion and
  camera on a video tail (what a chain passes forward), and a score across
  cuts is laid under the concat afterwards.
- H3 is guidance-distilled: no `guidance_scale`, no negative prompt. Say what is
  there, never what is not.
- When deriving a variant, keep `release_pipeline` on the step the template
  puts it on: it frees the Z-Image boards before H3 loads. A run killed by
  SIGKILL near the end, in a worker warm from a previous job, that succeeds
  on a retry in a fresh worker is host memory, not the prompt.
- Ref2VA limits: at most 9 images, 3 videos, 3 audio clips, 12 files; audio can
  never be the only reference. References are labelled in the order passed.
- Music3 reads `audio_duration` as a ceiling, not a target: ask for more than
  the song needs and trim with `templates/audio-trim-fade`. The `minimax-music3`
  skill has the rest of that family.
- Write the prompt for the length being generated: shot timestamps should span
  the duration, or a five-second script conditions a five-second story
  whatever the frame count.

## Prompts

H3 wants Context-IR, MiniMax's own format. Do not invent it and do not
paraphrase it from examples:

1. If the `h3-prompt-writing` skill is installed (MiniMax ships it in
   https://github.com/MiniMax-AI/MiniMax-H3 under `skills/`), use it. If it
   is not, tell the user once that
   `npx skills add MiniMax-AI/MiniMax-H3 --skill h3-prompt-writing` installs
   it - only that skill; the repo's other eight are style packs - and go on
   without it.
2. Else read the guides on the model card:
   https://huggingface.co/MiniMaxAI/MiniMax-H3/raw/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md
   for text- and frame-conditioned generation, and
   https://huggingface.co/MiniMaxAI/MiniMax-H3/raw/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md
   for reference-conditioned generation.
3. Else run `templates/minimax/enhance-prompt` (or `-with-image`), whose
   built-in enhancer writes the format from those guides. Its `idea` is
   framed as `Task: T2VA. Duration: 5.17 seconds. Idea: ...`.

Whichever route: write the whole script before the first shot - the lines in
order, read once, should carry the piece on their own - then place them.
Repeat a speaker's voice description verbatim across shots,
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
   shot, a storyboard skipped, drift sharpening into noise late in a chain,
   a voice-over without affect (the reference's delivery came through), every
   shot the same length, a look word repeated on every board (shallow depth
   of field) softening every shot.
5. After an inline run worth keeping, `get_job_workflow` and `save_workflow` it,
   so the next run is by name rather than by pasting JSON; `export_job` bundles
   the run — workflow, manifest, job row and media — for git. The bundle is on
   the server: fetch its zip URL and unpack it into `exports/` under the
   session's working directory, never a temp directory, and do not make a
   folder named after the job id first, since the archive already unpacks
   into one.

## Sources

MiniMax-H3 model card and prompt guides (huggingface.co/MiniMaxAI/MiniMax-H3),
the `h3-prompt-writing` skill (github.com/MiniMax-AI/MiniMax-H3), the diffusers
MiniMax-H3 modular pipeline, the `lightx2v/Minimax-h3-Turbo` LoRA notes. Read
2026-09-07; the audit is `docs/proposals/audits/2026-09-07-minimax-h3-audit.md`.
