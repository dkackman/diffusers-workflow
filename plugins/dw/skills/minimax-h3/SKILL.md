---
name: minimax-h3
description: Use when a dw MCP server is connected and the user wants MiniMax H3 video - a clip with its own sound, dialogue or a speaking character, a music video, a multi-shot short with cuts, a longer take, or a subject or voice kept consistent from a reference. Picks the template for the shape, states the rules that bite, quotes cost, and points at MiniMax's own prompt guides.
---

# MiniMax H3 on a dw server

H3 generates video and audio together: speech with lip sync, ambient sound,
score. Every template here fits a 24 GB card. This skill chooses the template
and the arguments; the prompt format is MiniMax's, from their text not here.

## Before anything

1. `get_server_info`: the device (H3 templates are CUDA; the quantized
   configurations do not run on mps) and which workspace this session is in.
2. `list_workflows(shape="shot")`, `list_workflows(shape="sequence")` and
   `list_workflows(shape="audio")`: the family's templates by their current
   names, with `summary`, `traits` and `cost`. Trust the listing over the
   names quoted below.
3. `get_workflow` on the one chosen, for its variables and their defaults.

## Which shape is the request

- **One clip, up to 14.4 seconds, from text**: `templates/minimax/video-with-audio`
  (960x544, fast), or `templates/minimax/video-with-audio-768p` for the render.
  From a one-line idea: `templates/minimax/enhance-prompt` writes it with the
  built-in Context-IR enhancer first.
- **Pinned to a picture**: first frame `templates/minimax/image-to-video`;
  first and last `templates/minimax/first-and-last-frame`; last only
  `templates/minimax/last-frame-only`; a one-line idea plus a picture
  `templates/minimax/enhance-prompt-with-image`.
- **A subject that must look the same**: `templates/minimax/reference-to-video`
  (an image fixes appearance, an audio clip voice);
  `templates/minimax/composable-references` adds a video reference for framing
  and camera, at about 3.4x the cost;
  `templates/minimax/generated-subject-reference` draws the subject with
  Z-Image first and references it in the same workflow;
  `templates/minimax/voice-timbre-reference` fixes a voice from a Bark-spoken line.
- **Several boards in one generation, one unbroken score**:
  `templates/minimax/storyboard` - H3 cuts between the boards inside a single
  generation, which no concat of separate clips can match for continuous audio.
  It is one beat with fixed cut points, not a building block; past one beat
  with a recurring cast, use the cuts pattern below.
- **Longer than 14.4 seconds**: decide first whether the seam is a cut or a
  continuation. Chain when the same action or line of speech crosses the seam;
  cut when the scene changes, and treat each cut as its own generation. Six
  distinct scenes are a cuts piece, not a chain.
- **Longer than 14.4 seconds as one take**: a chain. `templates/minimax/chained-segments`
  (last-frame continuity), `templates/minimax/chain-video-continuity` (the
  previous segment's tail rides along as a video reference - motion, camera and
  voice carry across the seam), `templates/minimax/chain-matched-to-audio` (a
  supplied track sets the length and is muxed back seamless),
  `templates/minimax/chain-matched-and-aligned` (all of it, per-segment
  prompts). Drift compounds per seam: reference the subject picture in every
  segment, prefer `last_segment` continuity, use the longest segments memory
  allows.
- **A piece with cuts**: fresh shots from shared portraits, then a concat.
  `templates/minimax/dialogue-short` (Z-Image draws the cast, one shot per
  `shots` entry on one loaded model, `concat_videos` splices) and
  `templates/minimax/music-video` (a song, one slice and one lip-synced shot
  per entry; its singer is the `singer_reference` argument - a `from_file`
  reference uses a cast portrait that exists and elides the drawing).
  `shots` is one list argument: a dialogue entry is `name`,
  `prompt`, `references` (portraits and voices: `from_previous_result` for
  one drawn here, `from_file` for an `asset:` cast that already exists) and
  `num_frames`; a music-video entry is `name`, `prompt` and `start_frame`.
  A six-shot piece is one more entry, not another file. The listing's
  `lists` block says what an entry carries; its `cost` carries
  `per_entry` when one shot was measured: quote
  `minutes - per_entry.minutes × per_entry.entries + per_entry.minutes × N`
  for N entries. Without `per_entry`, quote the total and say it is the
  default list's.
  A cut erases drift: the last shot is as clean as the first. Each shot
  generates its own audio, so write
  `non_diegetic_music: N/A` in every shot and lay one score under the concat
  afterwards: `templates/minimax/music` writes the track and
  `templates/assemble-and-score` shows the `pair_audio` step that mixes it
  under the world sound (it takes three shots; for more, author the concat
  and score steps the same way). A character who speaks in several
  shots keeps one voice by passing the same clip as an audio reference in each
  (the `voice-timbre-reference` pattern); a repeated description alone drifts.
  That clip carries delivery as well as timbre, and Bark's presets are
  conversational - for gravitas, `upload_asset` a read in that register.
  Each entry's `num_frames` is its own, so pace the cut.
- **Music alone**: `templates/minimax/music` (Music3); the `minimax-music3` skill.

If none fits, compose from `list_tasks` before authoring a new workflow, and
read the `workflows` guide's authoring section first.

## Hard rules

- `num_frames` is `17n + 5`, from 124 to 345, at a fixed 24 fps: 5.17 to 14.4
  seconds in one clip. Most default to 124 for fast iteration (storyboard 192);
  `num_frames=345` is the full length and fits the same 24 GB configuration.
  The 5-second floor is diffusers'; the model card says 4.
- Canvas: 768-pixel short edge, at most 768x1344, in multiples of 32, aspect
  1:4 to 4:1. Output audio is 32 kHz stereo.
- A checkpoint comes with a canvas, a sigma shift and an alpha, and they move
  together - change one, change all. `video_shift`, `audio_shift` and
  `lora_alpha` are variables everywhere, so a swap is arguments, not a file.
  Three combinations are tested, nothing else: 544p FL2VA turbo, 960x544,
  shift 12/3, alpha unset - the default; 768p FL2VA turbo, 1344x768, shift
  **6**/3, **alpha 128** - `video-with-audio-768p`; 768p Ref2VA turbo, shift
  12/3, alpha unset - every `ref2va` template. The two 768p LoRAs differ in
  shift; do not generalise. Never put an FL2VA LoRA on a reference template:
  a `ref2va` step holds `transformer_ref` alone, diffusers routes whatever it
  is handed there, and nothing complains - the output just degrades.
- Nine steps for an eight-step LoRA: the scheduler counts sigma grid points,
  terminal zero included, so `denoise_total_steps` reports 8. Expected. Do not
  read upstream's `--inference-steps 8` literally.
- Nothing carries between generations except what is passed as a reference:
  no latent memory and no extension mode, in the checkpoint, the API or
  diffusers. Identity rides on a picture, voice on an audio clip, motion and
  camera on a video tail (what a chain passes forward), and a score across
  cuts is laid under the concat.
- H3 is guidance-distilled: no `guidance_scale`, no negative prompt. Say what
  is there, not what is not.
- When deriving a variant, keep `release_pipeline` where the template puts it:
  it frees the Z-Image boards before H3 loads. A run SIGKILLed near the end in
  a warm worker that succeeds in a fresh one is host memory, not the prompt.
- Ref2VA limits: at most 9 images, 3 videos, 3 audio clips, 12 files; audio can
  never be the only reference. References are labelled in the order passed.
- Music3 reads `audio_duration` as a ceiling, not a target: ask for more than
  the song needs and trim with `templates/audio-trim-fade`.
- Write the prompt for the length generated: timestamps should span the
  duration, or a five-second script conditions a five-second story whatever
  the frame count.

## Prompts

H3 wants Context-IR, MiniMax's own format. Do not invent it and do not
paraphrase it from examples:

1. If the `h3-prompt-writing` skill is installed (MiniMax ships it in
   https://github.com/MiniMax-AI/MiniMax-H3 under `skills/`), use it. If not,
   say once that
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
order, read once, should carry the piece - then place them. Repeat a speaker's
voice description verbatim across shots, and when a reference picture should
fix identity but not framing, say so in the prompt itself, in the lines that
define the subject and what each reference keeps - or every shot inherits the
portrait's composition.

## Run and judge

1. `validate_workflow` first - free, and it catches arguments the pipeline
   rejects.
2. Quote `plan.estimate` from the validate answer (warm minutes; a first
   load or a `downloads_required` is longer). When `basis` is `unknown`,
   say so and give the shape instead: a 124-frame turbo clip is a few minutes
   on a 24 GB card, 345 frames three times that, an image reference twice a
   turbo clip, a video reference 3.4x again, a chain times its segments.
   Get the go-ahead, then `run_workflow` with `acknowledged_cost` = the
   plan's `{fingerprint, minutes, downloads}`.
3. `wait_for_job`, then `get_job` for the manifest. A cancelled H3 job runs
   on to its next step boundary, minutes on this model. Silence is no hang:
   `denoise_step` is null through the reference encode (~90 s; 629 s for one
   5 s 960x544 video reference on a 3090), and the block cache makes later
   steps uneven - two-minute gaps are healthy. Each entry carries
   `subfolder`: `final` is the deliverable (`episode`, `music_video`,
   `voyage`), `intermediate` the scratch; keep that split in anything you
   compose.
4. You cannot watch a video: no tool returns a frame from one. Hand the user
   the gallery `url` (`list_gallery`, or the manifest's file name) and ask them
   to look, and check what you can yourself - `get_job` for the manifest and
   its warnings, `get_gallery_metadata` for duration, size and whether audio
   is present. `get_output_image` works only on image steps - in this family
   the Z-Image portraits and boards of `dialogue-short`, `storyboard`,
   `generated-subject-reference` and `music-video`. Ask them to look for the
   family's failure modes: a character that changes between shots (reference
   the same portraits everywhere), a portrait imposing its framing on every
   shot, a storyboard skipped, drift sharpening into noise late in a chain, a
   voice-over without affect (the reference's delivery came through), every
   shot the same length, one look word on every board softening all of them.
5. After an inline run worth keeping, `get_job_workflow` and `save_workflow`
   it, so the next run is by name rather than pasted JSON; `export_job` bundles
   the run — workflow, manifest, job row and media — for git. It is on the
   server: fetch its zip URL and unpack it into `exports/` under the session's
   working directory, never a temp dir; the archive unpacks into a job-id
   folder, so do not make one first.

## Sources

MiniMax-H3 model card and prompt guides (huggingface.co/MiniMaxAI/MiniMax-H3),
the `h3-prompt-writing` skill (github.com/MiniMax-AI/MiniMax-H3), the diffusers
MiniMax-H3 modular pipeline, the `lightx2v/Minimax-h3-Turbo` LoRA notes. Read
2026-09-07; the audit is `docs/proposals/audits/2026-09-07-minimax-h3-audit.md`.
