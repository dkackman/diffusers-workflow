---
name: minimax-h3
description: Use when a dw MCP server is connected and the user wants MiniMax H3 video - a clip with its own sound, dialogue or a speaking character, a music video, a multi-shot short with cuts, a longer take, or a consistent subject/voice from a reference. Picks the template, states the rules that bite, quotes cost, points at MiniMax's prompt guides.
---

# MiniMax H3 on a dw server

H3 generates video and audio together: speech with lip sync, ambient sound,
score. Every template fits a 24 GB card. This skill picks the template and
arguments; prompt format is MiniMax's, from its text not here.

## Before anything

1. `get_server_info`: the device (H3 is CUDA; quantized configs don't
   run on mps) and the session's workspace.
2. `list_workflows(shape="shot")`, `list_workflows(shape="sequence")` and
   `list_workflows(shape="audio")`: the family's templates by name,
   `summary`, `traits`, `cost`. Trust the listing over names below.
3. `get_workflow` on the one chosen: its variables and defaults.

## Which shape is the request

- **One clip, up to 14.4 seconds, from text**: `templates/minimax/video-with-audio`
  (960x544, fast), or `templates/minimax/video-with-audio-768p` to render.
  To keep a 544p take but at 1344x768, `templates/minimax/upscale-refine`
  with its prompt and seed: its base pass is `video-with-audio`'s, refined
  4 steps at 1344x768 and keeping that take's audio. Native 768p re-rolls it.
  From a one-line idea: `templates/minimax/enhance-prompt` writes it with the
  built-in Context-IR enhancer first.
- **Pinned to a picture**: first frame `templates/minimax/image-to-video`;
  first and last `templates/minimax/first-and-last-frame`; last only
  `templates/minimax/last-frame-only`; a one-line idea plus a picture
  `templates/minimax/enhance-prompt-with-image`.
- **Pinned to a clip** (hold earlier footage, restyle a take): a `guides`
  argument on `t2va`/`fl2va` - up to 4 clips, each at a `frame` that is a
  multiple of 17. Rules and example: `get_guide("workflows",
  section="H3: holding a clip with `guides`")`.
- **A consistent subject**: `templates/minimax/reference-to-video`
  (an image fixes appearance, an audio clip voice);
  `templates/minimax/composable-references` adds a video reference for
  framing and camera, at ~3.4x cost;
  `templates/minimax/generated-subject-reference` draws the subject with
  Z-Image, references it in the same workflow;
  `templates/minimax/voice-timbre-reference` fixes a voice from a Bark line.
- **Several boards, one generation, unbroken score**:
  `templates/minimax/storyboard` - H3 cuts between boards in one
  generation, which no concat matches. One beat only; past that use the
  cuts pattern below.
- **Longer than 14.4 seconds**: chain when one action or line of speech
  crosses the seam, cut when the scene changes.
- **As one take (a chain)**: `templates/minimax/chained-segments`
  (last-frame continuity), `templates/minimax/chain-video-continuity`
  (previous segment's tail rides as a video reference - motion, camera,
  voice carry the seam), `templates/minimax/chain-matched-to-audio` (a track
  sets the length, muxed back seamless), `templates/minimax/chain-matched-and-aligned`
  (per-segment prompts). Drift compounds per seam: reference the subject
  picture each segment, prefer `last_segment` continuity, use the longest
  segments memory allows.
- **A piece with cuts**: fresh shots from shared portraits, then a concat.
  `templates/minimax/dialogue-short` (Z-Image draws the cast, one shot per
  `shots` entry, `concat_videos` splices) and `templates/minimax/music-video`
  (a song, one slice and one lip-synced shot per entry, padded then trimmed
  to its cut - `from_file` reuses an existing cast portrait and skips
  drawing).
  Before composing one, read `references/cuts.md`: the `shots` list (a music
  video's six fields, plan then render), cost per entry, scoring and one voice across cuts.
- **Dialogue into a song**, not a concat: `slice_audio`, `join_into_song`,
  `pair_audio`, in that reference's last section.
- **Unrelated shots, no cut**: `templates/minimax/shots-batch` - one H3
  step per `shots` entry, no shared cast, no concat. `keep_output` each
  clip, then `templates/assemble-and-score` cuts and scores.
- **Music alone**: `templates/minimax/music` (Music3); the `minimax-music3`
  skill - its `audio_duration` is a ceiling, not a length.

If none fits, compose from `list_tasks` before writing a new workflow and
read the `workflows` guide's authoring section.

## Hard rules

- `num_frames` is `17n + 5`, from 124 to 345, at a fixed 24 fps: 5.17 to 14.4
  seconds in one clip. Most default to 124 for fast iteration (storyboard 192);
  `num_frames=345` is the full length, and fits 24 GB at 544p only (#271).
  The 5-second floor is diffusers'; the model card says 4.
- Canvas: 768-pixel short edge, at most 768x1344, in multiples of 32, aspect
  1:4 to 4:1. Output audio is 32 kHz stereo.
- A checkpoint comes with a canvas, a sigma shift and an alpha and they move
  together - change one, change all. Before swapping one or adding a LoRA,
  read `references/checkpoints-and-loras.md`: the tested combinations,
  the LoRAs that need their own loaders, and the catalog.
  Never put an FL2VA LoRA on a reference template: `ref2va` holds
  `transformer_ref` alone, so anything handed there degrades output.
  `validate_workflow` refuses it and warns on a `weight_name` naming neither.
- Fit a crowd's action to what it holds: candle-holders asked to clap
  rendered three-handed people.
- A directed crowd move needs its landmark in frame: "marches up to the
  house" put it behind camera. Track from the side, landmark ahead.
- Two people in frame can lip-sync the wrong one, seemingly by whichever
  voice enters first: credit a shout to the crowd, spell the sung line in
  `<d>` tags, frame the singer face-on at medium close-up.
- State a crowd's age range; unstated skews young.
- A three-reference shot can open on the portraits' backdrop
  for ~1.5s (about 1 in 25): check a shot's first frames and say in
  `subject_definitions` that the portrait's backdrop, pose and framing are
  not reused.
- Ref2VA reaches 1344x768 too, not only the FL2VA 768p template - see the
  `recipes` guide's MiniMax-H3 section for the trade-off. At that canvas the
  24 GB ceiling on the `17n+5` grid drops with each reference: 243 frames at
  one, 209 at two, 175 at three, 141 at four. `validate_workflow` refuses
  the largest `for_each` shot over budget.
- Nine steps for an eight-step LoRA: the scheduler counts sigma grid points,
  terminal zero included, so `denoise_total_steps` reports 8. A null
  `lora_model_name` drops the LoRA; raise steps and shifts too.
- Nothing carries between generations except a passed reference or a
  held track: this pipeline has no latent memory or extension mode (a
  diffusers limit, not the model's). Identity rides on a picture, voice
  timbre on an audio clip, motion/camera on a video tail, score across
  cuts under concat.
- Lip sync to supplied audio: pass the track as an audio reference, as
  `music-video` and the `match_audio` chains do. `hold_audio` keeps the
  track exact and draws picture to it, but measured on a sung track the
  mouth followed it worse (open at 3 of 6 onsets vs 6 of 6); use it
  opt-in, for picture that must fit audio exactly, not for singing.
- H3 is guidance-distilled: no `guidance_scale` or negative prompt.
- Keep `release_pipeline` where the template puts it (frees Z-Image before
  H3 loads, H3 before a concat), or a warm worker may SIGKILL near the end.
- Ref2VA limits: at most 9 images, 3 videos, 3 audio clips, 12 files; audio can
  never be the only reference. References are labelled in order.
- Timestamps in the prompt span the generated length.

## Prompts

H3 wants Context-IR, MiniMax's format. `get_prompt` shows the shape;
the rules are MiniMax's:

1. If the `h3-prompt-writing` skill is installed, use it. If not, say once
   that `npx skills add MiniMax-AI/MiniMax-H3 --skill h3-prompt-writing`
   installs it and go on without it.
2. Else read the model card's guides:
   https://huggingface.co/MiniMaxAI/MiniMax-H3/raw/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md
   for text and frame conditioning, and
   https://huggingface.co/MiniMaxAI/MiniMax-H3/raw/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md
   for reference conditioning.
3. Else run `templates/minimax/enhance-prompt` (or `-with-image`), whose
   built-in enhancer writes the format from those guides. Its `idea` is
   framed as `Task: T2VA. Duration: 5.17 seconds. Idea: ...`.

Either way: write the whole script before the first shot. Repeat a speaker's voice description verbatim across shots and when a
reference picture should fix identity but not framing, say so in the prompt -
or every shot inherits the portrait's composition.

## Run and judge

1. `validate_workflow` first - free, catches rejected arguments.
2. Quote `plan.estimate` from the validate answer (warm minutes; a first
   load or `downloads_required` is longer). When `basis` is `unknown`,
   say so and give the shape: a 124-frame turbo clip is a few minutes
   on a 24 GB card, 345 frames three times that, an image reference twice a
   turbo clip, a video reference 3.4x again, a chain times its segments.
   Get the go-ahead, then `run_workflow` with `acknowledged_cost` = the
   plan's `{fingerprint, minutes, downloads}`.
3. `wait_for_job` with `timeout_seconds` = the estimate plus a margin
   (`timeout_capped` says the server's cap cut it; call again while
   `still_running`), then `get_job` for the manifest. A cancelled H3 job runs
   to its next step boundary. Silence is no hang: `denoise_step` is null
   through the reference encode (~90 s; 629 s for a video reference on a
   3090) and the block cache makes later steps uneven -
   two-minute gaps are healthy. `phase_stall` in `get_job_events` narrates
   it, not a fault; judge by `denoise_step`. Each entry carries `subfolder`:
   `final` is the deliverable (`episode`, `music_video`, `voyage`),
   `intermediate` the scratch.
4. Judge: `get_output_frames(count=12)` for a clip's shape,
   `seams=true` (each later shot's start frame) for a cut's joins - a
   character that changes between shots (reference the same portraits
   everywhere), a portrait imposing its framing on every shot - `at` late in
   a chain for drift sharpening to noise, and `get_output_audio` for a
   voice-over without affect. It returns sound, not text; to confirm a
   line, transcribe (`templates/transcribe-audio`, `get_output_text`). Dialogue gaps (`shot_dead_air`) need room tone:
   `find_loop_bed` the `output:` cut, then `slice_audio`->`loop_audio`->`mix_audio`
   its pick at its `gain`.
   Then `get_gallery_metadata` for duration/audio presence and hand the
   user the gallery `url` (`list_gallery`). `get_output_image` is for image
   steps only.
5. After a run worth keeping, `get_job_workflow` and `save_workflow` it;
   `export_job` bundles it on the server. `auth_required: false` - fetch
   `open_url` into `exports/` (never a temp dir). `true` - hand `open_url`
   to the person and keep using `get_output_image`/`_audio`/`_frames`.

## Sources

MiniMax-H3 model card and prompt guides, `h3-prompt-writing`, diffusers'
MiniMax-H3 modular pipeline, `lightx2v/Minimax-h3-Turbo` LoRA notes; read
2026-09-07; audit `docs/proposals/audits/2026-09-07-minimax-h3-audit.md`.
