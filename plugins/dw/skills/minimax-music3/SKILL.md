---
name: minimax-music3
description: Use when a dw MCP server is connected and the user wants music from MiniMax Music 3 - a song with sung lyrics, an instrumental, a score to lay under a film or a cuts piece, or the soundtrack a music video is cut to. Picks the template for the shape, states the ceiling and tag rules that bite, quotes cost, and points at MiniMax's own caption skill for the prompt.
---

# MiniMax Music 3 on a dw server

Music 3 writes a whole track from two strings: a caption that says how it
sounds and a lyrics body that says what happens section by section. It runs on
a dw server as one modular pipeline step, and the catalog already holds the
shapes; do not author a new workflow until the shape decision below fails.

## Before anything

1. `get_server_info`: the device. Music 3 is a CUDA model; on an `mps` or `cpu`
   server stop and say so.
2. `list_workflows(shape="audio")`, and `shape="sequence"` for the music video.
   Trust the names quoted below only after the listing confirms them.
3. `get_workflow` on the one chosen, for its variables and their defaults.

## Which shape is the request

- **A song**: `templates/minimax/music`. Lyrics with section tags plus a caption
  that names the vocal gender and timbre. This is the default shape.
- **An instrumental**: the same template with a tag-only lyrics body
  (`[intro]`, `[instrumental]`, `[solo]`, `[outro]`, one per line) and a caption
  that says instrumental and names the instrument carrying the lead. The
  diffusers pipeline has no instrumental flag and rejects empty lyrics, so the
  tags are how it is asked for.
- **A score under a film or a cuts piece**: an instrumental generated to a
  ceiling comfortably longer than the cut, trimmed and faded with
  `templates/audio-trim-fade`, then mixed under the picture the way
  `templates/assemble-and-score` does with `pair_audio`. Each H3 shot should
  have written `non_diegetic_music: N/A` so the two scores do not fight.
- **A music video**: `templates/minimax/music-video`. The song is written
  first, `slice_audio` deals frame-exact pieces to lip-synced H3 shots, and
  `pair_audio` lays the unbroken track back over the edit. The ceiling must
  exceed the total sliced length with real margin, not by a fraction of a
  second, since the model may stop early.

If none fits, compose from `list_tasks` (`slice_audio`, `fade_audio`,
`pair_audio`, `mix_audio`, `concat_videos`) before authoring, and read the
`workflows` guide's authoring section first.

## Hard rules

- `audio_duration` is a ceiling, not a target: the language model stops when
  the song ends, and generation is cut at the ceiling if it has not. Default
  60 seconds. Ask for more than the piece needs and trim; the run's time
  follows the length actually generated, so the margin is free.
- The engine caps a track at 9000 frames at 25 frames per second, 360 seconds.
  MiniMax supports five minutes; stay at or under 300.
- The caption is capped at 5,000 tokens and a longer one is an error, not a
  truncation. A 250-450 word caption is nowhere near it; a pasted pile of
  example captions is.
- Lyrics: a section tag stands alone on its line and is lower-cased on the way
  in; any words on the same line as a leading tag are dropped. The vocabulary
  on the model card is `[Intro]`, `[Verse]`, `[Pre-Chorus]`, `[Chorus]`,
  `[Post-Chorus]`, `[Bridge]`, `[Instrumental]`, `[Solo]`, `[Outro]`. A tag can
  carry a local direction for its section; the caption's arrangement text is
  where that direction is spelled out.
- Name the vocal gender and timbre in the caption, or the model may drift
  instrumental. An instrumental says so in the same place.
- Output is 44.1 kHz stereo, the vocoder's native rate. The model card's
  32 kHz is what MiniMax's reference server resamples to; the templates' result
  declares `sample_rate: 44100` because the modular output carries no rate of
  its own.
- The flow stage denoises in 200-frame windows at 30 steps each, with
  classifier-free guidance fixed at 1.7 on the pipeline's guider. Neither is a
  call argument; leave them unless a template exposes `num_inference_steps`.
- Memory: about 22 GB with the templates' auto CPU offload; the language model
  can be group-offloaded to run in 8 GB. A workflow that runs Music 3 before
  H3 frees it with `release_pipeline` first, as `music-video` does.
- Section tags and the caption are generative control, not guarantees: the
  vendor says tempo, key, structure and lyrics may not match every detail.
  Iterate at 30-60 seconds before asking for a long track.

## Prompts

The caption format is MiniMax's own. Do not invent it and do not paraphrase it
from the templates' examples:

1. If the `music-caption-rewriter` skill is installed (MiniMax ships it in
   https://github.com/MiniMax-AI/MiniMax-Music3 under `skills/`, with a genre
   router, 18 family indexes and 1,000 example captions), use it. If it is
   not, tell the user once that
   `npx skills add MiniMax-AI/MiniMax-Music3 --skill music-caption-rewriter`
   installs it; this is the family where the local skill pays off, since the
   templates are what a fetch of `SKILL.md` alone does not reach.
2. Else read its `SKILL.md` and `references/genre-router.md` at that path. The
   contract is three headings in order - Global Metadata, Vocal Details,
   Arrangement - in 250-450 English words, with no title, no reasoning, and
   no lyric line copied into the caption. The pipeline strips markdown
   headings and emphasis on the way in, so the vendor's caption pastes
   straight into `prompt`.
3. The concise one-paragraph form the templates' stored prompts use (genre,
   BPM, key, emotional progression, listening scenario, production profile,
   vocals, arrangement) is the model card's own example and works; the
   three-heading form is for precise control.

Lyrics carry the structure: tags for the sections, the words to sing under
them, and nothing else. The tag list is the model card's "Fine-Grained Music
Control" section.

## Run and judge

1. `validate_workflow` first - free, and it catches arguments the pipeline
   does not accept.
2. Quote the listing's `cost` (warm minutes on the card it was measured on;
   a first load is longer). When the listing declares none, say so and give
   the shape of the spend: the autoregressive stage runs at 25 frames per
   second of audio and dominates, so time scales with the length the model
   actually sings, not the ceiling. Get the user's go-ahead before
   `run_workflow` with `acknowledged_cost=true`.
3. `wait_for_job`, then `get_job` for the manifest.
4. You cannot listen: no tool returns audio inline. Hand the user the gallery
   `url` (`list_gallery`, or the manifest's file name) and check what you can
   yourself - `get_gallery_metadata` for the file's duration against the
   ceiling: `media.duration_seconds` within 0.2 s of `audio_duration` means
   the ceiling cut the track (raise it and rerun); well short of it means the
   song finished on its own. Also check the sample rate. Ask the user to
   listen for the family's failure
   modes: a song that went instrumental (name the vocals in the caption), an
   ending cut mid-note (raise the ceiling, then trim), a structure that ignored
   the tags (fewer sections, plainer directions).
5. To use the track in a later workflow, `keep_output` makes it an `asset:`;
   to trim it in the same run, chain `templates/audio-trim-fade` on the output.
6. After an inline run worth keeping, `get_job_workflow` and `save_workflow` it,
   so the next run is by name rather than by pasting JSON; `export_job` bundles
   the run — workflow, manifest, job row and media — for git. The bundle is on
   the server: fetch its zip URL and unpack it into `exports/` under the
   session's working directory, never a temp directory, and do not make a
   folder named after the job id first, since the archive already unpacks
   into one.

## Sources

MiniMax-Music3 model card (https://huggingface.co/MiniMaxAI/MiniMax-Music3,
2026-08): the tag vocabulary, the Structured Caption headings, the five-minute
range, the VRAM tiers. The `music-caption-rewriter` skill in
https://github.com/MiniMax-AI/MiniMax-Music3 (2026-09-08): the caption
contract. The diffusers `minimax_music3` modular pipeline and its docs page
(diffusers 0.41, 2026-09-08): the ceiling semantics, the 9000-frame and
5,000-token caps, the 200-frame window, the guider, the 44.1 kHz output, the
drift-instrumental tip. The repo's audit of all of it is
`docs/proposals/audits/2026-09-08-minimax-music3-audit.md`.
