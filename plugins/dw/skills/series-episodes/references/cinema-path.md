# Series episodes: the cinema path

The default `dialogue-short` shots are the TV path (960x544, turbo), whose
two-character faces smear (~60 px). The cinema path renders every shot at
1344x768 on the base model at int8, no LoRA; the `minimax-h3` skill's
quality-and-speed reference owns the settings, timings (CUDA
measurements), the `first_block` lever and the base-model prompt rules - read
it, not restated here. What a multi-shot episode adds:

- Wides as a 12-step draft, then a 30-step final; close-ups at 12 steps.
  One int8 job at a time. A five-shot episode ran ~156 GPU-min with three
  rejected drafts.
- Place every subject explicitly in every shot and name the set dressing, or
  the base model adds a person and each shot draws its own chair.
- Four references at 1344x768 cap a shot at 124 frames: write tags to 124 or
  drop a reference.
- Assemble at full size, no `fit`: `concat_videos` with `trim_frames: 0`,
  `audio_bleed_ms: 0`, `match_levels: "rms"`, a looped room-tone bed, then
  `normalize_audio` peak -1 (one episode: -16.7 LUFS).
- TV-path and cinema-path episodes look like different shows back to back:
  choose the path once, at episode 1, and keep it unless the user says
  otherwise.
- Choose the path once, at episode 1: TV-path and cinema-path episodes look like different shows back to back.
