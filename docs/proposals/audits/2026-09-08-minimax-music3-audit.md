# MiniMax Music 3 — knowledge audit of `diffusers-workflow`

Research date: 2026-09-08. Repository: `/Users/don/src/dkackman/diffusers-workflow` (branch `dw-plugin`).
Installed engine: diffusers `0.41.0.dev0` at
`/Users/don/src/dkackman/diffusers-workflow/venv/lib/python3.14/site-packages/diffusers`.

---

## Sources

| URL | Type | Date | Contributes |
| --- | --- | --- | --- |
| `venv/.../diffusers/modular_pipelines/minimax_music3/modular_blocks_minimax_music3.py` | PRIMARY (code, as installed) | diffusers 0.41.0.dev0, read 2026-09-08 | `audio_duration` default/ceiling docstring, `num_inference_steps` default 30, 200-frame window, lyrics tag contract, 44.1 kHz claim, `audios` output shape |
| `venv/.../minimax_music3/encoders.py` | PRIMARY (code) | read 2026-09-08 | Special-token prompt template, `_MAX_PROMPT_TOKENS = 5_000`, `_MAX_AUDIO_FRAMES = 9_000`, `_AR_CFG_SCALE = 1.5`, `_AR_CFG_TOP_K = 50`, `_AR_SAMPLING_TOP_K = 50`, `_normalize_lyrics`, `_clean_caption` markdown stripping |
| `venv/.../minimax_music3/before_denoise.py` | PRIMARY (code) | read 2026-09-08 | `_CHUNK_FRAMES = 200`, `_CHUNK_HOP = 100` |
| `venv/.../minimax_music3/denoise.py` | PRIMARY (code) | read 2026-09-08 | `ClassifierFreeGuidance` default `{"guidance_scale": 1.7}`, `num_inference_steps` default 30, `_OVERLAP_LATENT_LENGTH = 172` |
| `venv/.../minimax_music3/decoders.py` | PRIMARY (code) | read 2026-09-08 | `_CROP_LEFT_LATENT = 86`, `_CROP_RIGHT_LATENT = 344 - 86`, stereo stitch |
| `venv/.../minimax_music3/modular_pipeline.py` | PRIMARY (code) | read 2026-09-08 | `sampling_rate` = 44100, `frame_rate` = 25.0 Hz, `latent_hop_length` = 512, `num_channels_latents` = 128 |
| `venv/.../models/autoencoders/minimax_music3_vocoder.py` | PRIMARY (code) | read 2026-09-08 | `sampling_rate: int = 44100` (line 85), stereo `(batch, 2, samples)` |
| `venv/.../models/condition_embedders/condition_embedder_minimax_music3.py` | PRIMARY (code) | read 2026-09-08 | `input_sampling_rate=24000`, `input_hop_length=960` (→ 25 fps), `output_sampling_rate=44100`, `output_hop_length=512` |
| `venv/.../models/transformers/transformer_minimax_music3.py` | PRIMARY (code) | read 2026-09-08 | `in_channels: int = 128` |
| https://huggingface.co/MiniMaxAI/MiniMax-Music3/raw/main/README.md | PRIMARY (vendor model card, fetched verbatim by curl) | fetched 2026-09-08; card content dates from the 2026-08 launch | Five-minute max, tag vocabulary, Structured Caption contract, `max_new_tokens` @ 25 fps ceiling semantics, 5,000-token / 9,000-frame limits, VRAM tiers, 32 kHz WAV for the SGLang server, diffusers snippet with `audio_duration=60.0` |
| https://huggingface.co/MiniMaxAI/MiniMax-Music3/raw/main/vocoder/config.json | PRIMARY (checkpoint config) | fetched 2026-09-08 | `"sampling_rate": 44100`, `latent_channels: 128` |
| https://huggingface.co/MiniMaxAI/MiniMax-Music3/tree/main | PRIMARY (repo tree) | fetched 2026-09-08 | **No `docs/` or `skills/` directory on the HF repo**; only `assets/`, `figures/`, `scripts/`, and component folders |
| https://api.github.com/repos/MiniMax-AI/MiniMax-Music3/git/trees/main?recursive=1 | PRIMARY (repo tree) | fetched 2026-09-08 | 1,037 paths; `skills/music-caption-rewriter/` with `SKILL.md`, `references/genre-router.md` + 18 family indexes, **1,000** caption templates. No `docs/` directory |
| https://raw.githubusercontent.com/MiniMax-AI/MiniMax-Music3/main/skills/music-caption-rewriter/SKILL.md | PRIMARY (vendor agent skill) | fetched 2026-09-08 | The authoritative caption contract: three headings, instrumental handling, English default, 250–450 words, "never reproduce lyrics", precedence rules |
| https://raw.githubusercontent.com/MiniMax-AI/MiniMax-Music3/main/skills/music-caption-rewriter/references/genre-router.md | PRIMARY | fetched 2026-09-08 | 18 style families, alias/fusion routing rules |
| https://raw.githubusercontent.com/MiniMax-AI/MiniMax-Music3/main/skills/music-caption-rewriter/templates/acoustic-folk-singer-songwriter_0001.txt | PRIMARY (vendor example caption) | fetched 2026-09-08 | Concrete shape of a full Structured Caption |
| https://raw.githubusercontent.com/MiniMax-AI/MiniMax-Music3/main/skills/README.md | PRIMARY | fetched 2026-09-08 | Install path (`npx skills add MiniMax-AI/MiniMax-Music3 --skill music-caption-rewriter`), progressive-disclosure design |
| https://huggingface.co/docs/diffusers/main/en/api/pipelines/minimax_music3 | PRIMARY (diffusers docs) | fetched 2026-09-08 | The decisive 44.1 kHz vs 32 kHz statement, the "may drift instrumental" tip, CFG 1.7 swap recipe, ~23 GB VRAM, 25 fps dominates runtime |
| https://platform.minimax.io/docs/guides/music-generation | PRIMARY (vendor API docs) | fetched 2026-09-08; notice dated 2026-08-20 | Hosted API only: `music-3.0`, lyrics 10–1000 chars, `is_instrumental`, `lyrics_optimizer`, 44100 Hz / 256 kbps MP3, tags incl. `[Hook]`; **paid music API closed to new users from 2026-08-20** |
| https://www.minimax.io/blog/minimax-music-3-0-next-generation-open-weights-production-ready-versatile-music-model | PRIMARY (vendor blog) | published 2026-08-13 | Launch claims: five minutes, structured-caption philosophy, Mandarin + English demos |
| https://minimax3.com/blog/how-to-use-minimax-music-3 | SECONDARY | published 2026-08-14 | Mostly restates the card; original advice: iterate at 30–60 s before requesting a long track; "brief = how it sounds, lyrics = what happens per section" |
| https://www.minimax-music.com/minimax-music-3-prompt-guide, https://minimaxmusic3.ai/, https://minimax-ai.chat/models/minimax-music-3-0/ | SECONDARY (aggregators) | surfaced 2026-09-08 | All restate the model card's tag list and Structured Caption; no independent claims. Not relied on |
| https://huggingface.co/MiniMaxAI/MiniMax-Music3/discussions/17 | SECONDARY (community) | opened ~2026-08-15 ("24 days ago") | A community-authored system prompt; **no MiniMax staff reply**. Not authoritative |

Failed fetches: `https://github.com/huggingface/diffusers/blob/minimax-music3-integration/docs/source/en/api/pipelines/minimax_music3.md` → HTTP 404 (the integration branch is gone; the doc is now on `main` and was read there instead).

---

## Claim-by-claim verdict

### `workflows/templates/minimax/README.md`

| # | Claim | Verdict |
| --- | --- | --- |
| 1 | "music generation with [MiniMax-Music3](https://huggingface.co/MiniMaxAI/MiniMax-Music3)" | **CONFIRMED** — repo id correct, model card is live. |
| 2 | "fitted onto a single 24GB consumer GPU… Every example here runs on an RTX 3090" | **CONFIRMED** — model card "Low VRAM": full precision fits under 24 GB; auto CPU offload ≈22 GB. diffusers docs: "The full pipeline needs ~23 GB of VRAM in bfloat16." The templates use `enable_auto_cpu_offload`, the card's ~22 GB path. |
| 3 | The prompt-format paragraph ("The prompt format is MiniMax's own. Two guides… `docs/VIDEO_PROMPT_WRITING_GUIDE_*`… `skills/h3-prompt-writing`") | **CONFIRMED for H3, but the README never says the equivalent for Music3.** Music3 has its own vendor prompt authority — `skills/music-caption-rewriter` in `github.com/MiniMax-AI/MiniMax-Music3` — and the README does not mention it. See *Missing knowledge*. (There is no `docs/` folder in either the HF or the GitHub Music3 repo.) |
| 4 | music.json row: "The minimal modular pipeline: a `components_manager` owns device placement, and the output is audio, not video" | **CONFIRMED** — matches the model card's Low-VRAM snippet exactly (`ComponentsManager` + `enable_auto_cpu_offload` + `load_components(dtype=bfloat16)`), and `MiniMaxMusic3Blocks.outputs` is `audios` only. |
| 5 | "Music3 reads it as a ceiling rather than a target. The piece ends where the music ends" | **CONFIRMED, twice over.** `modular_blocks_minimax_music3.py`: "`audio_duration` … Upper bound on the generated audio length in seconds. The language model may stop earlier. Capped at 9000 frames (six minutes)." diffusers docs Tips: "`audio_duration` is an upper bound — the language model may end the song earlier with a stop token." Model card, SGLang path: "`max_new_tokens` sets the maximum number of audio frames at 25 frames per second. Generation may finish before this limit when the model emits an end-of-audio token." |
| 6 | "a value set to the length the song *should* be will guillotine the outro mid-decay" | **UNSOURCED (mechanism inverted).** No source describes truncation at the ceiling; the documented behaviour is the opposite — the model emits a stop token and the run *ends short*. A hard cut can happen when the model has not finished by `max_frames` (`encoders.py` breaks at `len(frame_hiddens) >= max_frames`), so the advice "ask for more time than the song needs" is *sound*, but the stated reason ("will guillotine the outro") is the repo's own inference, not a sourced claim. |
| 7 | "trim the tail afterwards - [templates/audio-trim-fade.json]" | **CONFIRMED (repo-internal)** — `workflows/templates/audio-trim-fade.json` exists. Note that `music.json`'s description points at a `tasks/` TrimFadeAudio path, which does **not** exist (see #14). |
| 8 | "Output audio is 32 kHz stereo" (canvas paragraph) | **CONFIRMED but scoped to H3, not Music3.** The sentence sits in the H3 canvas note and is correct there. If a reader carries it to Music3 it is wrong: the Music3 diffusers path emits 44.1 kHz (see #16). |
| 9 | music-video.json row: "a music video cut to a generated song: `slice_audio` deals frame-exact pieces to lip-synced shots, and `pair_audio` lays the unbroken track over the finished edit" | **CONFIRMED (repo-internal)** — the workflow does exactly that. No vendor bearing. |

### `workflows/templates/minimax/music.json`

| # | Setting / description sentence | Verdict |
| --- | --- | --- |
| 10 | `component_type: "ModularPipeline"`, `from_pretrained_arguments.model_name: "MiniMaxAI/MiniMax-Music3"` | **CONFIRMED** — the model card's own snippet is `ModularPipeline.from_pretrained("MiniMaxAI/MiniMax-Music3")`. (`MiniMaxMusic3ModularPipeline` — the class named in `docs/QUANTIZATION.md:187` and `docs/WORKFLOW_GUIDE.md:797` — is also real and resolves via `modular_config_dict`.) |
| 11 | `load_components.dtype: torch.bfloat16`; `components_manager.enable_auto_cpu_offload: true` | **CONFIRMED** — verbatim the card's Low-VRAM recipe. |
| 12 | Arguments `prompt`, `lyrics`, `audio_duration`, `output: "audios"` | **CONFIRMED** — exact match to the card's and the docs page's `pipe(prompt=…, lyrics=…, audio_duration=…, output="audios")`. |
| 13 | `"'audio_duration' is a ceiling, not a target: the model stops where the music stops"` | **CONFIRMED** — same citations as #5. |
| 14 | `"finish the tail with a tasks/ TrimFadeAudio path"` | **CONTRADICTED (repo-internal, stale path).** No such file exists under workflows/tasks; the file is `workflows/templates/audio-trim-fade.json`, which is what the README correctly names. |
| 15 | `"The result declares 'sample_rate' because a modular pipeline's dict output carries none of its own."` | **CONFIRMED (engine behaviour).** `MiniMaxMusic3Blocks.outputs` returns only the raw `audios` tensor; the rate lives on the pipeline (`MiniMaxMusic3ModularPipeline.sampling_rate`), not in the output. |
| 16 | `result.sample_rate: 44100` | **CONFIRMED — and this is the subtle one.** `minimax_music3_vocoder.py:85` `sampling_rate: int = 44100`; the checkpoint's `vocoder/config.json` is `"sampling_rate": 44100`; `condition_embedder_minimax_music3.py:40` `output_sampling_rate: int = 44100`. The **model card says "32 kHz, 16-bit stereo WAV"**, which is the SGLang server's resampled response format, not the diffusers output. The diffusers docs page settles it: "The pipeline returns the vocoder's native 44.1 kHz stereo output. The reference server additionally resamples to 32 kHz; apply your own resampling if you need that exact rate." The repo is **right**, and right for a non-obvious reason worth recording. |
| 17 | `result.content_type: "audio/mp3"` | **UNSOURCED / engine choice.** No vendor bearing. The vendor's own outputs are WAV (SGLang) or MP3 at 44100 Hz / 256 kbps (hosted API), so MP3 at 44.1 kHz is a defensible container choice, but the tensor is lossless and the repo re-encodes. |
| 18 | Stereo output implied by the audio result | **CONFIRMED** — `MiniMaxMusic3Vocoder.forward` returns `(batch, 2, samples)`; "the two audio channels are decoded as two folded `latent_channels // 2` streams". |
| 19 | Lyrics variable: `[verse]` / `[chorus]` tags each alone on a line, lower-case | **CONFIRMED.** `encoders.py::_normalize_lyrics` keeps only leading bracketed tags on a tag line ("text on the same line as a leading tag is dropped"), lower-cases every tag, and prepends `[start]`. Model card tag list: `[Intro]`, `[Verse]`, `[Pre-Chorus]`, `[Chorus]`, `[Post-Chorus]`, `[Bridge]`, `[Instrumental]`, `[Solo]`, `[Outro]`. The template uses only `[verse]`/`[chorus]` — legal, but the narrowest slice of the vocabulary (no `[intro]`, `[bridge]`, `[outro]`), which sits oddly beside the description's advice about the outro decaying. |
| 20 | `audio_duration: 120` (two minutes) | **CONFIRMED as legal** — under the 9,000-frame / 360 s engine cap and the vendor's five-minute maximum. Consistent with the ceiling advice (the lyrics are ~4 sections, far under two minutes). |
| 21 | `num_inference_steps` not set (falls to the default) | **CONFIRMED as the reference value** — default 30 in both `MiniMaxMusic3ChunkSetTimestepsStep` and `MiniMaxMusic3ChunkDenoiseInner`. |
| 22 | No `guidance_scale` set | **CONFIRMED as correct** — flow-stage CFG is a *guider* config, not a call argument: `ComponentSpec("guider", ClassifierFreeGuidance, config=FrozenDict({"guidance_scale": 1.7}))`. It cannot be passed as an argument; it is swapped with `pipe.update_components(guider=…)`. |
| 23 | `cost`: RTX 3090 / 24 GB / 3.6 minutes | **UNSOURCED** (repo measurement). Plausible: the docs page says "The autoregressive stage generates 25 frames per second of audio and dominates the runtime", so runtime tracks *generated* length, not `audio_duration`. |
| 24 | Prompt `prompt:minimax/acoustic_pop_song` — the "Genre: … BPM: … Key: … Vocals: … Arrangement: …" single-paragraph style | **CONFIRMED as a valid, vendor-shown form** — the model card's and the docs page's own diffusers example uses exactly this inline form ("Genre: acoustic pop. BPM: 96. Key: C major. … Vocals: … Arrangement: …"). It is the *concise* form; the vendor's recommended form for precise control is the three-heading Structured Caption. The prompt also names vocal gender and timbre ("soft female lead, close-mic breathy"), satisfying the docs tip in #26. |

### `workflows/templates/minimax/music-video.json` (Music3 parts only)

| # | Setting / sentence | Verdict |
| --- | --- | --- |
| 25 | `write_song` step: same pipeline class, arguments, `components_manager`, `content_type: audio/mp3`, `sample_rate: 44100`, `release_pipeline: true` | **CONFIRMED** — identical to #10–#12, #16, #18. `release_pipeline` is required here, per `docs/RECIPES_24GB.md:99` (host RAM, not VRAM, is the binding constraint before H3 loads). |
| 26 | `song_prompt` names a "smooth male crooner lead" | **CONFIRMED as best practice** — diffusers docs Tips: "The music description controls the vocals: describe the vocal gender and timbre explicitly (e.g. 'warm female vocal') or the model may drift instrumental." Both repo prompts do this. |
| 27 | `song_lyrics`: `[verse]` + `[chorus]`, tags on their own lines | **CONFIRMED** — same contract as #19. |
| 28 | `audio_duration: 21` against a `soundtrack` slice of 496 frames at 24 fps (= 20.667 s) and four 124-frame slices at offsets 0/124/248/372 | **CONTRADICTED by the repo's own rule.** `audio_duration` is a ceiling; the model may stop early. The workflow needs at least 20.667 s of *actual* audio and asks for a ceiling of 21 s — a 0.33 s margin. The README and `music.json` both say "ask for more time than the song needs"; this template asks for essentially exactly what it needs. An early stop, or a song that simply ends at 18 s, leaves `slice_4` and the `soundtrack` slice reading past the end of the waveform. A ceiling of ~30 s with the same 496-frame slicing would follow the repo's own advice. |
| 29 | `"MiniMax-Music3 writes the song, 'slice_audio' cuts it into frame-exact pieces (124 frames at 24 fps each)"` | **CONFIRMED (arithmetic)** — 124 / 24 = 5.167 s per slice; `sample_rate: 44100` is used for the slicing, matching #16. Note the two frame rates in play: Music3 generates at **25 Hz** internally, H3 renders at **24 fps**; slicing is done in H3 frames against the 44.1 kHz waveform, which is correct. |
| 30 | `"The generation models pass through one at a time - each is released before the next loads - so the workflow peaks no higher than its largest single model."` | **CONFIRMED (repo-internal)** — `release_pipeline: true` on `draw_singer` and `write_song`; consistent with `docs/RECIPES_24GB.md:95-101`. |

### Prompt library

| # | Claim | Verdict |
| --- | --- | --- |
| 31 | `prompts/minimax/acoustic_pop_song.json`, `otter_soul_song.json` — inline `Genre/BPM/Key/Emotional progression/Listening scenario/Production profile/Vocals/Arrangement` style | **CONFIRMED as well aligned.** These field names track the model card's Global Metadata list almost item for item ("genre, subgenre, BPM, key, scale, emotional progression, listening scenario, and production profile"), and cover Vocal Details and Arrangement. They are the card's vocabulary in the card's *concise* rendering. `intended_model: "minimax-music"` and the `music` tags are repo conventions. |
| 32 | No prompt in the library uses the three-heading Structured Caption form | **Gap, not an error.** `encoders.py::_clean_caption` explicitly strips markdown headings, bullets, bold and italics from the caption, so a markdown three-heading caption is *safe to paste* — the vendor's recommended form is fully supported and the library does not exercise it. |

### Docs mentions

| # | Claim | Verdict |
| --- | --- | --- |
| 33 | `docs/QUANTIZATION.md:187` / `docs/WORKFLOW_GUIDE.md:797` use `"component_type": "MiniMaxMusic3ModularPipeline"` with per-component `quantization_config` on `transformer` and `text_encoder` | **PARTLY CONTRADICTED.** The class name is real. But Music3 has **no component named `text_encoder`** — the pipeline's components are `tokenizer`, `language_model`, `rvq_depth_decoder`, `condition_encoder`, `transformer`, `scheduler`, `guider`, `vocoder` (`MiniMaxMusic3Blocks` docstring). The doc snippet reads as illustrative rather than runnable; as written the `text_encoder` entry would name nothing. The language model is the one worth quantizing/offloading (the card group-offloads `pipe.language_model` to reach 8 GB). |
| 34 | `docs/RECIPES_24GB.md:99` "a workflow that ran another model first (Z-Image drawing a subject, Music3 writing a song) must free it with `release_pipeline`" | **CONFIRMED (repo measurement)** — internally consistent with the templates; no vendor bearing. |
| 35 | `plugins/dw/skills/minimax-h3/SKILL.md:54,82` — "Music alone: `templates/minimax/music` (Music3)" and the ceiling rule | **CONFIRMED**, same citations as #5. Music3 currently exists in the plugin only as a footnote inside the H3 skill. |
| 36 | `docs/TASKS.md` | **No Music3 mentions** (grep clean). Not an error; `slice_audio`/`pair_audio`/trim-fade are documented model-agnostically. |

### Claims the task asked about that the repository does **not** make

- **Maximum duration**: unstated anywhere in the templates. Vendor: **five minutes** (model card, blog 2026-08-13). Engine: `_MAX_AUDIO_FRAMES = 9_000` at 25 fps = **360 s / six minutes** (`encoders.py`, and the docstring's "Capped at 9000 frames (six minutes)"). These disagree by a minute; the engine cap is the hard stop, the vendor's five minutes is the supported/trained range. Only the internal planning docs (`docs/proposals/agent-catalog-legibility.md:649`, `docs/superpowers/plans/2026-09-07-ltx-h3-catalog-repair.md:1062`) carry the 9000/six-minute figure, and they carry it without the five-minute caveat.
- **Full structure-tag vocabulary**: UNSOURCED in the repo — it only ever shows `[verse]`/`[chorus]`.
- **Instrumental-only**: not addressed anywhere in the repo. Vendor position (see below).
- **Language support**: not addressed. Vendor: blog demos are Mandarin and English; the caption skill says write the *caption* in English by default; no published language list.
- **Steps / guidance defaults**: not stated in any template or doc.
- **Prompt-token limit**: not stated. `_MAX_PROMPT_TOKENS = 5_000` raises a hard `ValueError`.

---

## Missing knowledge

Everything below is published, primary, and absent from the repository.

1. **The `music-caption-rewriter` agent skill** (GitHub `MiniMax-AI/MiniMax-Music3/skills/music-caption-rewriter`, current as of 2026-09-08). This is the exact Music3 counterpart of the H3 prompt-writing guides the README already points at, and the README does not mention it. It ships `SKILL.md`, a `references/genre-router.md`, **18 family indexes**, and **1,000 full caption templates**. Installable with `npx skills add MiniMax-AI/MiniMax-Music3 --skill music-caption-rewriter`. *(Repo gap dated 2026-09-08.)*
2. **The Structured Caption contract** (model card, 2026-08; SKILL.md "Output Contract"): exactly three top-level headings, in order — **Global Metadata** (genre/subgenres, tempo, emotional progression, sonic and production profile; exact BPM/key only when justified), **Vocal Details** (lead configuration, timbre, register, delivery, harmony, restrained FX), **Arrangement** (a section-by-section timeline: what enters, exits, changes, intensifies). Target **250–450 English words**. Do not include a title, template ID, reasoning trace, or any copied lyric line. The repo's prompts are the concise form only.
3. **The full tag vocabulary**: `[Intro]`, `[Verse]`, `[Pre-Chorus]`, `[Chorus]`, `[Post-Chorus]`, `[Bridge]`, `[Instrumental]`, `[Solo]`, `[Outro]` (model card, 2026-08; diffusers docs Tips lower-case the same list). The hosted API additionally lists `[Hook]` (platform.minimax.io, fetched 2026-09-08) — hosted-only, unverified for the open weights.
4. **Tags are directives, not just markers**: a section tag can change its *local* arrangement without replacing the global genre, and the caption skill is instructed to preserve section-local directives attached to lyric tags in the Arrangement text while the lyric text itself stays in `lyrics` (SKILL.md, 2026-09-08). Nothing in the repo says this.
5. **Instrumental music is supported, and how it is expressed**: SKILL.md — "Preserve an explicit instrumental request. Do not add vocals"; the Vocal Details heading must "state that the piece is instrumental and identify the instrument or texture carrying the lead melodic role". The hosted API has a dedicated `is_instrumental` flag (platform docs), **which the diffusers path does not expose** — `MiniMaxMusic3TokenizeStep.check_inputs` *requires* a non-empty `lyrics` string. So in this repo an instrumental means a tag-only `lyrics` (e.g. `[intro]\n[instrumental]\n[solo]\n[outro]`) plus a caption that says instrumental. *(2026-09-08; the diffusers-side consequence is an inference from `encoders.py`, not a vendor statement.)*
6. **The opposite failure mode — accidental instrumental**: diffusers docs Tips (fetched 2026-09-08): "describe the vocal gender and timbre explicitly … or the model may drift instrumental." The repo's prompts happen to do this; nothing tells an author it is mandatory.
7. **Prompt-token ceiling of 5,000** with a hard error (`encoders.py`, model card Limitations). A 450-word Structured Caption is nowhere near it, but a pasted 1,000-template dump would be.
8. **Flow-stage CFG is 1.7 and lives on the guider**, changed only via `pipe.update_components(guider=ClassifierFreeGuidance(guidance_scale=...))` (diffusers docs Tips + `denoise.py`). The AR stage has its own fixed 1.5 that is not user-settable.
9. **Runtime scales with generated length, not with `audio_duration`** — "The autoregressive stage generates 25 frames per second of audio and dominates the runtime" (diffusers docs). This is what makes the "ask for more than you need" advice cheap; the repo asserts the advice without the reason.
10. **The 8 GB path**: `apply_group_offloading(pipe.language_model, offload_type="leaf_level", use_stream=True)` (model card Low VRAM + docs). The repo's 24 GB framing never mentions that Music3 alone runs on far less.
11. **The 32 kHz vs 44.1 kHz distinction** — the vendor advertises 32 kHz, the diffusers pipeline emits 44.1 kHz, and the difference is a resample in the reference server (diffusers docs, 2026-09-08). Anyone reading the model card and the repo side by side will think one of them is wrong. Nothing in the repo explains it.
12. **Vendor five-minute maximum vs the engine's 9,000-frame / six-minute cap** (model card vs `encoders.py`). Undated conflict; both current as of 2026-09-08.
13. **Explicit vendor limitation**: "Section tags and music descriptions provide generative control rather than strict symbolic guarantees. The generated tempo, key, instrumentation, lyrics, and song structure may not always match every requested detail exactly" (model card Limitations, 2026-08). The repo's prompts specify exact BPM and key with no caveat.
14. **Inference requires CUDA** (model card Limitations, 2026-08). The diffusers docs snippet says `pipe.to("cuda")  # or "mps", "xpu", "cpu"`, so the two disagree; on this Mac the repo would fall back under `resolve_device()` with unknown results. Untested either way.
15. **The hosted music API closed to new users on 2026-08-20** (platform.minimax.io notice). Anything the repo might later want to borrow from the API surface (`is_instrumental`, `lyrics_optimizer`, cover mode) is now open-weights-only territory.
16. **Iterate short** — secondary but repeated: "Start with 30 or 60 seconds, fix the brief, and only then request a longer track" (minimax3.com, 2026-08-14). Consistent with #9. `music.json`'s default of 120 s is a slow first iteration.
17. **No newer checkpoint** — as of 2026-09-08 the only published open-weights music checkpoint is `MiniMaxAI/MiniMax-Music3`; the hosted model id is `music-3.0` and `music-cover`. No `Music3.1`/`Music4` found.

---

## What a skill should teach

**Shape first.** A *song* is lyrics with tags plus a caption naming vocal gender and timbre — the default. An *instrumental* is a tag-only `lyrics` body (`[intro]`/`[instrumental]`/`[solo]`/`[outro]`) plus a caption that says instrumental and names the lead instrument; diffusers has no `is_instrumental` flag and rejects empty lyrics. A *score under a film* is an instrumental generated to a ceiling comfortably longer than the cut, then trimmed and faded with `templates/audio-trim-fade`. A *music video* is `music.json` first, then frame-exact `slice_audio` into fresh H3 shots — and the ceiling must exceed the total sliced length with real margin, not by a third of a second.

**Hard numbers, each pinned to a constant.**
- `audio_duration` is an upper bound; the model may stop earlier — `MiniMaxMusic3AutoregressiveStep` InputParam docstring; default `60.0`.
- Frame rate 25 Hz — `MiniMaxMusic3ModularPipeline.frame_rate`, = `input_sampling_rate 24000` / `input_hop_length 960`.
- Hard cap `_MAX_AUDIO_FRAMES = 9_000` = 360 s; vendor supports five minutes. Prefer 300 s.
- Prompt cap `_MAX_PROMPT_TOKENS = 5_000` — hard `ValueError`.
- Output 44,100 Hz **stereo** — `minimax_music3_vocoder.py:85`, `vocoder/config.json`; the card's "32 kHz" is the reference server's resample.
- `num_inference_steps` default 30 per 200-frame window (`_CHUNK_FRAMES = 200`, `_CHUNK_HOP = 100`).
- Flow CFG 1.7 on the guider, not an argument (`ComponentSpec("guider", …, {"guidance_scale": 1.7})`).
- Tag lines: leading bracketed tags only; text on a tag line is dropped (`_normalize_lyrics`).
- ~22 GB with auto CPU offload; 8 GB with `language_model` leaf-level group offload.

**Point, don't transcribe.** Send the author to `skills/music-caption-rewriter/SKILL.md` and `references/genre-router.md` in `github.com/MiniMax-AI/MiniMax-Music3` for the three-heading caption contract, the 18-family routing, and the 1,000 templates; to the model card's "Fine-Grained Music Control" for the tag vocabulary; to the diffusers docs Tips for the drift-instrumental warning. Say only that markdown headings are stripped safely by `_clean_caption`, so the vendor's caption pastes straight in.

---

## Assessment

The repository is largely right, and right about the hard parts. The `audio_duration`-as-ceiling rule is confirmed verbatim by three independent primary sources. `sample_rate: 44100` is correct against the checkpoint config even though the model card says 32 kHz — the repo picked the right number for a reason it never records, and should. The pipeline wiring, `components_manager` recipe, `release_pipeline` discipline, and the prompt vocabulary all track the vendor's own examples.

Four things need correction. `music.json` points at a `tasks/` TrimFadeAudio path, which does not exist. `music-video.json` sets `audio_duration: 21` for 20.667 s of slices, violating the very margin rule it states. The README's "will guillotine the outro mid-decay" asserts a mechanism no source describes — the documented behaviour is an early stop. And `docs/QUANTIZATION.md`/`WORKFLOW_GUIDE.md` show a `text_encoder` component Music3 does not have.

The real gap is coverage, not accuracy: Music3 has a full vendor prompt-authority — the `music-caption-rewriter` skill with 1,000 templates — that the repo never mentions, alongside the full tag list, instrumental handling, the five-vs-six-minute ceiling, and the 32/44.1 kHz explanation.
