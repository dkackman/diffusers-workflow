# MiniMax-H3 knowledge verification — diffusers-workflow

Research date: 2026-09-07. Repository branch `agent-legibility`, commit b2b9a8b.
No repository file was modified.

## Headline

MiniMax publishes two official prompt-writing guides and an official
`h3-prompt-writing` Skill. The repository's `dw/workflows/h3_context_ir.json`
system prompt is, to a very high degree, a **faithful and compressed
transcription of those two guides** — most sentences are near-verbatim. The
gaps are omissions rather than errors, plus one wholly invented layer
("continuity modes"). The `workflows/templates/minimax/README.md` numeric
conventions check out against the installed diffusers source.

---

## Sources

| URL | Type | Date | Contributes |
| --- | --- | --- | --- |
| https://huggingface.co/MiniMaxAI/MiniMax-H3/raw/main/README.md | PRIMARY | model card, current as fetched 2026-09-07 | Output specs (4–15 s, 24 fps, 32 kHz stereo, 768p default / 2K, 11 languages, aspect ratios), variant/input limits (≤9 images, ≤3 videos, ≤3 audio, ≤12 files), three-module system (Context-IR / Base / Regenerate-2K), CFG-distilled checkpoints, three worked Context-IR outputs (T2VA, I2VA, Ref2VA) as real reference prompts, links to the prompting guides and skills |
| https://huggingface.co/MiniMaxAI/MiniMax-H3/raw/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md | PRIMARY | shipped with the model card | **The** spec for T2VA/I2VA/FL2VA/L2VA: instruction-block sentences verbatim, three core fields, shot/cut rules, the full camera-motion table, speaker IDs, `<d>`/voiceover/`<scenetrans>`/`<cutoff>`, on-screen text, `overall_soundscape` / `non_diegetic_music` rules, four cases |
| https://huggingface.co/MiniMaxAI/MiniMax-H3/raw/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md | PRIMARY | shipped with the model card | **The** spec for Ref2VA: six sections and their order, the four label types, `summary` task-type table, `retention_analysis` marker tables, `detailed_description` rules incl. 350–500 words, speaker/audio-source rules, complete example |
| https://github.com/MiniMax-AI/MiniMax-H3 (`skills/h3-prompt-writing/`) | PRIMARY | last commit touching skills 2026-08-15 (`d21241f`); "Tips for better results" added 2026-08-11 (`a107547`) | Official agent-facing Skill wrapping the two guides; adds duration-matching and label-consistency tips; `references/base-en.txt` and `ref-en.txt` are the same guides |
| https://platform.minimax.io/docs/api-reference/video-generation-v2-h3-context-ir | PRIMARY | fetched 2026-09-07 | Hosted Context-IR endpoint: duration 4–15 (integer), ratio vocabulary, per-modality file limits; **no continuation/extension mode exists** |
| diffusers 0.41.0.dev0, `modular_pipelines/minimax_h3/` (installed in repo venv) | PRIMARY | as installed | `MINIMAX_H3_FPS = 24`, `min_duration = 5.0`, `max_duration = 15.0`, `align_num_frames` (`17n + 5`), canvas rules (multiple of 32, short edge 768, max 768×1344, aspect 1:4–4:1), `reference_image_short_edge = 2048`, reference caps |
| diffusers 0.41.0.dev0, `modular_pipelines/minimax_music3/` | PRIMARY | as installed | `audio_duration` = "Upper bound on the generated audio length… The language model may stop earlier. Capped at 9000 frames (six minutes)" |
| https://huggingface.co/docs/diffusers/main/en/api/pipelines/minimax_h3 | PRIMARY | main branch, fetched 2026-09-07 | "24 fps, 5 to 15 seconds", `17n+5`, 768 short edge, guidance-distilled (no `guidance_scale`, no `negative_prompt`), `num_inference_steps` counts sigma grid points incl. terminal 0, reference ordering is semantic, 960×544 endorsed as the speed lever |
| https://github.com/ModelTC/Minimax-H3-Turbo (README) | PRIMARY (LoRA author) | repo README as fetched 2026-09-07 | Turbo LoRA spec table: task, training resolution, video/audio shifts, distillation NFE, recommended NFE; reference-resize policies (`match` / `max` / `diffusers`) and the recommendation to use `match` |
| https://huggingface.co/api/models/lightx2v/Minimax-h3-Turbo | PRIMARY | `lastModified` 2026-09-04, created 2026-08-07 | Actual file list — includes files newer than the ModelTC README table |
| https://huggingface.co/MiniMaxAI/MiniMax-Music3/raw/main/README.md | PRIMARY | fetched 2026-09-07 | Music3 usage; only shows `audio_duration=60.0`, no ceiling semantics stated |
| https://docs.comfy.org/tutorials/video/minimax/minimax-h3 | SECONDARY (vendor-adjacent, cited by the LoRA authors) | fetched 2026-09-07 | Reference tagging "in the exact order it was connected"; explicit role assignment advice; 20 steps default / ~25 for motion / 8-step turbo; `ref_image_size` match-vs-max; multiframe `frame_idx` guidance |
| https://www.rundiffusion.com/minimax-h3-prompt-guide | SECONDARY | Aug 2026 | Restates the six sections, both marker sets, bracketed prefixes; adds a ten-item failure-mode list and a 7 000-character prompt cap |
| https://deapi.ai/blog/…, https://www.dreampixelforge.com/blog/minimax-h3-prompts, https://minimax-h3-ai.com/blog/minimax-h3-prompt-guide/, https://domoai.app/blog/minimax-h3-prompt-guide, https://leadde.ai/blog/mini-max-h3-prompt-guide | SECONDARY | Aug 2026 | All restate the model card and the two official guides (three-field / six-section split, `[Shot 1]` untimestamped, MM:SS.mmm). One adds an unsourced "350–450 words for complex, 150–250 for simple" figure that conflicts with the official 350–500. No independent information. |
| https://medium.com/…/minimax-h3-…, https://www.runpod.io/blog/minimax-h3-… | SECONDARY | Aug 2026 | Architecture restatement only |

**Fetches attempted and not useful:** no arXiv/technical report for H3 exists as
of 2026-09-07 (searched; only unrelated omni-modal papers returned). The GitHub
mirror of the model repo has no `docs/` directory — the two guides live on the
Hugging Face repo only.

---

## Claim-by-claim verdict — `dw/workflows/h3_context_ir.json`

### Framing

| Claim | Verdict |
| --- | --- |
| "H3-Base is trained to consume this exact format and degrades on anything else" | **PARTLY UNSOURCED.** The model card says H3-Context-IR "is critical to the quality of the final output, so we strongly recommend incorporating it into your generation pipeline or following the 'Prompting Guidance'". It never says the base model degrades on other input, and the base checkpoints accept free text. Directionally right, rhetorically stronger than any source. |
| Output only the rewritten prompt, no commentary | **UNSOURCED** (harness convention, not a model fact). |
| Structure = optional instruction block, one blank line, then the core fields | **CONFIRMED.** base guide §2.1: "The instruction must be the first line of the final prompt, followed by one blank line before the core fields." |

### Instruction blocks

| Claim | Verdict |
| --- | --- |
| T2VA: none, begin with the core fields | **CONFIRMED**, base §2.1. |
| I2VA: `For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.` | **CONFIRMED verbatim**, base §2.1 and Case 2; also verbatim in the model card's own case-I2VA Context-IR output. |
| FL2VA: `How the reference pictures align with the target video - Picture 1 (from Shot 1) …; Picture 2 (from Shot N) aligns with the S.SS-second mark …` | **CONFIRMED**, base §2.1, with one cosmetic deviation: the guide uses an em dash (`—`) after "target video", the repo uses a hyphen. The guide's FL2VA form genuinely drops the angle brackets and square brackets that the L2VA form keeps; the repo reproduces that asymmetry correctly. |
| L2VA: `How the reference pictures align with the target video - <Picture 1> (from [Shot N]) aligns with the S.SS-second mark of the target video.` | **CONFIRMED**, base §2.1 (same em-dash nit). |
| `S.SS` is the duration to two decimals; `N` is the index of the final shot | **CONFIRMED verbatim**, base §2.1. |
| Ref2VA has no instruction block | **CONFIRMED** — ref guide §1 lists six sections beginning at `subject_definitions`; the model card's case-Ref2VA output starts at `subject_definitions:`. |

### Core fields (T2VA/I2VA/FL2VA/L2VA)

| Claim | Verdict |
| --- | --- |
| Order `integrated_multimodal_description` → `overall_soundscape` → `non_diegetic_music`, blank-line separated | **CONFIRMED**, base §2.2. |
| Their three one-line definitions | **CONFIRMED near-verbatim**, base §2.2 and §4.1. |
| **Omission:** the `N/A` convention | **MISSING.** base §4.6: `N/A` for `overall_soundscape` only when the user asks for complete silence; §4.7: `N/A` when there is no non-diegetic music. Both official examples use it. The repo never mentions `N/A`, so a model following the repo prompt will invent a score for a silent brief. |
| **Omission:** length guidance (1–4 sentences soundscape, 1–3 music) and "do not repeat dialogue/singing/diegetic music in `overall_soundscape`" | **MISSING**, base §4.6/§4.7. |
| **Omission:** "do not use abstract mood words or explain the emotional function of the score" in `non_diegetic_music` | **MISSING**, base §4.7. |

### Ref2VA six sections

| Claim | Verdict |
| --- | --- |
| Six sections in order `subject_definitions, summary, retention_analysis, detailed_description, overall_soundscape, non_diegetic_music` | **CONFIRMED**, ref guide §1 table; matches the model card's case-Ref2VA output. |
| `<Subject N>` = reusable visible content (person, animal, object, scene, costume, style, action) | **CONFIRMED**, ref §2.1. |
| `<Picture N>` = concrete frame / composition anchor; an image that only defines a subject is cited inside that subject's line | **CONFIRMED verbatim**, ref §2.2 ("do not create a standalone picture entry"). |
| `<Video N>` = whole-video relationship (edit source, continuation point, temporal structure) | **CONFIRMED verbatim**, ref §2.3. |
| `<Audio N>` = copied or referenced audio; name the speaker ID, `<Audio 1> is the voice-timbre reference for <Subject 1> (S1).` | **CONFIRMED verbatim**, ref §2.4 — the example sentence is identical. |
| "One subject may draw on several assets (`whose appearance comes from <Picture 1> and whose walking motion comes from <Video 1>`)" | **CONFIRMED verbatim**, ref §2.1. |
| Storyboard example `<Picture 3> is a storyboard reference for [Shot 1] and [Shot 2], defining their viewpoint, subject placement, and shot order.` | **CONFIRMED verbatim**, ref §2.2. |
| "A label keeps the same meaning in every later section" | **CONFIRMED**, ref §2 blockquote. |
| "The references appear in the order the request passes them, and that order is what the labels number" | **CONFIRMED but incomplete.** diffusers docs: order "labels them in the prompt presentation" and "a different order is a different request"; ComfyUI R2V tips: "Reference each input by tag in the exact order it was connected". **But** ref §2.5 adds a rule the repo omits: `<Video N>` and `<Audio N>` are numbered *independently within their own category*, so the same source video can be `<Video 1>` and `<Audio 2>`, and an ordinary reference video does not create an `<Audio N>` merely because the file has sound. Read as a single global counter, the repo's sentence is misleading. |

### `summary`

| Claim | Verdict |
| --- | --- |
| Square-bracketed task-type prefix + one short paragraph | **CONFIRMED**, ref §3. |
| The six type names, joined with ` + `, no repeats | **CONFIRMED verbatim** — `keyframe completion`, `reference generation`, `video editing`, `video continuation`, `audio reuse`, `audio reference`, and "combine the task types with ` + ` and do not repeat a type", ref §3. |
| Each type's one-line definition | **CONFIRMED near-verbatim**, ref §3 table. |
| "A video that only lends camera, cuts or rhythm is reference generation, not editing or continuation" | **CONFIRMED verbatim**, ref §3. |
| "A video edit opens with `The target video is an edited version of <Video N>.`" | **CONFIRMED**, ref §3 (guide writes `<Video 1>`; the repo's generalisation to `<Video N>` is correct). |
| **Omission:** "when editing a source video, use `audio reuse` as well if its original audio remains audible"; "when continuing without copying the signal, use `audio reference` if the new audio only continues the original's audible characteristics" | **MISSING**, ref §3. |
| **Omission:** "Do not introduce new reference labels in this section" | **MISSING**, ref §3. |

### `retention_analysis`

| Claim | Verdict |
| --- | --- |
| Visible-content markers `fully_preserved`, `partially_preserved`, `attribute_transfer`, `weak_reference` and their meanings | **CONFIRMED verbatim**, ref §4.1 — including that these are "fixed English values in the output format". |
| Audio markers `fully_copy`, `partially_copy`, `reference`, `weak_reference` and their meanings | **CONFIRMED verbatim**, ref §4.2. |
| Entry formats `<Subject 1> (appears in [Shot 1], [Shot 3]): fully_preserved - …`, `<Picture 2> ([Shot 1] first frame): fully_preserved - …`, `<Audio 1>: partially_copy - …` | **CONFIRMED verbatim** (the first two are the guide's own examples; the audio form matches `<Audio 1>: fully_copy - …`). |
| "Choose the marker within the role `subject_definitions` gave the label; new actions, backgrounds or events are not losses of fidelity" | **CONFIRMED near-verbatim**, ref §4.2 closing paragraph. |
| **Omission:** "Do not write `(Sx)` in `retention_analysis`" | **MISSING**, ref §5.4. This is an explicit prohibition the repo prompt does not carry. |
| **Omission:** the video-structure entry form `<Video 1> (cut and pacing structure): weak_reference - …` | **MISSING** (minor), ref §4.1. |

### `detailed_description`

| Claim | Verdict |
| --- | --- |
| One or two sentences of overall style *before* `[Shot 1]`, with the example `The target video is in a realistic photographic style with soft window light.` | **CONFIRMED** — ref §5.2 table: "Established in one or two English sentences before `[Shot 1]`"; the model card's Ref2VA output opens `The target video is in realistic photographic style.` |
| Then the same shot-by-shot timeline the other tasks put in `integrated_multimodal_description` | **CONFIRMED**, ref §5.1/§5.2. |
| Content checklist: composition, appearance, environment and lighting, actions and state changes, camera movement, sound, and where referenced content takes effect | **CONFIRMED verbatim**, ref guide preamble ("Description detail"). |
| "A generation task runs 350-500 words here" | **CONFIRMED verbatim**, ref §5.2 ("normally 350-500 English words"). Note one secondary source (dreampixelforge/deAPI family) gives 350–450 / 150–250 instead — **CONTRADICTED by the primary guide**; do not adopt it. |
| "Dialogue-dense content fits the whole spoken timeline first" | **CONFIRMED**, ref §5.2. |
| "Never reduce it to a plot summary or a list of reference relationships" | **CONFIRMED verbatim**, ref guide preamble. |
| **Omission:** "Video-editing descriptions scale with the complexity of the source video and do not have to follow the generation-task range" and "A single shot does not automatically justify a shorter description" | **MISSING**, ref §5.2. |
| **Omission:** the natural anchor phrasings `the shot begins from <Picture 1>` / `the shot's keyframe corresponds to <Picture 2>` / `the shot ends on <Picture 3>` | **MISSING**, ref §5.3. |

### Continuity modes (`standalone` / `continuation`)

**UNSOURCED — a repository invention.** Neither guide, the skill, the model card
nor the Context-IR API defines a continuity mode. `standalone` appears in the ref
guide only in the unrelated sense of "a standalone `<Picture N>` entry" / "a
standalone audio asset". The hosted Context-IR API documents no continuation or
segment-extension parameter at all.

The *content* of the repo's continuation mode is nevertheless a defensible
inference: `video continuation` is a real summary prefix (ref §3); marking the
continuation-point video `fully_preserved` is consistent with the §4.1 marker
definitions; and "one continuous shot, no cuts" is sound craft for chained
segments. But the framing — that H3 recognises two continuity modes — is the
repository's own and should be labelled as such in anything that teaches the
format.

### Shot-body rules

| Claim | Verdict |
| --- | --- |
| Open `[Shot 1]` with overall style and initial composition; in Ref2VA the style sentence sits before `[Shot 1]` | **CONFIRMED**, base §4.1 + ref §5.2. |
| Style vocabulary `Cinematic, live-action, 2D-animated, 3D CG, claymation, watercolor, vintage film` | **CONFIRMED verbatim**, base §4.1 (identical list, identical order). |
| Derive style from the reference image when there is one, otherwise from the user's text | **CONFIRMED verbatim**, base §4.1. Mild tension with the official Skill's later tip (2026-08-11): "Prefer concrete visual and audio details over abstract words like 'cinematic' or 'beautiful'." |
| Do not timestamp the first shot | **CONFIRMED**, base §4.2. |
| Later shots open with a strictly increasing cut time inside the duration, e.g. `[Shot 2] At 00:03.500, the camera cuts to …` | **CONFIRMED verbatim**, base §4.2; format `[Shot N] At MM:SS.mmm` per ref §5.1. |
| "A cut must introduce new information; when only framing or angle changes, use camera motion instead" | **CONFIRMED near-verbatim**, base §4.2 ("about the subject, space, state, viewpoint, or time"). The repo drops the enumeration of *what kind* of new information. |
| FL2VA favours a single continuous shot unless asked otherwise | **CONFIRMED**, base §3.2. **Omission:** "The last frame must be reached by the final `[Shot N]` at the end of the video." |
| Continuation mode allows no cut at all | **UNSOURCED** (part of the invented mode). |
| **Omission:** the cut-verb vocabulary — `the camera cuts to`, `the shot cuts to`, `the shot transitions to`, `the shot changes to`, `the shot switches to`; and cross-dissolve / fade / wipe only when the user asks | **MISSING**, base §4.2. |

### Camera motion

**CONFIRMED verbatim and complete.** All 20 motion types in the repo's list
(`Zoom In/Out, Push In, Pull Out, Pan Left/Right, Truck Left/Right, Tilt Up/Down,
Pedestal Up/Down, Arc Shot, Tracking Shot, Static Shot, Shake Slightly/Strongly,
POV, Roll Clockwise/Counterclockwise`) match base §4.3's table exactly, with no
additions and no omissions. The amplitude phrases (`with small amplitude`, `with
large amplitude`), the speed phrases (`at slow speed`, `at fast speed`), the
"omit when medium and normal" rule, and "written as a natural English action
within the shot rather than stacked as labels" are all verbatim.

### Speakers and dialogue

| Claim | Verdict |
| --- | --- |
| Stable IDs `(S1)`, `(S2)`, `(S1,S2)` for unison, kept across shots; non-vocalising characters get none | **CONFIRMED verbatim**, base §4.4. |
| First appearance establishes character type, age, gender, on/off-screen, pitch, timbre, speaking rate, accent | **CONFIRMED verbatim**, base §4.4. |
| Spoken/sung content inside `<d>[Language] …</d>` and nothing else; identity, action, delivery outside; wording and punctuation verbatim; never translate | **CONFIRMED verbatim**, base §4.4. |
| Voiceover uses the exact phrase `says in an off-screen voiceover`, and the text after every voiceover `<d>` block states the lips remain completely closed | **CONFIRMED verbatim**, base §4.4. |
| `<scenetrans>` at both connecting points of a line crossing a cut; `<cutoff>` when speech is truncated by the end of the video | **CONFIRMED**, base §4.4. **Omission:** the guide also requires *explicitly stating that the audio continues across the cut*, and supplies four phrasings (`continues seamlessly across the cut`, `continues uninterrupted into the next shot`, `carries over from the previous shot`, `remains audible across the transition`). |
| "Invent dialogue only when the user asked for speech without supplying any" | **UNSOURCED** (sensible, no source either way). |
| **Omission:** Ref2VA's combined form `<Subject N> (Sx)`, and `off-screen` marking for the same subject | **MISSING**, ref §5.4. |
| **Omission:** `(Sx)` is assigned once by the order of *actual vocal events in the target video*, never renumbered in an `<Audio N>` definition | **MISSING**, ref §5.4. |
| **Omission:** verbal cues that exist only inside a directly reused soundtrack use `<Audio N>` as the audible source and get no `(Sx)` | **MISSING**, ref §5.4. |
| **Omission:** `[unclear]` for unintelligible spans instead of guessing; punctuation standardised to `, . ? !`, decorative punctuation removed, terminal `.`/`?`/`!` before `</d>` | **MISSING**, ref §5.4. |
| **Omission:** "When only timbre, rhythm, emotion or delivery is referenced, do not carry the original dialogue from the reference audio into the target video." | **MISSING**, ref §5.4. |

### On-screen text and the reference image

| Claim | Verdict |
| --- | --- |
| Visible on-screen text in double quotation marks, verbatim, untranslated | **CONFIRMED verbatim**, base §4.5. |
| The description must match what the reference image actually contains; `[Shot 1]` opens on its framing; never contradict it | **CONFIRMED** for I2VA, base §3.1 ("Character identity, clothing, colors, key objects, and spatial relationships should remain consistent"). The continuation-mode variant is unsourced. |
| "Every detail must be something visible or audible" | **CONFIRMED verbatim**, base §4.1. |
| "No camera hardware specs, no artist names, no quality tags such as 8k or ultra-detailed" | **UNSOURCED** — neither guide says this. It is a sensible generalisation of §4.1 but is not in any source. |
| "No negative phrasing — H3 is guidance-distilled and has no negative prompt" | **CONFIRMED** on the mechanism: model card, "The released checkpoints are CFG-distilled Omni Transformer model weights"; diffusers docs, "Both transformer partitions are guidance-distilled … there is no guider, no `negative_prompt` and no `guidance_scale`". The prompt-writing *consequence* is the repo's inference, but a correct one. |

---

## Claim-by-claim verdict — `workflows/templates/minimax/README.md`

| Claim | Verdict |
| --- | --- |
| `num_frames` must be of the form `17n + 5` | **CONFIRMED.** `align_num_frames(num_frames, frames_per_chunk=17, latents_per_chunk=5)` in `modular_pipelines/minimax_h3/modular_pipeline.py`; diffusers docs and ComfyUI tips both state the `17k+5` grid. Note the pipeline *snaps up* rather than rejecting, and warns. |
| Between 124 and 345 | **CONFIRMED.** `min_duration = 5.0`, `max_duration = 15.0`, `fps = 24` → 120–360 frames → the `17n+5` members in range are 124 (n=7) … 345 (n=20). |
| At a fixed 24 fps | **CONFIRMED.** `MINIMAX_H3_FPS = 24`; model card "Output frame rate 24 FPS". |
| "that is 5.17 to 14.4 seconds" | **CONFIRMED.** 124/24 = 5.1667 s, 345/24 = 14.375 s. |
| "in a single clip" | **CONFIRMED.** No extension/continuation mode exists in the model, the API or diffusers. |
| **Nuance:** the model card and the Context-IR API say **4**–15 s | The 4-second floor is reachable only through the hosted API; the open checkpoint as integrated in diffusers enforces 5.0 s. The repo's floor is correct **for its own runtime**, but the README does not say why it differs from the model card, which will confuse anyone reading both. |
| "Write the prompt for the length you are generating" | **CONFIRMED in spirit** by the official Skill's tip (2026-08-11): "Always match the total duration of the description to the requested video length (4–15 seconds)", and by the guides' requirement that cut times fall inside the duration. |
| Music3 `audio_duration` is a ceiling, not a target; the piece ends where the music ends | **CONFIRMED verbatim** by `modular_blocks_minimax_music3.py`: "Upper bound on the generated audio length in seconds. The language model may stop earlier." The Music3 model card itself says nothing about it — diffusers is the only source. |
| **Omission:** `audio_duration` is capped at 9000 frames (six minutes) | **MISSING** from the README. |
| Turbo LoRA `lightx2v/Minimax-h3-Turbo` / `minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors` at `num_inference_steps: 9` | **CONFIRMED as internally correct, but unexplained.** ModelTC's spec table: this file is the **544p mixed-aspect** FL2VA/T2VA 8-step LoRA, recommended NFE 8 (or 4). diffusers docs: "`num_inference_steps` counts sigma grid points, the terminal `0` included, so it drives one model evaluation less" — so 9 grid points = 8 NFE, exactly the recommendation. Neither the README nor any template comment records this reasoning, so the `9` looks arbitrary. |
| Templates' default canvas 960×544 | **CONFIRMED as a legitimate choice.** diffusers docs endorse it explicitly ("960x544 runs about 2.3x faster per step than the trained 1344x768"), and it matches the 544p LoRA's training resolution, which is the *stronger* reason. But the README never states that H3's native canvas is a 768 short edge (1344×768 at 16:9), nor that the chosen LoRA is the 544p variant — so a reader raising the resolution would silently move off the LoRA's training distribution. |
| Chain drift advice (reference the original subject in every segment, `last_segment` continuity, longest segments possible) | **UNSOURCED.** No official source discusses chaining; it is repository craft. Consistent with the diffusers docs' "a generation as a reference" hand-off, which is the mechanism it relies on. |
| "Cuts erase drift; write shots, not takes" | **UNSOURCED** repository craft. Not contradicted by anything. |
| Per-example "what it introduces" descriptions | Accurate against the templates read (`video-with-audio.json`, `reference-to-video.json`, `storyboard.json`, `dialogue-short.json`). |

### One inconsistency found in the templates

`storyboard.json` is a **Ref2VA** request (three references, `<Picture 1>` first
frame plus two storyboard anchors) but loads
`minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors` — the **FL2VA/T2VA** LoRA —
at 9 steps. `dialogue-short.json`, `music-video.json` and
`chain-matched-and-aligned.json` do the same on reference-bearing requests. The
other Ref2VA templates (`reference-to-video`, `composable-references`,
`voice-timbre-reference`, `chain-video-continuity`,
`generated-subject-reference`) correctly use no LoRA at 20 steps. Per ModelTC,
the FL2VA LoRA is distilled against `transformer/`, while `ref2va` runs against
`transformer_ref/`; a dedicated Ref2VA turbo LoRA exists (below).

---

## Missing knowledge

Ordered by how much it would change what the repository does.

1. **The official prompt-writing guides and Skill exist and are not referenced.**
   `docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md`,
   `docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md` (in the model card repo), and
   `skills/h3-prompt-writing/` on GitHub (skills last touched **2026-08-15**).
   The repository's system prompt is derived from them but nothing says so, so
   there is no way to re-verify it or track upstream changes.

2. **A Ref2VA turbo LoRA exists** (`minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors`,
   and `minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors` present in the
   HF repo as of **2026-09-04**, the latter newer than ModelTC's spec table).
   Every Ref2VA template in the repository runs 20 unaccelerated steps.

3. **Newer FL2VA LoRA versions**: `4step_v1.1_768p` and `4step_v1.2_768p`
   (HF repo `lastModified` **2026-09-04**), plus `8step_v1.0_768p` — which
   LightX2V Studio itself deploys, "improved video and audio generation quality"
   (turbo README, **2026-08-27** screenshot). The repository pins the older
   544p 8-step v1.0.

4. **Scheduler shift values are part of the LoRA contract.** ModelTC: video/audio
   shift `12 / 3` for the 544p LoRAs, `6 / 3` for the 768p ones; diffusers ships
   `shift=12.0` video / `shift=3.0` audio by default. Moving to a 768p LoRA
   without changing the shifts is a silent mismatch. Nothing in the repository
   mentions shifts. (**2026-08/09**)

5. **Reference-image resize policy.** ModelTC recommends `match` (reference pixel
   area matched to the target canvas) because that is what distillation trained
   on; diffusers' fixed 2048-pixel short edge is the `diffusers` policy. Also in
   ComfyUI's tips. `storyboard.json` lowers `reference_image_short_edge` ad hoc
   for memory but the repository has no notion of matching it to the canvas.
   (**2026-08/09**)

6. **The `N/A` convention** for `overall_soundscape` and `non_diegetic_music`
   (base guide §4.6/§4.7) — the single most consequential prompt-format omission.

7. **Dialogue-fidelity rules** the repository prompt omits: `[unclear]` for
   unintelligible spans, punctuation standardisation and terminal punctuation
   before `</d>`, "do not carry original dialogue over when only timbre is
   referenced", "do not write `(Sx)` in `retention_analysis`", and the
   `<Audio N>`-as-vocal-source rule for reused soundtracks (ref §5.4).

8. **Cut-verb and audio-continuity vocabularies** (base §4.2, §4.4) — the
   repository teaches one cut phrasing and no continuity phrasings.

9. **`<Video N>` / `<Audio N>` are numbered independently within their own
   category** and a reference video does not automatically produce an
   `<Audio N>` (ref §2.5).

10. **Model-level facts absent from the repository README**: native canvas is a
    768-pixel short edge (max 768×1344), dimensions must be multiples of 32,
    aspect ratios from 1:4 to 4:1, 32 kHz stereo output, 11 stable dialogue
    languages, Ref2VA limits (≤9 images, ≤3 videos, ≤3 audio, ≤12 total; audio
    references can never be the only references), and that H3-Regenerate-2K
    (2K output) is API-only and not open-sourced.

11. **Step-count guidance**: ComfyUI documents 20 steps as the non-turbo default
    and ~25 for better motion quality. The repository's 20 matches the default
    but has no note about the motion trade-off. (**2026-09-07** fetch)

12. **Known failure modes** are documented nowhere in the repository. RunDiffusion
    (secondary, Aug 2026) lists ten, including character inconsistency across
    shots, reference-image dominance over the text, and storyboard skipping.
    Treat as unconfirmed but worth testing.

13. **`audio_duration` is capped at 9000 frames / six minutes** (Music3, diffusers).

14. **No technical report or paper exists** for H3 as of 2026-09-07. Anything
    claiming to cite one is fabricating.

---

## Assessment

The Context-IR spec in `h3_context_ir.json` is **an accurate, compressed
transcription of MiniMax's two official prompt-writing guides**, not an
inference from examples. Sentence after sentence — the I2VA/FL2VA/L2VA
instruction lines, the full 20-item camera-motion vocabulary, the six summary
task types, both four-item retention-marker sets, the entry formats, the 350–500
word figure, the voiceover phrase, the style list — is verbatim or near-verbatim
against `VIDEO_PROMPT_WRITING_GUIDE_base_en.md` and `_ref_en.md`. Whoever wrote
it had the guides open. Nothing in it is *wrong* about the format.

Trustworthy as-is for a hand-writing skill: the task taxonomy and instruction
blocks; the three core fields and the six Ref2VA sections with their order; every
label definition; the summary prefix vocabulary and combination rule; both
retention-marker vocabularies and entry shapes; the shot/timestamp/cut rules; the
entire camera-motion vocabulary; speaker IDs; `<d>[Language]`, voiceover,
`<scenetrans>`, `<cutoff>`; on-screen text.

Needs correction or flagging:

- **"Continuity modes" is invented.** No source defines standalone/continuation
  as a Context-IR concept, and the hosted Context-IR API has no continuation
  parameter. Keep the advice, relabel it as repository chaining craft.
- **`N/A` is missing**, so silent-audio and no-score briefs will be answered with
  fabricated sound. This is the one omission that produces materially wrong output.
- **Reference numbering** is stated as one global order; the guide numbers
  `<Video N>` and `<Audio N>` independently.
- **"Degrades on anything else"** and the "no 8k / no artist names" rules are
  unsourced editorialising — right in spirit, but should not be taught as the
  model's contract.
- The **dialogue-fidelity rules** of ref §5.4 are wholly absent and matter for any
  Ref2VA request that reuses supplied speech.

The README's numbers (`17n+5`, 124–345, 24 fps, `audio_duration` as a ceiling)
are all confirmed against the installed diffusers source. Its two soft spots are
the unexplained 5 s floor versus the model card's 4 s, and the unstated coupling
between the 544p turbo LoRA, the 960×544 canvas and `num_inference_steps: 9`.
