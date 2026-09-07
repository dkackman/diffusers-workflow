# Proposal: scripted dialogue shorts, and where text-to-speech belongs

Status: design only, no code changes. Records the conclusion of a design
discussion about supporting a flow like: *write a script of two people talking
→ give each a consistent voice and face → generate H3 clips of each speaking
their lines → cut between them into a finished dialogue scene.*

## The question

Should text-to-speech become a task (or a workflow shape), generating each
speaker's lines as audio up front, which H3 then follows? Or does predefined
textual content belong inside H3 itself?

## Conclusion: do not generate the dialogue audio first

The text being predefined is not an argument for TTS. H3 already takes
predefined dialogue — it speaks the exact line written inline in the prompt:

```
<Subject 1> (S1), a fast, nasal, high-pitched male tenor voice, tight with
disbelief, says <d>[English] Okay. Who finished the pistachio?</d>
```

Generating the audio first and asking H3 to follow it moves every line of
dialogue onto the model's weak pathway. H3 lip-syncs well when it *generates*
the speech (speech and mouth come out of one process, sync is free) and poorly
when it must *follow* an audio reference. Investigated 2026-09-04 on the
MarmotCroons rebuild; ruled out with a run each: conditioning-window choice,
prompt language, LoRA family, 30 steps without the turbo LoRA, and dw's own
plumbing (verified end to end). Nothing recovered it. Conditioning on a
centre-extracted vocal isolate was a partial win, not a fix.

So: script → TTS → `match_audio` trades away the one thing that works for free.

## The workflow already largely exists

[`workflows/templates/minimax/dialogue-short.json`](../../workflows/templates/minimax/dialogue-short.json)
is nearly the described flow already:

- two Z-Image steps draw the cast portraits (the second reuses the first's
  loaded pipeline; `release_pipeline` frees it before the video model loads)
- the first shot loads H3 once; the remaining shots are `pipeline_reference`
  steps rerunning that loaded model with a new prompt, references and length
- `concat_videos` splices the episode with hard cuts, and `audio_bleed_ms`
  rings each shot's room tone over the silent head of the next, so only the
  picture cuts hard

Consistency in this cut-based shape comes from repetition, not from a shared
audio track:

| What stays consistent | How |
| --- | --- |
| Identity | reference the same portrait in every shot |
| Voice | repeat the character's voice description **verbatim** in every prompt |
| Ambience | repeat `overall_soundscape` verbatim; `audio_bleed_ms` carries it across the cut |
| Quality | each shot is generated fresh from the portraits, so shot 12 is as clean as shot 1 |

Constraints to design around: a shot is 5.17–14.4 s (`num_frames` of the form
`17n + 5`, 124–345, at 24 fps), so a long line must be split across shots; and
roughly 13 shots is the ceiling for a single `dw.run` on a 24 GB box — beyond
that, batch and stitch.

## Where a TTS task does earn its place

There is no TTS in the repo today. It is worth adding, in these roles — none of
which is "the track the mouth follows":

1. **Voice *timbre* reference.** `ref2va` takes an `<Audio 1>` reference that
   fixes a voice while H3 still generates the speech — see
   [`MiniMaxH3Ref2VA.json`](../../workflows/templates/minimax/reference-to-video.json), whose
   `retention_analysis` says explicitly that only timbre, pitch and delivery are
   referenced and "none of its content is reused". A few seconds of TTS per
   character, referenced in every shot, makes voice consistency an actual
   conditioning signal rather than hoping prose like "flat, low, unhurried
   baritone" lands identically twelve times. Keeps the strong sync pathway.
   Highest value, smallest change.
2. **A voice that must be matched** — a cloned or specific voice, or a delivery
   H3 will not produce from description alone.
3. **Non-lip-sync audio** — narration over B-roll and cutaways, muxed with
   `pair_audio`, where no mouth has to follow it.

### Shape

Model it on [`dw/tasks/text_generation.py`](../../dw/tasks/text_generation.py):
a HuggingFace `text-to-speech` pipeline behind `cached_model`, returning a
waveform plus sample rate, so it composes with the existing `slice_audio`,
`pair_audio` and `concat_videos` tasks. A task, not an H3 pipeline mode.

## The larger gap: script → Context-IR shot list

Bigger than TTS. `generate_text` can write prose dialogue, but H3 wants
Context-IR: `subject_definitions` / `summary` / `retention_analysis` /
`detailed_description` / `overall_soundscape` / `non_diegetic_music`. Turning a
plain script into one Context-IR prompt **per shot** — voice descriptions
carried verbatim, speaker alternation driving the cuts — is the piece that does
not exist.

It must also strip the portrait of compositional authority, in
`subject_definitions` *and* in a `<Picture N>` `retention_analysis` entry
("carries no compositional detail: no framing, no shot size, no camera
position"). Without that the reference's framing wins and every shot comes back
as the same medium shot, so the cuts read as jump cuts. Measure framing variety
as mean absolute difference between shots' average frames; under ~10 means the
reference won.

## Suggested next step

Prototype either (a) the `generate_speech` task in its timbre-reference role, or
(b) the script → Context-IR shot-list step. (b) is the higher-value one.

## What was built (2026-09-06)

(a) is implemented: the `generate_speech` task
([`dw/tasks/speech_generation.py`](../../dw/tasks/speech_generation.py),
documented in [TASKS.md](../TASKS.md#speech-generation)), with
[`GenerateSpeech.json`](../../workflows/templates/generate-speech.json) and
[`MiniMaxH3GeneratedVoice.json`](../../workflows/templates/minimax/voice-timbre-reference.json)
showing the timbre-reference role.

It returns an `AudioTrack` - a waveform carrying the rate it was generated at -
rather than the bare waveform the other audio tasks return. Those rates are a
property of the workflow; a TTS model's rate is a property of the model, and every
one of them differs, so a declared 44100 against a 24 kHz model plays the speech
fast and low without ever failing. `normalize_audio` and `media_arguments` read
`.audio`/`.sample_rate` off whatever they are handed, the way the rest of the audio
plumbing already did, so a declared rate still wins where a workflow names one.

(b) was **not** built as a task, and the reasoning is worth keeping. Turning a
script into shots is authoring, not runtime. Putting it in the engine hides the
prompts until after the GPU has spent the time on them, and it needs shot *i*'s
prompt paired with speaker *i*'s portrait - a zip, where
[`previous_results.py`](../../dw/previous_results.py) deliberately does a cartesian
product. A generator that emits an ordinary workflow keeps the artifact
inspectable and needs no engine change; the missing piece is then the *format
knowledge* (the Context-IR fields, verbatim voice descriptions, the `<Picture N>`
clause stripping compositional authority, `17n + 5`, ~13 shots a run), which is
documentation and a template workflow rather than code. Revisit a `dw.script`
generator or an MCP tool once that shape has been used enough by hand to be sure
of it.
