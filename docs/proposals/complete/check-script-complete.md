# Script-adherence check: `check_script` and `templates/check-script` (#609)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out,
2026-10-07). Plan v2 approved by Don on 2026-10-05 as "build smaller"; three
stages, #643 (A), #644 (B), #645 (C), all verified. The plan comment on #609
is the full design; this doc records what shipped against it.

## The idea

A comparative review of another H3 workflow repo (concept only: its license
is incompatible with Apache-2.0, so nothing was copied) checked each take
against its script: `line_mismatch` below a similarity threshold, delivery
tags heard as speech, speech in a shot with no dialogue, and a last line
clipped by the end of the file. dw's `assess_output` checked dead air and
sync drift, but nothing checked that a take *says what the script said*:
the skills and WORKFLOW_GUIDE loop step 6 told the agent to transcribe the
take and compare by eye, which missed dropped lines (#559).

## Decisions

- **Plan verdict: build smaller.** A worker task plus a catalog template,
  not an `assess_output` input. The assess route is CPU-only and runs in the
  server process (`dw/server/assess.py`), so it can't transcribe; an input
  there would still need a separate transcribe job, would break the
  "exactly three probes" invariant, and would spend MCP surface budget the
  server didn't have (10 tokens of headroom at the time).
- **Corrections to the proposal**, found by the plan's sweep:
  - `no_speech_prob` isn't available from the HF ASR pipeline
    `transcribe_audio` uses, so the hallucination guard is **energy**
    (the #465 dead-air floor), not Whisper's own score.
  - dw has no delivery-tag vocabulary, so `tag_spoken` uses no fixed list:
    the markup stripped from a line becomes the words that must not be
    heard on it.
  - Shot records carry no dialogue, so lines name their shot.
  - Speaker sex is out of scope: `attribute_voices` (embeddings, never
    pitch).
- **Q1:** a worker task `check_script` plus `templates/check-script`; no MCP
  or `assess_output` change. **Q2:** lines may carry H3 markup, which is
  stripped. **Q3:** the task's default stays `openai/whisper-base`; stage A
  measures base against `openai/whisper-large-v3-turbo` and stage C sets the
  template's defaults from it. **Q4:** stage C retires the by-eye procedure.
  **Sequencing:** A lands before #608's A1 probes.

## What was built

**A, #643: the engine, whole-file.** `dw/tasks/script_check.py`, registered
in `dw/tasks/task.py` (`consumes_device=True`, returns json, not an
assessment probe, nothing added to `RULES`). It transcribes through
`transcribe_audio(timestamps="word")`, normalizes, strips markup, aligns
heard words to expected words with `difflib`, and scores each line. Rules
`line_mismatch` (0.85), `tag_spoken`, `speech_where_silent` (`lines: []`)
and `line_clipped_at_end` (final 0.25 s, above the dead-air floor). A
literal `lines` is refused at validate time by
`task_domains.script_lines_errors`, through the same `parse_lines` the task
runs. TASKS.md *Script check* section and a *Script check* row in
`docs/ARCHITECTURE.md`. Develop c3fc3661, b90a91d6 (merged 113301a8),
3a52f5c1 (merged 427586d1).

Deviations from plan v2, accepted at verify:
- **A second guard: repetition loops.** Energy alone couldn't keep the
  plan's promise that an instrumental gives no finding: Whisper's
  `Pre-pre-pre…` decoding loop over room tone or music is loud enough to
  pass the floor. Every chunk inside a run of a 1–4-word unit repeated 6+
  times is discarded with `reason: "repetition"` (`below_floor` for the
  energy guard). Cost: a real line that repeats one word 6+ times would be
  discarded and likely score as a mismatch.
- **A tag word heard on its line is left out of that line's similarity**;
  it is reported as `tag_spoken` only.
- `thresholds` in the answer echo `repeat_run_min` and
  `repeat_max_period`.

**The Q3 measurement** (job `af5de241ca83`, lem, develop @ c3fc3661): one
MMS take with a name, an abbreviation and numbers, checked against four
scripts (exact, swapped, dropped, numbers as digits) under both models.
Turbo scored correct lines 0.71–1.00 and swapped or missing lines 0.00;
base scored correct lines as low as 0.43 (mishearings such as `O'Connor`
and `415`). Turbo writes numbers as words, so a script with digits costs
similarity under it (0.43 on the `numbers` row).

**B, #644: shot-aware findings.** `{text, shot}` lines; a `shots` argument,
else the take's own `.shots`, else the manifest or `keep_output` sidecar,
through `assess.resolve_shots` (answer reports `shots` and
`shots_source`). Each shot is placed on the soundtrack with
`assess.sample_span` (made public from `_sample_span` for this). New rule
`speech_in_silent_shot`; `line_clipped_at_end` judged per shot as well as at
file end; `rules_skipped` with a reason when no shots are known; an unknown
shot or a name the map holds more than once is refused, naming the known
shots (`shots.duplicate_shot_names`, per file). A word that a line in
another shot matched is not counted as silent-shot speech, and `at` is
never earlier than the silent shot's start. Develop b849ad21, 64120597,
bf17655e.

**C, #645: the template, and the by-eye procedure retired.**
`workflows/templates/check-script.json` (shape `utility`; variables
`input_audio`, `lines`, `shots`, `model_name`, `similarity`; one step
`check` saving JSON). Defaults from Q3: `openai/whisper-large-v3-turbo` at
`similarity` **0.6**: any value in (0, 0.71] separated correct from wrong
on the measured take, and 0.6 leaves margin for real H3 voices. The
template says to write numbers as words. WORKFLOW_GUIDE loop step 6,
`series-episodes/SKILL.md` and `minimax-h3/SKILL.md` now run the template
instead of transcribing and reading. Pins:
`test_the_loop_carries_the_transcription_procedure`, the catalog's
`UTILITIES` and step-cache keys, `COMPACT_BUDGET` 10_450 → 10_550 (measured
10_498). Develop 17535810 (merged 18e7fd80).
- **No plugin version bump**, a deviation: `plugin.json`'s version is
  pinned to `pyproject.toml`'s and only the release script moves both.
- `attribute-lines` gained a `LEGITIMATE_MENTIONS` test-allowlist entry,
  since `lines` is now a catalog variable.

## Bounces per stage

| Stage | Bounces | Cause |
|---|---|---|
| A, #643 | 2 | (1) architecture review: `lines` validation was a second walk over task arguments beside `task_domains`, and references were detected by a colon rather than `dw/references.py`; (2) verify: C-F283 (a spoken `<pause>` lowered similarity) and C-F284 (Whisper's repetition loop over room tone and an instrumental raised `speech_where_silent`). C-F281 wasn't counted: it carried stage B's expectations (harnest#72). |
| B, #644 | 2 | (1) architecture review: `shot_spans` re-derived `assess._sample_span` (and disagreed on a missing `start_frame`), duplicate names were counted outside `shots`, and the URL test bypassed `locations`; (2) verify: C-F288's `omit` arm counted a word straddling the cut, already matched by shot 1's line, as silent-shot speech in shot 2. |
| C, #645 | 0 | |

No stage comment names `usage:` figures, so per-stage session cost is not
recorded. The plan estimated $9–15 for the three stages.

## Not built, and why

- **An `assess_output` input for expected lines** (the proposal's framing).
  Declined at Q1 for the reasons above. *Comes back if* the assess route
  gains device access or a transcript becomes an assessable artifact on its
  own.
- **Widening `line_clipped_at_end` to a last word that runs past its shot's
  end.** Kept the plan's ±0.25 s window: Whisper word times are ±0.1–0.3 s,
  so a "crosses the cut" test on one word would be noisy. *Comes back if* a
  take is seen clipped at a cut that the window missed.
- **A Whisper decode-time loop fix** (`no_repeat_ngram_size`,
  `compression_ratio_threshold`). The repetition guard filters a finished
  transcript instead, because `check_script` must use `transcribe_audio` as
  its only ASR path.

## Known edges

- Under `whisper-base` a TTS fade gave a false `line_clipped_at_end` at
  -64.36 dBFS against the -65 floor (noted at #643's verify; the plan's rule
  as written). The template's turbo default avoids the mishearings that
  caused most of base's spread, not this.
- The energy guard can discard real quiet speech (a whispered line);
  discarded words are reported, never hidden.

## Overlap left as it was

#608 (H3 dialogue rules) shipped first in practice; its probe 1 used
`transcribe_audio` word timestamps, and its Music3 vocal check uses the
`vocals` stem, not a transcript of the mix. #600's vocal/b-roll flags and
#488 (lip-sync target) can consume `speech_where_silent` and the per-line
timings; neither was changed here.
