# Words check for dialogue chains (#1262)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out session,
2026-11-09). Don filed the idea on 2026-11-07 from the constitution-walk H3
chain, where a client-side loop running `templates/check-script` after each
segment caught word defects that every picture and level measure passed. Plan
v2 is the plan comment on #1262. Don approved v1 with the Q1–Q5 defaults plus
two gates and one test, and v2 folded those in. It was built as three stages:
#1297, #1298 and #1299. The plan comment still holds the full design. This file
records what shipped, what the measurement found and what is still open.

## The problem

`templates/minimax/chained-segments` with `check: true` measured
`jump_ratio`, `look_ratio`, `cut_ratio`, `texture_ratio` and `audio_dbfs`, and
none of them hears speech. A `shot: dialogue` chain that dropped, garbled,
invented, looped or clipped words passed the check. Don then re-rendered whole
50–70 min chains by hand (`constitution-walk-2`: about 30 of its 86 jobs on
2026-11-08 were check-script runs judging re-renders).

## Verdict: build smaller

- **Built:** a `words` measure in the chain's per-segment check, re-rolling
  under the existing `retries`/`best_attempt` policy.
- **Cut (Q1): `guide_end: "pause"`**, ending the carried guide at the last
  pause rather than the final frame. Its evidence was one seam of one chain.
  It hits the same walls `guide_backoff` does: `chain_resume.check_plan` needs
  exact kept-frame counts, `_with_latent_tail` takes a fixed backoff, and
  `_duration_plan`/`plan_segments` ignore `tail_trim`. There is also an open
  17-frame latent-block question and a name collision with
  `h3_rules.guide_end_problem`. *What would bring it back:* words-check records
  showing carry-caused failures on two or more of Don's chains, meaning a
  segment whose every attempt fails words with insertions at the head of its
  kept span. Stage B's `every_attempt_failed` makes that visible.

Non-goals that stayed out: lip-sync and attribution (#488), music and singing
chains, any change to how a segment is rendered, and MCP, REST or UI surface.

## Plan history

- **v1 (approved 2026-11-09).** Q1 [scope] cut `guide_end`; Q2 [default]
  words on whenever `check` is on and a segment has a line; Q3 [scope] add the
  `lines` chain key; Q4 [default] number-word folding in the shared
  normaliser; Q5 [none] the ceiling from existing outputs, Whisper on CPU. Don
  added two gates: (1) no clean good/bad separation means B ships
  `words_similarity_min` default `null` and returns to him; (2) a CPU Whisper
  pass over ~20 s on an ~8 s span means a re-plan before B. He also asked for a
  "twenty-one" test after the hyphen split.
- **v2.** v1 plus that ruling, with no scope change.

## What was built

### Stage A (#1297): the words measure, unreferenced

Final merge `07a3a235` on `develop` (commits 0e8166c6, e36b99b2, 543b7553,
84d5f284).

- `dw/script_lines.dialogue_text(prompt)`: the `<d>…</d>` bodies of a prompt,
  joined in order. (`strip_markup` keeps the scene prose, so it couldn't be
  reused.)
- Number-word folding in `normalize_words` (`fold_numbers`, `numbers=True` by
  default): hyphenated compounds are split, then folded, so "twenty-one", "twenty
  one" and "21" compare equal. The thousands comma is dropped. Because the
  normaliser is shared, `check_script` gains the fold too, so its similarity
  can only rise on number lines.
- `chain_check.check_words(...)`: similarity, heard text, findings and passed,
  through `script_check.check()`, which is unchanged. It fails under the minimum
  or on `line_clipped_at_end`, `transcript_loop`, `tag_spoken` or
  `unscripted_speech`. **The seam-head rule** (this corrects the proposal): the
  guide's carried speech is already trimmed off the kept span, so the risk is a
  *partial* carried word at its head. Heard words before the first aligned
  expected word that match the tail of line k−1 are dropped. Invented words
  are kept, and they count.
- `bench/records/words.md`: the spread and the CPU timing (see "Measurement").

**Bounces: 2.**
1. Architecture review: `_seam_head` ran its own `SequenceMatcher` and its own
   per-chunk fold, which made it a second owner of the Script check row. It was
   rebuilt on `normalize_words(numbers=False)`, `fold_numbers` and
   `script_check.align`.
2. Verify:
   - C-F743(d): "Article 2 1." read 1.0 against heard "Article 21". The cause
     predates stage A: `line_similarity`'s compound join, `_merge_compounds`
     (#1172), joined any adjacent digit pieces. Digit-only pieces now join only
     as a year said in halves (`_year_halves`).
   - C-F744: C-F503's findings had moved.
   - C-F745: the check-script note is in the tasks guide, not the workflows
     guide.

Cases verified: C-F741 to C-F745.

### Stage B (#1298): wired into the H3 chain check

Final merge `ee4e18c1` on `develop` (commits dd0cf57a, 9dce522f, 38486c8e,
1ffa7ac4, 18fabc5f).

- `check.words_similarity_min`: a number in (0, 1], or null. **It defaults to
  `null`, because the separation gate fired.** The chain key `lines` holds one
  string per segment. Unlike `prompts`, its last entry is not repeated. Both are
  in `workflow_schema.json` and in `chain_rules` (`chain_check_finding`,
  `chain_lines_problem`). `lines` with nothing to read it is valid but warns.
- A segment's line is `lines[k]`, else `dialogue_text` of its prompt
  (`segment_line`). A segment with no line records `words: null` and is never
  re-rolled for words.
- Segment 1 is now words-checked; it had been checked for audio only.
- `best_attempt` breaks ties among words-only failures by the higher
  similarity. A kept failure warns `chain_check_failed` with `heard`,
  `similarity` and `findings`. When every attempt failed words, the warning
  adds `every_attempt_failed: true` and points to `repair-segment`.
- The estimate gains a Whisper term per checked attempt
  (`cost_scaling.words_check_term`) on the curated, observed and inherited bases.
  The term counts only segments with a line, and segment 1 counts as checked.
- Docs: the `WORKFLOW_GUIDE.md` measures table, `chain_check_failed`, `lines`.

**Bounces: 3** (and one driver park).
1. Verify: `lines: []` was refused without the counts, and the words term was
   missing from the observed and inherited estimate bases.
2. Verify: a words failure at similarity 1.0 (job `3fa646fcaca0`). This was by
   design: a finding fails words whatever the similarity. The fix was to
   make the re-roll log name the failing finding and the heard text. Whisper
   passes are now counted only for segments with a line.
3. Architecture review: `cost_scaling._is_literal_text` matched the reference
   prefixes with its own `startswith` loop. It now calls
   `references.is_ref(references.DEFERRED, value)`.

The driver then parked B after two sessions in a row ran out of budget partway
through the verify. Don handed it back. He accepted C-F749 segment 2 and
C-F750 on the take evidence: at seed 1262, H3 says L2 wrong (0.75, 6 of 8
words, in three jobs), and the words check correctly failed that take.
dkackman/harnest#201 re-picks the seed or line so those cases can run as
written. C-F748's kept-higher-similarity sub-check was accepted on unit tests.

Cases verified: C-F746 to C-F753 (C-F749 segment 2 and C-F750 on Don's ruling).

### Stage C (#1299): template, skills, docs

Commits 7e6a9176 and 167fd6b0 on `develop`.

- `templates/minimax/chained-segments` gains `lines` (default `null`), passed
  through as `chain.lines`.
- long-form's *Per seam* and `references/run-and-repair.md` list the words
  measure, its null default and the `every_attempt_failed` → `repair-segment`
  path. The minimax-h3 dialogue reference says to give each segment prompt one
  `<d>` line, or to use `lines`.

**Bounces: 1.** Verify (C-F755, C-F756): the template's `check` variable was
typed bool because its default was `false`, so the template refused
`{"words_similarity_min": 0.8}`, and `lines` could never reach a words check
through it. The default is now `null`, which means off. A null-default variable
is untyped, and the chain rules decide which values are valid.

Cases verified: C-F754 to C-F756.

## Measurement (`bench/records/words.md`)

- **Spread.** Good takes read 0.90–1.0 with number folding. The lowest are
  British spellings heard as American ("defence", "offence"). Bad takes read
  0.889–0.957. Three bad takes (0.9412, 0.9565) score above two good ones
  (0.90). Folding alone moved good takes such as "Amendment 6" from 0.90 to
  1.0.
- **Separation: none.** The bad takes share a word dropped at a segment's
  head, the seam the chain cuts. A dropped word costs a 12-word line only about
  0.04 of similarity. **Gate fired:** `words_similarity_min` defaults to `null`.
- **Ceiling.** `WORDS_SIMILARITY_MIN = 0.80` is the recommended explicit
  setting. It sits under the lowest good take with margin, and it catches only
  gross failures: a wrong or missing line, or a garbled segment.
- **Whisper CPU cost.** On lem (16 threads, whisper-large-v3-turbo, 8.0 s span)
  a pass took 3.00 s cold and 1.28 s / 1.27 s warm, well under the 20 s gate. So
  B was built as planned, on CPU, with no VRAM beside H3.

## Cost and bounces

| Stage | Bounces | Cost |
|---|---|---|
| #1297 | 2 (architecture, verify C-F743/744/745) | Not recorded on the stage |
| #1298 | 3 (verify ×2, architecture) plus one driver park | Not recorded on the stage. Two verify sessions ran out of budget |
| #1299 | 1 (verify C-F755/756, `check` typed bool) | Not recorded on the stage |

## Deferred and open

- **The gate's return to Don was skipped.** The plan said that if the separation
  gate fired, the parent would go to Don after B verified and before C. The gate
  fired, but stage C was built without that hand-off. What shipped matches the
  gate: the default is `null`, and the docs say to set `0.80` explicitly. Don
  has not yet ruled on whether a null default is where the measure should stay.
- **A head-of-segment drop is what similarity misses.** Every bad take in the
  record dropped a word at the line's head. A findings-style check on the
  first aligned expected word (a `line_head_dropped` finding, say) would
  separate them where a ceiling can't. It is not filed yet. It would be the
  route to turning words on by default.
- **`guide_end: "pause"`**: cut, on the trigger above.
- **The estimate's segment scaling** (#1305): a 3-segment checked chain quotes
  7.0 min against 6.9 min for 2 segments.
- **dkackman/harnest#201**: re-pick the seed or L2 so C-F749 and C-F750 run as
  written.
- Related: `check_narration` (#1272, parked) is the TTS sibling, sharing the
  `align`/`normalize` ideas.
