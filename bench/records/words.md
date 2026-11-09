# Words check: similarity spread and Whisper CPU cost (#1262 stage A, #1297)

Measured 2026-11-08. Does a dialogue segment's similarity to its line (the
`check_script` measure that `chain_check.check_words` uses on one line)
separate the takes Don kept from the ones he re-rendered? And what does one
Whisper pass on CPU cost? This used no new GPU time. The similarities come
from the existing `templates/check-script` outputs in workspace
`constitution-walk-2` (openai/whisper-large-v3-turbo, threshold 0.8),
re-scored offline with the #1297 normalisation (number words folded to
digits).

## Labels

A take is **bad** when Don re-rendered it for its words, as the job notes
record: am5c ("seg3 … Nor"), am6a ("orig last segment misspoke") and am9
("conscrued + dropped to"). A take is **good** when it was kept. am7a seg3
dropped its head word "and", and nobody noted it; it is marked bad? because,
by the script's words, it is.

## Spread

| Label | Take, line | Heard (the difference) | Before folding | With folding |
|---|---|---|---|---|
| good | about 25 lines (am1, am2, am5a, am5b, am5c r4/r5, am6a re-render, am6b, am7a, am7b, am10, …) | exact | 1.0 | 1.0 |
| good | am6a re-render seg1 | `Amendment 6` | 0.9000 | 1.0000 |
| good | am7b seg2 | `Amendment 8` | 0.9167 | 1.0000 |
| good | am10 seg1 | `Amendment 10` | 0.9231 | 1.0000 |
| good | am7a seg1 | `Amendment 7`, `$20` for "twenty dollars" | 0.8387 | 0.9677 |
| good | am6b seg4 | `defense` for "defence" | 0.9000 | 0.9000 |
| good | am5b seg2 | `offense` for "offence" | 0.9000 | 0.9000 |
| good? | am9 seg1 | `Amendment 9`, `conscrued` (a Whisper mishearing) | 0.8571 | 0.9286 |
| bad | am5c seg3, five renders | "Nor" dropped from the line's head | 0.9565 | 0.9565 |
| bad? | am7a seg3 | "and" dropped from the line's head | 0.9565 | 0.9565 |
| bad | am9 seg2 | "to" dropped from the line's head | 0.9412 | 0.9412 |
| bad | am6a seg4 | `asserturned` (misspoke "ascertained") | 0.8889 | 0.8889 |

- **Good:** 0.90 to 1.0 with folding. The lowest are spelling variants
  (British "defence"/"offence" heard as American), not wrong words.
- **Bad:** 0.889 to 0.957. A dropped word costs a long line 1 - 2/(2n+1),
  where n is the line's word count: about 0.04 for a 12-word line.

## Verdict: no clean separation

The ranges overlap. Three bad takes (0.9412 and 0.9565) score above two good
ones (0.90), so no single minimum passes every good take and fails every bad
one. What the bad takes share is a dropped word at a segment's **head**, the
seam the chain cuts. Similarity can't see that; an alignment can. **Gate
fired:** stage B ships `words_similarity` with a **null default**, so a chain
doesn't re-roll on similarity unless the caller sets a minimum.

**Ceiling:** `WORDS_SIMILARITY_MIN = 0.80`. That's the most a minimum can be
and still pass every good take seen, with margin under the lowest (0.90). It
catches only gross failures: a wrong or missing line, or a garbled segment.

## Whisper CPU cost

On lem, 16 torch threads (`CUDA_VISIBLE_DEVICES=` empty), whisper-large-v3-turbo
through `transcribe_audio(device="cpu", timestamps="word")`, on 8.0 s of a
constitution-walk-2 segment (am3, "Amendment 3. No soldier shall in time of
peace be quartered in any"):

| Pass | Seconds |
|---|---|
| first (pipeline built in the process, weights on disk) | 3.00 |
| second | 1.28 |
| third | 1.27 |

That's about 1.3 s for an 8 s segment once the model is loaded. **Gate not
fired:** that's well under the 20 s re-plan gate, so stage B builds as planned.
