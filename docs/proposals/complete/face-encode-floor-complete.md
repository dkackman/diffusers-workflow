# Face encode floor and `scored_from` in `analyze_face_drift` (#1418)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out session,
2026-11-17). #1400's stage A (#1406) found that `analyze_face_drift` scored one H3 clip
up to 0.045 SFace apart on a minimum (0.015 on a median), depending on whether it read
the in-job frames or the saved files. M-F399 bounced that stage on exactly this. Don
filed it as an `idea`. The plan comment on #1418 (v2) holds the full sweep. This file
records what was built and measured.

## Verdict: build smaller. Measure the floor, then make the score say its form

- **Measured on mini-ai, on the CPU:** in-memory scores and saved-file scores of the same
  clip differ by up to **0.016 on a median and 0.030 on a minimum**. This is a **lower
  bound**, from re-encoding frames that were already encoded. Almost all of it comes from
  the mp4 encode. The jpg save of the reference moves a median by 0.001 at most.
- **`analyze_face_drift` now says which form it scored:** a top-level
  `scored_from: {"video": "frames" | "file", "reference": "picture" | "file"}`. When the
  two halves differ, it adds one mixed-forms note that quotes the floor.
- **Cut: scoring the encoded form in-job (option b).** It comes back only with a median
  floor ≥ 0.03. 0.016 is below that.
- **Not triggered: the chain `face_drop` follow-up.** A fresh chain takes its base from
  memory and a resumed chain takes it from the file. Don ruled that this becomes its own
  issue only with a median floor ≥ 0.02. 0.016 is below that, so no issue was filed.

## Plan history

- **v1 (design, 2026-11-17).** Verdict: build smaller. Two stages: a probe that measures
  the floor and splits it by cause, then the provenance key and the docs. Option (b) was
  cut. Q1 [spend]: download the YuNet/SFace weights on mini-ai. Q2 [default]: add
  `scored_from` and the note. Q3 [none]: naming.
- **v2 (decompose, 2026-11-17).** Folded in Don's rulings on v1, all approved at the
  defaults, with the follow-up made conditional on a median floor ≥ 0.02. Scope unchanged.
- **Correction to the issue (found by the sweep):** the issue said the gap "comes from
  the encode". The #1400 number actually mixes four sources: the mp4 encode, the
  reference's jpg save, the YUV→RGB read-back, and possibly different frame selection
  (`assess.resolve_shots`). dw also has only one encode setting, so "across codecs and
  quality settings" reduces to its default, and the probe spans clips instead.

## What was built

### Stage A (#1420): face encode floor probe (bench only)

Commits f9525844 (merged as 9be01f4e), 331bfd6c (merged as f28499cf) and 6fcf7d30
(merged as b8a0cecd). Bench and tests only, so no server change.

- **`bench/workflows/probe-1418-face-encode-floor-prep.json` + `probe-1418-face-encode-floor.json`.**
  The prep run saves each clip's mp4 and its png and jpg reference. The arms job then
  scores them in six arms: FM (in-memory frames), FD (the saved mp4 decoded back into
  memory) and FF (the saved mp4 read as `output:`), each against the png (RP) and the
  dw-saved jpg (RJ). All arms use `samples_per_shot` 16 and the default threshold.
- **Clips:** five face clips on mini-ai (ep63, ep83, ep64, ep89, the priya-hal two-shot),
  plus repeats of ep63 under fresh inline ids. The repeat gap is 0.000.
- **Record:** `bench/records/face-encode-floor.md` + `.json`. The floor pairs are
  FD/RP−FM/RP (video encode), FM/RJ−FM/RP (reference) and FD/RJ−FM/RP (the #1400
  combination). Each pair matches on its shot record.

  | Pair | Median floor | Min floor |
  |---|---|---|
  | Video encode | 0.016 (ep83) | 0.030 (ep64) |
  | Reference | 0.001 | 0.002 |
  | Combination | 0.016 | 0.030 |

- **Source 4 (frame selection) is listed apart.** FF differs from FD only in its shot
  label (`none`/`whole` against `artifact`). FF−FD is 0.000 in every clip, so frame
  selection added nothing here.
- **Against #1400:** this lower bound's 0.030 minimum is below #1400's 0.045. That is
  what re-encoding already-encoded frames is expected to give, and it doesn't contradict
  the H3 number. `bench/README.md` says a face-score margin must clear the floor, that
  in-job scores are compared only with in-job scores, and that 0.030 is not enough
  margin for first-encode H3 output.
- **Tests:** `tests/test_bench_workflows.py` checks that the probe workflows validate,
  that the record's floor is the max |Δ| of its own table, and that both arms of each
  floor pair share a shot record.

### Stage B (#1421): `scored_from` and the floor in the docs (server)

Commit 035a1317, merged as 41cec78f, deployed to mini-ai by the driver.

- **`dw/tasks/face_drift.py`:** the `scored_from` key and the mixed-forms note, which
  quotes 0.016 / 0.030 as a lower bound and cites the record. `"file"` covers a stored
  path, `asset:`, `output:`, and a `save_segments` chain's frames. A chain's frames
  arrive as `ReadFrames` (a `list` that carries its `.path`), added in
  `dw/pipeline_processors/chain_spill.py` and `dw/previous_results.py`, so the task can
  still tell that they were decoded from a file.
- The chain check (`face_similarities`) is unchanged.
- **Docs:** the `analyze_face_drift` section of `docs/TASKS.md` (the key, its values and
  the floor), the `face_drop` row of `docs/WORKFLOW_GUIDE.md` (in-memory base on a fresh
  chain, kept-file base on a resume), and the description of
  `workflows/templates/face-drift-check.json`.
- **Tests:** `tests/test_face_drift.py` covers frames/picture, file/file on a real mp4,
  a saved chain giving `file`, the note on mixed pairs only, and the additive key set.
- The architecture review passed.

## Cost and bounces

| Stage | Bounces | Cost |
|---|---|---|
| #1420 | 2: C-F992 (the FM-vs-FF pairs, whose shot records differed, were folded into the floor; fixed by adding the FD arms), then a docs review (the record didn't address #1400's 0.045) | Session cost not recorded on the stage. Under 30 min of mini-ai CPU |
| #1421 | 0 | Session cost not recorded on the stage |

## Deferred and open

- **#1423 (open at close-out):** `analyze_face_drift` refuses a bare `video_frames`
  result as `FittedVideo` at run time, although validate passes it. This predates #1421.
  Because of it, C-F994 and C-F996 arm 1 can't be run as written; they pass with
  `trim_video` or `film_grain` as the in-memory source.
- **Upper bound:** the floor comes from re-encoding. A first-encode H3 clip on lem would
  give the upper bound, and #1400 saw 0.045 on a minimum there. It wasn't built.
- **Option (b), scoring the encoded form in-job:** cut, and it comes back only with a
  median floor ≥ 0.03.
- **Chain `face_drop` base across a resume:** not filed, because 0.016 < 0.02.
- **Not exercised over MCP:** a `save_segments` chain's `previous_result:` reading as
  `file`. It needs a video-model chain run, and unit tests cover it.
- Out of scope throughout: re-scoring past records, a crf/quality setting, and #1400.
