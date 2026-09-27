# `attribute_voices`: which reference singer sings each line, by timbre (#485)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out,
2026-09-27). Plan v1 approved by Don as "build smaller" on 2026-09-27;
stages #494 (A) and #495 (B), both verified.

## The report

A field report from an Opus 5.5 session on lem (2026-09-25/26): a 3:12
musical short, 18 lip-synced song shots from one Music 3 take. Music 3
ignores "Singer X sings section Y" (all three seeds), so the session had to
find out who sings what before staging lip-sync. Its pitch-based map
(demucs vocal stem, median pYIN per lyric line) called the choruses female:
the tenor melody sits at 220-293 Hz, which a mezzo also reaches, and the
phrases open on shouted ensemble attacks that pull the median up. Six chorus
shots were staged with the wrong singer and re-rendered (~3 GPU-hours).

What worked, done by hand outside dw (`who_sings.py`): separate the vocal
stem, embed a known stretch of each singer, score every line by cosine
similarity, roll lines up per shot by time overlap. It got all six right.

## Decisions (plan v1, approved as written)

- **Build smaller.** One task and a skill note. No template field wired to
  the attribution (it would break the pinned entry shapes in
  `tests/test_plugin_skills.py`; the agent writes the shot list from the
  JSON anyway). No reference-free diarization.
- **Q1: demucs is a hard dependency.** Separation is the default path; an
  optional extra would make the default fail on a plain install.
- **Q2: speechbrain ECAPA, not Resemblyzer.** speechbrain was already a
  dependency. The risk was that ECAPA is trained on speech, not singing;
  the stage-A known-answer case was the check, and it passed (held-out
  margins 0.36-0.74, reference cosine 0.053), so Q2 stands.
- **Q3: an explicit `assessment` flag on `register_command`.** A second
  `returns: "json"` command would otherwise have been filed under
  `list_tasks`' `assessment` bucket. The three probes set
  `assessment=True`; the bucket reads the flag.
- **#483 (timestamped transcribe) is not a dependency.** `lines` takes
  spans directly, in #483's proposed `{start, end, text}` chunk shape (or
  `start_seconds`/`duration_seconds`), so its output drops in via
  `previous_result:` when it lands.

## What was built

**Stage A, #494** (server): `dw/tasks/voice_attribution.py`, the
`assessment` flag (`dw/tasks/task.py`, `dw/introspection.py`), `demucs` in
`pyproject.toml`, and the `docs/TASKS.md` section (*Voice attribution*).

- Arguments: `audio`, `voices` (name -> span list into `audio`, or a clip
  path / `asset:`), `lines`, `windows`, `window_seconds` (2.0),
  `min_reference_seconds` (3.0), `separate` (true), `device`. No
  model-name argument, so nothing free-form reaches `torch.hub.load`.
- Returns per-line `scores`, `voice`, `margin`, `voiced_seconds`,
  `uncertain` (with `voice: null` and a reason when too little is voiced),
  per-window `share` rolled up by voiced overlap, and
  `reference_similarity` with a `voices_too_similar` warning.
- A step on it must save `content_type: application/json`, the rule the
  probes already follow.
- Found in verification and fixed:
  - bounce 1: a bare-path voice clip passed validation (the boundary held
    at run time; it was refused too late). Voice clips now go through the
    same location policy as `audio` (`dw/locations.py`). A literal `voices`
    is checked statically by `voices_errors`. `asset:`/`prompt:`/`output:`
    literals written straight into a step are now existence-checked at
    validation and submission, as caller arguments already were. The
    voiced floor became relative to the stem (95th-percentile frame rms
    less 35 dB, never below -60 dBFS) so a quiet sung section counts.
  - bounce 2: no code change. The failing line (Priya's opening, inside
    her reference span) holds ~0.44 s of voice, under the declared
    `MIN_VOICED_SECONDS` of 0.5, so `voice: null` was the specified answer.
    Filed as suite defects harnest#21 (that line and the 0-2 s window) and
    harnest#20 (`shot_join`).

**Stage B, #495** (plugin-only): one *Hard rules* bullet in the
`minimax-music3` skill (279 B, skill now 12,207 of 12,288 B): Music 3
ignores per-section singers, so run `attribute_voices` with a reference
span per singer before staging lip-sync, and never infer the singer from
pitch.

## Bounces per stage

| Stage | Bounces | Cause |
|---|---|---|
| A, #494 | 2 | 1: SE-F038 refused too late, voiced floor, direct literal existence check. 2: a case defect (harnest#20, #21), no code change. |
| B, #495 | 0 | |

No stage comment names `usage:` figures, so per-stage cost is not recorded.
The plan estimated ~$7-10 total.

## Deferred

- **Template wiring** (a per-shot `singer` chosen automatically). Back if a
  second report shows a session mis-wiring the output into a shot list.
- **Resemblyzer.** Back only if ECAPA fails on a real duet where
  Resemblyzer would not.
- **`plan.downloads_required` does not report the demucs weights** (from
  Meta's CDN via torch.hub), the same gap `transcribe_audio`'s Whisper and
  speechbrain already have. The fix is a per-task default-model table in
  `_collect_sources` (`dw/plan.py`), a separate follow-up.
- **demucs on MPS** is unverified; separation falls back to CPU with a
  warning.
- **Ensemble lines** (both singers at once) score between the two; `share`
  and `margin` show it, but `voice` names one.
- A `lines: "previous_result:transcribe"` case, once #483 lands.
