# Model narrative out of the MCP tool descriptions (#376, #101 recommendation 4)

Written by model `claude-opus-5-5` via provider `anthropic`, 2026-09-24, at
close-out. It replaces `mcp-context-cost-partial.md`. The analysis this
builds on is `mcp-context-cost-complete.md` (#101). The plan was plan v4 on
#376, approved by Don.

## Verdict: built smaller, as one stage to make room for #388

The partial doc was mostly out of date. #101's own follow-ups had already
done most of recommendation 4:

- **`wait_for_job`'s description was already rule-only.** The H3 figures
  (629 s, ~90 s, ~10 min, 140 s, RTX 3090) were in the docstring of the
  internal handler in `dw_mcp/diagnose.py`, which never reaches an agent.
- **`get_memory` already pointed at the guide** (`acceleration`, "Reading
  Memory While Offloading") instead of explaining the cache itself.
- **The H3 figures were already in `minimax-h3/SKILL.md`**, pinned by
  `tests/test_mcp_server.py`. Nothing moved into the skill.
- **The "~8,350 → ~4,000 chars" estimate no longer applied.** The largest
  item left was a worked transcription procedure, not H3 narrative.

What was left was about 200 tokens of resident surface, not ~1.1k. Its one
consumer was #388 (`assess_output`), which had 7 tokens of headroom under
`SURFACE_BUDGET`. Don chose #378's Q2 option (b): make room here first, then
have #388 raise the budget only by whatever it still needs.

## What was built (stage 1, #403)

`feat/376-a-trim-model-narrative` (dce8358), merged to `develop` as
`007717a`. Only descriptions and docs changed. There was no engine, REST,
syntax or schema change.

| Where | Removed | Kept |
|---|---|---|
| `get_output_audio` | The transcription walkthrough (its third copy) | A text-only client confirms the words by transcribing. It points at `WORKFLOW_GUIDE` "The loop", step 6, through `get_guide("workflows", section="Authoring a workflow from an agent")`. The "WAV" and "own encoding" rules stay. |
| `validate_workflow` | H3's `17 * n + 5` example | Constraints are checked and reported |
| `list_workflows` | The `17*n+5, 124-345, rounds up` example | Constraints appear in terse form |
| `list_prompts` | The family list, which went stale with each family added | `intended_model` filters by family |
| `get_job_events` | "(a video reference encode, block-cache gaps)" | Some models are silent for minutes; check the model's skill or guide |
| `dw_mcp/diagnose.py` (internal) | The H3 figures (their fourth copy) | The rule, plus a pointer to the skill |

- **`docs/WORKFLOW_GUIDE.md` "The loop", step 6** gained the "to confirm
  the words a clip speaks" paragraph. The procedure moved word for word.
- **Tests** check two things. `get_output_audio` points at "The loop" and no
  longer names `transcribe-audio`. The guide's step 6 names
  `templates/transcribe-audio` and `get_output_text`.
- **Surface** went from 13,883.0 to 13,670.5 tokens (measured on Python
  3.14). `SURFACE_BUDGET` stays at 13,890. The "Measured 2026-09-24" entry
  reserves the 219.5 tokens of headroom for #388.
- **Kept on purpose:** `wait_for_job`'s "video … minutes … uneven …
  `denoise_step` has moved" sentence, and `get_gallery_metadata`'s dBFS
  thresholds. Both are rules, not narrative.

## Deferred, and why

- **Addressable `###` subsections in `get_guide`** (Q4). "The loop" is a
  `###` under "Authoring a workflow from an agent", and `get_guide`
  resolves only `##` sections. Don chose to change the plan rather than the
  code. If this is wanted, it should be its own `idea`.
- **A second pass over the largest descriptions** (`list_gallery`,
  `download_output`, `upload_asset`) to cover all of `assess_output` (Q2
  option (c)). Not chosen. #388 raises the budget by the remainder and logs
  why.
- **`docs/MCP.md`'s H3 narrative** stays. It is for human readers and is not
  resident.

## Bounces

- **Stage 1 (#403): one bounce.** C-F114 step 1 called `get_guide("workflows",
  section="The loop")`, which the plan's own acceptance named but which
  could never resolve. The gap was in the plan, not the build. It was
  re-planned as v3 and v4 with no rebuild. C-F114 was replaced by C-F117,
  which passed with C-F113 and C-F115 on `develop @ 3ff4ff8`.

## Cost

Not recorded. The stage comments don't state `usage:` figures.

## Decisions (Don, 2026-09-24)

- **Q1.** Cut examples and specifics, and keep every rule, as the table
  shows.
- **Q2.** Option (b). No stage 2. #388 raises `SURFACE_BUDGET` by the
  remainder.
- **Q3.** Trim the internal `diagnose.py` docstring to the rule plus a
  pointer.
- **Q4.** Option (a). Reach step 6 through the `##` section, with no
  `get_guide` change.
