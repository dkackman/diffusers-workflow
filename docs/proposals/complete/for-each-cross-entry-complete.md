# Refuse a literal cross-entry reference inside one `for_each` group (#1101)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out,
2026-10-31). Plan v1 was approved by Don on 2026-10-31 with every question
at its default. Its only stage, A (#1111), shipped the same day at
`develop` @ `9b65ba80`, and an architecture-review fix followed at
`168a61ce`. It was verified on lem.

## The idea

Inside a `for_each` expansion, a member could read another entry of the same
list with a literal reference, such as `previous_result:shot@a` inside
`shot@b`. Validation let it through. `_rewrite_reference` only refuses a
*bare* reference to the member's own group, and
`previous_result_reference_errors` only checks that the name belongs to an
earlier step. It then resolved at run time, backward only. No test pinned
it, but `docs/WORKFLOW_GUIDE.md` promised it ("A member may reference an
*earlier* member of its own list directly"). #1101 was filed as a follow-up
to #1093, so Don could decide whether to pin it or refuse it.

Demand: #509, #511 and #813 show driving sessions chaining shots inside one
list this way. Each run failed after GPU time had been spent, because the
earlier member's result was a video. #813 patched only the video case.

## Verdict and scope (Don, 2026-10-31)

**Refuse it.** Validation rejects a literal reference from one entry of a
`for_each` into another entry of the same expansion. The error names both
entries and points at separate runs plus `output:` references (#1093's
recipe). A cutaway inside one workflow would be a new feature with its own
design.

Don required a check, before any change, that nothing used the form. It came
back clean: catalog `workflows/`, packaged `dw/workflows/`, `plugins/dw/`
and `tests/fixtures` were checked at design time. Every lem workspace was
checked at the start of the build (Q4), with `use_workspace`,
`list_workflows` and `get_workflow` on each list-driven workflow.

## Decisions (Don's review of plan v1, 2026-10-31)

- **Q1:** the ruling stands even though the guide promised the form. The
  guide paragraph is replaced by separate runs plus `output:` references.
- **Q2:** a *different* group over the same list (`trim@b` reading
  `shot@a`) stays legal. Every `shot@*` has run before any `trim@*`, so
  entry order carries no meaning there.
- **Q3:** one error, not two. The new check wins, and #813's
  `video_into_image_errors` skips a reference the new check refuses, since
  its `get_last_frame` advice can't be followed inside one list.
- **Q4:** the lem workspace check ran at the start of the build (clean).

## What was built

### Stage A, #1111: refuse the reference

- **The check.** `cross_entry_reference_errors` in `dw/previous_results.py`,
  registered as `cross_entry_references` in `dw/validation.py`'s
  `ERROR_CHECKS`, right after `previous_result_references`. Every path runs
  it: `/api/validate`, job submission, library save, MCP
  `validate_workflow`, the CLI and composed children.
  - **Refuses:** a member `G@k` holding a literal reference to `G@j`
    (j ≠ k). That covers the `previous_result:` prefix, `from_previous_result`
    (including inside an `item:`-spliced `references` entry), with or
    without a `.property` suffix, and index keys (`shot@1` → `shot@0`).
  - **Error:** `{path, message}`. The path points at the authored step. The
    message names both members and the sibling key, and gives the fix:
    "Generate 'a' in its own run and pass its file to this one as an
    output: reference (guide: for_each)."
- **The sibling rule's owner.** `cross_entry_key` in `dw/for_each.py`, next
  to `group_of`. Its three callers are the new check, the earlier-step skip
  in `previous_result_reference_errors`, and the skip in
  `video_into_image_errors`. `docs/ARCHITECTURE.md`'s previous-result check
  row names it.
- **One error per reference.** `video_into_image_errors` skips a refused
  reference (Q3). A *forward* sibling reference (`shot@a` reading `shot@b`)
  also gets only the cross-entry error, not also "names no earlier step".
  That second skip was a build deviation in the same spirit as Q3.
- **UI.** `memberRefusal` in `ui/src/lib/flow.ts` refuses the same form,
  with a reason naming the sibling. The shared case table
  `tests/fixtures/reference_cases.json` gains `cross_entry_same_group`
  (refused) and `other_group_other_key` (legal, Q2), so the engine and the
  UI are held to one verdict.

### Tests and docs

- `tests/test_previous_results.py::TestCrossEntryReferences`, every case
  through the real validate path. Refused: the prefix, `from_previous_result`
  in a spliced entry, `.property`, index keys and a forward sibling. Legal:
  same key (bare and literal), a later step, `gather:`, a different group
  over the same list, and a group over a different list.
- `tests/test_video_extensions.py::TestVideoIntoImage` is reframed. The real
  validate path asserts the cross-entry refusal and the absence of #813's
  message. The unit case moved to `refine@b` → `shot@a`, plus a sibling-skip
  case.
- `tests/test_validation.py`'s pinned `ERROR_ORDER` gains
  `cross_entry_references`.
- `docs/WORKFLOW_GUIDE.md`: entries of one list can't read each other (use
  separate runs plus `output:`), and same-key pairing is the only
  member-to-member read.
- `docs/ARCHITECTURE.md`: the previous-result check row gains the rule,
  `cross_entry_key` and its test.

## Verified

On lem, over MCP only, against `develop` @ `168a61ce`: C-F590 to C-F593.
These cover the refusal at `validate_workflow` and at `run_workflow`
submission (no job queued), the `references`/`.property`/index-key forms,
the forms that must stay legal (with one real CPU run of a cross-group read),
the catalog templates `templates/minimax/music-video` and
`templates/ltx2/restore-long` still validating clean, and one error, not
two, on a video read as `image`. The UI twin can't be checked over MCP. The
shared fixture case and `npm test` cover it.

## Bounces per stage

- **Stage A, #1111: 1.**
  - **Architecture review (1):** `cross_entry_key` first lived in
    `dw/references.py` and split member names itself, which made it a second
    owner of `for_each.group_of`'s rule. The map also didn't name it. Fixed
    by moving it into `dw/for_each.py` on top of `group_of` and adding it to
    the map row. Behaviour didn't change.
  - **Tester: 0.**

Cost per stage isn't recorded: the stage comments name no `usage:` figures.
The plan estimated about $3.

## Deferred

- **Chaining or cutaways inside one list.** Out of scope by Don's ruling. It
  would be a new feature with its own design. Until then, a cutaway's A→C
  source uses separate runs and `output:` references (#1093's recipe).
- **A direct Python `Workflow.run` that skips validation** would still
  resolve the reference. Every server, MCP and CLI path validates first, so
  the plan accepted this.
