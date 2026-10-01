# Phase 4 Implementation Plan: carried fixes, guardrails, context diet

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Leave dw in a state that holds without a freeze. The quirks the freeze parked are fixed, every architecture rule that can be a check is one (in dw's CI and in the harness), agent context is a map rather than a manual, and then the freeze lifts.

**Architecture:** Four stages, each merged to `develop` on its own with its own hot zone, as in Phases 2 and 3. A stage's detailed tasks are written when the previous stage merges, on the code that stage left. Stage 4a is detailed below. The order is:
- code first (4a), so the guardrails measure the code they will guard;
- guardrails (4b) before the diet, so CI is already enforcing when the diet re-baselines `claude_md_lines`;
- the seam map before the diet (both 4c), because the diet's pointers go to it;
- the gate (4d) last: the harness's stage C goes live *before* `FREEZE` is deleted.

**Tech Stack:** Python 3.10+, pytest, GitHub Actions; metrics from `scripts/arch_metrics.py` and `scripts/arch_report.py`.

**Spec:** [ROADMAP.md](ROADMAP.md) (Phase 4 row: "Context diet (CLAUDE.md <= 250 lines total) and guardrails installed"; gate: "Guardrails live in dw CI and the harness; freeze lifted"), [ASSESSMENT.md](ASSESSMENT.md) ("Guardrail principles"), and ROADMAP.md, Gate 3, "Carried to Phase 4". Don's rulings of 2026-10-01 bind the whole phase: "in the tension between freezing behavior and stabilizing the codebase, err on the side of stabilization", and the 1,000-line rule has hysteresis ("I don't want a refactor for +2 lines of SLOC").

## Where Phase 4 starts (gate 3, `ee5c4944`; develop `e768378c`, 0.7.0-alpha.1)

- **Ratchets** (`baseline.json`): modules 165, modules over 1,000 lines 0, functions over 150 lines 0, prefix literals 0, test patch targets 284, CLAUDE.md lines 957, duplicate blocks 5, complex functions 7, import cycles 0, modules in cycles 0.
- **Nothing in dw's CI runs the ratchet.** `ci.yml`'s `backend` job installs the dev extra (which carries grimp, networkx, pylint, pygount) and runs ruff and pytest only. `scripts/preflight.sh` doesn't run it either. The only enforcement is the harness's stage B hand-off gate.
- **Modules closest to 1,000 lines:** `workflow.py` 996, `workflow_run.py` 972, `arguments.py` 954, `tasks/task.py` 947, `plan.py` 926.
- **CLAUDE.md, 957 lines in four files:** root 674 (of which "Critical Gotchas" is 395), `ui/CLAUDE.md` 154, `dw_mcp/CLAUDE.md` 95, `dw/server/CLAUDE.md` 34. `.github/copilot-instructions.md` (114) is a sibling that the metric does not count. `tests/test_plugin_skills.py::test_a_skill_is_enumerated_where_the_plugin_describes_itself` requires every plugin skill's name in backticks in the root CLAUDE.md.
- **`Workflow`'s 8 one-line delegators** (gate 3 LCOM4: 9 components). Production callers: `cache_hits` (`worker.py`, `workflow.py`), `validation_context` (`validation.py`, `server/admission.py`), `null_variable_argument_warnings` (`server/routes/library.py`). The other five are called only from tests.

## Stages

| Stage | Scope | Done when |
| --- | --- | --- |
| 4a | **Carried code.** The behaviour items the freeze parked (each with a failing-first test and a 0.7.0 release note); `Workflow`'s 8 delegators replaced by their module functions; the remaining prefix spellings moved onto `references.py`'s helpers, with a metric that counts them. | Every "Carried to Phase 4" code item in ROADMAP.md, Gate 3, is fixed or ruled out in Decisions; `Workflow` LCOM4 is 1; the new prefix metric is 0 and ratcheted |
| 4b | **Guardrails in dw.** The size rule gets its warn band; `arch_metrics.py --check` runs in CI's `backend` job and in `preflight.sh`; the re-baseline rule is written down where the check prints it. | A PR that raises any ratchet fails CI; a module growing past 1,000 lines warns and does not fail until the ceiling |
| 4c | **Seam map, then the context diet.** `docs/ARCHITECTURE.md`: concept → owning module → the rule, one row each. Then every CLAUDE.md is cut to a map: each paragraph is deleted (the knowledge is already in a doc, a docstring or a test), moved into the owning module's docstring, or moved into the seam map. `.github/copilot-instructions.md` becomes a pointer. | the root CLAUDE.md <= 150 lines, every CLAUDE.md through the triage, `claude_md_lines` re-baselined; every plugin skill still named in the root CLAUDE.md; no paragraph moved to `docs/` verbatim without a ruling |
| 4d | **Gate 4.** The harness stage C prompt (`harness/stage-c-guardrails.md`), which Don runs; then `FREEZE` deleted and `hot-zone.txt` emptied; the metrics report, tag, re-baseline, lem deploy, ROADMAP and ASSESSMENT refreshed. | Stage C is committed in the harness, *then* the freeze is lifted on `develop` |

## Carried into Phase 4

From ROADMAP.md, Gate 3, "Carried to Phase 4", plus 3e's carried list. Each item goes to the stage whose files it touches:
- **4a:**
  - `for_each._copy_leaf` sharing leaves, and the step-cache snapshot via `copy_containers`: both **ruled out** in Decisions (4a), measured at gate 4.
  - One sub-workflow path resolver for `create_step_action` and `validation.sub_workflow_errors` (which resolves each path twice to keep its messages).
  - `_run_dir` reset at the top of `run`.
  - The kernels-hub "Cannot find a build variant" message's set order (nondeterministic).
  - `workflow_schema.json`'s `argument_template` description (still says `create_step_action` writes it into the definition).
  - 3d: `gain_audio`'s frame-based region end rounding (the pre-#557 way, one sample short).
  - 3d: `concat_videos` joining a track with no sample rate unresampled (`dissolve_videos` refuses it).
  - `Workflow`'s 8 delegators.
  - The remaining prefix spellings (`startswith` / `removeprefix` / slicing / alias constants, the `routes/assets.py` f-strings) and a metric that counts them.
- **4b:** the module-size warn band.

## Global Constraints (all stages)

- The freeze holds until 4d deletes `FREEZE`: no net-new features or functionality. Phase 4 may change behaviour where Don's 2026-10-01 ruling covers it (a parked quirk or duplicate path removed), and every such change gets a line under `### 0.7.0` in `docs/RELEASING.md` in the same commit.
- `scripts/arch_metrics.py --check docs/stabilization/baseline.json` passes at the end of every task. Phase 4 names no `modules` rise; a new module is a regression unless a stage's Decisions name it.
- Every new test fails before its fix. No count-pinning tests. Add no string `patch("dw...")` targets. A task that moves a patched name retargets the patch in the same commit.
- No compatibility shims: a moved or deleted name is imported from its home by every caller, tests included (Phase 3, Decisions).
- Behaviour-preserving tasks prove preservation with the existing suite: it passes unchanged, apart from the import paths, call spellings and patch targets the task changes.
- Tests: `venv/bin/python -m pytest -q -x -p no:cacheprovider` from the worktree root, plus `ruff check` and `ruff format --check` on `dw dw_mcp tests`. The worktree's `venv` is shared with the main checkout, so never run `pip install -e .` from the worktree.
- Never use `git stash`, in any form, including `git stash list`.
- Filesystem access keeps going through a `dw/security.py` validator, and a validator that moves is re-modelled in `.github/codeql/` in the same commit. Each stage's merge checks the CodeQL run on `develop` and treats a new alert as a finding.
- No lem deploy before gate 4.

## Decisions (rulings, 2026-10-01; each open to Don's correction)

- **The diet's budget is the root CLAUDE.md, which every session loads (Don, 2026-10-01: "economize tokens but not compromise quality").**
  - The root file is 51 KB, about 12-13k tokens, read by every agent session and subagent working in the repo. The three sub-files total about 18 KB and load only when work touches their directory.
  - Root: **<= 150 lines**, a hard target. Sub-files: the same per-paragraph triage (below), with no line quota. A rule that keeps a UI or MCP agent from drifting stays where that agent reads it.
  - The ROADMAP row's "<= 250 total" becomes "root <= 150; total whatever the triage leaves, then ratcheted". Expected total: about 250-300.
  - Cost if wrong: the sub-files keep a few hundred tokens they could have shed, paid only in sessions that work there.
- **The diet deletes before it moves.** Every CLAUDE.md paragraph goes one of three ways, decided per paragraph in a table committed with 4c: *deleted* (a doc, a docstring or a test already says it; the table names which), *docstring* (it is a rule about one module, so it moves to that module's docstring) or *seam map* (it is a rule spanning modules, so it becomes one row in `docs/ARCHITECTURE.md`, a sentence, not the paragraph). Nothing moves to `docs/*.md` verbatim.
  - Why: ASSESSMENT's "docs churn exceeds code churn". Moving 700 lines of prose from CLAUDE.md into docs takes it out of agent context and keeps all the churn.
  - Cost if wrong: something an agent needed falls out of context. The seam map is the net: it names every owner, so an agent finds the module, and the module's docstring holds the rule.
- **After the diet, `claude_md_lines` only goes down.** No later fix adds a CLAUDE.md line without a Decision; a fix that wants one writes it in the owning module's docstring or the seam map. This is the ratchet as ASSESSMENT intended it ("not more prose in agent context").
- **The size rule's warn band (Don, 2026-10-01: hysteresis).**
  - The ratchet `modules_over_1000_lines` is replaced by `modules_over_size_ceiling` at a ceiling of **1,100 lines**, ratcheted at 0.
  - `--check` (and the default output) also prints a non-failing `warning:` line for every module between 1,001 and 1,100 lines, naming it and its length. A warning is not a metric, so it never enters `baseline.json` and the harness's key-by-key comparison never sees it.
  - `functions_over_150_lines` stays a cliff. No ruling asked for a band there, and a 150-line function is already four screens.
  - Cost if wrong: a module can sit at 1,099 indefinitely. The warning makes that visible on every run; tightening is a one-line change.
  - The key rename is safe for stage B's harness ratchet: `regressions()` skips a key missing on either side, so the comparison across the rename sees neither key for one session and both sides of it afterwards.
- **The re-baseline rule.** `baseline.json` may be lowered by any commit that improves a ratchet, and is rewritten at every gate. It is raised only by a commit that names the rise and why in its message, and from 4d on the harness refuses that unless the issue carries `arch-approved` (stage C). The check prints this rule when it fails.
- **Release: open.** Phase 4 changes behaviour (4a) and develop is 0.7.0-alpha.1. Whether gate 4 ships 0.7.0 is Don's call at the gate; 4a collects the notes under `### 0.7.0` either way.

## Review Focus (all stages)

1. **A parked fix that changes a message an agent reads.** Every 4a behaviour change is listed under `### 0.7.0` in `docs/RELEASING.md`, and the MCP surface snapshot (`scripts/surface_snapshot.py`) diff at each merge is exactly that list.
2. **A delegator's caller left on the method.** After 4a, `git grep -nE "(workflow|candidate|self|wf|child|Workflow)\.(validation_context|sub_workflow_warnings|adapter_warnings|inherited_vram_warnings|slice_past_end_warnings|shot_span_warnings|null_variable_argument_warnings|cache_hits)\b" dw dw_mcp tests scripts` returns nothing. The receiver is in the pattern because the destination module functions share the names (`workflow_run.cache_hits`, `adapter_compatibility.adapter_warnings`).
3. **The CI ratchet passes on a broken environment.** If `arch_metrics.py` cannot import grimp, the CI step fails rather than skipping (stage B's "fails closed", in dw).
4. **A diet that drops a pinned name.** `test_a_skill_is_enumerated_where_the_plugin_describes_itself` and every other test reading a CLAUDE.md still pass after 4c, without the test being loosened.
5. **The freeze lifted before the harness can hold.** 4d deletes `FREEZE` only after Don confirms stage C is committed in the harness.

---

## Stage 4a: carried code

Work on branch `stabilization/phase-4a` in the worktree, from `develop` at `e768378c` or later.

**What exists (survey, 2026-10-01, at `e768378c`; full notes in [phase-4-surveys/carried-items.md](phase-4-surveys/carried-items.md)).** Line numbers are at that commit.

- **The 8 delegators** (`dw/workflow.py`):
  - `validation_context` :498 → `validation.workflow_context`;
  - `sub_workflow_warnings` :484, `adapter_warnings` :520, `inherited_vram_warnings` :531 (with an `if not index: return []` guard), `slice_past_end_warnings` :546, `shot_span_warnings` :556 and `null_variable_argument_warnings` :567, each → `validation.run_warning_check(self, "<name>", arguments[, ceiling_index=])`;
  - `cache_hits` :604 → `workflow_run.cache_hits`.
  - Production callers: `validation.py:662` and `:712`, `server/admission.py:158` (`validation_context`); `server/routes/library.py:298` (`null_variable_argument_warnings`); `worker.py:513` (`cache_hits`).
  - The other five are called only by tests and by `scripts/surface_snapshot.py:256` (`getattr(workflow, name)()` over its `WARNING_CHECKS` tuple).
  - Test seams on the methods: `tests/test_admission.py:499` monkeypatches `(Workflow, "validation_context")`; the fake workflows at `tests/test_worker_execute.py:287` and `:322` define their own `cache_hits`.
- **Sub-workflow paths are resolved at seven sites with four preambles** around the one resolver `library.resolve_sub_workflow` (`library.py:616`):
  - `Workflow.resolve_sub_workflow_path` (`workflow.py:439`): builtin branch with `builtin_root()` and an existence check; otherwise falls back to `catalog_root_dir(self.file_spec)`.
  - `Workflow._sub_workflow_action` (`workflow.py:908`, from `create_step_action`): **re-implements it inline** (:919-968). Its builtin root is `os.path.join(dirname(__file__), "workflows")`, it has no existence check, and both copies strip the prefix with `path.replace(references.BUILTIN, "")` (:450, :921), which removes the text anywhere in the path, not only as a prefix.
  - `Workflow.open_sub_workflow` (:475) calls `resolve_sub_workflow_path`.
  - `validation.sub_workflow_errors` resolves each path twice (:776 directly, :793 through `open_sub_workflow`) to keep its two message forms. `sub_workflow_argument_warnings` (:821) resolves a third time.
  - `realize.read_sub_workflow` (`realize.py:232`) and `server/routes/jobs.py:557` call `resolve_sub_workflow` with their own preamble, without the `catalog_root_dir` fallback.
- **`_run_dir`, `_run_version` and `_run_dir_inherited`** are class attributes (`workflow.py:167-177`), set in `workflow_run._claim_run_dir` (:557, :566) and by a parent for its child (`workflow.py:993`). `run()` (:623) resets `pipeline_ownership`, `manifest` and `_elided_steps` but not these. On a worker reusing a `Workflow`, if `prepare_run` raises before the claim, the `finally` (:713) sees the previous run's directory as its own and rewrites that run's `manifest.json` with the failed run's record.
- **The kernels message** comes from the third-party `kernels` package; dw embeds it as `f"'{value}' {KERNEL_FAULT_MARKER}: {e}"` (`kernel_availability.py:126`). Its per-variant lines (each starting `torch`) come out in set order. Only `scripts/surface_snapshot.py`'s `stable_message` sorts them, so an author sees a different order on every process.
- **`workflow_schema.json:107`**, `argument_template`'s description, says "Written by create_step_action from the step's 'arguments' block". Since 3e the handed arguments live on the child as `_handed_arguments` (`workflow.py:977`), and the property reads an authored `argument_template` only as the fallback. The schema is part of the surface snapshot.
- **`gain_audio`'s frame region** (`tasks/audio_utils.py:332-343`) rounds `start` and `length` separately (`frames_to_samples(num_frames, ...)`), so `start + length` can miss `round((start_frame + num_frames) / fps * sr)` by a sample. `slice_audio` goes through `task_domains.slice_region` (:406), which rounds the end once (#557).
- **`reconcile_sample_rates`** (`tasks/joins.py:70`) takes `skip_unrated=True` by default. `concat_videos` uses the default, so a track with no sample rate is left out of the rate choice and joined unresampled, which plays it at the wrong speed and pitch. `dissolve_videos` passes `skip_unrated=False` and fails in `dsp.resample_waveform` ("needs a sample_rate above zero"); `tests/test_dissolve_videos.py:94` pins that.
- **Prefix spellings: about 70 sites (84 lines) in 31 modules, none in `dw_mcp`.**
  - Alias constants (16): `ASSET_PREFIX` (`assets.py:30`), `OUTPUT_PREFIX` (`runs.py:57`), `PROMPT_PREFIX` (`prompts.py:28`), `BUILTIN_PREFIX` and `VARIABLE_PREFIX` (`realize.py:42-43`), and `_UNRESOLVED_PREFIXES` in eight modules: five bind it to `SUBSTITUTED` and three to `UNRESOLVED`, under one name. Also `RESERVED_TEXT_PREFIXES` (`prompts.py:33`, a six-tuple no named tuple matches) and `SHOT_REFERENCE_PREFIX` (`shots.py:42`, `previous_result:shot@`).
  - `startswith(<constant>)` 31 lines; `removeprefix` / `[len(<constant>):]` / `replace` 29 lines, about 12 of them `.removeprefix(X).strip()`.
  - Built by hand (8): `references.VARIABLE + name` (`elision.py:210`), `for_each.py:203`, `:263` and `:292`, `realize.py:201`, the log f-strings `routes/assets.py:363` and `:433` (literal `asset:`), and `"builtin:h3_context_ir.json"` (`server/enhancers.py:29`).
  - The JSON schema spells `^variable:` four times and `^constraint:` once (:103, :201, :211, :531, :400).
  - `prefix_literals` counts only an `ast.Constant` exactly equal to a bare prefix, so it sees none of this.

### Decisions (4a)

- **The 8 delegators are deleted; callers call the module function.** Production: `validation.workflow_context(workflow, ...)`, `validation.run_warning_check(workflow, "null_variable_argument_warnings")`, `workflow_run.cache_hits(workflow, arguments)`. Tests call `validation.run_warning_check(workflow, "<name>", ...)`. `surface_snapshot.py` does the same by name. `test_admission.py:499` patches `validation.workflow_context` with `patch.object`. The worker's fakes in `test_worker_execute.py` move to a `patch.object(workflow_run, "cache_hits", ...)`.
  - `inherited_vram_warnings`' empty-index guard moves into the check, if `run_warning_check` does not already return nothing for an empty index. The task checks this first.
  - `Workflow.validation_errors` and `validate` stay methods: they are the class's public entry points, not delegators to a check.
- **One sub-workflow resolver.** `resolve_sub_workflow_reference(path, file_spec, confine_to)` in `dw/library.py` owns the whole preamble: the builtin branch on `builtin_root()` with `ref_name(BUILTIN, ...)` and the existence check, then the `catalog_root_dir` fallback, `resolve_sub_workflow` and `validate_workflow_path`. It returns `(path, root)`.
  - `Workflow.resolve_sub_workflow_path` becomes a one-line call to it, and `_sub_workflow_action`'s inline copy is deleted in favour of that call.
  - `open_sub_workflow(path, resolved=None)` opens an already-resolved path without resolving again. `sub_workflow_errors` resolves once and hands the result on, with both message forms unchanged.
  - `realize.read_sub_workflow` and `routes/jobs.py:557` call it too. That gives them the `catalog_root_dir` fallback they lacked, so a catalog sub-workflow a run could open is now also digested and costed. That goes in the release notes.
  - Whether a builtin missing from `builtin_root()` now fails at `create_step_action` with `SubWorkflowNotFound` instead of later is checked against the tests; if its message changes, the change goes in the release notes.
- **`run()` resets `_run_dir` and `_run_version` at its top, unless `_run_dir_inherited`.** A child's values come from its parent, which sets them on a fresh `Workflow` just before `run`.
- **dw sorts the kernels variant lines itself.** `stable_message`'s rule moves into `kernel_availability.py` and applies where the fault is raised; `surface_snapshot.py` drops its copy. A message with no `Cannot find a build variant` text passes through untouched, so a change in the `kernels` package's format degrades to today's behaviour, not to an error.
- **`argument_template`'s description is reworded** to what the code does: the handed arguments are held on the child workflow at run time and never written into the definition, and an authored value is read as the fallback. This is a surface snapshot change and a release-note line.
- **`gain_audio`'s region goes through `slice_region`,** keeping its own "needs 'fps'" message and its whole-track branch. A frame-addressed region's end then matches `slice_audio`'s to the sample.
- **`concat_videos` refuses a track with no sample rate, as `dissolve_videos` does,** and `skip_unrated` is deleted from `reconcile_sample_rates`.
  - Why: playing a waveform at a rate it was not sampled at changes its speed and pitch. `as_track` and #140 already refuse that everywhere else.
  - The refusal names the input (`"concat_videos: '<name>' has audio with no sample rate"`), not the `resample_waveform` text dissolve surfaces today, and dissolve gets the same message. This is a release-note line.
  - Where an unrated track comes from: only from a pipeline none of whose components reports a rate. `attach_audio_sample_rate` (`pipeline_processors/pipeline.py:796`) then logs "Pipeline generated audio but no component reports its sample rate - set 'sample_rate' ... or 'audio_sample_rate' ... in the step result". The catalog's audio pipelines (LTX-2's vocoder, H3) report one. The ruling assumes the catalog's three `concat_videos` templates (`assemble-and-score`, `minimax/music-video`, `minimax/dialogue-short`) never feed it an unrated track. Task 4 checks that *first*. If one can, the fix moves upstream (the step's declared rate stamped onto the `AudioVideo` at extraction), and the task stops and reports instead.
  - The refusal names the same remedy as the generation-time warning (set the step result's `audio_sample_rate`).
  - Cost if wrong: a hand-written workflow over a pipeline that reports no rate stops joining. It was already joining wrong (wrong speed and pitch), with a warning at generation.
- **Ruled out: sharing leaves in `for_each._copy_leaf` and in the step-cache snapshot.** Both carry items assumed a leaf that is safe to share, and the survey found otherwise.
  - `_copy_leaf` deep-copies a member's media leaves on purpose. Tasks mutate their inputs in place (`conform_artifact` stamps fps onto what `select` hands back, the comment at `workflow.py:195`), so a shared leaf would carry one member's edit into the next.
  - The cache snapshot is the cache's *key*. A shared leaf mutated after the snapshot changes the stored key with it, and `deep_equal`'s `a is b` shortcut then reports a hit for a changed input.
  - What staying costs: memory and time per member and per cached step. That is measured, not assumed: the gate 4 lem timing runs a `for_each` over media. If it shows a real cost, sharing becomes its own design with a mutation audit, after the freeze.
- **References are built and read only through `references.py`.**
  - New in `references.py`: `RESERVED_TEXT` (the six-tuple `prompts.py` owns today) and `LAZY_MEDIA = (PREVIOUS_RESULT, VARIABLE)` (the pair spelled inline three times). `SHOT_REFERENCE_PREFIX` stays in `shots.py` as `make_ref(PREVIOUS_RESULT, "shot" + MEMBER_SEPARATOR)`: it is a shots concept built from the owned parts.
  - Every alias constant is deleted, and every caller uses `references.<NAME>` or a helper. `_UNRESOLVED_PREFIXES` goes away with them, so no module names a tuple after something it isn't.
  - `x.removeprefix(K).strip()` becomes `ref_name(K, x).strip()`. `ref_name` gets no strip variant: whitespace hygiene is the caller's rule. Where a site relied on `removeprefix` returning an unprefixed value unchanged, the task checks its guard and keeps the semantics. `ref_name` returns `None` there.
  - The JSON schema's patterns stay (a schema is data). A new test pins each `pattern` that begins `^variable:` or `^constraint:` to `references.VARIABLE` / `references.CONSTRAINT`.
- **The metric: `prefix_handling`, a second ratchet beside `prefix_literals`.** Outside `PREFIX_OWNERS`, in `dw` and `dw_mcp`, it counts:
  - (a) a call to `startswith`, `removeprefix`, `removesuffix`, `replace`, `split` or `partition` whose argument is a `references` constant or tuple (as `references.X`, an imported name `X`, or a name bound at module level to either);
  - (b) a slice whose start is `len(<that>)`;
  - (c) `+` or an f-string with one of them as an operand;
  - (d) a module-level assignment whose value is one of them;
  - (e) a string constant that *starts with* a prefix (`"builtin:h3_context_ir.json"`), or an f-string fragment that *ends with* one (`" as asset:"`, the fragment before a name is spliced in). Prose that names a prefix with no name after it (`"'asset:' reads ..."`) is not counted.
  - So a message that embeds a *reference* builds it with `make_ref`, error and log text included: `f"Kept output {name} as {make_ref(ASSET, asset_name)}"`. That is a ruling, not a side effect. Any f-string message that rule (e) catches is migrated in Task 6, not explained away.
  - It lands counting today's sites, failing first against a fixture tree in `tests/test_arch_metrics.py`. The baseline takes that count, and the migration task drives it to 0 and re-baselines.

### Review Focus (4a)

1. **A failed run on a reused `Workflow` leaves the previous run's manifest alone.** Test: run once into a nested-layout run directory, make `prepare_run` raise on a second run of the same instance, and assert the first `manifest.json` is byte-identical. A child's inherited directory still holds: the existing composed-run tests pass.
2. **A sub-workflow path the run can open, validation can open, and the other way round.** Tests through the one resolver:
   - a relative path that resolves only through `catalog_root_dir`;
   - `builtin:` with a name containing `builtin:` again (the `replace` bug);
   - a missing builtin;
   - a path escaping its root. That one is refused with the `validate_workflow_path` message, by every one of the seven call sites.
3. **Message parity where the plan promises it.** `sub_workflow_errors`' two message forms, `gain_audio`'s fps message and the seven warning checks' texts are unchanged; the surface snapshot diff is exactly the `argument_template` description.
4. **The prefix migration changes no answer.** For each `removeprefix` site turned into `ref_name`, an unprefixed input reaches the same result as before; the reviewer checks each guard. A test where the site has none.
5. **The metric counts what it claims and nothing it doesn't.** The fixture tree holds one instance of each form (a)-(e) and one prose mention. The count is exact, and the prose mention is not counted.

### Before Task 1: the hot zone goes live

Once Don approves this plan, and before any 4a code changes:
- commit the 4a list below into `docs/stabilization/hot-zone.txt` on `develop`, with the plan docs and ROADMAP.md's Phase 4 row (Plan: `[phase-4.md](phase-4.md) (staged: 4a-4d)`, as the Phase 3 row reads);
- push;
- start the `stabilization/phase-4a` branch from that commit.

### Task 1: The 8 delegators

Smallest first: it takes about 50 lines out of `workflow.py` (996), which the later tasks then have room in.

- [ ] **Step 1:** Check whether `run_warning_check(workflow, "inherited_vram_warnings", arguments, ceiling_index=None)` and `ceiling_index={}` already return `[]`. If not, move the guard into the check's registry entry first, with a test that fails first.
- [ ] **Step 2:** Repoint the five production call lines (Decisions), `scripts/surface_snapshot.py`, and every test caller (survey list: `test_validation.py`, `test_workflow.py`, `test_lora_disable.py`, `test_vram_inheritance.py`, `test_slice_preflight.py`, `test_shot_span_preflight.py`, `test_realize.py`, `test_workflow_step_cache.py`, `test_admission.py`, `test_worker_execute.py`).
- [ ] **Step 3:** Delete the 8 methods. Run `git grep` per Review Focus (all stages) item 2, then the suite, ruff, and `scripts/surface_snapshot.py`; the snapshot must be byte-identical.
- [ ] **Step 4:** Run `scripts/arch_report.py` for LCOM4 on `Workflow`; expect 1. `validate` reads `self.name` and calls `validation_errors`, so both join the core component. `validation_errors` passes `self` on and reads no attribute, but `validate` calling it links them. If it reads more than 1, record each remaining component in the report. Fix it in this task only when it is one more delegator.
- [ ] **Step 5:** Commit. No release note: no surface changes.

### Task 2: One sub-workflow resolver

- [ ] **Step 1:** Write the Review Focus 2 tests against `resolve_sub_workflow_reference`, plus one per call site showing it reaches the resolver: `create_step_action`, `open_sub_workflow`, `sub_workflow_errors`, `read_sub_workflow` and the observed-cost lookup. Run them; the resolver tests fail on the missing name, and the `replace` and `catalog_root_dir` cases fail on today's code.
- [ ] **Step 2:** Add `resolve_sub_workflow_reference` to `dw/library.py`. Move every call site onto it. Delete `_sub_workflow_action`'s inline copy, and add `open_sub_workflow`'s `resolved` parameter.
- [ ] **Step 3:** The suite, ruff and the snapshot. Any changed message goes under `### 0.7.0`.
- [ ] **Step 4:** Commit, with the release-note lines.

### Task 3: Three small fixes: the run directory, the kernels message, the schema description

- [ ] **Step 1:** Review Focus 1's test, and a kernels test: a fault carrying three `torch...` variant lines in reverse order comes out sorted, and the rest of the message keeps its place. Both fail.
- [ ] **Step 2:** Reset in `run()`; move `stable_message` into `kernel_availability.py` (applied where the fault is raised) and delete it from `surface_snapshot.py`; reword the `argument_template` description.
- [ ] **Step 3:** The suite, ruff and the snapshot. The only snapshot diff is the description.
- [ ] **Step 4:** Commit, with three release-note lines: a failed run no longer rewrites the previous run's manifest; the build-variant lines are sorted; the schema text.

### Task 4: The two audio rules

- [ ] **Step 1:** The ruling's premise. For each of the three catalog templates using `concat_videos`, trace each `videos` input back to the step that made it, and confirm that step's pipeline reports a rate (`_component_sample_rate`). Record that in the report. If any can arrive unrated, stop with DONE_WITH_CONCERNS (Decisions (4a)).
- [ ] **Step 2:** Tests, failing first:
  - `gain_audio` over a frame region where `round(start) + round(length) != round(end)`. Pick fps and rate so it bites, for example 24 fps at 44,100 Hz with `start_frame=1, num_frames=1`; compute it in the test.
  - `concat_videos` over two inputs, one with audio and no sample rate, refused with the input's name.
  - `dissolve_videos` with the same message.
- [ ] **Step 3:** `gain_audio` on `slice_region`; `reconcile_sample_rates` loses `skip_unrated` and raises the named refusal; update `test_dissolve_videos.py:94` to the new message.
- [ ] **Step 4:** The suite, ruff and the snapshot (a task description that mentions the rule may change; list it). Commit, with two release-note lines.

### Task 5: The `prefix_handling` metric

- [ ] **Step 1:** In `tests/test_arch_metrics.py`, add a fixture tree with one instance of each form (a)-(e), one prose mention, and one use inside `dw/references.py`. Assert `prefix_handling` is exactly 5. Fails: no such key.
- [ ] **Step 2:** Implement it in `scripts/arch_metrics.py` beside `prefix_literals`. Update the docstring's counting rules and ROADMAP.md's Metrics paragraph in two lines.
- [ ] **Step 3:** Run it on the worktree. The count should be close to the survey's (about 84 lines; the metric counts nodes, so pairs count twice). Explain any gap above 10% in the report before baselining.
- [ ] **Step 4:** Re-baseline (`--write`) with `prefix_handling` at today's count; the commit message names the new key and its number.

### Task 6: The prefix migration

- [ ] **Step 1:** Add `RESERVED_TEXT` and `LAZY_MEDIA` to `references.py`, and the schema-pattern test (fails only if a pattern drifts, so it is a characterization test; say so in its docstring).
- [ ] **Step 2:** Work module by module, per the survey's list, deleting each alias when its last importer has moved. After each module, run its tests.
- [ ] **Step 3:** For every `removeprefix` site, check its guard (Review Focus 4). Write a test where an unprefixed input reaches it and the semantics could differ.
- [ ] **Step 4:** `prefix_handling` and `prefix_literals` are 0. The suite, ruff and the snapshot are byte-identical, apart from the two log lines' text if their wording changed. Re-baseline both to 0.
- [ ] **Step 5:** Commit. No release note.

### Task 7: Stage 4a merge

- [ ] **Step 1: Re-baseline.** Run `venv/bin/python scripts/arch_metrics.py --write docs/stabilization/baseline.json`. The diff against the committed baseline is exactly: `prefix_handling` 0 (new), and anything that went down. `modules` is 165.
- [ ] **Step 2: Docs.** Find every sentence naming a deleted delegator, alias or `skip_unrated` (`git grep -nE "ASSET_PREFIX|OUTPUT_PREFIX|PROMPT_PREFIX|_UNRESOLVED_PREFIXES|skip_unrated|Workflow\.(cache_hits|validation_context|[a-z_]+_warnings)" CLAUDE.md '*/CLAUDE.md' docs/ ':!docs/stabilization/'`) and fix it. CLAUDE.md may only shrink.
- [ ] **Step 3: Hot zone.** Put `hot-zone.txt` back to the two standing entries.
- [ ] **Step 4: Merge.** Merge `stabilization/phase-4a` to `develop` with `--no-ff` and push. Check the CodeQL and CI runs on `develop`. No lem deploy. Report the changed metrics rows and `Workflow`'s LCOM4 to Don.
- [ ] **Step 5: Detail stage 4b** in this file, on the code 4a left, with its hot zone. Cross-check the design with Fable before Task 1 of 4b.

### Hot zone (4a)

```
dw/workflow.py
dw/workflow_run.py
dw/validation.py
dw/library.py
dw/realize.py
dw/worker.py
dw/kernel_availability.py
dw/workflow_schema.json
dw/references.py
dw/tasks/audio_utils.py
dw/tasks/joins.py
dw/tasks/concat_videos.py
dw/tasks/dissolve_videos.py
dw/server/admission.py
dw/server/routes/library.py
dw/server/routes/jobs.py
dw/server/routes/assets.py
dw/server/catalog.py
dw/server/enhancers.py
dw/server/exports.py
dw/server/jobs.py
dw/server/outputs.py
dw/adapter_compatibility.py
dw/argument_media.py
dw/arguments.py
dw/assets.py
dw/content_types.py
dw/elision.py
dw/for_each.py
dw/locations.py
dw/plan.py
dw/probe_paths.py
dw/prompts.py
dw/reference_limits.py
dw/reference_names.py
dw/runs.py
dw/shots.py
dw/step_value_checks.py
dw/subfolders.py
dw/variable_constraints.py
dw/video_extensions.py
dw/vram_estimate.py
dw/server/catalog_shape.py
scripts/arch_metrics.py
scripts/surface_snapshot.py
docs/stabilization/
```

It is a long list because the prefix migration touches a line or two in 31 modules. Most of those edits are one-line import or call changes, so a harness edit to one would be a small merge conflict, not a lost fix. If the harness needs one of these files for a field bug during 4a, Don can drop it from the list. Task 6 then rebases over that fix.

## Stage 4b: guardrails in dw

Detailed when 4a merges.

## Stage 4c: seam map, then the context diet

Detailed when 4b merges.

## Stage 4d: gate 4

Detailed when 4c merges.

The stage C prompt (`harness/stage-c-guardrails.md`, same format as A and B) covers the six guardrails stage B deferred to it, split by where each runs:
- **dw-side, already done by 4b/4c, which the prompt points at:** the CI ratchet (4b) and the seam map (`docs/ARCHITECTURE.md`, 4c).
- **Harness-side, which the prompt asks for:**
  - an architecture reviewer that reads the seam map and refuses a hand-off that adds a second owner for a concept;
  - new-module approval: stage A's new-file refusal under `dw/`, `dw_mcp/` and so on outlives `FREEZE`, still waived by `arch-approved`;
  - build-vs-buy: a new hand-rolled implementation of something a dependency already covers is refused, with the reviewer as the check;
  - the consolidation cadence: a periodic pass, run by the curator, that reads the gate report's change-coupling pairs and files consolidation issues for Don.
- **Also:** agents may not edit `baseline.json` upward without `arch-approved` (the re-baseline rule).

`FREEZE`'s own text says the freeze lifts "at the Phase 4 gate"; this ordering is what that means. Its first criterion: Don confirms the harness stage C prompt is committed and its tests pass in `/Users/don/testing/harnest`. Then, on `develop`: `FREEZE` deleted, `hot-zone.txt` emptied, `arch_report.py` with a Gate 4 column into ROADMAP.md, re-baseline, tag `stabilization-gate-4`, lem deploy and smoke, ROADMAP row 4 status, ASSESSMENT refreshed, the Claude Doc's metrics table, and memory.
