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
| 4b | **Guardrails in dw.** The size rule gets its warn band; `arch_metrics.py --check` runs in CI's `backend` job and in `preflight.sh`; the re-baseline rule is written down where the check prints it. | A PR that raises any ratchet fails CI (a direct push to `develop` goes red after landing; the harness's gate is what blocks those); a module growing past 1,000 lines warns and does not fail until the ceiling |
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
  - The key rename is safe for stage B's harness ratchet: stage B measures both trees with `origin/develop`'s script, so both sides carry the new key from the moment the rename lands.
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
- **One sub-workflow resolver.** `resolve_sub_workflow_reference(path, base_dir, confine_to)` in `dw/library.py` owns the whole preamble: the builtin branch on `builtin_root()` with `ref_name(BUILTIN, ...)` and the existence check, then the `catalog_root(base_dir)` fallback, `resolve_sub_workflow` and `validate_workflow_path`. It returns `(path, root)`.
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
  - New in `references.py`: `RESERVED_TEXT` (the six-tuple `prompts.py` owns today) and `LAZY_MEDIA = (PREVIOUS_RESULT, VARIABLE)` (the pair spelled inline three times). `SHOT_REFERENCE_PREFIX` is deleted: its branch in `shots.py` had a body identical to the plain `previous_result:` branch (a `shot@` member and a step are both read as `ref_name(...).split(".", 1)[0]`), so the branches merged and the constant went.
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

Work on branch `stabilization/phase-4b` in the worktree, from `develop` at `4bb997c5` (the 4a merge) or later.

**What exists (at `4bb997c5`).**
- `scripts/arch_metrics.py`:
  - `measure()` counts `modules_over_1000_lines` with a bare `> 1000`;
  - `regressions(current, baseline)` is a pure key-by-key "number rose", skipping a key missing on either side;
  - `main()` prints the metrics JSON, then each regression on `--check`, and exits 1 on any.
  - Nothing else is printed: no warnings, and no hint of what to do on failure.
- Other places that name `modules_over_1000_lines`: `scripts/arch_report.py:32` (a row label) and `tests/test_arch_metrics.py:42`. The harness's stage B compares whatever keys both sides have.
- `.github/workflows/ci.yml`'s `backend` job installs `requirements.txt` + `requirements-test.txt` (`-e .[dev]`, which carries grimp via import-linter, networkx, pylint and pygount), then runs ruff format/check on `dw dw_mcp tests` and `pytest -q`. It runs on push to `master`/`develop` and on every PR.
- `scripts/preflight.sh` runs `ruff format .` and `ruff check . --fix` over the whole repo, then pytest, integration and the UI preflight. Whole-repo `ruff format` rewrites the Python code blocks in `docs/**/*.md` (gate 3 had to revert those by hand). CI formats only `dw dw_mcp tests`.
- After 4a, the modules nearest the line are `workflow.py` (about 945), `workflow_run.py` 972, `arguments.py` 954 and `tasks/task.py` 947.

### Decisions (4b)

- **`modules_over_size_ceiling` replaces `modules_over_1000_lines`,** at `SIZE_CEILING = 1100`, with `SIZE_WARNING = 1000` beside it (the warn band; frame Decisions). `measure()` stays pure numbers. A new `size_warnings(root)` returns `[(path, lines)]` for modules in the band, and `main()` prints each as `warning: dw/x.py is 1,043 lines (warn above 1,000, fail above 1,100)` on every run, `--check` included, without changing the exit code. `arch_report.py`'s row is renamed to match; it measures every column with today's script, so earlier gates read on the new key.
- **The check says what to do when it fails.** After the regressions, `--check` prints dw's re-baseline rule in two lines:
  - lower `baseline.json` freely when a ratchet improves;
  - raise it only in a commit whose message names the rise and why.

  The harness enforces `arch-approved` itself; dw's script doesn't name it.
- **CI: one step in `backend`, after the tests:** `python scripts/arch_metrics.py --check docs/stabilization/baseline.json`.
  - It fails closed: `import_graph` runs grimp in a subprocess with `check=True`, so an ImportError there raises `CalledProcessError`, the script exits non-zero and the step fails. Task 1 pins that with a test. The subprocess runs with `cwd=root`, so a `grimp.py` that raises, dropped in the fixture tree's root, shadows the real one.
  - The step adds no install, because the dev extra is already there.
  - Cost: `--check` takes 7 s on the real tree (measured 2026-10-01), small next to the tests.
  - What it blocks and what it doesn't: a PR that raises a ratchet goes red before it merges. A direct push to `develop` (the harness and Don push it directly) goes red *after* it lands, so CI reports there rather than blocks. The blocker for direct pushes is the harness's stage B hand-off gate, and stage C after it.
- **Preflight runs the same check, and formats what CI formats.**
  - It gains a `run_step "architecture ratchet"` step.
  - `ruff format` / `ruff check --fix` move from `.` to `dw dw_mcp tests scripts`, so preflight stops rewriting the docs' code blocks. CI's own format and lint steps gain `scripts` to match.
- **`baseline.json` stays at `docs/stabilization/baseline.json`.** dw's CI reads it by that path, and from stage C on so does the harness's waiver check. Moving it later means moving both.

### Review Focus (4b)

1. **A module in the band warns and passes, and one over the ceiling fails.** Fixture modules at 1,000 lines (no warning), 1,001 (warning, exit 0) and 1,101 (`modules_over_size_ceiling: 0 -> 1`, exit 1). The test reads `main()`'s output and exit code.
2. **The check fails closed.** With a `grimp.py` that raises ImportError in the fixture tree's root (the subprocess runs there, so it shadows the real one), `--check` exits non-zero. It is never a pass with a skipped metric.
3. **CI's step can actually fail.** On the branch, before the merge, one throwaway commit raises a ratchet (an extra module) and is pushed to a draft PR, and the run is red at that step. A plain revert commit follows; no force-push.

### Before Task 1: the hot zone goes live

Commit the 4b list below into `docs/stabilization/hot-zone.txt` on `develop` with this section, push, and branch `stabilization/phase-4b` from that commit.

### Task 1: The size band and the failing check's message

- [ ] **Step 1:** Write the Review Focus 1 and 2 tests in `tests/test_arch_metrics.py`, plus one asserting that `--check` prints the two-line re-baseline rule only when it fails.
  - The existing `test_a_long_function_and_a_long_module_are_counted` builds about 1,052 lines, which is inside the new band. Grow it past 1,100 so it stays the "counted" case. Add a separate 1,001-1,100-line fixture for "warns, exit 0". Run them; they fail on today's script (no key, no warning, no rule text).
- [ ] **Step 2:** Implement the Decisions (4b) items in `scripts/arch_metrics.py`: the key rename, `SIZE_CEILING` and `SIZE_WARNING`, `size_warnings`, and the printing in `main()`. Rename the row in `arch_report.py`, and update `tests/test_arch_metrics.py:42` to the new key. Update the script's docstring.
- [ ] **Step 3:** Run it on the worktree. `modules_over_size_ceiling` is 0 and no module is in the band. Re-baseline (`--write`). The diff is the key rename only, with the same value 0.
- [ ] **Step 4:** ROADMAP.md's Metrics paragraph: one line for the warn band. The suite and ruff pass. Commit, with a message that names the key rename (the harness's stage B sees it).

### Task 2: CI and preflight

- [ ] **Step 1:** `ci.yml`'s `backend` job:
  - the new step after `Tests`;
  - `Format check` and `Lint` widened to `dw dw_mcp tests scripts`. Run `ruff format --check scripts` locally first and fix anything it reports in the same commit.
- [ ] **Step 2:** `preflight.sh`: the ruff steps scoped to `dw dw_mcp tests scripts`, and a `run_step "architecture ratchet" python scripts/arch_metrics.py --check docs/stabilization/baseline.json` step after the pytest steps. Its header comment names the new step.
- [ ] **Step 3:** Time `arch_metrics.py --check` on the worktree and record it in the report.
- [ ] **Step 4:** Run `scripts/preflight.sh` in full and confirm it leaves no edits under `docs/` (`git status`).
- [ ] **Step 5:** Commit. Push the branch and open a draft PR against `develop`. Confirm the new CI step runs and passes.
- [ ] **Step 6:** Review Focus 3: push one throwaway commit that adds an empty `dw/_ratchet_probe.py`. Confirm the run fails at the ratchet step with `modules: 165 -> 166`. Then push a plain `git revert` of it and confirm green. Record both run URLs in the report. The merge in Task 3 carries the probe and its revert. That's harmless, but say so in the merge commit.

### Task 3: Stage 4b merge

- [ ] **Step 1:** `arch_metrics.py --check` passes; the baseline diff since 4a is the key rename only.
- [ ] **Step 2:** Docs: `git grep -n "modules_over_1000_lines" -- ':!docs/stabilization/'` returns nothing. RELEASING.md gets no user-facing line; this is tooling only.
- [ ] **Step 3:** Put the hot zone back to the standing entries. ROADMAP Phase 4 row: "4b merged".
- [ ] **Step 4:** Close the draft PR. Merge `stabilization/phase-4b` to `develop` with `--no-ff` and push. The `develop` CI run is green, including the ratchet step. Report to Don.
- [ ] **Step 5:** Detail stage 4c in this file. Cross-check the design with Fable before Task 1 of 4c.

### Hot zone (4b)

```
scripts/arch_metrics.py
scripts/arch_report.py
scripts/preflight.sh
.github/workflows/ci.yml
tests/test_arch_metrics.py
docs/stabilization/
```

## Stage 4c: seam map, then the context diet

Work on branch `stabilization/phase-4c` in the worktree, from `develop` at `edb6e443` (the 4b merge) or later.

**What exists (at `edb6e443`).**
- Four CLAUDE.md files, 957 lines (`claude_md_lines`, ratcheted):
  - root, 674 lines and 51 KB, loaded by every session. "Critical Gotchas" is 395 of those lines, and "Type System" 69.
  - `ui/CLAUDE.md` 154 (design system, assets);
  - `dw_mcp/CLAUDE.md` 95;
  - `dw/server/CLAUDE.md` 34.
- `.github/copilot-instructions.md`, 114 lines. It is a parallel description of the architecture, partly stale (it still describes a 5-minute execution timeout and SHA-256 change detection), and the metric does not count it.
- What reads a CLAUDE.md mechanically:
  - `tests/test_plugin_skills.py::test_a_skill_is_enumerated_where_the_plugin_describes_itself`, which needs every plugin skill's name in backticks in the root file;
  - `arch_metrics.py` and `arch_report.py`, for counting.
- Docs that point into CLAUDE.md: `docs/AGENT_LOOP.md`, `docs/WORKFLOW_GUIDE.md` (its authoring section and the root's Type System say "change both when one changes") and `.claude/skills/model-family-onboarding/references/cold-drill-example.md`.
- No architecture map exists. An agent learns which module owns a concept from CLAUDE.md prose or by grepping.

### Decisions (4c)

- **The triage comes first, and it is a committed table:** `phase-4-surveys/claude-md-triage.md`. It has one row per paragraph or bullet of each CLAUDE.md and of `copilot-instructions.md`, with four columns:
  - where the paragraph is;
  - its first words;
  - the verdict: **delete**, **docstring**, **map** or **keep**;
  - the evidence.
  - Evidence by verdict:
    - delete: the file:line that already says it (a doc, a docstring, or a test whose name or docstring states the rule);
    - docstring: the module it moves to;
    - map: the seam-map row it becomes;
    - keep: why an agent needs it before it knows which module to open.

  A paragraph the triage can't place stays as **keep**, with that said. The diet then argues from the table, and the review checks the table, not 700 lines of prose.
- **What stays in the root CLAUDE.md (<= 150 lines):**
  - the project in two sentences, and the common commands;
  - a "where things are" block that points at the seam map;
  - the security rules, which an agent must know before it opens any file;
  - the plugin-skill names (pinned by the test);
  - the few gotchas that bite at the moment of editing and that no test or check catches. Each is one or two lines, and each ends with a pointer to its owner.
  - Everything that describes how a subsystem works goes to its module's docstring, or to a seam-map row.
- **The seam map is `docs/ARCHITECTURE.md`:** a table of concept, owning module(s), the rule in one sentence, and the test or check that enforces it, if any. It covers the concepts the triage sends there, plus every owner the stabilization created: `references`, `library`, the validation registry, `workflow_run`, `step_cache`, `worker_protocol`, `media`/`dsp`, `trust`/`security`, `plan`/`observed_cost`, `runs` versions, the server routers and the MCP tools.
  - It is not loaded into every session, so its length is not context cost. It must still stay a map, a sentence per row; a row that needs a paragraph links to the docstring that holds it.
  - A test (`tests/test_architecture_map.py`) checks that every backticked `dw/...` / `dw_mcp/...` / `ui/...` path in it exists. A map that names a deleted module is worse than none, and this is the drift that would happen silently. It fails first, against a fixture map naming a missing file.
- **A docstring move never changes what agents are served.** Destinations are module docstrings and internal-function docstrings only. Never an MCP tool's docstring, a `register_command` task description, or the schema: those are the served surface, and `scripts/surface_snapshot.py` pins them byte for byte. The snapshot does not cover the served guides, so anything touching `docs/WORKFLOW_GUIDE.md` or another `docs/*.md` is a pointer added, never content moved and never a heading changed. Tasks 3-5 each end with the snapshot byte-identical.
- **Delete needs evidence an agent reads before editing:** a doc or a docstring. A rule stated only by a test, in its name or docstring, becomes a **map** row with that test in the "enforced by" column. Nobody opens `tests/` first, so the map has to send them there.
- **Docstring moves keep the code's line budget in view.** A module receiving a moved rule must stay under the size warning (1,000 lines). If one would cross it, the rule goes to the map row instead, and the triage says so.
- **`copilot-instructions.md` becomes a pointer of 10 lines or fewer:** read `CLAUDE.md` and `docs/ARCHITECTURE.md`. Its stale facts go, and it is not counted by the metric either way.
- **"Change both when one changes" pairs are resolved, not kept.** Where the root CLAUDE.md and `docs/WORKFLOW_GUIDE.md` both describe the reference conventions, the guide is the owner (agents read it over MCP); CLAUDE.md points at it.
- **Sub-files get the same triage, with no quota** (frame Decisions). `ui/CLAUDE.md`'s design-system rules are read only by UI work and stay where that work reads them, unless the triage finds them already said in code or a test.

### Review Focus (4c)

1. **A "delete" whose evidence does not say it.** The reviewer samples at least 15 delete rows, weighted to Critical Gotchas, and reads each cited file:line. A row whose evidence is weaker than the paragraph (it names the module but not the rule) is a finding, and the paragraph is re-triaged.
2. **A rule that vanished.** Every paragraph of the old files appears in the triage table (counted against `git show <base>:CLAUDE.md`), and every docstring and map row the table promises exists after Task 3.
3. **The map names things that exist,** by the map test, and its owners are the real ones: the reviewer spot-checks 10 rows against the code.
4. **The mechanical readers still pass:** `test_plugin_skills` without loosening, `arch_metrics --check` after re-baselining `claude_md_lines`, and the docs that point into CLAUDE.md (AGENT_LOOP, WORKFLOW_GUIDE, the onboarding skill's drill) still point at text that exists.
5. **A cold agent can still find its way.** After Task 4, one fresh subagent with no session context gets only the new root CLAUDE.md and is asked three questions a harness implementer meets:
   - where `asset:` references are resolved, and what confines them;
   - what to change to add a validation check;
   - why a seeded rerun generates nothing.

   It records the files it opened, in order. It passes only when the chain is CLAUDE.md, then `docs/ARCHITECTURE.md` (or a named docstring), then the module, with no grep before the map. A right answer reached by grep is still a finding against the pointers.

### Before Task 1: the hot zone goes live

Commit the 4c list below into `hot-zone.txt` on `develop` with this section, push, and branch `stabilization/phase-4c`. Task 1's table names the modules that will receive docstrings; those are added to the hot zone in Task 1's commit.

### Task 1: The triage

- [ ] **Step 1:** Build the table in two dispatches on the most capable model: one for the root's "Critical Gotchas" (395 lines), one for the rest of the root plus the three sub-files and `copilot-instructions.md`. It is judgement work, and its mistakes are what the diet ships; its reviewer is on the same model. Cheap evidence sources: the long descriptive test names and docstrings, and the `docs/*.md` list.
- [ ] **Step 2:** For every delete row, open the cited evidence and confirm it states the rule, not just the topic. Downgrade to docstring or map where it doesn't.
- [ ] **Step 3:** Summarise the table at its top: rows per verdict per file; the projected root line count; the modules that will receive docstrings, with their current line counts against 1,000; the seam-map rows to be written.
- [ ] **Step 4:** Commit the table. Put the hot-zone additions (the docstring destinations) on `develop` too, pushed as a docs-only commit as the 4a and 4b lists were. The harness reads `hot-zone.txt` from `develop`, not from this branch.
- [ ] **Step 5:** Send Don the table's summary (counts per verdict per file, the keep rows, the projected root count) with "say if a delete should stay". Don't wait for an answer, but make it visible before Task 4 rewrites the file he works in.

### Task 2: The seam map

- [ ] **Step 1:** The map test, against a fixture map naming one missing path; it fails, since the test file does not exist yet.
- [ ] **Step 2:** Write `docs/ARCHITECTURE.md` from the table's map rows plus the owners listed in Decisions (4c). Each row's module path is checked by the test, and each rule sentence by reading the code.
- [ ] **Step 3:** Run the map test on the real file. Commit.

### Task 3: The docstring moves

- [ ] **Step 1:** For each docstring row, put the rule in the module's docstring (or the owning function's), rewritten to the module's voice, not pasted. Keep each module under 1,000 lines; `arch_metrics` prints a warning otherwise.
- [ ] **Step 2:** The suite, ruff and `arch_metrics --check` pass, with no new warnings, and the surface snapshot is byte-identical. Commit.

### Task 4: The root CLAUDE.md

- [ ] **Step 1:** Rewrite it to the Decisions (4c) shape from the keep rows, <= 150 lines, on the most capable model. Match the existing voice: short and specific, every claim naming a file. Don reads and edits this file himself. Point `docs/WORKFLOW_GUIDE.md`'s "change both" note, `docs/AGENT_LOOP.md` and `.claude/skills/model-family-onboarding/references/cold-drill-example.md` at whatever they referred to.
- [ ] **Step 2:** `test_plugin_skills` and the map test pass, and the surface snapshot is byte-identical. List each kept gotcha as a candidate check: ASSESSMENT's principle is mechanical over prose, so these become 4d / stage C follow-ups, not the end state.
- [ ] **Step 3:** Review Focus 5, the cold-agent check. Record its answers in the report, then fix the map or the root file for any miss.
- [ ] **Step 4:** Commit.

### Task 5: The sub-files and copilot-instructions

- [ ] **Step 1:** Apply the triage to `ui/CLAUDE.md`, `dw_mcp/CLAUDE.md` and `dw/server/CLAUDE.md`. Turn `copilot-instructions.md` into a pointer.
- [ ] **Step 2:** The UI preflight is unaffected (docs only); spot-check that `ui/CLAUDE.md`'s kept design rules still name real files. The surface snapshot is byte-identical. Commit.

### Task 6: Stage 4c merge

- [ ] **Step 1:** Re-baseline. `claude_md_lines` falls from 957 to the new total; nothing else moves.
- [ ] **Step 2:** ROADMAP Phase 4 row: record the final line counts per file. ASSESSMENT's "CLAUDE.md files total 964 lines" finding gets its gate-4 number at the gate, not here.
- [ ] **Step 3:** Hot zone back to the standing entries. Merge with `--no-ff`, push, and check CI. Report to Don with the per-file counts and the cold-agent answers.
- [ ] **Step 4:** Detail stage 4d. It includes writing the harness stage C prompt, which goes to Don before anything else in 4d.

### Hot zone (4c)

```
CLAUDE.md
ui/CLAUDE.md
dw_mcp/CLAUDE.md
dw/server/CLAUDE.md
.github/copilot-instructions.md
docs/ARCHITECTURE.md
docs/WORKFLOW_GUIDE.md
docs/AGENT_LOOP.md
.claude/skills/model-family-onboarding/references/cold-drill-example.md
tests/test_architecture_map.py
scripts/arch_metrics.py
docs/stabilization/
```

The harness implementer often adds a CLAUDE.md line with a field fix. While 4c runs, a fix that needs one is labelled `stabilization` and handed to Don, and its note goes into the triage instead.

## Stage 4d: gate 4

The gate, in this order. The first step waits on Don.

### Task 1: Stage C in the harness (Don)

- [ ] **Step 1:** Don reviews [harness/stage-c-guardrails.md](harness/stage-c-guardrails.md), pastes it into a session in `/Users/don/testing/harnest`, and confirms it is committed and its tests pass. It covers:
  - the harness-side guardrails: new-module approval through `arch-approved` plus a matching baseline raise, the architecture review against `docs/ARCHITECTURE.md`, and the consolidation cadence;
  - which stage A/B gates outlive FREEZE.

  The dw-side pair, the CI ratchet (4b) and the seam map (4c), are already on `develop`.
- [ ] **Step 2:** Nothing below starts until Don confirms Step 1. `FREEZE`'s own text says the freeze lifts "at the Phase 4 gate"; this ordering is what that means.

### Task 2: Lift the freeze

- [ ] **Step 1:** On `develop`, delete `docs/stabilization/FREEZE` and push. `hot-zone.txt` keeps `scripts/arch_metrics.py` permanently, because the harness must not edit its own ruler, and drops `docs/stabilization/`. The file stays: stage C reads it for later refactors. Don confirms that the harness's next pass runs `features_pass` and that the unpark pass ran.

### Task 3: The gate report and tag

- [ ] **Step 1:** `scripts/arch_report.py` with a Gate 4 column into ROADMAP.md, "Gate 4". The section also records:
  - each stage's merge;
  - the release notes collected under `### 0.7.0`;
  - LCOM4 (`Workflow` now 1);
  - the CLAUDE.md numbers per file;
  - the candidate checks for the kept gotchas (from 4c Task 4's report, "Step 2: candidate checks");
  - the six seam-map rules marked "—" (nothing enforces them), as follow-ups.
- [ ] **Step 2:** Re-baseline (`--write`); the diff against 4c's is nothing, or only decreases.
- [ ] **Step 3:** Tag `stabilization-gate-4` on `develop`, and push the tag.

### Task 4: lem deploy and smoke

- [ ] **Step 1:** `scripts/deploy.sh develop`, then health and version.
- [ ] **Step 2:** Run the same smoke as gate 3:
  - `templates/ltx2/two-stage`, cold, plus the B2 cached rerun;
  - the inline `for_each` workflow over a `prompt:` and an `asset:`.

  It also runs one catalog template that uses `concat_videos` (4a changed its rule; `dialogue-short` or `assemble-and-score`). The `for_each` run's time is compared with gate 3's (16.0 s cold), and its memory (`memory_status` before and after) is recorded as the first baseline for the ruled-out leaf sharing (Decisions (4a)). Gate 3 recorded no memory.

### Task 5: Close Phase 4

- [ ] **Step 1:** ROADMAP Phase 4 row: done, with the tag. ASSESSMENT.md refreshed with the gate-4 numbers ("CLAUDE.md files total 964 lines" gets its answer). The Claude Doc's metrics table gets a Gate 4 column.
- [ ] **Step 2:** The release. Ask Don whether gate 4 ships 0.7.0 (frame Decisions). If yes: the develop-to-master PR, then `scripts/release.sh 0.7.0 --next 0.8.0-alpha.1`, then the notes pasted from RELEASING.md's `### 0.7.0`.
- [ ] **Step 3:** Memory and the ledger: the stabilization is complete. Save the standing rules that outlive it (the CLAUDE.md-only-shrinks rule, `arch-approved` for a ratchet rise, the seam map as the first stop) where later sessions read them.
