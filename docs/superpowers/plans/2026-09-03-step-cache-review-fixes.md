# Step-cache and download review fixes

**Spec:** the verified review findings below are the requirements. Each task lists
its findings verbatim; the "Fix" lines are binding unless they contradict a finding.
Base commit: 3cc6e50 (memory manager already reverted — do not reintroduce it).

## Global Constraints

- Follow CLAUDE.md: validate paths through `dw/security.py`; no `eval`/`exec`/`shell=True`;
  `dw_mcp/` must import nothing from `dw.*`; `black dw/ tests/` clean.
- TDD: write the failing test first, then the fix. Tests live in `tests/`; UI tests in `ui/`.
- Run the covering test files, then `venv/bin/pytest -q -p no:cacheprovider` (full suite,
  ~2 min) before the final commit of a task. UI tasks: `npm run check && npm run lint && npm test`
  from `ui/`.
- Commit with `git add <specific files>` (never `-A`); other agents may be committing
  disjoint files on this branch concurrently — retry once on `index.lock`.
- End commit messages with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Do not change public function signatures used by callers outside the files you touch
  unless a task says so.

---

## Task 1: Step-cache hit path runs the step's bookkeeping (dw/workflow.py)

Files: `dw/workflow.py`, `tests/test_workflow_step_cache.py`, `tests/test_step_cache.py`
(only if the import move requires it), `docs/REPL_COMMANDS.md` (one sentence on `reused`).

Findings:
- F1. A step-cache hit at `dw/workflow.py:455` skips `create_step_action`, which is the only
  caller of `get_context().touch_pipeline(cache_key)` (line ~598) and of
  `cached_pipeline.publish_shared_components(shared_components)` (line ~608). So after an
  all-hit run `dw/worker.py:222` evicts every loaded pipeline as "no longer in workflow",
  and a later step whose `reused_components` must be resolved by a fresh `Pipeline.load()`
  raises "Cannot reuse component ... Shared so far: nothing" (LTX2TwoStage: stage one hits,
  stage two misses). `release_pipeline` on a hit step is a documented no-op ("Known gap",
  line ~507) and `_pipeline_reference_targets` (line 39) is a third workaround.
- F5. A hit republishes an earlier job's `saved_files` into this job's manifest and
  `step_end` event verbatim; `JobManager.job_for_file` (`dw/server/jobs.py:154`) then
  attributes the file to a job that never wrote it.
- F4a. When the workflow sets no `seed`, `default_seed = torch.Generator().seed()` is drawn
  fresh per run (line ~351), so no step can hit, yet every cacheable step still pays
  `copy.deepcopy(step_data)` (realized PIL images / frame lists) and `step_cache.put`
  pins its Result.
- C1. `dw/step_cache.py` imports `referenced_result_names` from `dw/workflow.py`, which is
  worked around with a function-scope `from .step_cache import step_cache` inside `run()`
  plus a regression test that `importlib.reload`s both modules.

Fix:
1. Always call `create_step_action(...)` for a cacheable step, hit or miss. On a hit skip
   only `step.run(...)` and `result.save(...)`. Delete `_pipeline_reference_targets` and
   the `pipeline_reference_targets` exclusion from `is_cacheable`, and delete the
   "Known gap" comment block above the `release_pipeline` pop — `_pipeline_keys_by_step`
   is now populated for hit steps too. `is_cacheable` becomes
   `"workflow" not in step_data and cache_enabled_this_run`.
2. `cache_enabled_this_run` is False when the workflow set no `seed` (i.e. `default_seed`
   was drawn randomly). When False, no deepcopy, no `get`, no `put`.
3. Manifest entry and `step_end` event for a hit carry `"reused": True`
   (miss entries carry nothing new — do not add `reused: False`). Update
   `JobManager.job_for_file` in `dw/server/jobs.py` to skip manifest entries with
   `reused` True so attribution stays with the job that wrote the file. Add a test.
4. Move `referenced_result_names` from `dw/workflow.py` into `dw/step_cache.py`;
   `dw/workflow.py` does `from .step_cache import step_cache, referenced_result_names`
   at module top. Delete the function-scope import, its comment, and the
   `test_import_order_workflow_then_step_cache_does_not_cycle` test. Fix any other
   importer (`grep -rn referenced_result_names`).

Tests (write first, each must fail before the fix):
- On a hit, `touch_pipeline` is called for the step's cache key (assert via
  `get_context().touched_pipelines` after `run`).
- Two-step fixture where step A declares `shared_components` and step B
  `reused_components`; run twice with fixed seed, then a third time with only B's prompt
  changed and B's pipeline removed from the `pipelines` dict: B loads without raising.
- `release_pipeline: true` on a step that hits pops its key from `pipelines`.
- Random seed (no `seed` in workflow): `step_cache.put` never called; `copy.deepcopy`
  never called on step_data (patch and assert).
- Hit manifest entry has `reused: True`; `job_for_file` prefers the writing job.

## Task 2: Step-cache identity, staleness and retention (dw/step_cache.py)

Files: `dw/step_cache.py`, `dw/workflow.py` (call sites only), `tests/test_step_cache.py`,
`tests/test_workflow_step_cache.py`, `tests/conftest.py`, `tests/test_server_downloads.py`.
Depends on Task 1.

Findings:
- F2. Entries are keyed by bare step name (`put`, line ~116) while saved files are named
  `f"{workflow_id}-{step.name}.{i}"`. A different workflow (or the same one after an `id`
  rename, or a sub-workflow sharing a step name with its parent) hits and republishes the
  other workflow's file paths, writing none of its own.
- F3. `_is_hit` (line ~99) checks only that each upstream step was hit *this run*, not that
  this entry was computed from the upstream entry now in the cache. A→B fixed seed: change
  A, run, cancel after A's `put` but before B's; next unchanged run serves B computed from
  the old A.
- F4b. `put` stores the live `Result`, whose `result_list` holds decoded frames or
  `output_type: latent` device tensors, so `release_unreferenced_results` removes a dict
  entry while the object stays alive for the life of the cache (bounded by count, not bytes).
- C2. `_is_hit` duplicates the "reference equals a name or extends it with `.prop`" rule that
  `release_unreferenced_results` and `dw/previous_results.py:135` also implement.
- C3. `_saved_files_exist` uses `getattr(result, "saved_files", None) or []`, which turns a
  result-like object without `saved_files` into a valid hit with no file check.
  `output_dir=None` defaults on `get`/`put` exist only so tests can omit it.
- C4. The autouse `_clear_step_cache_after_test` fixture lives only in
  `tests/test_workflow_step_cache.py`; any other test running a seeded workflow twice can
  now hit. `tests/test_server_downloads.py:18` copies the `server` fixture from
  `tests/test_server.py:129`.

Fix:
1. Key entries by `(workflow_id, step_name)`. `get`/`put` take `workflow_id` as a required
   positional; `output_dir` becomes required too. Update the two call sites in
   `dw/workflow.py`.
2. Each entry gets a monotonically increasing `generation` (module-level counter incremented
   on every `put`). `put` records `upstream_generations: {upstream_name: generation}` for
   every upstream the step references, read from the cache at put time. `get` hits only if
   every referenced upstream was hit this run AND its current entry's `generation` equals
   the recorded one. `hits_this_run` may stay a set of names.
3. Retention: `put` takes `retain_result: bool`. The caller passes True only when a later
   step in this run references the step (`referenced_result_names(steps[i+1:])`) or it is
   the last step (the workflow's return value). When False the entry stores a shallow
   `Result` copy with `result_list = []` and the same `saved_files`/definition. `get` takes
   `needs_result: bool` (same predicate for the current run) and misses when
   `needs_result` and the entry was not retained.
4. Extract one `reference_resolves_to(ref, name)` helper in `dw/step_cache.py` (no torch
   import) and use it from `_is_hit`, `release_unreferenced_results`, and
   `dw/previous_results.py:135` if its longest-prefix loop can use it without behavior
   change — otherwise leave `previous_results.py` alone and say so in the report.
5. `_saved_files_exist` reads `result.saved_files` directly.
6. Move the autouse step-cache clear into `tests/conftest.py` next to
   `_clear_task_model_cache`. Delete the copy in `test_workflow_step_cache.py`. Make
   `tests/test_server_downloads.py` import the `server` fixture from `tests/test_server.py`
   (or move it to conftest) instead of redeclaring it; if the download tests need the
   `Basic.json` seed file that the shared fixture writes, that is fine.

Tests (write first): two workflows with identical seeded step `main` and different ids do
not share an entry; id rename misses; the F3 cancel scenario misses on the third run;
unreferenced non-final step is stored with empty `result_list` and a later run that adds a
downstream reference misses; `saved_files` missing attribute raises rather than hitting.

## Task 3: Chain segments never overwrite; cached segment results stay valid

Files: `dw/pipeline_processors/chain.py`, `dw/step_cache.py` (small hook only if needed),
`tests/test_chain*.py`. Depends on Task 2.

Findings:
- F6. `SegmentSpill.write` (`chain.py:243`) builds
  `{base_name}.segment-{NNN}.mp4` through `validate_output_path` only, not
  `dw/result.py`'s `output_file_path`, so a rerun's segment-000 overwrites the previous
  run's salvageable segments — the one output the collision fix (`c1c84a2`) missed. The
  per-wrapper counter restarts at 0 each run.
- F6b. A cached `Result` whose `result_list` holds a `SegmentedFrames` can point at files
  `SegmentedFrames.cleanup()` (line 185) already removed; a downstream miss iterating it
  fails opening the segment files.

Fix:
1. Route the segment path through `output_file_path(self.output_dir, file_name)` from
   `dw/result.py` (it validates and dedupes). Keep the `-N` dedupe convention.
2. For F6b, pick the smallest correct rule and state it in the report: either
   (a) `Result` exposes `retainable` (False when any `result_list` item is a
   `SegmentedFrames` whose files have been cleaned up) and `step_cache.put` treats
   `retain_result=True` with `retainable=False` as store-without-result_list; or (b)
   `SegmentedFrames.cleanup()` is made idempotent and deferred until the Result is
   released. Prefer (a) unless it needs `dw/result.py` to import from chain.

Tests: two `SegmentSpill.write` calls with the same base name in one dir produce distinct
files; a rerun does not clobber; cached chain result after cleanup is not served to a
downstream miss.

## Task 4: MCP download_output — streaming, overwrite guard, honest docs

Files: `dw_mcp/media.py`, `dw_mcp/server.py` (tool description/param), `tests/test_mcp_media.py`,
`docs/MCP.md`, `docs/SECURITY.md`. No dependency.

Findings:
- F7. `download_output` (`dw_mcp/media.py:154-163`) expands, `abspath`s and writes any
  client-supplied `destination`, creating parents and silently overwriting. `docs/SECURITY.md:126`
  ("`dw_mcp/` introduces no new file access ... validated there") and `docs/MCP.md:301`
  ("purely a client of the same validated endpoints") are now false.
- F7b. It buffers the whole body via `client.get_bytes`; the tool exists for the large
  videos `get_output_image` cannot return. `DwClient._stream_request` and
  `get_bytes_if` (`dw_mcp/client.py:105-134`) show the streaming pattern.

Fix:
1. Stream: use `client._stream_request("GET", path)` (or add a small public
   `client.stream_to_file(path, destination)` next to `get_bytes_if`) and write
   `response.iter_bytes()` in chunks, counting bytes.
2. Add `overwrite: bool = False`. If the destination exists and `overwrite` is False,
   return an error result (do not raise through the MCP layer as a crash) naming the path.
   Reject a `destination` that contains a `..` path segment after `expanduser`
   (`".." in pathlib.PurePath(destination).parts`); absolute paths and `~` remain allowed —
   this is a local client acting for the user.
3. Docs: replace the SECURITY.md:126 and MCP.md:301 claims with one paragraph each stating
   the exception: `download_output` is the one dw_mcp tool that writes a local file, where
   it may write, the overwrite guard, and that `..` is refused.
4. Update the tool's docstring/description in `dw_mcp/server.py` for the new parameter.

Tests: existing file + overwrite False → error, no write; overwrite True → replaced;
`..` segment → error; body written in chunks (mock a streaming response with ≥2 chunks).

## Task 5: Query-token auth matched per route, not by suffix

Files: `dw/server/app.py`, `tests/test_server.py`. No dependency.

Finding:
- F8. `QUERY_TOKEN_ROUTE_SUFFIXES = ("/events", "/thumbnail", "/download")` with
  `path.endswith(...)` (`app.py:325`, `:448`). `GET /api/prompts/{name:path}` (line 843),
  `/api/workflows/{name:path}` (768) and `/api/classes/{name:path}` (641) match a resource
  literally named `download`/`events`/`thumbnail`, so a header-only full-read route accepts
  `?token=`. The `request.method == "GET"` guard also excludes HEAD, which Starlette adds
  to every `@app.get` route.

Fix: mark the five query-token routes (jobs events, gallery thumbnail, the three
downloads) with an attribute on the endpoint function (e.g. a `query_token_ok` decorator
setting `fn.query_token_ok = True`). In the middleware, resolve the matched route by
iterating `request.app.router.routes` and calling `route.matches(request.scope)`
(Starlette `Match.FULL`), then accept the query token only if the matched endpoint carries
the marker and the method is GET or HEAD. Remove `QUERY_TOKEN_ROUTE_SUFFIXES`. Update the
docs/SERVER.md sentence only if its wording becomes wrong.

Tests: `GET /api/prompts/download?token=<valid>` → 401 (save a prompt named `download`
first); `GET /api/gallery/x.png/thumbnail?token=` still 200; HEAD on that route with a
token → not 401; `POST /api/models/download?token=` still 401.

## Task 6: One DownloadLink component in the UI

Files: `ui/src/lib/DownloadLink.svelte` (new), `ui/src/lib/pages/GalleryPage.svelte:177`,
`ui/src/lib/pages/WorkflowPage.svelte:102`, `ui/src/lib/pages/PromptEditorPage.svelte:502`,
plus a vitest next to `CopyButton`'s if one exists. No dependency.

Finding:
- C5. The block `<a class="quiet icon" href=... download aria-label="Download"
  title="Download"><Download size={16} /></a>` is pasted into three pages, using
  `size={16}` where the neighbouring icon buttons use `size={14}`.

Fix: create `DownloadLink.svelte` modeled on `ui/src/lib/CopyButton.svelte` (same
class/aria/title conventions), props `href` and optional `label` (default "Download"),
icon `size={14}`. Replace the three copies. `npm run check && npm run lint && npm test`
clean; `npm run build` succeeds.
