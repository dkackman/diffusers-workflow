# When a job last used each Hub model (#1445)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out session,
2026-11-19). Don filed the ask on 2026-11-18: mark each Hub model with the time a job
last used it, the way quantized-cache entries already carry `last_hit`, so the Models
page's "last used" answers "what can I delete?". Plan v3 is the plan comment on #1445,
and it still holds the full sweep. This file records what was built, what was cut and
what stays.

## The problem

The Models page showed `repo.last_accessed`: huggingface_hub's `scan_cache_dir`
reading the blobs' filesystem **atime** (`dw/hub_cache.py`). It was wrong both ways:

- **It missed real use.** A job served from the quantized cache never reads the Hub
  weights, so H3 on lem (run every day from `~/.cache/dw/quantized`) looked idle.
  `relatime` and `noatime` mounts hide use too.
- **It counted things that aren't use.** A copy, a backup or a hash check bumps it. The
  2026-11-18 stream of H3's snapshot from lem to mini-ai marked it "used" on lem.

## Verdict: build smaller (two stages, no GPU)

- Cut the issue's `get_server_info` Hub summary (Q3): it would be new MCP surface, and
  `list_models` already answers the question.
- Cheaper alternatives rejected: rolling `last_hit` up onto the source repo fixes only
  the quantized case; deriving use from job history costs every `GET` and history is
  pruned.

## Plan history

- **v1 (2026-11-18).** The advisor, ruling on Don's behalf, asked for changes: no H3
  run anywhere in the mini-ai acceptance, a non-H3 quantized entry or a unit test for
  the quantized-hit clause, and a small repo for delete/re-download.
- **v2.** Acceptance moved to SD 1.5 (`templates/text-to-image`) and
  `openai/whisper-base` (`templates/transcribe-audio`). The quantized-hit clause
  became a unit test (mini-ai holds only LTX-2.5 entries). The index gained a
  1000-repo cap. Approved by the advisor with three minor adjustments.
- **v3.** Folded those in, scope unchanged: `repos_named` runs after argument
  substitution; the cap keeps the current call's stamps; the restart check uses the
  managed restart, not `kill -9`.

**Rulings (advisor, all defaults):**
- Q1 [none]: the index is `get_settings_dir()/model-use.json`, outside the Hub cache.
- Q2 [default]: "last used" is the start time of the latest job that named the repo,
  whatever its outcome. A load hook was rejected: there is no single Hub load point,
  and resident pipelines and cached task models are reused without loading again.
- Q3 [scope]: the `get_server_info` summary is cut.
- Q4 [none]: local-path and `from_single_file` URL loads aren't recorded.

## What was built

### Stage A (#1461): record and serve `last_job_use` (server)

Merged to `develop` at `88cb46d9` (branch `feat/1445-A-last-job-use`: `1b9eefbd` code,
`05bcf28a` tests and docs).

- **Which repos a job names:** `dw/plan_downloads.py` `repos_named()`. It shares one
  walk (`_walk`) with `downloads_required`, so it reaches step and component
  `model_name`, LoRA repos, `load_components`, task weights, file-weight repos and
  sub-workflows. It keeps only valid Hub ids. `Admission.repos_named` computes it after
  substitution, so a repo supplied through `variable:` is recorded resolved.
- **Carried to the job:** `JobManager.submit` and `rerun` take `repos_named` from all
  three submit paths (the jobs routes, the library run route and the restart-batch
  rerun), onto `Job.repos_named`.
- **Stamped at run start:** the worker's `run_start` event reaches
  `JobManager._on_run_start`, which calls `model_use.record(repos, now)`. A refused
  submission, a `validate_workflow`, or a job cancelled while queued stamps nothing. A
  quantized-cache hit stamps its source repo with no extra code, since the job names it.
- **The index:** `dw/model_use.py`, `{"version": 1, "repos": {id: epoch}}`. Each write
  merges under a file lock, keeps the newer stamp per repo and replaces the file
  atomically. At most 1000 entries; the repos just stamped always stay. A missing,
  corrupt or unwritable file reads as empty and never fails a job or a listing.
- **Served:** every `GET /api/models` row has `last_job_use` (epoch seconds, or `null`);
  MCP `list_models` passes it through. `last_accessed` is unchanged. Deleting a cached
  repo forgets its entry.
- **Deviations:** `ModelRepo.last_job_use` is a required field (always present,
  `null` when unrecorded), so the generated TS type is `number | null`. `last_accessed`
  gained a field description naming it the atime fallback.
- **Tests:** `tests/test_repos_named.py`, `tests/test_model_use.py`,
  `tests/test_models_last_job_use.py`.
- **Docs:** `docs/SERVER.md`, `docs/MCP.md` (the two dates and when to fall back),
  and a "Model-use index" row in `docs/ARCHITECTURE.md`.

**Bounces (0).** The architecture review passed. The tester passed C-F1019 to C-F1024
on the first verify (mps on mini-ai, develop @ `88cb46d9`). Every stamp landed within
0.3 s of its job's `started_at`.

**Not exercised on mini-ai** (unit tests cover each):
- C-F1025, restart survival: no restart fell between two reads in that session. It
  should run on a later pass with a deploy between its reads.
- C-F1026, the LoRA repo: no cached LoRA has a base mini-ai runs at verification size.
- C-F1027, delete forgets: no MCP tool deletes a Hub repo.
- The quantized-cache hit: mini-ai's quantized cache holds only LTX-2.5.
- CUDA hasn't seen the stage.

### Stage B (#1462): the Models page shows it (UI)

Merged to `develop` at `95a632e5` (branch `feat/1445-B-models-last-job-use`).

- The "last used" cell shows `day(last_job_use)` plain when it is set. Otherwise it
  shows `day(last_accessed)` muted, suffixed " (disk)", with the tooltip "No dw job has
  used this repo since tracking began. Date is the files' last access time, which
  copies and backups also bump." Both null shows "—". The header has a tooltip.
- **Deviation:** the cell is its own component, `ui/src/lib/LastUsedCell.svelte`,
  because inlining pushed `ModelsPage.svelte` past the UI's 600-line ceiling.
- **Tests:** `ui/src/lib/pages/ModelsPage.test.ts`, four cases.

**Bounces (0).** The first verify checked the data over MCP but couldn't see the page,
so it went to Don for a look. The advisor handed it back once Don had approved the
tester's `ui-check` runner, and the tester verified it black-box (develop @
`d687fa1c`): dw-used rows plain, the rest muted with "(disk)", and a repo whose files
were touched today but no job used read `Nov 19 (disk)`. The tooltips aren't
observable through `ui-check` and rest on the vitest case. A Web UI regression case
was added.

Stage costs aren't recorded: the stage comments name no `usage:` figures. No GPU time
beyond an SD 1.5 text-to-image and a few Whisper-base transcriptions.

## Cut, and what brings each back

- **The `get_server_info` Hub summary** (Q3). Comes back if Don wants "what can I
  delete?" at a glance in an MCP session; a few lines over `model_use.read()`.
- **Local-path and URL loads** (Q4), e.g. mini-ai's `/Volumes/NVME/cache/h3-prequantized`.
  They have no Models-page row to show it on. Comes back if the page lists local
  model directories.
- **Following component redirects** (`_component_plan` through the cache index). A
  redirect target can read older than it is. Comes back if it shows on lem's H3 or
  Kandinsky rows.
- **Backfill from past jobs.** Repos used before this shipped show the "(disk)"
  fallback until their next job, as the issue intended.
- **Automatic eviction.** Out of scope from the start.

## Known limits

- The number means "a job that started named this repo". A step skipped by a
  condition still counts, which errs toward keeping.
- On a settings dir over NFS the file lock is unreliable; the worst case is a lost
  stamp, never a failed job.
