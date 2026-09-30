> Survey taken at `9e25c49a` (2026-09-30) for the Phase 3 plan. Line numbers are that commit's: re-verify before relying on them.

# Survey: dw/server/app.py, dw_mcp/server.py, dw/serve.py, dw/server/jobs.py

Worktree: /Users/don/src/dkackman/dw-stabilization (branch stabilization/phase-3-plan). All line numbers are in this tree. `create_app` = app.py:741-4519 (ends at `return app`, 4519). Closure-locals analysis was done with `ast` (free-variable walk), so the "uses" columns are machine-derived.

---------------------------------------------------------------------------
## 1. Routes in create_app (all `@app.<verb>`, no router, no version prefix)

Span = decorator line to last line of handler. Total: 71 decorated things = 4 middlewares + 66 routes + 1 manual `Route` pair for /mcp (+ the SPA mount).

### jobs (1158-1532, 375 lines of routes; + admission helpers 1066-1156 = ~467 total)
| method | path | handler | span |
|---|---|---|---|
| POST | /api/jobs (201) | submit_job | 1158-1232 |
| GET | /api/jobs | list_jobs | 1234-1276 |
| GET | /api/jobs/{job_id} | get_job | 1278-1284 |
| GET | /api/jobs/{job_id}/workflow | get_job_workflow | 1286-1311 |
| POST | /api/jobs/{job_id}/rerun (201) | rerun_job (+ RerunRequest 1313) | 1325-1374 |
| POST | /api/jobs/{job_id}/export (201) | export_job_route | 1376-1427 |
| POST | /api/jobs/{job_id}/move | move_job (+ MoveRequest 1429) | 1432-1446 |
| POST | /api/jobs/{job_id}/cancel | cancel_job | 1448-1453 |
| GET | /api/jobs/{job_id}/events (SSE, async, `@query_token_ok`) | job_events | 1455-1491 |
| GET | /api/jobs/{job_id}/event-log | job_event_log | 1493-1532 |

### introspection / catalog-of-engine (1536-1626, 91 lines)
GET /api/pipelines (pipelines 1536), /api/pipelines/{name} (pipeline_description 1541), /api/tasks (1553), /api/tasks/{command} (get_task 1558), /api/classes (1567), /api/classes/{name:path} (class_description 1576), /api/schema (workflow_schema 1587), /api/guides (list_guides 1607), /api/guides/{name} (get_guide 1615). All use only `app`(decorator) - NO closure state at all. Trivially movable.

### validate (1628-1878, 251 lines)
POST /api/validate -> validate_workflow 1729-1861 (133 lines). Private helpers in closure: `_probe_command_for` 1628-1645, `_validation_plan` 1647-1727 (81), `_host_memory_warnings` 1863-1878.

### workspaces (1880-1986, ~107 lines; + lookup helpers 989-1033)
GET /api/workspaces (list_workspaces 1885-1913), POST /api/workspaces 201 (add_workspace 1915-1928), DELETE /api/workspaces/{name} (remove_workspace 1930-1986). WorkspaceRequest 1882.

### workflows (1992-2249, ~258 lines)
GET /api/workflows (list_workflows 1994-2046), PUT /api/workflows/{name:path} (save_workflow 2048-2102), DELETE same (delete_workflow 2104-2132), GET .../download (`@query_token_ok`, download_workflow 2134-2141), GET .../variables (get_workflow_variables 2145-2214; VARIABLE_VALUE_PREVIEW=200 at 1992), GET /api/workflows/{name:path} (get_workflow 2232-2249). Helper `_observed_for_name` 2216-2230 (also used by `_validation_plan`).
ORDER MATTERS: `{name:path}/download` and `/variables` are registered before the greedy `{name:path}` GET (2232) - preserve registration order when splitting.

### prompts (2253-2420, 168 lines) + enhancers (2424-2479, 56 lines)
GET /api/prompt-schema (2256), GET /api/prompts (list_prompts 2299-2353), PUT /api/prompts/{name:path} (save_prompt 2355-2374), DELETE (2376-2391), GET .../download (`@query_token_ok` 2393-2400), GET /api/prompts/{name:path} (get_prompt 2402-2420; again after /download), GET /api/enhancers (2436), POST /api/enhance 201 (enhance 2440-2479). Helpers: `referenceable` 2262, `_prompt_roots` 2269, `_find_prompt` 2281.

### gallery / outputs (2489-3665, ~1,177 lines: the biggest group; alone > 1,000)
Shared resolution helpers (not routes) 2489-2745 (~257): MEDIA_KINDS/RAW_MEDIA_EXTENSIONS/GALLERY_THUMBNAIL_MAX_DIM consts 2489-2512, `_strip_output_prefix` 2514, `_output_file` 2533, `_asset_in` 2566, `_asset_file` 2599, `_job_provenance` 2613, `_static_files_for` 2638, `_common_assets` 2648, `_asset_roots` 2657, `_resolution_roots` 2672, `_asset_roots_for_job` 2692, `_served_url` 2721, `_absolute_served_url` 2733. (The asset helpers are shared with the assets group and static-serving group.)
Listing helpers 2746-2895: `_iter_gallery_files` 2746, `_gallery_entries` 2794, `_iter_orphan_runs` 2858, `_orphan_entries` 2890.
Routes: GET /api/gallery (gallery 2897-2980); GET /api/gallery/{name:path}/metadata (2982-3063), /assess (3065-3103), /audio (3105-3194), /frames (gallery_frames 3202-3367 = **166 lines, the only handler >150**; consts FRAME_MIN_DIMENSION 3196, MAX_FRAME_MOMENTS 3200; helper `_encoded_tile` 3369), /thumbnail (`@query_token_ok` 3393-3447), /download (`@query_token_ok` 3449-3454); POST /api/gallery/archive (archive_outputs 3530-3546; ArchiveRequest 3460, MAX_ARCHIVE_FILES 3458, `_zip_download` 3463-3516, `_archive_selection` 3518); DELETE /api/gallery/{name:path} (delete_output 3635-3665; helpers 3550-3633: RUN_SIDECARS, `_remove_empty_identity_folders`, `_prune_empty_run_directory` 3569-3615, `_run_directory` 3617).
Order: POST /api/gallery/archive vs DELETE `{name:path}` differ by method so no clash; all `{name:path}/<suffix>` routes are before the bare `DELETE {name:path}`.

### uploads + assets (3669-4107, ~439 lines)
POST /api/uploads 201 (async upload_media 3678-3793; UPLOADS_SUBDIR, ALLOWED_UPLOAD_EXTENSIONS, MAX_UPLOAD_BYTES 3669-3676), GET /api/assets (list_assets 3812-3903; helper `_asset_origin` 3795), POST /api/assets/keep 201 (keep_output_as_asset 3924-4029; KeepRequest 3905), POST /api/assets/archive (archive_assets 4031-4055), DELETE /api/assets/{name:path} (delete_asset 4057-4107).

### models / downloads (4111-4165, 55 lines) and system/diffusers (4169-4230, 62 lines)
GET /api/models (4111), POST /api/models/download 202 (4121), GET /api/models/downloads (4129), POST /api/models/downloads/{download_id}/cancel (4133), DELETE /api/models?repo= (delete_cached_model 4142-4165). GET /api/system/diffusers (4171), POST /api/system/diffusers/update 202 (update_diffusers 4190-4230; UpdateDiffusersRequest 4177). `downloads = download_manager or DownloadManager()` at 4116; `updater = diffusers_updater or DiffusersUpdater()` at 4169.

### memory (4234-4263, 30 lines)
GET /api/memory (4234), POST /api/memory/clear (4241).

### health / server info (4265-4354, 90 lines)
GET /api/health (4265-4291; reports `__version__`, hostname, device, mcp flag), GET /api/server (server_info 4293-4354; uses host, port, token, wildcard_bind, local_addresses).

### mcp mount (4360-4380, 21 lines)
Not decorators: `app.router.routes.append(Route("/mcp", ...))` and `Route("/mcp/{sub_path:path}", ...)` only if `mcp_asgi`. Must be before the SPA mount; `require_bearer_token` gates exactly these two spellings.

### static output/inputs/exports (4382-4511, ~130 lines; NOT under /api, ungated)
`_sandbox_active_content` 4393 (CSP sandbox for ACTIVE_DOCUMENT_TYPES), GET /outputs/{name:path} (async output_file 4411-4441), GET /inputs/{name:path} (async input_file 4443-4467), `_export_download_name` 4469, GET /exports/{job_id}.zip (export_zip 4492-4511).

### ui (4514-4519)
`app.mount("/", StaticFiles(directory=resolved_ui, html=True), name="ui")` - must stay LAST (catch-all).

### Non-route setup in create_app
- 741-870: signature (15 params), manager construction + workflow_dir agreement check (766-778), lazy `build_mcp_app` (780-786), lifespan (788-802), FastAPI(...) (804-808), app.state population (810-865), allowed_hosts/wildcard_bind (867-870). ~130 lines.
- 872-987: 4 `@app.middleware("http")` (reject_foreign_origins 872-911, reject_foreign_hosts 923-935, require_bearer_token 937-975, browser_headers 982-987). Registration order defines middleware stack order (Starlette: last added = outermost). Preserve.

Approx. line budget by target module (for sizing): gallery/outputs+static resolution ~1,180 (must be split: e.g. output_files service ~260, gallery listing ~150, media routes (metadata/assess/audio/frames/thumbnail) ~470, archive/delete ~220); assets+uploads ~440; jobs ~470; validate ~250; workflows ~260; prompts+enhance ~230; workspaces ~110 + lookup ~90; models/system/memory/health ~240; static+exports ~130; middleware+app factory ~250.

---------------------------------------------------------------------------
## 2. Closure state (what handlers close over)

Everything routes read is either (a) `app.state.*` (already a shared-state object), (b) a create_app parameter/local, or (c) a closure helper. Machine-derived usage:

### Parameters / locals
| name | defined | used by |
|---|---|---|
| `manager` (JobManager) | 766 | lifespan; jobs group (all 10 routes + `_admit` indirect via submit/rerun); validate (`_validation_plan`); workspaces (remove_workspace); workflows (delete_workflow); enhance; models (delete_cached_model; update_diffusers); memory; health; gallery (`_output_file`, `_job_provenance`, `_asset_roots_for_job`) ; export (`_asset_roots_for_job`). Also `app.state.job_manager` (38 test refs). |
| `downloads` (DownloadManager) | 4116 | models group, update_diffusers (refuses while download active), delete_cached_model |
| `updater` (DiffusersUpdater) | 4169 | system/diffusers routes only |
| `token` | param | require_bearer_token, export_job_route (builds download URL), server_info, (mcp build) |
| `host`, `port`, `wildcard_bind`, `allowed_hosts` | param / 867-870 | middlewares (allowed_hosts, wildcard_bind), server_info (host, port, wildcard_bind) |
| `examples_dirs` | param | `_sources_for` only (and app.state init) |
| `mcp_asgi`, `mcp_server`, `mcp_client` | 780-786 | lifespan; mcp Route registration 4360; `app.state.mcp_mounted` |
| `workflow_dir`, `prompt_dir`, `asset_dir`, `workspace`, `output_dir`, `ui_dir`, `log_level` | params | consumed at construction to fill `app.state` / JobManager / `resolved_ui`; handlers go through app.state |
| `ceiling_indexes` (dict cache) | 1037 | only `_ceiling_index` (keyed by ws.name, validated by path+mtime signature) |
| `_example_libraries` | 826 | only fills `app.state.example_prompt_dirs/example_asset_dirs` |
| `resolved_ui` | 4515 | UI mount |

### app.state members written at 810-865 (and what reads them)
job_manager (jobs/many), observed_costs (workflows list/variables, `_observed_for_name`; 3 refs), workflow_dir (2), workflow_sources (1), prompt_dir (`_prompt_roots`, prompts, 7 test refs), example_prompt_dirs (`_prompt_roots`), example_asset_dirs (`_asset_roots`, `_asset_roots_for_job`), asset_dir, workspace, workspace_root (`_workspace_root`), default_workspace (`_workspace_for`), mcp_mounted (`health`), static_files_by_root (`_static_files_for`), media_kinds / raw_media_extensions (2508-2509; tests only - set "for tests" so they need not reach into the closure; `tests/test_server_downloads.py:241`).
Every handler receives `app` via closure (lexical) - a move to routers must switch to `request.app.state` / `Depends`.

### Closure helper functions and who uses them (free-variable derived)
Workspace lookup (989-1033):
- `_workspace_root` (991) <- `_workspace_for`, add_workspace, remove_workspace
- `_workspace_for` (1002) <- `selected_workspace`, submit_job, rerun_job, validate_workflow
- `selected_workspace` (1024, the FastAPI `Depends`) <- ~35 routes: jobs(submit, export), validate, workflows(list/save/delete/download/variables/get), enhance, every gallery route (gallery, metadata, assess, audio, frames, thumbnail, download, archive, delete), uploads, assets(list, keep, archive, delete), server_info, /outputs, /inputs, /exports
- `_sources_for` (1030) <- `_ceiling_index`, submit_job, validate_workflow, workflows(list, save, delete, download, variables, get). (Closes over `examples_dirs`.)
- `_ceiling_index` (1039) <- `_admit` only
Job admission (1067-1156): `_acknowledgement_form`, `_bound_plan_for`, `_check_bound_acknowledgement` <- submit_job, rerun_job; `_admit` (1146; closes `_ceiling_index`, `_prompt_roots`, `_resolution_roots`) <- submit_job, rerun_job, validate_workflow, enhance. NOTE `_admit` reaches into prompts group (`_prompt_roots`) and assets group (`_resolution_roots`) - cross-group coupling.
Validate: `_validation_plan` <- validate_workflow; `_probe_command_for` <- `_validation_plan`; `_host_memory_warnings` <- validate_workflow; `_observed_for_name` (2216, in workflows group) <- `_validation_plan`, get_workflow_variables -> validate depends on workflows helper.
Prompts: `_prompt_roots` <- `_admit`, list_prompts, `_find_prompt`; `_find_prompt` <- delete/download/get prompt; `referenceable` <- list_prompts.
Output/asset resolution (hub of the gallery group):
- `_strip_output_prefix` <- `_output_file`, gallery_metadata/assess/audio/frames, archive_outputs, delete_output, keep_output_as_asset, /outputs
- `_output_file` (closes `manager`, `_strip_output_prefix`) <- gallery_metadata/assess/audio/frames/thumbnail/download, archive_outputs, delete_output, keep_output_as_asset
- `_asset_in` <- `_asset_file`, archive_assets; `_asset_file` (closes `_resolution_roots`) <- metadata/assess/audio/frames, /outputs
- `_job_provenance` (closes manager) <- gallery_metadata, keep_output_as_asset
- `_static_files_for` (closes app.state.static_files_by_root) <- /outputs, /inputs
- `_common_assets` <- `_asset_roots`, `_asset_roots_for_job`, `_asset_origin`, upload_media, keep_output_as_asset
- `_asset_roots` <- `_resolution_roots`, `_asset_roots_for_job`, list_assets, delete_asset, /inputs
- `_resolution_roots` <- `_admit`, `_asset_file`, archive_assets
- `_asset_roots_for_job` <- export_job_route
- `_served_url`, `_absolute_served_url` <- `_gallery_entries`, list_assets, upload_media, export_job_route
- `_iter_gallery_files` (closes MEDIA_KINDS) <- `_gallery_entries`, list_assets
- `_gallery_entries` <- gallery; `_orphan_entries` <- gallery
- `_zip_download` (closes MEDIA_KINDS, RAW_MEDIA_EXTENSIONS) <- `_archive_selection` (-> archive_outputs, archive_assets), export_zip
- `_prune_empty_run_directory`, `_remove_empty_identity_folders`, `_run_directory` <- delete_output
- `_sandbox_active_content` <- /outputs, /inputs; `_export_download_name` <- export_zip
Constants closed over: MEDIA_KINDS (2489; also in `_iter_gallery_files`, `_zip_download`, metadata, assess, audio, frames, thumbnail), RAW_MEDIA_EXTENSIONS, GALLERY_THUMBNAIL_MAX_DIM, FRAME_MIN_DIMENSION, MAX_FRAME_MOMENTS, MAX_ARCHIVE_FILES, RUN_SIDECARS, UPLOADS_SUBDIR, ALLOWED_UPLOAD_EXTENSIONS, MAX_UPLOAD_BYTES, VARIABLE_VALUE_PREVIEW. None of these need to be closures - all are pure constants and can be module level.
Pydantic models defined inside the closure (should be module-level; FastAPI OpenAPI naming unaffected): RerunRequest 1313, MoveRequest 1429, WorkspaceRequest 1882, PromptRequest 2253, EnhanceRequest 2424, ArchiveRequest 3460, KeepRequest 3905, DownloadRequest 4118, UpdateDiffusersRequest 4177. (JobRequest 198, AcknowledgedCost 175 are already module level.)

Dependency-object implication: the real state is small: `manager`, `downloads`, `updater`, `token`, `host/port/wildcard_bind/allowed_hosts`, `examples_dirs`, `ceiling_indexes`, plus what is already on `app.state`. Nearly all the "helpers" are pure functions of (`ws`, `app.state`, `manager`) - they can become module-level service functions taking an explicit `AppContext`/`request.app.state`. `selected_workspace` must become a dependency reading `request.app.state` (today it closes over `app`). Module-level caches needing care: `_workflow_detail_cache` (226), `_prompt_detail_cache` (553) are process-global (shared across app instances - tests build many apps).

---------------------------------------------------------------------------
## 3. Module-level code in app.py outside create_app (lines 1-740; nothing after 4521)

Imports 8-167 (~160 lines of imports incl. `settings`, `guides`). Then (name, lines, callers):
- `logger` 169 - everywhere in app.py.
- `SSE_POLL_SECONDS` 172 - job_events (1485). Documented twins: `dw/run.py:32`, `dw_mcp/diagnose.py:17` (comments, not imports).
- `AcknowledgedCost` 175 / `ACKNOWLEDGED_COST_FIELD` 190 / `JobRequest` 198-216 - models for submit_job (1159), validate_workflow (1731), save_workflow (2050), rerun (1321-1322).
- `RUN_BOOKKEEPING_FILES` 223 - `_iter_orphan_runs` (2860, 2877).
- `_workflow_detail_cache` 226, `_prune_missing` 229-243 (-> `workflow_details` 423, `prompt_details` 590; **tests/test_server.py:6039,6056 call `app_module._prune_missing` and patch `app_module.os.path.exists`**).
- `collect_prompt_references` 246-259 (-> workflow_details 381; imported by tests/test_prompt_references.py:16).
- `_catalog_name_from_root` 262, `catalog_name_for` 277-286 (-> submit_job 1216, `_validation_plan` 1764).
- `attach_observed` 289-324 (-> list_workflows 2017; imported by tests/test_catalog_structure.py:16, tests/test_observed_cost.py:474).
- `workflow_details` 327-446 (120 lines; -> attach_observed, list_workflows 2018; imported by tests/test_catalog_structure.py:16; tests/test_catalog_shape.py touches it by name).
- `_write_bytes` 449 (-> upload_media 3771).
- `_unknown_workflow_detail` 454, `resolve_readable_workflow` 469 (-> delete/download/variables/get workflow), `resolve_writable_workflow` 484 (-> save_workflow 2065), `resolve_workflow_reference` 503-549 (-> submit_job 1165, validate 1758; `dw/run.py:38` only mentions its 400 message text; tests/test_cli.py:188 mentions it in a comment).
- `_prompt_detail_cache` 553, `prompt_details` 556-591 (-> list_prompts 2315/2327), `_matching_prompts` 594 (-> list_prompts 2321), `resolve_prompt_name` 621-640 (-> `_find_prompt`, save_prompt 2368).
- `default_ui_dir` 643-654 (-> create_app 4515; **dw/serve.py:248 imports it**; dw/server/guides.py only mentions it in a docstring).
- `_historical_log_note` 657-674 (-> job_event_log 1517).
- `LOOPBACK_HOSTS` 680, `WILDCARD_HOSTS` 683 (-> create_app 867-868; **dw/serve.py:154 imports LOOPBACK_HOSTS; dw/server/mcp_mount.py:25 imports both; dw_mcp/client.py:14 keeps a documented duplicate copy**).
- `MCP_PATH` 687 (-> server_info 4334).
- `ACTIVE_DOCUMENT_TYPES` 691-699 (-> `_sandbox_active_content` 4407).
- `query_token_ok` 702-710 (decorator; used at 1456, 2135, 2394, 3394, 3450; also read by `require_bearer_token` via `_matched_route`; tests/test_server.py references `query_token_ok` by name - check when moving).
- `_matched_route` 713-738 (-> require_bearer_token 960, 964). It walks `request.app.router.routes` for `Route` instances - still works with APIRouter routes (APIRoute subclasses starlette Route; include_router flattens into app.router.routes), but the `endpoint.query_token_ok` attribute on the endpoint function must survive (it will, decorators return the same fn).
Suggested homes: catalog listing (workflow_details, attach_observed, catalog_name_for, resolve_*_workflow, prompt_details, _matching_prompts) ~330 lines -> `dw/server/catalog.py` or `workflow_catalog.py`/`prompt_catalog.py`; request models (AcknowledgedCost, JobRequest) -> `schemas.py`; security-ish (query_token_ok, _matched_route, hosts constants, ACTIVE_DOCUMENT_TYPES) -> `dw/server/http_security.py` (and then mcp_mount/serve import the constants from there, which dissolves the cycle).

---------------------------------------------------------------------------
## 4. dw.server.app <-> dw.server.mcp_mount cycle

Both edges are lazy (function-local), so there is no import-time cycle, only a design cycle:
- app.py:780-786, inside `create_app` (only when `mcp=True`): `from .mcp_mount import build_mcp_app`.
- mcp_mount.py:25, inside `client_base_url()`: `from .app import LOOPBACK_HOSTS, WILDCARD_HOSTS` (comment at 22-24 explains it). Used once, to choose loopback vs bind host for the tool->server loopback URL.
- mcp_mount.py:75-82 also lazily imports `dw_mcp.client.DwClient` and `dw_mcp.server.build_server` (the only dw/ module importing the mcp SDK; tested by tests/test_server_mcp.py:169 and the torch-free-import test tests/test_mcp_server.py:1091).
Fix: move LOOPBACK_HOSTS/WILDCARD_HOSTS into a leaf module (e.g. `dw/server/hosts.py`); app.py, serve.py and mcp_mount.py all import from it. `dw_mcp/client.py:14` keeps its own copy on purpose (must not import `dw.*` because that pulls torch via dw/__init__.py).
Other import edges into app.py from outside dw/server: `dw/serve.py:154,246,248` only (LOOPBACK_HOSTS, create_app, default_ui_dir). dw_mcp never imports dw.server.

---------------------------------------------------------------------------
## 5. How tests reach app.py

String-target `patch("dw.server.app...")` (mock.patch): **0**. Only string target of any kind: `monkeypatch.setattr("dw.server.app.local_addresses", boom)` at tests/test_server_info.py:198 (1 use; the name is imported at app.py:160 and called at 4315 - if server_info moves, this target must move too). Other dw.server string patches: `"dw.server.updater.subprocess.run"` x2 (unaffected).
Module-object patching via `from dw.server import app as app_module` / `import dw.server.app as app_module` (all in tests/):
- `monkeypatch.setattr(app_module, "create_app", fake)` - test_serve_main.py:13/24, test_security_auth.py:376/384, test_security_trust_gate.py:507/509 (these patch what `dw.serve.main` imports: `from .server.app import create_app`). If `create_app` moves, serve.py and these 3 tests must agree; easiest is to keep `create_app` defined/re-exported in `dw.server.app`.
- `monkeypatch.setattr(app_module, "build_plan", boom/spy)` - tests/test_server.py:5074-5096, 5462-5485 (5 uses; `build_plan` is imported into app.py at :101 and called from `_validation_plan` ~1650-1727 and the `/api/jobs` path). **Breaks if validate/admission code moves to another module** - patches must target the new module's `build_plan` (or app.py must still be where it is looked up).
- `app_module.video_shape` read + `monkeypatch.setattr(app_module, "video_shape", counting_shape)` - tests/test_server.py:2145-2158 (imported app.py:97, used in gallery_frames). Breaks if frames move.
- `app_module._prune_missing(cache)` + `monkeypatch.setattr(app_module.os.path, "exists", ...)` - tests/test_server.py:6028-6056 (module-level helper 229; `app_module.os.path` patch is global on os.path, module-independent, but `_prune_missing` must still be importable from wherever the test looks).
Direct `from dw.server.app import <names>`:
- `create_app` - 16 test files (test_server, test_server_info, test_server_mcp, test_security_web_headers, test_vram_inheritance, test_admission, test_security_symlinks, test_server_guides, test_security_auth, test_rerun_new_seed, test_server_exports, test_security_decoder_bombs, test_library_sources, test_server_assess, test_security_input_caps, test_server_workspaces, test_validate_arguments).
- `collect_prompt_references` - tests/test_prompt_references.py:16
- `attach_observed`, `workflow_details` - tests/test_catalog_structure.py:16; `attach_observed` also tests/test_observed_cost.py:474
Also: tests/test_arch_report.py:40-43 uses the strings "dw.server.app"/"dw.server.jobs" as fixture edge data only (no real import). scripts/arch_report.py / arch_metrics.py measure modules (baseline tooling for the refactor; no hardcoded app.py path found in scripts/*.py).
`app.state.*` reads in tests/ and src (counts): job_manager 38, prompt_dir 7, workspace 4, workspace_root 3, observed_costs 3, mcp_mounted 3, example_asset_dirs 3, default_workspace 3, asset_dir 3, workflow_dir 2, static_files_by_root 2, example_prompt_dirs 2, workflow_sources 1, raw_media_extensions 1, media_kinds 1 -> keep these `app.state` names stable.
Conclusion: keep re-exports in `dw/server/app.py` (create_app, collect_prompt_references, attach_observed, workflow_details, LOOPBACK_HOSTS, default_ui_dir, `_prune_missing`) and update 4 patch sites (`build_plan` x5, `video_shape` x2, `local_addresses` x1) to the new module, or tests break. Nothing outside tests patches through dw.server.app.
Jobs module tests: `from dw.server.jobs import JobManager` (13+ files), `JobHistory` (11), `TERMINAL_STATES`, `Job`, `MAX_PERSISTED_EVENTS`, `TERMINAL_JOBS_KEPT`, `RUNNING/QUEUED/SUCCEEDED` - a jobs.py split needs a package `__init__` re-export or test edits.

---------------------------------------------------------------------------
## 6. dw_mcp/server.py `build_server` (65-1344; 1,279 lines)

Registration mechanism: NOT decorators. Tool functions are plain nested `def`s with docstrings inside `build_server`, registered by the inner helper `tool(fn, annotations)` (115-127) -> `server.add_tool(_anticipated(fn), name=fn.__name__, description=inspect.cleandoc(fn.__doc__), annotations=...)`, called in batches at the end of each section (e.g. 471-494 loop with READ_ONLY, 1071, 1292, 1340). `_anticipated` (45-58) wraps DwApiError -> ToolError. Annotation constants READ_ONLY/WRITES/OVERWRITES/DELETES at 27-42. Server instructions text 67-112 (46 lines, verbatim in the MCP server "instructions"; note the dw MCP instructions appear in the session prompt, so must not change byte-wise unless intended).
Closure state: only `client` (param) and `server` (+ helper `tool`). Nothing else - no caches, no locks. 65 `client` references, all passing it into handler modules.
Tool count: **59** nested functions (0 duplicates).
Line anatomy (ast): tool bodies total 1,062 lines, of which **657 are docstrings** (the agent-facing descriptions) and only ~405 executable; ~all tools are 2 executable lines (`return module.fn(client, ...)`). Exceptions with real logic living in server.py (violating the docstring's own "every tool body is a one-line call" claim): `get_output_frames` 563-650 (88 lines, 77 code: builds ImageContent/AudioContent parts + text telemetry), `get_output_image` 496-527 (23 code lines, ImageContent + telemetry), `get_output_audio` 529-561 (20 code), `list_gallery` 354-410 (19), `validate_workflow` 901-948 (17), `keep_output` (15), `enhance_prompt` (15), `upload_asset` (15), `run_workflow` 1079-1135 (21 code; also mutates `__doc__` to interpolate `{cap}` = `diagnose.MAX_WAIT_SECONDS` at 1132-1135; same for `wait_for_job` 1221-1224), `download_output` (13), `list_workflows` (13).
Grouping (section comments) with spans: catalog 127-~494 [list_workflows, get_workflow, get_schema, list_pipelines, get_pipeline_signature, list_classes, get_class, list_tasks, get_task, list_models, get_memory, clear_memory, get_health, get_server_info, list_jobs, list_gallery, get_gallery_metadata, list_guides, get_guide] (19); media 494-~756 [get_output_image, get_output_audio, get_output_frames, get_output_text, assess_output, delete_output, download_output] (7); assets 758-857 [list_assets, upload_asset, keep_output, delete_asset] (4); workspaces 859-899 (4: list/use/create/delete); authoring 901-994 [validate_workflow, save_workflow, delete_workflow] (3); prompts 996-1077 [list_prompts, get_prompt, get_prompt_schema, save_prompt, delete_prompt, list_enhancers, enhance_prompt] (7); run/diagnose 1079-1295 [run_workflow, get_job, get_job_workflow, get_job_events, wait_for_job, cancel_job, rerun_job, move_job, export_job] (9); models 1297-1343 [download_model, list_downloads, cancel_download, delete_model, get_diffusers_state, update_diffusers] (6). Sum 59.
Existing sibling modules (all plain `(client, **kwargs)` functions; MCP-SDK-free; only server.py imports the SDK): client.py 492 (DwClient, DwApiError, api_path, `_scoped` workspace query, `mounted` flag), catalog.py 362 (18 fns), media.py 648 (14), assets.py 438 (10), diagnose.py 371 (10), authoring.py 127 (4), prompts.py 104 (7), workspaces.py 212 (6), models.py 97 (6), exports.py 84 (1), guides.py 37 (2), __main__.py 133 (entry). So the pattern "tool body -> handler module" already exists and is followed for the data call; what remains in server.py is (a) docstrings, (b) MCP content-type construction for image/audio/frames, (c) annotation policy.
Natural split: `dw_mcp/tools/<group>.py` each exposing `register(server, client, tool)` (or a list of `(fn_factory, annotations)`), with server.py keeping only `build_server` + instructions + READ_ONLY/... + `_anticipated`; the ~90-line content-builders of media go to media-tool module. Docstrings must move verbatim (see gotcha).
Gotchas: tests/test_mcp_server.py:1524 `test_the_tool_surface_fits_the_budget` counts surface tokens (13,790.8 measured 2026-09-21; budgets comments at 1449-1502) - a move must keep description text byte-identical (inspect.cleandoc, 3.12/3.13 indent difference comment at 116-121). tests import `from dw_mcp.server import build_server` (test_mcp_server.py:16/140/764, test_server_mcp.py:169, test_mcp_main.py:27 patches `cli.build_server`, `dw_mcp/__main__.py:15`, `dw/server/mcp_mount.py:75`). tests/test_mcp_server.py:1091 asserts `import dw_mcp.server` pulls in none of torch/diffusers/dw (keep dw_mcp free of dw.* imports). Tool registration order = tool listing order in the session; keep if clients care.

---------------------------------------------------------------------------
## 7. dw/serve.py `main` (23-280; 258 lines; module is 285)

- 1-21: module docstring, imports, forced `multiprocessing.set_start_method("spawn")` (import-time side effect).
- 24-119 (96 lines): argparse declaration only - 13 options: --host, --port, --workspace, --workflow-dir, --output-dir, --prompt-dir, --examples-dir (append -> examples_dirs), --asset-dir, --output-layout, -l/--log_level, --token, --trust-workflows, --mcp. Pure data; extract to `build_parser()`.
- 119-152: `args = parse_args()`; workspace resolution + pin (`set_workspace(resolve_workspace(args.workspace))` 126), derive workflow_dir/output_dir/asset_dir (127-129), `makedirs` workflow_dir only if defaulted (137), `makedirs(asset_dir)` (141), pin `DW_ASSET_DIR` (145), output layout (147-150), token = flag or `DW_API_TOKEN` (152).
- 154-169: `from .server.app import LOOPBACK_HOSTS` (wait: imports server.app = fastapi etc. just to read a constant) and the hard error for `--mcp` on non-loopback without token (exit 2).
- 171-175: `set_trust_workflows(args.trust_workflows)` (pins env for spawned worker).
- 177-186: prompt dir discovery + pin `DW_PROMPT_DIR`.
- 188-219: example libraries: `example_libraries`, `set_library_fallbacks` for PROMPTS, WORKFLOWS, ASSETS (common_assets created, ordered common then examples).
- 220-227: uvicorn import check (SystemExit 1 with message).
- 229-248: `startup(log_level)`, logger, warning when bound non-loopback with no token; imports `create_app`, `default_ui_dir`.
- 250-264: `create_app(...)` with 12 kwargs; print banner (266-272).
- 273-281: `uvicorn.run(..., timeout_graceful_shutdown=5)`.
Split: `build_parser()` (~97), `configure_environment(args) -> ServerConfig` (env/dir pinning, 119-219 ~100 lines), `check_bind_safety`, `run(config)`; `main` ~20 lines. Tests patch `app_module.create_app` (test_serve_main.py, test_security_auth.py:376, test_security_trust_gate.py:507), so `create_app` has to keep being looked up as `dw.server.app.create_app` at call time (it is imported lazily inside main, which is what makes the patch work - keep that lazy import).
Order constraints (comments in file): env pins must precede `create_app` and the worker spawn; `--mcp` token check precedes `startup()` and any worker spawn.

---------------------------------------------------------------------------
## 7b. dw/server/jobs.py (1,567 lines)
Module constants 54-94 (states QUEUED/RUNNING/SUCCEEDED/FAILED/CANCELLED/TERMINAL_STATES; ACK_NONE/BOOLEAN/BOUND 66-68; RERUN_SPEC_KEYS 71; TERMINAL_JOBS_KEPT 90; MAX_PERSISTED_EVENTS 94).
- `JobHistory` 97-513 (417 lines): sqlite persistence; `__init__` 106 (75 lines: schema + migrations), record 194, recent_summaries 233, finished_runs 313, job_for_file 407, `_to_detail` 478, watermark 293 (used by ObservedCosts), etc.
- `Job` 516-744 (229 lines): in-memory record, events, `_note_progress` 582 (62), progress 645 (44).
- `JobManager` 747-1567 (821 lines): submit 786 (104), definition 898, realized 932, rerun_spec 984, rerun 1029, list 1104, cancel 1135, move 1156, `_run_loop`/`_run_job` 1187/1220, `_consume_results` 1309 (57), memory_status 1492 (45), clear_memory 1538, probe_cache 1459, restart_worker_if_idle 1419, plus manifest recording helpers 1259-1308.
Natural split (all >1000 eliminated): `jobs/history.py` (JobHistory ~420), `jobs/job.py` (Job ~230 + constants), `jobs/manager.py` (JobManager, itself 821 - ok under 1000; could peel memory_status/clear_memory/probe_cache/worker-memory caching ~160 lines and manifest helpers ~55 into mixin/helper modules). No function exceeds 150 (`submit` 104 is the longest). Keep `dw.server.jobs` re-exporting (13+ test files import JobManager/JobHistory/Job/TERMINAL_STATES/MAX_PERSISTED_EVENTS/TERMINAL_JOBS_KEPT; app.py:150 imports a tuple of names).

---------------------------------------------------------------------------
## 8. API version / client dependency on route paths

Versioning:
- Single source: `pyproject.toml:11` `version = "0.6.0-alpha.1"`; `dw.__version__` (dw/__init__.py:82-105) reads it from pyproject (regex) at import. `plugins/dw/.claude-plugin/plugin.json:4` version is the same, bumped by scripts/release.sh (104-115).
- Reported at: GET /api/health `version` (app.py:4274), GET /api/server `version` (4321; MCP `get_server_info`), dw_mcp/__main__.py:123 prints `health.get("version")`, run manifests `dw_version` (dw/workflow.py:1805), `sysinfo`/updater report diffusers version separately (updater.py:59-64).
- There is NO separate HTTP API version: `FastAPI(title=..., description=...)` (app.py:804) sets no `version=` (so /docs shows default 0.1.0), no `/api/v1` prefix, no Accept/version header negotiation. Routes are flat (`/api/...`, `/outputs`, `/inputs`, `/exports`, `/mcp`). Changing any route path is a breaking change for the clients below. If the plan wants a declared API version, it is a new addition (e.g. `app = FastAPI(version=__version__)`, `api_version` in /api/health and /api/server), not a rename.

Clients and the routes they call (paths verbatim; any move of handlers into routers must keep these identical):
- dw/run.py (thin client via dw_mcp.client.DwClient): POST /api/jobs (116, 130), GET /api/jobs/{id}/event-log?after= (157), GET /api/jobs/{id} (174), POST /api/jobs/{id}/cancel (246). Matches the 400 message prefix of `resolve_workflow_reference` (dw/run.py:38) - a wording dependency on app.py module-level helper text.
- dw_mcp (api_path / client calls): /api/health, /api/server, /api/assets (+ /{name} DELETE), POST /api/uploads, POST /api/validate, /api/workflows (+/{name}, /{name}/variables, PUT, DELETE), /api/schema, /api/pipelines(+/{name}), /api/classes(+/{name}), /api/tasks(+/{command}), /api/models, /api/memory, POST /api/memory/clear, /api/jobs (list/get/cancel/event-log/export/move/rerun/workflow), /api/gallery (list, /{name}/metadata|assess|audio|frames, DELETE /{name or run_dir}), /api/guides(+/{name}), /api/prompts(+/{name}), /api/prompt-schema, /api/enhancers, POST /api/enhance, POST /api/models/download, /api/models/downloads (+/{id}/cancel), DELETE /api/models, /api/system/diffusers(+/update), /api/workspaces (list/POST/DELETE /{name}), plus GET /outputs/{name} (media.py:53,405,625 - not under /api). Not used by MCP: SSE /api/jobs/{id}/events, /api/gallery/archive, /api/assets/archive, /api/gallery/{name}/thumbnail|download, /api/workflows/{name}/download, /api/prompts/{name}/download, /inputs, /exports.
- ui/src/lib/api.ts (+pages; all through encodePath/encodeURIComponent): /api/jobs (+?query, /{id}, /{id}/workflow, /rerun, /move, /export[?workspace=], /cancel, /events?after= via EventSource with ?token=), /api/health, /api/server, /api/memory, /api/models (+?repo=, /download, /downloads, /downloads/{id}/cancel), /api/system/diffusers(+/update), /api/pipelines(+/{name}), /api/tasks(+/{name}), /api/classes(?kind=, /{name}), /api/schema, /api/workflows(+/{name}), /api/validate, /api/prompts(+/{name}), /api/prompt-schema, /api/enhancers, /api/enhance, /api/gallery (?limit=, /{name}/..., /archive), /api/assets (+/keep, /archive, /{name}), /api/uploads?filename=, /api/workspaces(+/{name}), /inputs/..., /exports/<job id>.zip. Query-token (?token=) allowed only on the five `@query_token_ok` GET routes: job events, workflow download, prompt download, gallery thumbnail, gallery download.
- External/test surface of the route set: tests/test_server*.py etc. exercise via TestClient with literal paths (no import of handlers) - route paths unchanged means no test edits beyond the patch sites in section 5.
Route-order / middleware gotchas when splitting into APIRouters: (1) greedy `{name:path}` suffix routes (workflows, prompts, gallery, assets) must register their specific suffix routes before the bare one; (2) `/api/gallery/archive` and `/api/assets/archive` are POST literal paths next to `{name:path}` DELETE; (3) SPA `app.mount("/")` last; (4) `/mcp` routes before the SPA mount; (5) middleware order; (6) `require_bearer_token` gates path prefix `/api/` (and the exact `/mcp` spellings) - /outputs, /inputs, /exports are intentionally ungated; (7) `_matched_route` depends on `request.app.router.routes` containing starlette Route-compatible objects with `.endpoint` - APIRouter include_router satisfies that.

---------------------------------------------------------------------------
## 9. Duplicated logic noticed

1. Search-path construction written three+ times:
   - `_asset_roots(ws)` app.py:2657-2670 re-implements `dw/assets.asset_search_path` (dw/assets.py:76) - its docstring says "the same order 'asset:' resolves in (dw/assets.asset_search_path)", i.e. kept in sync by hand.
   - `_asset_roots_for_job` app.py:2692-2719 repeats the same abspath/dedupe/isdir loop with `asset_dir` swapped for `ws.assets` (near copy of `_asset_roots`).
   - `_prompt_roots` app.py:2269-2279 re-implements `dw/prompts.prompt_search_path` (dw/prompts.py:57).
   - `_sources_for` (1030) wraps `workflow_sources` (dw/workflow_sources.py) per workspace - OK but the workspace->roots mapping is in app.py.
2. Constants twinned by comment, not import: `LOOPBACK_HOSTS` (app.py:680 vs dw_mcp/client.py:14 - intentional because dw_mcp can't import dw.*, pure duplication otherwise); poll interval `SSE_POLL_SECONDS` (app.py:172) vs `dw/run.py:32` vs `dw_mcp/diagnose.py:17` `WAIT_POLL_SECONDS`.
3. The job-status vocabulary/validation for `GET /api/jobs` (app.py:1234-1276: unknown-status 400, limit semantics) is mirrored on the MCP side in dw_mcp/catalog.py:195-215 (joins statuses, reverses list, computes `truncated`); not identical logic (server cuts newest N, client reverses) but the contract lives in two places.
4. `_acknowledgement_form` (app.py:1067) classifies none|boolean|bound and jobs.py has ACK_NONE/ACK_BOOLEAN/ACK_BOUND (66-68): the classification is app.py's, the labels jobs.py's; `_bound_plan_for`/`_check_bound_acknowledgement` (1074-1144) are called by both submit_job and rerun_job with parallel argument preparation (submit 1158-1232 vs rerun 1325-1374 repeat the admission/acknowledgement/plan sequence).
5. Inside app.py: output-path resolution repeated per route - every gallery sub-route (metadata/assess/audio/frames) opens with the same `_strip_output_prefix` -> `_output_file` -> `_asset_file` fallback -> `MEDIA_KINDS` kind check sequence (2982-3447); `keep_output_as_asset` and `archive_assets` repeat root iteration. A single `resolve_media(ws, name)` would replace ~6 copies.
6. `workflows` get/delete/download/variables each repeat the `resolve_readable_workflow(_sources_for(ws), name)` pattern (fine) but `get_workflow_variables` (2145-2214) and `list_workflows` (1994-2046) both call `workflow_details`/`attach_observed`/observed-cost and the catalog_shape projections - plus `_validation_plan` calls `_observed_for_name` (2216) - the same "observed cost for a workflow" logic reached via three paths.
7. Health/server-info both compute `device`/`version` separately (4265-4291, 4293-4354), importing `__version__, get_device, get_device_type` locally in each (4269, 4312).
8. dw/run.py:38 and dw_mcp depend on the literal wording of server error messages (`resolve_workflow_reference` 400 prefix; MCP `_format_detail` rendering 409 plan) - fragile coupling to keep when moving helpers.

---------------------------------------------------------------------------
## Quick size facts
dw/server: app.py 4,521; jobs.py 1,567; catalog_shape.py 487; exports.py 480; guides.py 310; admission.py 305; observed_cost.py 379; updater 192; assess 135; enhancers 129; netinfo 124; mcp_mount 95; sysinfo 71. dw_mcp: server.py 1,344; media 648; client 492; assets 438; observed above. dw/serve.py 285.
Only function >150 lines in create_app besides create_app itself: gallery_frames (166). Others >100: validate_workflow 133, upload_media 116, workflow_details (module-level) 120, submit_job 75, `JobManager.submit` 104.
