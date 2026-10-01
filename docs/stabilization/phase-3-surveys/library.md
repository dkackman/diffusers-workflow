> Survey taken at `9e25c49a` (2026-09-30) for the Phase 3 plan. Line numbers are that commit's: re-verify before relying on them.

# Survey: library search path / root precedence (for the LibraryPath refactor)

Repo: /Users/don/src/dkackman/dw-stabilization. Paths below are relative to it. Line numbers are from the working tree at survey time.

## 0. Shape of the problem in one paragraph

There are four "libraries" (workflows, prompts, assets, outputs) and each has its own ad-hoc answer to "which roots, in which order, who can write".
The search-path rule is written in three layers that each re-derive it:
(a) the engine side (`dw/assets.py`, `dw/prompts.py`, `dw/workflow_sources.py` + `dw/workspace.library_fallbacks`, fed by env vars pinned by `dw/serve.py`);
(b) the server side (`dw/server/app.py` closures `_asset_roots`, `_prompt_roots`, `_sources_for`, fed by `app.state.example_*_dirs`);
(c) the tagging/writability rule (`origin`, `writable`), which is a *third* encoding, different per library.
Only outputs is a single root (no search path).

---

## 1. Where a search path / precedence is computed or applied

### 1.1 dw/workspace.py (constants, workspace resolution, generic env-pinned fallbacks)

- `WORKFLOWS_SUBDIR/PROMPTS_SUBDIR/ASSETS_SUBDIR/OUTPUTS_SUBDIR/COMMON_SUBDIR/EXPORTS_SUBDIR` :50-53, :71, :77. `RESERVED_WORKSPACE_NAMES = SUBDIRS + (EXPORTS, COMMON)` :~96. `NAMED_SUBDIRS` (no prompts) :~100.
- `Workspace` class :112-218. Library directories are *properties derived from a root*: `workflows` :165, `prompts` :169 (`_prompts_root or root/prompts`), `assets` :174, `outputs` :178, `common_assets` :182 (`(_common_root or root/common)/assets`). `describe()` :~198 reports them.
- `ConfiguredWorkspace(Workspace)` :648-717: same four properties but *injected* (loose `--workflow-dir/--asset-dir/--output-dir/--prompt-dir`), `common_assets` None when there is no root, `ensure()` raises NotImplementedError. This is the second way library roots are defined for one workspace; the server's default workspace is a ConfiguredWorkspace (app.py:850).
- `named_workspace(workspace, name)` :458-476: builds `Workspace(root/name, prompts_root=<root>/prompts, common_root=<root>/common)` from the *root Workspace's `.root`*. It therefore ignores a `--prompt-dir` / `--asset-dir` override given to `dw.serve` (see 1.8 disagreement D5).
- `resolve_workspace()` :235-266 (flag > DW_WORKSPACE(+DW_WORKSPACE_SOURCE) > settings > cwd if `looks_like_workspace` > ~/diffusers-workspace).
- `LIBRARY_PATH_ENV_VARS` :300-304 maps prompts/assets/workflows -> `DW_PROMPT_PATH`/`DW_ASSET_PATH`/`DW_WORKFLOW_PATH`.
- `library_fallbacks(subdir, primary)` :307-329: reads the env var, splits on `os.pathsep`, abspath/expanduser, drops `primary`, dups, and non-existent dirs. Returns *only the read-only tail*; the writable root is never in it. No origin information at all.
- `set_library_fallbacks(subdir, roots)` :332-342: the writer of those env vars (called only by dw/serve.py).
- `example_libraries(examples_dirs)` :345-370: for each `--examples-dir`, looks for `prompts/` and `assets/` in the dir itself, then in its parent. Returns `{prompts:[...], assets:[...]}`. (Workflows trees are the `--examples-dir` themselves; not in this function.)
- `discover_library(subdir, env_var, base_dir)` :373-414: THIRD precedence rule, used by `get_prompt_dir` and `get_asset_dir`: env var (DW_PROMPT_DIR / DW_ASSET_DIR) > explicit workspace's `<subdir>` > `./<subdir>` if exists > walk up from `base_dir` > workspace `<subdir>`.
- `_holds_a_workspace`/`workspace_names`/`workspace_contents`/`_own_folders` :417-560: enumerate library folders again (workflows/assets/outputs) for usage accounting and listing.

### 1.2 dw/workflow_sources.py (workflows; already the closest thing to LibraryPath)

- Origins: `WORKSPACE_ORIGIN/EXAMPLES_ORIGIN/BUILTIN_ORIGIN` :27-29, plus `COMMON_ORIGIN` :33 which is "not a workflow source" but lives here because it is the asset library's tag. So the origin vocabulary is defined in the workflows module and imported by the server for prompts and assets.
- `WorkflowSource(root, origin, writable)` :61-87: `contains(path)` (via `validate_path`), `names()`, `to_dict()` -> `{root, origin, writable}`.
- `workflow_sources(workflow_dir, examples_dirs, include_builtin)` :109-135: `[WorkflowSource(workflow_dir, workspace, True)] + examples (False)`; dedups a read-only root equal to the writable one (checkout case).
- `writable_source` :138, `source_for_path` :146, `resolve_in_source(source, name, allow_create)` :156 (appends `.json`, `validate_path` against root), `find_workflow(sources, name)` :170 (front-to-back first hit -> `(path, source)`), `listing(sources)` :227 (first source wins a name: shadowing, with no record of what was shadowed), `workflow_names(root)` :90 (os.walk for `*.json`, skipping symlinks that escape) — **also used by the prompt listing** (app.py:2311), so the "list a json library" primitive lives here and is reused for prompts.
- `fallback_roots(primary)` :237-247: engine-side read-only workflow roots = `library_fallbacks(WORKFLOWS_SUBDIR, primary)` i.e. env DW_WORKFLOW_PATH. A *second* workflow search path, separate from `workflow_sources()`; it is examples-only (no builtin) and untagged.
- `resolve_sub_workflow(path, base_dir, confine_to)` :283-383: applies the path — order: relative to base_dir, +.json, the run's own root (`confine_to`), then each fallback root. It constructs throwaway `WorkflowSource(root, EXAMPLES_ORIGIN, False)` at :~329 and :~353 purely to reuse `contains`/`resolve_in_source`; the origin there is a lie (the confine_to root may be the writable workspace).
- `catalog_root(directory)` :42 (nearest ancestor named `workflows`); `builtin_root()` :37.

### 1.3 dw/assets.py (assets, engine side)

- `ASSET_DIR_ENV_VAR = "DW_ASSET_DIR"` :30 (defined here; `DW_PROMPT_DIR` is an inline string in prompts.py:54 and serve.py:186 — inconsistent).
- `_active_asset_dir` ContextVar :39; `activate_asset_dir/deactivate_asset_dir` :42-48 (the worker sets it per job; admission sets it per validate request: admission.py:~140).
- `get_asset_dir(base_dir)` :51-68: active contextvar > `discover_library(ASSETS_SUBDIR, DW_ASSET_DIR, base_dir)`.
- `asset_search_path(asset_dir, base_dir)` :76-90: `[primary] + library_fallbacks(ASSETS, primary)`. So the engine path is `[active-or-discovered workspace assets] + env DW_ASSET_PATH` = `[ws.assets, common/assets, *example assets]` (because serve.py pins that order into the env).
- `resolve_asset_reference` :93-126: first root where `validate_path(join(root,name), root)` isfile wins. Raises ValueError naming every searched root.
- Note: `resolve_asset_reference(ref, asset_dir=root)` **still appends the env fallbacks** — so "probe one root" (admission.py:291) is not probing one root (see D7).

### 1.4 dw/prompts.py (prompts, engine side)

- `get_prompt_dir(base_dir)` :39-54 = `discover_library(PROMPTS, "DW_PROMPT_DIR", base_dir)`.
- `prompt_search_path(prompt_dir, base_dir)` :57-72: `[primary] + library_fallbacks(PROMPTS, primary)` (env DW_PROMPT_PATH).
- `resolve_prompt_reference` :75-102: first root with `<root>/<name>.json`; `validate_prompt_path(path, root)`.
- `fetch_prompt` :130-160 (reads text, rejects reserved prefixes).
- Identical structure to assets.py lines 76-126 (same function written twice, differing only in extension handling and validator).

### 1.5 dw/runs.py + dw/locations.py + dw/realize.py (outputs, and a consumer of asset roots)

- Outputs is a single root, not a search path. `_active_output_root` ContextVar :98-104 (set in `Workflow.run`, workflow.py:1149, 1228 with `self.output_dir`); `output_root()` :107-115: active > `resolve_workspace().outputs`. `resolve_output_reference(ref, root)` :210; `fetch_output` :261. `OUTPUT_LAYOUT_ENV_VAR = DW_OUTPUT_LAYOUT` :43 (set by `set_output_layout`).
- Who names the output root: server routes use `ws.outputs` (app.py: :1125, :1183, :1208, :2534..., :3290 gallery etc.); admission uses `outputs=workspace.outputs` (admission.py:299); exports use `output_root` param; `realize.py:194` uses `output_root or default_output_root()`. No env pin for outputs (job carries `output_dir` in spec, jobs.py:758,842-876).
- `dw/locations.py:media_roots` :72-110: `[base_dir] + asset_search_path(base_dir=base_dir) + [output_root()]`, all realpath'd; confinement set for `validate_media_path`. This is a consumer that must see the *same* asset path as the resolver.

### 1.6 dw/serve.py (the only writer of the pins)

- :124-126 `set_workspace(resolve_workspace(args.workspace))` -> writes `DW_WORKSPACE` + `DW_WORKSPACE_SOURCE` (workspace.py:282-283).
- :127-130 derive `workflow_dir = args.workflow_dir or ws.workflows`, `output_dir = args.output_dir or ws.outputs`, `asset_dir = abspath(args.asset_dir or ws.assets)`.
- :145 `os.environ["DW_ASSET_DIR"] = asset_dir`.
- :181-186 `prompt_dir = abspath(args.prompt_dir or get_prompt_dir(base_dir=abspath(workflow_dir)))` (note: uses the discover_library walk for prompts but NOT for assets) then `os.environ["DW_PROMPT_DIR"] = prompt_dir`.
- :197-206 `example_dirs = example_libraries(args.examples_dirs)`; `set_library_fallbacks(PROMPTS, example_dirs[PROMPTS])`; `set_library_fallbacks(WORKFLOWS, args.examples_dirs)`.
- :208-220 creates `workspace.common_assets` (makedirs) and `set_library_fallbacks(ASSETS, [common_assets] + example_dirs[ASSETS])`.
- Then `create_app(...)` is called with the same `examples_dirs` (see 1.7), which re-runs `example_libraries` (app.py:826). So examples->roots is computed twice per process (serve.py:201 and app.py:826), and passed to two consumers by two different channels (env for worker, `app.state` for API).

### 1.7 dw/server/app.py (API side, the second derivation)

State setup (:815-863):
- :815 `app.state.workflow_dir = workflow_dir`; :819 `app.state.workflow_sources = workflow_sources(workflow_dir, examples_dirs)` — **dead**: grep finds no reader in dw/ or tests/ (routes call `_sources_for(ws)` instead).
- :820 `app.state.prompt_dir`; :826-828 `example_libraries(examples_dirs)` -> `app.state.example_prompt_dirs`, `app.state.example_asset_dirs`; :832 `app.state.asset_dir`; :837 `app.state.workspace`; :842 `workspace_root = Workspace(workspace,"flag")`; :850 `default_workspace = ConfiguredWorkspace(workflows=workflow_dir, assets=asset_dir, outputs=?, prompts=prompt_dir, root=workspace)`.
- :1002 `_workspace_for(name)` default -> `default_workspace`; else `named_workspace(root, name)` and must pass `_holds_a_workspace`. `selected_workspace` :1024 is the FastAPI dependency. `_workspace_root` :991.

Workflows:
- `_sources_for(ws)` :1030 = `workflow_sources(ws.workflows, examples_dirs)` (examples_dirs is a closure var).
- `resolve_readable_workflow` :473 (find_workflow or 404 with suggestions), `resolve_writable_workflow` :484 (always the writable source, **409** "no writable workflow directory" if none, 404 on traversal), `resolve_workflow_reference` :~510 (run by name/path, 400 outside every source).
- Listing: `GET /api/workflows` :1994; `workflow_details(sources_by_name)` :~325-430 attaches `origin` and `writable` per entry (with a cache that re-stamps them :349-355, :417-421), `attach_observed(... writable)` :289-320 uses `writable` to scope observed history.
- `PUT /api/workflows/{name}` :2048, `DELETE` :2104 (403 when `not source.writable`, :2113-2118), `GET` :2232 (headers `X-Workflow-Origin`, `X-Workflow-Writable`), `/download` :2134, `/variables` :2145.
- `_ceiling_index(ws)` :1039-1045 iterates `listing(_sources_for(ws))`.

Prompts:
- `resolve_prompt_name(prompt_dir, name, allow_create)` :621-643: name rule + `validate_path`; 400 on bad name for save, 404 for read.
- `_prompt_roots()` :2269-2278: `[app.state.prompt_dir] + [example dirs not equal to primary]` — no isdir filter (the engine's `library_fallbacks` has one), independent of workspace (prompts are shared).
- `_find_prompt(name)` :2281-2296: `(path, writable = index == 0)`.
- `GET /api/prompts` :2299: walks roots with `workflow_names(root)`, `referenceable(name)` filter, origin `WORKSPACE_ORIGIN if index==0 else EXAMPLES_ORIGIN` :2314 (first root wins a name; no shadowed list).
- `PUT` :2355 always `app.state.prompt_dir`; `DELETE` :2376 403 via `not writable` :2382; `GET` :2402 headers `X-Prompt-Origin` (:2414 `WORKSPACE_ORIGIN if writable else EXAMPLES_ORIGIN`) and `X-Prompt-Writable`; `/download` :2393; `/api/prompt-schema` :2256.
- `_admit(workspace)` :1146 passes `prompt_roots=_prompt_roots()` and `asset_roots=_resolution_roots(workspace)` to `admit()` — then admission.py:290-296 re-implements "try each root" with `resolve_*_reference(..., asset_dir=root)` (D7).
- `_bound_plan_for` :1074-1093 and validation plan :1702 call `build_plan(prompt_dir=workspace.prompts ...)` (engine `prompt_search_path`, env-based).

Assets:
- `_asset_in(name, roots)` :2566-2597 (name check once, per-root `validate_path`, symlink escape = miss, 404 naming roots). `_asset_file(ref, ws)` :2599 -> `_asset_in(name, _resolution_roots(ws))`.
- `_common_assets(ws)` :2648 = `getattr(ws, "common_assets", None)`.
- `_asset_roots(ws)` :2657-2669: `[ws.assets, common, *app.state.example_asset_dirs]`, abspath, dedupe, drop non-dirs (isdir filter). Matches `asset_search_path` ordering by convention, not by shared code.
- `_resolution_roots(ws)` :2672-2689: `_asset_roots` or fallback to `[abspath(ws.assets)]` (so a 404 names something) or `[]`.
- `_asset_roots_for_job(job_id, ws)` :2692-2719: the same loop *a third time*, with the job's `spec["asset_dir"]` replacing `ws.assets` (export).
- `_asset_origin(ws, root)` :3795-3810: origin by **directory equality** (`ws.assets` -> workspace, `_common_assets` -> common, else examples) — deliberately not by position (comment :3799-3803), unlike prompts (by index) and workflows (by stored attribute).
- `GET /api/assets` :3812-3922: builds `libraries=[{origin, dir, writable: origin != EXAMPLES_ORIGIN}]`, walks each root with `_iter_gallery_files(root, group_runs=False)`, reports later duplicates under `shadowed` with `shadowed_by` (origin of winner); sorted by mtime.
- `POST /api/uploads` :3678: writes `ws.assets or ws.outputs` (fallback to outputs when no asset dir!), or `_common_assets(ws)` when `shared` (409 if none); under `uploads/` subfolder; `validate_output_path`.
- `POST /api/assets/keep` :3924: `_common_assets(ws) if shared else ws.assets` (409 if none), hardlink/copy from `ws.outputs`, 409 if exists without `overwrite`.
- `POST /api/assets/archive` :4031 (uses `_resolution_roots`).
- `DELETE /api/assets/{name}` :4057: walks `_asset_roots`, first file hit; `_asset_origin == EXAMPLES_ORIGIN` -> **403**; 409 when no roots; 400 bad name; 404 none.
- `GET /inputs/{name}` :4443: first root holding it via `validate_path`; `GET /outputs/{name}` :4411: `asset:` reference -> `_asset_file`, else `ws.outputs`. Both use `_static_files_for(root)`.
- Job export: `:1396 _asset_roots_for_job`, dw/server/exports.py:186-300 `_copy_assets(asset_roots)` re-loops roots (fourth loop).

Other places roots are enumerated:
- dw/server/admission.py:~131-140 activate_asset_dir(workspace.assets); :192-300 `argument_reference_errors(asset_roots, prompt_roots, outputs)` with `over_roots` helper.
- dw/server/jobs.py submit :794-878 stores `output_dir`, `asset_dir`, `workflow_dir`, `workspace` per job; :1042-1077 and :1237-1243 replay them.
- dw/worker.py :20,:471-479,:696-706 `activate_asset_dir(command["asset_dir"])` per job/probe.

### 1.8 Where the same rule is written twice or disagrees

- D1 Asset search order: `dw/assets.py:76-90` (primary + env DW_ASSET_PATH) vs `app.py:2657-2669` (`ws.assets, common, example_asset_dirs`) vs `app.py:2692-2719` (job variant) vs serve.py:215-219 (writes the env in `[common]+examples` order) — one order, four writers. The env is the truth for the worker, `app.state` for the API; nothing enforces they agree except convention.
- D2 Prompt search order: `dw/prompts.py:57-72` vs `app.py:2269-2278`. Differ in isdir filtering (engine drops missing example dirs, API keeps them) and in primary source (`discover_library` vs `app.state.prompt_dir`).
- D3 Workflow search: `workflow_sources()` (tagged, includes examples + optional builtin) vs `fallback_roots()`/DW_WORKFLOW_PATH (untagged examples). Two lists derived from `args.examples_dirs` at serve.py:206 and app.py:1033.
- D4 Origin/writable tagging is three different mechanisms: workflows = attribute on `WorkflowSource`; prompts = `index == 0` (app.py:2286, 2314, 2414); assets = directory-equality against `ws.assets`/`common` (app.py:3795), with `writable = origin != EXAMPLES_ORIGIN` (app.py:3839, 4090). Prompts have no `common`; assets have no `builtin`.
- D5 `named_workspace` builds `prompts_root = <root>/prompts` (workspace.py:~470) regardless of `--prompt-dir`, whereas the server's default ConfiguredWorkspace carries `app.state.prompt_dir` (app.py:850-855). `ws.prompts` is used at app.py:1087 and :1702 (plan building). With `--prompt-dir X` and a named workspace, plan building resolves `prompt:` via root/prompts (+env fallbacks) while listing/saving/running use X. Worth a test before relying on; likely a real inconsistency.
- D6 `discover_library` (workspace.py:373) vs serve.py:124-145: assets skip discovery (straight `args.asset_dir or ws.assets`), prompts use it with the walk-up. Three precedence rules for "where is the writable library": `resolve_workspace`, `discover_library`, serve.py's explicit derivation.
- D7 `resolve_asset_reference(ref, asset_dir=root)` (assets.py:76-90 via 93) re-appends the env fallbacks, so admission.py:290-296 "check each root" (`over_roots`) and the API's `_asset_in` (comment at app.py:2566-2580 explicitly works around this) are two different implementations of "find name in roots".
- D8 `WorkflowSource(root, EXAMPLES_ORIGIN, False)` constructed in `resolve_sub_workflow` (:329, :353) as a container, mislabelled (the confine_to root can be the writable workspace).
- D9 Env var names: `ASSET_DIR_ENV_VAR` constant (assets.py:30) exists, prompts use inline `"DW_PROMPT_DIR"` (prompts.py:54, serve.py:186), serve.py:145 inline `"DW_ASSET_DIR"`; `DW_OUTPUT_LAYOUT` and `DW_TRUST_WORKFLOWS` are in runs.py/security.py.
- D10 Dead `app.state.workflow_sources` (app.py:819).
- D11 Upload with no asset dir falls back to writing into outputs (`ws.assets or ws.outputs`, app.py ~3734) whereas keep/delete 409; inconsistent "no library" behaviour.

---

## 2. Env var pinning

| Var | Written by | Read by | Content |
|---|---|---|---|
| `DW_WORKSPACE` | `set_workspace` workspace.py:282 (only caller: serve.py:126) | `resolve_workspace` workspace.py:247 (and thus `discover_library`, `output_root` fallback runs.py:113-115, anything in worker) | workspace root |
| `DW_WORKSPACE_SOURCE` | `set_workspace` :283 | `resolve_workspace` :249 | how it was chosen (flag/environment/settings/working directory/default) so an inferred workspace is not promoted to explicit in the worker |
| `DW_ASSET_DIR` | serve.py:145 (`args.asset_dir or ws.assets`); tests set it widely | `discover_library(ASSETS, ASSET_DIR_ENV_VAR)` via `get_asset_dir` (assets.py:68) — lower priority than the `_active_asset_dir` contextvar | writable asset root of the *default* workspace |
| `DW_PROMPT_DIR` | serve.py:186 | `get_prompt_dir` prompts.py:54 via `discover_library` | writable (and only) prompt root, shared |
| `DW_PROMPT_PATH` | `set_library_fallbacks(PROMPTS, ...)` serve.py:202 | `library_fallbacks` (prompts.py:72 via `prompt_search_path`) | os.pathsep list of example `prompts/` dirs |
| `DW_ASSET_PATH` | serve.py:215-219: `[common/assets] + example assets` | `library_fallbacks` (assets.py:90) | read-only tail in resolution order |
| `DW_WORKFLOW_PATH` | serve.py:206: `args.examples_dirs` | `fallback_roots` (workflow_sources.py:247) -> `resolve_sub_workflow` | read-only workflow trees |
| `DW_OUTPUT_LAYOUT` | `set_output_layout` (serve.py:~147) | `output_layout()` runs.py:266 | run/flat |
| `DW_TRUST_WORKFLOWS` | `set_trust_workflows` (serve.py) | security | not a library var, but same channel |
| `DW_MCP_WORKSPACE` | user | dw_mcp/client.py:128-136 | *workspace name* (not a dir), unrelated to DW_WORKSPACE |
| `DW_DEVICE`, `DW_PUBLIC_URL`, `DW_API_TOKEN` | n/a | unrelated | |

Notes:
- The outputs root and the named-workspace asset root are NOT pinned in env; they travel in the job spec (`output_dir`, `asset_dir`, `workflow_dir`, `workspace`; jobs.py:794-878) and are activated per job (`activate_asset_dir` worker.py:478, `activate_output_root` workflow.py:1149).
- Because `DW_ASSET_PATH` contains `common/assets` of the *default* workspace root and named workspaces share the same root, this works; a `ConfiguredWorkspace` with no root yields no common (workspace.py:~700).
- Precedence inside worker for assets: contextvar (job) > DW_ASSET_DIR > discover chain.

---

## 3. dw.security <-> dw.workspace

Both directions are function-local imports; there is NO module-level cycle today.

- security -> workspace: `dw/security.py:875` `from .workspace import RESERVED_WORKSPACE_NAMES` inside `validate_workspace_name` (def at :862).
- workspace -> security: `dw/workspace.py:465` `from .security import validate_workspace_name` inside `named_workspace`; `dw/workspace.py:486` same inside `create_workspace`.
- Also lazy: workspace -> settings (`:256`, `from .settings import load_settings`), workflow_sources -> workspace (`workflow_sources.py:245`, inside `fallback_roots`).
- Module-level edges that would matter: `assets.py:20-21` imports security and workspace at top; `prompts.py:15-17` imports schema, security, workspace at top; `workflow_sources.py:23` imports security at top; `workspace.py` imports only stdlib at top (os, time, pathlib). `security.py` top-level imports are stdlib only (:5-10).
- Would a LibraryPath module break it? Not if it follows the same rule. Safe layouts:
  1. New leaf module (say `dw/library.py`) importing `.security` at top level (for `validate_path`, `contained`, `SecurityError`) and importing NO workspace symbols at module level. `workspace.py`, `assets.py`, `prompts.py`, `workflow_sources.py` may then import it at top. `security.py` must not import it at top.
  2. If LibraryPath needs `SUBDIRS`/env-var names from workspace.py and workspace.py needs LibraryPath (to return one from `Workspace.library('assets')`), one of them must import lazily — or move the constants (`*_SUBDIR`, `RESERVED_WORKSPACE_NAMES`, env var names) into the new leaf module and have `workspace.py` re-export them. That also removes the only security->workspace edge (`RESERVED_WORKSPACE_NAMES`) if `validate_workspace_name` reads it from the leaf; `security` -> leaf at top would be fine *only if the leaf does not import security at top*, so choose: either leaf constants module (no imports) + LibraryPath module (imports security), or keep the lazy import at :875.
  - Tests import `RESERVED_WORKSPACE_NAMES`/subdir constants via `dw.workspace`, so keep re-exports.
- CodeQL: `.github/codeql/dw-security/` models `validate_path(path, base)` as a barrier only with `base is not None` — LibraryPath must do its containment through `validate_path(..., root)` (not its own normalize-then-check) or the path-injection alerts come back.

---

## 4. Existing classes close to LibraryPath

- `WorkflowSource` (workflow_sources.py:61) is already `{root, origin, writable}` + `contains`, `names`, `to_dict`. `workflow_sources()` is already "writable first, examples after, dedupe the equal root". `find_workflow`, `listing`, `writable_source`, `resolve_in_source` are already LibraryPath's read/shadow/write-target operations — but hard-coded to `.json` names and `workflows`.
- What is missing to generalize it (one concept instead of four):
  - name->file mapping per library: workflows/prompts `<name>.json` (+ `.json` optional on input), assets `<name>` literal with extension. A `suffix`/`name_to_file` parameter.
  - name validator per library: `validate_prompt_reference`, `validate_asset_reference`, (workflow: none beyond containment).
  - listing walker: `workflow_names` (json) for prompts/workflows, `_iter_gallery_files(root, group_runs=False)` for assets (app.py) — assets list by media kind, not by suffix.
  - origin vocabulary: `workspace | common | examples | builtin`. `COMMON_ORIGIN` already lives in workflow_sources.py.
  - shadow reporting: `listing()` drops shadowed names; assets report `shadowed` with `shadowed_by`. LibraryPath.entries() should yield both.
  - env pinning: `library_fallbacks`/`set_library_fallbacks`/`LIBRARY_PATH_ENV_VARS` are the serialization of "the read-only tail" — generalize to serializing the whole path with origins (e.g. `DW_ASSET_PATH="common:/x;examples:/y"`) so the worker's tags match the API's; today the worker only gets untagged dirs.
  - per-job primary override (`activate_asset_dir` contextvar; job spec `asset_dir`) — LibraryPath for a job = `LibraryPath.with_primary(job.asset_dir)`.
- `Workspace.describe()` (workspace.py:~198) and the `sources`/`libraries`/`prompt_dirs` fields on listings are the three serializations.
- `media_roots` (locations.py:72) and `_copy_assets` (exports.py:280) and `admission.over_roots` are consumers that should take a LibraryPath instead of a bare list of dirs (they currently each re-run validate_path + isfile).

---

## 5. HTTP / MCP surface

### 5.1 Routes (all under `/api` unless noted)

Workflows (take `?workspace=`):
- GET `/api/workflows` list; GET `/api/workflows/{name}` read; GET `/{name}/variables`; GET `/{name}/download`; PUT `/{name}` save; DELETE `/{name}`.
Prompts (NO `workspace` param — shared by design):
- GET `/api/prompts` (`tag`, `intended_model`, `include_text`); GET `/api/prompts/{name}`; GET `/{name}/download`; PUT `/{name}`; DELETE `/{name}`; GET `/api/prompt-schema`.
Assets (take `?workspace=`):
- GET `/api/assets` list; POST `/api/uploads?filename&asset_name&shared` (create); POST `/api/assets/keep` body `{name, asset_name, overwrite, shared}`; POST `/api/assets/archive`; DELETE `/api/assets/{name}`; GET `/inputs/{name}` and GET `/outputs/asset:<name>` (serve). No "get one asset's metadata" route, no PUT.
Outputs (for contrast): GET/DELETE `/api/gallery...`, `/outputs/{name}`.
Workspaces: GET/POST `/api/workspaces`, DELETE `/api/workspaces/{name}`.

### 5.2 MCP tools (dw_mcp/server.py -> dw_mcp/*.py)

- workflows: `list_workflows` (server.py:130, catalog.py:9), `get_workflow` (:175), `save_workflow` (:950, authoring.py:69, client-side JSON-merge-patch), `delete_workflow` (:980, takes `workspace`), `validate_workflow`, `run_workflow`.
- prompts: `list_prompts` (:996, prompts.py:24), `get_prompt` (prompts.py:41), `save_prompt` (:56), `delete_prompt` (:63) — none take `workspace`.
- assets: `list_assets(detail, workspace)` (:758, assets.py:207), `upload_asset(..., shared, workspace)` (:772), `keep_output(..., shared, workspace)` (:814), `delete_asset(name, workspace)` (:839).
- MCP `client.get_json` returns body only (client.py:372-377), so `X-Workflow-Origin/Writable` and `X-Prompt-Origin/Writable` headers never reach an agent; an agent learns a prompt's/workflow's origin only from the listing (prompts: `origins` map; workflows: `details[name].origin` — but the *compact* view keeps only `COMPACT_FIELDS` (catalog_shape.py:360-372), which has neither `origin` nor `writable`; `sources` at top level survives).

### 5.3 Listing response shapes (the inconsistencies)

| | workflows `GET /api/workflows` | prompts `GET /api/prompts` | assets `GET /api/assets` |
|---|---|---|---|
| writable dir key | `workflow_dir` | `prompt_dir` | `asset_dir` |
| roots list | `sources: [{root, origin, writable}]` | `prompt_dirs: [str]` (no origin/writable) | `asset_dirs: [str]` **and** `libraries: [{origin, dir, writable}]` |
| names | `workflows: [name]` + `details{name:{..., origin, writable}}` | `prompts: [name]` + `details{name:{...}}` + `origins{name: origin}` | `assets: [entry{name, reference, folder, kind, size, mtime, origin, url, absolute_url?}]`, `folders` |
| per-entry origin | in `details[name].origin` | in separate `origins` map | in each entry |
| per-entry writable | `details[name].writable` | absent | absent (derive from `libraries`) |
| shadowed | not reported | not reported | `shadowed: [... shadowed_by]` |
| workspace echoed | `workspace` | no | no (upload/keep/delete echo it) |
| read-only field | `writable` | `writable` (header only) | `writable` (libraries) |
| empty-case | normal | normal | special early return with `libraries: []`, `shadowed: []` |
| sort | by name | by name | by mtime desc |

Field-name disagreements: `root` (WorkflowSource.to_dict) vs `dir` (assets libraries) vs bare strings (prompt_dirs); `sources` vs `libraries` vs `prompt_dirs`; `origin` values `workspace | examples | builtin` (workflows), `workspace | examples` (prompts), `workspace | common | examples` (assets; `COMMON_ORIGIN`).

Single-item read: workflows/prompts return the raw JSON body with origin in headers (`X-Workflow-*`, `X-Prompt-*`); assets have none.

Write/delete response bodies:
- save workflow `{name, workspace, origin, warnings, shape, traits, summary}` (app.py:2092-2102); save prompt `{name}` (:2375); upload `{reference, workspace, url, shared, absolute_url?}` (or, with no asset dir, `{workspace, url}`); keep `{reference, name, workspace, linked, shared}`.
- delete workflow `{name, workspace, origin, deleted}`; delete prompt `{name, deleted}`; delete asset `{name, workspace, reference, deleted, origin}`.

Status codes:
- Delete read-only: 403 in all three (workflow :2113, prompt :2382, asset :4091) but three different message texts.
- No writable library: workflows PUT 409 "no writable workflow directory" (:493); assets keep/delete 409; upload instead silently writes to outputs; prompts cannot occur.
- Bad name: prompt save 400 / read 404 (`resolve_prompt_name`, :621); asset delete 400, asset read 404; workflow traversal 404 on save (:498) but 400 via `resolve_workflow_reference` for runs.
- Exists: keep 409 unless `overwrite`; upload with `asset_name` overwrites silently (no `overwrite` flag); prompt/workflow PUT overwrite silently.
- Save over a read-only shadowed name: workflow -> writes copy to workspace; prompt -> writes to prompt_dir (same effect); assets -> upload writes to `ws.assets` (same effect), keep 409s only if the *writable* file exists.
- `shared` (write into `common`) exists only for assets.

UI coupling (must change with any shape change): `ui/src/lib/api.ts` :253-299 (workflows `sources`, headers), :483-496 (assets), :571-587 (prompts `origins`, `prompt_dirs`); `ui/src/lib/types.ts` :161-191, :298-315 (`AssetFile.origin`, `AssetLibrary`, `ShadowedAsset`); `ui/src/lib/api.test.ts` (2 hits). MCP: `dw_mcp/assets.py:61-100` reads `libraries[].writable/dir` to decide legal upload sources; `ASSET_SUMMARY_FIELDS` :196.

---

## 6. Test coupling

No `patch("dw.<module>....")` string targets exist for workspace.py, workflow_sources.py, assets.py, prompts.py (the only string patch targets in this area are `dw.locations.socket.getaddrinfo` in tests/test_locations.py:166 and tests/test_security_ssrf.py:78,237,311,334,371,400,423,444,467). Coupling is by direct import and by environment variables.

Direct imports (name -> test):
- `dw.workspace`: `Workspace` (test_admission, test_server, test_server_exports, test_server_info, test_server_workspaces, test_validate_arguments, test_vram_inheritance, test_library_sources, test_workspace); `create_workspace` (test_server_exports, test_server_info, test_workspace); `delete_workspace`, `workspace_names`, `named_workspace` (test_workspace); `forget_workspace_usage` (test_server_workspaces); `WORKSPACE_ENV_VAR`, `WORKSPACE_SOURCE_ENV_VAR` (test_assets, test_output_references, test_workspace); `WORKFLOW_PATH_ENV_VAR` (test_workflow); `example_libraries`, `library_fallbacks`, `set_library_fallbacks`, `ASSETS_SUBDIR`, `PROMPTS_SUBDIR` (test_library_sources); `DEFAULT, ENVIRONMENT, FLAG, SETTINGS, WORKING_DIRECTORY, looks_like_workspace, resolve_workspace, set_workspace` (test_workspace).
- `dw.workflow_sources`: `WorkflowSource, listing` (test_catalog_structure); `listing, workflow_sources` (test_vram_inheritance); `EXAMPLES_ORIGIN, WORKSPACE_ORIGIN` (test_library_sources); `workflow_names` (test_security_symlinks); `SubWorkflowNotFound`, `builtin_root` (test_workflow); test_workflow_sources.py imports a larger set (see its header).
- `dw.assets`: `asset_search_path, resolve_asset_reference` (test_library_sources); `resolve_asset_reference` (test_security_symlinks); `get_asset_dir` (test_worker_execute); test_assets.py imports a block.
- `dw.prompts`: `fetch_prompt, prompt_search_path, resolve_prompt_reference` (test_library_sources); `fetch_prompt` (test_assets, test_output_references, test_security_symlinks); `PROMPT_PREFIX, resolve_prompt_reference` (test_prompt_references); `get_prompt_dir` (test_workspace); `RESERVED_TEXT_PREFIXES` (test_server_guides); test_prompts.py imports a block.

Env-var coupling (monkeypatch.setenv/delenv on the pins): `DW_ASSET_DIR` in test_assess:798, test_slice_preflight:36, test_shot_span_preflight:25, test_security_symlinks:238,319, test_dissolve_frame_errors:49, test_rule_parity:49,194, test_workflow:1223, test_security_trust_gate:514,806, test_server:1058, test_security_auth:389, test_library_sources:99; `DW_PROMPT_DIR` in test_serve_main:28, test_previous_results:161, test_assets:167, test_prompts:30-85, test_workspace:225-247, test_output_references:177, test_security_auth:388, test_security_trust_gate:513, test_library_sources:98; `DW_ASSET_PATH` test_security_symlinks:239, test_library_sources:102; `DW_PROMPT_PATH` test_library_sources:101; `DW_WORKSPACE(+_SOURCE)` test_serve_main:32-33, test_security_auth:390-391, test_security_trust_gate:515-516. So the env-var contract (names, ordering of contextvar > DW_*_DIR > discover) is a test-pinned public interface; keep env names or migrate these ~30 sites.

Server/HTTP tests that pin the shapes in section 5: tests/test_server.py (28 hits for origin/libraries/shadowed/prompt_dirs/asset_dirs/sources/writable/X-Prompt/X-Workflow), tests/test_mcp_assets.py (10), tests/test_server_workspaces.py (6), tests/test_library_sources.py (5), tests/test_security_auth.py (2), tests/test_catalog_shape.py (2), tests/test_mcp_authoring.py (1). `tests/test_server.py` also exports `ScriptedWorkerManager`, `success_script`, `valid_workflow` which test_library_sources imports.

tests reach closures only through HTTP (`create_app(..., examples_dirs=..., prompt_dir=..., asset_dir=..., workspace=...)`); nothing references `app.state.example_*`, `_asset_roots`, `_prompt_roots` or `app.state.workflow_sources` by name — those can be refactored freely. The `create_app` kwargs (`workflow_dir, prompt_dir, asset_dir, examples_dirs, workspace`, app.py:741-757) are the stable seam.

---

## 7. Suggested inputs for the plan (observations, not decisions)

1. Generalize `WorkflowSource` into the LibraryPath entry and `workflow_sources()` into its constructor; add `suffix`, `validate_name`, `lister`, and origin vocabulary (`workspace|common|examples|builtin`) in one module that imports only `security` at top.
2. One constructor per library taking a `Workspace` + examples dirs: `ws.library("assets")` returns the ordered tagged path (`ws.assets`, `common`, examples), replacing `_asset_roots`, `_asset_roots_for_job` (primary override), `asset_search_path` (primary override via contextvar), `_prompt_roots`, `prompt_search_path`, `_sources_for`, `fallback_roots`.
3. Serialize the tagged path into the env once (serve.py) so the worker rebuilds the same object; retire `DW_*_PATH` untagged lists (or keep them as the tail format).
4. Unify write-target and read-only refusal in the object (`writable_root()`, `require_writable(entry)` raising one error type the routes map to 403) — replaces `index==0`, `origin != EXAMPLES_ORIGIN`, `source.writable`.
5. Unify listing/response shape (`libraries: [{origin, dir, writable}]`, per-entry `origin`/`writable`, `shadowed`, `workspace`) across workflows/prompts/assets, plus the single-read origin (headers vs body). UI/MCP touch points listed in 5.3.
6. Fix or explicitly decide D5 (named-workspace prompts vs `--prompt-dir`), D7 (single-root probe), D10 (dead state), D11 (upload-to-outputs fallback).
