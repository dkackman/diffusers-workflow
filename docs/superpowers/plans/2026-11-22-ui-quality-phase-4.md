# UI quality phase 4: results - find, understand, reuse. Implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to carry out this plan task by task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** a person can find what they made, see it large, understand what made it and use it again, the same way on every page that shows a file. One tile per kind, one viewer, one recipe, and a metric that fails when the next page draws its own `<img>`.

**Architecture:** Three building blocks in `ui/src/lib/ui/`: `OutputTile` (one tile per kind: image, video poster, audio duration, text excerpt), `MediaViewer` (a window-filling overlay with zoom, previous/next, swipe and fullscreen; the page keeps the open file in the hash as `open=`), and `RecipePanel` ("What made this", owning its metadata read). Two small pieces sit under them: `MediaBody` (one file drawn by its kind, the only other place a raw `<img>`/`<video>`/`<audio>` may appear) and `ReferenceChip` (a copyable `output:`/`asset:` reference). The kind-to-glyph map and the viewer's pure logic live in `ui/src/lib/media.ts` and `ui/src/lib/mediaViewer.ts`. The Gallery, Assets, the job page, request takes, Compare, the Jobs list, the Overview, the catalog, the workflow page and the media picker are then moved onto them. One server task adds two opt-in listing fields (`prompts=true`) and a probe memo, so search can match prompts; nothing changes for a caller that does not ask.

**Tech stack:** Svelte 5 runes, vitest with @testing-library/svelte, Playwright, ESLint flat config, Node 24 type stripping for `scripts/arch-metrics.mjs`. Server: FastAPI 0.143.0 / Pydantic 2.14.1 for the OpenAPI dump, pytest.

**Spec:** `docs/superpowers/specs/2026-11-22-ui-quality-design.md`. This plan covers:
- principle P7 (*Show the proof, and show it the same way*);
- the building-blocks row *Media* (`MediaViewer`, `OutputTile`, `RecipePanel`);
- the phase-4 row of *Phases*: GA-5..10, GA-11 (bulk star), GA-12, GA-15, GA-16, GA-L*, JB-2..4, JB-10, JB-14..16, RQ-5, SH-12, SH-5, and the journey gate `find-and-reuse.spec.ts`;
- the survey items D1 (media tile), D2 (recipe and detail panel) and its §15 `ReferenceChip`, from `docs/superpowers/specs/2026-11-22-ui-duplication-survey.md` §1, §2, §15 and the *Shared pieces that pages bypass* row `assetThumbnailUrl`;
- *Keeping it from coming back*: the kind-to-glyph map comes only from its owning module.

GA-L* expands, from the spec's appendix, to: **crops** (tiles are square crops), **audio/text tiles** (a glyph and a name, nothing else), **star wrap** (the star and note marks wrap inside the 4-line caption), **filter width**, **Compare tools** (a cell cannot be removed or opened), **asset detail** (no download, a different media size, a separate fetch), **prompt cards** (the model chip truncates the name, `biomechanical_da...`), **single-delete confirm**, and **picker gaps** (a clip in the picker has no play badge, and its own glyph map). The audit (`docs/superpowers/specs/2026-11-22-ui-audit.md`, Tier 1 #15-16, Tier 2 §B and §C, Tier 3 *Jobs* and *Gallery and assets*) has the longer text for each.

**Out of scope:**
- **#1599**, the created and last-changed timestamps pass. `OutputTile` has a `meta` snippet slot and `RecipePanel` keeps the size-and-time line the detail panels show today (`formatBytes · formatMtime`), so #1599 has a place to put a time. This phase does no timestamp inventory, no `<Time>` component and no fill-in. The Gallery's date grouping (GA-9) reads the listing's existing `mtime` through the existing `formatDay`.
- **#1600-#1602**, the phase-3 follow-ups, except the items folded below. Each task that touches a component a #1600 item names says so.
- **Phase 6 request behaviour**, including "accepting also stars". Task 16 opens takes in the viewer and offers Accept there with the same behaviour Accept has today.
- **Phase 5's `ReferenceField`.** `MediaArgumentInput`'s preview and `Suggest`'s thumbnail stay as they are (the two remaining `raw_media` exceptions it will burn down). Task 7's *Use as input* fills a workflow page field through the existing prefill, never a new input.

**Folded in from #1600** (each in the task named, each cheap and in a component the task already rewrites):
1. The empty Compare page flashes a skeleton, and a failed gallery read shows Retry instead of the empty copy (Task 15).
2. `join('/')` for the Compare route is ambiguous for a segment holding an encoded `/` (Task 15).
3. "(run root)" in `JobResults.svelte:146` uses "run" as a noun (Task 12).
4. A group toggle under an active filter flips its stored state with no visible effect, in `FolderGroups` (Task 8; MediaPicker and ShadowedAssets keep theirs, #1600 still owns those).
5. The asset listing GET runs on every Save as asset press, and a failed listing blanks the editors' shared asset list (Task 6).
6. Hand-edited `roles=` values are not validated, and roles are encoded as `%2C` in the hash (Task 8).
7. GalleryDetail's job link uses `workspace.current` (Task 5: `RecipePanel` links the job in the file's own workspace; WorkerList and EnhancePanel stay with #1600).

## Measured at the start (develop bcfab7c0)

**Files that show media today.** 27 raw `<img>`/`<video>`/`<audio>` elements in 14 `.svelte` files (`grep -rnE '<img|<video|<audio' src --include='*.svelte'`):

| File | Elements | What |
|---|---|---|
| `src/lib/gallery/GalleryDetail.svelte` (447 lines) | 3 (`:254`, `:257`, `:259`) | the docked detail panel; media `max-width: min(480px, 100%)` (`:417`), Settings collapsed in `<details>` (`:292-341`), Made from (`:343-364`) |
| `src/lib/assets/AssetDetail.svelte` (173) | 3 (`:121`, `:124`, `:126`) | the asset popout; media `max-height: 60vh` (`:166`), a copy button for `asset:` (`:53`), no download |
| `src/lib/job/JobResults.svelte` (392) | 3 (`:184`, `:189`, `:193`) | each output inline at `min(480px, 100%)` (`:310`) beside model/seed/prompt (`:217-239`); Made from (`:122-140`) |
| `src/lib/pages/ComparePage.svelte` (249) | 2 (`:128`, `:142`) | full media per cell, one transport |
| `src/lib/pages/GalleryPage.svelte` (504) | 2 (`:342`, `:351`) | grid tile, square crop, 4-line caption with inline role, version, note and star marks (`:333-395`, CSS `:421-503`) |
| `src/lib/pages/AssetsPage.svelte` (521) | 2 (`:419`, `:421`) | grid tile: the **original** image and a **`<video preload="metadata">` per tile** (GA-12), 110 px, accent hover (CSS `:465-520`) |
| `src/lib/pages/OverviewPage.svelte` (439) | 2 (`:165`, `:277`) | recent strip linking to the gallery root (`:160-176`, SH-12); workflow cards (`:274-289`); jobs panel links only the id (`:220-232`) |
| `src/lib/pages/WorkflowsPage.svelte` (439) | 2 (`:267`, `:274`) | card proof, `<img>` duplicated in both branches |
| `src/lib/pages/WorkflowPage.svelte` (445) | 2 (`:298`, `:304`) | proofs linking to the gallery root (`:291-314`) |
| `src/lib/request/ItemProof.svelte` (164) | 1 (`:55`) | request proof: 96 px row, **120 px take** (RQ-5), 48 px cover |
| `src/lib/assets/MediaPicker.svelte` (335) | 1 (`:225`) | picker tile, 96 px, no play badge on a clip |
| `src/lib/editor/MediaArgumentInput.svelte` | 2 (`:204`, `:206`) | phase 5's (`ReferenceField`) |
| `src/lib/ui/Suggest.svelte` | 1 (`:136`) | a suggestion row's thumbnail; phase 5's |
| `src/lib/job/ChainSegments.svelte` | 1 (`:238`) | a segment thumb seeked to its first frame; stays, named |

So: **4 viewers** (GalleryDetail, AssetDetail, JobResults, ComparePage) and **9 tile implementations** (Gallery, Assets, ShadowedAssets `:44-60` with no picture, MediaPicker, Workflows, Overview ×2, WorkflowPage, ItemProof). `ItemProof` is used by `AttemptStrip.svelte:62`, `RequestItem.svelte:232` and `RequestsPage.svelte:87`.

**Kind-to-glyph maps: 5.** `refcomplete.svelte.ts:19` (`lut` → `¶`), `MediaPicker.svelte:49`, `ItemProof.svelte:73`, `AssetsPage.svelte:424` (inverted condition), `GalleryPage.svelte:362`.

**Already done, or no longer applicable:**
- D2's "GalleryDetail takes 17 props": it takes 7 now and reads its metadata through `resource()` (phase 2). The fetch still moves, into `RecipePanel`.
- GA-10's rename to **Save as asset** and its in-app name dialog with Replace or Keep both (phase 3). What is left is *Copy reference* and *Use as input*.
- GA-11's delete half (phase 1). The bulk star is left.
- GA-16's empty state (phase 3, XC-16). The skeleton flash (#1600) and the URL are left.
- GA-L *filter width*: phase 3's `PageHeader` `filter` slot holds one `.pagehead .filter` rule. Only the Gallery's placeholder changes here.
- GA-L *single-delete confirm*: phase 3's `fileConfirm` through `destroyConfirm`.
- JB-14's ETA in raw seconds: phase 3's `formatApprox`. The raw JSON rows, the log that does not follow, and the step-only estimate labelled as the job's are left.
- JB-10's stale workflow after the job finishes: the definition is a `resource` watching `jobs` since phase 2 (`JobPage.svelte:52-66`). Task 13 pins it with a test; the seed chip after a restart is still open (`seed` is read only from `workflow_start` events, `JobPage.svelte:184-187`).
- GA-5 is **still open**: `filterActive` counts `roles.value.length < 2` (`GalleryPage.svelte:111`), and the Gallery opens on `['final']`, so every folder is forced open.

**Metrics** (`cd ui && node scripts/arch-metrics.mjs`):
`files_over_size_ceiling 0, complex_functions 13, long_functions 0, import_cycles 0, modules_in_import_cycles 0, prefix_literals 3, a11y_suppressions 3, raw_error_toasts 0, empty_outside_loadstate 0, unwatched_reads 0, window_dialogs 0, unguarded_mutations 0, banned_terms 0, duplicate_blocks 30`. Against `docs/stabilization/ui/baseline.json`, develop is red on `complex_functions 11 -> 13` and `prefix_literals 0 -> 3`, as it was at the start of phase 3. Neither is this phase's to fix, and neither may rise. 11 of the 30 `duplicate_blocks` are in files this phase rewrites (AssetDetail, ShadowedAssets ×2, GalleryDetail ×3, JobResults ×2, AssetsPage ×3).

Python (`venv/bin/python scripts/arch_metrics.py`): `modules 281`, `claude_md_lines 129`, `modules_over_size_ceiling 5`. `dw/server/api_models.py` 1105 lines; `dw/server/job_history.py` 1000; `dw/server/routes/gallery.py` 680; `dw/server/routes/jobs.py` 1041; `dw/server/outputs.py` 762. `ui/CLAUDE.md` is 12 lines.

**Page sizes** (the limit is 600): `app.css` 600 (at the limit), `Sidebar` 584, `FlowView` 576, `App.svelte` 561, `RequestItem` 550, `AssetsPage` 521, `EditorPage` 509, `ModelsPage` 505, `GalleryPage` 504, `JobsPage` 475, `JobPage` 465, `PromptEditorPage` 461, `StepEditor` 458, `GalleryDetail` 447, `WorkflowPage` 445, `WorkflowsPage` 439, `OverviewPage` 439, `JobResults` 392, `ComparePage` 249, `patterns.css` 155, `frames.css` 119.

## Global constraints

- **MCP behaviour is unchanged.** Every server change is additive and opt-in: a caller that does not pass `prompts=true` gets the same payload, key for key. OpenAPI is regenerated only through `scripts/dump_openapi.py` under the pinned FastAPI 0.143.0 / Pydantic 2.14.1 (`constraints-openapi.txt`): install them into a scratch overlay with `venv/bin/python -m pip install --target "$SCRATCH/oapi" -c constraints-openapi.txt fastapi pydantic`, then `PYTHONPATH="$SCRATCH/oapi" venv/bin/python scripts/dump_openapi.py && cd ui && npm run gen:api`. Never hand-edit `ui/src/lib/generated/`.
- **Size limits.**
  - Python `modules` stays at 281. No task here needs a new module; any rise needs Don's approval, and the task must stop and ask.
  - `claude_md_lines` stays at 129: `ui/CLAUDE.md` stays at 12 lines, rewritten, never grown.
  - `dw/server/api_models.py` stays at or below 1105 lines. Task 2 adds two fields and pays for them by folding descriptions onto their lines.
  - No page over 600 lines. `app.css` is at 600, so every new global rule (the viewer's sheet, the tile) goes in `src/patterns.css`; component styles stay in the component.
  - No function over complexity 10 or 60 lines, in TS or Python.
- **The UI metrics gate: no metric rises against a start measurement.** Before Task 1, run `node scripts/arch-metrics.mjs > "$SCRATCH/ui-start.json"` (any scratch folder). After Task 1, run it again into `"$SCRATCH/ui-p4.json"`: it holds `raw_media`'s start.
  - Each task's gate is `node scripts/arch-metrics.mjs > "$SCRATCH/ui-now.json" && node scripts/arch-metrics.mjs --compare "$SCRATCH/ui-now.json" "$SCRATCH/ui-p4.json"`, and it must print nothing.
  - `npm run metrics -- --check ../docs/stabilization/ui/baseline.json` may print only the two lines already red.
  - `window_dialogs`, `unguarded_mutations` and `banned_terms` (phase 3), and `raw_error_toasts`, `empty_outside_loadstate` and `unwatched_reads` (phase 2), stay at 0. `duplicate_blocks` stays at or below 30.
  - A task that lowers a count lowers that key in `baseline.json` in the same commit.
- **The phase gate** is `ui/e2e/journeys/find-and-reuse.spec.ts` passing, and `raw_media` at 4 (Task 19).
- **Staging and commits.** Stage files by name; never `git add -A ui`, because Don edits `ui/package.json` locally. Commit headlines use a scope (`ui(media): …`, `server(gallery): …`, `docs(…)`), never `fix: #N`. Each commit ends with the `Co-Authored-By` line.
- **Checks.** UI checks run from `ui/`: `npm run check`, `npm run lint`, `npx vitest run <file>`, and `npx playwright test <file>` (port 8971). Python: `venv/bin/python -m pytest <file> -q` and `venv/bin/python scripts/arch_metrics.py`. Tests run with `DW_STRICT_RESPONSES=1`, so a key with no model field fails at once.
- WCAG contrast is guidance only, never a check. Keyboard and screen-reader basics (names, focus, `aria-pressed`, a dialog's label) are in scope.
- **Building blocks.** Each new building block gets a row in the UI table of `docs/ARCHITECTURE.md`, in the task that builds it, naming its owner module and its test. A row an existing building block changes (*UI overlays and suggestions* for `Modal`'s `fill`, *A page's place* for `open=`) is edited in that task.
- **Known flaky tests.** The smoke test "editor opens a workflow with introspected arguments" (#1603) and `tests/test_ledger_job_link.py::TestDelete::test_a_request_made_again_under_the_id_takes_the_jobs_back`. Neither is this phase's to fix. Don't chase them.
- **Copy fixed by this plan.** Use it exactly. `<…>` is filled in.
  - **The viewer:** the dialog is labelled by the file's leaf name. Its buttons' labels: `Previous file`, `Next file`, `Zoom in`, `Zoom out`, `Fit to window`, `Actual size`, `Full screen`, `Exit full screen`, `Close`. Its position line: `<i> of <n>`. A file the URL names that is not listed: notice `<leaf> is not in this view any more.`
  - **RecipePanel:** heading `What made this`. Row labels, in this order: `Workflow`, `Job`, `Step`, `Model`, `Seed`, `Prompt`, `Negative prompt`, `Size` (`<w> × <h>`), `Length` (`formatDuration`), `Arguments` (every argument, open), `Made from`, `Location`. With no embedded recipe and no job on record: `No recipe is saved with this file.`, then the hint `A step saves one when its result sets embed_metadata.` (this replaces both "no embedded metadata - enable embed_metadata…" lines, the audit's §F contradiction). An asset saved from a job: `Saved from <workflow file name> v<n>` (replaces "kept from"). An asset's users: `Used by <workflows>`.
  - **Reuse:** the reference chip's copy button is titled `Copy the reference a workflow argument takes`. **Use as input…** opens a dialog titled `Use <leaf> as input`, saying `Workflows in this workspace that take <a|an> <kind>:`, or `No workflow in this workspace takes <a|an> <kind>.`
  - **Bulk star:** BulkBar buttons **Star** and **Unstar**; undo toasts `Starred <n> files` and `Unstarred <n> files`, each with **Undo**.
  - **Gallery find:** the filter's placeholder `search names, prompts and notes…`; the sort `Segmented` options `Newest`, `Oldest`, `Name`, labelled `Sort`; the group `Segmented` options `Folder`, `Date`, labelled `Group by`; date headings `Today`, `Yesterday`, `This week`, `Earlier`. The count chip: `<n> of <m> files` while a filter hides some, else `<m> files`.
  - **Folder toggle while filtering:** the heading's toggle is disabled, titled `Showing every match while the filter is on`.
  - **Job errors** (lead, remedy, action):
    - out of memory: `The card ran out of memory.` / `Run it again with a smaller size or fewer frames.` / **Run again with changes…**
    - a size not divisible by 8: `The width and height must be multiples of 8.` / `Run it again with a size divisible by 8.` / **Run again with changes…**
    - a gated model: `This model needs Hugging Face access.` / `Accept its terms on Hugging Face, then sign in on the server with hf auth login.` / **Open Models**
    - anything else: `The job stopped with an error.` / no remedy / no action.
    - Every one is preceded, when the step is known, by `Failed in step <step>.`, and followed by `<Details summary="Engine detail">` holding the error and the traceback.
  - **The ETA:** `about <duration> left in this step`.
  - **The job page's results head:** **Compare** (2 to 4 pictures or clips; with more, its title says `Compares the first 4`), **Download all**.
  - **Takes:** **Compare takes**.
  - **The event log's rows:** `step_start` → `Started <step>`, `step_end` → `Finished <step>`, `pipeline_step` → `Denoise <i> of <n>`, `workflow_start` → `Started the workflow`, `job_status` → `Status: <status>`, `queue_wait` → `Waiting for a card`, `warning` → its `message`. Any other event: its event name, with the payload behind `Details` (`Event data`). The toggle: **Follow new events** (`aria-pressed`).
  - **"(run root)"** becomes `(top folder)` (#1600).

## Review focus

1. **A gallery of thousands, many of them clips.** No tile ever opens a media stream; durations arrive after the grid is drawn, in one background read, without blanking or reordering it. Pinned in Task 3 (`the grid renders no <video>, and fills durations from one background read`).
2. **The open file goes away under the viewer.** Deleted with the Delete key, deleted by an agent, or hidden by a filter or an unstar: the viewer steps to the neighbour from the file's last place, or closes when nothing is left; an `open=` naming a missing file shows the page with the notice in Global constraints, never a crash or a blank viewer. Pinned in Task 6 (`the viewer steps on when its file is deleted elsewhere`, `an open= naming a missing file says so`).
3. **Keys inside the viewer.** Arrows, S and Delete act on the file only when focus is not in an editable field (the note box), never while a confirm is open over the viewer, and the arrows step even after a click on a video (GA-15). Pinned in Task 4 (`arrows step after a click on the video`, `keys typed in a field or under a dialog do nothing`).
4. **A workspace switch with tiles or the viewer on screen** (SH-5). Every thumbnail and file URL names the workspace its listing came from; the viewer closes when the workspace changes. Pinned in Task 3 (`tiles take their workspace from the listing`) and Task 6 (`switching workspace closes the viewer`).
5. **Odd media.** A 400×1600 image fits the window with the controls on screen (GA-6); a thumbnail the server refuses (over `MAX_DECODE_PIXELS`, or a 404) shows the kind's glyph, not a broken image; an empty text file shows `(empty)`. Pinned in Task 3 (`a thumbnail that fails shows the kind glyph`) and Task 4 (`a tall image fits with its controls visible`).

---

### Task 1: The media module, the `raw_media` metric and the glyph rule

**Files:**
- Create: `ui/src/lib/media.ts`, `ui/src/lib/media.test.ts`
- Modify: `ui/scripts/arch-metrics.mjs`, `ui/scripts/arch-metrics.test.ts`, `ui/scripts/design-rules.test.ts`, `docs/stabilization/ui/baseline.json`
- Modify (the glyph maps): `ui/src/lib/refcomplete.svelte.ts:19`, `ui/src/lib/assets/MediaPicker.svelte:49`, `ui/src/lib/request/ItemProof.svelte:73`, `ui/src/lib/pages/AssetsPage.svelte:424`, `ui/src/lib/pages/GalleryPage.svelte:362`

**Interfaces:**
- Produces, in `media.ts`:
  - `export type MediaFile = { name: string; kind: OutputKind | AssetKind; url: string; label: string; workspace: string; thumb?: string | null; duration?: number | null; starred?: boolean; note?: string }`. `OutputKind` and `AssetKind` come from `types.ts`.
  - `export function kindGlyph(kind: string): string`: `audio` → `♪`; `text`, `lut` and anything else that has no picture → `¶`; `image` and `video` → `''`.
  - `export const hasPicture = (kind: string): boolean` (image or video).
  - `export function outputMedia(file: GalleryFile, workspace: string): MediaFile` (thumb from `galleryThumbnailUrl(file.name, workspace)` for a picture, `duration` from `duration_seconds`) and `export function assetMedia(asset: AssetFile, workspace: string): MediaFile` (thumb from `assetThumbnailUrl(asset.name)`). These are the only two places a tile's thumbnail URL is built.
- Produces, in `arch-metrics.mjs`: **`raw_media`**: every `<img>`, `<video>` and `<audio>` element in a `.svelte` file outside `src/lib/ui/OutputTile.svelte` and `src/lib/ui/MediaBody.svelte`. Detail: `file:line tag`.
- `design-rules.test.ts` gains `the kind glyphs are spelled only in media.ts`: no file under `src/` but `src/lib/media.ts` contains `♪` or `¶` (tests excluded).

- [ ] **Step 1: Write the failing tests.**
  - `media.test.ts`: `maps every kind to its one glyph` (`audio` `♪`, `text` `¶`, `lut` `¶`, `image` `''`, `video` `''`); `builds an output's thumbnail in the listing's workspace` (`outputMedia(file, 'studio').thumb` contains `ws=studio` or the scoped path `scopeTo` makes); `an audio file has no thumbnail`.
  - `arch-metrics.test.ts`: `raw_media counts img, video and audio outside OutputTile and MediaBody`.
  - `design-rules.test.ts`: the glyph case above.
- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/media.test.ts scripts/arch-metrics.test.ts scripts/design-rules.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement**, and replace the 5 glyph maps with `kindGlyph`. `refcomplete`'s `lut` now reads `¶` as it did.
- [ ] **Step 4: Run** the tests, then `node scripts/arch-metrics.mjs`. Expected: PASS and `raw_media 27`. If it differs, check the detail against *Measured at the start*. Write `raw_media: 27` into `baseline.json` and save `$SCRATCH/ui-p4.json`.
- [ ] **Step 5: Commit** `ui(media): one kind-glyph map, and the raw_media metric`.

### Task 2: Prompts on the listings, and a probe memo (server)

What search by prompt (GA-9, JB-4) and audio durations (GA-L) need. Additive and opt-in.

**Files:**
- Modify: `dw/server/outputs.py` (`prompt_text`), `dw/server/job_history.py` (`submitted_prompts`), `dw/server/routes/gallery.py` (`prompts`, `_probed`), `dw/server/routes/jobs.py` (`prompts`), `dw/server/api_models.py` (`GalleryList.prompts`, `JobList.prompts`)
- Modify: `ui/src/lib/generated/*` (regenerated only), `ui/src/lib/api.ts` (`gallery` and `listJobs` take `{ prompts?: boolean; media?: boolean }`), `ui/src/lib/jobsListing.ts`
- Test: `tests/test_gallery_prompts.py` (new test file, not a module), `tests/test_jobs_listing.py`, `tests/test_api_contract.py` (unchanged; it must pass)
- Docs: `docs/SERVER.md` (the two parameters); ARCHITECTURE: the *What the server says about an output file* row names `prompt_text`

**Interfaces:**
- `outputs.prompt_text(arguments: dict | None) -> str | None`: the `prompt` argument as text (a string, or a list's strings joined by `"\n"`), cut to `PROMPT_TEXT_MAX = 2000` characters; `None` when absent, empty or not text.
- `JobHistory.submitted_prompts(workspace: str | None) -> list[tuple[str, str | None, str]]`: `(job id, run id, prompt)` for each job of `workspace` (every workspace for `None`) whose stored `arguments` carry a prompt, through `prompt_text`. One query.
- `GET /api/gallery?prompts=true` adds `prompts: dict[str, str]` to `GalleryList`: run id → the prompt its job was submitted with, for the runs on the returned page. A run whose job history no longer holds has no entry.
- `GET /api/jobs?prompts=true` adds `prompts: dict[str, str]` to `JobList`: job id → prompt, live jobs (their `spec["arguments"]`) and historical ones alike.
- Both fields are `sometimes(...)`: absent unless asked. `api_models.py` stays at or below 1105 lines: fold two multi-line `Field(description=…)` blocks of the gallery section onto fewer lines to pay for the two fields.
- `media=true` durations go through `gallery._probed(path) -> dict | None`, memoised on `(path, st_mtime_ns, st_size)` with `functools.lru_cache(maxsize=4096)`. The answer is the same as before, only cheaper on a second read.
- `api.ts`: `gallery(roles?, requestId?, item?, opts?: { prompts?: boolean; media?: boolean })`, `listJobs(…, opts?: { prompts?: boolean })`. Both keys stay classified as reads.

- [ ] **Step 1: Write the failing tests.**
  - `tests/test_gallery_prompts.py`:
    - `test_prompt_text_reads_a_string_a_list_and_nothing_else`
    - `test_the_gallery_lists_each_runs_prompt_when_asked`
    - `test_the_gallery_says_nothing_about_prompts_unless_asked` (the key is absent)
    - `test_a_second_media_listing_probes_nothing_again` (patch `probe_media`, list twice, one call per file)
    - `test_a_rewritten_file_is_probed_again`
  - `tests/test_jobs_listing.py`: `test_jobs_carry_their_prompts_when_asked`, `test_a_queued_jobs_prompt_is_listed`, `test_jobs_say_nothing_about_prompts_unless_asked`.
- [ ] **Step 2: Run** `venv/bin/python -m pytest tests/test_gallery_prompts.py tests/test_jobs_listing.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement**, then regenerate OpenAPI as in Global constraints, then `cd ui && npm run gen:api`.
- [ ] **Step 4: Run** the two files, `tests/test_api_contract.py`, `tests/test_ui_twins.py`, `tests/test_server_jobs.py`, `tests/test_gallery_roles.py`, then `wc -l dw/server/api_models.py` (expected ≤ 1105), `venv/bin/python scripts/arch_metrics.py` (`modules 281`), and `cd ui && npm run check`. Expected: PASS.
- [ ] **Step 5: Commit** `server(gallery): prompts on the gallery and jobs listings when asked, and a probe memo`, staging the named files and the four generated files.

### Task 3: `OutputTile`, on the Gallery and Assets grids

D1 (survey §1), GA-12, and GA-L *crops*, *audio/text tiles* and *star wrap*.

**Files:**
- Create: `ui/src/lib/ui/OutputTile.svelte`, `ui/src/lib/ui/OutputTile.test.ts`
- Modify: `ui/src/patterns.css` (`.tile`), `ui/src/lib/pages/GalleryPage.svelte:315-397` and its CSS `:421-503`, `ui/src/lib/pages/AssetsPage.svelte:391-433` and its CSS `:465-520`, `ui/src/lib/assets/ShadowedAssets.svelte`
- Test: `ui/src/lib/pages/GalleryPage.test.ts`, `ui/src/lib/pages/AssetsPage.test.ts`
- Docs: ARCHITECTURE, a new UI row *Media tiles*: owner `OutputTile.svelte` and `media.ts`; rule "a file is drawn as a tile only through OutputTile; a clip is always a poster with a play badge; a tile never opens a stream"; enforced by `OutputTile.test.ts`, `raw_media`, `design-rules.test.ts`

**Interfaces:**
- Consumes: Task 1's `MediaFile`, `kindGlyph`, `outputMedia`, `assetMedia`.
- `OutputTile.svelte` props: `{ file: MediaFile; size?: 'grid' | 'card' | 'strip' | 'take' | 'mini'; caption?: 'none' | 'line' | 'clamp'; href?: string; onclick?: () => void; active?: boolean; picked?: boolean; dim?: boolean; alt?: string; badges?: Snippet; meta?: Snippet }`.
  - A picture is the thumbnail, letterboxed (`object-fit: contain` on `--panel-2`), never cropped; a clip adds frames.css's `.videobadge` and, when `duration` is known, its `formatDuration` in the corner.
  - Audio is `♪`, its label and its duration.
  - Text is `¶` and the first 3 lines of the file, read once when the tile first scrolls into view (`IntersectionObserver`, the existing `api.fileText`), `(empty)` for an empty file.
  - A thumbnail `onerror` falls back to the kind's glyph.
  - The marks (star, note) sit in a fixed corner row from the `badges` snippet, never inside the caption.
  - Sizes: `grid` fills its column at 1:1; `card` 4:3; `strip` 96 px; `take` 240 px wide (RQ-5); `mini` 48 px. One hover and one selection look: the ink border (`active`), the ink edge (`picked`).
  - `meta` is the slot #1599 may fill with a time; nothing fills it in this phase.
- The Gallery's grid and the Assets grid render `OutputTile` (`size="grid"`, `caption="clamp"` and `"line"`). The Gallery's tile passes `version` and `RoleTag` in `badges`. The Assets grid uses `assetMedia`, so no `<video>` and no original. The Gallery listing is read without `media`, then a second resource, `durations` (`lazy: false`, `watch: ['gallery']`), reads `api.gallery(undefined, req, item, { media: true })` and the tile takes `duration` by name from it.
- `ShadowedAssets` renders `OutputTile` with `dim` and no thumbnail, dropping its copy of the assets CSS (the 33-line clone, two `duplicate_blocks` runs).

- [ ] **Step 1: Write the failing tests.**
  - `OutputTile.test.ts`:
    - `a clip is a poster with a play badge and its duration`
    - `an image is letterboxed, never cropped` (the img has `object-fit: contain` through its class)
    - `audio shows its glyph, name and duration`
    - `a text tile reads its first lines once, when it is first seen`
    - `a thumbnail that fails shows the kind glyph`
    - `the star and note marks sit outside the caption`
  - `GalleryPage.test.ts`: `the grid renders no <video>, and fills durations from one background read`; `tiles take their workspace from the listing` (the listing says `studio` while `workspace.current` is `default`: every thumbnail URL names `studio`).
  - `AssetsPage.test.ts`: `the asset grid shows thumbnails, never the original or a video`.
- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/ui/OutputTile.test.ts src/lib/pages/GalleryPage.test.ts src/lib/pages/AssetsPage.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement.** Run `wc -l` on GalleryPage and AssetsPage before and after; both shrink.
- [ ] **Step 4: Run** the tests, `npx vitest run src/lib/pages/`, `npx playwright test e2e/smoke.spec.ts e2e/responsive.spec.ts`, and the metrics gate. Expected: PASS, `raw_media` 23, `duplicate_blocks` below 30. Lower both in `baseline.json`.
- [ ] **Step 5: Commit** `ui(media): OutputTile, one tile per kind, on the gallery and asset grids`.

### Task 4: `MediaBody` and `MediaViewer`

The *Media* row's viewer: fitted to the window, zoom, previous/next, swipe, fullscreen. GA-6, GA-7, GA-15.

**Files:**
- Create: `ui/src/lib/ui/MediaBody.svelte`, `ui/src/lib/ui/MediaViewer.svelte`, `ui/src/lib/mediaViewer.ts`, `ui/src/lib/mediaViewer.test.ts`, `ui/src/lib/ui/MediaViewer.test.ts`
- Modify: `ui/src/lib/ui/Modal.svelte` (a `fill` prop), `ui/src/patterns.css` (`.ui-sheet-fill`, the stage and aside layout)
- Docs: ARCHITECTURE, a new UI row *Media viewer*: owners `MediaViewer.svelte`, `MediaBody.svelte`, `mediaViewer.ts`; rule "one file is shown large only in the viewer, fitted to the window; a page keeps the open file in the hash as `open=`, and the viewer's keys act only when no field or dialog has them"; enforced by `mediaViewer.test.ts`, `MediaViewer.test.ts`, `raw_media`. The *UI overlays and suggestions* row names `fill` and says its styles are in `patterns.css` (app.css is full).

**Interfaces:**
- `MediaBody.svelte` props: `{ file: MediaFile; fit?: boolean; onvideo?: (el: HTMLVideoElement | null) => void; controls?: boolean }`. It draws an image, a `<video controls loop playsinline>`, an `<audio controls>` or a text file's contents (`api.fileText`, `ErrorLine` on a failure). The video has `tabindex="-1"`.
- `mediaViewer.ts` (pure):
  - `export const ZOOM_MAX = 8`
  - `export function nextZoom(scale: number, fitScale: number, by: 1 | -1): number`: steps ×1.5 from `fitScale` up to `ZOOM_MAX`, never below `fitScale`.
  - `export function swipeStep(dx: number, dy: number, ms: number): 1 | -1 | 0`: a step when `|dx| ≥ 50`, `|dx| > 2·|dy|` and `ms ≤ 600`; a swipe left is `1` (next).
  - `export function viewerKey(e: KeyboardEvent): 'prev' | 'next' | 'zoomIn' | 'zoomOut' | 'fit' | 'actual' | 'fullscreen' | 'seekBack' | 'seekOn' | null`: `ArrowLeft`/`ArrowRight` step, `Shift`+arrow seeks 5 s, `+`/`=`, `-`, `0`, `1`, `f`. A modifier other than Shift is never the viewer's.
  - `export function clampPan(pan: {x: number; y: number}, scale: number, box: {w: number; h: number}, media: {w: number; h: number}): {x: number; y: number}`.
- `MediaViewer.svelte` props: `{ items: MediaFile[]; open: string | null (bindable); actions?: Snippet<[MediaFile]>; aside?: Snippet<[MediaFile]>; onkey?: (e: KeyboardEvent, file: MediaFile) => boolean }`.
  - It is a `Modal` with `fill`, labelled by the open file's leaf. Escape and the **Close** button set `open = null`.
  - The stage fits the file to the window (no scroll at fit); the aside is a 22rem column beside the stage at ≥ 900 px and below it under that.
  - Its top bar: the leaf, `<i> of <n>`, **Previous file** / **Next file**, the zoom buttons, **Full screen** (hidden when `document.fullscreenEnabled` is false), then the page's `actions`, then **Close**.
  - Previous and next walk `items` through `neighbor()` from the file's last known place, so a file that left `items` steps on from where it stood.
  - A drag pans when zoomed; a double click toggles fit and actual size; a swipe steps when at fit.
  - Keys go through one `keydown` on the sheet in the capture phase: when the target is editable (`isEditable`) or `overlayOpen()` counts a layer above the viewer's own, nothing happens; else `viewerKey`, then the page's `onkey`. A handled key is `preventDefault`ed, so a focused video never seeks on an arrow.
  - `open` naming no item renders nothing and calls nothing; the page shows the notice (Task 6).

- [ ] **Step 1: Write the failing tests.**
  - `mediaViewer.test.ts`: `zoom steps from fit and stops at 8`, `never zooms out past fit`, `a short flat swipe steps, a slow or steep one does not`, `the arrows step and Shift-arrows seek`, `a pan is clamped to the picture's edges`.
  - `MediaViewer.test.ts`:
    - `opens on the named file and says where it is` (`2 of 3`)
    - `Next and Previous walk the items, and stop at the ends`
    - `arrows step after a click on the video`
    - `keys typed in a field or under a dialog do nothing`
    - `a tall image fits with its controls visible` (the img is bounded by the stage, and **Close** is in the document's first screen)
    - `Escape closes, and the page's onkey sees S`
- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/mediaViewer.test.ts src/lib/ui/MediaViewer.test.ts src/lib/ui/Modal.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the tests, `npx vitest run src/lib/ui/`, and the metrics gate. Expected: PASS. `raw_media` does not rise (MediaBody is an owner).
- [ ] **Step 5: Commit** `ui(media): MediaViewer, fitted to the window, with zoom, steps, swipe and full screen`.

### Task 5: `ReferenceChip` and `RecipePanel`

GA-8 (the recipe hidden and forgotten), D2 and survey §15.

**Files:**
- Create: `ui/src/lib/ui/ReferenceChip.svelte`, `ui/src/lib/ui/RecipePanel.svelte`, `ui/src/lib/recipe.ts`, `ui/src/lib/recipe.test.ts`, `ui/src/lib/ui/RecipePanel.test.ts`, `ui/src/lib/ui/ReferenceChip.test.ts`
- Modify: `ui/src/lib/outputMeta.ts` (`describeOutputMeta` keeps its shape; `recipe.ts` builds on it)
- Docs: ARCHITECTURE, a new UI row *What made this*: owners `RecipePanel.svelte` (its own read of `GET /api/gallery/{name}/metadata`, then `GET /api/jobs/{id}` when nothing is embedded), `recipe.ts`, `ReferenceChip.svelte`; rule "every surface that says what made a file says it through RecipePanel, open, with every argument; a reference is built by `references.ts`, never spelled"; enforced by the three test files and `prefix_literals`

**Interfaces:**
- `ReferenceChip.svelte` props: `{ prefix: typeof OUTPUT | typeof ASSET; name: string }`. It shows `<code>` of `reference(prefix, name)` and a `CopyButton` with the Global constraints title.
- `recipe.ts`:
  - `export type RecipeRow = { key: 'workflow' | 'job' | 'step' | 'model' | 'seed' | 'prompt' | 'negative' | 'size' | 'length' | 'argument'; label: string; value: string; copy?: boolean; href?: string }`
  - `export function recipeRows(meta: GalleryMetadata, job: JobDetail | null, workspace: string): RecipeRow[]`, in the Global constraints order. An image's rows come from its embedded `metadata`; with none, from `job.arguments` and `job.workflow_name`. `prompt`, `negative` and `seed` carry `copy: true`. `size` reads `media.width`/`media.height`, `length` `media.duration_seconds`. Each argument not already a row is one `argument` row, its value through `compare.ts`'s `valueText`. `job` links `jobHref(job.id, workspace)` in the file's own workspace (#1600 item 7).
  - `export function savedFrom(meta: GalleryMetadata): string` (`Saved from <file name> v<n>`, or `''`), replacing `AssetDetail`'s `keptFrom`.
- `RecipePanel.svelte` props: `{ name: string; source: 'output' | 'asset'; workspace: string; children?: Snippet<[GalleryMetadata]> }`.
  - It reads the metadata as a `resource` (`watch: ['gallery']` for an output, `['assets']` for an asset; `workspace: () => workspace`), and the job only when `metadata` is null and `job` is known.
  - It renders the `What made this` heading, the rows (a `.formgrid`), **Open as workflow** when the image embeds its workflow (the behaviour `GalleryDetail.openAsWorkflow` has today, moved here), `Made from` with a `ReferenceChip` and a link per input, `Location`, the empty copy, then `children(meta)`, so a page adds its own lines (notes, findings) to one read.
  - Nothing in it is collapsed.

- [ ] **Step 1: Write the failing tests.**
  - `recipe.test.ts`: `an image's rows come from its embedded recipe, in order`, `a clip's rows come from its job's arguments`, `every argument is listed once`, `the job links its own workspace`, `an asset saved from a job says so`.
  - `RecipePanel.test.ts`: `shows the recipe open, with copy buttons on the prompt and seed`, `with no recipe and no job says so once`, `reads the job only when nothing is embedded`, `children see the same read`.
  - `ReferenceChip.test.ts`: `copies output:<name>`, `copies asset:<name>`.
- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/recipe.test.ts src/lib/ui/RecipePanel.test.ts src/lib/ui/ReferenceChip.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the tests, `npx vitest run src/lib/freshness.test.ts` (RecipePanel's resource is declared), and the metrics gate. Expected: PASS.
- [ ] **Step 5: Commit** `ui(media): RecipePanel and ReferenceChip, what made a file, open and copyable`.

### Task 6: The Gallery opens files in the viewer

GA-6, GA-7, GA-15, D2's GalleryDetail; the open file in the URL. Folds #1600 item 5.

**Files:**
- Create: `ui/src/lib/gallery/GalleryAside.svelte` (the note editors, `GalleryFindings`, `FileAgentNotes`, under `RecipePanel`'s `children`)
- Delete: `ui/src/lib/gallery/GalleryDetail.svelte`; its test cases move to `GalleryPage.test.ts` and `GalleryAside.test.ts`
- Modify: `ui/src/lib/pages/GalleryPage.svelte`, `ui/src/lib/gallery/saveAsAsset.ts` (#1600 item 5)
- Test: `ui/src/lib/pages/GalleryPage.test.ts`, `ui/src/lib/gallery/GalleryAside.test.ts`, `ui/src/lib/gallery/saveAsAsset.test.ts`, `ui/e2e/gallery-findings.spec.ts` (selectors only)
- Docs: the *A page's place* row adds `open` (`hashState('open', '')`): the open file is in the hash, written in place like a filter, so Escape closes the viewer and Back leaves the page

**Interfaces:**
- Consumes: Tasks 3-5.
- `GalleryPage`:
  - `const open = hashState('open', '')`, bound to `MediaViewer`'s `open` (empty string ↔ `null`); `items` is `visible` through `outputMedia`.
  - The viewer's `actions(file)`: Star (`aria-pressed`, the existing `starring` action), **Save as asset**, **Use as input…** (Task 7 adds it; leave the snippet's place), `ReferenceChip` (`OUTPUT`), **Assess** (as today), a `DownloadLink`, delete (the existing `removing` action and `fileConfirm`).
  - Its `aside(file)`: `RecipePanel` (`source="output"`) with `GalleryAside` as its children.
  - Its `onkey`: S stars, Delete or Cmd+Backspace deletes. `GalleryPage`'s own `cullKey`, `step` and `selected` go; `selectedAt` becomes the viewer's.
  - When the open file leaves `visible` (deleted here or elsewhere, filtered, unstarred under Starred), the viewer steps to the neighbour from its last place; with nothing left, it closes.
  - An `open` naming no listed file, once the listing has loaded, shows the notice in Global constraints (an `.hint` line under the header, dismissible) and clears `open`.
  - A workspace change clears `open`.
- `saveAsAsset(file, existing)` changes to `saveAsAsset(file, listNames: () => Promise<string[]>)`: the library is listed only after a 409, and a failed listing leaves `assetLibrary.items` as it was (#1600 item 5).

- [ ] **Step 1: Write the failing tests.**
  - `GalleryPage.test.ts`:
    - `clicking a tile opens the viewer and writes open= to the hash`
    - `a hash with open= opens that file`
    - `an open= naming a missing file says so`
    - `the viewer steps on when its file is deleted elsewhere`
    - `S stars the open file and Delete asks, then steps on`
    - `switching workspace closes the viewer`
    - the existing cull, star-undo, note and findings cases, re-pointed at the viewer
  - `GalleryAside.test.ts`: `a file inheriting its job's note shows that one, and Note this file swaps in its own`.
  - `saveAsAsset.test.ts`: `lists the library only after the name is taken`, `a failed listing leaves the shared list alone`.
- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/pages/GalleryPage.test.ts src/lib/gallery/`. Expected: FAIL.
- [ ] **Step 3: Implement.** `GalleryPage` ends below 504.
- [ ] **Step 4: Run** the tests, `npx playwright test e2e/gallery-findings.spec.ts e2e/journeys/consistency.spec.ts`, and the metrics gate. Expected: PASS; `raw_media` 20; `duplicate_blocks` lower (GalleryDetail's three runs go). Lower both in `baseline.json`.
- [ ] **Step 5: Commit** `ui(gallery): files open in the viewer, with the recipe open beside them`.

### Task 7: Use as input

GA-10's reuse path, the half phase 3 left.

**Files:**
- Create: `ui/src/lib/gallery/useAsInput.ts`, `ui/src/lib/gallery/useAsInput.test.ts`, `ui/src/lib/gallery/UseAsInput.svelte`
- Modify: `ui/src/lib/mediaArguments.ts` (export `kindByName` as `mediaKindByName`), `ui/src/lib/pages/GalleryPage.svelte` (the viewer action), `ui/src/lib/pages/WorkflowPage.svelte` (no change to `takePrefill`'s contract; a test pins it)
- Test: `ui/src/lib/pages/WorkflowPage.test.ts`

**Interfaces:**
- `useAsInput.ts`:
  - `export function inputTargets(cards: Record<string, WorkflowCard>, kind: MediaKind): { workflow: string; variable: string }[]`: the workspace's workflows (not the examples) with a `variable_names` entry whose `mediaKindByName` is `kind`, the first such variable each, sorted by name.
  - `export function useAsInput(target: { workflow: string; variable: string }, file: string): void`: writes `WORKFLOW_PREFILL_KEY` as `{ workflow, arguments: { [variable]: reference(OUTPUT, file) } }` and goes to the workflow's page.
- `UseAsInput.svelte` props `{ file: MediaFile }`: **Use as input…** for an image or a clip; it opens a `Modal` with the Global constraints copy and one button per target. The catalog comes from `api.listWorkflows()` through a `resource` read when the dialog opens (`lazy`).

- [ ] **Step 1: Write the failing tests.**
  - `useAsInput.test.ts`: `offers workflows with a variable of the file's kind, first variable each`, `offers none for audio`, `writes the prefill and goes to the workflow`.
  - `WorkflowPage.test.ts`: `opens with an output: reference filled in from Use as input`.
- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/gallery/useAsInput.test.ts src/lib/pages/WorkflowPage.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the tests, `npx vitest run src/lib/freshness.test.ts`, and the metrics gate. Expected: PASS.
- [ ] **Step 5: Commit** `ui(gallery): Use as input opens a workflow with the file filled in`.

### Task 8: Finding a file in the Gallery

GA-9 (search by path only; no sort or date; no "N of M") and GA-5 (folder collapse does nothing). Folds #1600 items 4 and 6.

**Files:**
- Modify: `ui/src/lib/gallery/order.ts` (`gridFiles`, `dateGroup`), `ui/src/lib/pages/GalleryPage.svelte`, `ui/src/lib/FolderGroups.svelte`, `ui/src/lib/ui/PageHeader.svelte` (`count.of`), `ui/src/lib/router.svelte.ts` (`codec.list` writes a plain comma; `codec.oneOf`)
- Test: `ui/src/lib/gallery/order.test.ts`, `ui/src/lib/pages/GalleryPage.test.ts`, `ui/src/lib/FolderGroups.test.ts`, `ui/src/lib/router.svelte.test.ts`, `ui/src/lib/ui/PageHeader.test.ts`

**Interfaces:**
- Consumes: Task 2's `prompts`.
- `gridFiles(files, query: { text: string; roles: OutputRole[]; starredOnly: boolean; sort: 'newest' | 'oldest' | 'name'; prompts: Record<string, string> }): GalleryFile[]`. The text matches, case-insensitively and word by word (every word must match somewhere), the name, the label, the note and the run's prompt. `newest` keeps today's order (finals first inside a run), `oldest` reverses the runs, `name` sorts by name.
- `export function dateGroup(mtime: number, now: Date): 'Today' | 'Yesterday' | 'This week' | 'Earlier'` (local days; "This week" is the 7 days before yesterday).
- The Gallery reads its listing with `{ prompts: true }`. Two `hashState`s: `sort` (`codec.oneOf(['newest','oldest','name'])`, default `newest`) and `group` (`codec.oneOf(['folder','date'])`, default `folder`), each a `Segmented` in the header's actions. `group = date` passes `FolderGroups` a `groupOf` of `dateGroup`.
- **GA-5**: `filterActive` no longer counts the roles at their default (`['final']`), so folders collapse; it counts the text, Starred, the request, and roles other than the default.
- `FolderGroups`: while `filterActive`, the heading's toggle is disabled with the Global constraints title (#1600 item 4).
- `PageHeader`'s `count` gains `of?: number`: `<n> of <m> <nouns>` when given and different.
- `codec.list` formats with a literal `,` (no `%2C`); `codec.oneOf(values: readonly string[])` parses an unknown value as the initial one, and `roles` parses through a filter of the known roles, an empty list as the default (#1600 item 6).

- [ ] **Step 1: Write the failing tests.**
  - `order.test.ts`: `matches a word of the prompt, the note or the name`, `needs every word`, `sorts oldest and by name`, `dates fall into Today, Yesterday, This week and Earlier`.
  - `GalleryPage.test.ts`: `collapsing a folder hides it with the default roles` (GA-5), `searching a prompt finds its file`, `the count says n of m while filtered`, `sort and group survive Back`.
  - `FolderGroups.test.ts`: `the toggle is disabled and says why while filtering`.
  - `router.svelte.test.ts`: `a list is written with a plain comma`, `an unknown value reads as the initial one`.
  - `PageHeader.test.ts`: `renders n of m`.
- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/gallery/order.test.ts src/lib/pages/GalleryPage.test.ts src/lib/FolderGroups.test.ts src/lib/router.svelte.test.ts src/lib/ui/PageHeader.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement.** Logic goes in `order.ts`; `GalleryPage` stays below 504.
- [ ] **Step 4: Run** the tests, `npx playwright test e2e/journeys/consistency.spec.ts`, and the metrics gate. Expected: PASS.
- [ ] **Step 5: Commit** `ui(gallery): search prompts and notes, sort, group by date, and folders that collapse`.

### Task 9: Star a selection

GA-11's bulk star.

**Files:**
- Modify: `ui/src/lib/BulkBar.svelte` (`onstar`), `ui/src/lib/picks.svelte.ts` (`star`), `ui/src/lib/pages/GalleryPage.svelte`
- Test: `ui/src/lib/picks.test.ts`, `ui/src/lib/pages/GalleryPage.test.ts`

**Interfaces:**
- `BulkBar` gains `onstar?: (next: boolean) => void` and `allStarred?: boolean`: **Star** when any picked file is unstarred, else **Unstar**.
- `Picks.star(set: (name: string, next: boolean) => Promise<unknown>, noun: string)` returns `{ busy; run(next: boolean) }`, an `action()` whose `undo` is `undoToast`'s with the Global constraints copy and the Gallery's `STAR_TARGET`; its revert sets each file back to what it was, not to `!next`. Failures go through `actOnEach`/`reportFailures`.
- The Gallery passes the edits through `patch`, so a quiet reload never marks the page's own stars.

- [ ] **Step 1: Write the failing tests.**
  - `picks.test.ts`: `stars every pick once, and Undo puts each back as it was`.
  - `GalleryPage.test.ts`: `stars a selection, offers Undo, and marks nothing changed`, `a double click on Star sends one round`.
- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/picks.test.ts src/lib/pages/GalleryPage.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the tests and the metrics gate (`unguarded_mutations` 0). Expected: PASS.
- [ ] **Step 5: Commit** `ui(gallery): star or unstar a selection, with Undo`.

### Task 10: Assets open in the viewer

GA-L *asset detail*, D2's AssetDetail.

**Files:**
- Delete: `ui/src/lib/assets/AssetDetail.svelte` (its cases move to `AssetsPage.test.ts`)
- Create: `ui/src/lib/assets/AssetAside.svelte` (the note editor and `Used by`, under `RecipePanel`'s `children`)
- Modify: `ui/src/lib/pages/AssetsPage.svelte`
- Test: `ui/src/lib/pages/AssetsPage.test.ts`

**Interfaces:**
- Consumes: Tasks 3-5.
- `AssetsPage` keeps the open asset in `hashState('open', '')`, its viewer walking the open section's assets; `actions`: `ReferenceChip` (`ASSET`), a download (`DownloadLink` on the asset's `url`, `download` attribute), delete (the existing action; hidden for a read-only `examples` asset, with `read-only` shown); `aside`: `RecipePanel` (`source="asset"`), which says `Saved from …`, and `AssetAside`.

- [ ] **Step 1: Write the failing tests** in `AssetsPage.test.ts`: `an asset opens in the viewer with its reference and a download`, `a read-only asset offers no delete`, `says which job it was saved from`, `open= survives Back`.
- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/pages/AssetsPage.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement.** `AssetsPage` ends below 521.
- [ ] **Step 4: Run** the tests, `npx playwright test e2e/journeys/editor-safety.spec.ts` (uploads), and the metrics gate. Expected: PASS; `raw_media` 17. Lower `raw_media` and `duplicate_blocks` in `baseline.json`.
- [ ] **Step 5: Commit** `ui(assets): assets open in the viewer, with their reference, download and origin`.

### Task 11: Tiles everywhere else; the Overview links the file

SH-5, SH-12, GA-L *picker gaps*, and the rest of D1.

**Files:**
- Modify: `ui/src/lib/pages/OverviewPage.svelte:157-176,212-232,272-290`, `ui/src/lib/pages/WorkflowsPage.svelte:264-284` (and its `.cardframe` CSS), `ui/src/lib/pages/WorkflowPage.svelte:291-314`, `ui/src/lib/assets/MediaPicker.svelte:205-232`, `ui/src/lib/request/ItemProof.svelte`
- Modify: `ui/src/lib/routes.ts` (`fileHref`), `ui/src/lib/proofs.ts` (`proofByRun`)
- Test: `ui/src/lib/proofs.test.ts`, `ui/src/lib/pages/OverviewPage.test.ts`, `ui/src/lib/pages/WorkflowsPage.test.ts, `ui/src/lib/pages/WorkflowPage.test.ts`, `ui/src/lib/assets/MediaPicker.test.ts`, `ui/src/lib/routes.test.ts`, `ui/src/lib/request/RequestItem.test.ts`

**Interfaces:**
- `proofs.ts`: `export function proofByRun(files: GalleryFile[]): Record<string, GalleryFile>`: each run id's first image, else its first clip, by `latestProofs`' rule.
- `routes.ts`: `export function fileHref(workspace: string, section: 'gallery' | 'assets', name: string): string` (`#/ws/<ws>/<section>?open=<name>`).
- Overview: the recent strip's tiles are `OutputTile` (`size="strip"`) built with the listing's own `workspace` (SH-5) and linking `fileHref` (SH-12). The jobs panel's rows are each one link to the job (the whole row), showing `StatusChip`, the workflow's file name and the job's proof tile (`size="mini"`, through `proofByRun` below, over the gallery listing the page already reads), with `job <id>` as its title instead of the bare id.
- Workflows, Workflow: their proofs are `OutputTile` (`size="card"` and `"strip"`); the workflow page's link `fileHref`. The duplicated `<img>` branches and `.cardframe` go.
- MediaPicker: tiles are `OutputTile` (`size="strip"`), so a clip has its play badge; its own `GLYPHS` is gone (Task 1).
- `ItemProof` keeps its dashed slot for a job with no file and draws a file through `OutputTile` (sizes `strip` for a row, `take` for a take, `mini` for a cover).

- [ ] **Step 1: Write the failing tests.**
  - `OverviewPage.test.ts`: `a recent tile opens that file in the gallery`, `tiles keep the workspace they were listed in after a switch`, `a job row is one link, titled with its id`.
  - `WorkflowPage.test.ts`: `a proof opens its file in the gallery`.
  - `MediaPicker.test.ts`: `a clip shows its play badge`.
  - `routes.test.ts`: `fileHref puts the name in open=`.
  - `proofs.test.ts`: `a run's proof is its first image, else its first clip`.
- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/proofs.test.ts src/lib/pages/OverviewPage.test.ts src/lib/pages/WorkflowsPage.test.ts src/lib/pages/WorkflowPage.test.ts src/lib/assets/MediaPicker.test.ts src/lib/routes.test.ts src/lib/request/`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the tests, `npx playwright test e2e/smoke.spec.ts e2e/requests.spec.ts e2e/responsive.spec.ts`, and the metrics gate. Expected: PASS; `raw_media` 9. Lower it in `baseline.json`.
- [ ] **Step 5: Commit** `ui(media): every tile is an OutputTile; overview tiles open their file`.

### Task 12: The job page shows its results first, and opens them in the viewer

JB-2 (results below the fold), JB-15 (layout leftovers), JB-16 (missing actions). Folds #1600 item 3.

**Files:**
- Modify: `ui/src/lib/pages/JobPage.svelte`, `ui/src/lib/job/JobResults.svelte`, `ui/src/lib/job/JobProgress.svelte`
- Test: `ui/src/lib/pages/JobPage.test.ts`, `ui/src/lib/job/JobResults.test.ts`

**Interfaces:**
- Consumes: Tasks 3-5, 9's star action shape.
- The finished job's order: header, the error summary (Task 13), **Results**, the note, the workflow, then **Progress** (on a finished job a one-line summary: `<k> of <n> steps` and the time taken, with the list behind a disclosure), the log, the event log. A running job keeps Progress first.
- JB-15: the note sits in its own `.panel`; the JSON toggle sits in the Workflow head beside `realized` / `as submitted`, not on a row of its own.
- `JobResults`: each output is an `OutputTile` (`size="card"`, `caption="line"`) in a grid per step, with one line of recipe under it (model, seed); clicking opens `MediaViewer` over the job's outputs, with `open` in the job route's hash (`hashState('open', '')`), `actions`: Star, **Save as asset**, **Use as input…**, `ReferenceChip`, download, and "Note this file" when the page gives `onnote`; `aside`: `RecipePanel`. `runMetadataReader` stays for the one-line recipe.
- The Results head: **Compare** (opens Task 15's compare with the first 4 picture or clip outputs, finals first) and **Download all** (`api.archiveOutputs` with the job's files).
- "(run root)" reads `(top folder)`.

- [ ] **Step 1: Write the failing tests.**
  - `JobPage.test.ts`: `a finished job shows Results before the workflow and Progress`, `a running job keeps Progress first`, `a finished job's progress is one line until opened`.
  - `JobResults.test.ts`: `an output opens in the viewer with its recipe`, `Compare takes the first four pictures, finals first`, `Download all archives every file`, `a folder at the job's top reads (top folder)`.
  - The existing cases that read `img[src]` beside metadata are re-pointed at the tiles and the viewer.
- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/pages/JobPage.test.ts src/lib/job/JobResults.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement.** `JobPage` stays under 600; `JobResults` shrinks.
- [ ] **Step 4: Run** the tests, `npx playwright test e2e/job-event-log.spec.ts e2e/chain-segments.spec.ts`, and the metrics gate. Expected: PASS; `raw_media` 6. Lower it and `duplicate_blocks` in `baseline.json`.
- [ ] **Step 5: Commit** `ui(job): results first, as tiles that open in the viewer, with Compare and Download all`.

### Task 13: A failed job says what failed and what to do; the seed survives a restart

JB-3 (errors buried, not actionable; failed step not red) and the rest of JB-10.

**Files:**
- Create: `ui/src/lib/job/jobError.ts`, `ui/src/lib/job/jobError.test.ts`, `ui/src/lib/job/JobError.svelte`
- Modify: `ui/src/lib/pages/JobPage.svelte` (the error panel at `:411-417` goes; `failedNode` and `seed` fall back)
- Test: `ui/src/lib/pages/JobPage.test.ts`

**Interfaces:**
- `jobError.ts`: `export function explainJobError(error: string): { title: string; remedy: string | null; action: 'rerun' | 'models' | null }`. Patterns: `/out of memory|OutOfMemoryError|CUDA out of memory|MPS backend out of memory/i` → OOM; `/divisible by 8|multiple of 8/i` → size; `/GatedRepoError|gated repo|403 Client Error.*huggingface/i` → gated; else the fallback. Copy as in Global constraints.
- `JobError.svelte` props `{ job: JobDetail; step: string | null; onrerun: () => void }`: an `error-edge` panel with `role="alert"`: `Failed in step <step>.` when known, the title, the remedy, the action button (**Run again with changes…** opens `JobHeader`'s existing rerun dialog through `onrerun`; **Open Models** links `#/server/models`), then `Details` (`Engine detail`) with `job.error` and `job.traceback`.
- `failedNode`: from the events as now, else `job.progress?.step` (a job read from history has no events), so the failed step is red after a restart.
- `seed`: from `workflow_start` as now, else `job.arguments[seedVariable]` when it is a number. JB-10's stale workflow is pinned by a test, with no code change.

- [ ] **Step 1: Write the failing tests.**
  - `jobError.test.ts`: one case per pattern, and `an unknown error has no remedy or action`.
  - `JobPage.test.ts`: `a failed job leads with what failed and the remedy, the traceback folded`, `the failed step is red on a job read from history`, `the seed shows on a job read from history`, `the workflow updates when the job finishes` (JB-10, passes today).
- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/job/jobError.test.ts src/lib/pages/JobPage.test.ts`. Expected: FAIL (the last case PASS).
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the tests and the metrics gate. Expected: PASS.
- [ ] **Step 5: Commit** `ui(job): a failed job leads with the step, the reason and the remedy`.

### Task 14: Job rows you can tell apart

JB-4 (rows indistinguishable; filter).

**Files:**
- Modify: `ui/src/lib/pages/JobsPage.svelte`, `ui/src/lib/jobsListing.ts`
- Test: `ui/src/lib/pages/JobsPage.test.ts`, `ui/src/lib/jobsListing.test.ts`

**Interfaces:**
- Consumes: Task 2's `JobList.prompts`, Task 3's `OutputTile`, Task 11's `proofByRun`.
- `listJobs` asks with `{ prompts: true }`; the live resource keeps `rows`, `queueSize` and `prompts`. A second resource, `proofs` (`watch: ['gallery']`), reads `api.gallery()` and maps through `proofByRun`.
- A running row also reads its `api.getJob(id)` in the live load (one per running job) for `progress`, shown as `<n>%` of `denoise_step / denoise_total_steps`.
- Each row: the proof (`OutputTile size="mini"`, or the status chip's slot when none), the workflow's file name, the first line of its prompt (muted, one line, ellipsis), its note's first line, the chips as today.
- The filter's placeholder becomes `search names, prompts, notes and ids…`, and it matches, word by word, the file name, the id, the note and the prompt.

- [ ] **Step 1: Write the failing tests.**
  - `JobsPage.test.ts`: `each row shows its picture, prompt and note`, `searching a prompt or an id finds the job`, `a running row shows its percentage`.
- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/pages/JobsPage.test.ts src/lib/jobsListing.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement.** `JobsPage` stays under 600; row markup over 30 lines moves to `ui/src/lib/job/JobRow.svelte`.
- [ ] **Step 4: Run** the tests, `npx playwright test e2e/restart-banner.spec.ts`, and the metrics gate. Expected: PASS.
- [ ] **Step 5: Commit** `ui(jobs): each row shows its picture, prompt and note, and search finds them`.

### Task 15: Compare

GA-16 (the URL), GA-L *Compare tools*. Folds #1600 items 1 and 2.

**Files:**
- Modify: `ui/src/lib/pages/ComparePage.svelte`, `ui/src/lib/routes.ts` (`compareHref`), `ui/src/lib/pages/GalleryPage.svelte` (BulkBar's `oncompare`)
- Test: `ui/src/lib/pages/ComparePage.test.ts`, `ui/src/lib/routes.test.ts`

**Interfaces:**
- `compareHref(workspace: string, names: string[]): string` is the one builder: each name one path segment, `encodeURIComponent`ed. `parseView` compares `rest` as the decoded array, never a `join('/')` (#1600 item 2). The Gallery, the job page (Task 12) and takes (Task 16) all use it.
- Under 2 names the page renders the empty copy at once, with no read and no skeleton; a failed gallery read with 2 or more shows `LoadState`'s error (#1600 item 1).
- Each cell draws through `MediaBody` (`onvideo` registers the transport's video), is a button that opens `MediaViewer` over the compared files (`open` in the hash), and has **Remove from comparison** (`aria-label`), which `replaceRoute`s to `compareHref` without it.

- [ ] **Step 1: Write the failing tests** in `ComparePage.test.ts`: `with fewer than two names shows the empty copy without loading`, `a name holding a slash survives the round trip`, `removing a cell rewrites the URL`, `a cell opens in the viewer`, `one transport plays every clip`; and `routes.test.ts`: `compareHref round-trips names with slashes`.
- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/pages/ComparePage.test.ts src/lib/routes.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the tests and the metrics gate. Expected: PASS; `raw_media` 4 (Suggest 1, MediaArgumentInput 2, ChainSegments 1). Lower it in `baseline.json`.
- [ ] **Step 5: Commit** `ui(compare): one URL form, cells that open and remove, no skeleton when empty`.

### Task 16: Takes are big enough to judge

RQ-5 (takes too small to compare).

**Files:**
- Modify: `ui/src/lib/request/AttemptStrip.svelte`, `ui/src/lib/request/RequestItem.svelte` (stays under 600), `ui/src/lib/pages/RequestPage.svelte`
- Test: `ui/src/lib/request/RequestItem.test.ts`, `ui/src/lib/pages/RequestPage.test.ts`

**Interfaces:**
- Consumes: Tasks 3, 4, 5, 15.
- A take's tile is `size="take"` (240 px, Task 3) and a button: it opens `MediaViewer` over the item's takes that have a file, oldest to newest, with `open` in the request route's hash. The viewer's `actions(file)`: **Accept** / **Accepted** (the existing `onaccept`, unchanged behaviour, no star), **Open the job** (`jobHref`); `aside`: `RecipePanel`.
- The take's job link moves to its caption (`take <n>` links the job).
- **Compare takes** in the strip's head when 2 or more takes have a picture or clip: `compareHref` with the newest 4.

- [ ] **Step 1: Write the failing tests** in `RequestItem.test.ts`: `a take opens in the viewer, and Accept there accepts it`, `Compare takes opens the newest four`, `a take with no file stays a slot and does not open`.
- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/request/ src/lib/pages/RequestPage.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the tests, `npx playwright test e2e/requests.spec.ts`, and the metrics gate. Expected: PASS.
- [ ] **Step 5: Commit** `ui(requests): takes open large, with Accept, and compare side by side`.

### Task 17: The event log reads like sentences and follows the job

JB-14 (raw JSON rows; the log doesn't follow; a step's ETA labelled as the job's).

**Files:**
- Modify: `ui/src/lib/job/eventLog.ts` (`eventSummary`), `ui/src/lib/job/EventLog.svelte`, `ui/src/lib/job/JobProgress.svelte:50`
- Test: `ui/src/lib/job/eventLog.test.ts`, `ui/src/lib/job/JobProgress.test.ts`, `ui/src/lib/pages/JobPage.test.ts`

**Interfaces:**
- `eventSummary(event: JobEvent): { text: string; data: string | null }`: the sentences in Global constraints; `data` is the remaining payload as indented JSON, or null. `EventLog` renders `text`, and `data` behind `Details` (`Event data`).
- `EventLog` gains **Follow new events** (a `button.togglechip` with `aria-pressed`, on by default for a live job): while on, a new row scrolls into view; a user's scroll up turns it off.
- `JobProgress`: `about <duration> left in this step`.

- [ ] **Step 1: Write the failing tests.** `eventLog.test.ts`: one case per named event, and `any other event shows its name, the data folded`. `JobProgress.test.ts`: `the estimate names the step`. `JobPage.test.ts`: `the event log follows a live job until the user scrolls up`.
- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/job/ src/lib/pages/JobPage.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the tests, `npx playwright test e2e/job-event-log.spec.ts`, and the metrics gate (`banned_terms` 0). Expected: PASS.
- [ ] **Step 5: Commit** `ui(job): the event log in sentences, following the job; the estimate names its step`.

### Task 18: Prompt cards show their names

GA-L *prompt cards*.

**Files:**
- Modify: `ui/src/lib/pages/PromptsPage.svelte:88-130` and its CSS
- Test: `ui/src/lib/pages/PromptsPage.test.ts`

**Interfaces:**
- The card's name has the card's full width and wraps to at most two lines (`line-clamp: 2`, `overflow-wrap: anywhere`), titled with the full name; the model chip moves to the start of the tag row.

- [ ] **Step 1: Write the failing test** `the name has its own line, and the model chip sits with the tags`.
- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/pages/PromptsPage.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the test, `npx playwright test e2e/responsive.spec.ts`, and the metrics gate. Expected: PASS.
- [ ] **Step 5: Commit** `ui(prompts): a prompt card's name is never cut by its model chip`.

### Task 19: The journey, the rules for the next agent, and the phase gate

**Files:**
- Create: `ui/e2e/journeys/find-and-reuse.spec.ts`
- Modify: `ui/e2e/serve_fixture.py` (media to find), `ui/GLOSSARY.md` (*Before you write markup*), `ui/CLAUDE.md` (rewrite, no new line), `docs/ARCHITECTURE.md` (*UI architecture ratchet* names `raw_media`), `docs/stabilization/ui/baseline.json`

**Interfaces:**
- The fixture adds, with the standard library only (`zlib`, `struct`, `wave`):
  - `e2e-find/20260910-120000-0f1d0f1d/lighthouse.png`: 400×1600, solid colour, with a `parameters` `tEXt` chunk before `IDAT` holding `{"prompt": "a lighthouse at dusk", "seed": 4242, "model_name": "e2e/tiny", "arguments": {"prompt": "a lighthouse at dusk", "seed": 4242, "steps": 12}}`;
  - a second file in that run, `harbour.png`, 64×64, no recipe;
  - `e2e-find/20260910-120000-0f1d0f1d/hum.wav`, 2 s of silence at 8 kHz, and `notes.txt` with three lines;
  - a history row for job `e2efind00001`, status `succeeded`, `arguments` `{"prompt": "a lighthouse at dusk"}`, run id `20260910-120000-0f1d0f1d`.
- `GLOSSARY.md`'s table gains rows: a file shown small (`OutputTile`, `media.ts`), a file shown large (`MediaViewer`, with the page's `open` `hashState`), what made a file (`RecipePanel`), a reference to copy (`ReferenceChip`), a file to reuse (`UseAsInput`).
- `ui/CLAUDE.md` stays at 12 lines: bullet 2's "Show the proof: a list of things that produce images shows the images." becomes "Show the proof, one way: a file is an `OutputTile`, opened in `MediaViewer` (the page's `open` hash), and what made it is a `RecipePanel` (`raw_media` holds the rest at 4)." That is paid for by cutting bullet 2's "(`unguarded_mutations` holds it at 0)" and "(ESLint holds it)", which the ARCHITECTURE rows already say.

- [ ] **Step 1: Write the journey** `find-and-reuse.spec.ts`:
  1. In the Gallery, type `lighthouse dusk`: one tile is left and the count reads `1 of <m> files`.
  2. Open it: the URL holds `open=`, the image fits the window with **Close** visible, and `What made this` shows the prompt and `Seed 4242` without a click.
  3. **Zoom in**, then **Fit to window**; press → and ←: the position line changes and comes back.
  4. Copy the reference: the clipboard holds `output:e2e-find/20260910-120000-0f1d0f1d/lighthouse.png`.
  5. **Save as asset** under `lighthouse-ref`: `Saved as asset lighthouse-ref`; on Assets, its tile is a thumbnail (no `<video>`, no original `src`), and it opens with `Saved from`.
  6. Back in the Gallery, tick two files, **Star**, then **Undo**: neither is starred.
  7. Tick two, **Compare**, remove one: the URL has one name and the empty copy shows.
  8. On the Overview, the recent tile for `lighthouse.png` opens it in the Gallery viewer.
  9. Open job `e2efind00001`: **Results** comes before the workflow, and its tile opens the same viewer.
  10. The `hum.wav` tile shows its duration and the `notes.txt` tile its first line.
- [ ] **Step 2: Run** `cd ui && npx playwright test e2e/journeys/find-and-reuse.spec.ts`. Expected: PASS. A failure names the task that left a gap; fix it here.
- [ ] **Step 3: Write the rules** as in Interfaces.
- [ ] **Step 4: Run** `npm run check`, `npm run lint`, `npm run build`, `npm test`, `npx playwright test`, the metrics gate, `npm run metrics -- --check ../docs/stabilization/ui/baseline.json` (only the two pre-existing lines may print), `venv/bin/python scripts/arch_metrics.py` (`modules` 281, `claude_md_lines` 129), `wc -l dw/server/api_models.py` (≤ 1105), and `venv/bin/python -m pytest tests/test_ui_twins.py tests/test_api_contract.py -q`. Expected: PASS. `baseline.json` reads `raw_media 4` and `duplicate_blocks` at its measured value, below 30.
- [ ] **Step 5: Commit** `ui(media): the find-and-reuse journey, and the rules for the next agent`.

---

## Phase exit

- `raw_media` is 4 in `docs/stabilization/ui/baseline.json` (Suggest, MediaArgumentInput ×2, ChainSegments), and the kind glyphs are spelled only in `media.ts`.
- `window_dialogs`, `unguarded_mutations`, `banned_terms`, `raw_error_toasts`, `empty_outside_loadstate` and `unwatched_reads` are 0; `duplicate_blocks` is below 30; no other metric is above `$SCRATCH/ui-start.json`.
- `modules` is 281, `claude_md_lines` 129, `api_models.py` at most 1105 lines, and no page over 600.
- `GET /api/gallery` and `GET /api/jobs` answer as before unless asked for `prompts`.
- ARCHITECTURE has the rows *Media tiles*, *Media viewer* and *What made this*.
- `find-and-reuse.spec.ts` holds the phase.
