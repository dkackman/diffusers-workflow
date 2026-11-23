# Svelte duplication audit: ui/src (develop)

Read from `/Users/don/src/dkackman/diffusers-workflow/ui/src` on `develop`. The spec's planned pieces are taken from
`.worktrees/ui-quality-phase-1/docs/superpowers/specs/2026-11-22-ui-quality-design.md`, in the table "Architecture: the shared building blocks".
The phase-1 worktree has uncommitted edits to EditorPage, PromptEditorPage, WorkflowFileBar, StepEditor and the router. Line numbers in those files may have moved there.

## Method

- **Exact near-duplicates.** A script normalised whitespace, dropped trivial lines and hashed sliding 6-line windows across all 118 `.svelte` files. It found little copy-paste. The largest exact clone is 33 lines (AssetsPage and ShadowedAssets `.cell`). The codebase is already factored at the copy-paste level.
- **CSS rule blocks.** A second script parsed every `<style>` block into rule bodies (4+ declarations, sorted) and grouped identical bodies across files. There were 24 identical cross-file blocks (listed inline below).
- **Concepts.** Most duplication is *conceptual*: the same idea built 3 to 9 times with drifting props. I inventoried that by grep, for these:
  - thumbnails and tiles
  - toggles and chips
  - label/field grids
  - `.head` rows
  - loaders and `changes().watch`
  - `confirmDialog` and `notify.error`
  - `hashQuery`
  - keyboard handlers
  - status chips
  - tables
  - `--danger` and other tone classes
  - `#/jobs/` links
  - `window.prompt`

Then I read each occurrence.
- **Line estimates.** These are rough: lines that disappear from the call sites, net of the new shared piece.

Legend:
- **UX** = the variations are the same concept behaving or looking differently. That is an inconsistency users see.
- **harmless** = only maintainability.
- **Planned** = one of the spec's building blocks absorbs it.
- **NEW** = no planned piece covers it.

---

## Ranked candidates

### 1. Media tile / thumbnail (9 implementations). Planned: `OutputTile` (UX: yes, includes a perf bug)

| # | Where | Shape |
|---|---|---|
| a | `lib/pages/GalleryPage.svelte:395-425` + CSS `:516-583` | button.cell, aspect 1, radius-2, hover ink, poster+badge, `¶/♪`+label, 4-line clamped caption |
| b | `lib/pages/AssetsPage.svelte:459-479` + CSS `:519-574` | button.cell, height 110px, radius-frame, hover **accent**, **`<video preload="metadata">` per tile**, `♪/¶`, 1-line ellipsis caption |
| c | `lib/assets/ShadowedAssets.svelte:36-48` + CSS `:66-122` | exact clone of (b)'s `.cell`/`.caption`/placeholder CSS (33 lines, flagged by the hash scan) |
| d | `lib/assets/MediaPicker.svelte:205-224` + CSS `:280-337` | button.tile, height 96px, hover accent, thumbnail for video **with no play badge**, own `GLYPHS` map |
| e | `lib/pages/WorkflowsPage.svelte:279-297` + `.cardframe` CSS `:463-476` | img for image, identical img + videobadge for "else" |
| f | `lib/pages/OverviewPage.svelte:261-277` | `a.frame.tile`: image / video+badge / muted label |
| g | `lib/pages/OverviewPage.svelte:376-389` + `.cardframe` CSS `:540-552` | identical to (e). The `.cardframe` block is byte-identical to WorkflowsPage's (12 lines) |
| h | `lib/pages/WorkflowPage.svelte:310-328` | `a.frame.proof`, image / else video+badge |
| i | `lib/request/ItemProof.svelte:54-74` | img + badge if video, glyph `¶/♪` |

Related:
- `ComparePage.svelte:157-178` uses full media, not tiles.
- `ChainSegments.svelte:253` uses a lazy `<video>` thumb.

**What differs (the props needed):**
- `kind`
- `src` / thumb URL
- `href` | `onclick`
- caption mode (none / 1 line / clamped)
- `active` / `current`
- `size` (aspect or height)
- badges (role, version, note, star)
- `alt`

**User-visible inconsistencies:**
- **The assets grid renders a live `<video preload="metadata">` per tile** (`AssetsPage.svelte:467`). GalleryPage's own comment at `:410` says exactly that "hung the page". It also ignores `assetThumbnailUrl`, which MediaPicker and refcomplete already use for the same assets.
- MediaPicker shows a video as a still with no play badge. Everywhere else, the badge is what says "this is a clip".
- Hover and selection colour: accent on assets and the picker, ink on gallery and cards. Radius is `--radius-frame` vs `--radius-2`. Tile heights are 110 / 96 / aspect-1 / 4:3.
- Kind-to-glyph mapping is written 5 times: GalleryPage:423, AssetsPage:471 (inverted condition), ItemProof:73, MediaPicker:46, refcomplete.svelte.ts:19. They map `lut` differently.
- `decoding="async"` appears only on WorkflowsPage. Alt text is `""` / label / name depending on page.
- (e), (g) and (h) duplicate the `<img>` in both branches, where only the badge differs.

**Bypass:** WorkflowsPage and OverviewPage `.cardframe` re-implement `frames.css`'s `.frame` (border, panel-2 ground, object-fit) instead of using it.

**Proposal:** `ui/src/lib/ui/OutputTile.svelte` as planned.
- Props: `{ file: {name, kind, label}, thumb?: string, href?, onclick?, active?, size?: 'square'|'card'|'strip', caption?: 'none'|'line'|'clamp' }`.
- Snippet: `badges`.
- Plus `kindGlyph(kind)` in a `.ts` module (one map).
- Assets must use `assetThumbnailUrl`, never `<video>`.

**Lines removable:** about 250, roughly 110 markup and 140 CSS.

---

### 2. "What made this" recipe + detail panel (3 implementations). Planned: `MediaViewer` + `RecipePanel` (UX: yes)

**Occurrences:**
- `lib/gallery/GalleryDetail.svelte:193-404`
  - bar `:194-271`
  - media body `:273-285`
  - Settings `<details>` with step/model/seed/prompt/negative/job/workflow `:314-366`
  - "Made from" `:368-389`
  - CSS `:407-469`
- `lib/assets/AssetDetail.svelte:51-121`
  - same bar skeleton (selname, flex, "open file", size/mtime, delete, flex, close; the hash scan matched `:82-93` vs GalleryDetail `:263-273`)
  - same sticky `.detail` CSS
- `lib/job/JobResults.svelte`
  - "Made from" `:161-178`, near-identical to GalleryDetail's (hash match)
  - per-file model/seed/prompt/negative `:254-277`
  - `.meta`/`.prompt` CSS `:365-394` identical to GalleryDetail's `.meta`/`.prompt`

**User-visible inconsistencies:**
- An asset's panel has a **copy-reference** button (`asset:`). An output's panel (gallery, job page) has none for `output:`, though both are references a workflow takes.
- Download exists for outputs only. Text files are fetched and shown in the gallery, but only linked from assets.
- Media size is 480px max on the gallery and 60vh max on assets.
- The job page's "Made from" links have no titles, and it omits step and job, which the gallery shows.
- AssetDetail fetches `galleryMetadata` itself (`:33-49`) to say "kept from". GalleryPage fetches it in the page and passes 10 props down.

**Proposal:** the planned `RecipePanel` (`{ metadata, inputs, role, sourceJob }`) used by GalleryDetail, AssetDetail and JobResults, and `MediaViewer` for the bar and body.

**Lines removable:** about 160.

**Over-wide component:** `GalleryDetail` takes **17 props**, because GalleryPage owns the metadata fetch (`GalleryPage.svelte:185-247`: `metadata`, `metadataLoading`, `sourceJob`, `role`, `inputs`, `findings`, `linkedJobs`, `noteInfo`). The fetch belongs inside RecipePanel/MediaViewer through `resource()`.

---

### 3. Label/field form grid (9 copies). NEW: `.formgrid` in app.css (UX: yes, misaligned columns)

| File:line | Label track | Collapse at | Label style |
|---|---|---|---|
| `editor/StepEditor.svelte:409-425` | 170px, field max 480px | 400px | 600 muted |
| `editor/PipelineOptions.svelte:347-376` | **150px** | 400px | 600 muted |
| `editor/ComponentEditor.svelte:132-151` | **120px** | **380px** | 600 muted, pad-top .35 |
| `editor/ArgumentsEditor.svelte:260-287` | 170px + auto | 400px | pad-top .42 |
| `editor/MappingEditor.svelte:95-115` | minmax(140px,40%) + auto | **420px** | 600 muted |
| `editor/VariablesForm.svelte:272-297` | minmax(140px,40%) | 400px | pad-top .45 |
| `editor/EnhancePanel.svelte:129-148` | minmax(120px,40%) | 420px | .85rem |
| `pages/PromptEditorPage.svelte:443-462` | minmax(120px,40%), identical to EnhancePanel (hash match) | 420px | .85rem |
| `editor/WorkflowFileBar.svelte:212-227` | auto | 400px | .85rem |

**User-visible:** StepEditor nests PipelineOptions, which nests ComponentEditor, which nests ArgumentsEditor, with MappingEditor alongside. One step's panel therefore shows label columns at 170, 150, 120, 170 and 40%. Fields do not line up down the page, and they collapse to one column at three different widths.

**Proposal:** an `app.css` class `.formgrid`.
- A `--label-w` custom property, default `minmax(140px, 40%)`.
- `.formgrid > label` gets one style.
- One `@container (max-width: 400px)` collapse rule.
- Modifier `.formgrid.withaction` adds a trailing `auto` column.
- Not a component: the children are heterogeneous inputs.

**Lines removable:** about 100.

---

### 4. Toggle controls: segmented switches and filter chips (7). NEW: `Segmented.svelte` + `.togglechip` (UX: yes)

**Occurrences:**
- `editor/ViewSwitch.svelte` (segmented, selected = accent outline, **no `aria-pressed`**).
- `editor/StepEditor.svelte:205-222` + CSS `:460-480`. A second copy of ViewSwitch's segmented CSS, including `.activebtn` (hash match), **bypassing ViewSwitch**. It also has no `aria-pressed`.
- `pages/WorkflowsPage.svelte:211-246` + CSS `:381-438`. The shape row is selected = **ink fill**; trait chips are ink fill.
- `pages/GalleryPage.svelte:323-345` + CSS `:489-510`. `.rolechip` is the same rule set as WorkflowsPage `.traitchip` (padding 0.6 vs 0.5rem).
- `assets/MediaPicker.svelte:162-179` + `:252-255`. Tabs as `quiet` + `.on` accent border.
- `pages/PromptsPage.svelte:121-147` + `:226-243`. Tag and model chips as filter buttons, a third chip look (panel-2 ground, accent hover).

**What differs:** selected state is accent-outline vs ink-fill; pill vs square; screen readers are told the state on 3 of 6.

**Proposal:**
- `ui/src/lib/ui/Segmented.svelte` with `{ options: {value,label,title?,icon?}[], value, onselect, size?: 'sm'|'md', label }`. It always sets `role="group"` and `aria-pressed`. ViewSwitch becomes a thin wrapper or is deleted. StepEditor and MediaPicker use it.
- A global `button.togglechip` (+ `.on`) in app.css for the multi-select chips (traits, roles, starred, prompt tags).

**Lines removable:** about 90.

---

### 5. Two note editors with different behaviour. NEW: shared draft/keys logic (UX: yes, same concept, different keys)

**Occurrences:**
- `lib/ui/NoteEditor.svelte` (162 lines), used by GalleryDetail, AssetDetail, OverviewPage and JobPage for workspace, asset, run and file notes.
- `lib/request/ItemNote.svelte` (404 lines), for request and item notes.

**Behaviour diff:**

| Behaviour | ItemNote | NoteEditor |
|---|---|---|
| Esc | closes, keeps the draft | none |
| Ctrl/Cmd+Enter | saves | none |
| Draft across a reload | kept (`request/draft.ts`) | none |
| Cancel with a draft | asks before discarding | discards silently |
| Character counter | near 4000 | none (`maxlength` only) |

A person learns Ctrl+Enter on a request note and it does nothing on a file note. A half-written file note is lost on reload.

**Proposal:** lift ItemNote's draft engine and keys into `ui/src/lib/noteDraft.svelte.ts`, re-using `request/draft.ts`'s `readDraft`, `keepDraft` and `draftAtRisk`, and give NoteEditor the same keys. A full merge is not advised: ItemNote's plan/intent coupling is request-specific.

**Lines removable:** about 30 net. The value is consistency.

---

### 6. Request filter chip, re-implemented on the Jobs page. NEW props on the existing `RequestFilter` (UX: yes, bypass)

**Occurrences:**
- `gallery/RequestFilter.svelte`: chip + a picker when none is chosen + hash sync.
- `pages/JobsPage.svelte:248-266`: its own `wschip active-filter` chip with a different label (`requestLabel(...)` vs `request <code>id</code>`).
  - No picker: on Jobs you can only clear a request filter, never choose one.
  - Its own `req` hash plumbing (`:41`, `:47-64`).

**Proposal:** give `RequestFilter` a `workspace` prop (Jobs-all pins it, per `listingWorkspace`) and a `pickable` flag. JobsPage uses it.

**Lines removable:** about 25.

---

### 7. Hash-backed filter state. NEW: `hashState()` in `router.svelte.ts` (UX: yes)

**Occurrences of the same three-part dance** (read `hashQuery()` initially, `$effect(setHashQuery)`, a `hashchange` listener to re-sync):
- `pages/JobsPage.svelte:30-64`
- `pages/RequestsPage.svelte:31-37`
- `gallery/RequestFilter.svelte:20-30`

**User-visible:** only Jobs (status, q, ws, req), Requests (show) and the gallery's `req` survive opening an item and pressing Back. These are lost on Back and not shareable:
- the gallery's text filter, role chips and Starred
- Workflows' shape and traits
- Prompts' and Assets' text filters

**Proposal:** `hashState(key, default, {parse, format})` returns a `$state`-backed box kept in step both ways. Each page then declares its filters in one line. This needs to land after phase 1's router leave-guard changes (`router.svelte.ts` is modified in the worktree).

**Lines removable:** about 25, plus filter persistence on 4 pages for free.

---

### 8. Server status rows. NEW: `ServerHealthList.svelte` (UX: yes)

**Occurrences:**
- `lib/StatusPopover.svelte:33-51`
- `lib/pages/ServerPage.svelte:84-99`

Both have the Worker(s) / Queue / Memory `<dt>/<dd>` rows. The `dl`/`dt` CSS is identical (hash match, StatusPopover `:55-69` vs ServerPage `:357-372`).

**Differences:**
- "unreachable" is `--bad` red in the popover and `--warn` amber on the page.
- With health unknown, the popover shows **both** "unreachable" and "not started — spawns with the first job", because `!health?.worker_alive` is true when health is null. The page guards it.

WorkerList and CardMemoryList are already shared; the rows around them are not.

**Proposal:** `ServerHealthList.svelte` (`{ health, cards, onnavigate? }`) renders the `dl`. Also a global `dl.kv` class: OverviewPage `:555-566` and KeyboardHelp `:38` use the same grid.

**Lines removable:** about 45.

---

### 9. Validate's plan and verdict, shown two ways. NEW: `PlanLines.svelte` (UX: yes, mild)

**Occurrences:**
- `editor/ValidationPanel.svelte:48-60` + `.plan` CSS `:93-107`
- `workflow/RunCheckPanel.svelte:33-42` + `.plan` CSS `:73-86`

The plan list markup and its 9-declaration CSS block are identical (hash match). The verdicts differ: edge colour + icons + "schema-valid, no argument warnings" in the editor, vs tone text + findings on the workflow page, for one `/validate` reply.

**Proposal:** `PlanLines.svelte` (`{ plan }`, calling `describePlan` inside). Consider a shared `Verdict` while there.

**Lines removable:** about 25.

---

### 10. Job status chip (5 sites). NEW: `StatusChip.svelte` (UX: yes)

**Occurrences:**
- `job/JobHeader.svelte:146`
- `pages/JobsPage.svelte:308`
- `pages/OverviewPage.svelte:325` (hard-codes "running") and `:334`
- `request/AttemptStrip.svelte:31-37,80` (maps `cancelled → idle`)
- `request/RunDialog.svelte:193`

**User-visible:** a cancelled job is an amber `chip cancelled` on Jobs, the job header and Overview, but a grey `chip idle` as a request take. It is the same job status with two looks.

**Proposal:** `StatusChip.svelte` (`{ status }`) owns the status-to-tone map. Request *item* chips (`itemState.ts`) stay separate by design (StateTag's comment).

**Lines removable:** about 10. The value is consistency.

---

### 11. Name-input dialog (window.prompt ×2). NEW: `nameDialog()` beside `chooseDialog` in `confirm.svelte.ts` (UX: yes)

**Occurrences:**
- `gallery/GalleryDetail.svelte:98-131`: keep as asset, then on "already exists" a confirm, then retry with replace.
- `pages/AssetsPage.svelte:266-276`: upload names through `window.prompt`. No exists or replace path.

`chooseDialog` (phase 1) is buttons only. Nothing planned takes text.

**Proposal:** `nameDialog(question, { initial, validate: isNameSegment })` returns `string|null`, inside the app's Modal. Pair it with `saveDocument`'s `If-None-Match: *` so both flows get the same exists → replace choice. This also burns down the spec's `window_dialogs` metric.

**Lines removable:** about 30.

---

### 12. Page heads that bypass `.pagehead` (7). Planned: `PageHeader` (UX: yes, small)

**`.pagehead` users:** Prompts, Workflows, Assets, Gallery, Jobs, Requests, Compare.

**Own `.head` CSS:**
- `ServerPage.svelte:337` (no bottom margin)
- `SchemaPage.svelte:63` (no bottom margin)
- `WorkflowPage.svelte:411`
- `EditorPage.svelte:484`
- `PromptEditorPage.svelte:374` (h1 forced to 1.1rem, off-token)
- `ModelsPage.svelte:387` (gap 0.4/1rem)
- `OverviewPage.svelte:449`

Gaps run 0.4/0.6, 0.4/0.8 and 0.4/1rem.

**Back-links in three forms:**
- `EditorPage.svelte:354` (nav.crumbs)
- `PromptEditorPage.svelte:233` ("← prompts")
- `ComparePage.svelte:125` ("back to the gallery")

**Dead CSS:** `.pagehead .flex {flex:1}` (`PromptsPage:172`, `WorkflowsPage:367`) repeats app.css's global `.flex`. `.filter {max-width:220px}` appears 4 times (Prompts, Workflows, Gallery, Assets) and Jobs is a fifth: give `PageHeader` a `filter` slot, or add `.pagehead .filter` to app.css. The count chip is `{n} files` on Gallery and Assets, a bare `{n}` on Workflows, and missing on Prompts. Give `PageHeader` a `count` prop.

**Lines removable:** about 70.

---

### 13. Loading, error and empty, and the loader itself (9 pages). Planned: `resource()` + `LoadState` (UX: yes)

**Loaders** (each about 20-30 lines of `latestOnly()` + `{quiet}` + `.then/.catch` + `$effect(void workspace.current)` + a second `$effect(changes().watch)`):
- `PromptsPage.svelte:22-53`
- `WorkflowsPage.svelte:44-97`
- `AssetsPage.svelte:81-121`
- `GalleryPage.svelte:61-104`
- `RequestsPage.svelte:39-68`
- `JobsPage`
- `ComparePage`
- `RequestPage`
- `OverviewPage.svelte:` 5 loaders, 6 watches

Small effect-loaders with hand-written stale flags:
- `AssetDetail.svelte:33-49`
- `GalleryDetail.svelte:80-95`
- `RequestFilter.svelte:32-44`
- `LorasEditor`

**User-visible drift:**
- **Error copy:** "Could not load prompts:" / "Could not read the asset library:" / "Could not read the gallery:". Only Requests offers Retry.
- **Loading:** Overview, Compare, Requests and RequestPage show Skeleton. Prompts, Workflows, Assets and Gallery show nothing until loaded.
- **Quiet refresh:** a failed quiet refresh re-shows the error on Gallery before first load (`GalleryPage.svelte:88`) but never elsewhere.
- **Filtered-empty:** "Nothing matches "x"." (Prompts, Assets), "Nothing matches those filters." (Workflows), and "No jobs match the filter." + a **clear filters** button (Jobs only). Suggest LoadState grows a `filtered` state with `onclear`.

**Lines removable:** about 300.

---

### 14. Editor save location and toolbar. NEW: generalise `WorkflowFileBar` into `DocumentFileBar` (UX: yes)

**Occurrences:**
- `editor/WorkflowFileBar.svelte`: a collapsible panel with folder, name, description and a path chip.
- `pages/PromptEditorPage.svelte:268-292`: an inline "saving as" bar, with the same FolderPicker, name input and dir hint, plus a ref hint and CopyButton.

The editor heads differ too:

| | Workflow editor | Prompt editor |
|---|---|---|
| Unsaved marker | "unsaved" chip + dirtydot on a *quiet* Save | dirtydot on a *primary* Save |
| Duplicate / Download / Delete | none (they are on WorkflowPage) | in the head |

`VIEWS` arrays are near-identical (`EditorPage.svelte:75-102`, `PromptEditorPage.svelte:53-72`, hash match). The Ctrl/Cmd+S handlers are written twice (`EditorPage:251`, `PromptEditorPage:126`). Neither checks `overlayOpen()`, and neither handles `'S'` with Caps Lock.

**Proposal:**
- `DocumentFileBar` (`{ ed, dir, folders, reference?, extra snippet }`). The workflow's workspace and start-from go in the snippet.
- Export `EDITOR_VIEWS` and an `editorKeys(ed, { save, run? })` from `editorShell.svelte.ts`.

Partly in phase 1's lane (`DocumentEditor`), so co-ordinate with the in-flight edits.

**Lines removable:** about 60.

---

### 15. Copy-a-reference. NEW: `ReferenceChip.svelte` (UX: yes, folds into #2)

**Occurrences:**
- `PromptEditorPage.svelte:283-290`: `<code>prompt:x</code>` + CopyButton.
- `AssetDetail.svelte:53-57`: strong name + CopyButton.

There is none for outputs (GalleryDetail, JobResults) or builtins. `CopyButton` itself is fine and widely used.

**Proposal:** `ReferenceChip` (`{ prefix, name }`) builds the text via `references.ts`, so a prefix is never spelled at the call site. MediaViewer/RecipePanel then use it for `output:`.

**Lines removable:** about 15.

---

### 16. Data-table CSS (6 copies, 4 variants). NEW: `table.datatable` in app.css (harmless, slight visual drift)

**Occurrences:**
- `CachesPanel.svelte:299-320`
- `ServerPage.svelte:408-426` (identical to CachesPanel; both render on the same page)
- `ModelsPage.svelte:396,522-540`
- `ComparePage.svelte:268-284`
- `ChainSegments.svelte:349-365`
- `EventLog.svelte:233-249`

Header size is 0.75rem in some and inherited in others. Row rules are border-top in some and border-bottom in others.

**Proposal:** `.tablewrap` + `table.datatable` (th, td, row rule), with a `.dense` mono modifier for EventLog.

**Lines removable:** about 70.

---

### 17. Tone text classes and a missing token. NEW: global `.tone-good/.tone-warn/.tone-bad` + `.hint` in app.css (UX: one real bug)

**Spread:**
- `.error {color: var(--bad)}` in 10 files
- `.warn` in 7
- `.bad` in 4
- `.hint` in 9, at three sizes (0.75rem / 0.8rem / `--t-sm`)

**Bug:**
- `ModelsPage.svelte:454` uses `color: var(--danger, #c33)`. **`--danger` is not a token**, so the Models page's warnings are a hard-coded #c33 in both themes. Every other `.warn` is amber `--warn`.
- `VariablesForm.svelte:301` uses `var(--danger, var(--muted))`, so its "danger" text is silently muted.

**Lines removable:** about 60.

---

### 18. Panel heading with icon. NEW: `.panel > h2` rule in app.css (harmless)

The same `h2 {display:flex; align-items:center; gap}` appears in 6 files:
- `CachesPanel:283`
- `ServerPage:346` (also forces 0.95rem, off the `--t-md` token)
- `OverviewPage:461`
- `EnhancePanel:121`
- `PromptEditorPage:424`
- `ChainSegments:327`

**Lines removable:** about 25.

---

### 19. Bulk actions on a selectable grid. NEW: helpers in `picks.svelte.ts`, or the planned `action()` (harmless; one behavioural difference)

**Occurrences:**
- `GalleryPage.svelte:133-170`
- `AssetsPage.svelte:208-252`

Both have a `downloadPicked` (identical apart from the API call, hash match) and a `removePicked`: confirm → busy → `actOnEach` → `keepFailed` → "Could not delete X of N" → busy=false. The Escape handlers are also identical: `GalleryPage:300-304`, `AssetsPage:342-351`.

**Difference:** the gallery drops deleted rows locally; assets reloads the whole listing.

**Proposal:** `bulkDelete(picks, del, { confirm, noun })` / `bulkDownload` next to `actOnEach`, or route both through `action({confirm})`. Also a `gridEscape(picks, clearSelected)`.

**Lines removable:** about 40.

---

### 20. Collapsible group headings. NEW: `collapsedState(key)` rune + `GroupHeading.svelte` (harmless, a11y gap)

**Occurrences:**
- `FolderGroups.svelte:29-43,46-68,82-127`: storage-backed collapse plus heading.
- `AssetsPage.svelte:61-65,151-158`: re-implements that collapse persistence for library sections.
- `assets/LibraryRow.svelte`: a second heading look.
- `assets/ShadowedAssets.svelte:24-28,52-63`: copies FolderGroups' heading CSS, with a comment admitting "the same look rather than the same rule".
- `MediaPicker.svelte:207`: a third heading.

None sets `aria-expanded`.

**Lines removable:** about 40.

---

### 21. Disk meter. NEW: `DiskMeter.svelte` (UX: mild)

**Occurrences:**
- `ModelsPage.svelte:176-183` + CSS `:439-452`
- `CachesPanel.svelte:143-153` + CSS `:340-363`

The CSS is identical (hash match). The text differs: "X GB free" vs "X / Y GB free". Both render on server views. Also inline GB maths in `WorkerList.svelte:29` instead of `gbFromMb`.

**Lines removable:** about 25.

---

### 22. List-row CSS. NEW: `.panel.list > .row` in app.css (harmless)

**Occurrences:**
- `JobsPage.svelte:430-477`
- `RequestsPage.svelte:165-197`

Both share `.pagehead h1 {margin:0; flex:1}`, `.list {padding:.3rem}`, `.row` padding/radius/hover and `.namecol`. JobsPage collapses on `@media` and RequestsPage on `@container`, although `app.css`'s responsive convention says page-level layout uses `@media`.

Overview's job list (`OverviewPage.svelte:323-338`) is a third row design: only the id is a link, and the running marker is a hand-rolled `--live` inset (`:514`) instead of the global `.row-live`.

**Lines removable:** about 20.

---

### 23. Legacy `#/jobs/<id>` links. NEW: `jobHref(id, ws?)` in `routes.ts` (UX: minor)

**Occurrences:**
- `WorkerList.svelte:33`
- `GalleryDetail.svelte:345`
- `EditorPage`/`EnhancePanel.svelte:93`
- `App.svelte:257`

Each lands in the *last-used* workspace and is then corrected by `location.replace` in `JobPage.svelte:86-89`: a flash and a double navigation. Everything else uses `wsHref`.

**Lines removable:** 0. Correctness only.

---

## Shared pieces that pages bypass

| Existing piece | Bypassed by | Ref |
|---|---|---|
| `frames.css` `.frame` | WorkflowsPage and OverviewPage `.cardframe` | #1 |
| `RequestFilter` | JobsPage's own chip and hash plumbing | #6 |
| `ViewSwitch` | StepEditor `.modeswitch` | #4 |
| `FolderGroups` collapse | AssetsPage `toggleLibrary`, ShadowedAssets heading | #20 |
| `.pagehead` | 7 pages' own `.head` | #12 |
| `.row-live` | OverviewPage running row | #22 |
| `gbFromMb` | WorkerList | #21 |
| `assetThumbnailUrl` | AssetsPage grid (`<video>` per tile) | #1 |
| global `.flex` | dead `.pagehead .flex` copies in Prompts and Workflows | #12 |

## Over-general or over-wide components

- **Nothing is over-general.** BulkBar, FolderGroups, NoteEditor, ClampedText, Suggest, Modal/Popover/ConfirmDialog and HintBar each have narrow, used props.
- **`GalleryDetail` is over-wide** (17 props) because its parent holds its fetch state; see #2.
- **`ItemNote` is right to stay separate**, but its draft and keys engine should be shared; see #5.

## Totals

About 1,500 lines removable in all (rough):
- **About 800 absorbed by planned pieces:**
  - `OutputTile`: 250
  - `RecipePanel`/`MediaViewer`: 160 (+15 via ReferenceChip)
  - `resource`/`LoadState`: 300
  - `PageHeader`: 70
- **About 700 needing NEW pieces:**
  - formgrid: 100
  - Segmented/togglechip: 90
  - datatable: 70
  - tone classes: 60
  - DocumentFileBar + editor keys: 60
  - ServerHealthList: 45
  - bulk helpers: 40
  - collapse heading: 40
  - nameDialog: 30
  - noteDraft: 30
  - PlanLines: 25
  - RequestFilter props: 25
  - hashState: 25
  - DiskMeter: 25
  - panel h2: 25
  - list rows: 20
  - StatusChip: 10
