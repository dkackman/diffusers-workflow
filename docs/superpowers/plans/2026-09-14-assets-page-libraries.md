# Assets page: libraries as sections, review fixes, and cleanups

Branch: `assets-page` (over `develop`). Spec: the code review findings in this
session plus the section-by-library design accepted by the user. There is no
separate spec file; the Global Constraints below are the binding text.

## Global Constraints

- Python: never `eval`/`exec`/`shell=True`; every disk read of a request-named
  path goes through `validate_path(path, base)` with a non-None base (see
  CLAUDE.md *Security Rules* - CodeQL models the validators as barriers).
- Tests: Python `python -m pytest tests/test_server.py tests/test_server_downloads.py tests/test_mcp_assets.py -q`
  (run the focused file while iterating, the three once before committing);
  UI `cd ui && npm test` and `npm run check` must both pass before a commit
  that touches `ui/`.
- Every UI string a test asserts is quoted in the task; use it verbatim.
- Existing response fields of `GET /api/assets` (`asset_dir`, `asset_dirs`,
  `assets`, `folders`, and every field of an `assets` entry) keep their names
  and meaning - MCP `list_assets` returns this body verbatim.
- Commit per task, message in the repo's voice (imperative subject, a body
  saying *why*), ending with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Comments explain *why*, in the style of the surrounding code; no narration
  of what the line does.
- Do not touch `ui/src/lib/pages/GalleryPage.svelte` beyond what a task names.

## Task 1: Archive routes - one search probe, one compression policy, one tail

Files: `dw/server/app.py`, `tests/test_server_downloads.py`, `tests/test_server.py`,
`docs/SERVER.md`, `dw_mcp/CLAUDE.md`.

1. **`_asset_in(name, roots)`** (app.py ~2262): stop calling
   `resolve_asset_reference` per root (each call already searches the pinned
   fallbacks, so the loop re-walks them - measured 107 syscalls per hit vs
   29). New body: `validate_asset_reference(name)` once (import from
   `dw.assets` if not already; it raises `SecurityError` subclasses), then for
   each root `candidate = os.path.join(root, name)`; if
   `os.path.isfile(candidate)` return `validate_path(candidate, root)`. On a
   total miss raise `HTTPException(404, detail=f"Unknown asset {name!r}: not found in {', '.join(roots)}")`;
   when `roots` is empty the detail is
   `f"Unknown asset {name!r}: this workspace has no asset library"`. A
   `SecurityError` from validation becomes a 404 with `str(e)` as today.
   Keep the docstring's *why* (bulk callers resolve many names against one
   path). Fix the comment in `archive_assets` (~3075) so it claims only what
   the code does.
2. **`_asset_file(reference, ws)`** stays a one-line wrapper; add a module-level
   helper inside `create_app` named `_resolution_roots(ws)` returning
   `_asset_roots(ws) or [ws.assets]` with a docstring saying why a missing
   library is still named (so the 404 points at the caller's own directory),
   and use it at the three sites (validate route ~1498, `_asset_file`,
   `archive_assets`).
3. **`archive_assets`**: strip each name and dedupe preserving order before
   resolving (`names = list(dict.fromkeys(n.strip() for n in body.names))`);
   the zip entry name is the stripped name. Test: a body
   `["iris.png", "iris.png "]` yields one entry named `iris.png`.
4. **Compression policy**: replace `COMPRESSIBLE_EXTENSIONS` with
   `RAW_MEDIA_EXTENSIONS = {".bmp", ".wav"}` defined directly beneath
   `MEDIA_KINDS` (~2237) with a comment: the allowlist members that are not
   already-compressed containers. In `_zip_download`, an entry is
   `ZIP_STORED` only when `extension in MEDIA_KINDS and extension not in RAW_MEDIA_EXTENSIONS`;
   everything else (`.json`, `.md`, `.txt`, `.bmp`, `.wav`, unknown) is
   `ZIP_DEFLATED`. Tests: (a) `assert RAW_MEDIA_EXTENSIONS <= set(MEDIA_KINDS)`
   - expose both via `app.state` or import; pick whatever the existing test
   for `MEDIA_KINDS` (if any) does, else attach them to `app.state` -; (b) the
   export zip's `workflow.json` entry has `compress_type == zipfile.ZIP_DEFLATED`
   (extend an existing export test in `tests/test_server_exports.py` or
   `test_server_downloads.py`); (c) the existing
   `test_archive_stores_already_compressed_media_rather_than_deflating_it`
   still passes.
5. **One tail**: add `_archive_selection(entries, kind)` beside
   `_zip_download`: it builds the zip via `_zip_download(entries, f"dw-{kind}s-{stamp}.zip")`
   with `stamp = datetime.now().strftime("%Y%m%d-%H%M%S")`, and logs
   `f"Archived {len(entries)} {kind} files"` *after* the archive is written.
   Both archive routes call it (`kind` = `"output"` / `"asset"`), and the
   duplicated three-line tails go. Check the download filename the UI/tests
   expect - keep whatever `dw-outputs-*.zip` / `dw-assets-*.zip` names exist
   today.
6. **Export route** (~3424): restore a plain two-line loop naming `path` and
   `entry` (`entry = os.path.relpath(path, current).replace(os.sep, "/")`,
   append `(f"{job_id}/{entry}", path)`), and restore the dropped "not a
   second permanent copy" comment if git shows it (`git show develop:dw/server/app.py`
   around the old export route).
7. **Docs**: `docs/SERVER.md` `POST /api/assets/archive` paragraph - replace
   the "stored rather than deflated unless `.bmp`, `.wav`" sentence with the
   new rule (media stored, everything else deflated; the export zip's text
   files deflate). `dw_mcp/CLAUDE.md` line 10: "the gallery's bulk zip" ->
   "the two bulk zips (gallery and assets)".

## Task 2: `GET /api/assets` names its libraries and what they shadow

Files: `dw/server/app.py` (`list_assets` ~2906), `tests/test_server.py`,
`docs/SERVER.md`, `docs/MCP.md` (the `list_assets` row), `dw_mcp/assets.py`
docstring, `ui/src/lib/types.ts`.

1. Response gains `libraries`: a list in search-path order, one per root
   `_asset_roots(ws)` returns, `{"origin": <"workspace"|"common"|"examples">, "dir": <abs path>, "writable": <origin != "examples">}`.
   `asset_dirs` stays (it is `[lib["dir"] for lib in libraries]`).
2. Response gains `shadowed`: entries the loop currently skips at
   `if relative in seen: continue`. Each has the same fields as an `assets`
   entry **except `url`** (the URL would serve the shadowing file) plus
   `"shadowed_by": <origin of the root that holds the shadowing file>`.
   `assets` is unchanged - exactly the files `asset:` resolves to.
   Record the shadowing origin as you go (a dict `name -> origin` filled when
   a name is first seen).
3. Tests in `tests/test_server.py` beside `test_the_asset_library_lists_what_it_holds`:
   (a) `libraries` lists the workspace root first with `writable: true`; an
   examples root (see `test_gallery_metadata_finds_an_asset_an_examples_tree_brought`
   for how one is configured) is `writable: false`; (b) a name present in
   both the workspace and an examples library appears once in `assets`
   (origin `workspace`) and once in `shadowed` with `shadowed_by: "workspace"`
   and no `url` key; (c) with no library configured the body has
   `libraries: []` and `shadowed: []`.
4. `ui/src/lib/types.ts`: extend `AssetListing`'s type (find where
   `listAssets` return type is declared - `api.ts` or `types.ts`) with
   `libraries: AssetLibrary[]` and `shadowed: ShadowedAsset[]`;
   `AssetLibrary = { origin: AssetFile['origin']; dir: string; writable: boolean }`,
   `ShadowedAsset = Omit<AssetFile, 'url'> & { shadowed_by: AssetFile['origin'] }`.
   Update `AssetsPage.test.ts`'s `listing` fixture to include
   `libraries: [{ origin: 'workspace', dir: '/ws/assets', writable: true }]`
   and `shadowed: []` so `npm run check` passes; do not change the page.
5. Docs: `docs/SERVER.md` `GET /api/assets` bullet gains one sentence each
   for `libraries` and `shadowed` (why: a client that only sees the
   resolved list cannot show which of its names hide a shared one).
   `docs/MCP.md` `list_assets` row and `dw_mcp/assets.py` docstring mention
   that `shadowed` lists names a nearer library hides.

## Task 3: A bulk bar counts what its actions will touch

Files: `ui/src/lib/picks.svelte.ts`, `ui/src/lib/picks.test.ts`,
`ui/src/lib/BulkBar.svelte`, `ui/src/lib/pages/AssetsPage.svelte`,
`ui/src/lib/pages/GalleryPage.svelte`, their tests, `ui/CLAUDE.md`.

1. `Picks.size` returns `this.names.length` (ticks the filter hides are
   inert: not counted, not acted on, kept until the filter lifts). Delete
   `get hidden()` and its test. Update the doc comment on `size`.
2. `keepFailed(attempted: string[], failed: string[])`: delete every name in
   `attempted` that is not in `failed`; touch nothing else (a hidden tick
   survives). Update both callers (`removePicked` in each page passes
   `names, failed`).
3. `picks.test.ts`: replace the `hidden` test with (a) "size counts only what
   the order includes" - tick two, shrink the order to one, `size === 1`,
   restore the order, `size === 2`; (b) "keepFailed keeps ticks the action
   never attempted".
4. `AssetsPage.svelte` / `GalleryPage.svelte`: `downloadPicked` /
   `removePicked` early-return when `picks.names.length === 0` (the bar is
   hidden then anyway, but a keyboard path could still reach them).
5. `ui/CLAUDE.md` lines ~113-116: rewrite the `Picks.hidden` / open-question
   sentences to state the rule now chosen: the count is the visible
   selection, a hidden tick waits.
6. Run `npm test` and `npm run check`; fix any test that asserted the old
   count.

## Task 4: The assets page sections by library

Files: `ui/src/lib/pages/AssetsPage.svelte`, `ui/src/lib/pages/AssetsPage.test.ts`,
`ui/CLAUDE.md`, `ui/src/lib/api.ts` only if `listAssets` needs its type.

Depends on Task 2's `libraries`/`shadowed` fields and Task 3's `Picks`.

Structure - the search path is the top level, folders inside:

1. Derive `sections` from `libraries` in order: for each library,
   `{ origin, dir, writable, label, assets: visible.filter(a => a.origin === origin), shadowed: listing.shadowed.filter(s => s.origin === origin && matches filter) }`.
   Labels (verbatim): `workspace` -> `This workspace`, `common` ->
   `Shared library`, `examples` -> `Examples`. A library with no assets and
   no shadowed entries still renders its header (so an empty workspace shows
   where an upload would land) unless a filter is active, in which case an
   empty section is skipped.
2. Each section: a header row with the label, `(N)` count, the `dir` in
   `.path` mono, a collapse chevron (state in `storageGet/storageSet` under
   key `collapsed-asset-libraries`, a `Record<origin, boolean>`; while the
   filter is active every section is open, as `FolderGroups` does), and:
   - `writable` and origin `workspace`: an `Upload` button (the one filled
     button on the page).
   - `writable` and origin `common`: a `.quiet` `Upload to shared` button with
     `title="lands in the shared library - visible from every workspace under this root and cannot be moved afterwards"`.
   - not writable: a muted `read-only` span.
   One hidden `<input type="file">`; a `let uploadTarget: 'workspace' | 'shared'`
   set by whichever button was clicked before `fileInput.click()`. The
   `window.prompt` text keeps its shape: `Upload — name in the ${uploadTarget} asset library:`.
3. Inside an open section: `FolderGroups` with `collapseKey="collapsed-asset-folders-${origin}"`,
   the existing card snippet. Then, if the section has shadowed entries, one
   more grid titled with a muted `shadowed/` heading rendered the way
   `FolderGroups` renders a folder name (reuse its markup style, not the
   component): tiles with class `cell shadowed` (opacity 0.45, no checkbox,
   not a button - a `div`), caption = leaf name, `title="shadowed by this workspace's {name} - asset:{name} resolves to that file"`
   where the shadowing origin's label is used in place of "this workspace's"
   when `shadowed_by !== 'workspace'` (`shadowed by the shared library's ...`).
4. Remove: the `origin` state and `<select class="originpick">`, `origins`,
   `borrowed`, `originOffered`, the `uploadTo` select, the `.origin` pill on
   tiles, the footer `read from ... uploads land in ...` paragraph. The head
   keeps `<h1>`, `WorkspacePicker`, a `.num muted` `{assets.length} files`,
   and the filter input. `filterActive` becomes `filter !== ''`.
5. Tiles from a non-writable library render **no checkbox** (nothing bulk
   can do to them). Tiles from `common` keep the checkbox.
6. Delete confirms:
   - single, origin `workspace`: unchanged text.
   - single, origin `common`: `Delete ${reference} from the shared library? Every workspace under this root loses it, and any workflow still carrying that reference stops loading.`
   - bulk: `Delete ${n} asset${s}?` followed, when `shared > 0`, by
     ` ${shared} ${shared === 1 ? 'is' : 'are'} in the shared library and go${shared === 1 ? 'es' : ''} away for every workspace.` then
     ` Any workflow still carrying one of those references stops loading.`
   The detail bar's delete button is hidden for origin `examples` as today.
7. HintBar text (verbatim): `An asset is input a workflow names by reference: an argument set to asset:name loads this file at run time, whatever run produced it. The shared library and any examples library sit on every workspace's search path, so they follow you between workspaces; only this workspace's own section changes with the picker, and a name here hides the same name further down.`
8. Tests (`AssetsPage.test.ts`): rewrite the origin-select / borrowed-count /
   upload-to tests; add: (a) sections render in library order with their
   labels; (b) an examples tile has no checkbox; (c) a shadowed entry renders
   dimmed with its title and no checkbox; (d) clicking `Upload to shared`
   then choosing a file calls `uploadMedia(file, name, true)`; (e) the shared
   delete confirm text; (f) the bulk confirm names the shared count; (g)
   collapsing a library section hides its grid and survives a re-render
   (storage). Keep the tests that still apply (filter, Escape, picks).
9. `ui/CLAUDE.md`: replace the paragraph about `borrowed`/the library pick
   (~118-130) with one on the sections: why the search path is the top
   level and folders are inside it, why upload lives on the section, why a
   shadowed entry is shown at all.

## Task 5: Cleanups the review named

Files: `ui/src/lib/format.ts` (new), `ui/src/lib/format.test.ts` (new),
`ui/src/lib/pages/AssetsPage.svelte`, `ui/src/lib/pages/GalleryPage.svelte`,
`ui/src/lib/pages/ModelsPage.svelte`, `ui/src/lib/pages/ServerPage.svelte`,
`ui/src/app.css`, `ui/src/lib/BulkBar.svelte`.

1. `format.ts`: `formatBytes(size: number): string` - `< 1024*1024` ->
   `Math.max(1, Math.round(size / 1024)) + ' KB'`, else
   `(size / 1024 / 1024).toFixed(1) + ' MB'`, and `>= 1024**3` ->
   `(size / 1024**3).toFixed(2) + ' GB'`; `formatMtime(mtime: number): string`
   -> `new Date(mtime * 1000).toLocaleString()`. Tests for the three byte
   ranges and the boundary at exactly 1 MB.
2. Replace the local `day`/`kb`/`mb`/`size` helpers in the four pages with
   these; if a page's tests assert a specific rendering, update the
   assertion to the new function's output.
3. `app.css`: add `.flex { flex: 1 }` and `button.withicon, .withicon { display: inline-flex; align-items: center; gap: 0.35rem }`
   beside `button.quiet`/`button.bare`, with a comment saying they were
   scoped copies in a dozen components; remove the copies from
   `BulkBar.svelte`, `AssetsPage.svelte`, `GalleryPage.svelte` only (the
   others are out of scope).
4. `app.css` `.cellwrap.picked .cell` -> `.cellwrap.picked > .cell, .cellwrap.picked > .cell:hover`
   is still (0,3,0)+(0,3,0); instead raise specificity with
   `.cellwrap.picked > .cell.cell` (a doubled class is the established trick
   for beating a scoped tie) and a comment explaining that the page's scoped
   `.cell.active`/`.cell:hover` tie with a global rule and win on order.
   Verify visually is not possible; assert in `AssetsPage.test.ts` nothing -
   this is CSS-only.
5. `npm test`, `npm run check`, `npm run lint` clean; commit.
