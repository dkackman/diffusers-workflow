# Workspace-centric web UI

**Date:** 2026-09-21
**Status:** approved design, awaiting plan

## Problem

The web UI has every piece a workspace holds - workflows, jobs, gallery,
assets, editor - but nothing shows how they relate. The workspace itself is
a `<select>` that appears on three pages, only when more than one workspace
exists; Jobs spans every workspace behind a filter; workspace create/delete
sits at the bottom of the Server page; and the selection lives in
localStorage rather than the URL, so `#/gallery` means "the gallery of
whatever I last picked" and no link can name a workspace.

The rework makes the workspace the landing point and the frame every
per-workspace page sits in, with the things every workspace shares and the
things that belong to the server each given their own place.

## Decisions already made

- Selecting a workspace lands on an **Overview** page, not straight into a
  section.
- The workspace is **part of the URL**. Legacy routes redirect.
- Shared things get **both** a sidebar `Shared` node (the one place to
  manage them) **and** stay visible inline as labelled sections inside a
  workspace's own pages (so what a workflow can reference is visible where
  you compose). Prompts, being one library for every workspace, move to
  `Shared` and leave the per-workspace nav.
- Implementation shape: **the route is the single source of truth** for the
  selected workspace. `workspace.current` is derived from the route;
  localStorage only remembers the last-used workspace for redirects. No
  router library.

## 1. Shell

```
┌──────────┬─────────────────────────────────────────────┐
│ dw       │ header: breadcrumb   ● running   VRAM  ☾  ? │
│          ├─────────────────────────────────────────────┤
│ ▾ studio │                                             │
│  Overview│   content                                   │
│  Workflows                                             │
│  Jobs    │                                             │
│  Gallery │                                             │
│  Assets  │                                             │
│  Editor  │                                             │
│ ▸ default│                                             │
│ ▸ scratch│                                             │
│ + new    │                                             │
│ ──────── │                                             │
│ ▾ Shared │                                             │
│  Prompts │                                             │
│  Assets  │                                             │
│  Examples│                                             │
│ ──────── │                                             │
│ ▾ Server │                                             │
│  Models  │                                             │
│  Schema  │                                             │
│  Status  │                                             │
└──────────┴─────────────────────────────────────────────┘
```

**Sidebar** (`lib/Sidebar.svelte`), three groups in this order:

1. **Workspaces** - one entry per name from `workspace.names`, the
   current one expanded into its sections (Overview, Workflows, Jobs,
   Gallery, Assets, Editor); the others collapsed. Clicking a collapsed
   workspace's name goes to its Overview. Each entry shows its usage
   (`workspace.usage[name].bytes`, `formatBytes`) as a muted `.num`
   figure when the server sends one. `+ new` opens an inline name field
   and calls `POST /api/workspaces`; on success it invalidates the listing
   and navigates to the new workspace's Overview. The workspace name is
   validated client-side against the reserved names (`workflows`,
   `prompts`, `assets`, `outputs`, `exports`, `common`) before the request,
   and the server's 4xx detail is shown verbatim otherwise.
2. **Shared** - Prompts, Assets, Examples.
3. **Server** - Models, Schema, Status.

The sidebar collapses to an icon rail; the collapsed state is persisted
under `dw-sidebar` in localStorage (try/catch, as every other storage read).
Below 900px it is collapsed by default and opens as an overlay from a
hamburger in the header. Keyboard: the rail is a `<nav>` of links; no new
shortcuts.

A single-workspace server gets the same shell. The structure is the point,
and the sidebar is where `+ new` lives.

**Header** keeps what it has (running-job link, VRAM meter, theme, `?`),
drops the top nav, and gains a breadcrumb that mirrors the route:
`studio / gallery`, `shared / prompts`, `server / models`. The brand link
goes to `#/`.

**Design system** - unchanged rules from `ui/CLAUDE.md`: no accent in the
chrome, `--live` only for the running job / VRAM pressure / focus, workspace
and section names in `--font-mono` (they are names the engine resolves),
the selected sidebar entry reads as a heavier ink edge rather than a colour.

## 2. Routes

`router.svelte.ts` keeps its hash-array parse and grows a typed view over
it (`lib/routes.ts`):

```
#/ws/<name>/overview
#/ws/<name>/workflows[/<workflow name>]
#/ws/<name>/jobs[/<job id>]
#/ws/<name>/gallery
#/ws/<name>/assets
#/ws/<name>/edit[/<workflow name>]
#/shared/prompts
#/shared/prompt-edit[/<prompt name>]
#/shared/assets
#/shared/examples[/<workflow name>]
#/server/models
#/server/schema
#/server/status
```

`route.workspace` is the `<name>` for a `ws` route and `null` otherwise.
`workspace.current` becomes a `$derived` of that, falling back to the
last-used name (localStorage `dw-workspace`, as today) for non-`ws` routes -
so a page under `Shared` or `Server` that does call a scoped route (Status's
job files, Examples' proofs) still scopes to something sensible. Navigating
to any `ws` route records that name as last-used. `selectWorkspace()` is
removed; `go()` gains `goWs(section, ...rest)` which fills in the current
workspace, so a page that links to `jobs/<id>` does not spell the prefix.

**Redirects**, applied once at parse time (`location.replace`, so they do
not pollute history):

| From | To |
|------|----|
| `#/`, `#/workflows` | `#/ws/<last>/overview` and `#/ws/<last>/workflows` |
| `#/gallery`, `#/assets`, `#/edit/...`, `#/workflows/<n>` | same section under `<last>` |
| `#/jobs` | `#/server/status` |
| `#/jobs/<id>` | `#/ws/<job's workspace>/jobs/<id>` - the job is fetched first; while it loads the page shows the job page under `<last>`, and corrects the URL when the job answers. An unknown job id stays where it is and the job page reports 404 as it does now |
| `#/prompts`, `#/prompt-edit/...` | `#/shared/...` |
| `#/models`, `#/schema`, `#/server` | `#/server/...` |

A `ws` route naming a workspace the listing does not hold redirects to
`#/ws/default/overview` with a toast naming what was missing - replacing
today's silent fallback in `loadWorkspaces()`. The check runs once the
listing lands; until then the route is trusted, so the first paint is not
delayed by the listing.

## 3. Overview page (new: `pages/OverviewPage.svelte`)

Panels, each a link to its full page, each loading independently so none
waits on another:

- **Recent outputs** - a contact-sheet strip of the newest eight gallery
  entries (`api.gallery()`), in `.frame`s. A workspace that has produced
  nothing shows the empty state, not eight grey plates.
- **Jobs** - the running job with its progress bar if it belongs to this
  workspace, then the last five finished jobs with their status chips
  (`api.listJobs(workspace, limit)`). Links to the workspace's Jobs page.
- **Workflows** - count; the ones with proofs first, up to six cards using
  the catalog's existing proof matching; a `New workflow` link to the
  editor. This is the page's one filled button.
- **Assets and disk** - asset count from `api.listAssets()` (the
  workspace's own library only), usage from `workspace.usage[name]`, and
  the workspace's directory on disk (`workspace.root` joined with the name;
  the root itself for `default`).
- **Manage** - `Delete workspace`, `.quiet`, with the existing
  `confirmDialog` wording (what it holds, that it is not undoable), refused
  for `default` and shown disabled with the reason. On success:
  invalidate the listing, navigate to `#/ws/default/overview`, toast.
  Rename is not an API; it is not offered.

## 4. Existing pages

- `WorkspacePicker.svelte` is deleted; `WorkflowsPage`, `GalleryPage`,
  `AssetsPage` stop mounting it.
- `JobsPage` takes a prop: under a workspace it lists that workspace's jobs
  and shows no workspace filter; under `Server → Status` it is the current
  page - every workspace, with the filter and a workspace column. A job
  row links to `#/ws/<its workspace>/jobs/<id>`.
- `ServerPage` loses the workspace-management block and keeps everything
  else; it is `Server → Status`'s upper half, with the all-workspaces
  `JobsPage` below it.
- `PromptsPage` / `PromptEditorPage` move under `Shared` unchanged.
- `AssetsPage` under a workspace is unchanged (its origin sections are the
  inline half of the sharing decision). Under `Shared → Assets` the same
  component takes `shared` and hides the workspace section, so the page
  shows `common` and `examples` only; `Upload to shared` becomes the
  filled button there.
- `WorkflowsPage` under `Shared → Examples` takes `readOnlyOnly` and lists
  only entries from read-only roots. Opening one goes to `#/shared/examples/<name>`,
  whose run form submits into the *current* workspace (the server already
  confines the run to the examples root and writes outputs to the job's
  workspace); its `Edit` goes to `#/ws/<current>/edit/<name>`, and a save
  from there writes a copy into the workspace, as `PUT /api/workflows`
  already does.
- `EditorPage` and `WorkflowPage` change only their links.

## 5. Out of scope

- Rename workspace (no API).
- Moving a job or an output between workspaces from the UI (`move_job`
  exists over MCP; not this change).
- Any server-side change. Every route the UI needs exists.

## 6. Testing

- **Vitest**: `routes.test.ts` - the parse table above, every redirect
  row, `route.workspace` for each shape, `goWs`. `workspace.svelte.test.ts` -
  `current` derives from the route, falls back to last-used off a `ws`
  route, records last-used on a `ws` route. `Sidebar.test.ts` - expansion
  follows the route, collapse persists, `+ new` validates reserved names
  and navigates on success, unknown workspace toasts and redirects.
  `OverviewPage.test.ts` - each panel renders from a mocked `api`, an
  empty workspace shows empty states, delete is disabled for `default`.
- **Playwright**: switch workspaces through the sidebar and assert the
  gallery request carries the new `?workspace=`; a legacy `#/gallery`
  bookmark lands under the last-used workspace; `#/jobs/<id>` lands under
  the job's workspace.
- **Docs**: `docs/SERVER.md` "The pages" section rewritten around the
  three sidebar groups; `ui/CLAUDE.md`'s scoping paragraph updated (the
  route sets `workspace.current`, no picker); screenshots in `docs/img/`
  regenerated.
