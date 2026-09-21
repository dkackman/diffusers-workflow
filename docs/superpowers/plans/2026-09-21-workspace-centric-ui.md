# Workspace-centric web UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the workspace the landing point and frame of the web UI: a collapsible sidebar of workspaces (each with Overview, Workflows, Jobs, Gallery, Assets, Editor), a `Shared` group (Prompts, Assets, Examples) and a `Server` group (Models, Schema, Status), with the workspace carried in the URL.

**Architecture:** The hash route is the single source of truth for the selected workspace. `lib/routes.ts` parses `#/ws/<name>/<section>/...`, `#/shared/...`, `#/server/...` into a typed view and maps every legacy hash to its new home; `router.svelte.ts` applies redirects with `location.replace` and, on every route change, sets `workspace.current` from the route (falling back to the last-used name off a `ws` route). `api.ts`'s `scoped()` keeps reading `workspace.current`, so no page's data fetching changes - pages only change the links they build. `App.svelte` becomes sidebar + header + content. No server changes.

**Tech Stack:** Svelte 5 runes, TypeScript, Vitest + @testing-library/svelte, Playwright, lucide icons. Commands run from `ui/`: `npm test`, `npm run check`, `npm run lint`, `npm run build`, `npx playwright test` (starts its own server on 8971 - do not have `dw.serve` on that port).

**Spec:** `docs/superpowers/specs/2026-09-21-workspace-centric-ui-design.md`

## Global Constraints

- No server-side change. Every route the UI needs exists (`GET/POST/DELETE /api/workspaces`, `?workspace=` on scoped routes, `GET /api/jobs?workspace=&limit=`).
- Design system (`ui/CLAUDE.md`): no accent colour in chrome; `--live` only for the running job, VRAM pressure and focus; names the engine resolves (workspace names, section names, workflow names) in `--font-mono`; the selected sidebar entry reads as a heavier ink edge, not a colour; one filled button per view.
- Every `localStorage` read/write is wrapped in try/catch (private mode). Use `storageGet`/`storageSet` from `lib/storage.ts` for JSON values.
- Reserved workspace names: `workflows`, `prompts`, `assets`, `outputs`, `exports`, `common`.
- The default workspace is named `default` (`DEFAULT_WORKSPACE` in `lib/workspace.svelte.ts`); it cannot be deleted.
- Legacy localStorage key `dw-workspace` keeps meaning "last-used workspace" so an existing install lands where it was.
- Commit after each task. Commit message prefix `feat(ui):` / `refactor(ui):` / `test(ui):` / `docs(ui):`, ending with `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.
- Run `npm run check && npm run lint && npm test` before every commit; `npm run build` before the final one (the server serves `ui/dist`, so the UI is invisible until built).

---

## File map

| File | Responsibility |
|------|----------------|
| `ui/src/lib/routes.ts` (new) | Pure: `parseView(parts)` → typed `RouteView`; `legacyRedirect(parts, lastUsed)`; `wsHref`, `sharedHref`, `serverHref` builders; section/group constants |
| `ui/src/lib/routes.test.ts` (new) | Parse and redirect tables |
| `ui/src/lib/router.svelte.ts` (modify) | Applies redirects, exposes `route.view`, calls `applyRouteWorkspace` on change, adds `goWs` |
| `ui/src/lib/workspace.svelte.ts` (modify) | `current` set from the route; `lastUsedWorkspace()` / `rememberWorkspace()`; unknown-workspace redirect + toast in `loadWorkspaces` |
| `ui/src/lib/workspaceActions.ts` (new) | `RESERVED_WORKSPACE_NAMES`, `workspaceNameError`, `createWorkspaceAndGo`, `deleteWorkspaceWithConfirm` - shared by Sidebar and Overview |
| `ui/src/lib/proofs.ts` (new) | `latestProofs(files)` - the newest gallery entry per workflow folder, lifted out of `WorkflowsPage` for the Overview |
| `ui/src/lib/Sidebar.svelte` (new) | The three groups, expansion, collapse persistence, `+ new` |
| `ui/src/lib/Breadcrumb.svelte` (new) | `studio / gallery` from `route.view` |
| `ui/src/lib/pages/OverviewPage.svelte` (new) | Workspace landing page |
| `ui/src/lib/pages/StatusPage.svelte` (new) | `Server → Status`: `ServerPage` content above an all-workspaces `JobsPage` |
| `ui/src/App.svelte` (modify) | Layout, header, route → page switch |
| `ui/src/main.ts` (modify) | No `restoreWorkspace()`; the router sets the workspace at load |
| `ui/src/lib/WorkspacePicker.svelte` (delete) | Replaced by the sidebar |
| Pages (modify links) | `EditorPage`, `PromptsPage`, `PromptEditorPage`, `JobPage`, `JobsPage`, `WorkflowPage`, `WorkflowsPage`, `GalleryPage`, `AssetsPage`, `ServerPage` |
| `ui/e2e/workspaces.spec.ts` (new), `ui/e2e/smoke.spec.ts` (modify) | Legacy-hash landing, sidebar switching, job redirect |
| `docs/SERVER.md`, `ui/CLAUDE.md` (modify) | Pages section around the three groups; scoping paragraph |

---

### Task 1: Route parsing and legacy redirects (`routes.ts`)

**Files:**
- Create: `ui/src/lib/routes.ts`
- Test: `ui/src/lib/routes.test.ts`

**Interfaces:**
- Produces:
  ```ts
  export const WS_SECTIONS = ['overview','workflows','jobs','gallery','assets','edit'] as const
  export type WsSection = (typeof WS_SECTIONS)[number]
  export const SHARED_SECTIONS = ['prompts','prompt-edit','assets','examples'] as const
  export type SharedSection = (typeof SHARED_SECTIONS)[number]
  export const SERVER_SECTIONS = ['models','schema','status'] as const
  export type ServerSection = (typeof SERVER_SECTIONS)[number]
  export type RouteView =
    | { kind: 'ws'; workspace: string; section: WsSection; rest: string[] }
    | { kind: 'shared'; section: SharedSection; rest: string[] }
    | { kind: 'server'; section: ServerSection }
  export function parseView(parts: string[]): RouteView | null   // null = legacy or unknown
  export function legacyRedirect(parts: string[], lastUsed: string): string[] | null
  export function wsHref(workspace: string, section: WsSection, ...rest: string[]): string
  export function sharedHref(section: SharedSection, ...rest: string[]): string
  export function serverHref(section: ServerSection): string
  ```

- [ ] **Step 1: Write the failing tests**

`ui/src/lib/routes.test.ts`:
```ts
import { describe, expect, it } from 'vitest'
import {
  legacyRedirect,
  parseView,
  serverHref,
  sharedHref,
  wsHref,
} from './routes'

describe('parseView', () => {
  it('reads a workspace route with its section and rest', () => {
    expect(parseView(['ws', 'studio', 'jobs', 'abc'])).toEqual({
      kind: 'ws',
      workspace: 'studio',
      section: 'jobs',
      rest: ['abc'],
    })
  })
  it('defaults a bare workspace, or an unknown section, to overview', () => {
    expect(parseView(['ws', 'studio'])).toMatchObject({ section: 'overview' })
    expect(parseView(['ws', 'studio', 'nope'])).toMatchObject({
      section: 'overview',
      rest: [],
    })
  })
  it('keeps a slash-joined name in rest', () => {
    expect(
      parseView(['ws', 'default', 'workflows', 'models', 'z-image']),
    ).toMatchObject({ rest: ['models', 'z-image'] })
  })
  it('reads shared and server routes', () => {
    expect(parseView(['shared', 'prompts'])).toEqual({
      kind: 'shared',
      section: 'prompts',
      rest: [],
    })
    expect(parseView(['server', 'models'])).toEqual({
      kind: 'server',
      section: 'models',
    })
  })
  it('answers null for a legacy or empty hash', () => {
    expect(parseView([])).toBeNull()
    expect(parseView(['gallery'])).toBeNull()
    expect(parseView(['ws'])).toBeNull()
    expect(parseView(['shared', 'nope'])).toBeNull()
    expect(parseView(['server'])).toBeNull()
  })
})

describe('legacyRedirect', () => {
  const last = 'studio'
  it.each([
    [[], ['ws', 'studio', 'overview']],
    [['workflows'], ['ws', 'studio', 'workflows']],
    [['workflows', 'models', 'z'], ['ws', 'studio', 'workflows', 'models', 'z']],
    [['gallery'], ['ws', 'studio', 'gallery']],
    [['assets'], ['ws', 'studio', 'assets']],
    [['edit'], ['ws', 'studio', 'edit']],
    [['edit', 'a', 'b'], ['ws', 'studio', 'edit', 'a', 'b']],
    [['jobs'], ['server', 'status']],
    [['jobs', 'j1'], ['ws', 'studio', 'jobs', 'j1']],
    [['prompts'], ['shared', 'prompts']],
    [['prompt-edit', 'p'], ['shared', 'prompt-edit', 'p']],
    [['models'], ['server', 'models']],
    [['schema'], ['server', 'schema']],
    [['server'], ['server', 'status']],
    [['ws'], ['ws', 'studio', 'overview']],
    [['shared'], ['shared', 'prompts']],
    [['server', 'nope'], ['server', 'status']],
  ])('%j -> %j', (from, to) => {
    expect(legacyRedirect(from, last)).toEqual(to)
  })
  it('leaves a parseable route alone', () => {
    expect(legacyRedirect(['ws', 'x', 'gallery'], last)).toBeNull()
    expect(legacyRedirect(['shared', 'assets'], last)).toBeNull()
  })
  it('sends an unknown top level to the overview', () => {
    expect(legacyRedirect(['whatever'], last)).toEqual([
      'ws',
      'studio',
      'overview',
    ])
  })
})

describe('href builders', () => {
  it('encode every segment', () => {
    expect(wsHref('my ws', 'workflows', 'a/b')).toBe(
      '#/ws/my%20ws/workflows/a%2Fb',
    )
    expect(sharedHref('prompt-edit', 'p')).toBe('#/shared/prompt-edit/p')
    expect(serverHref('models')).toBe('#/server/models')
  })
})
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd ui && npx vitest run src/lib/routes.test.ts`
Expected: FAIL - cannot resolve `./routes`.

- [ ] **Step 3: Implement `routes.ts`**

```ts
/** The shape of the hash: '#/ws/<name>/<section>/...', '#/shared/<section>/...',
 * '#/server/<section>'. Pure - the router applies it. Anything else is a
 * legacy hash (the pre-workspace '#/gallery' and friends) and is mapped to
 * its new home by `legacyRedirect`, so an old bookmark still lands. */

export const WS_SECTIONS = [
  'overview',
  'workflows',
  'jobs',
  'gallery',
  'assets',
  'edit',
] as const
export type WsSection = (typeof WS_SECTIONS)[number]

export const SHARED_SECTIONS = [
  'prompts',
  'prompt-edit',
  'assets',
  'examples',
] as const
export type SharedSection = (typeof SHARED_SECTIONS)[number]

export const SERVER_SECTIONS = ['models', 'schema', 'status'] as const
export type ServerSection = (typeof SERVER_SECTIONS)[number]

export type RouteView =
  | { kind: 'ws'; workspace: string; section: WsSection; rest: string[] }
  | { kind: 'shared'; section: SharedSection; rest: string[] }
  | { kind: 'server'; section: ServerSection }

const isWs = (s: string): s is WsSection =>
  (WS_SECTIONS as readonly string[]).includes(s)
const isShared = (s: string): s is SharedSection =>
  (SHARED_SECTIONS as readonly string[]).includes(s)
const isServer = (s: string): s is ServerSection =>
  (SERVER_SECTIONS as readonly string[]).includes(s)

export function parseView(parts: string[]): RouteView | null {
  const [group, second, third, ...rest] = parts
  if (group === 'ws' && second) {
    // An unknown section is the overview rather than a 404: the workspace
    // is the thing the link names, and the overview is where it starts
    if (third === undefined) {
      return { kind: 'ws', workspace: second, section: 'overview', rest: [] }
    }
    if (isWs(third)) return { kind: 'ws', workspace: second, section: third, rest }
    return { kind: 'ws', workspace: second, section: 'overview', rest: [] }
  }
  if (group === 'shared' && second && isShared(second)) {
    return { kind: 'shared', section: second, rest: third ? [third, ...rest] : [] }
  }
  if (group === 'server' && second && isServer(second)) {
    return { kind: 'server', section: second }
  }
  return null
}

/** Where a hash that `parseView` refuses should go. `lastUsed` is the
 * workspace the scoped legacy routes land in. Null when nothing needs
 * redirecting. */
export function legacyRedirect(
  parts: string[],
  lastUsed: string,
): string[] | null {
  if (parseView(parts)) return null
  const [top, ...rest] = parts
  const ws = (section: WsSection, ...tail: string[]) => [
    'ws',
    lastUsed,
    section,
    ...tail,
  ]
  switch (top) {
    case undefined:
    case '':
    case 'ws':
      return ws('overview')
    case 'workflows':
      return ws('workflows', ...rest)
    case 'gallery':
      return ws('gallery')
    case 'assets':
      return ws('assets')
    case 'edit':
      return ws('edit', ...rest)
    case 'jobs':
      // The queue spans every workspace; one job belongs to one. The job's
      // own workspace is not known until it loads, so the page corrects the
      // URL then (JobPage) - lastUsed is the first guess
      return rest.length ? ws('jobs', ...rest) : ['server', 'status']
    case 'prompts':
      return ['shared', 'prompts']
    case 'prompt-edit':
      return ['shared', 'prompt-edit', ...rest]
    case 'shared':
      return ['shared', 'prompts']
    case 'models':
      return ['server', 'models']
    case 'schema':
      return ['server', 'schema']
    case 'server':
      return ['server', 'status']
    default:
      return ws('overview')
  }
}

const join = (parts: string[]) =>
  '#/' + parts.map(encodeURIComponent).join('/')

export function wsHref(
  workspace: string,
  section: WsSection,
  ...rest: string[]
): string {
  return join(['ws', workspace, section, ...rest])
}
export function sharedHref(section: SharedSection, ...rest: string[]): string {
  return join(['shared', section, ...rest])
}
export function serverHref(section: ServerSection): string {
  return join(['server', section])
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd ui && npx vitest run src/lib/routes.test.ts`
Expected: PASS (all cases).

- [ ] **Step 5: Commit**

```bash
git add ui/src/lib/routes.ts ui/src/lib/routes.test.ts
git commit -m "feat(ui): route table for workspace-centric hashes with legacy redirects"
```

---

### Task 2: Workspace follows the route (`workspace.svelte.ts`, `router.svelte.ts`, `main.ts`)

**Files:**
- Modify: `ui/src/lib/workspace.svelte.ts`
- Modify: `ui/src/lib/router.svelte.ts`
- Modify: `ui/src/main.ts`
- Test: `ui/src/lib/workspace.svelte.test.ts`, `ui/src/lib/router.svelte.test.ts` (new)

**Interfaces:**
- Consumes: `parseView`, `legacyRedirect` from Task 1.
- Produces:
  ```ts
  // workspace.svelte.ts
  export const workspace: { current: string; names: string[] | undefined; root: string | null; usage: Record<string, {files:number;bytes:number}> }
  export function lastUsedWorkspace(): string          // localStorage 'dw-workspace' or 'default'
  export function rememberWorkspace(name: string): void
  export function applyRouteWorkspace(view: RouteView | null): void  // sets workspace.current
  export function loadWorkspaces(): Promise<void>       // unchanged signature; redirects+toasts on an unknown route workspace
  export function invalidateWorkspaces(): void
  // removed: selectWorkspace, restoreWorkspace
  // router.svelte.ts
  export const route: { parts: string[]; view: RouteView }
  export function go(...parts: string[]): void
  export function goWs(section: WsSection, ...rest: string[]): void   // under workspace.current
  ```

- [ ] **Step 1: Write the failing workspace tests**

Replace `ui/src/lib/workspace.svelte.test.ts` with:
```ts
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

const listWorkspaces = vi.hoisted(() =>
  vi.fn(() =>
    Promise.resolve({
      workspace_root: '/ws',
      default: 'default',
      workspaces: [
        { name: 'default', default: true, workflows: '/ws/workflows', assets: null, outputs: '/ws/outputs', prompts: null },
        { name: 'studio', default: false, workflows: '/ws/studio/workflows', assets: null, outputs: '/ws/studio/outputs', prompts: null },
      ],
    }),
  ),
)
const errorToast = vi.hoisted(() => vi.fn())
vi.mock('./api', () => ({ api: { listWorkspaces: () => listWorkspaces() } }))
vi.mock('./toast', () => ({ notify: { error: errorToast, success: vi.fn() } }))

beforeEach(() => {
  listWorkspaces.mockClear()
  errorToast.mockClear()
  localStorage.clear()
  location.hash = ''
})
afterEach(() => vi.resetModules())

it('shares one fetch across concurrent and repeated calls', async () => {
  const { loadWorkspaces, workspace } = await import('./workspace.svelte')
  await Promise.all([loadWorkspaces(), loadWorkspaces()])
  expect(listWorkspaces).toHaveBeenCalledTimes(1)
  expect(workspace.names).toEqual(['default', 'studio'])
})

it('refetches only after invalidateWorkspaces()', async () => {
  const { invalidateWorkspaces, loadWorkspaces } = await import('./workspace.svelte')
  await loadWorkspaces()
  await loadWorkspaces()
  expect(listWorkspaces).toHaveBeenCalledTimes(1)
  invalidateWorkspaces()
  await loadWorkspaces()
  expect(listWorkspaces).toHaveBeenCalledTimes(2)
})

it('a ws route sets current and is remembered as last used', async () => {
  const { applyRouteWorkspace, lastUsedWorkspace, workspace } = await import('./workspace.svelte')
  applyRouteWorkspace({ kind: 'ws', workspace: 'studio', section: 'gallery', rest: [] })
  expect(workspace.current).toBe('studio')
  expect(lastUsedWorkspace()).toBe('studio')
  expect(localStorage.getItem('dw-workspace')).toBe('studio')
})

it('the default workspace clears the stored last-used name', async () => {
  localStorage.setItem('dw-workspace', 'studio')
  const { applyRouteWorkspace, lastUsedWorkspace } = await import('./workspace.svelte')
  applyRouteWorkspace({ kind: 'ws', workspace: 'default', section: 'overview', rest: [] })
  expect(localStorage.getItem('dw-workspace')).toBeNull()
  expect(lastUsedWorkspace()).toBe('default')
})

it('a shared or server route keeps scoping to the last-used workspace', async () => {
  localStorage.setItem('dw-workspace', 'studio')
  const { applyRouteWorkspace, workspace } = await import('./workspace.svelte')
  applyRouteWorkspace({ kind: 'shared', section: 'prompts', rest: [] })
  expect(workspace.current).toBe('studio')
  applyRouteWorkspace({ kind: 'server', section: 'models' })
  expect(workspace.current).toBe('studio')
})

it('an unknown workspace in the route redirects to default and says so', async () => {
  location.hash = '#/ws/gone/gallery'
  // The router is what applies the hash to `workspace.current`
  await import('./router.svelte')
  const { loadWorkspaces } = await import('./workspace.svelte')
  await loadWorkspaces()
  expect(location.hash).toBe('#/ws/default/overview')
  expect(errorToast).toHaveBeenCalledWith(expect.stringContaining('gone'))
})

it('a known workspace in the route is left alone', async () => {
  location.hash = '#/ws/studio/gallery'
  await import('./router.svelte')
  const { loadWorkspaces } = await import('./workspace.svelte')
  await loadWorkspaces()
  expect(location.hash).toBe('#/ws/studio/gallery')
  expect(errorToast).not.toHaveBeenCalled()
})
```

- [ ] **Step 2: Write the failing router tests**

`ui/src/lib/router.svelte.test.ts`:
```ts
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

vi.mock('./api', () => ({ api: { listWorkspaces: vi.fn(() => new Promise(() => {})) } }))
vi.mock('./toast', () => ({ notify: { error: vi.fn(), success: vi.fn() } }))

beforeEach(() => {
  localStorage.clear()
  location.hash = ''
})
afterEach(() => vi.resetModules())

const tick = () => new Promise((r) => setTimeout(r, 0))

it('redirects a legacy hash to the last-used workspace at load', async () => {
  localStorage.setItem('dw-workspace', 'studio')
  location.hash = '#/gallery'
  const { route } = await import('./router.svelte')
  expect(location.hash).toBe('#/ws/studio/gallery')
  expect(route.view).toEqual({ kind: 'ws', workspace: 'studio', section: 'gallery', rest: [] })
})

it('an empty hash lands on the default overview', async () => {
  const { route } = await import('./router.svelte')
  expect(location.hash).toBe('#/ws/default/overview')
  expect(route.view).toMatchObject({ kind: 'ws', workspace: 'default', section: 'overview' })
})

it('a hash change updates the view and the current workspace', async () => {
  const { route } = await import('./router.svelte')
  const { workspace } = await import('./workspace.svelte')
  location.hash = '#/ws/studio/jobs/j1'
  await tick()
  expect(route.view).toMatchObject({ workspace: 'studio', section: 'jobs', rest: ['j1'] })
  expect(workspace.current).toBe('studio')
  location.hash = '#/shared/prompts'
  await tick()
  expect(route.view).toMatchObject({ kind: 'shared', section: 'prompts' })
  expect(workspace.current).toBe('studio')
})

it('goWs builds a link under the current workspace', async () => {
  const { goWs } = await import('./router.svelte')
  location.hash = '#/ws/studio/overview'
  await tick()
  goWs('jobs', 'j1')
  expect(location.hash).toBe('#/ws/studio/jobs/j1')
})
```

- [ ] **Step 3: Run to verify they fail**

Run: `cd ui && npx vitest run src/lib/workspace.svelte.test.ts src/lib/router.svelte.test.ts`
Expected: FAIL - `applyRouteWorkspace`, `lastUsedWorkspace` are not exported; `route.view` undefined.

- [ ] **Step 4: Rewrite `workspace.svelte.ts`**

```ts
import { api } from './api'
import { notify } from './toast'
import type { RouteView } from './routes'

const STORAGE_KEY = 'dw-workspace'
export const DEFAULT_WORKSPACE = 'default'

/** Which workspace the UI is looking at, and what the server offers.
 *
 * The route is the source of truth: `applyRouteWorkspace` (called by the
 * router on every change) sets `current` from a `#/ws/<name>/...` hash, and
 * off one (a shared or server page) leaves it at the last workspace a `ws`
 * route named, so a scoped request from those pages still means something.
 * `api.ts` reads `current` directly to scope every request, so a page only
 * has to read `current` inside its load effect to refetch on a switch.
 * `names` stays undefined until a listing lands, so "not loaded yet" is
 * distinguishable from "only the default exists". */
export const workspace = $state<{
  current: string
  names: string[] | undefined
  root: string | null
  /** Roughly how much disk each workspace holds, by name - the server
   * computes it per listing and caches it briefly, so it is a glance
   * rather than a live figure. Absent for a server that does not send it. */
  usage: Record<string, { files: number; bytes: number }>
}>({ current: DEFAULT_WORKSPACE, names: undefined, root: null, usage: {} })

/** The workspace a bare legacy hash ('#/gallery') lands in: the last one a
 * route named, so a reload or an old bookmark goes back where the user was. */
export function lastUsedWorkspace(): string {
  try {
    return localStorage.getItem(STORAGE_KEY) || DEFAULT_WORKSPACE
  } catch {
    return DEFAULT_WORKSPACE
  }
}

export function rememberWorkspace(name: string): void {
  try {
    if (name === DEFAULT_WORKSPACE) localStorage.removeItem(STORAGE_KEY)
    else localStorage.setItem(STORAGE_KEY, name)
  } catch {
    /* the selection still holds for this session */
  }
}

export function applyRouteWorkspace(view: RouteView | null): void {
  if (view?.kind === 'ws') {
    workspace.current = view.workspace
    rememberWorkspace(view.workspace)
  } else {
    workspace.current = lastUsedWorkspace()
  }
}

// The in-flight/completed listing, shared across every caller - the sidebar
// and any page that needs names call loadWorkspaces() and would otherwise
// refetch a listing that has not changed.
let workspacesPromise: Promise<void> | null = null

export function loadWorkspaces(): Promise<void> {
  if (!workspacesPromise) {
    workspacesPromise = (async () => {
      try {
        const result = await api.listWorkspaces()
        workspace.names = result.workspaces.map((entry) => entry.name)
        workspace.root = result.workspace_root
        workspace.usage = Object.fromEntries(
          result.workspaces
            .filter((entry) => entry.usage)
            .map((entry) => [entry.name, entry.usage!]),
        )
        // A route naming a workspace that has gone away (deleted elsewhere,
        // or a stale bookmark) would scope every request to a 404. Said out
        // loud rather than silently swapped, since the URL was explicit.
        if (!workspace.names.includes(workspace.current)) {
          const missing = workspace.current
          rememberWorkspace(DEFAULT_WORKSPACE)
          workspace.current = DEFAULT_WORKSPACE
          notify.error(`No workspace named ${missing} - showing default`)
          location.replace('#/ws/default/overview')
        }
      } catch {
        workspace.names = [DEFAULT_WORKSPACE]
      }
    })()
  }
  return workspacesPromise
}

/** Drop the cached listing so the next `loadWorkspaces()` refetches - call
 * after a create or delete changes what the server offers. */
export function invalidateWorkspaces(): void {
  workspacesPromise = null
}
```

Note: the unknown-workspace branch runs for a `shared`/`server` route too when `lastUsedWorkspace()` is stale - correct, since every scoped request from those pages would 404; the redirect to the default overview is the right landing either way.

- [ ] **Step 5: Rewrite `router.svelte.ts`**

```ts
import { legacyRedirect, parseView, type RouteView, type WsSection } from './routes'
import { applyRouteWorkspace, lastUsedWorkspace, workspace } from './workspace.svelte'

/** Minimal hash router: '#/ws/studio/jobs/abc' -> parts ['ws','studio','jobs','abc']
 * and a typed `view`. A legacy hash is rewritten in place (no history entry)
 * before it is parsed, so '#/gallery' lands in the last-used workspace. */
function decode(part: string): string {
  // A hand-edited or truncated hash can hold a stray '%', which
  // decodeURIComponent throws on - at module load that would blank the whole
  // app, so an undecodable segment is used as it was typed
  try {
    return decodeURIComponent(part)
  } catch {
    return part
  }
}

function split(): string[] {
  const hash = location.hash.replace(/^#\/?/, '')
  return hash ? hash.split('/').map(decode) : []
}

function parse(): { parts: string[]; view: RouteView } {
  let parts = split()
  const redirect = legacyRedirect(parts, lastUsedWorkspace())
  if (redirect) {
    location.replace('#/' + redirect.map(encodeURIComponent).join('/'))
    parts = redirect
  }
  const view = parseView(parts)!
  applyRouteWorkspace(view)
  return { parts, view }
}

export const route = $state(parse())

window.addEventListener('hashchange', () => {
  const next = parse()
  route.parts = next.parts
  route.view = next.view
})

export function go(...parts: string[]) {
  location.hash = '/' + parts.map(encodeURIComponent).join('/')
}

/** Navigate to a section of the workspace the UI is in. */
export function goWs(section: WsSection, ...rest: string[]) {
  go('ws', workspace.current, section, ...rest)
}
```

`location.replace` on a hash fires `hashchange` too; `parse()` on that second pass finds nothing to redirect. That is one extra parse, not a loop.

- [ ] **Step 6: Update `main.ts`**

Remove the `restoreWorkspace` import and call and the comment above it. The router module (imported by `App.svelte`) sets the workspace at load, before any page effect runs.

- [ ] **Step 7: Run the tests**

Run: `cd ui && npx vitest run src/lib/workspace.svelte.test.ts src/lib/router.svelte.test.ts`
Expected: PASS.

Then `npm run check` - it will fail in `ServerPage.svelte` (imports `selectWorkspace`) and `WorkspacePicker.svelte`. Leave those; Task 3 and Task 5 fix them. Do not commit a red `check` - fold Step 8 into Task 3's commit if you prefer one green commit; otherwise stub: in `ServerPage.svelte` replace `selectWorkspace(DEFAULT_WORKSPACE)` with `go('ws', DEFAULT_WORKSPACE, 'overview')` (import `go` from `../router.svelte`) and in `WorkspacePicker.svelte` replace `selectWorkspace(value)` with `go('ws', value, route.view.kind === 'ws' ? route.view.section : 'overview')` (import `route`, `go`). Both are deleted/reworked later.

- [ ] **Step 8: Commit**

```bash
cd ui && npm run check && npm run lint && npm test
git add ui/src/lib/workspace.svelte.ts ui/src/lib/workspace.svelte.test.ts ui/src/lib/router.svelte.ts ui/src/lib/router.svelte.test.ts ui/src/main.ts ui/src/lib/pages/ServerPage.svelte ui/src/lib/WorkspacePicker.svelte
git commit -m "feat(ui): the route is the source of truth for the selected workspace"
```

---

### Task 3: Shared workspace actions and proofs helper

**Files:**
- Create: `ui/src/lib/workspaceActions.ts`, `ui/src/lib/workspaceActions.test.ts`
- Create: `ui/src/lib/proofs.ts`, `ui/src/lib/proofs.test.ts`
- Modify: `ui/src/lib/pages/WorkflowsPage.svelte` (use `latestProofs`)
- Modify: `ui/src/lib/pages/ServerPage.svelte` (remove the workspaces panel and its script)
- Modify: `ui/src/lib/pages/ServerPage.test.ts` (drop any assertion on the workspaces panel, if present)

**Interfaces:**
- Produces:
  ```ts
  // workspaceActions.ts
  export const RESERVED_WORKSPACE_NAMES: readonly string[]
  export function workspaceNameError(name: string): string | null
  export async function createWorkspaceAndGo(name: string): Promise<string | null>  // null on success (navigated), else message
  export async function deleteWorkspaceWithConfirm(name: string): Promise<boolean>  // true when deleted (navigated to default overview)
  // proofs.ts
  export function latestProofs(files: GalleryFile[]): Record<string, GalleryFile>
  ```

- [ ] **Step 1: Write the failing tests**

`ui/src/lib/workspaceActions.test.ts`:
```ts
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

const createWorkspace = vi.hoisted(() => vi.fn())
const deleteWorkspace = vi.hoisted(() => vi.fn())
const confirm = vi.hoisted(() => vi.fn())
vi.mock('./api', () => ({
  api: {
    createWorkspace: (n: string) => createWorkspace(n),
    deleteWorkspace: (n: string, ack?: boolean) => deleteWorkspace(n, ack),
    listWorkspaces: vi.fn(() =>
      Promise.resolve({ workspace_root: '/ws', default: 'default', workspaces: [{ name: 'default', default: true }] }),
    ),
  },
}))
vi.mock('./confirm.svelte', () => ({ confirmDialog: (...a: unknown[]) => confirm(...a) }))
vi.mock('./toast', () => ({ notify: { error: vi.fn(), success: vi.fn() } }))

beforeEach(() => {
  createWorkspace.mockReset()
  deleteWorkspace.mockReset()
  confirm.mockReset()
  localStorage.clear()
  location.hash = '#/ws/default/overview'
})
afterEach(() => vi.resetModules())

it('refuses reserved and malformed names before asking the server', async () => {
  const { workspaceNameError } = await import('./workspaceActions')
  expect(workspaceNameError('assets')).toMatch(/reserved/)
  expect(workspaceNameError('common')).toMatch(/reserved/)
  expect(workspaceNameError('')).toMatch(/name/)
  expect(workspaceNameError('a/b')).toMatch(/letters/)
  expect(workspaceNameError('studio-2')).toBeNull()
})

it('creates, then navigates to the new overview', async () => {
  createWorkspace.mockResolvedValue({ name: 'studio' })
  const { createWorkspaceAndGo } = await import('./workspaceActions')
  expect(await createWorkspaceAndGo('studio')).toBeNull()
  expect(createWorkspace).toHaveBeenCalledWith('studio')
  expect(location.hash).toBe('#/ws/studio/overview')
})

it('returns the server detail on failure and stays put', async () => {
  createWorkspace.mockRejectedValue(new Error('already exists'))
  const { createWorkspaceAndGo } = await import('./workspaceActions')
  expect(await createWorkspaceAndGo('studio')).toBe('already exists')
  expect(location.hash).toBe('#/ws/default/overview')
})

it('asks with the server detail, then deletes acknowledged and goes to default', async () => {
  deleteWorkspace
    .mockRejectedValueOnce(new Error('holds 3 files'))
    .mockResolvedValueOnce({ name: 'studio', deleted: true })
  confirm.mockResolvedValue(true)
  location.hash = '#/ws/studio/overview'
  const { deleteWorkspaceWithConfirm } = await import('./workspaceActions')
  expect(await deleteWorkspaceWithConfirm('studio')).toBe(true)
  expect(confirm).toHaveBeenCalledWith(expect.stringContaining('holds 3 files'), expect.anything())
  expect(deleteWorkspace).toHaveBeenLastCalledWith('studio', true)
  expect(location.hash).toBe('#/ws/default/overview')
})

it('a declined confirm deletes nothing', async () => {
  deleteWorkspace.mockRejectedValueOnce(new Error('holds 3 files'))
  confirm.mockResolvedValue(false)
  const { deleteWorkspaceWithConfirm } = await import('./workspaceActions')
  expect(await deleteWorkspaceWithConfirm('studio')).toBe(false)
  expect(deleteWorkspace).toHaveBeenCalledTimes(1)
})
```

`ui/src/lib/proofs.test.ts`:
```ts
import { expect, it } from 'vitest'
import { latestProofs } from './proofs'
import type { GalleryFile } from './types'

const file = (name: string, folder: string, kind: GalleryFile['kind']): GalleryFile => ({
  name, folder, subfolder: '', url: '/' + name, kind, size: 1, mtime: 1, label: name,
})

it('keeps the first entry seen per folder, preferring an image over a video', () => {
  const proofs = latestProofs([
    file('a.mp4', 'shot', 'video'),
    file('a.png', 'shot', 'image'),
    file('b.png', 'still', 'image'),
    file('c.png', 'still', 'image'),
    file('loose.png', '', 'image'),
  ])
  expect(proofs.shot.name).toBe('a.png')
  expect(proofs.still.name).toBe('b.png')
  expect(Object.keys(proofs)).toEqual(['shot', 'still'])
})
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd ui && npx vitest run src/lib/workspaceActions.test.ts src/lib/proofs.test.ts`
Expected: FAIL - modules missing.

- [ ] **Step 3: Implement `proofs.ts`** (lifted verbatim from `WorkflowsPage.svelte`'s gallery effect)

```ts
import type { GalleryFile } from './types'

/** The newest output each workflow has produced, by workflow identity. A run
 * writes to <outputs>/<identity>/<run id>/, and the gallery listing reports
 * that identity as an entry's `folder` with the run id already stripped, so
 * a workflow's name matches its folder directly. Entries arrive newest
 * first, so the first one seen for a folder is that workflow's latest.
 * Images win over video because only images have a thumbnail endpoint; a
 * video-only workflow falls back to its video, which renders its first
 * frame. */
export function latestProofs(files: GalleryFile[]): Record<string, GalleryFile> {
  const latest: Record<string, GalleryFile> = {}
  for (const file of files) {
    if (!file.folder) continue
    const held = latest[file.folder]
    if (!held) latest[file.folder] = file
    else if (held.kind !== 'image' && file.kind === 'image')
      latest[file.folder] = file
  }
  return latest
}
```

In `WorkflowsPage.svelte` replace the body of the `.then((result) => { ... proofs = latest })` in the gallery effect with `.then((result) => { proofs = latestProofs(result.files) })` and import `latestProofs` from `'../proofs'`. Delete the now-duplicated comment there.

- [ ] **Step 4: Implement `workspaceActions.ts`**

```ts
import { api } from './api'
import { confirmDialog } from './confirm.svelte'
import { notify } from './toast'
import {
  DEFAULT_WORKSPACE,
  invalidateWorkspaces,
  loadWorkspaces,
} from './workspace.svelte'

/** The names a workspace cannot take: the folders the root itself holds
 * (dw/workspace.py's RESERVED). Checked here so the sidebar can say why
 * before a round trip, but the server has the last word. */
export const RESERVED_WORKSPACE_NAMES = [
  'workflows',
  'prompts',
  'assets',
  'outputs',
  'exports',
  'common',
] as const

const NAME = /^[A-Za-z0-9_][A-Za-z0-9_-]*$/

export function workspaceNameError(name: string): string | null {
  if (!name) return 'A workspace needs a name'
  if ((RESERVED_WORKSPACE_NAMES as readonly string[]).includes(name))
    return `${name} is reserved`
  if (!NAME.test(name))
    return 'Use letters, digits, - and _ only'
  return null
}

/** Create a workspace and land on its overview. Answers the message to show
 * on failure, null on success. */
export async function createWorkspaceAndGo(name: string): Promise<string | null> {
  const invalid = workspaceNameError(name)
  if (invalid) return invalid
  try {
    await api.createWorkspace(name)
  } catch (e) {
    return e instanceof Error ? e.message : String(e)
  }
  invalidateWorkspaces()
  await loadWorkspaces()
  notify.success(`Created workspace ${name}`)
  location.hash = `#/ws/${encodeURIComponent(name)}/overview`
  return null
}

/** Delete a workspace after an informed confirm - the server's unacknowledged
 * refusal says what it holds - and land on the default overview. */
export async function deleteWorkspaceWithConfirm(name: string): Promise<boolean> {
  if (name === DEFAULT_WORKSPACE) return false
  let detail = ''
  try {
    // Unacknowledged first: the server answers with what it would remove,
    // which is what makes the confirmation an informed one
    await api.deleteWorkspace(name)
  } catch (e) {
    detail = e instanceof Error ? e.message : String(e)
  }
  if (
    !(await confirmDialog(`${detail}\n\nDelete workspace "${name}"?`.trim(), {
      confirmLabel: 'Delete',
    }))
  )
    return false
  try {
    await api.deleteWorkspace(name, true)
  } catch (e) {
    notify.error(e instanceof Error ? e.message : String(e))
    return false
  }
  invalidateWorkspaces()
  await loadWorkspaces()
  notify.success(`Deleted workspace ${name}`)
  location.hash = `#/ws/${DEFAULT_WORKSPACE}/overview`
  return true
}
```

- [ ] **Step 5: Strip the workspaces panel from `ServerPage.svelte`**

Remove: the `// ---- workspaces` block (`newWorkspace`, `workspaceError`, `addWorkspace`, `removeWorkspace`), the `<div class="panel"><h2>Workspaces</h2>…</div>` markup, the `.workspaces`, `.workspaces li`, `.workspaces .size`, `.newworkspace` styles, and now-unused imports (`Trash2`, `confirmDialog`, `invalidateWorkspaces`, `loadWorkspaces`, `DEFAULT_WORKSPACE`, `workspace`, `formatBytes`, `go` if Task 2 added it) - keep any still used elsewhere in the file (`notify` is used by the copy buttons; check with `npm run lint`). Remove the `loadWorkspaces()` call in the server effect and its comment. Under the `directories` list, keep the `workspace` directory line as is. Check `ServerPage.test.ts` still passes; delete any assertion about the `Workspaces` heading.

- [ ] **Step 6: Run the tests**

Run: `cd ui && npm run check && npm run lint && npm test`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add ui/src/lib/workspaceActions.ts ui/src/lib/workspaceActions.test.ts ui/src/lib/proofs.ts ui/src/lib/proofs.test.ts ui/src/lib/pages/WorkflowsPage.svelte ui/src/lib/pages/ServerPage.svelte ui/src/lib/pages/ServerPage.test.ts
git commit -m "refactor(ui): workspace create/delete and gallery proofs as shared helpers"
```

---

### Task 4: Sidebar, breadcrumb and the app shell

**Files:**
- Create: `ui/src/lib/Sidebar.svelte`, `ui/src/lib/Sidebar.test.ts`
- Create: `ui/src/lib/Breadcrumb.svelte`
- Modify: `ui/src/App.svelte`
- Delete: `ui/src/lib/WorkspacePicker.svelte`
- Modify: `ui/src/lib/pages/WorkflowsPage.svelte`, `GalleryPage.svelte`, `AssetsPage.svelte` (remove the picker import and `<WorkspacePicker />`)
- Modify: `ui/src/lib/pages/WorkflowsPage.test.ts` (drop the `listWorkspaces` mock line and its comment)

**Interfaces:**
- Consumes: `route`, `go` (Task 2); `workspace`, `loadWorkspaces` (Task 2); `createWorkspaceAndGo`, `workspaceNameError` (Task 3); `wsHref`, `sharedHref`, `serverHref`, `WS_SECTIONS` (Task 1); `formatBytes` (`lib/format.ts`); `storageGet`/`storageSet` (`lib/storage.ts`).
- Produces: `Sidebar` props `{ collapsed: boolean; onToggle: () => void }`; `Breadcrumb` no props. `App.svelte` renders pages by `route.view` - later tasks add `OverviewPage` and `StatusPage` to that switch; until then `overview` renders `WorkflowsPage` and `status` renders `ServerPage` as placeholders.

- [ ] **Step 1: Write the failing sidebar tests**

`ui/src/lib/Sidebar.test.ts`:
```ts
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/svelte'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import Sidebar from './Sidebar.svelte'

const listWorkspaces = vi.hoisted(() =>
  vi.fn(() =>
    Promise.resolve({
      workspace_root: '/ws',
      default: 'default',
      workspaces: [
        { name: 'default', default: true, usage: { files: 2, bytes: 2048 } },
        { name: 'studio', default: false },
      ],
    }),
  ),
)
const createWorkspace = vi.hoisted(() => vi.fn())
vi.mock('./api', () => ({
  api: {
    listWorkspaces: () => listWorkspaces(),
    createWorkspace: (n: string) => createWorkspace(n),
  },
}))
vi.mock('./toast', () => ({ notify: { error: vi.fn(), success: vi.fn() } }))

beforeEach(async () => {
  localStorage.clear()
  location.hash = '#/ws/studio/gallery'
  createWorkspace.mockReset()
  // A fresh router/workspace module per test so the cached listing and the
  // parsed route do not leak between cases
  vi.resetModules()
  await import('./router.svelte')
})
afterEach(cleanup)

it('lists workspaces, expands the current one and marks its section', async () => {
  render(Sidebar, { collapsed: false, onToggle: () => {} })
  await waitFor(() => expect(screen.getByText('studio')).toBeTruthy())
  const gallery = screen.getByRole('link', { name: /gallery/i })
  expect(gallery.getAttribute('href')).toBe('#/ws/studio/gallery')
  expect(gallery.getAttribute('aria-current')).toBe('page')
  // the other workspace is collapsed: its sections are not rendered
  expect(screen.getAllByRole('link', { name: /workflows/i })).toHaveLength(1)
  // usage shows beside a workspace the server measured
  expect(screen.getByText('2.0 KB')).toBeTruthy()
})

it('a collapsed workspace links to its overview', async () => {
  render(Sidebar, { collapsed: false, onToggle: () => {} })
  await waitFor(() => expect(screen.getByText('default')).toBeTruthy())
  expect(screen.getByRole('link', { name: 'default' }).getAttribute('href')).toBe('#/ws/default/overview')
})

it('shared and server groups link to their sections', async () => {
  render(Sidebar, { collapsed: false, onToggle: () => {} })
  expect(screen.getByRole('link', { name: /prompts/i }).getAttribute('href')).toBe('#/shared/prompts')
  expect(screen.getByRole('link', { name: /examples/i }).getAttribute('href')).toBe('#/shared/examples')
  expect(screen.getByRole('link', { name: /models/i }).getAttribute('href')).toBe('#/server/models')
  expect(screen.getByRole('link', { name: /status/i }).getAttribute('href')).toBe('#/server/status')
})

it('+ new refuses a reserved name without calling the server', async () => {
  render(Sidebar, { collapsed: false, onToggle: () => {} })
  await waitFor(() => expect(screen.getByText('studio')).toBeTruthy())
  await fireEvent.click(screen.getByRole('button', { name: /new workspace/i }))
  const input = screen.getByPlaceholderText('name')
  await fireEvent.input(input, { target: { value: 'assets' } })
  await fireEvent.keyDown(input, { key: 'Enter' })
  await waitFor(() => expect(screen.getByText(/reserved/)).toBeTruthy())
  expect(createWorkspace).not.toHaveBeenCalled()
})

it('+ new creates and lands on the new overview', async () => {
  createWorkspace.mockResolvedValue({ name: 'fresh' })
  render(Sidebar, { collapsed: false, onToggle: () => {} })
  await waitFor(() => expect(screen.getByText('studio')).toBeTruthy())
  await fireEvent.click(screen.getByRole('button', { name: /new workspace/i }))
  const input = screen.getByPlaceholderText('name')
  await fireEvent.input(input, { target: { value: 'fresh' } })
  await fireEvent.keyDown(input, { key: 'Enter' })
  await waitFor(() => expect(location.hash).toBe('#/ws/fresh/overview'))
})

it('collapsed shows icons only and the toggle reports', async () => {
  const onToggle = vi.fn()
  render(Sidebar, { collapsed: true, onToggle })
  await fireEvent.click(screen.getByRole('button', { name: /expand sidebar/i }))
  expect(onToggle).toHaveBeenCalled()
  expect(screen.queryByText('studio')).toBeNull()
})
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd ui && npx vitest run src/lib/Sidebar.test.ts`
Expected: FAIL - `Sidebar.svelte` missing.

- [ ] **Step 3: Implement `Sidebar.svelte`**

```svelte
<script lang="ts">
  import { onMount } from 'svelte'
  import {
    Database,
    FolderOpen,
    Images,
    LayoutDashboard,
    Layers,
    ListTodo,
    ListTree,
    MessageSquareText,
    PanelLeftClose,
    PanelLeftOpen,
    Plus,
    Server,
    SquarePen,
    BookCopy,
  } from '@lucide/svelte'
  import { route } from './router.svelte'
  import {
    serverHref,
    sharedHref,
    wsHref,
    type ServerSection,
    type SharedSection,
    type WsSection,
  } from './routes'
  import { loadWorkspaces, workspace } from './workspace.svelte'
  import { createWorkspaceAndGo } from './workspaceActions'
  import { formatBytes } from './format'

  let { collapsed, onToggle }: { collapsed: boolean; onToggle: () => void } =
    $props()

  onMount(loadWorkspaces)

  // The workspace whose sections are open: the route's when it names one,
  // else the one the UI is scoped to (a shared or server page still sits
  // inside a workspace's context)
  const expanded = $derived(
    route.view.kind === 'ws' ? route.view.workspace : workspace.current,
  )
  const wsSection = $derived(
    route.view.kind === 'ws' ? route.view.section : null,
  )
  const sharedSection = $derived(
    route.view.kind === 'shared' ? route.view.section : null,
  )
  const serverSection = $derived(
    route.view.kind === 'server' ? route.view.section : null,
  )

  const WS_ITEMS: { section: WsSection; label: string; icon: typeof Layers }[] = [
    { section: 'overview', label: 'Overview', icon: LayoutDashboard },
    { section: 'workflows', label: 'Workflows', icon: Layers },
    { section: 'jobs', label: 'Jobs', icon: ListTodo },
    { section: 'gallery', label: 'Gallery', icon: Images },
    { section: 'assets', label: 'Assets', icon: FolderOpen },
    { section: 'edit', label: 'Editor', icon: SquarePen },
  ]
  const SHARED_ITEMS: { section: SharedSection; label: string; icon: typeof Layers }[] = [
    { section: 'prompts', label: 'Prompts', icon: MessageSquareText },
    { section: 'assets', label: 'Assets', icon: FolderOpen },
    { section: 'examples', label: 'Examples', icon: BookCopy },
  ]
  const SERVER_ITEMS: { section: ServerSection; label: string; icon: typeof Layers }[] = [
    { section: 'models', label: 'Models', icon: Database },
    { section: 'schema', label: 'Schema', icon: ListTree },
    { section: 'status', label: 'Status', icon: Server },
  ]
  // prompt-edit sits under Prompts in the nav
  const sharedActive = (s: SharedSection) =>
    sharedSection === s || (s === 'prompts' && sharedSection === 'prompt-edit')

  let adding = $state(false)
  let newName = $state('')
  let addError = $state('')
  let nameInput = $state<HTMLInputElement | null>(null)

  function startAdd() {
    adding = true
    addError = ''
    newName = ''
    // the input mounts on the next tick
    queueMicrotask(() => nameInput?.focus())
  }
  async function submitAdd() {
    const failure = await createWorkspaceAndGo(newName.trim())
    if (failure) addError = failure
    else adding = false
  }
</script>

<aside class:collapsed aria-label="navigation">
  <div class="top">
    <a class="brand plain" href={wsHref(workspace.current, 'overview')}
      >dw</a
    >
    <button
      class="bare icon"
      onclick={onToggle}
      title={collapsed ? 'expand sidebar' : 'collapse sidebar'}
      aria-label={collapsed ? 'expand sidebar' : 'collapse sidebar'}
    >
      {#if collapsed}<PanelLeftOpen size={15} />{:else}<PanelLeftClose
          size={15}
        />{/if}
    </button>
  </div>

  <nav>
    <div class="group" aria-label="workspaces">
      {#each workspace.names ?? [workspace.current] as name (name)}
        {#if name === expanded}
          <div class="ws open">
            <span class="wsname" title={name}>
              {#if !collapsed}{name}{/if}
              {#if !collapsed && workspace.usage[name]}
                <span class="num muted size" title="{workspace.usage[name].files} files"
                  >{formatBytes(workspace.usage[name].bytes)}</span
                >
              {/if}
            </span>
            {#each WS_ITEMS as item (item.section)}
              <a
                class="plain item"
                href={wsHref(name, item.section)}
                aria-current={wsSection === item.section ? 'page' : undefined}
                title={item.label}
              >
                <item.icon size={15} />{#if !collapsed}<span class="label"
                    >{item.label}</span
                  >{/if}
              </a>
            {/each}
          </div>
        {:else if !collapsed}
          <a class="plain ws shut" href={wsHref(name, 'overview')} title={name}>
            <span class="wsname">{name}</span>
            {#if workspace.usage[name]}
              <span class="num muted size">{formatBytes(workspace.usage[name].bytes)}</span>
            {/if}
          </a>
        {/if}
      {/each}
      {#if !collapsed && workspace.root}
        {#if adding}
          <form class="newws" onsubmit={(e) => { e.preventDefault(); submitAdd() }}>
            <input
              bind:this={nameInput}
              bind:value={newName}
              placeholder="name"
              aria-label="new workspace name"
              onkeydown={(e) => {
                if (e.key === 'Enter') { e.preventDefault(); submitAdd() }
                if (e.key === 'Escape') adding = false
              }}
            />
            {#if addError}<span class="muted err">{addError}</span>{/if}
          </form>
        {:else}
          <button class="bare item" onclick={startAdd} title="new workspace" aria-label="new workspace">
            <Plus size={15} /><span class="label">new</span>
          </button>
        {/if}
      {/if}
    </div>

    <div class="rule" aria-hidden="true"></div>
    <div class="group" aria-label="shared">
      {#if !collapsed}<span class="grouplabel muted">Shared</span>{/if}
      {#each SHARED_ITEMS as item (item.section)}
        <a class="plain item" href={sharedHref(item.section)} aria-current={sharedActive(item.section) ? 'page' : undefined} title={item.label}>
          <item.icon size={15} />{#if !collapsed}<span class="label">{item.label}</span>{/if}
        </a>
      {/each}
    </div>

    <div class="rule" aria-hidden="true"></div>
    <div class="group" aria-label="server">
      {#if !collapsed}<span class="grouplabel muted">Server</span>{/if}
      {#each SERVER_ITEMS as item (item.section)}
        <a class="plain item" href={serverHref(item.section)} aria-current={serverSection === item.section ? 'page' : undefined} title={item.label}>
          <item.icon size={15} />{#if !collapsed}<span class="label">{item.label}</span>{/if}
        </a>
      {/each}
    </div>
  </nav>
</aside>

<style>
  aside {
    display: flex;
    flex-direction: column;
    width: 200px;
    min-width: 200px;
    border-right: 1px solid var(--line);
    background: var(--panel);
    font-family: var(--font-mono);
    font-size: var(--t-sm);
  }
  aside.collapsed {
    width: 44px;
    min-width: 44px;
  }
  .top {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.55rem 0.6rem;
    border-bottom: 1px solid var(--line);
  }
  aside.collapsed .top { justify-content: center; }
  aside.collapsed .brand { display: none; }
  .brand { font-weight: 600; }
  nav { display: flex; flex-direction: column; padding: var(--space-2) 0; overflow-y: auto; }
  .group { display: flex; flex-direction: column; }
  .grouplabel { padding: var(--space-2) var(--space-4) var(--space-1); font-size: var(--t-xs); }
  .rule { border-top: 1px solid var(--line); margin: var(--space-2) 0; }
  .ws.shut, .wsname {
    display: flex; align-items: center; justify-content: space-between; gap: var(--space-2);
    padding: 0.35rem var(--space-4);
  }
  .ws.open .wsname { font-weight: 600; }
  .size { font-size: var(--t-xs); }
  .item {
    display: flex; align-items: center; gap: 0.5rem;
    padding: 0.35rem var(--space-4) 0.35rem calc(var(--space-4) + 0.6rem);
    color: var(--muted);
    border-left: 2px solid transparent;
    font: inherit;
    width: 100%;
    text-align: left;
  }
  aside.collapsed .item { padding: 0.45rem 0; justify-content: center; }
  .item:hover { color: var(--ink); background: var(--panel-2); }
  /* Selection is the user's state, not the machine's: a heavier ink edge,
     never a colour */
  .item[aria-current='page'] { color: var(--ink); border-left-color: var(--ink); font-weight: 600; }
  .newws { display: flex; flex-direction: column; gap: var(--space-1); padding: 0.2rem var(--space-4); }
  .newws input { width: 100%; font-family: var(--font-mono); font-size: var(--t-sm); }
  .err { font-family: var(--font-sans); font-size: var(--t-xs); }
</style>
```

Check the icons exist in `@lucide/svelte` (`LayoutDashboard`, `PanelLeftClose`, `PanelLeftOpen`, `BookCopy`); if one does not, substitute a sibling (`Gauge`, `ChevronsLeft`, `ChevronsRight`, `Library`) - `npm run check` will say.

- [ ] **Step 4: Implement `Breadcrumb.svelte`**

```svelte
<script lang="ts">
  import { route } from './router.svelte'
  import { serverHref, sharedHref, wsHref } from './routes'

  // group / section, each a link; the section's rest (a workflow or job
  // name) is the page's own h1, not the crumb's
  const crumbs = $derived.by(() => {
    const view = route.view
    if (view.kind === 'ws')
      return [
        { label: view.workspace, href: wsHref(view.workspace, 'overview') },
        { label: view.section, href: wsHref(view.workspace, view.section) },
      ]
    if (view.kind === 'shared')
      return [
        { label: 'shared', href: sharedHref('prompts') },
        { label: view.section, href: sharedHref(view.section) },
      ]
    return [
      { label: 'server', href: serverHref('status') },
      { label: view.section, href: serverHref(view.section) },
    ]
  })
</script>

<nav class="crumbs" aria-label="breadcrumb">
  {#each crumbs as crumb, i (crumb.href)}
    {#if i > 0}<span class="sep muted">/</span>{/if}
    <a class="plain" href={crumb.href}>{crumb.label}</a>
  {/each}
</nav>

<style>
  .crumbs { display: flex; align-items: center; gap: 0.4rem; font-family: var(--font-mono); font-size: var(--t-sm); }
  .crumbs a { color: var(--muted); }
  .crumbs a:last-child { color: var(--ink); }
</style>
```

- [ ] **Step 5: Rework `App.svelte`**

Script: remove the nav icon imports no longer used (`Database`, `FolderOpen`, `Images`, `Layers`, `ListTodo`, `ListTree`, `MessageSquareText`, `Server`, `SquarePen`), add `Menu` from lucide, import `Sidebar`, `Breadcrumb`, `storageGet`/`storageSet`. Add:

```ts
  let sidebarCollapsed = $state(storageGet<boolean>('dw-sidebar', false))
  function toggleSidebar() {
    sidebarCollapsed = !sidebarCollapsed
    storageSet('dw-sidebar', sidebarCollapsed)
  }
  // Narrow viewports: the sidebar is an overlay opened from the header
  let drawerOpen = $state(false)
  $effect(() => {
    // any navigation closes the drawer
    void route.parts
    drawerOpen = false
  })
  const view = $derived(route.view)
  const wide = $derived(
    (view.kind === 'ws' && view.section === 'edit') ||
      (view.kind === 'shared' && view.section === 'prompt-edit'),
  )
```

Markup: replace everything from `<header>` through `</main>` with:

```svelte
<div class="shell" class:drawer={drawerOpen}>
  <Sidebar collapsed={sidebarCollapsed && !drawerOpen} onToggle={toggleSidebar} />
  {#if drawerOpen}
    <button class="scrim" aria-label="close navigation" onclick={() => (drawerOpen = false)}></button>
  {/if}
  <div class="column">
    <header>
      <div class="navrow">
        <button class="bare icon menu" onclick={() => (drawerOpen = !drawerOpen)} aria-label="open navigation" title="navigation">
          <Menu size={15} />
        </button>
        <Breadcrumb />
        <span class="flex"></span>
        <div class="state">
          <!-- unchanged: running link, VRAM meter / status button, StatusPopover,
               token button, theme button, help links -->
        </div>
      </div>
    </header>

    <main class:wide>
      {#if view.kind === 'server'}
        {#if view.section === 'schema'}<SchemaPage />
        {:else if view.section === 'models'}<ModelsPage />
        {:else}<ServerPage />{/if}
      {:else if view.kind === 'shared'}
        {#if view.section === 'prompts'}<PromptsPage />
        {:else if view.section === 'prompt-edit'}<PromptEditorPage name={view.rest.join('/')} />
        {:else if view.section === 'assets'}<AssetsPage />
        {:else if view.rest.length}<WorkflowPage name={view.rest.join('/')} />
        {:else}<WorkflowsPage />{/if}
      {:else if view.section === 'gallery'}<GalleryPage />
      {:else if view.section === 'assets'}<AssetsPage />
      {:else if view.section === 'edit'}<EditorPage name={view.rest.join('/')} />
      {:else if view.section === 'jobs' && view.rest[0]}<JobPage jobId={view.rest[0]} />
      {:else if view.section === 'jobs'}<JobsPage />
      {:else if view.section === 'workflows' && view.rest.length}<WorkflowPage name={view.rest.join('/')} />
      {:else}<WorkflowsPage />{/if}
    </main>
  </div>
</div>
```

(`overview` falls to `WorkflowsPage` until Task 6; `status` to `ServerPage` until Task 7; `shared/assets` and `shared/examples` get their props in Task 5.) Keep `{#key}`-free: the page components already refetch on `workspace.current`.

Styles: replace the `.navrow` nav-link rules (`nav a`, `.navlabel`, `.navrule`, `.second`, `.active`, the 1080px media query on nav) with:

```css
  .shell { display: flex; min-height: 100vh; }
  .column { flex: 1; min-width: 0; display: flex; flex-direction: column; }
  header { border-bottom: 1px solid var(--line); background: var(--panel); position: sticky; top: 0; z-index: 10; }
  .navrow { display: flex; align-items: center; gap: 0.5rem 1rem; padding: 0.55rem 1.2rem; }
  .flex { flex: 1; }
  .menu { display: none; }
  .scrim { display: none; }
  @media (max-width: 900px) {
    .menu { display: inline-flex; }
    .shell :global(aside) { position: fixed; inset: 0 auto 0 0; z-index: 20; transform: translateX(-100%); transition: transform 0.15s ease; }
    .shell.drawer :global(aside) { transform: none; }
    .shell.drawer .scrim { display: block; position: fixed; inset: 0; z-index: 15; background: rgb(0 0 0 / 0.35); border: 0; }
  }
```

Keep the existing `main` / `main.wide` / `.state` / `.meter` / `.live` / `.pulse-dot` rules. Remove the "One row..." comment above the header and replace with one line: `<!-- The sidebar carries navigation; the header carries where you are and what the GPU is doing -->`.

- [ ] **Step 6: Delete the picker**

`git rm ui/src/lib/WorkspacePicker.svelte`. In `WorkflowsPage.svelte`, `GalleryPage.svelte`, `AssetsPage.svelte` remove the import and the `<WorkspacePicker />` element. In `WorkflowsPage.test.ts` remove the `listWorkspaces` mock and the two comment lines above it. In `AssetsPage.svelte`'s HintBar text change "only this workspace's own section changes with the picker" to "only this workspace's own section changes between workspaces".

- [ ] **Step 7: Run everything**

Run: `cd ui && npm run check && npm run lint && npm test`
Expected: PASS. Then `npm run dev` and eyeball at 1200px and 600px: sidebar collapses, drawer opens, breadcrumb reads `default / workflows`.

- [ ] **Step 8: Commit**

```bash
git add -A ui/src
git commit -m "feat(ui): sidebar of workspaces, shared and server groups replaces the top nav"
```

---

### Task 5: Pages link into the new routes; per-page props

**Files:**
- Modify: `ui/src/lib/pages/EditorPage.svelte`, `PromptsPage.svelte`, `PromptEditorPage.svelte`, `JobPage.svelte`, `JobsPage.svelte`, `WorkflowPage.svelte`, `WorkflowsPage.svelte`, `GalleryPage.svelte`, `AssetsPage.svelte`
- Modify: `ui/src/lib/api.ts` (`listJobs` gains `limit`)
- Modify: `ui/src/App.svelte` (pass the new props)
- Test: `ui/src/lib/pages/WorkflowsPage.test.ts`, `ui/src/lib/pages/AssetsPage.test.ts`, `ui/src/lib/pages/JobPage.test.ts`

**Interfaces:**
- Consumes: `goWs`, `route` (Task 2); `wsHref`, `sharedHref` (Task 1).
- Produces:
  - `JobsPage` props `{ scope?: 'workspace' | 'all' }` (default `'workspace'`).
  - `AssetsPage` props `{ shared?: boolean }` (default false).
  - `WorkflowsPage` props `{ examples?: boolean }` (default false).
  - `api.listJobs(workspace?: string, limit?: number)`.

- [ ] **Step 1: Write the failing tests**

Append to `WorkflowsPage.test.ts` (the mock listing gains `sources` and per-workflow `origin`/`writable`):
```ts
it('the examples view lists only read-only workflows and links under shared', async () => {
  listing.details['models/flux-dev'] = { ...listing.details['models/flux-dev'], origin: 'examples', writable: false }
  listing.details['templates/tti'] = { ...listing.details['templates/tti'], origin: 'workspace', writable: true }
  render(WorkflowsPage, { examples: true })
  await waitFor(() => expect(card('flux-dev')).toBeTruthy())
  expect(card('tti')).toBeNull()
  expect(card('flux-dev')!.getAttribute('href')).toBe('#/shared/examples/models/flux-dev')
  expect(screen.queryByTitle('new workflow')).toBeNull()
})

it('the workspace view links a card under the current workspace', async () => {
  location.hash = '#/ws/studio/workflows'
  await renderPage('tti')
  expect(card('tti')!.getAttribute('href')).toBe('#/ws/studio/workflows/templates/tti')
})
```
(Import `location`-setting into `beforeEach`: `location.hash = '#/ws/default/workflows'`. The page reads `workspace.current`, so the router module must be imported in the test file: add `import '../router.svelte'` after the component import.)

Append to `AssetsPage.test.ts` (find its `libraries` mock; it returns `workspace`, `common`, `examples` entries):
```ts
it('the shared view hides the workspace section and makes shared upload the filled button', async () => {
  render(AssetsPage, { shared: true })
  await waitFor(() => expect(screen.getByText(/shared library/i)).toBeTruthy())
  expect(screen.queryByText('This workspace')).toBeNull()
  const upload = screen.getByRole('button', { name: /upload to shared/i })
  expect(upload.classList.contains('quiet')).toBe(false)
})
```

Append to `JobPage.test.ts` (find how it mocks `api.getJob`; the job fixture carries `workspace`):
```ts
it('corrects the URL to the job\'s own workspace', async () => {
  location.hash = '#/ws/default/jobs/j1'
  jobFixture.workspace = 'studio'   // whatever the file names its fixture
  render(JobPage, { jobId: 'j1' })
  await waitFor(() => expect(location.hash).toBe('#/ws/studio/jobs/j1'))
})
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd ui && npx vitest run src/lib/pages`
Expected: the three new cases FAIL.

- [ ] **Step 3: `api.listJobs` limit**

```ts
  listJobs: (workspace?: string, limit?: number) => {
    const query = new URLSearchParams()
    if (workspace) query.set('workspace', workspace)
    if (limit) query.set('limit', String(limit))
    const qs = query.toString()
    return request<{ jobs: JobSummary[]; total?: number }>(
      qs ? `/api/jobs?${qs}` : '/api/jobs',
      undefined,
      { scope: false },
    )
  },
```
Update the comment above it: "Unscoped on purpose: `workspace` is explicit so Status can span every workspace and a workspace's Jobs page can name its own."

- [ ] **Step 4: Update each page**

- `EditorPage.svelte`: `go('jobs', job.id)` → `goWs('jobs', job.id)`; `href="#/workflows"` → `href={wsHref(workspace.current, 'workflows')}`; find `workflowHref` (the crumb link to the read-only view) and make it `wsHref(workspace.current, 'workflows', ...name.split('/'))`. Import `goWs` from `'../router.svelte'`, `wsHref` from `'../routes'`, `workspace` from `'../workspace.svelte'`.
- `WorkflowPage.svelte`: `go('edit')` → `goWs('edit')`; `go('workflows')` → `goWs('workflows')`; `go('jobs', job.id)` → `goWs('jobs', job.id)`; back link `href="#/workflows"` → `href={route.view.kind === 'shared' ? sharedHref('examples') : wsHref(workspace.current, 'workflows')}`; proof `href="#/gallery"` → `href={wsHref(workspace.current, 'gallery')}`. The `Edit` button/link that opens the editor for this workflow: ensure it goes to `goWs('edit', ...name.split('/'))` (a read-only example opened this way saves a copy into the workspace, as the server already does).
- `WorkflowsPage.svelte`: add `let { examples = false }: { examples?: boolean } = $props()`. Card href helper (`'#/workflows/' + ...`) → `examples ? sharedHref('examples', ...name.split('/')) : wsHref(workspace.current, 'workflows', ...name.split('/'))`. The `visible` derivation gains a first clause: `if (examples && details[name]?.writable !== false) return false`. Wrap the `<a class="newlink" href="#/edit">` in `{#if !examples}` and change its href to `wsHref(workspace.current, 'edit')`. Heading: `{examples ? 'Examples' : 'Workflows'}`.
- `GalleryPage.svelte`: `go('edit')` → `goWs('edit')`.
- `AssetsPage.svelte`: add `let { shared = false }: { shared?: boolean } = $props()`. In `sections`, add `.filter((section) => !shared || section.origin !== 'workspace')`. Where the two upload buttons render, the workspace one is inside `{#if !shared}` already by the section filter; the shared button's class becomes `class:quiet={!shared}` so it is the filled one under Shared. Heading: `{shared ? 'Shared assets' : 'Assets'}`.
- `PromptsPage.svelte`: `href="#/prompt-edit"` → `href={sharedHref('prompt-edit')}`; the per-prompt card href (find `'#/prompt-edit/'`) → `sharedHref('prompt-edit', ...name.split('/'))`.
- `PromptEditorPage.svelte`: `go('prompts')` → `go('shared', 'prompts')`; `go('prompt-edit')` → `go('shared', 'prompt-edit')`; `href="#/prompts"` → `href={sharedHref('prompts')}`.
- `JobPage.svelte`: `go('jobs', id)` (rerun) → `go('ws', job!.workspace, 'jobs', id)` (a rerun keeps its workspace); `href="#/jobs"` → `href={wsHref(job?.workspace ?? workspace.current, 'jobs')}`. Add, where the job first loads (the effect that sets `job = await api.getJob(jobId)`):
  ```ts
  // A legacy '#/jobs/<id>' lands under the last-used workspace as a first
  // guess; the job knows its own, so the URL is corrected once it answers
  if (route.view.kind === 'ws' && route.view.workspace !== job.workspace)
    location.replace(wsHref(job.workspace, 'jobs', jobId))
  ```
- `JobsPage.svelte`: add `let { scope = 'workspace' }: { scope?: 'workspace' | 'all' } = $props()`. Replace `workspaceFilter`'s use in the poll with `const filter = scope === 'all' ? workspaceFilter : workspace.current` (read inside the effect so a switch refetches). Wrap the workspace `<select>` in `{#if scope === 'all' && ...}`. The `wschip` shows only when `scope === 'all'`. Row href `'#/jobs/' + job.id` → `wsHref(job.workspace, 'jobs', job.id)`. `move()`'s refetch uses the same `filter` expression. Heading: `{scope === 'all' ? 'All jobs' : 'Jobs'}`. Remove `onMount(loadWorkspaces)` only if `scope==='workspace'` never needs names - simpler: keep it (cached).

- `App.svelte`: `shared/assets` → `<AssetsPage shared />`; `shared/examples` (no rest) → `<WorkflowsPage examples />`.

- [ ] **Step 5: Run everything**

Run: `cd ui && npm run check && npm run lint && npm test`
Expected: PASS. Grep for leftovers: `grep -rn "href=\"#/\|go('jobs'\|go('edit'\|go('workflows'\|go('prompts'" ui/src/lib | grep -v test` should return nothing.

- [ ] **Step 6: Commit**

```bash
git add -A ui/src
git commit -m "feat(ui): pages link under their workspace; jobs, assets and workflows take a scope"
```

---

### Task 6: Overview page

**Files:**
- Create: `ui/src/lib/pages/OverviewPage.svelte`, `ui/src/lib/pages/OverviewPage.test.ts`
- Modify: `ui/src/App.svelte` (route `overview` to it)

**Interfaces:**
- Consumes: `api.gallery`, `api.listJobs(ws, limit)`, `api.listWorkflows`, `api.listAssets`, `api.galleryThumbnailUrl`, `api.health`; `latestProofs` (Task 3); `deleteWorkspaceWithConfirm` (Task 3); `workspace` (Task 2); `wsHref` (Task 1); `formatBytes`; `stepProgress` is not needed - the running job's row shows status only, the job page shows progress.

- [ ] **Step 1: Write the failing tests**

`ui/src/lib/pages/OverviewPage.test.ts`:
```ts
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/svelte'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import OverviewPage from './OverviewPage.svelte'

const data = vi.hoisted(() => ({
  files: [] as unknown[],
  jobs: [] as unknown[],
  workflows: [] as string[],
  assets: [] as unknown[],
}))
const deleteWorkspace = vi.hoisted(() => vi.fn())
vi.mock('../api', () => ({
  api: {
    gallery: () => Promise.resolve({ files: data.files }),
    listJobs: () => Promise.resolve({ jobs: data.jobs }),
    listWorkflows: () => Promise.resolve({ workflow_dir: '/ws/workflows', workflows: data.workflows, details: {} }),
    listAssets: () => Promise.resolve({ assets: data.assets, libraries: [], shadowed: [], folders: [], asset_dir: null, asset_dirs: [] }),
    galleryThumbnailUrl: (n: string) => `/thumb/${n}`,
    listWorkspaces: () => Promise.resolve({ workspace_root: '/ws', default: 'default', workspaces: [{ name: 'default', default: true }, { name: 'studio', default: false, usage: { files: 4, bytes: 4096 } }] }),
    deleteWorkspace: (n: string, a?: boolean) => deleteWorkspace(n, a),
  },
}))
vi.mock('../confirm.svelte', () => ({ confirmDialog: vi.fn(() => Promise.resolve(true)) }))
vi.mock('../toast', () => ({ notify: { error: vi.fn(), success: vi.fn() } }))

beforeEach(async () => {
  data.files = []; data.jobs = []; data.workflows = []; data.assets = []
  deleteWorkspace.mockReset()
  localStorage.clear()
  location.hash = '#/ws/studio/overview'
  vi.resetModules()
  await import('../router.svelte')
})
afterEach(cleanup)

it('an empty workspace shows empty states, not empty frames', async () => {
  render(OverviewPage)
  await waitFor(() => expect(screen.getByText(/nothing generated yet/i)).toBeTruthy())
  expect(document.querySelectorAll('.frame')).toHaveLength(0)
  expect(screen.getByText(/no jobs yet/i)).toBeTruthy()
  expect(screen.getByText(/no workflows yet/i)).toBeTruthy()
})

it('shows recent outputs, jobs and workflows, each linking to its page', async () => {
  data.files = [{ name: 'shot/r1/a.png', folder: 'shot', subfolder: '', url: '/o/a.png', kind: 'image', size: 1, mtime: 2, label: 'a' }]
  data.jobs = [{ id: 'j1', workflow: 'shot', status: 'succeeded', created_at: 1, started_at: 1, finished_at: 2, workspace: 'studio' }]
  data.workflows = ['shot', 'still']
  data.assets = [{ name: 'x.png', reference: 'asset:x.png', folder: '', kind: 'image', origin: 'workspace' }, { name: 'y.png', reference: 'asset:y.png', folder: '', kind: 'image', origin: 'common' }]
  render(OverviewPage)
  await waitFor(() => expect(document.querySelectorAll('.frame').length).toBe(1))
  expect(screen.getByRole('link', { name: /recent outputs/i }).getAttribute('href')).toBe('#/ws/studio/gallery')
  expect(screen.getByRole('link', { name: 'j1' }).getAttribute('href')).toBe('#/ws/studio/jobs/j1')
  // the workflow with a proof sorts first and carries its picture
  const cards = screen.getAllByRole('link', { name: /^(shot|still)$/ })
  expect(cards[0].textContent).toContain('shot')
  expect(screen.getByText('1 asset')).toBeTruthy()   // the workspace's own only
  expect(screen.getByText('4.0 KB')).toBeTruthy()
})

it('delete is offered off the default and goes through the shared action', async () => {
  deleteWorkspace.mockResolvedValue({ name: 'studio', deleted: true })
  render(OverviewPage)
  await waitFor(() => expect(screen.getByRole('button', { name: /delete workspace/i })).toBeTruthy())
  await fireEvent.click(screen.getByRole('button', { name: /delete workspace/i }))
  await waitFor(() => expect(location.hash).toBe('#/ws/default/overview'))
})

it('the default workspace cannot be deleted', async () => {
  location.hash = '#/ws/default/overview'
  vi.resetModules(); await import('../router.svelte')
  render(OverviewPage)
  await waitFor(() => expect(screen.getByRole('button', { name: /delete workspace/i })).toBeTruthy())
  expect((screen.getByRole('button', { name: /delete workspace/i }) as HTMLButtonElement).disabled).toBe(true)
})
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd ui && npx vitest run src/lib/pages/OverviewPage.test.ts`
Expected: FAIL - component missing.

- [ ] **Step 3: Implement `OverviewPage.svelte`**

```svelte
<script lang="ts">
  import { Images, Inbox, Layers, Plus } from '@lucide/svelte'
  import { api } from '../api'
  import Empty from '../Empty.svelte'
  import { formatBytes } from '../format'
  import { latestProofs } from '../proofs'
  import { wsHref } from '../routes'
  import type { GalleryFile, JobSummary } from '../types'
  import { DEFAULT_WORKSPACE, loadWorkspaces, workspace } from '../workspace.svelte'
  import { deleteWorkspaceWithConfirm } from '../workspaceActions'

  // Each panel loads on its own: the page is a glance at the workspace, and
  // a slow gallery must not hold the jobs list back
  let recent = $state<GalleryFile[] | null>(null)
  let jobs = $state<JobSummary[] | null>(null)
  let workflows = $state<string[] | null>(null)
  let proofs = $state<Record<string, GalleryFile>>({})
  let assetCount = $state<number | null>(null)

  const name = $derived(workspace.current)
  const RECENT = 8
  const WORKFLOWS = 6
  const JOBS = 5

  $effect(() => {
    void workspace.current
    recent = null; jobs = null; workflows = null; assetCount = null
    loadWorkspaces()
    api.gallery().then((r) => {
      recent = r.files.slice(0, RECENT)
      proofs = latestProofs(r.files)
    }).catch(() => (recent = []))
    api.listJobs(workspace.current, JOBS + 1).then((r) => (jobs = r.jobs.reverse())).catch(() => (jobs = []))
    api.listWorkflows().then((r) => (workflows = r.workflows)).catch(() => (workflows = []))
    api.listAssets().then((r) => (assetCount = r.assets.filter((a) => a.origin === 'workspace').length)).catch(() => (assetCount = 0))
  })

  // The ones that have produced something first - the same order the catalog
  // uses, so the overview and the catalog agree on what is "recent"
  const featured = $derived(
    [...(workflows ?? [])]
      .sort((l, r) => (proofs[l] ? 0 : 1) - (proofs[r] ? 0 : 1) || l.localeCompare(r))
      .slice(0, WORKFLOWS),
  )
  const running = $derived(jobs?.find((j) => j.status === 'running') ?? null)
  const finished = $derived((jobs ?? []).filter((j) => j.status !== 'running').slice(0, JOBS))
  const directory = $derived(
    workspace.root ? (name === DEFAULT_WORKSPACE ? workspace.root : `${workspace.root}/${name}`) : null,
  )
  const usage = $derived(workspace.usage[name])
  const leaf = (n: string) => n.split('/').pop() ?? n
</script>

<div class="head">
  <h1>{name}</h1>
  {#if directory}<code class="muted dir">{directory}</code>{/if}
</div>

<section class="panel">
  <h2><a class="plain" href={wsHref(name, 'gallery')}><Images size={15} /> Recent outputs</a></h2>
  {#if recent === null}
    <p class="muted">loading…</p>
  {:else if recent.length === 0}
    <Empty>{#snippet icon()}<Images size={36} strokeWidth={1.5} />{/snippet}Nothing generated yet — run a workflow.</Empty>
  {:else}
    <div class="strip">
      {#each recent as file (file.name)}
        <a class="plain frame tile" href={wsHref(name, 'gallery')} title={file.label}>
          {#if file.kind === 'image'}<img src={api.galleryThumbnailUrl(file.name)} alt={file.label} loading="lazy" />
          {:else if file.kind === 'video'}<video src={file.url} muted playsinline preload="metadata"></video>
          {:else}<span class="muted">{file.label}</span>{/if}
        </a>
      {/each}
    </div>
  {/if}
</section>

<div class="two">
  <section class="panel">
    <h2><a class="plain" href={wsHref(name, 'jobs')}>Jobs</a></h2>
    {#if jobs === null}
      <p class="muted">loading…</p>
    {:else if jobs.length === 0}
      <Empty>{#snippet icon()}<Inbox size={36} strokeWidth={1.5} />{/snippet}No jobs yet.</Empty>
    {:else}
      <ul class="jobs">
        {#if running}
          <li class="runningnow"><span class="chip running">running</span><span class="wf">{running.workflow}</span><a class="plain" href={wsHref(name, 'jobs', running.id)}>{running.id}</a></li>
        {/if}
        {#each finished as job (job.id)}
          <li><span class="chip {job.status}">{job.status}</span><span class="wf">{job.workflow}</span><a class="plain" href={wsHref(name, 'jobs', job.id)}>{job.id}</a></li>
        {/each}
      </ul>
    {/if}
  </section>

  <section class="panel">
    <h2>
      <a class="plain" href={wsHref(name, 'workflows')}><Layers size={15} /> Workflows</a>
      {#if workflows}<span class="num muted">{workflows.length}</span>{/if}
      <span class="flex"></span>
      <a class="button withicon new" href={wsHref(name, 'edit')} title="new workflow"><Plus size={15} /> New workflow</a>
    </h2>
    {#if workflows === null}
      <p class="muted">loading…</p>
    {:else if workflows.length === 0}
      <Empty>{#snippet icon()}<Layers size={36} strokeWidth={1.5} />{/snippet}No workflows yet — create one in the editor, or run an example.</Empty>
    {:else}
      <div class="cards">
        {#each featured as wf (wf)}
          <a class="plain card" href={wsHref(name, 'workflows', ...wf.split('/'))}>
            {#if proofs[wf]}
              <span class="frame cardframe">
                {#if proofs[wf].kind === 'image'}<img src={api.galleryThumbnailUrl(proofs[wf].name)} alt="" loading="lazy" />
                {:else}<video src={proofs[wf].url} muted playsinline preload="metadata"></video>{/if}
              </span>
            {/if}
            <span class="cardname">{leaf(wf)}</span>
          </a>
        {/each}
      </div>
    {/if}
  </section>
</div>

<div class="two">
  <section class="panel">
    <h2><a class="plain" href={wsHref(name, 'assets')}>Assets and disk</a></h2>
    <dl>
      <dt>assets</dt><dd>{#if assetCount === null}…{:else}{assetCount} {assetCount === 1 ? 'asset' : 'assets'}{/if}</dd>
      {#if usage}<dt>on disk</dt><dd class="num" title="{usage.files} files">{formatBytes(usage.bytes)}</dd>{/if}
    </dl>
  </section>
  <section class="panel">
    <h2>Manage</h2>
    <p class="muted">Deleting removes the workspace's workflows, assets and outputs on the server. The prompt library and the shared asset library are untouched.</p>
    <button class="quiet danger" onclick={() => deleteWorkspaceWithConfirm(name)} disabled={name === DEFAULT_WORKSPACE}
      title={name === DEFAULT_WORKSPACE ? 'the default workspace is the root itself and cannot be deleted' : 'delete this workspace and everything in it'}>
      Delete workspace
    </button>
  </section>
</div>

<style>
  .head { display: flex; align-items: baseline; gap: var(--space-3); margin-bottom: var(--space-4); }
  .dir { font-size: var(--t-xs); }
  section { margin-bottom: var(--space-4); }
  h2 { display: flex; align-items: center; gap: var(--space-2); }
  h2 a { display: inline-flex; align-items: center; gap: 0.35rem; }
  .flex { flex: 1; }
  .new { font-family: var(--font-sans); font-size: var(--t-sm); }
  .two { display: grid; grid-template-columns: 1fr 1fr; gap: var(--space-4); }
  @container (max-width: 640px) { .two { grid-template-columns: 1fr; } }
  .strip { display: grid; grid-template-columns: repeat(auto-fill, minmax(120px, 1fr)); gap: var(--space-2); }
  .tile { aspect-ratio: 1; }
  .jobs { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: var(--space-2); }
  .jobs li { display: flex; align-items: center; gap: var(--space-3); font-family: var(--font-mono); font-size: var(--t-sm); }
  .jobs .wf { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .jobs .runningnow { box-shadow: inset 2px 0 0 var(--live); padding-left: var(--space-2); }
  .cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap: var(--space-2); }
  .card { display: flex; flex-direction: column; gap: var(--space-1); font-family: var(--font-mono); font-size: var(--t-sm); }
  .cardframe { aspect-ratio: 4 / 3; }
  dl { display: grid; grid-template-columns: auto 1fr; gap: var(--space-1) var(--space-3); margin: 0; }
  dt { color: var(--muted); }
  dd { margin: 0; font-family: var(--font-mono); }
</style>
```

The `New workflow` link is the page's one filled button: add a global `a.button` rule in `app.css` next to `button` if one does not exist (`a.button { display: inline-flex; background: var(--accent); color: var(--accent-ink); border: 1px solid var(--accent); border-radius: var(--radius-1); padding: 0.4rem 0.9rem; text-decoration: none; }`) - check first with `grep -n "a.button" ui/src/app.css`.

- [ ] **Step 4: Route it**

In `App.svelte` add `import OverviewPage from './lib/pages/OverviewPage.svelte'` and, in the `ws` branch, `{:else if view.section === 'overview'}<OverviewPage />` before the `gallery` branch.

- [ ] **Step 5: Run**

Run: `cd ui && npm run check && npm run lint && npm test`
Expected: PASS. `npm run dev` and look at an overview with and without outputs.

- [ ] **Step 6: Commit**

```bash
git add ui/src/lib/pages/OverviewPage.svelte ui/src/lib/pages/OverviewPage.test.ts ui/src/App.svelte ui/src/app.css
git commit -m "feat(ui): workspace overview page"
```

---

### Task 7: Server → Status page

**Files:**
- Create: `ui/src/lib/pages/StatusPage.svelte`
- Modify: `ui/src/App.svelte`

**Interfaces:**
- Consumes: `ServerPage` (stripped in Task 3), `JobsPage` with `scope="all"` (Task 5).

- [ ] **Step 1: Implement**

```svelte
<script lang="ts">
  import ServerPage from './ServerPage.svelte'
  import JobsPage from './JobsPage.svelte'
</script>

<!-- What this server is, then what it is doing: the queue across every
     workspace lives here because the queue is the server's, not any one
     workspace's -->
<ServerPage />
<div class="alljobs">
  <JobsPage scope="all" />
</div>

<style>
  .alljobs { margin-top: var(--space-5); padding-top: var(--space-5); border-top: 1px solid var(--line); }
</style>
```

In `App.svelte`: import `StatusPage`; in the `server` branch replace the final `{:else}<ServerPage />` with `{:else}<StatusPage />` and drop the direct `ServerPage` import if nothing else uses it.

- [ ] **Step 2: Run**

Run: `cd ui && npm run check && npm run lint && npm test`
Expected: PASS. In `npm run dev`, `#/server/status` shows the server panel with the all-workspaces job list (and its workspace filter once two workspaces exist).

- [ ] **Step 3: Commit**

```bash
git add ui/src/lib/pages/StatusPage.svelte ui/src/App.svelte
git commit -m "feat(ui): server status page carries the all-workspaces queue"
```

---

### Task 8: End-to-end tests

**Files:**
- Create: `ui/e2e/workspaces.spec.ts`
- Modify: `ui/e2e/smoke.spec.ts`, `ui/e2e/responsive.spec.ts`, `ui/e2e/chrome.spec.ts` (any `goto('/#/…')` and any header-nav assertions)

- [ ] **Step 1: Update the legacy gotos**

Every `page.goto('/#/<legacy>')` in the e2e specs still works through the redirect, so leave the paths; but update assertions that look for the old top nav (`chrome.spec.ts` / `responsive.spec.ts` - grep for `navlabel`, `nav a`, `getByRole('navigation')`) to look for the sidebar: `page.getByRole('complementary', { name: 'navigation' })` (an `<aside aria-label>` is `complementary`). The smoke test `page.goto('/')` then `heading 'Workflows'` must become `heading 'default'` (the overview) - or goto `/#/ws/default/workflows`. Run `npx playwright test` and fix what fails; do not weaken a test that was asserting behaviour that still exists.

- [ ] **Step 2: Write `workspaces.spec.ts`**

```ts
import { expect, test } from '@playwright/test'

test('a legacy hash lands under the last-used workspace', async ({ page }) => {
  await page.goto('/#/gallery')
  await expect(page).toHaveURL(/#\/ws\/default\/gallery$/)
  await expect(page.getByRole('heading', { name: 'Gallery' })).toBeVisible()
})

test('the root lands on the overview and the sidebar marks it', async ({ page }) => {
  await page.goto('/')
  await expect(page).toHaveURL(/#\/ws\/default\/overview$/)
  const nav = page.getByRole('complementary', { name: 'navigation' })
  await expect(nav.getByRole('link', { name: 'Overview' })).toHaveAttribute('aria-current', 'page')
})

test('create a workspace, work in it, delete it', async ({ page }) => {
  await page.goto('/')
  const nav = page.getByRole('complementary', { name: 'navigation' })
  await nav.getByRole('button', { name: 'new workspace' }).click()
  await nav.getByPlaceholder('name').fill('e2e-ws')
  await nav.getByPlaceholder('name').press('Enter')
  await expect(page).toHaveURL(/#\/ws\/e2e-ws\/overview$/)
  // its gallery is scoped: the request carries the workspace
  const galleryRequest = page.waitForRequest((r) => r.url().includes('/api/gallery') && r.url().includes('workspace=e2e-ws'))
  await nav.getByRole('link', { name: 'Gallery' }).click()
  await galleryRequest
  // back to default through the sidebar, and the scope follows
  const defaultRequest = page.waitForRequest((r) => r.url().includes('/api/gallery') && !r.url().includes('workspace='))
  await nav.getByRole('link', { name: 'default' }).click()
  await nav.getByRole('link', { name: 'Gallery' }).click()
  await defaultRequest
  // delete from its overview
  await page.goto('/#/ws/e2e-ws/overview')
  await page.getByRole('button', { name: 'Delete workspace' }).click()
  await page.getByRole('alertdialog').getByRole('button', { name: 'Delete' }).click()
  await expect(page).toHaveURL(/#\/ws\/default\/overview$/)
  await expect(nav.getByText('e2e-ws')).toHaveCount(0)
})

test('an unknown workspace falls back to default and says so', async ({ page }) => {
  await page.goto('/#/ws/nope/gallery')
  await expect(page).toHaveURL(/#\/ws\/default\/overview$/)
  await expect(page.getByText(/No workspace named nope/)).toBeVisible()
})

test('shared prompts and server models are reachable from the sidebar', async ({ page }) => {
  await page.goto('/')
  const nav = page.getByRole('complementary', { name: 'navigation' })
  await nav.getByRole('link', { name: 'Prompts' }).click()
  await expect(page).toHaveURL(/#\/shared\/prompts$/)
  await nav.getByRole('link', { name: 'Models' }).click()
  await expect(page).toHaveURL(/#\/server\/models$/)
})
```

The fixture server runs `--workspace <scratch>` (see `e2e/serve_fixture.py`), so `POST /api/workspaces` creates `e2e-ws` under the scratch root and the delete removes it - nothing in the repo is touched.

- [ ] **Step 3: Run**

Run: `cd ui && npx playwright test`
Expected: PASS (serialized on one worker; the first run compiles the server import chain, allow ~2 min).

- [ ] **Step 4: Commit**

```bash
git add ui/e2e
git commit -m "test(ui): e2e coverage for workspace routes and the sidebar"
```

---

### Task 9: Docs, build, and release note

**Files:**
- Modify: `docs/SERVER.md` ("The pages" section; the `![The Server page …]` screenshot caption)
- Modify: `ui/CLAUDE.md` (the scoping paragraph; a paragraph on the sidebar)
- Modify: `CLAUDE.md` top-level "Workspaces on the server" paragraph - one sentence: the web UI is organised by workspace (`ui/src/lib/Sidebar.svelte`), with the workspace in the hash.
- Build: `ui/dist`

- [ ] **Step 1: `docs/SERVER.md`**

Replace the intro of "The pages" with:

> The UI is organised by workspace. A sidebar lists every workspace on the server; the selected one opens into **Overview, Workflows, Jobs, Gallery, Assets, Editor**, and the hash carries the workspace (`#/ws/studio/gallery`), so a link names where it points and an old `#/gallery` bookmark redirects to the last workspace you were in. Below the workspaces, **Shared** holds what every workspace sees - the prompt library, the `common` asset library and the read-only example workflows - and **Server** holds Models, Schema and Status.

Then per bullet: **Overview** (new bullet: recent outputs, jobs, workflows, assets/disk, delete); **Workflows** - drop the "a picker here chooses" sentence, say the workspace's own plus, under Shared → Examples, the read-only ones; **Prompts** - lead with "Under Shared:"; **Jobs** - "A workspace's Jobs page lists its own; Server → Status lists every workspace's with a filter"; **Server** → **Status** - "…and, below, the queue across every workspace. Workspaces are created from the sidebar and deleted from their Overview." Update the screenshot caption. Regenerate `docs/img/ui-server-dark.png` and add `docs/img/ui-overview-dark.png` if you have a server to shoot; otherwise note in the commit that screenshots are owed.

- [ ] **Step 2: `ui/CLAUDE.md`**

Replace the paragraph starting "Every request is scoped to the selected workspace in one place" with:

> Every request is scoped to the selected workspace in one place: `scoped()` in `lib/api.ts` reads `workspace.current` and appends `?workspace=`. The route is what sets it: `lib/routes.ts` parses `#/ws/<name>/<section>/...`, and `router.svelte.ts` calls `applyRouteWorkspace` on every change, so a `ws` route names the workspace and a `shared`/`server` route keeps the last one named (localStorage `dw-workspace`, which is only ever a fallback). There is no picker; the sidebar (`lib/Sidebar.svelte`) is links. A page refetches on a switch by reading `workspace.current` inside its load effect, as before. Legacy hashes (`#/gallery`) are rewritten in place by `legacyRedirect`; `#/jobs/<id>` lands under the last-used workspace and `JobPage` corrects the URL to the job's own once it loads. A job's files still load from its own workspace (`outputUrl(path, version, workspace)`).

Add under Design system: "The sidebar's selected entry is a heavier ink left edge (`aria-current="page"`), never a colour; workspace and section names are mono."

- [ ] **Step 3: Build and full check**

Run: `cd ui && npm run preflight`
Expected: every stage passes; `ui/dist` rebuilt.

- [ ] **Step 4: Commit**

```bash
git add docs/SERVER.md ui/CLAUDE.md CLAUDE.md ui/dist
git commit -m "docs(ui): workspace-centric UI - pages by sidebar group; scoping follows the route"
```

Release note (for the next release's notes, not a file in this plan): "The web UI is organised by workspace: a sidebar of workspaces, each with an Overview; the workspace is in the URL (`#/ws/<name>/…`, old links redirect); Prompts moved under Shared; the all-workspaces queue moved to Server → Status; workspaces are created from the sidebar and deleted from their Overview."
