import { untrack } from 'svelte'
import {
  legacyRedirect,
  placeOf,
  queryString,
  resolveView,
  type RouteView,
  type WsSection,
} from './routes'
import {
  applyRouteWorkspace,
  lastUsedWorkspace,
  setRouteReplacer,
  workspace,
} from './workspace.svelte'

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

/** The hash's path and its '?query' tail - '#/ws/studio/jobs?status=failed'.
 * The query is a page's own view state (a list's filters); it is not part
 * of the route, so changing it is not a navigation. */
function hashParts(): [string, string] {
  const hash = location.hash.replace(/^#\/?/, '')
  const at = hash.indexOf('?')
  return at < 0 ? [hash, ''] : [hash.slice(0, at), hash.slice(at + 1)]
}

function split(): string[] {
  const [path] = hashParts()
  return path ? path.split('/').map(decode) : []
}

/** The current hash's query, as key -> value. */
export function hashQuery(): Record<string, string> {
  const query: Record<string, string> = {}
  for (const pair of hashParts()[1].split('&')) {
    if (!pair) continue
    const at = pair.indexOf('=')
    const key = at < 0 ? pair : pair.slice(0, at)
    query[decode(key)] = at < 0 ? '' : decode(pair.slice(at + 1))
  }
  return query
}

/** The hash's query, reactive: setHashQuery's replaceState raises no event,
 * so a link built from the query (the sidebar's) reads it here. */
export const liveQuery = $state({ now: hashQuery() })
window.addEventListener('hashchange', () => (liveQuery.now = hashQuery()))

/** The current path with `values` as its query, empty values dropped: what
 * setHashQuery writes in place, and what a push of a query names. */
export function queryHash(values: Record<string, string>): string {
  return '#/' + hashParts()[0] + queryString(values)
}

/** Replace the hash's query in place: no history entry and no hashchange,
 * so typing in a filter neither floods Back nor re-runs the route. Empty
 * values are dropped, so a cleared filter leaves a clean link. Nothing
 * while a leave guard is asking or its decline is walking back: the browser
 * is on another page's entry, and writing there would commit its URL. */
export function setHashQuery(values: Record<string, string>) {
  if (asking || undoing) return
  const hash = queryHash(values)
  if (hash !== location.hash) history.replaceState(history.state, '', hash)
  committed = location.hash
  liveQuery.now = hashQuery()
}

/** How a box's value is written into the hash's query and read back. */
export type HashCodec<T> = {
  parse: (raw: string) => T
  format: (value: T) => string
}
const text: HashCodec<string> = { parse: (raw) => raw, format: (v) => v }
/** A list, comma-joined: `roles=final,intermediate` */
const list: HashCodec<string[]> = {
  parse: (raw) => (raw ? raw.split(',') : []),
  format: (value) => value.join(','),
}
export const codec = {
  list,
  /** One of `values`, the first being the box's initial one; anything else
   * (a hand-edited or stale link) reads as that first one */
  oneOf: <T extends string>(values: readonly T[]): HashCodec<T> => ({
    parse: (raw) => (values.includes(raw as T) ? (raw as T) : values[0]),
    format: (value) => value,
  }),
  /** Some of `values`, comma-joined, read back in their order with no
   * repeats: an unknown one is dropped, and none left reads as `initial` */
  someOf: <T extends string>(
    values: readonly T[],
    initial: T[],
  ): HashCodec<T[]> => ({
    parse: (raw) => {
      const named = list.parse(raw)
      const known = values.filter((v) => named.includes(v))
      return known.length ? known : initial
    },
    format: list.format,
  }),
  /** On or off: `starred=1`, or the key left out */
  flag: {
    parse: (raw) => raw === '1',
    format: (value) => (value ? '1' : ''),
  } as HashCodec<boolean>,
}

// Every box on screen, by key, as the hash would carry it: a box writing its
// own key writes the others' too, so no filter ever drops another
// eslint-disable-next-line svelte/prefer-svelte-reactivity -- not reactive state
const boxes = new Set<{
  key: string
  written: () => string
  sync: () => void
}>()

// One listener for every box: a browser runs microtasks between listeners, so
// with one each, the first box's write effect flushed before the next box had
// read the new hash, and wrote that box's old value back into it
window.addEventListener('hashchange', () => {
  for (const box of boxes) box.sync()
})

/** A list's filter kept in the hash's query under `key`, so opening an item
 * and pressing Back - or sharing the link - keeps the narrowed view. Writing
 * the box replaces the query in place (no history entry); a hashchange
 * re-reads it, and a hash without the key (a bare nav link) puts it back to
 * `initial`, which the hash never spells out. Call it in component init. */
export function hashState(
  key: string,
  initial: string,
  codec?: HashCodec<string>,
): { value: string }
export function hashState<T>(
  key: string,
  initial: T,
  codec: HashCodec<T>,
): { value: T }
export function hashState<T>(
  key: string,
  initial: T,
  { parse, format }: HashCodec<T> = text as unknown as HashCodec<T>,
): { value: T } {
  const read = (): T => {
    const query = hashQuery()
    return key in query ? parse(query[key]) : initial
  }
  let value = $state(read())
  const bare = format(initial)
  const box = {
    key,
    written: () => {
      const raw = format(value)
      return raw === bare ? '' : raw
    },
    sync: () => {
      value = read()
    },
  }
  $effect(() => {
    boxes.add(box)
    return () => {
      boxes.delete(box)
    }
  })
  $effect(() => {
    const own = box.written()
    untrack(() => {
      const query = hashQuery()
      for (const other of boxes) query[other.key] = other.written()
      setHashQuery({ ...query, [key]: own })
    })
  })
  return {
    get value() {
      return value
    },
    set value(next: T) {
      value = next
    },
  }
}

function parse(): { parts: string[]; view: RouteView } {
  let parts = split()
  const redirect = legacyRedirect(parts, lastUsedWorkspace())
  if (redirect) {
    // replaceState raises no second hashchange: the settle that called this
    // (or the module's load) commits the redirected entry
    // the query is the page's own view state: it survives the move
    const query = hashParts()[1]
    const hash =
      '#/' +
      redirect.map(encodeURIComponent).join('/') +
      (query ? '?' + query : '')
    history.replaceState(history.state, '', hash)
    parts = redirect
  }
  const view = resolveView(parts)
  applyRouteWorkspace(view)
  return { parts, view }
}

export const route = $state(parse())

/** The hash of the route the app is showing. */
let committed = location.hash

type LeaveGuard = () => boolean | Promise<boolean>
// eslint-disable-next-line svelte/prefer-svelte-reactivity -- not reactive state
const guards = new Set<LeaveGuard>()
let asking = false
let undoing = false
// Set by replaceRoute until its own hashchange arrives: that entry is the
// committed one renamed, so it keeps the committed index. Only that arrival:
// a push while the replace is asked about is still a push
let replacing = false
// The committed entry's state, as replaceRoute found it, kept until the
// answer so a decline can rename the entry back
let replacedState: unknown = null

/** Every entry the app commits carries its position in `history.state`, so a
 * declined navigation can tell a pushed link from a Back or Forward
 * traversal and walk the history back to the committed entry. */
function indexOf(): number | undefined {
  const at = (history.state as { dwIndex?: unknown } | null)?.dwIndex
  return typeof at === 'number' ? at : undefined
}

function stamp(at: number) {
  const state = history.state
  const kept = state && typeof state === 'object' ? state : {}
  history.replaceState({ ...kept, dwIndex: at }, '')
}

const hadIndex = indexOf() !== undefined
let committedIndex = indexOf() ?? 0
stamp(committedIndex)
/** The index of the entry the browser was last seen on. */
let lastSeenIndex = committedIndex

// Where each entry was left, by its index, for Back and Forward to go back
// to; the newest 50. The browser's own restore is off: it would land before
// the page it restores into has rendered. Mirrored to sessionStorage so a
// reload - which keeps the entry's index in history.state - finds its place
const SCROLLS_KEPT = 50
const SCROLLS_KEY = 'dw-scrolls'
function loadScrolls(fresh: boolean): [number, number][] {
  // an entry with no index is a new tab or a hand-typed address: none of the
  // saved places belong to it
  if (fresh) return []
  try {
    const saved: unknown = JSON.parse(sessionStorage.getItem(SCROLLS_KEY) ?? '')
    if (!Array.isArray(saved)) return []
    return saved
      .filter(
        (e): e is [number, number] =>
          Array.isArray(e) &&
          typeof e[0] === 'number' &&
          typeof e[1] === 'number',
      )
      .slice(-SCROLLS_KEPT)
  } catch {
    return []
  }
}
function saveScrolls() {
  try {
    sessionStorage.setItem(SCROLLS_KEY, JSON.stringify([...scrolls]))
  } catch {
    // storage refused: the places stay in memory
  }
}
// eslint-disable-next-line svelte/prefer-svelte-reactivity -- not reactive state
const scrolls = new Map<number, number>(loadScrolls(!hadIndex))
history.scrollRestoration = 'manual'
function keepScroll(at: number) {
  scrolls.delete(at)
  scrolls.set(at, window.scrollY)
  const oldest = scrolls.keys().next().value
  if (scrolls.size > SCROLLS_KEPT && oldest !== undefined)
    scrolls.delete(oldest)
  saveScrolls()
}
// A reload or a close leaves the entry the page is on: keep its place too
window.addEventListener('pagehide', () => keepScroll(committedIndex))

// What the user does to scroll the page: any of these ends a restore
const USER_SCROLLS = ['wheel', 'touchstart', 'keydown', 'pointerdown']
// How long a restore waits for a slow list to render tall enough
const RESTORE_MS = 10_000
let ending = () => {}

/** End the restore in flight, if any: its timer, observer and listeners go.
 * A new navigation ends it, and the shell ends it on unmount. */
export function stopRestore() {
  ending()
}

/** Whether the page moved from where the restore last put it in a way the
 * page's own growing or shrinking can't explain: a scrollbar click (which
 * Firefox raises no pointerdown for), find-in-page, assistive tech. */
function movedByUser(set: number, room: number): boolean {
  const expected = Math.min(set, Math.max(0, room))
  return Math.abs(window.scrollY - expected) > 1
}

/** Scroll to where the entry at `index` was left. The page it lands on may
 * still be loading, so while it is too short it goes as far down as the
 * page reaches - never back to the top, whatever a reload renders meanwhile
 * - and goes again each time the document resizes, until the target is
 * reachable, the user scrolls, a new navigation starts or `ms` pass. */
export function restoreScroll(index: number, { ms = RESTORE_MS } = {}) {
  stopRestore()
  const target = scrolls.get(index) ?? 0
  let set = -1
  // Scroll anchoring off while it runs: content that grows above the list
  // would make the browser shift the page to keep its anchor in view, which
  // reads as the user's scroll and ends the restore short of its target
  const root = document.documentElement.style
  root.overflowAnchor = 'none'
  // the arrows reach `stop` and `step` only once they are defined
  const timer = setTimeout(() => stop(), ms)
  // jsdom has no ResizeObserver: the first step stands
  const observer =
    typeof ResizeObserver === 'undefined'
      ? undefined
      : new ResizeObserver(() => step())
  const stop = () => {
    clearTimeout(timer)
    root.overflowAnchor = ''
    observer?.disconnect()
    for (const type of USER_SCROLLS) window.removeEventListener(type, stop)
    if (ending === stop) ending = () => {}
  }
  const step = () => {
    const room = document.documentElement.scrollHeight - window.innerHeight
    if (set >= 0 && movedByUser(set, room)) return stop()
    set = Math.max(0, Math.min(target, room))
    window.scrollTo(0, set)
    if (room >= target) stop()
  }
  ending = stop
  for (const type of USER_SCROLLS)
    window.addEventListener(type, stop, { passive: true })
  step()
  if (ending === stop) observer?.observe(document.documentElement)
}

// The entry and path the page was last placed for. A traversal to an entry
// with a saved place goes back there, a new path starts at the top, and
// anything else - a filter typed, a pushed query, a rename (`renameRoute`) -
// leaves the page where it is
let placed: string | null = null
let placedIndex: number | null = null

/** Place the page after the route lands: Back or Forward to an entry goes
 * where it was left, even on the same path; a new path starts at the top;
 * otherwise the page stays. The shell calls it on every navigation. */
export function placeScroll() {
  const path = pathOf(committed)
  const traversed =
    committedIndex !== placedIndex && scrolls.has(committedIndex)
  const moved = path !== placed
  placed = path
  placedIndex = committedIndex
  if (traversed) restoreScroll(committedIndex)
  else if (moved) {
    stopRestore()
    window.scrollTo(0, 0)
  }
}

/** Stamp the entry a hashchange arrived on, guarded or not: a pushed entry
 * is one past the last one seen, a replaced one keeps its index. So the
 * walk back always knows how far, however many entries arrived while a
 * guard was asking. */
function arrive() {
  const fresh = indexOf() === undefined
  const at = indexOf() ?? (replacing ? lastSeenIndex : lastSeenIndex + 1)
  // a new entry reuses the index of one Back left behind: not its scroll
  if (fresh) {
    scrolls.delete(at)
    saveScrolls()
  }
  replacing = false
  stamp(at)
  lastSeenIndex = at
}

/** The browser is on the committed entry, under the committed URL. */
function home(): boolean {
  return indexOf() === committedIndex && location.hash === committed
}

/** Ask before the route leaves its current path (an editor with unsaved
 * edits). Return true to allow. Guards are asked in registration order and
 * never for a query-only change. Returns a function that removes the guard. */
export function addLeaveGuard(guard: LeaveGuard): () => void {
  const entry: LeaveGuard = () => guard()
  guards.add(entry)
  return () => guards.delete(entry)
}

function pathOf(hash: string): string {
  return hash.replace(/^#\/?/, '').split('?')[0]
}

async function allowed(): Promise<boolean> {
  for (const guard of [...guards]) if (!(await guard())) return false
  return true
}

// A hand-off on its way: the session-storage keys `goCarrying` wrote, and
// the path whose page takes them
let carried: { path: string; keys: string[]; unguarded: boolean } | null = null

/** The navigation the hand-off rode on is not happening: take it back, so a
 * later visit does not open it. */
function dropCarried() {
  for (const key of carried?.keys ?? []) sessionStorage.removeItem(key)
  carried = null
}

/** Take the entry the browser is on as the route. */
function settle() {
  // arrived: the keys are the destination page's to take now
  if (carried?.path === pathOf(location.hash)) carried = null
  else dropCarried()
  stopRestore()
  replacing = false
  replacedState = null
  const at = indexOf() ?? committedIndex + 1
  if (at !== committedIndex) keepScroll(committedIndex)
  const next = parse()
  committed = location.hash
  route.parts = next.parts
  route.view = next.view
  committedIndex = at
  lastSeenIndex = at
  stamp(at)
}

/** Back on the committed entry: a replace that renamed it while a guard was
 * asking is renamed back, so the URL matches the route. */
function restore() {
  if (location.hash !== committed)
    history.replaceState(replacedState ?? history.state, '', committed)
  replacing = false
  replacedState = null
  lastSeenIndex = committedIndex
}

/** Walk the history back to the committed entry: the index difference,
 * whether the entries were pushed links or a traversal. The entries left
 * behind keep their own URLs, so Back and Forward still work. A declined
 * replace is no walk at all, only the committed URL put back. */
function undo() {
  const at = indexOf()
  const delta = at === undefined ? -1 : committedIndex - at
  if (!delta) {
    restore()
    return
  }
  undoing = true
  history.go(delta)
}

/** Ask the guards; meanwhile every hashchange only records where the
 * browser is. Allowed, the route settles on that entry, the user's latest
 * choice; declined, the history walks back to the committed one. */
function ask() {
  asking = true
  allowed()
    .catch(() => false)
    .then((ok) => {
      asking = false
      // allowed but back on the committed entry is no navigation: the
      // route keeps its arrays, as a decline there would
      if (ok && !home()) settle()
      else {
        dropCarried()
        undo()
      }
    })
}

// The landing of our own history.go. popstate, not hashchange: a walk
// between two entries of the same hash raises no hashchange, and a flag
// waiting for one would swallow the next navigation
window.addEventListener('popstate', () => {
  if (!undoing) return
  undoing = false
  restore()
})

window.addEventListener('hashchange', () => {
  arrive()
  // asking, it only records where the browser is; on the committed entry
  // (our own walk back, already restored) there is nothing to do
  if (asking || home()) return
  const path = pathOf(location.hash)
  const exempt = carried?.unguarded && carried.path === path
  if (!guards.size || exempt || placeOf(path) === placeOf(pathOf(committed)))
    settle()
  else ask()
})

/** Replace the entry the app is on with `hash`, through the guards - a
 * page correcting its own URL. Declined, the committed URL is put back in
 * place with no history traversal; allowed, the route settles on the same
 * entry, at the same index. */
export function replaceRoute(hash: string) {
  const before = location.hash
  replacing = true
  replacedState = history.state
  location.replace(hash)
  // the same hash raises no hashchange, so nothing will answer the flag
  if (location.hash === before) {
    replacing = false
    replacedState = null
  }
}

/** Rename the entry the app is on - an editor's first save gives its
 * document a path - with no history entry, no hashchange and no guard: the
 * page stays, only what the route calls it changes. `route.parts` is
 * updated in place, so the shell does not take it as a navigation (which
 * scrolls to the top); `route.view` is replaced, which renames the page. */
export function renameRoute(hash: string) {
  history.replaceState(history.state, '', hash)
  committed = location.hash
  placed = pathOf(committed)
  const next = parse()
  route.parts.splice(0, route.parts.length, ...next.parts)
  route.view = next.view
}

function hrefOf(parts: string[]): string {
  return '/' + parts.map(encodeURIComponent).join('/')
}

export function go(...parts: string[]) {
  goHref(hrefOf(parts))
}

/** `go`, with a hand-off the destination page reads from session storage.
 * The entries last only if the route arrives there: a leave guard's decline,
 * or landing anywhere else, removes them. `unguarded` skips the leave guards,
 * for a navigation that takes the unsaved edits with it. */
export function goCarrying(
  parts: string[],
  entries: Record<string, string>,
  { unguarded = false } = {},
) {
  dropCarried()
  for (const [key, value] of Object.entries(entries))
    sessionStorage.setItem(key, value)
  const href = hrefOf(parts)
  carried = {
    path: pathOf('#' + href),
    keys: Object.keys(entries),
    unguarded,
  }
  goHref(href)
}

/** Navigate, as `go` does (a new history entry), to a hash a builder in
 * `routes.ts` spelled - `compareHref`, the one place a comparison is */
export function goHref(hash: string) {
  location.hash = hash
}

/** Navigate to a section of the workspace the UI is in. */
export function goWs(section: WsSection, ...rest: string[]) {
  go('ws', workspace.current, section, ...rest)
}

setRouteReplacer(replaceRoute)
