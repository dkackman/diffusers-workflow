# UI/UX Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eleven web UI improvements, from a tab title that tracks the running
job up to a side-by-side compare view, each shipped and tested on its own.

**Architecture:**
- Most tasks are UI-only (Svelte 5 runes, `ui/src/lib/`). Pure logic goes in a
  small `.ts` module with its own Vitest file, and the component only wires it
  up. That is the codebase's existing pattern (`progress.ts`, `picks.svelte.ts`,
  `outputMeta.ts`).
- Two tasks change the server:
  - **Stars:** saved in each run's existing `notes.json` through `dw/notes.py`,
    with one new route.
  - **Rerun lineage:** a `rerun_of` field on the job spec, plus one new
    job-history column.
- The UI's server response types are generated (`python scripts/dump_openapi.py`
  then `npm run gen:api`) and are never written by hand.

**Tech Stack:**
- Svelte 5, Vitest with jsdom, `@testing-library/svelte`, `@lucide/svelte`,
  svelte-sonner (through `ui/src/lib/toast.ts`);
- FastAPI with pydantic, pytest (xdist), sqlite job history.

**Spec:** the suggestion list agreed in the 2026-10-14 session, restated here
because there is no separate design doc. Numbers match the suggestion list.

1. Tab title shows the running job; an alert fires when a job finishes.
2. Notes are rendered as Markdown.
3. Side-by-side compare of 2–4 outputs, with synced video playback.
4. ←/→ and Delete in the gallery detail.
5. Sidebar: hide workspaces, and group them by name prefix. (Every workspace
   name already shows in full on hover - `Sidebar.svelte` sets `title={name}`.)
6. Drag-and-drop uploads (Assets page, media argument inputs).
7. Star ("keep") an output, and filter the gallery to starred ones.
8. Measured run time from this server's history (`observed`), shown on the
   workflow page and the cards.
9. Placeholder blocks instead of "loading…" on the Overview panels.
10. ~~Theme toggle~~ - **already exists** (`cycleTheme` in `ui/src/App.svelte`),
    dropped.
11. A note you expanded stays expanded (per browser).
12. Jobs list: time in the queue, and the job each rerun came from.

## Global Constraints

- Colour (`ui/src/app.css` header): `--live` is machine state only, and
  `--select` is the user's place only. Where they may appear is enforced by
  `ui/scripts/design-rules.test.ts`. New UI uses ink, `--muted`, `--line` and
  `--panel-2` only.
- Type: anything the engine reads literally (names, ids, paths, numbers) is
  `--font-mono`; text written for a person is `--font-sans`.
- Every token pair passes WCAG AA in both themes.
- Persisted UI state goes through `ui/src/lib/storage.ts` (`storageGet` /
  `storageSet`, keys auto-prefixed `dw-`) and must degrade silently when
  storage is unavailable.
- User text is never rendered as HTML: no `{@html}`, no `innerHTML`.
  `NoteEditor.test.ts` "shows the note as text, never markup" must keep
  passing.
- Server: paths go through `dw/security.py` validators. Never use
  `eval`/`exec`/`shell=True`. Notes and stars are written only by the server.
- `ui/src/lib/generated/` comes from `python scripts/dump_openapi.py` (repo
  root) then `npm run gen:api` (in `ui/`), never hand-edited.
- Checks: from `ui/` run `npm run check`, `npm run lint`, `npm test`. For
  server tasks also run `pytest tests/ -v`. `dw.serve` needs a restart to pick
  up server changes.
- Match the surrounding code: comment density, prose-style comments explaining
  *why*, and the existing component structure.

## Review Focus

1. **The health poll fails once** (server restart, network blip). A failed
   poll must not count as "every running job ended": no false "finished"
   toast or alert. Pinned in Task 1 (`watchJobs` ignores a null health).
2. **The UI is opened over plain HTTP from another machine** (`http://lem:8765`
   - the user's real setup). Browser notifications need a secure context, so
   the alert must fall back to a chime and the button must say so, not fail
   silently. Pinned in Task 1 (`alertMode` test).
3. **Arrow or Delete key pressed while typing in a note textarea, or while a
   focused `<video>` is seeking.** It must not change the selection or delete
   a file. Pinned in Task 3 (`isEditable` covers media elements, and a test
   presses ← inside the note box).
4. **Link markup in a note** (`[x](javascript:alert(1))`). It must render as
   plain text, never as a link. Pinned in Task 7 (`safeHref` and parse tests).
5. **A star write and a note write on the same run.** Neither may erase the
   other in `notes.json`. Pinned in Task 10 (a round-trip test in both
   directions).

---

## File Map

| File | Task | Responsibility |
|---|---|---|
| `ui/src/lib/jobWatch.ts` (+test) | 1 | running ids, ended ids, percent, tab title text |
| `ui/src/lib/finishAlert.ts` (+test) | 1 | desktop notification or chime fallback |
| `ui/src/lib/editable.ts` (+test) | 1, 3 | "is this keypress inside something editable" (moved out of App) |
| `ui/src/App.svelte` | 1, 12 | wire job watching, alert toggle; compare route |
| `ui/src/lib/format.ts` (+test) | 2 | `formatMinutes` (moved from WorkflowsPage) |
| `ui/src/lib/observed.ts` (+test) | 2 | text for the server's `observed` block |
| `ui/src/lib/pages/WorkflowPage.svelte`, `WorkflowsPage.svelte` | 2 | show measured time |
| `ui/src/lib/neighbor.ts` (+test) | 3 | next/previous in a list |
| `ui/src/lib/pages/GalleryPage.svelte` | 3, 10, 12 | keys, star filter, compare action |
| `ui/src/lib/KeyboardHelp.svelte` | 3, 10 | document the new keys |
| `ui/src/lib/expandedNotes.ts` (+test) | 4 | remembered expanded notes |
| `ui/src/lib/ui/ClampedText.svelte` (+test) | 4, 7 | remember key; Markdown mode; max-height clamp |
| `ui/src/lib/ui/Skeleton.svelte` (+test) | 5 | placeholder blocks |
| `ui/src/lib/jobTimes.ts` (+test) | 6 | queue wait |
| `ui/src/lib/pages/JobsPage.svelte` | 6, 11 | wait + rerun chip |
| `ui/src/lib/markdown.ts` (+test) | 7 | safe Markdown subset → node tree |
| `ui/src/lib/ui/Markdown.svelte` (+test) | 7 | render the tree with Svelte nodes |
| `ui/src/lib/dropFiles.ts` (+test) | 8 | drag-and-drop Svelte action |
| `ui/src/lib/pages/AssetsPage.svelte`, `editor/MediaArgumentInput.svelte` | 8 | accept drops |
| `ui/src/lib/wsGroups.ts` (+test) | 9 | group/hide workspace names |
| `ui/src/lib/hiddenWorkspaces.svelte.ts` | 9 | persisted hidden list |
| `ui/src/lib/Sidebar.svelte`, `pages/OverviewPage.svelte` | 9 | groups, hidden toggle |
| `dw/notes.py`, `dw/server/routes/media.py`, `dw/server/routes/gallery.py`, `dw/server/api_models.py` | 10 | stars |
| `tests/test_stars.py` | 10 | star store + API |
| `dw/server/jobs.py`, `job_record.py`, `job_history.py`, `api_models.py` | 11 | `rerun_of` |
| `tests/test_rerun_lineage.py` | 11 | lineage live + historical + migration |
| `ui/src/lib/compare.ts` (+test) | 12 | differing arguments |
| `ui/src/lib/pages/ComparePage.svelte` (+test) | 12 | compare grid |
| `ui/src/lib/BulkBar.svelte`, `routes.ts` | 12 | compare action + route |

---

## Phase A - small, UI-only

### Task 1: Tab title and job-finished alerts

**Files:**
- Create: `ui/src/lib/editable.ts`, `ui/src/lib/editable.test.ts`
- Create: `ui/src/lib/jobWatch.ts`, `ui/src/lib/jobWatch.test.ts`
- Create: `ui/src/lib/finishAlert.ts`, `ui/src/lib/finishAlert.test.ts`
- Modify: `ui/src/App.svelte` (the `isEditable` function, the status `$effect`
  around line 69, and the header `.state` block)

**Interfaces:**
- Produces: `isEditable(target: EventTarget | null): boolean` (Task 3 uses it)
- Produces: `runningIds(health: HealthInfo | null): string[]`,
  `endedSince(before: readonly string[], now: readonly string[]): string[]`,
  `percentDone(progress: JobDetail['progress']): number | null`,
  `tabTitle(job: TitleJob | null, othersRunning?: number): string`,
  `endedTitle(job: TitleJob & { status: string }): string`,
  `finishedText(job: JobDetail): string`, `BASE_TITLE`
- Produces: `alertMode(): 'notification' | 'chime'`,
  `enableAlerts(): Promise<boolean>`, `unlockChime(): void`,
  `alertFinished(text: string, href: string): void`

- [ ] **Step 1: Write the failing tests**

`ui/src/lib/editable.test.ts`:
```ts
import { describe, expect, it } from 'vitest'
import { isEditable } from './editable'

describe('isEditable', () => {
  it('is true for text entry and media, false for plain elements', () => {
    expect(isEditable(document.createElement('input'))).toBe(true)
    expect(isEditable(document.createElement('textarea'))).toBe(true)
    expect(isEditable(document.createElement('select'))).toBe(true)
    // a focused <video> seeks on the arrow keys itself
    expect(isEditable(document.createElement('video'))).toBe(true)
    expect(isEditable(document.createElement('div'))).toBe(false)
    expect(isEditable(null)).toBe(false)
  })
})
```

`ui/src/lib/jobWatch.test.ts`:
```ts
import { describe, expect, it } from 'vitest'
import type { HealthInfo, JobDetail } from './types'
import {
  BASE_TITLE,
  endedSince,
  endedTitle,
  percentDone,
  runningIds,
  tabTitle,
} from './jobWatch'

const health = (ids: (string | null)[], current: string | null = null) =>
  ({
    current_job: current ?? ids.find(Boolean) ?? null,
    workers: ids.map((id) => ({ current_job: id })),
  }) as unknown as HealthInfo

const progress = (p: Partial<NonNullable<JobDetail['progress']>>) =>
  p as JobDetail['progress']

describe('runningIds', () => {
  it('names every card’s job once', () => {
    expect(runningIds(health(['a', null, 'b']))).toEqual(['a', 'b'])
  })
  it('falls back to current_job when workers is absent', () => {
    expect(
      runningIds({ current_job: 'x' } as unknown as HealthInfo),
    ).toEqual(['x'])
  })
  it('is empty for no health', () => {
    expect(runningIds(null)).toEqual([])
  })
})

describe('endedSince', () => {
  it('is what was running and is not now', () => {
    expect(endedSince(['a', 'b'], ['b', 'c'])).toEqual(['a'])
  })
})

describe('percentDone', () => {
  it('counts finished steps plus the denoise fraction of the current one', () => {
    expect(
      percentDone(
        progress({
          step_index: 1,
          total_steps: 4,
          denoise_step: 5,
          denoise_total_steps: 10,
        }),
      ),
    ).toBe(37)
  })
  it('never claims 100 while running', () => {
    expect(
      percentDone(
        progress({
          step_index: 3,
          total_steps: 4,
          denoise_step: 10,
          denoise_total_steps: 10,
        }),
      ),
    ).toBe(99)
  })
  it('is null with no step count', () => {
    expect(percentDone(null)).toBeNull()
    expect(percentDone(progress({ total_steps: 0 }))).toBeNull()
  })
})

describe('tabTitle', () => {
  it('is the app name when idle', () => {
    expect(tabTitle(null)).toBe(BASE_TITLE)
  })
  it('shows percent, the workflow leaf name and other running jobs', () => {
    const job = {
      workflow: 'minimax/ref-to-video',
      workflow_name: 'templates/minimax/ref-to-video',
      progress: progress({ step_index: 0, total_steps: 2 }),
    }
    expect(tabTitle(job, 1)).toBe(`▶ 0% · ref-to-video +1 — ${BASE_TITLE}`)
  })
  it('leaves the percent out when there is none', () => {
    expect(
      tabTitle({ workflow: 'wf', workflow_name: null, progress: null }),
    ).toBe(`▶ wf — ${BASE_TITLE}`)
  })
})

describe('endedTitle', () => {
  it('marks the outcome', () => {
    const base = { workflow: 'wf', workflow_name: null, progress: null }
    expect(endedTitle({ ...base, status: 'succeeded' })).toBe(
      `✓ wf — ${BASE_TITLE}`,
    )
    expect(endedTitle({ ...base, status: 'failed' })).toBe(
      `✗ wf — ${BASE_TITLE}`,
    )
    expect(endedTitle({ ...base, status: 'cancelled' })).toBe(
      `⊘ wf — ${BASE_TITLE}`,
    )
  })
})
```

`ui/src/lib/finishAlert.test.ts`:
```ts
import { afterEach, describe, expect, it, vi } from 'vitest'
import { alertMode } from './finishAlert'

describe('alertMode', () => {
  afterEach(() => vi.unstubAllGlobals())

  it('uses desktop notifications on a secure origin', () => {
    vi.stubGlobal('isSecureContext', true)
    vi.stubGlobal('Notification', function () {})
    expect(alertMode()).toBe('notification')
  })
  it('falls back to a chime over plain http from another machine', () => {
    vi.stubGlobal('isSecureContext', false)
    vi.stubGlobal('Notification', function () {})
    expect(alertMode()).toBe('chime')
  })
  it('falls back to a chime where the API does not exist', () => {
    vi.stubGlobal('isSecureContext', true)
    vi.stubGlobal('Notification', undefined)
    expect(alertMode()).toBe('chime')
  })
})
```

- [ ] **Step 2: Run them to see them fail**

Run: `cd ui && npx vitest run src/lib/editable.test.ts src/lib/jobWatch.test.ts src/lib/finishAlert.test.ts`
Expected: FAIL, because the modules do not exist.

- [ ] **Step 3: Implement**

`ui/src/lib/editable.ts`:
```ts
/** Whether a keypress landed somewhere that owns its own keys: text entry,
 * a select, contenteditable, or a media element (a focused <video> seeks on
 * the arrow keys). A page-wide shortcut stands aside for all of them. */
export function isEditable(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false
  return (
    target instanceof HTMLInputElement ||
    target instanceof HTMLTextAreaElement ||
    target instanceof HTMLSelectElement ||
    target instanceof HTMLMediaElement ||
    target.isContentEditable
  )
}
```

`ui/src/lib/jobWatch.ts`:
```ts
/** What the tab title and the finish alert read off the 5-second health
 * poll. A job runs for minutes to half an hour, usually in a tab nobody is
 * looking at, so the title is where its state has to be. */
import { formatDuration } from './format'
import type { HealthInfo, JobDetail } from './types'

export const BASE_TITLE = 'diffusers-workflow'

export type TitleJob = Pick<JobDetail, 'workflow' | 'workflow_name' | 'progress'>

/** Every job a card is running now; `current_job` alone on a server too old
 * to report `workers`. */
export function runningIds(health: HealthInfo | null): string[] {
  if (!health) return []
  const ids = (health.workers ?? [])
    .map((w) => w.current_job)
    .filter((id): id is string => !!id)
  if (health.current_job && !ids.includes(health.current_job))
    ids.push(health.current_job)
  return ids
}

export function endedSince(
  before: readonly string[],
  now: readonly string[],
): string[] {
  return before.filter((id) => !now.includes(id))
}

/** 0-99: steps behind it plus how far the current step's denoise loop is.
 * Never 100 - a running job is not done. Null with nothing to go on. */
export function percentDone(progress: JobDetail['progress']): number | null {
  if (!progress?.total_steps) return null
  const inner = progress.denoise_total_steps
    ? Math.min(1, (progress.denoise_step ?? 0) / progress.denoise_total_steps)
    : 0
  const done = ((progress.step_index ?? 0) + inner) / progress.total_steps
  return Math.min(99, Math.max(0, Math.floor(done * 100)))
}

const leaf = (job: TitleJob) =>
  (job.workflow ?? job.workflow_name ?? 'job').split('/').pop() || 'job'

export function tabTitle(job: TitleJob | null, othersRunning = 0): string {
  if (!job) return BASE_TITLE
  const pct = percentDone(job.progress)
  const more = othersRunning > 0 ? ` +${othersRunning}` : ''
  return `▶ ${pct === null ? '' : `${pct}% · `}${leaf(job)}${more} — ${BASE_TITLE}`
}

const MARKS: Record<string, string> = {
  succeeded: '✓',
  failed: '✗',
  cancelled: '⊘',
}

export function endedTitle(job: TitleJob & { status: string }): string {
  return `${MARKS[job.status] ?? '•'} ${leaf(job)} — ${BASE_TITLE}`
}

/** "ref-to-video succeeded in 12m 54s" - the toast and the alert body. */
export function finishedText(job: JobDetail): string {
  const took =
    job.started_at && job.finished_at
      ? ` in ${formatDuration(job.finished_at - job.started_at)}`
      : ''
  return `${leaf(job)} ${job.status}${took}`
}
```

`ui/src/lib/finishAlert.ts`:
```ts
/** The alert a finished job raises in a tab nobody is looking at.
 *
 * A desktop notification needs a secure context - https, or localhost - and
 * this UI is routinely opened over plain http from another machine
 * (http://<gpu-box>:8765), where the Notification API is not offered. There
 * the alert is a short chime instead. The browser only lets a page make
 * sound after the person has interacted with it, so the chime is unlocked on
 * the click that turns alerts on, and again on the first click after a
 * reload. */
export type AlertMode = 'notification' | 'chime'

export function alertMode(): AlertMode {
  return typeof Notification === 'function' && globalThis.isSecureContext
    ? 'notification'
    : 'chime'
}

let audio: AudioContext | null = null

export function unlockChime(): void {
  try {
    audio ??= new AudioContext()
    void audio.resume()
  } catch {
    /* no Web Audio - the tab title still carries the news */
  }
}

/** Asks for permission where that applies; false when it was refused. */
export async function enableAlerts(): Promise<boolean> {
  if (alertMode() === 'notification') {
    if (Notification.permission === 'granted') return true
    return (await Notification.requestPermission()) === 'granted'
  }
  unlockChime()
  return true
}

function chime(): void {
  if (!audio) return
  const now = audio.currentTime
  const tone = audio.createOscillator()
  const gain = audio.createGain()
  tone.frequency.value = 880
  gain.gain.setValueAtTime(0.15, now)
  gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.6)
  tone.connect(gain).connect(audio.destination)
  tone.start(now)
  tone.stop(now + 0.6)
}

export function alertFinished(text: string, href: string): void {
  if (alertMode() === 'notification' && Notification.permission === 'granted') {
    const note = new Notification('diffusers-workflow', { body: text })
    note.onclick = () => {
      window.focus()
      location.hash = href.replace(/^#/, '')
      note.close()
    }
    return
  }
  chime()
}
```

- [ ] **Step 4: Run the new tests**

Run: `cd ui && npx vitest run src/lib/editable.test.ts src/lib/jobWatch.test.ts src/lib/finishAlert.test.ts`
Expected: PASS

- [ ] **Step 5: Wire App.svelte**

1. Delete the local `isEditable` function and add
   `import { isEditable } from './lib/editable'`.
2. Add these imports: `Bell, BellOff` (to the existing `@lucide/svelte`
   import), `TERMINAL_STATUSES` (to the existing `./lib/api` import),
   `import { notify } from './lib/toast'`,
   `import { wsHref } from './lib/routes'`,
   `import type { JobDetail } from './lib/types'` (merge it with the existing
   type import), the `jobWatch` functions, and
   `import { alertFinished, alertMode, enableAlerts, unlockChime } from './lib/finishAlert'`.
3. Below the `currentJob` derived value, add:

```ts
  // The tab title and the finish alert (jobWatch.ts, finishAlert.ts). The
  // first poll only learns what is running - a job already going when the
  // page opened has not "just finished" when the next poll misses it
  let watched: string[] | null = null
  let runningJob = $state<JobDetail | null>(null)
  let othersRunning = $state(0)
  let endedMark = $state<string | null>(null)
  let alertsOn = $state(storageGet<boolean>('alert-finished', false))

  async function watchJobs(info: HealthInfo | null) {
    // A failed poll says nothing about the jobs - treating it as "none
    // running" would announce every running job as finished
    if (!info) return
    const now = runningIds(info)
    const ended = watched === null ? [] : endedSince(watched, now)
    watched = now
    othersRunning = Math.max(0, now.length - 1)
    runningJob = now[0] ? await api.getJob(now[0]).catch(() => null) : null
    for (const id of ended) {
      const job = await api.getJob(id).catch(() => null)
      if (job && TERMINAL_STATUSES.includes(job.status)) announce(job)
    }
  }

  function announce(job: JobDetail) {
    const text = finishedText(job)
    if (document.hidden) endedMark = endedTitle(job)
    // The job's own page already shows its outcome
    const onItsPage =
      view.kind === 'ws' && view.section === 'jobs' && view.rest[0] === job.id
    if (!onItsPage) {
      if (job.status === 'succeeded') notify.success(text)
      else notify.error(text)
    }
    if (alertsOn && document.hidden)
      alertFinished(text, wsHref(job.workspace, 'jobs', job.id))
  }

  async function toggleAlerts() {
    if (alertsOn) {
      alertsOn = false
    } else if (await enableAlerts()) {
      alertsOn = true
    } else {
      notify.error(
        'The browser blocked notifications for this site - allow them in its site settings',
      )
      return
    }
    storageSet('alert-finished', alertsOn)
  }

  $effect(() => {
    document.title = endedMark ?? tabTitle(runningJob, othersRunning)
  })
  $effect(() => {
    const back = () => {
      if (!document.hidden) endedMark = null
    }
    // A chime may only play after a gesture; after a reload the first
    // click anywhere is that gesture
    const unlock = () => {
      if (alertsOn && alertMode() === 'chime') unlockChime()
    }
    document.addEventListener('visibilitychange', back)
    window.addEventListener('pointerdown', unlock, { once: true })
    return () => {
      document.removeEventListener('visibilitychange', back)
      window.removeEventListener('pointerdown', unlock)
    }
  })
```

4. In `refreshStatus`, after `health = ...`, add
   `void watchJobs(health)`.
5. `view` is declared further down (`const view = $derived(route.view)`).
   Move that line above the new block so `announce` can read it.
6. In the header's `.state` block, put a button next to the theme toggle,
   copying its classes:

```svelte
          <button
            class="bare icon"
            onclick={toggleAlerts}
            aria-pressed={alertsOn}
            aria-label={alertsOn ? 'stop job-finished alerts' : 'alert me when a job finishes'}
            title={alertsOn
              ? 'alerts on - click to stop'
              : alertMode() === 'chime'
                ? 'chime when a job finishes while this tab is in the background (this page is not a secure origin, so desktop notifications are unavailable)'
                : 'notify me when a job finishes while this tab is in the background'}
          >
            {#if alertsOn}<Bell size={15} />{:else}<BellOff size={15} />{/if}
          </button>
```

- [ ] **Step 6: Run the UI checks**

Run: `cd ui && npm run check && npm run lint && npm test`
Expected: all pass. Then check by hand on `npm run dev` against a running
`dw.serve`: start a job, switch tabs, and watch the title count up then show
✓; turn alerts on and confirm the chime plays over a LAN http URL.

- [ ] **Step 7: Commit**

```bash
git add ui/src/lib/editable.ts ui/src/lib/editable.test.ts ui/src/lib/jobWatch.ts ui/src/lib/jobWatch.test.ts ui/src/lib/finishAlert.ts ui/src/lib/finishAlert.test.ts ui/src/App.svelte
git commit -m "feat(ui): tab title tracks the running job; alert when a job finishes"
```

---

### Task 2: Show measured run time (`observed`)

The server already computes, for each workflow, how long its finished jobs took
on this machine (`dw/server/observed_cost.py`). `GET /api/workflows` returns it
as `details[name].observed`, with `cold_minutes`, `warm_minutes`, `runs` and
optional `name`/`device`/`since`. The UI never shows it.

**Files:**
- Modify: `ui/src/lib/format.ts`, `ui/src/lib/format.test.ts` (move `formatMinutes` here)
- Create: `ui/src/lib/observed.ts`, `ui/src/lib/observed.test.ts`
- Modify: `ui/src/lib/pages/WorkflowsPage.svelte` (`formatMinutes`, `formatCost`, `cardTitle`, ~line 130-165 and the card around line 297)
- Modify: `ui/src/lib/pages/WorkflowPage.svelte` (load the detail, show it next to Run)

**Interfaces:**
- Produces: `formatMinutes(minutes: number): string` in `format.ts`
- Produces: `observedText(observed: Record<string, unknown> | null | undefined): string | null`,
  `observedTitle(observed: Record<string, unknown> | null | undefined): string | null`

- [ ] **Step 1: Write the failing tests**

`ui/src/lib/observed.test.ts`:
```ts
import { describe, expect, it } from 'vitest'
import { observedText, observedTitle } from './observed'

describe('observedText', () => {
  it('prefers the warm figure - what the next run costs with the model resident', () => {
    expect(
      observedText({ warm_minutes: 12.9, cold_minutes: 14.5, runs: 3 }),
    ).toBe('~12.9 min here')
  })
  it('falls back to the cold figure', () => {
    expect(observedText({ cold_minutes: 0.4, runs: 1 })).toBe('<1 min here')
  })
  it('is null without a figure', () => {
    expect(observedText(null)).toBeNull()
    expect(observedText({ runs: 2 })).toBeNull()
  })
})

describe('observedTitle', () => {
  it('names both figures, the run count and the card', () => {
    expect(
      observedTitle({
        warm_minutes: 12.9,
        warm_runs: 2,
        cold_minutes: 14.5,
        cold_runs: 1,
        runs: 3,
        name: 'RTX 4090',
        since: '2026-10-01T00:00:00Z',
      }),
    ).toBe(
      'measured on this server (RTX 4090): ~12.9 min with the model loaded (2 runs), ~14.5 min from cold (1 run); 3 runs since 2026-10-01',
    )
  })
})
```

Add to `ui/src/lib/format.test.ts`:
```ts
describe('formatMinutes', () => {
  it('rounds to two places and says <1 below a minute', () => {
    expect(formatMinutes(12.946)).toBe('~12.95 min')
    expect(formatMinutes(0.4)).toBe('<1 min')
  })
})
```
(add `formatMinutes` to that file's import from `./format`).

- [ ] **Step 2: Run them to see them fail**

Run: `cd ui && npx vitest run src/lib/observed.test.ts src/lib/format.test.ts`
Expected: FAIL

- [ ] **Step 3: Implement**

Move the arrow function `formatMinutes` from `WorkflowsPage.svelte` (around
line 133) into `format.ts` as an export, keeping its doc comment:
```ts
export const formatMinutes = (minutes: number) =>
  minutes < 1 ? '<1 min' : `~${Number(minutes.toFixed(2))} min`
```
Then import it in `WorkflowsPage.svelte`.

`ui/src/lib/observed.ts`:
```ts
/** The server's own history for a workflow - `observed` on a catalog
 * detail (dw/server/observed_cost.py) - said for a person. The curated
 * `cost` is a maintainer's claim about some other card; this is what this
 * box actually took, which is the number that predicts the next run. */
import { formatMinutes } from './format'

type Observed = Record<string, unknown> | null | undefined
const num = (v: unknown) => (typeof v === 'number' ? v : null)
const runs = (n: number) => `${n} ${n === 1 ? 'run' : 'runs'}`

export function observedText(observed: Observed): string | null {
  const minutes = num(observed?.warm_minutes) ?? num(observed?.cold_minutes)
  return minutes === null ? null : `${formatMinutes(minutes)} here`
}

export function observedTitle(observed: Observed): string | null {
  if (!observed) return null
  const parts: string[] = []
  const warm = num(observed.warm_minutes)
  const cold = num(observed.cold_minutes)
  if (warm !== null)
    parts.push(
      `${formatMinutes(warm)} with the model loaded (${runs(num(observed.warm_runs) ?? 0)})`,
    )
  if (cold !== null)
    parts.push(
      `${formatMinutes(cold)} from cold (${runs(num(observed.cold_runs) ?? 0)})`,
    )
  if (!parts.length) return null
  const card = observed.name ?? observed.device
  const since =
    typeof observed.since === 'string' ? ` since ${observed.since.slice(0, 10)}` : ''
  return `measured on this server${card ? ` (${card})` : ''}: ${parts.join(', ')}; ${runs(num(observed.runs) ?? 0)}${since}`
}
```

`WorkflowsPage.svelte`:
- In the card (around line 297), after the curated-cost `{#if}` block, add:
  ```svelte
  {#if !detail?.cost?.length && observedText(detail?.observed)}
    <span class="cost num muted">{observedText(detail?.observed)}</span>
  {/if}
  ```
- In `cardTitle`, after the `measured on` line, add:
  ```ts
  const seen = observedTitle(detail?.observed)
  if (seen) lines.push(seen)
  ```

`WorkflowPage.svelte`:
- Add state `let observed = $state<Record<string, unknown> | null>(null)`.
- In the effect that runs when `name` changes (the one calling
  `api.getWorkflow(name)`), also call:
  ```ts
  observed = null
  api
    .listWorkflows()
    .then((r) => {
      observed = (r.details?.[name]?.observed as Record<string, unknown>) ?? null
    })
    .catch(() => {})
  ```
- Right before the Run `<button>`, add:
  ```svelte
  {#if observedText(observed)}
    <span class="num muted" title={observedTitle(observed) ?? ''}
      >{observedText(observed)}</span
    >
  {/if}
  ```

- [ ] **Step 4: Run the tests and checks**

Run: `cd ui && npx vitest run src/lib/observed.test.ts src/lib/format.test.ts && npm run check && npm run lint && npm test`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ui/src/lib/format.ts ui/src/lib/format.test.ts ui/src/lib/observed.ts ui/src/lib/observed.test.ts ui/src/lib/pages/WorkflowsPage.svelte ui/src/lib/pages/WorkflowPage.svelte
git commit -m "feat(ui): show this server's measured run time beside Run and on cards"
```

---

### Task 3: Gallery ←/→ and Delete

**Files:**
- Create: `ui/src/lib/neighbor.ts`, `ui/src/lib/neighbor.test.ts`
- Modify: `ui/src/lib/pages/GalleryPage.svelte` (the `<svelte:window onkeydown>` around line 207, and `removeFile` around line 183)
- Modify: `ui/src/lib/pages/GalleryPage.test.ts`
- Modify: `ui/src/lib/KeyboardHelp.svelte`

**Interfaces:**
- Consumes: `isEditable` (Task 1), `overlayOpen` (`ui/layers.svelte.ts`)
- Produces: `neighbor(items: readonly string[], current: string, step: 1 | -1): string | null`

- [ ] **Step 1: Write the failing tests**

`ui/src/lib/neighbor.test.ts`:
```ts
import { describe, expect, it } from 'vitest'
import { neighbor } from './neighbor'

describe('neighbor', () => {
  const items = ['a', 'b', 'c']
  it('steps forward and back', () => {
    expect(neighbor(items, 'b', 1)).toBe('c')
    expect(neighbor(items, 'b', -1)).toBe('a')
  })
  it('stops at the ends rather than wrapping', () => {
    expect(neighbor(items, 'c', 1)).toBeNull()
    expect(neighbor(items, 'a', -1)).toBeNull()
  })
  it('is null for a name no longer listed', () => {
    expect(neighbor(items, 'z', 1)).toBeNull()
  })
})
```

Add to `GalleryPage.test.ts`. Follow that file's existing helpers: it renders
`GalleryPage` with `listing.files` and opens a file by clicking its tile.
Reuse whatever helper the "opens details" tests use.
```ts
it('the arrow keys step through the files the grid shows', async () => {
  render(GalleryPage)
  await openFile('a.png') // the file's existing helper that clicks a tile
  await fireEvent.keyDown(window, { key: 'ArrowRight' })
  expect(galleryMetadata).toHaveBeenLastCalledWith('b.png')
  await fireEvent.keyDown(window, { key: 'ArrowLeft' })
  expect(galleryMetadata).toHaveBeenLastCalledWith('a.png')
})

it('an arrow key typed in the note box does not move the selection', async () => {
  render(GalleryPage)
  await openFile('a.png')
  const before = galleryMetadata.mock.calls.length
  const box = document.createElement('textarea')
  document.body.append(box)
  await fireEvent.keyDown(box, { key: 'ArrowRight' })
  expect(galleryMetadata.mock.calls.length).toBe(before)
  box.remove()
})
```
If the file has no `openFile` helper, write one at the top of the new tests:
`const openFile = (label: string) => fireEvent.click(screen.getByTitle(label))`.
Match how the existing tests find a tile.

- [ ] **Step 2: Run them to see them fail**

Run: `cd ui && npx vitest run src/lib/neighbor.test.ts src/lib/pages/GalleryPage.test.ts`
Expected: FAIL

- [ ] **Step 3: Implement**

`ui/src/lib/neighbor.ts`:
```ts
/** The item `step` places from `current` in `items`, or null at either end
 * (no wrap: arriving back at the first file reads as "nothing happened") or
 * when `current` is no longer listed. */
export function neighbor(
  items: readonly string[],
  current: string,
  step: 1 | -1,
): string | null {
  const at = items.indexOf(current)
  if (at === -1) return null
  return items[at + step] ?? null
}
```

`GalleryPage.svelte`:
- Import `isEditable` from `'../editable'` and `neighbor` from `'../neighbor'`.
- Replace the window handler with:
```svelte
<svelte:window
  onkeydown={(e) => {
    if (overlayOpen()) return
    // Escape closes the thing on top, and a dialog answers it itself. The
    // selection is the more recent, more surprising state to be stuck in,
    // so it clears first and the detail panel on a second press
    if (e.key === 'Escape') {
      if (picks.size) picks.clear()
      else selected = null
      return
    }
    // Culling a run: step through what the grid shows, delete as you go.
    // A key typed into the note box, or a focused video seeking, is theirs
    if (!selected || isEditable(e.target)) return
    if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
      const next = neighbor(
        visible.map((f) => f.name),
        selected.name,
        e.key === 'ArrowRight' ? 1 : -1,
      )
      if (next) {
        e.preventDefault()
        select(byName.get(next)!)
      }
    } else if (e.key === 'Delete' || (e.key === 'Backspace' && e.metaKey)) {
      e.preventDefault()
      void removeFile()
    }
  }}
/>
```
- In `removeFile`, work out the next file *before* the delete. After a
  successful delete, open that file instead of closing the panel:
```ts
    const name = selected.name
    const order = visible.map((f) => f.name)
    const after = neighbor(order, name, 1) ?? neighbor(order, name, -1)
    try {
      await api.deleteOutput(name)
      files = files.filter((f) => f.name !== name)
      picks.drop(name)
      // Culling goes on to the next file rather than back to the grid
      if (after && byName.get(after)) select(byName.get(after)!)
      else selected = null
```

`KeyboardHelp.svelte`, after the `Esc` entry:
```svelte
    <dt><kbd>←</kbd> / <kbd>→</kbd></dt>
    <dd>previous / next file (gallery details)</dd>
    <dt><kbd>Delete</kbd> or <kbd>⌘</kbd> + <kbd>⌫</kbd></dt>
    <dd>delete the open file and show the next (gallery details)</dd>
```

- [ ] **Step 4: Run the tests and checks**

Run: `cd ui && npx vitest run src/lib/neighbor.test.ts src/lib/pages/GalleryPage.test.ts && npm run check && npm run lint`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ui/src/lib/neighbor.ts ui/src/lib/neighbor.test.ts ui/src/lib/pages/GalleryPage.svelte ui/src/lib/pages/GalleryPage.test.ts ui/src/lib/KeyboardHelp.svelte
git commit -m "feat(ui): arrow keys and Delete in the gallery details for culling a run"
```

---

### Task 4: Remember expanded notes

**Files:**
- Create: `ui/src/lib/expandedNotes.ts`, `ui/src/lib/expandedNotes.test.ts`
- Modify: `ui/src/lib/ui/ClampedText.svelte`, `ui/src/lib/ui/ClampedText.test.ts`
- Modify: `ui/src/lib/ui/NoteEditor.svelte`

**Interfaces:**
- Produces: `isRemembered(key: string): boolean`, `remember(key: string, open: boolean): void`
- Produces: `ClampedText` prop `rememberKey?: string`

- [ ] **Step 1: Write the failing tests**

`ui/src/lib/expandedNotes.test.ts`:
```ts
import { beforeEach, describe, expect, it } from 'vitest'
import { isRemembered, remember } from './expandedNotes'

describe('expandedNotes', () => {
  beforeEach(() => localStorage.clear())

  it('remembers and forgets', () => {
    remember('workspace:a', true)
    expect(isRemembered('workspace:a')).toBe(true)
    remember('workspace:a', false)
    expect(isRemembered('workspace:a')).toBe(false)
  })
  it('keeps only the newest 200', () => {
    for (let i = 0; i < 205; i++) remember(`k${i}`, true)
    expect(isRemembered('k0')).toBe(false)
    expect(isRemembered('k204')).toBe(true)
  })
})
```

Add to `ClampedText.test.ts`:
```ts
  it('opens already expanded when its key was left expanded', () => {
    localStorage.clear()
    fakeHeights(400, 80)
    const first = render(ClampedText, {
      props: { text: 'long text', rememberKey: 'k' },
    })
    return fireEvent
      .click(screen.getByRole('button', { name: 'Show more' }))
      .then(() => {
        first.unmount()
        render(ClampedText, { props: { text: 'long text', rememberKey: 'k' } })
        expect(
          screen.getByText('long text').classList.contains('clamped'),
        ).toBe(false)
      })
  })
```

- [ ] **Step 2: Run them to see them fail**

Run: `cd ui && npx vitest run src/lib/expandedNotes.test.ts src/lib/ui/ClampedText.test.ts`
Expected: FAIL

- [ ] **Step 3: Implement**

`ui/src/lib/expandedNotes.ts`:
```ts
/** Which long notes this browser left expanded, so a note someone opened
 * to read stays open when they come back to it. Newest last, capped, so the
 * list cannot grow without bound across thousands of run notes. */
import { storageGet, storageSet } from './storage'

const KEY = 'expanded-notes'
const CAP = 200

export function isRemembered(key: string): boolean {
  return storageGet<string[]>(KEY, []).includes(key)
}

export function remember(key: string, open: boolean): void {
  const list = storageGet<string[]>(KEY, []).filter((k) => k !== key)
  if (open) list.push(key)
  storageSet(KEY, list.slice(-CAP))
}
```

`ClampedText.svelte`:
- Props become
  `let { text, lines = 4, rememberKey }: { text: string; lines?: number; rememberKey?: string } = $props()`.
- Import `isRemembered, remember` from `'../expandedNotes'`.
- Make the initial state and the reset depend on the key:
```ts
  const recalled = () => (rememberKey ? isRemembered(rememberKey) : false)
  // svelte-ignore state_referenced_locally
  let expanded = $state(recalled())
  ...
  // svelte-ignore state_referenced_locally
  let seen = `${rememberKey ?? ''}\u0000${text}`
  $effect(() => {
    const now = `${rememberKey ?? ''}\u0000${text}`
    if (now === seen) return
    seen = now
    expanded = recalled()
  })
  function toggle() {
    expanded = !expanded
    if (rememberKey) remember(rememberKey, expanded)
  }
```
- The button's `onclick` becomes `onclick={toggle}`.

`NoteEditor.svelte`: pass
`rememberKey={`${workspace ?? ''}|${target}`}` to `<ClampedText>`.

- [ ] **Step 4: Run the tests and checks**

Run: `cd ui && npx vitest run src/lib/expandedNotes.test.ts src/lib/ui && npm run check && npm run lint`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ui/src/lib/expandedNotes.ts ui/src/lib/expandedNotes.test.ts ui/src/lib/ui/ClampedText.svelte ui/src/lib/ui/ClampedText.test.ts ui/src/lib/ui/NoteEditor.svelte
git commit -m "feat(ui): a note left expanded stays expanded"
```

---

### Task 5: Placeholder blocks on the Overview

**Files:**
- Create: `ui/src/lib/ui/Skeleton.svelte`, `ui/src/lib/ui/Skeleton.test.ts`
- Modify: `ui/src/lib/pages/OverviewPage.svelte` (the three `<p class="muted">loading…</p>`)
- Modify: `ui/src/lib/pages/OverviewPage.test.ts` if any test looks for the "loading…" text

**Interfaces:**
- Produces: `Skeleton` props `{ lines?: number; tiles?: number; label?: string }`

- [ ] **Step 1: Write the failing test**

`ui/src/lib/ui/Skeleton.test.ts`:
```ts
import { cleanup, render, screen } from '@testing-library/svelte'
import { afterEach, describe, expect, it } from 'vitest'
import Skeleton from './Skeleton.svelte'

describe('Skeleton', () => {
  afterEach(cleanup)
  it('announces loading and draws the requested shape', () => {
    render(Skeleton, { props: { tiles: 4, label: 'loading outputs' } })
    const status = screen.getByRole('status', { name: 'loading outputs' })
    expect(status.querySelectorAll('.tile')).toHaveLength(4)
  })
  it('draws lines by default', () => {
    render(Skeleton)
    expect(screen.getByRole('status').querySelectorAll('.line')).toHaveLength(3)
  })
})
```

- [ ] **Step 2: Run it to see it fail**

Run: `cd ui && npx vitest run src/lib/ui/Skeleton.test.ts`
Expected: FAIL

- [ ] **Step 3: Implement**

`ui/src/lib/ui/Skeleton.svelte`:
```svelte
<script lang="ts">
  // The shape of what is coming, in the panel's own grey, so a slow panel
  // does not jump when it lands. Neutral on purpose: --live is machine
  // state and a load is not one. Still under reduced motion
  let {
    lines = 3,
    tiles = 0,
    label = 'loading',
  }: { lines?: number; tiles?: number; label?: string } = $props()
</script>

<div class="skeleton" role="status" aria-label={label}>
  {#if tiles}
    <div class="tiles">
      {#each { length: tiles } as _, i (i)}<span class="tile"></span>{/each}
    </div>
  {:else}
    {#each { length: lines } as _, i (i)}
      <span class="line" style:width="{90 - (i % 3) * 20}%"></span>
    {/each}
  {/if}
</div>

<style>
  .skeleton {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
  }
  .tiles {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
    gap: var(--space-2);
  }
  .tile,
  .line {
    display: block;
    background: var(--panel-2);
    border-radius: var(--radius-2);
    animation: pulse 1.4s ease-in-out infinite;
  }
  .tile {
    aspect-ratio: 1;
  }
  .line {
    height: 0.9em;
  }
  @keyframes pulse {
    50% {
      opacity: 0.55;
    }
  }
  @media (prefers-reduced-motion: reduce) {
    .tile,
    .line {
      animation: none;
    }
  }
</style>
```

`OverviewPage.svelte`: import `Skeleton from '../ui/Skeleton.svelte'` and
replace each `loading…`:
- Recent outputs: `<Skeleton tiles={RECENT} label="loading recent outputs" />`
- Jobs: `<Skeleton lines={JOBS} label="loading jobs" />`
- Workflows: `<Skeleton lines={3} label="loading workflows" />`

If `OverviewPage.test.ts` asserts `getByText('loading…')`, change it to
`getByRole('status', { name: /loading/ })`.

- [ ] **Step 4: Run the tests and checks**

Run: `cd ui && npx vitest run src/lib/ui/Skeleton.test.ts src/lib/pages/OverviewPage.test.ts && npm run check && npm run lint && npm test`
Expected: PASS (the design-rules test included)

- [ ] **Step 5: Commit**

```bash
git add ui/src/lib/ui/Skeleton.svelte ui/src/lib/ui/Skeleton.test.ts ui/src/lib/pages/OverviewPage.svelte ui/src/lib/pages/OverviewPage.test.ts
git commit -m "feat(ui): placeholder blocks instead of 'loading…' on the overview"
```

---

### Task 6: Time in queue on the Jobs list

**Files:**
- Create: `ui/src/lib/jobTimes.ts`, `ui/src/lib/jobTimes.test.ts`
- Modify: `ui/src/lib/pages/JobsPage.svelte` (the `.started` span around line 193)

**Interfaces:**
- Produces: `queueWait(job: Pick<JobSummary, 'created_at' | 'started_at' | 'status'>, now: number): number | null`

- [ ] **Step 1: Write the failing test**

`ui/src/lib/jobTimes.test.ts`:
```ts
import { describe, expect, it } from 'vitest'
import { queueWait } from './jobTimes'

describe('queueWait', () => {
  it('is the time from submit to start for a job that started', () => {
    expect(
      queueWait({ created_at: 100, started_at: 160, status: 'succeeded' }, 999),
    ).toBe(60)
  })
  it('counts up to now for a job still queued', () => {
    expect(
      queueWait({ created_at: 100, started_at: null, status: 'queued' }, 400),
    ).toBe(300)
  })
  it('is null for a job cancelled before it started, or with no clock', () => {
    expect(
      queueWait({ created_at: 100, started_at: null, status: 'cancelled' }, 400),
    ).toBeNull()
    expect(
      queueWait({ created_at: null, started_at: null, status: 'queued' }, 400),
    ).toBeNull()
  })
})
```

- [ ] **Step 2: Run it to see it fail**

Run: `cd ui && npx vitest run src/lib/jobTimes.test.ts`
Expected: FAIL

- [ ] **Step 3: Implement**

`ui/src/lib/jobTimes.ts`:
```ts
import type { JobSummary } from './types'

/** Seconds a job waited for a card: submit to start, or submit to now for
 * one still waiting. Null when there is no wait to speak of - no clock, or
 * cancelled before it ever started. */
export function queueWait(
  job: Pick<JobSummary, 'created_at' | 'started_at' | 'status'>,
  now: number,
): number | null {
  if (!job.created_at) return null
  if (job.started_at) return Math.max(0, job.started_at - job.created_at)
  return job.status === 'queued' ? Math.max(0, now - job.created_at) : null
}
```

`JobsPage.svelte`: import `queueWait`. Replace
`<span class="muted started">{when(job.started_at)}</span>` with:
```svelte
        {@const wait = queueWait(job, Date.now() / 1000)}
        {#if job.status === 'queued'}
          <span class="muted started" title="time spent waiting for a card"
            >waiting {wait === null ? '' : formatDuration(wait)}</span
          >
        {:else}
          <span
            class="muted started"
            title={wait !== null && wait >= 1
              ? `waited ${formatDuration(wait)} in the queue`
              : ''}>{when(job.started_at)}</span
          >
        {/if}
```
`{@const}` must be the first child of a block. If Svelte rejects it here,
compute the value in a small function `waitOf(job)` in the script and call
that instead.

- [ ] **Step 4: Run the tests and checks**

Run: `cd ui && npx vitest run src/lib/jobTimes.test.ts src/lib/pages/JobsPage.test.ts && npm run check && npm run lint`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ui/src/lib/jobTimes.ts ui/src/lib/jobTimes.test.ts ui/src/lib/pages/JobsPage.svelte
git commit -m "feat(ui): jobs list shows how long a job waited for a card"
```

---

## Phase B - UI, medium

### Task 7: Notes rendered as Markdown (a safe subset)

The renderer is our own small parser, and it outputs **Svelte nodes, never
HTML**, so no sanitizer or new dependency is needed and the
"never markup" rule holds. The subset:
- `#`/`##`/`###` headings;
- `-`/`*` and `1.` lists;
- fenced code blocks;
- paragraphs (single newlines kept as line breaks);
- inline `**bold**`, `*italic*`, `` `code` ``, and `[text](https://…)`.

Underscore emphasis is left out on purpose: notes are full of `snake_case`
names (`cold_minutes`, `first_block`).

`-webkit-line-clamp` does not reliably clamp across block children (`<p>`,
`<ul>`). So `ClampedText` switches to a max-height clamp (`lines × line-height`)
with a fade, which works for both plain and Markdown text.

**Files:**
- Create: `ui/src/lib/markdown.ts`, `ui/src/lib/markdown.test.ts`
- Create: `ui/src/lib/ui/Markdown.svelte`, `ui/src/lib/ui/Markdown.test.ts`
- Modify: `ui/src/lib/ui/ClampedText.svelte` (prop `markdown`, max-height clamp)
- Modify: `ui/src/lib/ui/NoteEditor.svelte` (render notes as Markdown; hint under the textarea)

**Interfaces:**
- Produces: types `Inline`, `Block`; `parseMarkdown(src: string): Block[]`;
  `parseInline(src: string): Inline[]`; `safeHref(url: string): string | null`
- Produces: `Markdown` props `{ source: string }`; `ClampedText` prop `markdown?: boolean`

- [ ] **Step 1: Write the failing tests**

`ui/src/lib/markdown.test.ts`:
```ts
import { describe, expect, it } from 'vitest'
import { parseInline, parseMarkdown, safeHref } from './markdown'

describe('safeHref', () => {
  it('allows http, https and mailto only', () => {
    expect(safeHref('https://x.dev/a')).toBe('https://x.dev/a')
    expect(safeHref('mailto:a@b.c')).toBe('mailto:a@b.c')
    expect(safeHref('javascript:alert(1)')).toBeNull()
    expect(safeHref('JaVaScRiPt:alert(1)')).toBeNull()
    expect(safeHref('data:text/html,x')).toBeNull()
    expect(safeHref('/relative')).toBeNull()
  })
})

describe('parseInline', () => {
  it('reads bold, italic, code and links', () => {
    expect(parseInline('a **b** *c* `d` [e](https://x.dev)')).toEqual([
      { kind: 'text', text: 'a ' },
      { kind: 'strong', children: [{ kind: 'text', text: 'b' }] },
      { kind: 'text', text: ' ' },
      { kind: 'em', children: [{ kind: 'text', text: 'c' }] },
      { kind: 'text', text: ' ' },
      { kind: 'code', text: 'd' },
      { kind: 'text', text: ' ' },
      {
        kind: 'link',
        href: 'https://x.dev/',
        children: [{ kind: 'text', text: 'e' }],
      },
    ])
  })
  it('leaves an unsafe link as its literal text', () => {
    expect(parseInline('[x](javascript:alert(1))')).toEqual([
      { kind: 'text', text: '[x](javascript:alert(1))' },
    ])
  })
  it('leaves snake_case alone', () => {
    expect(parseInline('cold_minutes and warm_minutes')).toEqual([
      { kind: 'text', text: 'cold_minutes and warm_minutes' },
    ])
  })
  it('keeps markup-looking text as text', () => {
    expect(parseInline('<b>x</b>')).toEqual([
      { kind: 'text', text: '<b>x</b>' },
    ])
  })
})

describe('parseMarkdown', () => {
  it('reads headings, lists, code and paragraphs', () => {
    const src = [
      '## RULE',
      'Two paths, pick by target.',
      'Second line.',
      '',
      '- TV: L0 -> L7',
      '- Cinema: B3 -> P1',
      '',
      '1. one',
      '2. two',
      '',
      '```',
      'dw validate x',
      '```',
    ].join('\n')
    expect(parseMarkdown(src)).toEqual([
      { kind: 'heading', level: 2, children: [{ kind: 'text', text: 'RULE' }] },
      {
        kind: 'paragraph',
        children: [
          { kind: 'text', text: 'Two paths, pick by target.' },
          { kind: 'br' },
          { kind: 'text', text: 'Second line.' },
        ],
      },
      {
        kind: 'list',
        ordered: false,
        items: [
          [{ kind: 'text', text: 'TV: L0 -> L7' }],
          [{ kind: 'text', text: 'Cinema: B3 -> P1' }],
        ],
      },
      {
        kind: 'list',
        ordered: true,
        items: [[{ kind: 'text', text: 'one' }], [{ kind: 'text', text: 'two' }]],
      },
      { kind: 'code', text: 'dw validate x' },
    ])
  })
  it('a plain note is one paragraph', () => {
    expect(parseMarkdown('just text')).toEqual([
      { kind: 'paragraph', children: [{ kind: 'text', text: 'just text' }] },
    ])
  })
  it('an unclosed fence runs to the end', () => {
    expect(parseMarkdown('```\nx')).toEqual([{ kind: 'code', text: 'x' }])
  })
})
```

`ui/src/lib/ui/Markdown.test.ts`:
```ts
import { cleanup, render, screen } from '@testing-library/svelte'
import { afterEach, describe, expect, it } from 'vitest'
import Markdown from './Markdown.svelte'

describe('Markdown', () => {
  afterEach(cleanup)
  it('renders the subset as elements', () => {
    render(Markdown, {
      props: { source: '## Rule\n\n- **one**\n\n[docs](https://x.dev)' },
    })
    expect(screen.getByRole('heading', { name: 'Rule' })).toBeTruthy()
    expect(screen.getByText('one').tagName).toBe('STRONG')
    const link = screen.getByRole('link', { name: 'docs' })
    expect(link.getAttribute('href')).toBe('https://x.dev/')
    expect(link.getAttribute('rel')).toBe('noopener noreferrer')
  })
  it('never renders markup in the source', () => {
    render(Markdown, { props: { source: '<img src=x onerror=alert(1)>' } })
    expect(document.querySelector('img')).toBeNull()
    expect(screen.getByText('<img src=x onerror=alert(1)>')).toBeTruthy()
  })
  it('never links a javascript: url', () => {
    render(Markdown, { props: { source: '[x](javascript:alert(1))' } })
    expect(screen.queryByRole('link')).toBeNull()
  })
})
```

- [ ] **Step 2: Run them to see them fail**

Run: `cd ui && npx vitest run src/lib/markdown.test.ts src/lib/ui/Markdown.test.ts`
Expected: FAIL

- [ ] **Step 3: Implement the parser**

`ui/src/lib/markdown.ts`:
```ts
/** A small, safe Markdown subset for notes, parsed to a tree that
 * Markdown.svelte renders as Svelte nodes. Nothing here produces HTML, so
 * text that looks like markup stays text, and a link is only ever an
 * http(s)/mailto URL the browser's own URL parser accepted.
 *
 * Underscore emphasis is left out on purpose: notes quote snake_case
 * names (`cold_minutes`, `first_block`) constantly. */

export type Inline =
  | { kind: 'text'; text: string }
  | { kind: 'br' }
  | { kind: 'code'; text: string }
  | { kind: 'strong' | 'em'; children: Inline[] }
  | { kind: 'link'; href: string; children: Inline[] }

export type Block =
  | { kind: 'heading'; level: 1 | 2 | 3; children: Inline[] }
  | { kind: 'paragraph'; children: Inline[] }
  | { kind: 'list'; ordered: boolean; items: Inline[][] }
  | { kind: 'code'; text: string }

const SAFE_PROTOCOLS = new Set(['http:', 'https:', 'mailto:'])

export function safeHref(url: string): string | null {
  try {
    const parsed = new URL(url)
    return SAFE_PROTOCOLS.has(parsed.protocol) ? parsed.href : null
  } catch {
    return null
  }
}

// One alternation, tried left to right at each position: code first, so a
// `**` inside backticks stays literal
const INLINE =
  /`([^`]+)`|\*\*(.+?)\*\*|\*([^*\s][^*]*?)\*|\[([^\]]+)\]\(([^)\s]+(?:\([^)\s]*\))?[^)\s]*)\)/g

function pushText(out: Inline[], text: string) {
  if (!text) return
  const last = out[out.length - 1]
  if (last?.kind === 'text') last.text += text
  else out.push({ kind: 'text', text })
}

export function parseInline(src: string): Inline[] {
  const out: Inline[] = []
  let at = 0
  for (const m of src.matchAll(INLINE)) {
    pushText(out, src.slice(at, m.index))
    at = m.index + m[0].length
    if (m[1] !== undefined) out.push({ kind: 'code', text: m[1] })
    else if (m[2] !== undefined)
      out.push({ kind: 'strong', children: parseInline(m[2]) })
    else if (m[3] !== undefined)
      out.push({ kind: 'em', children: parseInline(m[3]) })
    else {
      const href = safeHref(m[5])
      if (href) out.push({ kind: 'link', href, children: parseInline(m[4]) })
      else pushText(out, m[0])
    }
  }
  pushText(out, src.slice(at))
  return out
}

const HEADING = /^(#{1,3})\s+(.*)$/
const BULLET = /^\s*[-*]\s+(.*)$/
const NUMBERED = /^\s*\d+[.)]\s+(.*)$/
const FENCE = /^\s*```/

const startsBlock = (line: string) =>
  FENCE.test(line) ||
  HEADING.test(line) ||
  BULLET.test(line) ||
  NUMBERED.test(line)

export function parseMarkdown(src: string): Block[] {
  const lines = src.replace(/\r\n?/g, '\n').split('\n')
  const blocks: Block[] = []
  let i = 0
  while (i < lines.length) {
    const line = lines[i]
    if (!line.trim()) {
      i++
      continue
    }
    if (FENCE.test(line)) {
      const body: string[] = []
      i++
      while (i < lines.length && !FENCE.test(lines[i])) body.push(lines[i++])
      i++ // the closing fence, or past the end when it never closed
      blocks.push({ kind: 'code', text: body.join('\n') })
      continue
    }
    const heading = HEADING.exec(line)
    if (heading) {
      blocks.push({
        kind: 'heading',
        level: heading[1].length as 1 | 2 | 3,
        children: parseInline(heading[2]),
      })
      i++
      continue
    }
    const list = BULLET.test(line) ? BULLET : NUMBERED.test(line) ? NUMBERED : null
    if (list) {
      const items: Inline[][] = []
      while (i < lines.length && list.test(lines[i]))
        items.push(parseInline(list.exec(lines[i++])![1]))
      blocks.push({ kind: 'list', ordered: list === NUMBERED, items })
      continue
    }
    // A paragraph runs until a blank line or the start of another block;
    // its single newlines are kept as line breaks - notes are written as
    // lines, not reflowed prose
    const children: Inline[] = []
    while (i < lines.length && lines[i].trim() && !startsBlock(lines[i])) {
      if (children.length) children.push({ kind: 'br' })
      children.push(...parseInline(lines[i++]))
    }
    blocks.push({ kind: 'paragraph', children })
  }
  return blocks
}
```

- [ ] **Step 4: Implement the component, and switch ClampedText and NoteEditor**

`ui/src/lib/ui/Markdown.svelte`:
```svelte
<script lang="ts">
  // Renders parseMarkdown's tree as Svelte nodes - never as HTML - so a note
  // that contains markup shows that markup as text. Headings are styled
  // paragraphs with a heading role two levels below the page's own, so a
  // note's "# RULE" does not compete with the page title in the outline
  import { parseMarkdown, type Inline } from '../markdown'

  let { source }: { source: string } = $props()
  const blocks = $derived(parseMarkdown(source))
</script>

{#snippet inline(nodes: Inline[])}
  {#each nodes as node, i (i)}
    {#if node.kind === 'text'}{node.text}{:else if node.kind === 'br'}<br
      />{:else if node.kind === 'code'}<code>{node.text}</code
      >{:else if node.kind === 'strong'}<strong>{@render inline(node.children)}</strong
      >{:else if node.kind === 'em'}<em>{@render inline(node.children)}</em
      >{:else}<a href={node.href} target="_blank" rel="noopener noreferrer"
        >{@render inline(node.children)}</a
      >{/if}
  {/each}
{/snippet}

<div class="md">
  {#each blocks as block, i (i)}
    {#if block.kind === 'heading'}
      <p class="h h{block.level}" role="heading" aria-level={block.level + 2}>
        {@render inline(block.children)}
      </p>
    {:else if block.kind === 'paragraph'}
      <p>{@render inline(block.children)}</p>
    {:else if block.kind === 'list' && block.ordered}
      <ol>
        {#each block.items as item, j (j)}<li>{@render inline(item)}</li>{/each}
      </ol>
    {:else if block.kind === 'list'}
      <ul>
        {#each block.items as item, j (j)}<li>{@render inline(item)}</li>{/each}
      </ul>
    {:else}
      <pre><code>{block.text}</code></pre>
    {/if}
  {/each}
</div>

<style>
  .md > :first-child {
    margin-top: 0;
  }
  .md > :last-child {
    margin-bottom: 0;
  }
  p,
  ul,
  ol,
  pre {
    margin: 0 0 0.5em;
  }
  ul,
  ol {
    padding-left: 1.4em;
  }
  .h {
    font-weight: 600;
    margin-top: 0.75em;
  }
  .h1 {
    font-size: 1.1em;
  }
  pre {
    background: var(--panel-2);
    padding: var(--space-2);
    border-radius: var(--radius-2);
    overflow-x: auto;
    white-space: pre;
  }
  code {
    font-family: var(--font-mono);
    font-size: 0.92em;
  }
</style>
```

`ClampedText.svelte`:
- Add the prop `markdown = false` (type `markdown?: boolean`) and
  `import Markdown from './Markdown.svelte'`.
- The text element body becomes
  `{#if markdown}<Markdown source={text} />{:else}{text}{/if}` and gets
  `class:plain={!markdown}`.
- Replace the CSS for `.text` / `.clamped`:
```css
  .text {
    line-height: 1.5;
    overflow-wrap: anywhere;
  }
  .plain {
    white-space: pre-wrap;
  }
  /* A height clamp, not -webkit-line-clamp: that one does not reliably
     clamp across block children, and a Markdown note is paragraphs and
     lists. The fade says there is more */
  .clamped {
    max-height: calc(var(--lines) * 1.5em);
    overflow: hidden;
    mask-image: linear-gradient(to bottom, #000 70%, transparent);
  }
```
- Update the comment at the top of the file to describe the height clamp.

`NoteEditor.svelte`:
- Pass `markdown` to `<ClampedText>`.
- Under the textarea, before `.actions`, add
  `<p class="hint muted">Markdown: # heading · - list · **bold** · *italic* · \`code\` · [link](https://…)</p>`
  with `.hint { font-size: var(--t-sm); margin: 0.25rem 0 0; }`.

- [ ] **Step 5: Run the tests and checks**

Run: `cd ui && npx vitest run src/lib/markdown.test.ts src/lib/ui && npm run check && npm run lint && npm test`
Expected: PASS. `NoteEditor.test.ts` "never markup" passes unchanged.

- [ ] **Step 6: Commit**

```bash
git add ui/src/lib/markdown.ts ui/src/lib/markdown.test.ts ui/src/lib/ui/Markdown.svelte ui/src/lib/ui/Markdown.test.ts ui/src/lib/ui/ClampedText.svelte ui/src/lib/ui/NoteEditor.svelte
git commit -m "feat(ui): notes render a safe Markdown subset"
```

---

### Task 8: Drag-and-drop uploads

**Files:**
- Create: `ui/src/lib/dropFiles.ts`, `ui/src/lib/dropFiles.test.ts`
- Modify: `ui/src/app.css` (one `.dropping` rule)
- Modify: `ui/src/lib/pages/AssetsPage.svelte` (`upload` around line 235; wrap the page content from `.pagehead` (line 297) through the `<FolderGroups>`)
- Modify: `ui/src/lib/editor/MediaArgumentInput.svelte` (`pick` around line 74; the root `<div class="media">`)

**Interfaces:**
- Produces: `dropFiles: Action<HTMLElement, DropOptions>` with
  `DropOptions = { ondrop: (files: File[]) => void; accept?: (file: File) => boolean; disabled?: boolean }`

- [ ] **Step 1: Write the failing test**

`ui/src/lib/dropFiles.test.ts`:
```ts
import { describe, expect, it, vi } from 'vitest'
import { dropFiles } from './dropFiles'

function dragEvent(type: string, files: File[]) {
  const event = new Event(type, { bubbles: true, cancelable: true })
  Object.defineProperty(event, 'dataTransfer', {
    value: { types: ['Files'], files, dropEffect: 'none' },
  })
  return event
}

describe('dropFiles', () => {
  it('hands over the accepted files and marks the target while dragging', () => {
    const node = document.createElement('div')
    const ondrop = vi.fn()
    const png = new File(['x'], 'a.png', { type: 'image/png' })
    const txt = new File(['x'], 'b.txt', { type: 'text/plain' })
    const action = dropFiles(node, {
      ondrop,
      accept: (f) => f.type.startsWith('image/'),
    })
    node.dispatchEvent(dragEvent('dragenter', [png, txt]))
    expect(node.classList.contains('dropping')).toBe(true)
    const drop = dragEvent('drop', [png, txt])
    node.dispatchEvent(drop)
    expect(drop.defaultPrevented).toBe(true)
    expect(ondrop).toHaveBeenCalledWith([png])
    expect(node.classList.contains('dropping')).toBe(false)
    action?.destroy?.()
  })

  it('ignores drops while disabled', () => {
    const node = document.createElement('div')
    const ondrop = vi.fn()
    dropFiles(node, { ondrop, disabled: true })
    node.dispatchEvent(dragEvent('drop', [new File(['x'], 'a.png')]))
    expect(ondrop).not.toHaveBeenCalled()
  })
})
```

- [ ] **Step 2: Run it to see it fail**

Run: `cd ui && npx vitest run src/lib/dropFiles.test.ts`
Expected: FAIL

- [ ] **Step 3: Implement**

`ui/src/lib/dropFiles.ts`:
```ts
/** Drop files from the desktop onto an element. A drag that carries no
 * files (text, a link, a tile being dragged inside the page) is left alone
 * so it keeps its native behaviour. `dropping` marks the element while
 * files hover; enter/leave are counted because they fire for every child
 * the pointer crosses. */
import type { Action } from 'svelte/action'

export interface DropOptions {
  ondrop: (files: File[]) => void
  accept?: (file: File) => boolean
  disabled?: boolean
}

export const dropFiles: Action<HTMLElement, DropOptions> = (node, initial) => {
  let options = initial
  let depth = 0
  const carriesFiles = (e: DragEvent) =>
    !options.disabled &&
    Array.from(e.dataTransfer?.types ?? []).includes('Files')
  const clear = () => {
    depth = 0
    node.classList.remove('dropping')
  }
  const enter = (e: DragEvent) => {
    if (!carriesFiles(e)) return
    e.preventDefault()
    depth++
    node.classList.add('dropping')
  }
  const over = (e: DragEvent) => {
    if (!carriesFiles(e)) return
    e.preventDefault()
    if (e.dataTransfer) e.dataTransfer.dropEffect = 'copy'
  }
  const leave = () => {
    if (depth > 0 && --depth === 0) node.classList.remove('dropping')
  }
  const drop = (e: DragEvent) => {
    if (!carriesFiles(e)) return
    e.preventDefault()
    clear()
    const files = Array.from(e.dataTransfer?.files ?? []).filter(
      (f) => options.accept?.(f) ?? true,
    )
    if (files.length) options.ondrop(files)
  }
  node.addEventListener('dragenter', enter)
  node.addEventListener('dragover', over)
  node.addEventListener('dragleave', leave)
  node.addEventListener('drop', drop)
  return {
    update(next) {
      options = next
    },
    destroy() {
      node.removeEventListener('dragenter', enter)
      node.removeEventListener('dragover', over)
      node.removeEventListener('dragleave', leave)
      node.removeEventListener('drop', drop)
    },
  }
}
```

`ui/src/app.css`, near the other global utility rules:
```css
/* A drop target with files hovering over it (dropFiles.ts). Ink, not
   --select: it is the pointer's place, not a selection */
.dropping {
  outline: 2px dashed var(--ink);
  outline-offset: 4px;
}
```

`AssetsPage.svelte`:
- Split `upload(event)` into the input handler and `uploadFiles(files: File[])`:
```ts
  async function upload(event: Event) {
    const input = event.target as HTMLInputElement
    const files = Array.from(input.files ?? [])
    // Clear it either way, so picking the same file twice still fires
    input.value = ''
    await uploadFiles(files)
  }

  // One file asks for its library name; a dropped batch keeps each file's
  // own, since a prompt per file would be a dialog storm
  async function uploadFiles(files: File[]) {
    if (!files.length) return
    let names: (string | undefined)[] = files.map(() => undefined)
    if (files.length === 1) {
      const assetName = window.prompt(
        `Upload — name in the ${uploadTarget} asset library:`,
        files[0].name,
      )
      if (assetName === null) return
      names = [assetName || undefined]
    }
    busy = true
    const failed: string[] = []
    for (const [i, file] of files.entries()) {
      try {
        const result = await api.uploadMedia(
          file,
          names[i],
          uploadTarget === 'shared',
        )
        if (files.length === 1)
          notify.success(`Uploaded as ${result.reference ?? result.url}`)
      } catch (e) {
        failed.push(`${file.name}: ${e instanceof Error ? e.message : String(e)}`)
      }
    }
    if (files.length > 1 && failed.length < files.length)
      notify.success(`Uploaded ${files.length - failed.length} of ${files.length} files`)
    if (failed.length) notify.error(failed.join('\n'))
    busy = false
    await load()
  }
```
- Import `dropFiles`. Wrap the content from `<div class="pagehead">` through
  the end of `<FolderGroups>…</FolderGroups>` in
  `<div class="dropzone" use:dropFiles={{ ondrop: uploadFiles, disabled: busy }}>`.
  Give it `.dropzone { min-height: 60vh; }` so an empty library is still a
  target. Add `multiple` to the file `<input>` (line 314), since the handler
  now takes several.

`MediaArgumentInput.svelte`:
- Split `pick` into the event handler and `uploadFile(file: File)`. The
  handler reads `e.currentTarget.files?.[0]`, clears the input, and calls
  `uploadFile(file)` when there is a file. The body after `if (!file) return`
  moves into `uploadFile`.
- On the root `<div class="media">` add
  `use:dropFiles={{ ondrop: (files) => uploadFile(files[0]), accept: (f) => f.type.startsWith(`${kind}/`), disabled: uploading }}`.
- Add to its title text: "or drop a file here".

- [ ] **Step 4: Run the tests and checks**

Run: `cd ui && npx vitest run src/lib/dropFiles.test.ts src/lib/pages/AssetsPage.test.ts src/lib/editor/MediaArgumentInput.test.ts && npm run check && npm run lint && npm test`
Expected: PASS. If an existing `AssetsPage.test.ts` upload test fires
`change` with one file, it still works: one file still prompts.

- [ ] **Step 5: Commit**

```bash
git add ui/src/lib/dropFiles.ts ui/src/lib/dropFiles.test.ts ui/src/app.css ui/src/lib/pages/AssetsPage.svelte ui/src/lib/editor/MediaArgumentInput.svelte
git commit -m "feat(ui): drop files onto the assets page and media inputs to upload"
```

---

### Task 9: Sidebar - hide workspaces, group by prefix

The hidden list is **per browser** (`localStorage`). Workspaces are shared
server state, and hiding one is a viewing preference, not a change to the
workspace. The open workspace is always shown even if hidden, so you can't
lose the page you're on. Workspace names cannot contain `/`
(`^[a-zA-Z_][a-zA-Z0-9_-]*$`), so groups key on the part before the first
`-`. A group forms only when at least 3 names share that prefix.

**Files:**
- Create: `ui/src/lib/wsGroups.ts`, `ui/src/lib/wsGroups.test.ts`
- Create: `ui/src/lib/hiddenWorkspaces.svelte.ts`
- Modify: `ui/src/lib/Sidebar.svelte` (the `{#each workspace.names …}` block, lines 144-200)
- Modify: `ui/src/lib/Sidebar.test.ts`
- Modify: `ui/src/lib/pages/OverviewPage.svelte` (Manage panel)

**Interfaces:**
- Produces: `type WsEntry = { kind: 'ws'; name: string } | { kind: 'group'; prefix: string; names: string[] }`;
  `groupWorkspaces(names: readonly string[], opts: { hidden: readonly string[]; open: string; min?: number }): { entries: WsEntry[]; hidden: string[] }`
- Produces: `hiddenWorkspaces: { names: string[] }` ($state),
  `setHidden(name: string, hidden: boolean): void`

- [ ] **Step 1: Write the failing tests**

`ui/src/lib/wsGroups.test.ts`:
```ts
import { describe, expect, it } from 'vitest'
import { groupWorkspaces } from './wsGroups'

const names = [
  'acorn-wars',
  'regression-co',
  'h3-quality',
  'regression-mod',
  'regression-smoke',
  'sci0fi',
]

describe('groupWorkspaces', () => {
  it('groups three or more names that share a prefix, where the first one stood', () => {
    expect(groupWorkspaces(names, { hidden: [], open: 'h3-quality' })).toEqual({
      entries: [
        { kind: 'ws', name: 'acorn-wars' },
        {
          kind: 'group',
          prefix: 'regression',
          names: ['regression-co', 'regression-mod', 'regression-smoke'],
        },
        { kind: 'ws', name: 'h3-quality' },
        { kind: 'ws', name: 'sci0fi' },
      ],
      hidden: [],
    })
  })
  it('leaves hidden names out, except the open one', () => {
    const r = groupWorkspaces(names, {
      hidden: ['sci0fi', 'h3-quality'],
      open: 'h3-quality',
    })
    expect(r.hidden).toEqual(['sci0fi'])
    expect(r.entries).toContainEqual({ kind: 'ws', name: 'h3-quality' })
    expect(r.entries).not.toContainEqual({ kind: 'ws', name: 'sci0fi' })
  })
  it('does not group a pair', () => {
    const r = groupWorkspaces(['a-1', 'a-2', 'b'], { hidden: [], open: 'b' })
    expect(r.entries.every((e) => e.kind === 'ws')).toBe(true)
  })
  it('hiding names can dissolve a group below the minimum', () => {
    const r = groupWorkspaces(names, {
      hidden: ['regression-co'],
      open: 'h3-quality',
    })
    expect(r.entries.some((e) => e.kind === 'group')).toBe(false)
  })
})
```

Add to `Sidebar.test.ts`, following that file's existing setup for
`workspace.names`:
```ts
it('folds a run of same-prefix workspaces into one group row', async () => {
  workspace.names = ['regression-a', 'regression-b', 'regression-c', 'main']
  workspace.current = 'main'
  render(Sidebar, { props: { collapsed: false, onToggle: () => {} } })
  const group = screen.getByRole('button', { name: /regression/ })
  expect(group.getAttribute('aria-expanded')).toBe('false')
  expect(screen.queryByText('regression-a')).toBeNull()
  await fireEvent.click(group)
  expect(screen.getByText('regression-a')).toBeTruthy()
})
```
Use whatever props and state setup `Sidebar.test.ts` already uses. The names
above are the only new part.

- [ ] **Step 2: Run them to see them fail**

Run: `cd ui && npx vitest run src/lib/wsGroups.test.ts src/lib/Sidebar.test.ts`
Expected: FAIL

- [ ] **Step 3: Implement**

`ui/src/lib/wsGroups.ts`:
```ts
/** The sidebar's workspace list, folded: names that share a prefix (the
 * part before the first '-') become one group row once there are `min` of
 * them, at the place the first one stood; names this browser hid are left
 * out. The open workspace is never hidden - hiding the page you are on
 * would strand you. */
export type WsEntry =
  | { kind: 'ws'; name: string }
  | { kind: 'group'; prefix: string; names: string[] }

const prefixOf = (name: string) => {
  const at = name.indexOf('-')
  return at > 0 ? name.slice(0, at) : null
}

export function groupWorkspaces(
  names: readonly string[],
  {
    hidden,
    open,
    min = 3,
  }: { hidden: readonly string[]; open: string; min?: number },
): { entries: WsEntry[]; hidden: string[] } {
  const shown = names.filter((n) => n === open || !hidden.includes(n))
  const counts = new Map<string, number>()
  for (const n of shown) {
    const p = prefixOf(n)
    if (p) counts.set(p, (counts.get(p) ?? 0) + 1)
  }
  const entries: WsEntry[] = []
  const groups = new Map<string, { kind: 'group'; prefix: string; names: string[] }>()
  for (const n of shown) {
    const p = prefixOf(n)
    if (p && (counts.get(p) ?? 0) >= min) {
      let group = groups.get(p)
      if (!group) {
        group = { kind: 'group', prefix: p, names: [] }
        groups.set(p, group)
        entries.push(group)
      }
      group.names.push(n)
    } else entries.push({ kind: 'ws', name: n })
  }
  return {
    entries,
    hidden: names.filter((n) => n !== open && hidden.includes(n)),
  }
}
```

`ui/src/lib/hiddenWorkspaces.svelte.ts`:
```ts
/** Workspaces this browser keeps out of the sidebar - a viewing choice, so
 * it is local, not a change to the workspace on the server. */
import { storageGet, storageSet } from './storage'

export const hiddenWorkspaces = $state({
  names: storageGet<string[]>('hidden-workspaces', []),
})

export function setHidden(name: string, hidden: boolean): void {
  const rest = hiddenWorkspaces.names.filter((n) => n !== name)
  hiddenWorkspaces.names = hidden ? [...rest, name] : rest
  storageSet('hidden-workspaces', hiddenWorkspaces.names)
}
```

`Sidebar.svelte`:
1. Pull the body of the existing `{#each … as name (name)}` (the
   open-workspace `<div class="ws open">` branch and the shut `<a>` branch)
   into a snippet `{#snippet wsRow(name: string)} … {/snippet}`, unchanged.
2. Add these imports: `groupWorkspaces` from `./wsGroups`,
   `hiddenWorkspaces` from `./hiddenWorkspaces.svelte`, `storageGet` and
   `storageSet` from `./storage`, and `Eye` from `@lucide/svelte`.
3. Add this state:
```ts
  const folded = $derived(
    groupWorkspaces(workspace.names ?? [workspace.current], {
      hidden: hiddenWorkspaces.names,
      open: workspace.current,
    }),
  )
  let openGroups = $state(storageGet<string[]>('open-ws-groups', []))
  function toggleGroup(prefix: string) {
    openGroups = openGroups.includes(prefix)
      ? openGroups.filter((p) => p !== prefix)
      : [...openGroups, prefix]
    storageSet('open-ws-groups', openGroups)
  }
  let showHidden = $state(false)
```
4. Replace the `{#each}` with:
```svelte
      {#each folded.entries as entry (entry.kind === 'ws' ? entry.name : `group:${entry.prefix}`)}
        {#if entry.kind === 'ws'}
          {@render wsRow(entry.name)}
        {:else}
          {@const isOpen =
            openGroups.includes(entry.prefix) ||
            entry.names.includes(workspace.current)}
          {#if collapsed}
            {#each entry.names as name (name)}{@render wsRow(name)}{/each}
          {:else}
            <button
              class="plain ws shut wsgroup"
              aria-expanded={isOpen}
              onclick={() => toggleGroup(entry.prefix)}
              title="{entry.names.length} workspaces named {entry.prefix}-…"
            >
              {#if isOpen}<ChevronDown size={13} class="chev" />{:else}<ChevronRight
                  size={13}
                  class="chev"
                />{/if}
              <span class="name">{entry.prefix}-…</span>
              <span class="num muted size">{entry.names.length}</span>
            </button>
            {#if isOpen}
              <div class="groupkids">
                {#each entry.names as name (name)}{@render wsRow(name)}{/each}
              </div>
            {/if}
          {/if}
        {/if}
      {/each}
      {#if folded.hidden.length && !collapsed}
        <button
          class="bare muted hiddentoggle"
          aria-expanded={showHidden}
          onclick={() => (showHidden = !showHidden)}
          ><Eye size={13} />{folded.hidden.length} hidden</button
        >
        {#if showHidden}
          <div class="groupkids dim">
            {#each folded.hidden as name (name)}{@render wsRow(name)}{/each}
          </div>
        {/if}
      {/if}
```
5. Styles:
   `.groupkids { padding-left: var(--space-3); }`,
   `.dim { opacity: 0.7; }` (check AA contrast against `--muted` in both
   themes; if it fails, drop the opacity and rely on the label),
   `.hiddentoggle { display: inline-flex; gap: 0.35rem; align-items: center; font-size: var(--t-sm); padding: var(--space-1) var(--space-2); }`,
   and `.wsgroup { width: 100%; text-align: left; }`.

`OverviewPage.svelte`, in the Manage panel, before the delete button:
```svelte
    <button
      class="quiet withicon"
      onclick={() => setHidden(name, !hiddenWorkspaces.names.includes(name))}
      disabled={name === DEFAULT_WORKSPACE}
      title="a choice for this browser only - the workspace itself is unchanged"
    >
      {#if hiddenWorkspaces.names.includes(name)}<Eye size={14} />Show in sidebar{:else}<EyeOff
          size={14}
        />Hide from sidebar{/if}
    </button>
```
(import `Eye, EyeOff` and `hiddenWorkspaces, setHidden`).

- [ ] **Step 4: Run the tests and checks**

Run: `cd ui && npx vitest run src/lib/wsGroups.test.ts src/lib/Sidebar.test.ts && npm run check && npm run lint && npm test`
Expected: PASS (design-rules included: the group row uses no `--select`)

- [ ] **Step 5: Commit**

```bash
git add ui/src/lib/wsGroups.ts ui/src/lib/wsGroups.test.ts ui/src/lib/hiddenWorkspaces.svelte.ts ui/src/lib/Sidebar.svelte ui/src/lib/Sidebar.test.ts ui/src/lib/pages/OverviewPage.svelte
git commit -m "feat(ui): group same-prefix workspaces in the sidebar; hide workspaces per browser"
```

---

## Phase C - server and UI

### Task 10: Star outputs

Stars are saved in the run's existing `notes.json` as `"starred": [names]`.
That file is already the server-only sidecar for that run's per-file data,
and it already travels with the run. `read_notes` keeps its current return
shape, because existing tests compare it exactly. Stars get their own reader,
and the shared write path keeps both kinds of data.

**Files:**
- Modify: `dw/notes.py`
- Modify: `dw/server/routes/media.py` (a `StarRequest` model and a `put_star` route next to `put_note`)
- Modify: `dw/server/routes/gallery.py` (listing, lines ~124-131)
- Modify: `dw/server/api_models.py` (`GalleryFile.starred`, new `StarSet`)
- Create: `tests/test_stars.py`
- Modify: `docs/SERVER.md` (notes section, ~line 804)
- Regenerate: `ui/src/lib/generated/openapi.json`, `api-schema.ts`
- Modify: `ui/src/lib/types.ts` (`StarSet`), `ui/src/lib/api.ts` (`setStarred`)
- Modify: `ui/src/lib/gallery/GalleryDetail.svelte`, `ui/src/lib/pages/GalleryPage.svelte` (+ test), `ui/src/lib/KeyboardHelp.svelte`

**Interfaces:**
- Produces (Python): `read_starred(directory) -> set[str]`,
  `set_file_starred(directory, name, starred: bool) -> bool`. The
  `drop_file_note(directory, name)` function now also clears the star.
- Produces (HTTP): `PUT /api/stars` with body `{name: str, starred: bool}`,
  returning `{name, starred}`. A run directory or a `workspace:`/`asset:`
  target gets 400; a flat-layout file gets 409.
- Produces (TS): `api.setStarred(name: string, starred: boolean): Promise<StarSet>`.
  `GalleryFile.starred?: boolean`.

- [ ] **Step 1: Write the failing server tests**

`tests/test_stars.py`:
```python
"""Stars: the "keep" mark on an output file, in the run's notes.json beside
its notes (dw/notes.py), set through PUT /api/stars and carried on the
gallery listing."""

from dw import notes as N

from .test_notes_api import (  # noqa: F401 - fixtures
    client,
    make_run,
    output_root,
    run_name,
    studio,
)


def test_star_round_trip_keeps_notes(tmp_path):
    N.set_file_note(str(tmp_path), "a.png", "keeper")
    assert N.set_file_starred(str(tmp_path), "a.png", True) is True
    assert N.read_starred(str(tmp_path)) == {"a.png"}
    # A note written after the star leaves it, and the other way round
    N.set_run_note(str(tmp_path), "run note")
    assert N.read_starred(str(tmp_path)) == {"a.png"}
    assert N.read_notes(str(tmp_path))["files"] == {"a.png": "keeper"}


def test_unstarring_the_last_file_with_no_notes_removes_the_sidecar(tmp_path):
    N.set_file_starred(str(tmp_path), "a.png", True)
    N.set_file_starred(str(tmp_path), "a.png", False)
    assert not (tmp_path / N.NOTES_FILE_NAME).exists()


def test_dropping_a_file_drops_its_star(tmp_path):
    N.set_file_starred(str(tmp_path), "a.png", True)
    N.drop_file_note(str(tmp_path), "a.png")
    assert N.read_starred(str(tmp_path)) == set()


def test_put_star_and_the_listing_carries_it(client, output_root, make_run):
    run_dir = make_run("wf", ["a.png", "b.png"])
    run = run_name(run_dir, output_root)
    r = client.put("/api/stars", json={"name": f"{run}/a.png", "starred": True})
    assert r.status_code == 200
    assert r.json() == {"name": f"{run}/a.png", "starred": True}
    files = {f["label"]: f for f in client.get("/api/gallery").json()["files"]}
    assert files["a.png"]["starred"] is True
    assert "starred" not in files["b.png"]
    client.put("/api/stars", json={"name": f"{run}/a.png", "starred": False})
    files = {f["label"]: f for f in client.get("/api/gallery").json()["files"]}
    assert "starred" not in files["a.png"]


def test_only_an_output_file_can_be_starred(client, output_root, make_run):
    run = run_name(make_run("wf", ["a.png"]), output_root)
    for name in (run, "workspace:default", "asset:x.png"):
        r = client.put("/api/stars", json={"name": name, "starred": True})
        assert r.status_code == 400, name


def test_put_star_refuses_traversal(client):
    r = client.put("/api/stars", json={"name": "../../etc/passwd", "starred": True})
    assert r.status_code in (400, 404)
```

- [ ] **Step 2: Run them to see them fail**

Run: `pytest tests/test_stars.py -v -p no:xdist`
Expected: FAIL (`set_file_starred` is not defined; the route returns 404/405)

- [ ] **Step 3: Implement the store (`dw/notes.py`)**

Update the module docstring's `notes.json` bullet to:
`{"run": ..., "files": {...}, "starred": [<name>, ...]}` - starred is the
"keep" mark on an output file.

Replace `_store` and `_update`, and update the callers:
```python
def read_starred(directory):
    """The names starred in `directory`'s notes.json, as a set."""
    starred = _read_json(os.path.join(directory, NOTES_FILE_NAME)).get("starred")
    if not isinstance(starred, list):
        return set()
    return {name for name in starred if isinstance(name, str)}


def _store(directory, notes, starred):
    path = os.path.join(directory, NOTES_FILE_NAME)
    if notes["run"] is None and not notes["files"] and not starred:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        return
    data = {k: v for k, v in notes.items() if v}
    if starred:
        data["starred"] = sorted(starred)
    write_json_atomic(path, data)


def _update(directory, change):
    """Read, change and write notes.json under its lock. `change` gets the
    notes and the starred set and edits them in place - both, so a write of
    one never drops the other."""
    with _lock(directory):
        notes = read_notes(directory)
        starred = read_starred(directory)
        change(notes, starred)
        _store(directory, notes, starred)
```
`set_run_note` becomes
`_update(directory, lambda notes, _starred: notes.__setitem__("run", note))`.
`set_file_note`'s inner `change` becomes `def change(notes, _starred):`.
Add:
```python
def set_file_starred(directory, name, starred):
    starred = bool(starred)

    def change(_notes, names):
        if starred:
            names.add(name)
        else:
            names.discard(name)

    _update(directory, change)
    return starred


def drop_file_note(directory, name):
    """Forget a deleted file: its note and its star."""
    if os.path.exists(os.path.join(directory, NOTES_FILE_NAME)):

        def change(notes, names):
            notes["files"].pop(name, None)
            names.discard(name)

        _update(directory, change)
```
(this replaces the old `drop_file_note`).

- [ ] **Step 4: Implement the route, model and listing**

`dw/server/api_models.py`:
- In `GalleryFile`, after `note`, add:
  `starred: bool = sometimes("True when the file is starred; absent otherwise.")`
- Next to `NoteSet`, add:
```python
class StarSet(ApiModel):
    name: str
    starred: bool
```

`dw/server/routes/media.py`, after `put_note`:
```python
class StarRequest(BaseModel):
    name: str = Field(description="A gallery file name")
    starred: bool


@router.put("/api/stars", response_model=StarSet)
def put_star(
    request: Request, body: StarRequest, ws: Workspace = Depends(selected_workspace)
):
    """Star or unstar one output file - the "keep" mark a person culls a run
    with. It lives in the run's notes.json beside the notes, so it is
    resolved exactly as a file note is, and only a file in a run directory
    can carry one."""
    if is_ref(WORKSPACE, body.name) or is_asset_reference(body.name):
        raise HTTPException(status_code=400, detail="Only an output file can be starred")
    directory, name, kind = resolve_note_target(request.app.state, ws, body.name)
    if kind != "file":
        raise HTTPException(status_code=400, detail="Only an output file can be starred")
    try:
        starred = set_file_starred(directory, name, body.starred)
    except OSError:
        raise internal_error(f"Could not write the star on {body.name!r}")
    return {"name": body.name, "starred": starred}
```
Import `set_file_starred` with the other `..notes` imports and `StarSet` with
the other `..api_models` imports. These have to be added to existing import
lists in that file, so check them.

`dw/server/routes/gallery.py`: import `read_starred`. Next to `notes_by_run`,
add `starred_by_run = {}`. In the `if split:` block:
```python
            if split[0] not in starred_by_run:
                starred_by_run[split[0]] = read_starred(split[0])
            if split[1] in starred_by_run[split[0]]:
                entry["starred"] = True
```
Find where `notes_by_run = {}` is set and add `starred_by_run = {}` next to it.

`docs/SERVER.md`, after the notes paragraph (~line 807):
```markdown
`PUT /api/stars` with `{"name": <gallery file name>, "starred": true|false}`
stars or unstars one output file - the "keep" mark for culling a run. A star
is stored in the run's `notes.json` (`"starred": [...]`) beside its notes, and
the gallery listing carries `"starred": true` on a starred file. Only a file
in a run directory can be starred; a run, asset or workspace target is a 400.
```

- [ ] **Step 5: Run the server tests**

Run: `pytest tests/test_stars.py tests/test_notes.py tests/test_notes_api.py tests/test_notes_bookkeeping.py -v -p no:xdist`
Expected: PASS

- [ ] **Step 6: Regenerate the UI types**

Run: `python scripts/dump_openapi.py && cd ui && npm run gen:api`
Expected: `openapi.json` and `api-schema.ts` now contain `StarSet` and `GalleryFile.starred`.

- [ ] **Step 7: Write the failing UI test**

In `GalleryPage.test.ts`, add `setStarred` to the mocked `api` object, using
the file's `vi.hoisted` pattern:
`const setStarred = vi.hoisted(() => vi.fn((name: string, starred: boolean) => Promise.resolve({ name, starred })))`.
Then add:
```ts
it('stars the open file and can filter to starred files', async () => {
  render(GalleryPage)
  await openFile('a.png')
  await fireEvent.click(screen.getByRole('button', { name: 'star this file' }))
  expect(setStarred).toHaveBeenCalledWith('a.png', true)
  await fireEvent.click(screen.getByRole('button', { name: 'Starred' }))
  expect(screen.queryByTitle('b.png')).toBeNull()
})
```

Run: `cd ui && npx vitest run src/lib/pages/GalleryPage.test.ts`
Expected: FAIL

- [ ] **Step 8: Implement the UI**

`types.ts`: `export type StarSet = Schemas['StarSet']`.

`api.ts`, after `deleteOutput`:
```ts
  setStarred: (name: string, starred: boolean) =>
    request<StarSet>('/api/stars', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, starred }),
    }),
```
(import the `StarSet` type.)

`GalleryDetail.svelte`:
- Add the props `starred: boolean` and `onstar: () => void`.
- Import `Star`.
- Before the delete button, add:
```svelte
    <button
      class="quiet icon"
      onclick={onstar}
      aria-pressed={starred}
      aria-label={starred ? 'unstar this file' : 'star this file'}
      title={starred ? 'starred - press S to unstar' : 'star to keep - press S'}
      ><Star size={14} fill={starred ? 'currentColor' : 'none'} /></button
    >
```

`GalleryPage.svelte`:
- Add state `let starredOnly = $state(false)`.
- `filterActive` adds `|| starredOnly`.
- `visible`'s filter adds `&& (!starredOnly || f.starred)`.
- Add:
```ts
  async function toggleStar() {
    if (!selected) return
    const name = selected.name
    const next = !selected.starred
    try {
      await api.setStarred(name, next)
      files = files.map((f) => (f.name === name ? { ...f, starred: next } : f))
      if (selected?.name === name) selected = { ...selected, starred: next }
    } catch (e) {
      notify.error(e instanceof Error ? e.message : String(e))
    }
  }
```
- In the keydown handler, after the arrow branch, add
  `else if (e.key === 's' && !e.metaKey && !e.ctrlKey) { e.preventDefault(); void toggleStar() }`.
- Pass `starred={!!selected.starred}` and `onstar={toggleStar}` to `<GalleryDetail>`.
- After the role chips, add
  `<button class="rolechip" class:on={starredOnly} aria-pressed={starredOnly} onclick={() => (starredOnly = !starredOnly)} title="only the files you starred">Starred</button>`.
- On the tile, next to the note marker (`aria-label="Has a note"`), add
  `{#if file.starred}<span class="badge" aria-label="Starred"><Star size={11} fill="currentColor" /></span>{/if}`,
  copying the note marker's classes.

`KeyboardHelp.svelte`: add
`<dt><kbd>S</kbd></dt><dd>star / unstar the open file (gallery details)</dd>`.

- [ ] **Step 9: Run all checks**

Run: `cd ui && npm run check && npm run lint && npm test && cd .. && pytest tests/ -v`
Expected: PASS

- [ ] **Step 10: Commit**

```bash
git add dw/notes.py dw/server/routes/media.py dw/server/routes/gallery.py dw/server/api_models.py tests/test_stars.py docs/SERVER.md ui/src/lib/generated ui/src/lib/types.ts ui/src/lib/api.ts ui/src/lib/gallery/GalleryDetail.svelte ui/src/lib/pages/GalleryPage.svelte ui/src/lib/pages/GalleryPage.test.ts ui/src/lib/KeyboardHelp.svelte
git commit -m "feat: star output files to keep them; gallery filters to starred"
```

---

### Task 11: Rerun lineage

**Files:**
- Modify: `dw/server/jobs.py` (`submit` param plus spec around line 320; `rerun` passes it, ~line 545)
- Modify: `dw/server/job_record.py` (`__init__` around line 94; `summary()`)
- Modify: `dw/server/job_history.py` (migration, `record`, `recent_summaries`, `get`, `_to_detail`)
- Modify: `dw/server/api_models.py` (`JobSummary.rerun_of`)
- Create: `tests/test_rerun_lineage.py`
- Regenerate: UI types
- Modify: `ui/src/lib/pages/JobsPage.svelte`, `ui/src/lib/job/JobHeader.svelte`

**Interfaces:**
- Produces: `JobSummary.rerun_of?: string`, which `JobDetail` inherits. It is
  present only on a rerun and is kept in history.

- [ ] **Step 1: Write the failing tests**

`tests/test_rerun_lineage.py`:
```python
"""A rerun names the job it reran (`rerun_of`), live and from history."""

import sqlite3

from fastapi.testclient import TestClient

from dw.server.app import create_app
from dw.server.job_history import JobHistory
from dw.server.jobs import JobManager

from .test_server import ScriptedWorkerManager, success_script, valid_workflow
from .test_server import wait_for_status


def make_client(tmp_path):
    (tmp_path / "workflows").mkdir(exist_ok=True)
    manager = JobManager(
        str(tmp_path / "outputs"),
        worker_manager=ScriptedWorkerManager(success_script),
        history_path=str(tmp_path / "jobs.sqlite"),
    )
    app = create_app(
        workflow_dir=str(tmp_path / "workflows"),
        output_dir=str(tmp_path / "outputs"),
        job_manager=manager,
    )
    return TestClient(app, base_url="http://localhost")


def run_and_rerun(client):
    original = client.post(
        "/api/jobs", json={"workflow": valid_workflow("lineage"), "arguments": {}}
    ).json()["id"]
    wait_for_status(client, original, ["succeeded"])
    rerun = client.post(f"/api/jobs/{original}/rerun").json()["id"]
    wait_for_status(client, rerun, ["succeeded"])
    return original, rerun


def test_a_rerun_names_the_job_it_reran(tmp_path):
    with make_client(tmp_path) as client:
        original, rerun = run_and_rerun(client)
        assert client.get(f"/api/jobs/{rerun}").json()["rerun_of"] == original
        assert "rerun_of" not in client.get(f"/api/jobs/{original}").json()
        listed = {j["id"]: j for j in client.get("/api/jobs").json()["jobs"]}
        assert listed[rerun]["rerun_of"] == original


def test_the_lineage_survives_a_restart(tmp_path):
    with make_client(tmp_path) as client:
        original, rerun = run_and_rerun(client)
    with make_client(tmp_path) as client:
        assert client.get(f"/api/jobs/{rerun}").json()["rerun_of"] == original
        listed = {j["id"]: j for j in client.get("/api/jobs").json()["jobs"]}
        assert listed[rerun]["rerun_of"] == original


def test_an_older_history_database_gains_the_column(tmp_path):
    path = tmp_path / "old.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE jobs (id TEXT PRIMARY KEY, workflow TEXT, status TEXT,"
            " created_at REAL, started_at REAL, finished_at REAL, arguments TEXT,"
            " spec TEXT, manifest TEXT, warnings TEXT, error TEXT)"
        )
    JobHistory(path)
    with sqlite3.connect(path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(jobs)")}
    assert "rerun_of" in columns
```

- [ ] **Step 2: Run them to see them fail**

Run: `pytest tests/test_rerun_lineage.py -v -p no:xdist`
Expected: FAIL (KeyError `rerun_of`, and the column is missing)

- [ ] **Step 3: Implement**

`dw/server/jobs.py`:
- `submit` gains a keyword parameter `rerun_of=None`. Document it in the
  docstring: "the job this one reruns, recorded so the list can say so".
- After `if note: spec["note"] = note`, add
  `if rerun_of: spec["rerun_of"] = rerun_of`.
- In `rerun`, pass `rerun_of=job_id` to `self.submit(...)`.

`dw/server/job_record.py`:
- In `__init__`, after `self.note = spec.get("note")`, add
  `self.rerun_of = spec.get("rerun_of")`.
- In `summary()`, after the `note` entry, add
  `**({"rerun_of": self.rerun_of} if self.rerun_of else {}),`.

`dw/server/job_history.py`:
- Migration, after the `note` column block:
```python
            # The job a rerun reran. NULL for a fresh submit and for every
            # row before the column
            if "rerun_of" not in columns:
                connection.execute("ALTER TABLE jobs ADD COLUMN rerun_of TEXT")
```
- `record`: add `rerun_of` to the end of the column list, add one more `?`
  (25 in total), and add `getattr(job, "rerun_of", None),` as the last tuple
  value.
- `recent_summaries`: append `, rerun_of` after the `CASE … END` expression
  in the SELECT, so it is `row[15]`. In the dict, add
  `**({"rerun_of": row[15]} if row[15] else {}),`.
- `get`: append `, rerun_of` after `note` in the SELECT (`row[20]`). In
  `_to_detail`, add `**({"rerun_of": row[20]} if row[20] else {}),`.

`dw/server/api_models.py`, in `JobSummary` after `note`:
`rerun_of: str = sometimes("The job this one reran; absent for a job submitted fresh.")`

- [ ] **Step 4: Run the server tests**

Run: `pytest tests/test_rerun_lineage.py tests/test_rerun_new_seed.py tests/test_server_jobs.py tests/test_job_history_device.py -v -p no:xdist`
Expected: PASS

- [ ] **Step 5: Regenerate the UI types and show the lineage**

Run: `python scripts/dump_openapi.py && cd ui && npm run gen:api`

`JobsPage.svelte`, after the `run_version` chip:
```svelte
          {#if job.rerun_of}
            <span class="wschip muted" title="a rerun of {job.rerun_of}"
              >rerun</span
            >
          {/if}
```
(the row is itself an `<a>`, so the chip is not a link.)

`JobHeader.svelte`: next to the job id / `CopyButton` in the title row, add:
```svelte
    {#if job?.rerun_of}
      <span class="muted"
        >rerun of <a class="plain" href={wsHref(job.workspace, 'jobs', job.rerun_of)}
          ><code>{job.rerun_of}</code></a
        ></span
      >
    {/if}
```

Add to `JobsPage.test.ts`, following its existing listing mock:
```ts
it('marks a rerun', async () => {
  // a job in the mocked listing with rerun_of: 'abc'
  ...render as the other tests do...
  expect(screen.getByTitle('a rerun of abc')).toBeTruthy()
})
```
Fill in the setup by copying the nearest existing test in that file. The only
new part is the extra `rerun_of: 'abc'` field on one listed job.

- [ ] **Step 6: Run all checks**

Run: `cd ui && npm run check && npm run lint && npm test && cd .. && pytest tests/ -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add dw/server/jobs.py dw/server/job_record.py dw/server/job_history.py dw/server/api_models.py tests/test_rerun_lineage.py ui/src/lib/generated ui/src/lib/pages/JobsPage.svelte ui/src/lib/pages/JobsPage.test.ts ui/src/lib/job/JobHeader.svelte
git commit -m "feat: a rerun records and shows the job it reran"
```

---

### Task 12: Compare view

Pick 2–4 files in the gallery, then click **Compare**. They show side by side,
each with its version, model, seed, run time and note. Below the grid, a
"What differs" table lists only the arguments whose values differ between the
files. Videos play, pause and scrub together from one control.

**Files:**
- Create: `ui/src/lib/compare.ts`, `ui/src/lib/compare.test.ts`
- Create: `ui/src/lib/pages/ComparePage.svelte`, `ui/src/lib/pages/ComparePage.test.ts`
- Modify: `ui/src/lib/routes.ts` (`WS_SECTIONS` adds `'compare'`)
- Modify: `ui/src/App.svelte` (route branch)
- Modify: `ui/src/lib/BulkBar.svelte` (optional `oncompare`)
- Modify: `ui/src/lib/pages/GalleryPage.svelte` (pass `oncompare`)

**Interfaces:**
- Consumes: `api.gallery()`, `api.galleryMetadata(name)` (which returns
  `{ metadata, job, ... }`), `api.getJob(id)`, `describeOutputMeta`,
  `formatDuration`
- Produces: `differingArguments(metas: (Record<string, unknown> | null)[]): string[]`,
  `argumentOf(meta: Record<string, unknown> | null, key: string): unknown`,
  `valueText(value: unknown): string`, `COMPARE_MAX = 4`

- [ ] **Step 1: Write the failing tests**

`ui/src/lib/compare.test.ts`:
```ts
import { describe, expect, it } from 'vitest'
import { argumentOf, differingArguments, valueText } from './compare'

const meta = (args: Record<string, unknown>, extra = {}) => ({
  arguments: args,
  ...extra,
})

describe('differingArguments', () => {
  it('lists only the keys whose values differ, model and seed first', () => {
    expect(
      differingArguments([
        meta({ prompt: 'a', steps: 12, width: 1344 }, { model_name: 'h3', seed: 1 }),
        meta({ prompt: 'a', steps: 30, width: 1344 }, { model_name: 'h3', seed: 2 }),
      ]),
    ).toEqual(['seed', 'steps'])
  })
  it('a key one side lacks is a difference', () => {
    expect(
      differingArguments([meta({ lora: 'x' }), meta({})]),
    ).toEqual(['lora'])
  })
  it('compares structured values by content', () => {
    expect(
      differingArguments([meta({ q: { a: 1 } }), meta({ q: { a: 1 } })]),
    ).toEqual([])
  })
  it('missing metadata counts as unknown, not as different', () => {
    expect(differingArguments([meta({ steps: 1 }), null])).toEqual([])
  })
})

describe('argumentOf / valueText', () => {
  it('reads model and seed from the top level, the rest from arguments', () => {
    const m = meta({ steps: 12 }, { model_name: 'h3', seed: 9 })
    expect(argumentOf(m, 'model')).toBe('h3')
    expect(argumentOf(m, 'seed')).toBe(9)
    expect(argumentOf(m, 'steps')).toBe(12)
  })
  it('prints strings bare, the rest as JSON, cut long', () => {
    expect(valueText('a cat')).toBe('a cat')
    expect(valueText({ a: 1 })).toBe('{"a":1}')
    expect(valueText(undefined)).toBe('—')
    expect(valueText('x'.repeat(300))).toHaveLength(201)
  })
})
```

`ui/src/lib/pages/ComparePage.test.ts`:
```ts
import { cleanup, render, screen } from '@testing-library/svelte'
import { afterEach, describe, expect, it, vi } from 'vitest'

const file = (name: string, kind = 'image') => ({
  name,
  label: name.split('/').pop(),
  kind,
  url: `/outputs/${name}`,
  folder: 'wf',
  subfolder: '',
  run_id: 'r',
  size: 1,
  mtime: 1,
})

vi.mock('../api', () => ({
  api: {
    gallery: () =>
      Promise.resolve({ files: [file('wf/r1/a.png'), file('wf/r2/b.png')] }),
    galleryMetadata: (name: string) =>
      Promise.resolve({
        metadata: {
          model_name: 'h3',
          seed: name.includes('a') ? 1 : 2,
          arguments: { steps: 12 },
        },
        job: null,
      }),
    galleryThumbnailUrl: (n: string) => `/thumb/${n}`,
    getJob: () => Promise.reject(new Error('none')),
  },
}))

import ComparePage from './ComparePage.svelte'

describe('ComparePage', () => {
  afterEach(cleanup)
  it('shows each file and only the argument that differs', async () => {
    render(ComparePage, { props: { names: ['wf/r1/a.png', 'wf/r2/b.png'] } })
    expect(await screen.findByAltText('wf/r1/a.png')).toBeTruthy()
    expect(await screen.findByRole('rowheader', { name: 'seed' })).toBeTruthy()
    expect(screen.queryByRole('rowheader', { name: 'steps' })).toBeNull()
  })
  it('says when a file is gone', async () => {
    render(ComparePage, { props: { names: ['wf/r1/a.png', 'wf/gone.png'] } })
    expect(await screen.findByText(/no longer in the gallery/)).toBeTruthy()
  })
})
```

- [ ] **Step 2: Run them to see them fail**

Run: `cd ui && npx vitest run src/lib/compare.test.ts src/lib/pages/ComparePage.test.ts`
Expected: FAIL

- [ ] **Step 3: Implement the logic**

`ui/src/lib/compare.ts`:
```ts
/** What a side-by-side compare reads off each file's metadata: the
 * arguments that actually differ between the files. The rest are shared and
 * would only bury the one change the comparison is about. */

export const COMPARE_MAX = 4

type Meta = Record<string, unknown> | null

export function argumentOf(meta: Meta, key: string): unknown {
  if (!meta) return undefined
  if (key === 'model') return meta.model_name
  if (key === 'seed')
    return meta.seed ?? (meta.arguments as Record<string, unknown>)?.seed
  return (meta.arguments as Record<string, unknown> | undefined)?.[key]
}

const same = (a: unknown, b: unknown) => JSON.stringify(a) === JSON.stringify(b)

/** Keys that differ across the files that have metadata - model and seed
 * first, then the arguments alphabetically. A file whose metadata could
 * not be read is unknown, not different. */
export function differingArguments(metas: Meta[]): string[] {
  const known = metas.filter((m): m is Record<string, unknown> => !!m)
  if (known.length < 2) return []
  const keys = new Set<string>()
  for (const m of known)
    for (const k of Object.keys((m.arguments as object) ?? {})) keys.add(k)
  keys.delete('seed')
  const ordered = ['model', 'seed', ...[...keys].sort()]
  return ordered.filter((key) => {
    const first = argumentOf(known[0], key)
    return known.some((m) => !same(argumentOf(m, key), first))
  })
}

export function valueText(value: unknown): string {
  if (value === undefined || value === null) return '—'
  const text = typeof value === 'string' ? value : JSON.stringify(value)
  return text.length > 200 ? `${text.slice(0, 200)}…` : text
}
```

- [ ] **Step 4: Implement the page, the route and the action**

`routes.ts`: add `'compare'` to `WS_SECTIONS`, after `'gallery'`.
`WS_ITEMS` in `Sidebar.svelte` is a separate list, so no nav item appears.

`App.svelte`: import `ComparePage` and add, beside the gallery branch:
`{:else if view.section === 'compare'}<ComparePage names={view.rest} />`

`BulkBar.svelte`:
- Add the optional prop `oncompare?: () => void`.
- Import `Columns2` and `COMPARE_MAX`.
- Before the download button, add:
```svelte
    {#if oncompare}
      <button
        class="quiet icon"
        onclick={oncompare}
        disabled={busy || picks.size < 2 || picks.size > COMPARE_MAX}
        title={picks.size > COMPARE_MAX
          ? `compare up to ${COMPARE_MAX} at a time`
          : 'compare the selected side by side'}
        aria-label="compare the selected {noun}s side by side"
      >
        <Columns2 size={14} />
      </button>
    {/if}
```

`GalleryPage.svelte`: import `go` from `'../router.svelte'` and `workspace`
from `'../workspace.svelte'` (check which are already imported). Pass
`oncompare={() => go('ws', workspace.current, 'compare', ...picks.names)}`
to `<BulkBar>`.

`ui/src/lib/pages/ComparePage.svelte`:
```svelte
<script lang="ts">
  // Two to four outputs side by side - the seed sweep, the int4 vs int8
  // pair, the 12 vs 30 steps - with the arguments that differ called out
  // and every video on one transport, so a difference in motion is seen at
  // the same frame rather than remembered across two players
  import { Pause, Play } from '@lucide/svelte'
  import { api } from '../api'
  import { differingArguments, argumentOf, valueText } from '../compare'
  import { formatDuration } from '../format'
  import { wsHref } from '../routes'
  import type { GalleryFile } from '../types'
  import { workspace } from '../workspace.svelte'

  let { names }: { names: string[] } = $props()

  type Cell = {
    name: string
    file: GalleryFile | null
    meta: Record<string, unknown> | null
    took: number | null
  }
  let cells = $state<Cell[]>([])
  let loaded = $state(false)

  $effect(() => {
    const wanted = [...names]
    loaded = false
    void api
      .gallery()
      .then(async ({ files }) => {
        const byName = new Map(files.map((f) => [f.name, f]))
        cells = await Promise.all(
          wanted.map(async (name) => {
            const file = byName.get(name) ?? null
            if (!file) return { name, file, meta: null, took: null }
            const r = await api.galleryMetadata(name).catch(() => null)
            const job = r?.job ? await api.getJob(r.job).catch(() => null) : null
            const took =
              job?.started_at && job.finished_at
                ? job.finished_at - job.started_at
                : null
            return {
              name,
              file,
              meta: (r?.metadata as Record<string, unknown>) ?? null,
              took,
            }
          }),
        )
      })
      .finally(() => (loaded = true))
  })

  const differing = $derived(differingArguments(cells.map((c) => c.meta)))

  // One transport for every video: the first is the clock the slider reads
  let videos = $state<HTMLVideoElement[]>([])
  let playing = $state(false)
  let time = $state(0)
  let length = $state(0)
  const present = () => videos.filter(Boolean)
  function playPause() {
    playing = !playing
    for (const v of present()) void (playing ? v.play() : v.pause())
  }
  function seek(to: number) {
    time = to
    for (const v of present()) v.currentTime = to
  }
</script>

<div class="pagehead baseline">
  <h1>Compare</h1>
  <a class="plain muted" href={wsHref(workspace.current, 'gallery')}>back to the gallery</a>
</div>

{#if !loaded}
  <p class="muted">loading…</p>
{:else}
  {#if cells.some((c) => c.file?.kind === 'video')}
    <div class="transport">
      <button class="quiet icon" onclick={playPause} aria-label={playing ? 'pause all' : 'play all'}>
        {#if playing}<Pause size={14} />{:else}<Play size={14} />{/if}
      </button>
      <input
        type="range"
        min="0"
        max={length || 0}
        step="0.01"
        value={time}
        oninput={(e) => seek(Number(e.currentTarget.value))}
        aria-label="position in every video"
      />
      <span class="num muted">{time.toFixed(2)} s</span>
    </div>
  {/if}

  <div class="grid" style:--cols={cells.length}>
    {#each cells as cell, i (cell.name)}
      <figure class="cell">
        {#if !cell.file}
          <p class="muted">{cell.name} is no longer in the gallery.</p>
        {:else if cell.file.kind === 'video'}
          <!-- svelte-ignore a11y_media_has_caption -->
          <video
            bind:this={videos[i]}
            src={cell.file.url}
            loop
            muted
            playsinline
            ontimeupdate={(e) => {
              if (i === 0) time = e.currentTarget.currentTime
            }}
            onloadedmetadata={(e) => {
              length = Math.max(length, e.currentTarget.duration)
            }}
          ></video>
        {:else if cell.file.kind === 'image'}
          <img src={cell.file.url} alt={cell.name} />
        {:else}
          <p class="muted">{cell.file.label}</p>
        {/if}
        {#if cell.file}
          <figcaption>
            <span class="mono">{cell.file.label}</span>
            {#if cell.file.version}<span class="num muted">v{cell.file.version}</span>{/if}
            <span class="num muted">
              {valueText(argumentOf(cell.meta, 'model'))} · seed {valueText(
                argumentOf(cell.meta, 'seed'),
              )}{cell.took !== null ? ` · ${formatDuration(cell.took)}` : ''}
            </span>
            {#if cell.file.note}<span class="note">{cell.file.note}</span>{/if}
          </figcaption>
        {/if}
      </figure>
    {/each}
  </div>

  {#if differing.length}
    <section class="panel">
      <h2>What differs</h2>
      <table>
        <tbody>
          {#each differing as key (key)}
            <tr>
              <th scope="row">{key}</th>
              {#each cells as cell (cell.name)}
                <td class="mono">{valueText(argumentOf(cell.meta, key))}</td>
              {/each}
            </tr>
          {/each}
        </tbody>
      </table>
    </section>
  {:else if cells.filter((c) => c.meta).length > 1}
    <p class="muted">Same model, seed and arguments - the files differ only in what the run did.</p>
  {/if}
{/if}

<style>
  .grid {
    display: grid;
    grid-template-columns: repeat(var(--cols), minmax(0, 1fr));
    gap: var(--space-3);
    margin-bottom: var(--space-4);
  }
  @media (max-width: 640px) {
    .grid {
      grid-template-columns: 1fr;
    }
  }
  .cell {
    margin: 0;
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
  }
  .cell img,
  .cell video {
    width: 100%;
    background: var(--panel-2);
    border: 1px solid var(--line);
  }
  figcaption {
    display: flex;
    flex-wrap: wrap;
    gap: var(--space-2);
    font-size: var(--t-sm);
  }
  .mono {
    font-family: var(--font-mono);
  }
  .note {
    flex-basis: 100%;
    white-space: pre-wrap;
  }
  .transport {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    margin-bottom: var(--space-3);
  }
  .transport input {
    flex: 1;
  }
  table {
    border-collapse: collapse;
    width: 100%;
    font-size: var(--t-sm);
  }
  th {
    text-align: left;
    font-family: var(--font-mono);
    color: var(--muted);
    padding-right: var(--space-3);
    white-space: nowrap;
  }
  td {
    padding: var(--space-1) var(--space-2);
    border-top: 1px solid var(--line);
    overflow-wrap: anywhere;
  }
</style>
```
If Task 5 has landed, swap the `loading…` line for
`<Skeleton tiles={names.length} label="loading the comparison" />`.

- [ ] **Step 5: Run the tests and checks**

Run: `cd ui && npx vitest run src/lib/compare.test.ts src/lib/pages/ComparePage.test.ts src/lib/pages/GalleryPage.test.ts && npm run check && npm run lint && npm test`
Expected: PASS. Then check by hand on `npm run dev`: tick two videos, click
Compare, and confirm play, pause and scrub move both together.

- [ ] **Step 6: Commit**

```bash
git add ui/src/lib/compare.ts ui/src/lib/compare.test.ts ui/src/lib/pages/ComparePage.svelte ui/src/lib/pages/ComparePage.test.ts ui/src/lib/routes.ts ui/src/App.svelte ui/src/lib/BulkBar.svelte ui/src/lib/pages/GalleryPage.svelte
git commit -m "feat(ui): compare 2-4 outputs side by side with synced video and the differing arguments"
```

---

## Finish

- [ ] From `ui/`: `npm run check && npm run lint && npm test && npx playwright test`.
  Playwright starts its own server, so stop `dw.serve` on the same port first.
- [ ] From the repo root: `pytest tests/ -v`
- [ ] Rebuild the UI (`cd ui && npm run build`) and restart `dw.serve` so Tasks
  10 and 11 take effect.
- [ ] Optional follow-up, not in this plan: rewrite the long h3-quality
  workspace note as Markdown (RULE / TIERS / LESSONS / PERF headings with
  lists) so Task 7 shows off on real content. The MCP `set_note` tool can do
  this.
