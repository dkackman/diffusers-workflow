/** What the tab title and the finish alert read off the 5-second health
 * poll. A job runs for minutes to half an hour, usually in a tab nobody is
 * looking at, so the title is where its state has to be. */
import { formatDuration } from './format'
import { BASE_TITLE, pageTitle } from './pageFrame.svelte'
import { workflowName } from './terms'
import type { HealthInfo, JobDetail } from './types'

export { BASE_TITLE }

export type TitleJob = Pick<
  JobDetail,
  'workflow' | 'workflow_name' | 'progress'
>

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

export type Watch = {
  watched: string[] | null
  ended: string[]
  running: string[]
}

/** One health poll's step: what is running now and which watched jobs ended.
 * The first poll (`watched` null) only learns what is running - a job
 * already going when the page opened has not "just finished" when the next
 * poll misses it - unless its id was adopted.
 *
 * The health reading counts queued jobs but does not name them, so an
 * adopted id that is not running can only be called ended when nothing is
 * queued; with a queue it may still be waiting, and is held rather than
 * re-read every poll - except on a `recheck` reading, which reads them all
 * (a still-queued one is adopted again by the caller). A failed poll (`info` null) says nothing about the
 * jobs: reading it as "none running" would announce every running job as
 * finished, so it leaves the state as it was. */
const holdsAdopted = (info: HealthInfo, recheck: boolean) =>
  !recheck && (info.queued ?? 0) > 0

export function nextWatch(
  watched: string[] | null,
  info: HealthInfo | null,
  adopted: ReadonlySet<string> = new Set(),
  recheck = false,
): Watch {
  if (!info) return { watched, ended: [], running: watched ?? [] }
  const running = runningIds(info)
  const mine = holdsAdopted(info, recheck) ? [] : [...adopted]
  const before = [...new Set([...(watched ?? []), ...mine])]
  const ended =
    watched === null && !mine.length ? [] : endedSince(before, running)
  return { watched: running, ended, running }
}

/** Ids a submit, rerun or enhance returned. A job that starts and ends
 * between two polls is never in `running`, so the id the server handed back
 * is what makes it watched from the moment it was submitted. Insertion
 * order is age; the oldest go first past the cap. */
const adopted = new Set<string>()
const ADOPTED_CAP = 50

export const adoptedIds = (): ReadonlySet<string> => adopted

export function adoptJob(id: string): void {
  adopted.delete(id)
  adopted.add(id)
  for (const old of adopted) {
    if (adopted.size <= ADOPTED_CAP) break
    adopted.delete(old)
  }
}

/** Announced, seen running, in flight, or gone from the server. */
export function dropAdopted(id: string): void {
  adopted.delete(id)
}

export function clearAdopted(): void {
  adopted.clear()
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

// A job is named by its workflow's file, never the JSON id (ui/GLOSSARY.md)
const leaf = workflowName

/** `base` is the page's own title (`pageTitle`): the job's state goes in
 * front of it, so the tab still says which page it is. */
export function tabTitle(
  job: TitleJob | null,
  othersRunning = 0,
  base = pageTitle(),
): string {
  if (!job) return base
  const pct = percentDone(job.progress)
  const more = othersRunning > 0 ? ` +${othersRunning}` : ''
  return `▶ ${pct === null ? '' : `${pct}% · `}${leaf(job)}${more} — ${base}`
}

const MARKS: Record<string, string> = {
  succeeded: '✓',
  failed: '✗',
  cancelled: '⊘',
}

export function endedTitle(
  job: TitleJob & { status: string },
  base = pageTitle(),
): string {
  return `${MARKS[job.status] ?? '•'} ${leaf(job)} — ${base}`
}

/** "ref-to-video succeeded in 12m 54s" - the toast and the alert body. */
export function finishedText(job: JobDetail): string {
  const took =
    job.started_at && job.finished_at
      ? ` in ${formatDuration(job.finished_at - job.started_at)}`
      : ''
  return `${leaf(job)} ${job.status}${took}`
}
