/** The alert a finished job raises in a tab nobody is looking at.
 *
 * A desktop notification needs a secure context - https, or localhost - and
 * this UI is routinely opened over plain http from another machine
 * (http://<gpu-box>:8765), where the Notification API is not offered. There
 * the alert is a short chime instead. The browser only lets a page make
 * sound after the person has interacted with it, so the chime is unlocked on
 * the click that turns alerts on, and again on the first click after a
 * reload. */
import { api, TERMINAL_STATUSES } from './api'
import {
  adopted,
  endedTitle,
  finishedText,
  nextWatch,
  tabTitle,
} from './jobWatch'
import { go } from './router.svelte'
import { notify } from './toast'
import { wsHref } from './routes'
import type { HealthInfo, JobDetail } from './types'

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

// The tab title and the finish alert: what a poll means for the watched
// jobs is nextWatch's to decide, this holds the state it leaves.
/** `alertsOn` is passed in because alerts.svelte.ts imports this module. */
export type FollowContext = {
  onItsPage: (id: string) => boolean
  alertsOn?: () => boolean
}

let watched: string[] | null = null
let endedMark: string | null = null
let titleJob: JobDetail | null = null
let others = 0

const showTitle = () => {
  document.title = endedMark ?? tabTitle(titleJob, others)
}

/** The tab is back in front: the ended mark has been seen. */
export function clearEndedMark(): void {
  if (document.hidden) return
  endedMark = null
  showTitle()
}

export function resetFollow(): void {
  watched = null
  endedMark = titleJob = null
  others = 0
  adopted.clear()
}

function announce(job: JobDetail, ctx: FollowContext): void {
  const text = finishedText(job)
  if (document.hidden) endedMark = endedTitle(job)
  // The job's own page already shows its outcome
  if (!ctx.onItsPage(job.id)) {
    const open = {
      label: 'Open',
      onClick: () => go('ws', job.workspace, 'jobs', job.id),
    }
    if (job.status === 'succeeded') notify.news(text, open)
    else notify.error(text, undefined, open)
  }
  if (ctx.alertsOn?.() && document.hidden)
    alertFinished(text, wsHref(job.workspace, 'jobs', job.id))
}

/** One health reading's worth of watching: the title, and a finish toast
 * (and alert) for each watched job that ended. */
export async function followJobs(
  info: HealthInfo | null,
  ctx: FollowContext,
): Promise<void> {
  const next = nextWatch(watched, info, adopted)
  watched = next.watched
  // A failed poll leaves the title showing what it last knew
  if (info) {
    for (const id of next.running) adopted.delete(id)
    others = Math.max(0, next.running.length - 1)
    titleJob = next.running[0]
      ? await api.getJob(next.running[0]).catch(() => null)
      : null
    showTitle()
  }
  for (const id of next.ended) {
    const job = await api.getJob(id).catch(() => null)
    if (!job || !TERMINAL_STATUSES.includes(job.status)) continue
    adopted.delete(id)
    announce(job, ctx)
  }
  showTitle()
}
