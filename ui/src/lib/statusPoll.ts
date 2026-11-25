// The app shell's status poll (memory, health, job-finished alerts): the
// one UI timer that keeps running while the tab is hidden. The tab title
// and the job-finished alert it drives exist for the hidden tab, so it may
// never stop there (Don's ruling on #1536, Q2; ui/CLAUDE.md). Its readings
// land in `connection` (connection.svelte.ts).
import { poll } from './poll'
import { api } from './api'
import { connection } from './connection.svelte'
import { setAttentionBackends } from './attentionBackends.svelte'

export const STATUS_POLL_MS = 5000

/** Ask for the server's status now and every STATUS_POLL_MS, hidden or
 * not, until the returned function stops it */
export function pollStatus(refresh: () => unknown): () => void {
  return poll(refresh, STATUS_POLL_MS, { whileHidden: true })
}

/** Read memory and health once. Settled apart: memory answers 503 while the
 * worker is unreachable, and that must not blank the "running" indicator */
export async function checkNow(): Promise<void> {
  const [memory, health] = await Promise.allSettled([
    api.memory(),
    api.health(),
  ])
  connection.memory = memory.status === 'fulfilled' ? memory.value : null
  connection.health = health.status === 'fulfilled' ? health.value : null
  setAttentionBackends(connection.health?.attention_backends)
}

/** The shell's status poll, until the returned function stops it */
export function startStatusPoll(): () => void {
  return pollStatus(checkNow)
}
