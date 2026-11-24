<script lang="ts">
  import ErrorLine from '../ui/ErrorLine.svelte'
  import Details from '../ui/Details.svelte'
  // A job's event log, read from GET /api/jobs/{id}/event-log rather than
  // the live stream. Its state is its own: the page's `events` - which the
  // Progress list, the flow view and its chips read - never sees these rows.
  // Collapsed by default, and nothing is fetched until it is first opened.
  import { ChevronDown, ChevronRight, RefreshCw } from '@lucide/svelte'
  import { tick, untrack } from 'svelte'
  import { ApiError } from '../api'
  import { formatDuration } from '../format'
  import type { JobEvent } from '../types'
  import {
    EVENT_LOG_CAP,
    eventKind,
    eventSummary,
    fetchEventLog,
    type EventLogFetch,
  } from './eventLog'

  let { jobId, live }: { jobId: string; live: boolean } = $props()

  let open = $state(false)
  let loaded = $state(false)
  let loading = $state(false)
  let rows = $state<JobEvent[]>([])
  let lastSeq = $state(-1)
  let note = $state<string | null>(null)
  let capped = $state(false)
  let missing = $state(false)
  let error = $state<unknown>(null)
  let kind = $state('')
  let body = $state<HTMLDivElement | null>(null)
  // Follow new events: on for a live job. Only a scroll up by the user turns
  // it off; `lastTop` is where the log last stood, so the scrolls this
  // component makes (always to the bottom) never read as the user's.
  let follow = $state(untrack(() => live))
  let lastTop = 0

  // Another job is another log: drop this one's and close
  $effect(() => {
    void jobId
    open = false
    loaded = false
    loading = false
    rows = []
    lastSeq = -1
    note = null
    capped = false
    missing = false
    error = null
    kind = ''
    follow = untrack(() => live)
    lastTop = 0
  })

  function take(page: EventLogFetch, after: number) {
    rows = after < 0 ? page.events : [...rows, ...page.events]
    lastSeq = page.lastSeq
    note = page.note ?? note
    capped = page.capped
    loaded = true
  }

  // A live log is read from its tail: the rows a load or Refresh just
  // brought in are the newest, so bring them into view rather than
  // leaving them below the box's fold (#1499)
  async function showTail(id: string) {
    await tick()
    if (id !== jobId || !body) return
    body.scrollTop = body.scrollHeight
    lastTop = body.scrollTop
  }

  function scrolled() {
    if (!body) return
    if (body.scrollTop < lastTop - 1) follow = false
    lastTop = body.scrollTop
  }

  function toggleFollow() {
    follow = !follow
    if (follow) showTail(jobId)
  }

  function failed(e: unknown) {
    if (e instanceof ApiError && e.status === 404) missing = true
    else error = e
    loaded = true
  }

  async function load(after: number) {
    const id = jobId
    loading = true
    error = null
    try {
      const page = await fetchEventLog(id, after, EVENT_LOG_CAP)
      if (id !== jobId) return
      take(page, after)
      if (follow && page.events.length) await showTail(id)
    } catch (e) {
      if (id === jobId) failed(e)
    } finally {
      if (id === jobId) loading = false
    }
  }

  // The job ended while its log was loaded: Refresh goes with `live`, so
  // read the events that came after the last one shown, the failure among
  // them, rather than leave the log short of its end
  let wasLive = untrack(() => live)
  $effect(() => {
    const ended = wasLive && !live
    wasLive = live
    if (ended) untrack(() => loaded && !missing && !loading && load(lastSeq))
  })

  function toggle() {
    open = !open
    if (open && !loaded && !loading) load(-1)
  }

  const kinds = $derived([...new Set(rows.map(eventKind))].sort())
  const shown = $derived(
    kind ? rows.filter((event) => eventKind(event) === kind) : rows,
  )
  // Says how many rows the box holds, so rows past its fold are not
  // mistaken for rows that never came
  const count = $derived(
    rows.length
      ? (kind ? `${shown.length} of ` : '') +
          `${rows.length.toLocaleString('en-US')} events, seq ${rows[0].seq}–${rows[rows.length - 1].seq}`
      : '',
  )
</script>

<section class="panel eventlog" aria-label="Event log">
  <div class="head">
    <h2>
      <button
        class="bare toggle"
        onclick={toggle}
        aria-expanded={open}
        aria-controls="event-log-body"
      >
        {#if open}<ChevronDown size={14} />{:else}<ChevronRight
            size={14}
          />{/if}
        Event log
      </button>
    </h2>
    {#if open && loaded && !missing && rows.length}
      <label class="filter">
        <span class="muted">kind</span>
        <select bind:value={kind} aria-label="filter by kind">
          <option value="">all</option>
          {#each kinds as k (k)}<option value={k}>{k}</option>{/each}
        </select>
      </label>
    {/if}
    {#if open && loaded && !missing}
      <button
        class="togglechip"
        class:on={follow}
        aria-pressed={follow}
        onclick={toggleFollow}
      >
        Follow new events
      </button>
    {/if}
    {#if open && live && loaded && !missing}
      <button
        class="quiet icon"
        onclick={() => load(lastSeq)}
        disabled={loading}
        aria-label="Refresh event log"
        title="fetch the events since the last one shown"
      >
        <RefreshCw size={14} />
      </button>
    {/if}
  </div>

  {#if open}
    <div id="event-log-body" bind:this={body} onscroll={scrolled}>
      {#if loading && !loaded}
        <p class="muted">Loading…</p>
      {:else if missing}
        <p class="muted">no event log for this job</p>
      {:else}
        {#if error}<ErrorLine {error} doing="read the log" />{/if}
        {#if note}<p class="muted note">{note}</p>{/if}
        {#if count}<p class="muted count">{count}</p>{/if}
        {#if loaded && !rows.length && !error}
          <p class="muted">no events</p>
        {:else if shown.length}
          <table class="datatable dense">
            <thead>
              <tr><th>seq</th><th>time</th><th>kind</th><th>event</th></tr>
            </thead>
            <tbody>
              {#each shown as event (event.seq)}
                {@const said = eventSummary(event)}
                <tr>
                  <td class="num">{event.seq}</td>
                  <td class="num">
                    {typeof event.at === 'number'
                      ? '+' + formatDuration(event.at)
                      : ''}
                  </td>
                  <td class="kind">{eventKind(event)}</td>
                  <td class="body">
                    <span class="text" title={said.text}>{said.text}</span>
                    {#if said.data}
                      <Details summary="Event data" detail={said.data} />
                    {/if}
                  </td>
                </tr>
              {/each}
            </tbody>
          </table>
        {/if}
        {#if capped}
          <p class="muted">
            showing the first {rows.length.toLocaleString('en-US')}; more on the
            server
          </p>
        {/if}
      {/if}
    </div>
  {/if}
</section>

<style>
  .eventlog {
    margin-bottom: 1rem;
  }
  .head {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.4rem 1rem;
  }
  .toggle {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0;
    font: inherit;
    color: var(--ink);
  }
  .head h2 {
    margin: 0;
  }
  .filter {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    font-size: var(--t-xs);
  }
  .note {
    margin: var(--space-2) 0;
  }
  #event-log-body {
    margin-top: var(--space-2);
    max-height: 32rem;
    overflow: auto;
  }
  th {
    position: sticky;
    top: 0;
    background: var(--panel);
  }
  .num {
    text-align: right;
    white-space: nowrap;
  }
  .kind {
    white-space: nowrap;
  }
  /* One line a row, so a long payload (a memory reading) does not push
     the rows after it out of the box; the whole of it is in the title */
  .body {
    width: 100%;
    max-width: 0;
  }
  .text {
    display: block;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .count {
    margin: var(--space-2) 0;
  }
</style>
