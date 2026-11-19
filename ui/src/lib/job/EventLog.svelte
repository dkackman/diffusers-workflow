<script lang="ts">
  // A job's event log, read from GET /api/jobs/{id}/event-log rather than
  // the live stream. Its state is its own: the page's `events` - which the
  // Progress list, the flow view and its chips read - never sees these rows.
  // Collapsed by default, and nothing is fetched until it is first opened.
  import { ChevronDown, ChevronRight, RefreshCw } from '@lucide/svelte'
  import { ApiError } from '../api'
  import { formatDuration } from '../format'
  import type { JobEvent } from '../types'
  import {
    EVENT_LOG_CAP,
    eventKind,
    eventSummary,
    fetchEventLog,
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
  let error = $state('')
  let kind = $state('')

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
    error = ''
    kind = ''
  })

  async function load(after: number) {
    const id = jobId
    loading = true
    error = ''
    try {
      const page = await fetchEventLog(id, after, EVENT_LOG_CAP)
      if (id !== jobId) return
      rows = after < 0 ? page.events : [...rows, ...page.events]
      lastSeq = page.lastSeq
      note = page.note ?? note
      capped = page.capped
      loaded = true
    } catch (e) {
      if (id !== jobId) return
      if (e instanceof ApiError && e.status === 404) missing = true
      else error = e instanceof Error ? e.message : String(e)
      loaded = true
    } finally {
      if (id === jobId) loading = false
    }
  }

  function toggle() {
    open = !open
    if (open && !loaded && !loading) load(-1)
  }

  const kinds = $derived([...new Set(rows.map(eventKind))].sort())
  const shown = $derived(
    kind ? rows.filter((event) => eventKind(event) === kind) : rows,
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
    {#if open && live && loaded && !missing}
      <button
        class="quiet icon"
        onclick={() => load(lastSeq)}
        disabled={loading}
        aria-label="Refresh"
        title="fetch the events since the last one shown"
      >
        <RefreshCw size={14} />
      </button>
    {/if}
  </div>

  {#if open}
    <div id="event-log-body">
      {#if loading && !loaded}
        <p class="muted">Loading…</p>
      {:else if missing}
        <p class="muted">no event log for this job</p>
      {:else}
        {#if error}<p class="error">{error}</p>{/if}
        {#if note}<p class="muted note">{note}</p>{/if}
        {#if loaded && !rows.length && !error}
          <p class="muted">no events</p>
        {:else if shown.length}
          <table>
            <thead>
              <tr><th>seq</th><th>time</th><th>kind</th><th>event</th></tr>
            </thead>
            <tbody>
              {#each shown as event (event.seq)}
                <tr>
                  <td class="num">{event.seq}</td>
                  <td class="num"
                    >{typeof event.at === 'number'
                      ? '+' + formatDuration(event.at)
                      : ''}</td
                  >
                  <td class="kind">{eventKind(event)}</td>
                  <td class="body">{eventSummary(event)}</td>
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
  table {
    width: 100%;
    border-collapse: collapse;
    font-family: var(--font-mono);
    font-size: var(--t-xs);
  }
  th {
    text-align: left;
    position: sticky;
    top: 0;
    background: var(--panel);
  }
  th,
  td {
    padding: 0.15rem 0.6rem 0.15rem 0;
    vertical-align: top;
  }
  .num {
    text-align: right;
    white-space: nowrap;
  }
  .kind {
    white-space: nowrap;
  }
  .body {
    word-break: break-word;
  }
  .error {
    color: var(--bad);
  }
</style>
