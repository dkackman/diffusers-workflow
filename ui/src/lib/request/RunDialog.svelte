<script lang="ts">
  import { TriangleAlert } from '@lucide/svelte'
  import StatusChip from '../ui/StatusChip.svelte'
  import { api } from '../api'
  import { formatMinutes } from '../format'
  import { describeEstimateTotal } from '../plan'
  import { wsHref } from '../routes'
  import { notify } from '../toast'
  import Details from '../ui/Details.svelte'
  import Modal from '../ui/Modal.svelte'
  import Skeleton from '../ui/Skeleton.svelte'
  import type { PlanState } from './itemState'
  import { listening } from './listening.svelte'
  import {
    afterwardsLine,
    heldLine,
    planNotes,
    planSaid,
    quoteLine,
    type RunPlan,
  } from './runPlan'
  import { RunQueue, type RunRow } from './runQueue.svelte'

  // The one way this page spends GPU time: every item priced before the
  // button that queues them can be pressed, and nothing shown as queued
  // until the server has queued it. Mounted for one opening; closing it
  // drops the pricing still out
  let {
    run,
    plan,
    workspace,
    requestId,
    onclose,
  }: {
    /** What would run and what would not (runPlan.ts) */
    run: RunPlan
    /** The plan it is a run of, for what becomes ready afterwards */
    plan: PlanState
    workspace: string
    requestId: string
    /** The dialog has closed; `queued` jobs were queued from it */
    onclose: (queued: number) => void
  } = $props()

  // svelte-ignore state_referenced_locally
  const queue = new RunQueue(run.candidates, { workspace, requestId, api })
  void queue.price()

  let open = $state(true)
  let cancel = $state<HTMLButtonElement>()
  let told = false
  function closed() {
    if (told) return
    told = true
    queue.close()
    onclose(queue.queued.length)
  }
  $effect(() => {
    if (!open) closed()
  })
  $effect(() => closed)

  // Every chosen row queued: there is nothing left to look at
  $effect(() => {
    if (!queue.finished || queue.busy) return
    const n = queue.queued.length
    notify.success(`Queued ${n} ${n === 1 ? 'job' : 'jobs'}`)
    open = false
  })

  const only = $derived(run.single ? run.candidates[0] : null)
  const heading = $derived(
    !only
      ? 'Run what can run now'
      : run.retake
        ? `Run another take of ${only.title}`
        : `Run ${only.title}`,
  )
  const rows = $derived(queue.rows.filter((row) => row.state !== 'refused'))
  // a failed item offered again sits apart, unticked, so it is never run
  // by accident
  const firstRows = $derived(rows.filter((row) => !row.candidate.optional))
  const againRows = $derived(rows.filter((row) => row.candidate.optional))
  const wont = $derived([
    ...queue.rows
      .filter((row) => row.state === 'refused')
      .map((row) => ({
        key: row.candidate.key,
        title: row.candidate.title,
        reason: row.error ?? '',
        detail: row.detail ?? '',
      })),
    ...run.skipped.map((skip) => ({ ...skip, detail: '' })),
  ])
  /** The rows the total and the "afterwards" line speak of */
  const counted = $derived(
    rows.filter((row) => row.ticked && row.state !== 'pricing'),
  )
  const total = $derived(
    describeEstimateTotal(counted.map((row) => row.plan?.estimate ?? null)),
  )
  const afterwards = $derived(
    afterwardsLine(
      plan,
      counted.map((row) => row.candidate.key),
    ),
  )
  const n = $derived(queue.chosen.length)

  // A held item says that running it is the go-ahead, read off the plan
  // as it is now: its review may end while the dialog is open
  const channel = $derived(listening(workspace))
  const held = (key: string) =>
    heldLine(
      plan.live.find((state) => state.item.key === key)?.review ?? null,
      channel.listening,
    )

  const quote = (row: RunRow) => quoteLine(row.plan?.estimate ?? null)
  const said = (row: RunRow) =>
    planSaid(row.candidate.planMinutes, row.plan?.estimate.minutes)
  const now = (row: RunRow) => {
    const minutes = row.plan?.estimate.minutes
    return minutes == null
      ? 'Now with no figure.'
      : `Now ${formatMinutes(minutes)}.`
  }
</script>

{#snippet runRow(row: RunRow)}
  {@const item = row.candidate}
  {@const hold = held(item.key)}
  <li class="runrow" data-key={item.key}>
    <div class="who">
      {#if !run.single}
        <input
          type="checkbox"
          bind:checked={row.ticked}
          disabled={row.state !== 'priced' || queue.started || queue.busy}
          aria-label="Run {item.title}"
        />
      {/if}
      <span class="name">{item.title}</span>
      <code class="muted">{item.key}</code>
      <code class="flow">{item.workflow}</code>
    </div>
    {#if hold}
      <p class="held" class:warn={hold.tone === 'warn'}>
        {#if hold.tone === 'warn'}<TriangleAlert
            size={13}
            aria-hidden="true"
          />{/if}
        <span>{hold.text}</span>
      </p>
    {/if}
    <div class="quote">
      {#if row.state === 'pricing'}
        <Skeleton lines={1} label="pricing {item.title}" />
      {:else}
        {@const line = quote(row)}
        <p class:warn={line.tone === 'warn'}>
          {#if line.tone === 'warn'}<TriangleAlert
              size={13}
              aria-hidden="true"
            />{/if}
          <span>{line.text}</span>
        </p>
        {#if line.detail}<Details
            summary="Why there's no estimate"
            detail={line.detail}
          />{/if}
        {#if row.verdict}
          <p class="verdict" class:warn={row.verdict.tone === 'warn'}>
            <TriangleAlert size={13} aria-hidden="true" />
            <span>
              {row.verdict.text}{#if row.verdict.remedy}
                {row.verdict.remedy}{/if}
            </span>
          </p>
        {/if}
        {#if row.plan}
          {#each planNotes(row.plan) as note (note.text)}
            <p class:warn={note.tone === 'warn'}>
              {#if note.tone === 'warn'}<TriangleAlert
                  size={13}
                  aria-hidden="true"
                />{/if}
              <span>{note.text}</span>
            </p>
          {/each}
        {/if}
        {#if said(row)}<p class="muted">{said(row)}</p>{/if}
      {/if}
    </div>
    {#if row.state === 'sending'}
      <p class="outcome muted" role="status">queuing…</p>
    {:else if row.state === 'queued' && row.job}
      <p class="outcome" role="status">
        <StatusChip status="queued" />
        <a href={wsHref(workspace, 'jobs', row.job)}>Open job</a>
      </p>
    {:else if row.state === 'repriced'}
      <p class="outcome" role="alert">
        <span>The price moved. {now(row)}</span>
        <button
          type="button"
          class="quiet"
          disabled={queue.busy}
          onclick={() => queue.accept(item.key)}
        >
          Run at this price
        </button>
      </p>
    {:else if row.state === 'failed'}
      <div class="outcome bad" role="alert">
        <p>{row.error}</p>
        <Details summary="Engine detail" detail={row.detail ?? ''} />
      </div>
    {/if}
  </li>
{/snippet}

<Modal
  bind:open
  label={heading}
  large
  onopenfocus={(event) => {
    event.preventDefault()
    cancel?.focus()
  }}
>
  <h2>{heading}</h2>
  <div class="body">
    {#if rows.length}
      {#if firstRows.length}
        <ul class="rows" aria-label="Will run">
          {#each firstRows as row (row.candidate.key)}{@render runRow(
              row,
            )}{/each}
        </ul>
      {/if}
      {#if againRows.length}
        <section class="again">
          <h3>Failed before. Tick one to run it again.</h3>
          <ul class="rows" aria-label="Failed before">
            {#each againRows as row (row.candidate.key)}{@render runRow(
                row,
              )}{/each}
          </ul>
        </section>
      {/if}
      {#if !run.single}
        <p class="total num" class:warn={total.tone === 'warn'}>{total.text}</p>
      {/if}
    {:else if queue.settled}
      <p class="muted">Nothing here can be run.</p>
    {/if}

    {#if wont.length}
      <section class="wont">
        <h3>Will not run</h3>
        <ul>
          {#each wont as skip (skip.key)}
            <li>
              <span class="name">{skip.title}</span>: {skip.reason}<Details
                summary="Engine detail"
                detail={skip.detail}
              />
            </li>
          {/each}
        </ul>
      </section>
    {/if}
    {#if run.retake}<p class="after">{run.retake}</p>{/if}
    {#if afterwards}<p class="after">Afterwards: {afterwards}</p>{/if}
  </div>
  <div class="ui-actions">
    <button
      type="button"
      class="bare"
      bind:this={cancel}
      onclick={() => (open = false)}
    >
      {queue.queued.length ? 'Close' : 'Cancel'}
    </button>
    <button
      type="button"
      disabled={!queue.settled || queue.busy || n === 0}
      onclick={() => queue.submit()}
    >
      {queue.busy ? 'Queuing…' : `Run ${n} ${n === 1 ? 'job' : 'jobs'}`}
    </button>
  </div>
</Modal>

<style>
  h2 {
    font-family: var(--font-sans);
    font-size: var(--t-md);
    margin: 0 0 var(--space-3);
    overflow-wrap: anywhere;
  }
  h3 {
    font-family: var(--font-sans);
    font-size: var(--t-sm);
    margin: 0 0 var(--space-1);
  }
  .body {
    overflow-y: auto;
    min-height: 0;
    display: flex;
    flex-direction: column;
    gap: var(--space-3);
    margin-bottom: var(--space-4);
    font-size: var(--t-sm);
  }
  ul {
    list-style: none;
    margin: 0;
    padding: 0;
  }
  p {
    margin: 0;
  }
  .runrow {
    display: flex;
    flex-direction: column;
    gap: var(--space-1);
    padding: var(--space-2) 0;
  }
  .runrow + .runrow {
    border-top: 1px solid var(--line);
  }
  .who {
    display: flex;
    flex-wrap: wrap;
    align-items: baseline;
    gap: 0 var(--space-2);
  }
  .who input {
    align-self: center;
    flex: none;
    width: auto;
    margin: 0;
  }
  .name {
    font-weight: 600;
    overflow-wrap: anywhere;
  }
  .who code {
    font-size: var(--t-xs);
  }
  .flow {
    margin-left: auto;
  }
  .quote {
    display: flex;
    flex-direction: column;
    gap: 0.1rem;
  }
  .quote p,
  .outcome {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: var(--space-1) var(--space-2);
  }
  .quote :global(svg),
  .held :global(svg) {
    flex: none;
  }
  .held {
    display: flex;
    align-items: center;
    gap: var(--space-1) var(--space-2);
  }
  .quote span {
    min-width: 0;
    flex: 1;
  }
  .outcome p {
    margin: 0;
  }
  .outcome button {
    padding: 0.2rem 0.6rem;
  }
  .total {
    font-weight: 600;
    text-align: right;
    border-top: 1px solid var(--line);
    padding-top: var(--space-2);
  }
  .wont li {
    color: var(--muted);
    padding: 0.1rem 0;
  }
  .wont .name {
    color: var(--ink);
    font-weight: 500;
  }
  .after {
    color: var(--muted);
  }
  @media (pointer: coarse) {
    .who input {
      width: 24px;
      height: 24px;
    }
  }
</style>
