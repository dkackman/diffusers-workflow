<script lang="ts">
  import type { Snippet } from 'svelte'
  import { Check, TriangleAlert } from '@lucide/svelte'
  import ItemProof from './ItemProof.svelte'
  import { takesOf, type Take } from './itemState'
  import { formatSpent } from '../format'
  import { requestQuery } from '../requestFilter'
  import { wsHref } from '../routes'
  import type { LedgerItem } from '../types'

  // An item's takes - every job linked to it - oldest to newest, as
  // pictures: which one is the result, and whether the user chose it or it
  // is only the latest. Read-only; `action` is where a control under each
  // take goes
  let {
    item,
    workspace,
    requestId,
    action,
  }: {
    item: LedgerItem
    workspace: string
    requestId: string
    /** Rendered under each succeeded take */
    action?: Snippet<[Take]>
  } = $props()

  const takes = $derived(takesOf(item))
  const CHIP = {
    queued: 'queued',
    running: 'running',
    failed: 'failed',
    cancelled: 'idle',
    succeeded: 'succeeded',
  }
  const MARK = {
    accepted: 'accepted',
    'accepted-gone': 'accepted, files gone',
    latest: 'shown as the result (latest)',
  }

  // The newest take is the one being looked for: the strip starts at its end
  let strip = $state<HTMLElement>()
  $effect(() => {
    void takes.length
    if (strip) strip.scrollLeft = strip.scrollWidth
  })
</script>

{#if item.accepted_missing}
  <p class="gone">
    <TriangleAlert size={14} aria-hidden="true" />
    The accepted take's files are gone. Showing the latest take instead.
  </p>
{/if}
<ul class="strip" bind:this={strip} aria-label="Takes of {item.title}">
  {#each takes as take (take.attempt.job)}
    {@const attempt = take.attempt}
    <li
      class="tile"
      class:result={take.mark === 'accepted' || take.mark === 'latest'}
    >
      <ItemProof
        output={attempt.output}
        state={attempt.status}
        {workspace}
        href={wsHref(workspace, 'jobs', attempt.job)}
        alt="take {take.n} of {item.title}"
        size="take"
        picked={take.mark === 'accepted'}
      />
      <span class="caption">
        <span class="n">take {take.n}</span>
        {#if attempt.status !== 'succeeded'}
          <span class="chip {CHIP[attempt.status]}">{attempt.status}</span>
        {/if}
        {#if attempt.minutes != null}
          <span class="num muted">{formatSpent(attempt.minutes)}</span>
        {/if}
      </span>
      {#if take.mark}
        <span class="mark" class:muted={take.mark === 'accepted-gone'}>
          {#if take.mark === 'accepted'}<Check
              size={12}
              aria-hidden="true"
            />{/if}
          {MARK[take.mark]}
        </span>
      {/if}
      {#if attempt.note}
        <span class="note muted" title={attempt.note}>{attempt.note}</span>
      {/if}
      {#if action && attempt.status === 'succeeded'}{@render action(take)}{/if}
    </li>
  {/each}
</ul>
<p class="links">
  <a href={wsHref(workspace, 'gallery') + requestQuery(requestId, item.key)}
    >open these in the gallery</a
  >
</p>

<style>
  .strip {
    list-style: none;
    margin: 0;
    padding: 0 0 var(--space-1);
    display: flex;
    gap: var(--space-3);
    overflow-x: auto;
  }
  .tile {
    flex: none;
    width: 120px;
    display: flex;
    flex-direction: column;
    gap: 0.15rem;
    font-size: var(--t-xs);
  }
  .caption {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.15rem var(--space-2);
  }
  .n {
    font-weight: 600;
  }
  .mark {
    display: inline-flex;
    align-items: center;
    gap: 0.2rem;
    font-weight: 600;
  }
  .note {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .gone {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    margin: 0 0 var(--space-2);
    font-size: var(--t-sm);
  }
  .links {
    margin: var(--space-1) 0 0;
    font-size: var(--t-sm);
  }
</style>
