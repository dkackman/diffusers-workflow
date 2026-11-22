<script lang="ts">
  import { Check, TriangleAlert } from '@lucide/svelte'
  import ItemProof from './ItemProof.svelte'
  import { takesOf, type Take } from './itemState'
  import { formatSpent } from '../format'
  import { wsHref } from '../routes'
  import type { LedgerItem } from '../types'

  // An item's takes - every job linked to it - oldest to newest, as
  // pictures: which one is the result, and whether the user chose it or it
  // is only the latest. With `onaccept`, each good take has the button
  // that makes it the result - or, pressed again on the chosen one, clears
  // the choice. With `onnote`, each take has "Note this take", which adds
  // it to the item's new note for the agent
  let {
    item,
    workspace,
    onaccept,
    onnote,
  }: {
    item: LedgerItem
    workspace: string
    /** Accept this take, or with null clear the one accepted. Left out
     * where the plan would not keep it: a dropped item */
    onaccept?: (take: Take, job: string | null) => void
    /** Add this take to the new note. Left out for a dropped item */
    onnote?: (take: Take) => void
  } = $props()

  const takes = $derived(takesOf(item))
  const CHIP = {
    queued: 'queued',
    running: 'running',
    failed: 'failed',
    cancelled: 'idle',
    succeeded: 'succeeded',
  }
  const chosen = (take: Take) =>
    take.mark === 'accepted' || take.mark === 'accepted-gone'
  const MARK = {
    accepted: 'accepted',
    'accepted-gone': 'accepted, files gone',
    latest: 'result (latest)',
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
    {#if onaccept}Accept another to settle it.{/if}
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
        <!-- On the caption line, so every take's buttons sit at one height -->
        {#if take.mark}
          <span class="mark" class:muted={take.mark === 'accepted-gone'}>
            {#if take.mark === 'accepted'}<Check
                size={12}
                aria-hidden="true"
              />{/if}
            {MARK[take.mark]}
          </span>
        {/if}
      </span>
      {#if attempt.note}
        <span class="note muted" title={attempt.note}>{attempt.note}</span>
      {/if}
      {#if onaccept && attempt.status === 'succeeded'}
        <button
          type="button"
          class="quiet accept"
          aria-pressed={chosen(take)}
          aria-label="Accept take {take.n} for {item.title}"
          title={chosen(take)
            ? 'Accepted as the result. Press to clear the choice'
            : 'Make this take the result'}
          onclick={() => onaccept(take, chosen(take) ? null : attempt.job)}
        >
          {chosen(take) ? 'Accepted' : 'Accept'}
        </button>
      {/if}
      {#if onnote}
        <button
          type="button"
          class="bare notetake"
          aria-label="Note take {take.n} of {item.title} for the agent"
          onclick={() => onnote(take)}
        >
          Note this take
        </button>
      {/if}
    </li>
  {/each}
</ul>

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
  /* The strip's tiles are one height: the buttons go to the bottom, so
     they line up whether or not a take carries the result mark */
  .accept {
    margin-top: auto;
    padding: 0.2rem 0.5rem;
    font-size: var(--t-xs);
  }
  .notetake {
    align-self: flex-start;
    padding: 0.1rem 0;
    font-size: var(--t-xs);
    text-decoration: underline;
    text-underline-offset: 2px;
  }
  .accept[aria-pressed='true'] {
    color: var(--ink);
    border-color: var(--ink);
    font-weight: 600;
  }
  @media (pointer: coarse) {
    .accept,
    .notetake {
      min-height: 44px;
    }
  }
  .gone {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    margin: 0 0 var(--space-2);
    font-size: var(--t-sm);
  }
</style>
