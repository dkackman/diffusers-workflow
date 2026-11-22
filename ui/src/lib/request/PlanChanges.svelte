<script lang="ts">
  import ClampedText from '../ui/ClampedText.svelte'
  import { formatExact, formatMtime } from '../format'
  import { foldChanges, stateChange, who } from './planChanges'
  import type { LedgerHistoryEntry } from '../types'

  // The request's history, newest first: the agent's `why` on each write
  // is the only thing it ever says to the user, so its latest leads. A run
  // of the user's own writes is one line until asked for
  let {
    history,
    revision,
    heading = true,
  }: {
    history: LedgerHistoryEntry[]
    revision: number
    /** False inside a fold whose summary already names it */
    heading?: boolean
  } = $props()

  const FIRST = 5
  const changes = $derived(foldChanges(history))
  const oldest = $derived(Math.min(...history.map((entry) => entry.revision)))
  let all = $state(false)
  // The runs the reader opened, by the revision of their newest entry
  let opened = $state<number[]>([])
  const shown = $derived(all ? changes : changes.slice(0, FIRST))
  const seconds = (at: string) => Date.parse(at) / 1000
</script>

{#snippet line(entry: LedgerHistoryEntry)}
  {@const turned = stateChange(entry, entry.revision === oldest)}
  <div class="entry" id="plan-change-r{entry.revision}">
    <p class="head">
      <span class="num">r{entry.revision}</span>
      <span class="muted" title={formatExact(seconds(entry.at))}>
        {formatMtime(seconds(entry.at))}
      </span>
      <span class="by">{who(entry)}</span>
      {#if turned}<span class="state">{turned}</span>{/if}
    </p>
    <p class="why">{entry.why}</p>
    {#if entry.goal && entry.revision !== oldest}
      <div class="goal">
        <span class="muted">goal changed:</span>
        <ClampedText text={entry.goal} lines={2} />
      </div>
    {/if}
  </div>
{/snippet}

<section class="changes">
  {#if heading}<h2>Plan changes ({history.length})</h2>{/if}
  <p class="muted now">now at <span class="num">r{revision}</span></p>
  <ol>
    {#each shown as change (change.kind === 'entry' ? change.entry.revision : change.entries[0].revision)}
      <li>
        {#if change.kind === 'entry'}
          {@render line(change.entry)}
        {:else}
          {@const id = change.entries[0].revision}
          {@const last = change.entries[change.entries.length - 1]}
          <p class="head">
            <span class="num">r{last.revision}–r{id}</span>
            <span class="by">you, {change.entries.length} edits</span>
            <button
              type="button"
              class="bare fold"
              aria-expanded={opened.includes(id)}
              onclick={() =>
                (opened = opened.includes(id)
                  ? opened.filter((r) => r !== id)
                  : [...opened, id])}
            >
              {opened.includes(id) ? 'hide' : 'show'}
            </button>
          </p>
          {#if opened.includes(id)}
            <div class="run">
              {#each change.entries as entry (entry.revision)}
                {@render line(entry)}
              {/each}
            </div>
          {/if}
        {/if}
      </li>
    {/each}
  </ol>
  {#if changes.length > FIRST}
    <button
      type="button"
      class="bare fold"
      aria-expanded={all}
      onclick={() => (all = !all)}
    >
      {all ? 'show fewer' : `show all ${changes.length}`}
    </button>
  {/if}
</section>

<style>
  h2 {
    font-family: var(--font-sans);
    font-size: var(--t-base);
    margin: 0;
  }
  .now {
    font-size: var(--t-xs);
    margin: 0 0 var(--space-2);
  }
  ol {
    list-style: none;
    margin: 0;
    padding: 0;
    display: flex;
    flex-direction: column;
    gap: var(--space-3);
    font-size: var(--t-sm);
  }
  p {
    margin: 0;
  }
  .head {
    display: flex;
    flex-wrap: wrap;
    align-items: baseline;
    gap: 0 var(--space-2);
    font-size: var(--t-xs);
  }
  .by,
  .state {
    font-family: var(--font-mono);
  }
  .state {
    font-weight: 600;
  }
  .why {
    overflow-wrap: anywhere;
  }
  .goal {
    margin-top: var(--space-1);
  }
  .run {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
    margin-top: var(--space-2);
    padding-left: var(--space-3);
    border-left: 1px solid var(--line);
  }
  .fold {
    padding: 0;
    font-size: var(--t-xs);
    text-decoration: underline;
    text-underline-offset: 2px;
  }
</style>
