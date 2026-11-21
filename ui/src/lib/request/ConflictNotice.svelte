<script lang="ts">
  import { TriangleAlert, X } from '@lucide/svelte'
  import type { LedgerHistoryEntry } from '../types'
  import {
    ANSWERS,
    appliedText,
    changedText,
    goneText,
    type Conflict,
  } from './conflict'

  // What the page says when the plan moved under an edit. Three cases:
  // the edit went on top of the other write (`applied`, a status line);
  // the item changed, so the edit was held back and the user is asked
  // (`changed`); the edit no longer applies (`gone`). The last two are
  // alerts: something the user did was not done
  let {
    applied,
    conflict,
    onagain,
    ondismiss,
    onsee,
  }: {
    /** The write the user's change went on top of; null when the history
     * no longer holds it. Leave out for a conflict card */
    applied?: LedgerHistoryEntry | null
    conflict?: Conflict
    /** Make the held-back edit anyway */
    onagain?: () => void
    ondismiss: () => void
    /** Show that write among the plan's changes */
    onsee?: (revision: number) => void
  } = $props()

  const answers = $derived(conflict ? ANSWERS[conflict.intent.kind] : undefined)
</script>

{#if conflict?.kind === 'changed'}
  <div class="panel warn-edge card" role="alert">
    <p>
      <TriangleAlert size={14} aria-hidden="true" />
      <span>{changedText(conflict)}</span>
    </p>
    <div class="answers">
      <button type="button" class="quiet danger" onclick={onagain}
        >{answers?.[0] ?? 'Do it anyway'}</button
      >
      <button type="button" class="quiet" onclick={ondismiss}
        >{answers?.[1] ?? 'Leave it'}</button
      >
    </div>
  </div>
{:else if conflict}
  <div class="panel warn-edge card" role="alert">
    <p>
      <TriangleAlert size={14} aria-hidden="true" />
      <span>{goneText(conflict)}</span>
    </p>
    <div class="answers">
      <button type="button" class="quiet" onclick={ondismiss}>Dismiss</button>
    </div>
  </div>
{:else}
  <div class="panel line" role="status">
    <span>{appliedText(applied ?? null)}</span>
    <span class="answers">
      {#if applied && onsee}
        <button
          type="button"
          class="bare see"
          onclick={() => onsee(applied.revision)}>See changes</button
        >
      {/if}
      <button
        type="button"
        class="bare icon"
        aria-label="Dismiss"
        title="Dismiss"
        onclick={ondismiss}><X size={14} aria-hidden="true" /></button
      >
    </span>
  </div>
{/if}

<style>
  .card,
  .line {
    padding: 0.55rem 0.8rem;
    font-size: var(--t-sm);
  }
  .card {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
  }
  .line {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-2) var(--space-4);
  }
  p {
    margin: 0;
    display: flex;
    gap: var(--space-2);
  }
  p :global(svg) {
    flex: none;
    margin-top: 0.25em;
  }
  .answers {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: var(--space-2);
  }
  .see {
    text-decoration: underline;
    text-underline-offset: 2px;
  }
</style>
