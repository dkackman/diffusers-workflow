<script lang="ts">
  import type { Rung } from '../preferences'
  import { serverHref } from '../routes'
  import { downloadNote, reasonText, refusalLead } from './preferenceCopy'

  // One tier of a preference's read view: what this machine runs for it, the
  // models passed over and why, and the models after the pick, which are not
  // checked here (#1762)
  let {
    tier,
    isDefault,
    rung,
    names,
  }: {
    tier: string
    isDefault: boolean
    /** This host's answer for the tier; undefined when the read had none */
    rung: Rung | undefined
    /** The tier's targets by name, in ladder order */
    names: readonly string[]
  } = $props()

  const download = $derived(rung ? downloadNote(rung.notes) : null)
  const fallback = $derived.by(() => {
    if (!rung?.target) return []
    const at = names.indexOf(rung.target)
    return at === -1 ? [] : names.slice(at + 1)
  })
</script>

<div class="tier" data-tier={tier}>
  <h3>
    {tier}{#if isDefault}<span class="default">default</span>{/if}
  </h3>
  {#if rung?.error}
    <p class="answer">
      <span class="dot bad" aria-hidden="true"></span>
      {refusalLead(rung.error, rung.skipped)}
    </p>
  {:else if rung?.target}
    <p class="answer">
      <span
        class="dot"
        class:good={!download}
        class:warn={!!download}
        aria-hidden="true"
      ></span>
      <code>{rung.target}</code>
      {#if download}
        <a class="note" href={serverHref('models', 'downloaded')}>
          {download.text}
        </a>
      {:else}
        <span class="note">ready</span>
      {/if}
    </p>
  {/if}
  {#if rung?.skipped.length}
    <ul class="skipped">
      {#each rung.skipped as skip, i (i + skip.workflow)}
        <li>
          <s>{skip.workflow}</s>
          <span class="muted">{reasonText(skip.reason)}</span>
        </li>
      {/each}
    </ul>
  {/if}
  {#if fallback.length}
    <p class="muted fallback">
      then {fallback.join(', ')} (fallback, not checked here)
    </p>
  {/if}
</div>

<style>
  .tier {
    font-size: var(--t-sm);
  }
  h3 {
    margin: 0 0 var(--space-1);
    font-size: var(--t-sm);
    font-weight: 600;
  }
  .default {
    margin-left: var(--space-2);
    padding: 0 0.4em;
    border: 1px solid var(--select);
    border-radius: var(--radius-1);
    color: var(--select);
    font-weight: 400;
    font-size: var(--t-xs);
  }
  p {
    margin: 0;
  }
  .answer {
    display: flex;
    flex-wrap: wrap;
    align-items: baseline;
    gap: var(--space-1) var(--space-2);
  }
  .dot {
    width: 0.6em;
    height: 0.6em;
    border-radius: 50%;
    background: currentColor;
    display: inline-block;
  }
  .dot.good {
    color: var(--good);
  }
  .dot.warn {
    color: var(--warn);
  }
  .dot.bad {
    color: var(--bad);
  }
  .note {
    color: var(--muted);
  }
  .skipped {
    margin: var(--space-1) 0 0;
    padding: 0;
    list-style: none;
  }
  .fallback {
    margin-top: var(--space-1);
  }
</style>
