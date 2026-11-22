<script lang="ts">
  import { outputUrl } from '../apiUrls'
  import { wsHref } from '../routes'
  import type { LedgerNoteRef, LedgerRequest } from '../types'
  import { refLabel } from './noteRefs'

  // One thing a note for the agent is about, as a chip: a take's file
  // (linked to the file), a take (linked to its job), or an item (its
  // row). A reference that no longer holds is kept as written, greyed
  let {
    ref,
    request,
    workspace,
    ongo,
  }: {
    ref: LedgerNoteRef
    request: LedgerRequest
    workspace: string
    /** Go to an item's row, where the page has it; else the item's chip
     * links to the request page */
    ongo?: (key: string) => void
  } = $props()

  const label = $derived(refLabel(ref, request))
  const href = $derived(
    ref.output && ref.job
      ? outputUrl(ref.output, ref.job, workspace)
      : ref.job
        ? wsHref(workspace, 'jobs', ref.job)
        : wsHref(workspace, 'requests', request.id),
  )
</script>

{#if ref.gone}
  <span class="ref gone" title="no longer exists">{label}</span>
{:else if !ref.job && ongo}
  <button type="button" class="bare ref" onclick={() => ongo(ref.item)}>
    {label}
  </button>
{:else if ref.output}
  <a class="plain ref" {href} target="_blank" rel="noopener">{label}</a>
{:else}
  <a class="plain ref" {href}>{label}</a>
{/if}

<style>
  .ref {
    display: inline-block;
    max-width: 24ch;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    vertical-align: bottom;
    padding: 0.05rem 0.45rem;
    border: 1px solid var(--line);
    border-radius: 999px;
    color: var(--ink);
    font-size: var(--t-xs);
    text-decoration: none;
  }
  .ref:hover {
    border-color: var(--ink);
    background: transparent;
  }
  .gone {
    color: var(--muted);
    text-decoration: line-through;
  }
  .gone:hover {
    border-color: var(--line);
  }
</style>
