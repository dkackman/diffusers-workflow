<script lang="ts">
  // The one panel for data that never arrived: what failed, a Retry, and -
  // for an editor whose file does not exist - a way to start it. LoadState
  // shows it for a page's data, where the raw message never reaches the
  // page; an editor, whose document is not a resource, shows it itself,
  // with the engine's detail (`detail`) on request: what is wrong with the
  // file being edited
  import { describeError } from '../apiErrors'
  import Details from './Details.svelte'

  let {
    label,
    error,
    onretry,
    oncreate,
    verb = 'open',
    what,
    detail = false,
  }: {
    label: string
    error: unknown
    onretry: () => void
    /** "Couldn't <verb> <label>" */
    verb?: 'open' | 'load'
    /** What a 404 is about, as in "<what> wasn't found" */
    what?: string
    /** Offer the engine's own message, folded */
    detail?: boolean
    /** Offered only when the file is missing, not the workspace */
    oncreate?: () => void
  } = $props()

  const described = $derived(describeError(error, what))
</script>

<div class="panel loadfailed" role="alert">
  <p>Couldn't {verb} {label}: {described.title}</p>
  {#if described.remedy}<p>{described.remedy}</p>{/if}
  {#if detail}
    <Details summary="Engine detail" detail={described.detail ?? ''} />
  {/if}
  <div class="actions">
    <button onclick={onretry}>Retry</button>
    {#if oncreate && described.kind === 'not_found'}
      <button class="quiet" onclick={oncreate}>
        Start a new one named {label}
      </button>
    {/if}
  </div>
</div>

<style>
  .loadfailed {
    margin: 1rem 0;
  }
  .loadfailed p {
    margin: 0 0 0.6rem;
    overflow-wrap: anywhere;
  }
  .actions {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
  }
</style>
