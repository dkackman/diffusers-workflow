<script lang="ts">
  import { tick } from 'svelte'
  import { ApiError } from '../api'
  import { action } from '../action.svelte'
  import { notify } from '../toast'
  import type { LedgerRequest } from '../types'
  import type { PlanStore } from './planStore.svelte'

  // The agent's proposal to close the request (ledger.py `close_proposal`):
  // only the user closes one, so a pending proposal waits here for Close,
  // which writes the proposed state as the menu's Mark done does (and
  // settles the proposal as accepted), or Keep open, which declines it
  // with an optional one-line note the agent reads through get_request.
  // A declined proposal stays as a quiet line until the next one.
  let {
    request,
    store,
    onclose,
    closing = false,
  }: {
    request: LedgerRequest
    store: PlanStore
    /** Write the proposed state: the menu's state write, its 409 message
     * and its Undo */
    onclose: (state: LedgerRequest['state']) => void
    /** That write is in flight */
    closing?: boolean
  } = $props()

  /** The server takes one line of at most 200 characters */
  const NOTE_MAX = 200

  const proposal = $derived(
    request.state === 'open' ? (request.close_proposal ?? null) : null,
  )
  const pending = $derived(proposal?.outcome === 'pending' ? proposal : null)
  const declined = $derived(proposal?.outcome === 'declined' ? proposal : null)
  const what = $derived(
    pending?.state === 'abandoned' ? 'abandon this' : 'mark this done',
  )

  let asking = $state(false)
  let note = $state('')
  let field = $state<HTMLInputElement>()

  async function ask() {
    asking = true
    await tick()
    field?.focus()
  }
  function cancel() {
    asking = false
    note = ''
  }

  /** The decline, through the store as every edit is: over the proposal
   * this page showed, so one the agent replaced since is never answered */
  const declining = action(
    async (revision: number) => {
      const text = note.trim()
      try {
        const outcome = await store.push({
          kind: 'decline',
          revision,
          ...(text ? { note: text } : {}),
        })
        if (outcome.status === 'saved') cancel()
      } catch (error) {
        if (error instanceof ApiError && error.code === 'stale_revision') {
          notify.error(
            'The plan changed since this page read it. Look again, then try that.',
          )
          return
        }
        throw error
      }
    },
    { doing: 'keep it open' },
  )
  const decline = () => {
    if (pending) void declining.run(pending.revision)
  }

  function onkeydown(event: KeyboardEvent) {
    if (event.key === 'Escape') {
      event.preventDefault()
      cancel()
    }
  }
</script>

{#if pending}
  <div
    class="panel warn-edge proposal"
    role="region"
    aria-label="The agent's proposal to close this request"
    data-testid="close-proposal"
  >
    <p class="said">
      The agent proposes to {what}: <span class="reason">{pending.reason}</span>
    </p>
    {#if asking}
      <form
        class="answer"
        onsubmit={(event) => {
          event.preventDefault()
          decline()
        }}
      >
        <input
          bind:this={field}
          bind:value={note}
          {onkeydown}
          maxlength={NOTE_MAX}
          placeholder="Why it stays open (optional)"
          aria-label="Why it stays open, for the agent (optional)"
        />
        <button type="submit" disabled={declining.busy}>Keep open</button>
        <button
          type="button"
          class="quiet"
          disabled={declining.busy}
          onclick={cancel}
        >
          Cancel
        </button>
      </form>
    {:else}
      <div class="answer">
        <button
          type="button"
          disabled={closing}
          onclick={() => onclose(pending.state)}
        >
          Close
        </button>
        <button type="button" onclick={ask}>Keep open</button>
      </div>
    {/if}
  </div>
{:else if declined}
  <p class="muted kept" data-testid="close-declined">
    {declined.note
      ? `You kept this open: ${declined.note}`
      : 'You kept this open.'}
  </p>
{/if}

<style>
  .proposal {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-2) var(--space-3);
  }
  .said {
    margin: 0;
    min-width: 0;
    overflow-wrap: anywhere;
  }
  .reason {
    font-weight: 600;
  }
  .answer {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: var(--space-2);
  }
  .answer input {
    min-width: 16rem;
  }
  .kept {
    margin: 0;
    font-size: var(--t-sm);
    overflow-wrap: anywhere;
  }
</style>
