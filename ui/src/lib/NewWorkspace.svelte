<script lang="ts">
  // The sidebar's new-workspace form. It opens with focus in the field and
  // closes on Escape, Cancel, success (which lands on the new overview) or a
  // focus that leaves it while the field is empty: a typed name stays until
  // Cancel or Escape (and the parent holds it, so a collapse keeps it). A refused name is said beside the field, the draft
  // kept and focus back in it; Enter twice creates it once
  import { onMount } from 'svelte'
  import { createWorkspaceAndGo } from './workspaceActions'
  import { action } from './action.svelte'

  let {
    onclose,
    name = $bindable(''),
  }: { onclose?: () => void; name?: string } = $props()

  let error = $state('')
  let form = $state<HTMLFormElement | null>(null)
  let input = $state<HTMLInputElement | null>(null)

  onMount(() => input?.focus())

  const creating = action(
    async (wanted: string) => {
      const failure = await createWorkspaceAndGo(wanted)
      if (failure) {
        error = failure
        input?.focus()
      } else onclose?.()
      return failure
    },
    { doing: (wanted) => `create workspace ${wanted}` },
  )
  const submit = () => void creating.run(name.trim())

  function onkeydown(e: KeyboardEvent) {
    if (e.key === 'Escape') onclose?.()
    else if (e.key === 'Enter' && e.target === input) {
      e.preventDefault()
      submit()
    }
  }
  function onfocusout(e: FocusEvent) {
    const to = e.relatedTarget as Node | null
    if (!form?.contains(to) && !name.trim()) onclose?.()
  }
</script>

<form
  class="newws"
  bind:this={form}
  aria-busy={creating.busy}
  onsubmit={(e) => {
    e.preventDefault()
    submit()
  }}
  {onfocusout}
>
  <input
    bind:this={input}
    bind:value={name}
    placeholder="name"
    aria-label="new workspace name"
    {onkeydown}
  />
  {#if error}<span class="muted err">{error}</span>{/if}
  <div class="buttons">
    <button type="submit" class="quiet" {onkeydown}>Create</button>
    <button type="button" class="bare" onclick={() => onclose?.()} {onkeydown}>
      Cancel
    </button>
  </div>
</form>

<style>
  .newws {
    display: flex;
    flex-direction: column;
    gap: var(--space-1);
    padding: 0.2rem var(--space-4);
  }
  .newws input {
    width: 100%;
    font-family: var(--font-mono);
    font-size: var(--t-sm);
  }
  .err {
    font-family: var(--font-sans);
    font-size: var(--t-xs);
  }
  .buttons {
    display: flex;
    gap: var(--space-1);
  }
  .buttons button {
    padding: 0.2rem 0.6rem;
    font-size: var(--t-xs);
  }
</style>
