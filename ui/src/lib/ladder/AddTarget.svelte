<script lang="ts">
  import { Plus } from '@lucide/svelte'
  import Popover from '../ui/Popover.svelte'
  import Suggest from '../ui/Suggest.svelte'
  import { DEVICES } from './ladderDraft'

  // The `+ add` at the end of a tier's chain (#1762): a popover with the
  // workflow picker over the catalog and an optional "only on" device
  // choice. What is picked is the caller's to append
  let {
    tier,
    catalog,
    onadd,
  }: {
    tier: string
    /** The workflow names a target may name */
    catalog: readonly string[]
    onadd: (name: string, devices: string[]) => void
  } = $props()

  let open = $state(false)
  let trigger = $state<HTMLButtonElement | null>(null)
  let name = $state('')
  let devices = $state<string[]>([])

  // each opening starts blank: Bits keeps typed text in a field whose bound
  // value is cleared, and the content is remade on open
  $effect(() => {
    if (!open) {
      name = ''
      devices = []
    }
  })

  function add(event: SubmitEvent) {
    event.preventDefault()
    if (!name.trim()) return
    onadd(name.trim(), [...devices])
    open = false
  }
</script>

<button
  type="button"
  class="quiet add"
  bind:this={trigger}
  aria-label="Add a model to {tier}"
  aria-haspopup="dialog"
  aria-expanded={open}
  onclick={() => (open = !open)}
>
  <Plus size={14} aria-hidden="true" /> add
</button>
<Popover bind:open label="Add a model to {tier}" anchor={trigger}>
  <form class="adding" onsubmit={add}>
    <Suggest
      aria-label="workflow to add to {tier}"
      placeholder="a workflow…"
      suggestions={catalog}
      bind:value={name}
    />
    <fieldset>
      <legend>Only on these devices — leave empty for any</legend>
      {#each DEVICES as device (device)}
        <label>
          <input type="checkbox" value={device} bind:group={devices} />{device}
        </label>
      {/each}
    </fieldset>
    <button type="submit" disabled={!name.trim()}>Add</button>
  </form>
</Popover>

<style>
  .add {
    display: inline-flex;
    align-items: center;
    gap: 0.2em;
    font-size: var(--t-sm);
  }
  .adding {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: var(--space-2);
    padding: var(--space-2);
    font-size: var(--t-sm);
    min-width: 16rem;
  }
  fieldset {
    margin: 0;
    padding: 0;
    border: 0;
    display: flex;
    flex-wrap: wrap;
    gap: var(--space-1) var(--space-3);
  }
  legend {
    padding: 0;
    margin-bottom: var(--space-1);
    color: var(--muted);
  }
  label {
    display: inline-flex;
    align-items: center;
    gap: 0.2em;
  }
</style>
