<script lang="ts">
  import { Plus } from '@lucide/svelte'
  import { variableNameProblem } from './variableTypes'

  let {
    taken,
    onadd,
  }: {
    taken: string[]
    /** Called with a name the engine accepts and no variable has yet */
    onadd: (name: string) => void
  } = $props()

  let name = $state('')
  let problem = $state<string | null>(null)

  function add() {
    if (!name) return
    problem = variableNameProblem(name, taken)
    if (problem) return
    onadd(name)
    name = ''
  }
</script>

<input
  placeholder="new variable name…"
  bind:value={name}
  oninput={() => (problem = null)}
  onkeydown={(e) => e.key === 'Enter' && add()}
  aria-invalid={problem ? true : undefined}
/>
<button
  class="quiet icon addvar"
  onclick={add}
  disabled={!name}
  title="add this variable to the workflow"
  aria-label="add this variable to the workflow"
>
  <Plus size={14} />
</button>
<span></span>
{#if problem}<p class="error problem" role="alert">{problem}</p>{/if}

<style>
  .problem {
    grid-column: 2 / -1;
    margin: 0;
  }
  .addvar {
    justify-self: start;
  }
</style>
