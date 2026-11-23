<script lang="ts">
  // The gallery's role and star chips: which roles the grid shows, and
  // whether only starred files
  import type { OutputRole } from '../types'

  let {
    roles = $bindable(),
    starredOnly = $bindable(),
  }: { roles: OutputRole[]; starredOnly: boolean } = $props()

  // Never both off: an empty grid is not a state worth offering
  function toggleRole(which: OutputRole) {
    const next = roles.includes(which)
      ? roles.filter((r) => r !== which)
      : [...roles, which]
    roles = next.length ? next : [which === 'final' ? 'intermediate' : 'final']
  }
</script>

<div class="rolechips" role="group" aria-label="filter by role">
  {#each ['final', 'intermediate'] as const as which (which)}
    <button
      class="rolechip"
      class:on={roles.includes(which)}
      aria-pressed={roles.includes(which)}
      onclick={() => toggleRole(which)}
      title={which === 'final'
        ? 'what the runs were for'
        : "the steps' working files, kept along the way"}
    >
      {which === 'final' ? 'Final' : 'Intermediate'}
    </button>
  {/each}
  <button
    class="rolechip"
    class:on={starredOnly}
    aria-pressed={starredOnly}
    onclick={() => (starredOnly = !starredOnly)}
    title="only the files you starred"
  >
    Starred
  </button>
</div>

<style>
  .rolechips {
    display: flex;
    gap: 0.3rem;
  }
  .rolechip {
    background: none;
    border: 1px solid var(--line);
    border-radius: 999px;
    color: var(--muted);
    font-size: var(--t-xs);
    font-weight: 500;
    padding: 0.05rem 0.6rem;
    cursor: pointer;
  }
  .rolechip:hover {
    color: var(--ink);
    border-color: var(--muted);
    filter: none;
  }
  .rolechip.on {
    background: var(--ink);
    border-color: var(--ink);
    color: var(--panel);
  }
</style>
