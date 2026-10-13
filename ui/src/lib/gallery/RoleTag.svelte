<script lang="ts">
  // A file's role in its run, as a small tag - the gallery's main axis.
  // Nothing for a file whose run does not say
  let {
    role,
  }: { role: 'final' | 'intermediate' | 'input' | null | undefined } = $props()
  const text = { final: 'Final', intermediate: 'Intermediate', input: 'Input' }
</script>

{#if role}
  <span class="role-tag" data-role={role}>{text[role]}</span>
{/if}

<style>
  .role-tag {
    display: inline-block;
    padding: 0.05rem 0.4rem;
    border-radius: 999px;
    font-family: var(--font-sans);
    font-size: var(--t-xs);
    font-weight: 600;
    line-height: 1.4;
    white-space: nowrap;
  }
  .role-tag[data-role='final'] {
    background: var(--accent);
    color: var(--accent-ink);
  }
  .role-tag[data-role='intermediate'] {
    background: var(--panel-2);
    color: var(--muted);
  }
  .role-tag[data-role='input'] {
    background: transparent;
    color: var(--muted);
    border: 1px solid var(--line);
  }
</style>
