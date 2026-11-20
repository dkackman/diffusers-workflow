<script lang="ts">
  import type { LedgerRequest } from '../types'

  // A request's state is an engine value, so it is mono in a square tag;
  // an item's status is a pill (.chip). The two levels never read alike
  let { value }: { value: LedgerRequest['state'] } = $props()
</script>

<span class="tag {value}">{value}</span>

<style>
  .tag {
    font-family: var(--font-mono);
    font-size: var(--t-xs);
    font-weight: 400;
    color: var(--muted);
    border: 1px solid var(--line);
    border-radius: 4px;
    padding: 0 0.3rem;
    white-space: nowrap;
  }
  .open {
    color: var(--ink);
    border-color: var(--ink);
  }
  .done {
    color: var(--good);
  }
</style>
