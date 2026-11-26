<script lang="ts">
  // A list row whose request the agent proposes to close: it waits on the
  // user (the request page's Close or Keep open), so it takes attention's
  // colour, in the state tag's square
</script>

<span
  class="tag"
  data-testid="close-proposed-tag"
  title="The agent proposes to close this request: open it to close or keep it open"
  >close proposed</span
>

<style>
  .tag {
    font-family: var(--font-mono);
    font-size: var(--t-xs);
    font-weight: 400;
    color: var(--warn);
    border: 1px solid var(--warn);
    border-radius: 4px;
    padding: 0 0.3rem;
    white-space: nowrap;
  }
</style>
