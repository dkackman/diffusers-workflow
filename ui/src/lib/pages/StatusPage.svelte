<script lang="ts">
  import ServerPage from './ServerPage.svelte'
  import JobsPage from './JobsPage.svelte'
</script>

<!-- What this server is, then what it is doing: the queue across every
     workspace lives here because the queue is the server's, not any one
     workspace's -->
<ServerPage />
<div class="alljobs">
  <JobsPage scope="all" />
</div>

<style>
  .alljobs {
    margin-top: var(--space-5);
    padding-top: var(--space-5);
    border-top: 1px solid var(--line);
  }
</style>
