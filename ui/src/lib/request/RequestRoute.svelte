<script lang="ts">
  import RequestPage from '../pages/RequestPage.svelte'
  import { workspace } from '../workspace.svelte'

  // A page per request: another id, or another workspace, is a new page with
  // a new store, and the one left writes what it still holds
  let { requestId }: { requestId: string } = $props()
</script>

{#key `${workspace.current}/${requestId}`}
  <RequestPage {requestId} />
{/key}
