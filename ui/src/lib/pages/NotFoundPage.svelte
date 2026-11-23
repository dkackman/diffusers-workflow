<script lang="ts">
  import { route } from '../router.svelte'
  import { wsHref } from '../routes'
  import { workspace } from '../workspace.svelte'

  const path = $derived(route.view.kind === 'missing' ? route.view.path : '')
</script>

<h1>Page not found</h1>
<p>There is no page at <code>{path}</code>.</p>
<p><a href={wsHref(workspace.current, 'overview')}>Go to the overview</a></p>
