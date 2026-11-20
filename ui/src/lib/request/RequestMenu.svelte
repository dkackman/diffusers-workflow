<script lang="ts">
  import { notify } from '../toast'
  import type { LedgerRequest } from '../types'
  import ItemMenu from './ItemMenu.svelte'
  import { requestMenu } from './menuEntries'
  import type { PlanStore } from './planStore.svelte'

  // The request's own actions, in its header: mark it done, abandon it,
  // reopen it - each a write of its state through the store, as any edit is
  let {
    request,
    store,
  }: {
    request: LedgerRequest
    store: PlanStore
  } = $props()

  async function setState(state: LedgerRequest['state']) {
    try {
      await store.push({ kind: 'state', state })
    } catch (error) {
      const said = error instanceof Error ? error.message : String(error)
      notify.error(`Not saved: ${said}`)
    }
  }
</script>

<ItemMenu
  label="More actions for this request"
  entries={requestMenu(request.state)}
  onselect={(id) => setState(id as LedgerRequest['state'])}
/>
