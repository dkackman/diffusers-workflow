<script lang="ts">
  import { api, ApiError } from '../api'
  import { confirmDialog } from '../confirm.svelte'
  import { notify } from '../toast'
  import type { LedgerRequest } from '../types'
  import ItemMenu from './ItemMenu.svelte'
  import { requestMenu } from './menuEntries'
  import type { PlanStore } from './planStore.svelte'

  // The request's own actions, in its header: mark it done, abandon it,
  // reopen it - each a write of its state through the store, as any edit
  // is - and, once closed, delete it from the ledger
  let {
    request,
    store,
    workspace,
    onreload,
    ondeleted,
  }: {
    request: LedgerRequest
    store: PlanStore
    workspace: string
    /** Read the request again: a delete found it had moved on */
    onreload: () => void
    /** The request is gone from the ledger */
    ondeleted: () => void
  } = $props()

  async function setState(state: LedgerRequest['state']) {
    try {
      await store.push({ kind: 'state', state })
    } catch (error) {
      if (error instanceof ApiError && error.code === 'stale_revision') {
        notify.error(
          'The plan changed since this page read it. Look again, then try that.',
        )
      } else {
        notify.failure('save that', error)
      }
    }
  }

  const REFUSED: Record<string, string> = {
    request_has_active_jobs:
      'One of its jobs is queued or running. Cancel it or let it finish, then delete.',
    stale_revision:
      'The plan changed since this page read it. Look again, then delete.',
  }

  /** Its jobs and their files stay: only the ledger's record goes */
  async function remove() {
    const { title, id } = request
    const sure = await confirmDialog(
      `Delete "${title}"?\n\nIts plan, notes and history are removed from the ledger for good. Its jobs and their outputs are kept.`,
      { confirmLabel: 'Delete' },
    )
    if (!sure) return
    try {
      await api.deleteRequest(workspace, id, store.read?.revision ?? 0)
    } catch (error) {
      const code = error instanceof ApiError ? error.code : undefined
      if (code === 'stale_revision') onreload()
      const refused = code && REFUSED[code]
      if (refused) notify.error(`Not deleted: ${refused}`)
      else notify.failure('delete that', error)
      return
    }
    notify.success(`Deleted ${title}`)
    ondeleted()
  }

  const select = (id: string) =>
    id === 'delete' ? remove() : setState(id as LedgerRequest['state'])
</script>

<ItemMenu
  label="More actions for this request"
  entries={requestMenu(request.state)}
  onselect={select}
/>
