<script lang="ts">
  import { api, ApiError } from '../api'
  import { action } from '../action.svelte'
  import { destroyConfirm, undoToast } from '../destructive'
  import { notify } from '../toast'
  import type { LedgerRequest } from '../types'
  import ItemMenu from './ItemMenu.svelte'
  import { requestMenu, type RequestAction } from './menuEntries'
  import { onThisRequest, type PlanStore } from './planStore.svelte'

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

  /** A state write through the store, which reads again and re-applies it
   * when the plan moved on; whether it was written. `from` is the state it
   * must find there (an Undo's). Refused for good, it says why here, or
   * through the store's notice when the state moved on, and never throws */
  async function pushState(
    state: LedgerRequest['state'],
    from?: LedgerRequest['state'],
  ): Promise<boolean> {
    try {
      const outcome = await store.push(
        from ? { kind: 'state', state, from } : { kind: 'state', state },
      )
      return outcome.status === 'saved'
    } catch (error) {
      if (error instanceof ApiError && error.code === 'stale_revision') {
        notify.error(
          'The plan changed since this page read it. Look again, then try that.',
        )
      } else {
        notify.failure('save that', error)
      }
      return false
    }
  }

  const UNDONE: Partial<Record<LedgerRequest['state'], (t: string) => string>> =
    {
      done: (title) => `Marked "${title}" done`,
      abandoned: (title) => `Abandoned "${title}"`,
    }

  /** Undo's write, unless the agent reopened the request since: then there
   * is nothing to take back, and it says so */
  async function undo(
    previous: LedgerRequest['state'],
    state: LedgerRequest['state'],
  ) {
    const { title } = request
    if (store.read?.state === 'open')
      return void notify.success(`"${title}" is already open`)
    await pushState(previous, state)
  }

  /** Mark done and Abandon act at once and offer Undo, which writes the
   * state it had back through the same store path - only over the state
   * this wrote, so a change the agent made since stays; Reopen just writes
   * and says so. The page's closed line calls this for its Reopen button:
   * one writer for both */
  async function write(state: LedgerRequest['state']) {
    const { title, state: previous } = request
    const said = UNDONE[state]
    if (!(await pushState(state))) return
    if (state === 'open') return void notify.success(`Reopened "${title}"`)
    if (!said) return
    undoToast(said(title), () => undo(previous, state), onThisRequest(store))
  }

  // One write at a time: a second press while one is in flight does nothing
  const writing = action(write, { doing: 'save that' })
  export const setState = (state: LedgerRequest['state']) => writing.run(state)
  /** A state write is in flight (the closed line's Reopen shows it) */
  export const busy = () => writing.busy

  const REFUSED: Record<string, string> = {
    request_has_active_jobs:
      'One of its jobs is queued or running. Cancel it or let it finish, then delete.',
    stale_revision:
      'The plan changed since this page read it. Look again, then delete.',
  }

  /** Its jobs and their files stay: only the ledger's record goes */
  const removing = action(
    async ({ title, id }: LedgerRequest) => {
      try {
        await api.deleteRequest(workspace, id, store.read?.revision ?? 0)
      } catch (error) {
        const code = error instanceof ApiError ? error.code : undefined
        if (code === 'stale_revision') onreload()
        const refused = code && REFUSED[code]
        // Any other failure is the action's to report
        if (!refused) throw error
        notify.error(`Not deleted: ${refused}`)
        return
      }
      notify.success(`Deleted ${title}`)
      ondeleted()
    },
    {
      doing: () => 'delete that',
      confirm: ({ title }) =>
        destroyConfirm({
          name: `"${title}"`,
          noun: 'request',
          consequence:
            'Its plan, notes and history are deleted from the ledger. Its jobs and their outputs are kept.',
          place: 'ledger',
        }),
    },
  )
  const remove = () => removing.run(request)

  const select = (id: RequestAction) =>
    id === 'delete' ? remove() : setState(id)
</script>

<ItemMenu
  label="More actions for this request"
  entries={requestMenu(request.state)}
  onselect={select}
/>
