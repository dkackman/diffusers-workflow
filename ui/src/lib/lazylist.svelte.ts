import { workspace } from './workspace.svelte'

/** A workspace's listing as the editors' pickers share it. `items` stays
 * undefined until a listing lands, so not-yet-loaded is distinguishable
 * from empty; `workspace` says whose it is. */
export type Listing<T> = {
  items: T[] | undefined
  workspace: string | undefined
}

/** A listing loaded on first use rather than with every page - only a
 * field that starts a reference needs it - and again when the workspace
 * changes. `ensure` is safe to call from a derivation: it starts at most
 * one request, and the listing updating re-runs whatever read it. A failed
 * fetch reads as empty - completions just stay absent - and is tried again
 * by an ensure after RETRY_MS, so a server restart doesn't leave the
 * library empty for the session (not sooner: the failure itself re-runs
 * the derivation that asked, which would refetch in a loop). */
const RETRY_MS = 5000

export function lazyListing<T>(fetch: () => Promise<T[]>) {
  const listing = $state<Listing<T>>({ items: undefined, workspace: undefined })
  let loading: Promise<void> | null = null
  let failedAt: number | null = null

  /** Refetch unconditionally - after an upload, say. */
  function load(): Promise<void> {
    const wanted = workspace.current
    // Started inside the chain, so even a synchronous failure lands in the
    // catch below rather than in the caller
    loading = Promise.resolve()
      .then(fetch)
      .then((items) => {
        listing.items = items
        failedAt = null
      })
      .catch(() => {
        listing.items = []
        failedAt = Date.now()
      })
      .finally(() => {
        listing.workspace = wanted
        loading = null
      })
    return loading
  }

  /** Fetch unless the listing already holds this workspace's. */
  function ensure(): void {
    if (loading) return
    const retry = failedAt !== null && Date.now() - failedAt >= RETRY_MS
    if (listing.workspace === workspace.current && !retry) return
    void load()
  }

  return { listing, load, ensure }
}
