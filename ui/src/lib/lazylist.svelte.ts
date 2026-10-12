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
 * fetch reads as empty: completions just stay absent. */
export function lazyListing<T>(fetch: () => Promise<T[]>) {
  const listing = $state<Listing<T>>({ items: undefined, workspace: undefined })
  let loading: Promise<void> | null = null

  /** Refetch unconditionally - after an upload, say. */
  function load(): Promise<void> {
    const wanted = workspace.current
    // Started inside the chain, so even a synchronous failure lands in the
    // catch below rather than in the caller
    loading = Promise.resolve()
      .then(fetch)
      .then((items) => {
        listing.items = items
      })
      .catch(() => {
        listing.items = []
      })
      .finally(() => {
        listing.workspace = wanted
        loading = null
      })
    return loading
  }

  /** Fetch unless the listing already holds this workspace's. */
  function ensure(): void {
    if (listing.workspace === workspace.current || loading) return
    void load()
  }

  return { listing, load, ensure }
}
