import { api } from './api'
import type { AssetFile } from './types'
import { workspace } from './workspace.svelte'

/** The asset library as the editors' pickers share it: every asset the
 * workspace's search path resolves, so an `asset:` completion offers what
 * a run would find. Loaded on first use rather than with every page - only
 * a field that starts an `asset:` reference needs it - and again when the
 * workspace changes. `assets` stays undefined until a listing lands. */
export const assetLibrary = $state<{
  assets: AssetFile[] | undefined
  workspace: string | undefined
}>({ assets: undefined, workspace: undefined })

let loading: Promise<void> | null = null

/** Fetch the listing unless the store already holds this workspace's.
 * Safe to call from a derivation: it starts at most one request, and the
 * store updating re-runs whatever read it. */
export function ensureAssetLibrary(): void {
  if (assetLibrary.workspace === workspace.current || loading) return
  void loadAssetLibrary()
}

/** Refetch unconditionally - after an upload, say. */
export function loadAssetLibrary(): Promise<void> {
  const wanted = workspace.current
  // Started inside the chain, so even a synchronous failure lands in the
  // catch below rather than in the caller
  loading = Promise.resolve()
    .then(() => api.listAssets())
    .then((result) => {
      assetLibrary.assets = result.assets
    })
    .catch(() => {
      /* no asset library - completions just stay absent */
      assetLibrary.assets = []
    })
    .finally(() => {
      assetLibrary.workspace = wanted
      loading = null
    })
  return loading
}

/** The asset an `asset:` name resolves to in the loaded listing. */
export function assetNamed(name: string): AssetFile | undefined {
  return assetLibrary.assets?.find((asset) => asset.name === name)
}
