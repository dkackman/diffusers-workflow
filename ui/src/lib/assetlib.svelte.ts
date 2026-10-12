import { api } from './api'
import { lazyListing } from './lazylist.svelte'
import type { AssetFile } from './types'

/** The asset library as the editors' pickers share it: every asset the
 * workspace's search path resolves, so an `asset:` completion offers what
 * a run would find. */
const assets = lazyListing<AssetFile>(() =>
  api.listAssets().then((result) => result.assets),
)
export const assetLibrary = assets.listing
export const ensureAssetLibrary = assets.ensure
export const loadAssetLibrary = assets.load

/** The asset an `asset:` name resolves to in the loaded listing. */
export function assetNamed(name: string): AssetFile | undefined {
  return assetLibrary.items?.find((asset) => asset.name === name)
}
