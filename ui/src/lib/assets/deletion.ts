// What deleting assets costs, in destroyConfirm's wording. A shared asset
// goes for every workspace under the root, which is a different thing from
// deleting one of this workspace's own - so the confirm says so, and a
// bulk delete counts them rather than leaving the selection to be read.
import type { ConfirmSpec } from '../action.svelte'
import { destroyConfirm } from '../destructive'
import type { AssetFile } from '../types'

/** One asset, by its reference */
export function assetConfirm(asset: AssetFile): ConfirmSpec {
  const consequence =
    asset.origin === 'common'
      ? 'It is in the shared asset library, so every workspace under this root ' +
        'loses it, and any workflow still carrying that reference stops ' +
        'loading.'
      : 'Any workflow still carrying that reference stops loading.'
  return destroyConfirm({ name: asset.reference, noun: 'asset', consequence })
}

/** A ticked selection, `shared` of them from the shared asset library */
export function assetsConfirm(count: number, shared: number): ConfirmSpec {
  const them =
    count === 1 ? 'It is' : `${shared} of them ${shared === 1 ? 'is' : 'are'}`
  const goes = count === 1 || shared === 1 ? 'goes' : 'go'
  const sharedPart = shared
    ? `${them} in the shared asset library and ${goes} away for every workspace. `
    : ''
  return destroyConfirm({
    count,
    noun: 'asset',
    consequence: `${sharedPart}Any workflow still carrying one of those references stops loading.`,
  })
}
