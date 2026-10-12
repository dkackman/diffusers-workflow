import { afterEach, expect, it, vi } from 'vitest'
import { lazyListing } from './lazylist.svelte'

afterEach(() => {
  vi.useRealTimers()
})

it('fetches once per workspace', async () => {
  const fetch = vi.fn(() => Promise.resolve(['a']))
  const lib = lazyListing(fetch)
  lib.ensure()
  lib.ensure()
  await vi.waitFor(() => expect(lib.listing.items).toEqual(['a']))
  lib.ensure()
  expect(fetch).toHaveBeenCalledTimes(1)
})

it('tries a failed fetch again, but not straight away', async () => {
  vi.useFakeTimers()
  const fetch = vi
    .fn<() => Promise<string[]>>()
    .mockRejectedValueOnce(new Error('server restarting'))
    .mockResolvedValue(['a'])
  const lib = lazyListing(fetch)
  lib.ensure()
  await vi.waitFor(() => expect(lib.listing.items).toEqual([]))
  // The failure re-runs whatever asked; that must not refetch in a loop
  lib.ensure()
  expect(fetch).toHaveBeenCalledTimes(1)
  vi.advanceTimersByTime(5000)
  lib.ensure()
  await vi.waitFor(() => expect(lib.listing.items).toEqual(['a']))
  expect(fetch).toHaveBeenCalledTimes(2)
})
