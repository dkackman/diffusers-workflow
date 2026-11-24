<!-- dw's own disk caches on the server page: the registry's table with a
  clear per cache and, where a cache lists its entries, a delete per entry.
  Read on open, after each clear, and when the models feed moves (a clear
  or an entry delete anywhere) - every read walks the cache's tree, so it
  is never polled. -->
<script lang="ts">
  import {
    ChevronDown,
    ChevronRight,
    HardDrive,
    Trash2,
    TriangleAlert,
  } from '@lucide/svelte'
  import { api } from './api'
  import { resource } from './resource.svelte'
  import LoadState from './ui/LoadState.svelte'
  import DiskMeter from './ui/DiskMeter.svelte'
  import ErrorLine from './ui/ErrorLine.svelte'
  import { action } from './action.svelte'
  import { cacheClearConfirm, cacheEntryConfirm } from './destructive'
  import { formatBytes, formatMtime } from './format'
  import type { DiskCacheEntry, DiskCacheInfo } from './types'

  let clearError = $state<unknown>(null)

  const read = resource(() => api.listCaches(), { watch: ['models'] })
  const caches = $derived(read.data?.caches ?? [])
  const refreshCaches = () => read.reload()

  // formatBytes floors at 1 KB, so an empty cache would read as 1 KB
  const cacheSize = (bytes: number) => (bytes ? formatBytes(bytes) : '0 bytes')

  // A refusal (a 409: a job running or queued) carries the server's reason,
  // which ErrorLine shows; the cache list is read again either way
  async function settle(run: () => Promise<unknown>) {
    clearError = null
    try {
      await run()
    } catch (e) {
      clearError = e
    } finally {
      await refreshCaches()
    }
  }

  const clear = action(
    (cache: DiskCacheInfo) => settle(() => api.clearCache(cache.name)),
    {
      doing: (cache) => `clear the ${cache.name} cache`,
      confirm: (cache) => cacheClearConfirm(cache, cacheSize(cache.bytes)),
    },
  )

  // caches whose entry table is expanded, by name
  let expanded = $state<string[]>([])
  const toggle = (name: string) =>
    (expanded = expanded.includes(name)
      ? expanded.filter((n) => n !== name)
      : [...expanded, name])

  const entryLabel = (e: DiskCacheEntry) =>
    e.unreadable ? 'unreadable' : e.component || e.model || e.id.slice(0, 12)

  const entryWhat = (e: DiskCacheEntry) =>
    e.unreadable
      ? `unreadable entry ${e.id.slice(0, 12)}`
      : `${e.component} of ${e.model}`

  const deleteEntry = action(
    (cache: DiskCacheInfo, entry: DiskCacheEntry) =>
      settle(() => api.deleteCacheEntry(cache.name, entry.id)),
    {
      doing: (cache, entry) => `delete ${entryLabel(entry)} from ${cache.name}`,
      confirm: (cache, entry) =>
        cacheEntryConfirm(
          cache.name,
          entryWhat(entry),
          cacheSize(entry.bytes),
          cache.cost_of_clearing,
        ),
    },
  )
  const working = $derived(clear.busy || deleteEntry.busy)
</script>

<div class="panel caches">
  <h2><HardDrive size={15} />Caches</h2>
  <LoadState of={read} label="the caches" empty={caches.length === 0}>
    {#snippet emptyText()}No disk caches registered.{/snippet}
    {#if clearError}<ErrorLine error={clearError} doing="clear that" />{/if}
    <div class="tablewrap">
      <table class="datatable">
        <thead>
          <tr>
            <th>cache</th>
            <th>root</th>
            <th>size</th>
            <th>files</th>
            <th>last used</th>
            <th>enabled</th>
            <th>disk</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          {#each caches as cache (cache.name)}
            <tr>
              <td>{cache.name}</td>
              <td><code>{cache.root}</code></td>
              <td>{cacheSize(cache.bytes)}</td>
              <td>{cache.files}</td>
              <td class="muted">
                {cache.last_used ? formatMtime(cache.last_used) : '—'}
              </td>
              <td>{cache.enabled ? 'yes' : 'no'}</td>
              <td>
                {#if cache.disk_total && cache.disk_free != null}
                  <DiskMeter
                    freeBytes={cache.disk_free}
                    totalBytes={cache.disk_total}
                    title="disk holding the {cache.name} cache"
                  />
                {:else}
                  <span class="muted">—</span>
                {/if}
              </td>
              <td>
                <button
                  class="quiet icon"
                  disabled={!cache.clearable || working}
                  title={cache.clearable
                    ? `clear the ${cache.name} cache`
                    : (cache.not_clearable_reason ?? 'not clearable')}
                  aria-label="clear the {cache.name} cache"
                  onclick={() => clear.run(cache)}
                >
                  <Trash2 size={14} />
                </button>
              </td>
            </tr>
            {#if cache.enabled && cache.low_disk}
              <tr class="reason">
                <td></td>
                <td colspan="7" class="warn lowdisk">
                  <TriangleAlert size={14} />volume under the {cache.name}
                  cache's free-space floor - new entries are not saved
                </td>
              </tr>
            {/if}
            {#if Array.isArray(cache.entries)}
              {@const open = expanded.includes(cache.name)}
              <tr class="reason">
                <td></td>
                <td colspan="7">
                  <button
                    class="quiet withicon entrytoggle"
                    aria-expanded={open}
                    onclick={() => toggle(cache.name)}
                  >
                    {#if open}<ChevronDown size={14} />{:else}<ChevronRight
                        size={14}
                      />{/if}
                    {cache.entries.length}
                    {cache.entries.length === 1 ? 'entry' : 'entries'}
                  </button>
                </td>
              </tr>
              {#if open}
                <tr class="entries">
                  <td></td>
                  <td colspan="7">
                    <div class="tablewrap">
                      <table class="datatable entrytable">
                        <thead>
                          <tr>
                            <th>component</th>
                            <th>model</th>
                            <th>subfolder</th>
                            <th>size</th>
                            <th>created</th>
                            <th>last hit</th>
                            <th>id</th>
                            <th></th>
                          </tr>
                        </thead>
                        <tbody>
                          {#each cache.entries as entry (entry.id)}
                            <tr>
                              <td class:muted={entry.unreadable}>
                                {entry.unreadable
                                  ? 'unreadable'
                                  : entry.component}
                              </td>
                              <td>{entry.model || '—'}</td>
                              <td>{entry.subfolder || '—'}</td>
                              <td>{cacheSize(entry.bytes)}</td>
                              <td class="muted">
                                {entry.created
                                  ? formatMtime(entry.created)
                                  : '—'}
                              </td>
                              <td class="muted">
                                {entry.last_hit != null
                                  ? formatMtime(entry.last_hit)
                                  : '—'}
                              </td>
                              <td>
                                <code title={entry.id}>
                                  {entry.id.slice(0, 12)}
                                </code>
                              </td>
                              <td>
                                <button
                                  class="quiet icon"
                                  disabled={!cache.clearable || working}
                                  title={cache.clearable
                                    ? `delete ${entryLabel(entry)} from the ${cache.name} cache`
                                    : (cache.not_clearable_reason ??
                                      'not clearable')}
                                  aria-label="delete {entryLabel(
                                    entry,
                                  )} from the {cache.name} cache"
                                  onclick={() => deleteEntry.run(cache, entry)}
                                >
                                  <Trash2 size={14} />
                                </button>
                              </td>
                            </tr>
                          {/each}
                        </tbody>
                      </table>
                    </div>
                  </td>
                </tr>
              {/if}
            {/if}
            {#if !cache.clearable && cache.not_clearable_reason}
              <tr class="reason">
                <td></td>
                <td colspan="7" class="muted">{cache.not_clearable_reason}</td>
              </tr>
            {/if}
          {/each}
        </tbody>
      </table>
    </div>
  </LoadState>
</div>

<style>
  .warn {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    max-width: 80ch;
  }

  tr.reason td {
    border-top: none;
    padding-top: 0;
    font-size: 0.8rem;
  }

  tr.entries > td {
    border-top: none;
  }

  .entrytable {
    font-size: 0.85rem;
  }

  .lowdisk {
    display: table-cell;
  }
</style>
