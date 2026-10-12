<!-- dw's own disk caches on the server page: the registry's table with a
  clear per cache and, where a cache lists its entries, a delete per entry.
  Fetched once on mount and again after each clear - every read walks the
  cache's tree, so it is not polled. -->
<script lang="ts">
  import {
    ChevronDown,
    ChevronRight,
    HardDrive,
    Trash2,
    TriangleAlert,
  } from '@lucide/svelte'
  import { api } from './api'
  import { confirmDialog } from './confirm.svelte'
  import { formatBytes, formatMtime, gbFromBytes } from './format'
  import type { DiskCacheEntry, DiskCacheInfo } from './types'

  let caches = $state<DiskCacheInfo[] | null>(null)
  let cachesError = $state('')
  let clearing = $state<string | null>(null)
  let clearError = $state('')

  async function refreshCaches() {
    try {
      caches = (await api.listCaches()).caches
      cachesError = ''
    } catch (e) {
      cachesError = e instanceof Error ? e.message : String(e)
    }
  }
  $effect(() => {
    refreshCaches()
  })

  // formatBytes floors at 1 KB, so an empty cache would read as 1 KB
  const cacheSize = (bytes: number) => (bytes ? formatBytes(bytes) : '0 bytes')

  async function clear(cache: DiskCacheInfo) {
    if (
      !(await confirmDialog(
        `Clear the ${cache.name} cache (${cacheSize(cache.bytes)}, ` +
          `${cache.files} files) at ${cache.root}?\n${cache.cost_of_clearing}`,
        { confirmLabel: 'Clear' },
      ))
    )
      return
    clearing = cache.name
    clearError = ''
    try {
      await api.clearCache(cache.name)
    } catch (e) {
      // a 409 (a job running or queued) carries the server's reason
      clearError = e instanceof Error ? e.message : String(e)
    } finally {
      clearing = null
      await refreshCaches()
    }
  }

  // caches whose entry table is expanded, by name
  let expanded = $state<string[]>([])
  const toggle = (name: string) =>
    (expanded = expanded.includes(name)
      ? expanded.filter((n) => n !== name)
      : [...expanded, name])

  const entryLabel = (e: DiskCacheEntry) =>
    e.unreadable ? 'unreadable' : e.component || e.model || e.id.slice(0, 12)

  async function deleteEntry(cache: DiskCacheInfo, entry: DiskCacheEntry) {
    const what = entry.unreadable
      ? `unreadable entry ${entry.id.slice(0, 12)}`
      : `${entry.component} of ${entry.model}`
    if (
      !(await confirmDialog(
        `Delete the ${cache.name} cache entry ${what} ` +
          `(${cacheSize(entry.bytes)})?\n${cache.cost_of_clearing}`,
        { confirmLabel: 'Delete' },
      ))
    )
      return
    clearing = `${cache.name}/${entry.id}`
    clearError = ''
    try {
      await api.deleteCacheEntry(cache.name, entry.id)
    } catch (e) {
      // a 409 (a job or download active) carries the server's reason
      clearError = e instanceof Error ? e.message : String(e)
    } finally {
      clearing = null
      await refreshCaches()
    }
  }

  const usedPct = (cache: DiskCacheInfo) =>
    cache.disk_total && cache.disk_free != null
      ? Math.round(
          (100 * (cache.disk_total - cache.disk_free)) / cache.disk_total,
        )
      : null
</script>

<div class="panel caches">
  <h2><HardDrive size={15} />Caches</h2>
  {#if cachesError}
    <p class="warn">{cachesError}</p>
  {:else if !caches}
    <p class="muted">loading caches…</p>
  {:else if caches.length === 0}
    <p class="muted">No disk caches registered.</p>
  {:else}
    {#if clearError}
      <p class="warn" role="alert"><TriangleAlert size={14} />{clearError}</p>
    {/if}
    <div class="tablewrap">
      <table>
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
            {@const pct = usedPct(cache)}
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
                {#if pct != null}
                  <span
                    class="disk"
                    title="disk holding the {cache.name} cache"
                  >
                    <span class="diskbar"
                      ><span style="width: {pct}%"></span></span
                    >
                    {gbFromBytes(cache.disk_free ?? 0)} / {gbFromBytes(
                      cache.disk_total ?? 0,
                    )} GB free
                  </span>
                {:else}
                  <span class="muted">—</span>
                {/if}
              </td>
              <td>
                <button
                  class="quiet withicon danger"
                  disabled={!cache.clearable || clearing !== null}
                  title={cache.clearable
                    ? `clear the ${cache.name} cache`
                    : (cache.not_clearable_reason ?? 'not clearable')}
                  onclick={() => clear(cache)}
                >
                  <Trash2 size={14} />
                  {clearing === cache.name ? 'clearing…' : 'Clear'}
                </button>
              </td>
            </tr>
            {#if cache.low_disk}
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
                      <table class="entrytable">
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
                                <code title={entry.id}
                                  >{entry.id.slice(0, 12)}</code
                                >
                              </td>
                              <td>
                                <button
                                  class="quiet withicon danger"
                                  disabled={!cache.clearable ||
                                    clearing !== null}
                                  title={cache.clearable
                                    ? `delete ${entryLabel(entry)} from the ${cache.name} cache`
                                    : (cache.not_clearable_reason ??
                                      'not clearable')}
                                  onclick={() => deleteEntry(cache, entry)}
                                >
                                  <Trash2 size={14} />
                                  {clearing === `${cache.name}/${entry.id}`
                                    ? 'deleting…'
                                    : 'Delete'}
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
  {/if}
</div>

<style>
  h2 {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    margin: 0 0 0.6rem;
    font-size: 0.95rem;
  }

  .warn {
    color: var(--warn);
    display: flex;
    align-items: center;
    gap: 0.4rem;
    max-width: 80ch;
  }

  .tablewrap {
    overflow-x: auto;
  }

  table {
    width: 100%;
    border-collapse: collapse;
  }

  th {
    text-align: left;
    font-size: 0.75rem;
    text-transform: none;
    color: var(--muted);
    font-weight: 600;
    padding: 0.2rem 0.5rem 0.2rem 0;
  }

  td {
    padding: 0.25rem 0.5rem 0.25rem 0;
    border-top: 1px solid var(--line);
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

  .disk {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    color: var(--muted);
    font-size: 0.85rem;
    white-space: nowrap;
  }

  .diskbar {
    width: 90px;
    height: 6px;
    border-radius: 3px;
    background: var(--line);
    overflow: hidden;
    display: inline-block;
  }

  .diskbar > span {
    display: block;
    height: 100%;
    background: var(--muted);
    opacity: 0.7;
  }
</style>
