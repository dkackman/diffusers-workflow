<script lang="ts">
  import { api } from '../api'
  import { resource } from '../resource.svelte'
  import { serverHref, type ModelsTab } from '../routes'
  import { gbFromBytes as gb } from '../format'
  import DiskMeter from '../ui/DiskMeter.svelte'
  import PageHeader from '../ui/PageHeader.svelte'
  import ModelsDownloaded from './ModelsDownloaded.svelte'
  import PreferencesPage from './PreferencesPage.svelte'

  let { tab }: { tab: ModelsTab } = $props()

  // The hub cache follows the models feed (a delete or a download, here or
  // by an agent); the header, the tab count and the Downloaded table all show
  // this one read
  const cacheRead = resource(() => api.listModels(), { watch: ['models'] })
  const cache = $derived(cacheRead.data ?? null)
</script>

<PageHeader title="Models" baseline>
  {#if cache}
    <span class="muted" title={cache.cache_dir}>
      {cache.repos.length} cached · {gb(cache.size_on_disk)} GB
    </span>
  {/if}
  {#snippet actions()}
    {#if cache && cache.disk_total && cache.disk_free != null}
      <DiskMeter
        freeBytes={cache.disk_free}
        totalBytes={cache.disk_total}
        title="disk holding the hub cache"
        icon
      />
    {/if}
  {/snippet}
</PageHeader>

<nav class="tabs" aria-label="Models">
  <a
    href={serverHref('models', 'preferences')}
    aria-current={tab === 'preferences' ? 'page' : undefined}
  >
    Preferences
  </a>
  <a
    href={serverHref('models', 'downloaded')}
    aria-current={tab === 'downloaded' ? 'page' : undefined}
  >
    Downloaded{cache ? ` · ${cache.repos.length}` : ''}
  </a>
</nav>

{#if tab === 'downloaded'}
  <ModelsDownloaded {cacheRead} />
{:else}
  <PreferencesPage />
{/if}

<style>
  .tabs {
    display: inline-flex;
    margin-bottom: 0.8rem;
  }
  .tabs a {
    padding: 0.3rem 0.8rem;
    border: 1px solid var(--line);
    color: var(--muted);
    text-decoration: none;
    font-size: var(--t-sm);
  }
  .tabs a:first-child {
    border-radius: var(--radius-1) 0 0 var(--radius-1);
  }
  .tabs a:last-child {
    border-radius: 0 var(--radius-1) var(--radius-1) 0;
  }
  .tabs a + a {
    margin-left: -1px;
  }
  .tabs a[aria-current='page'] {
    border-color: var(--accent);
    color: var(--accent);
    position: relative;
    z-index: 1;
  }
</style>
