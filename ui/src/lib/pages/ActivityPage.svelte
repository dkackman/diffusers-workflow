<script lang="ts">
  import { api } from '../api'
  import ActiveDownloads from '../ActiveDownloads.svelte'
  import JobsPage from './JobsPage.svelte'
  import ServerHealthList from '../ServerHealthList.svelte'
  import { resource } from '../resource.svelte'
  import { serverHref } from '../routes'
  import PageHeader from '../ui/PageHeader.svelte'

  // Static while the server runs; a reconnect or an accepted token raises
  // the epoch, which reads it again, so the page recovers on its own
  const read = resource(() => api.server(), { watch: [] })
  const info = $derived(read.data ?? null)

  // An update started or finished elsewhere moves models; a running one is
  // polled so this line goes when it finishes. SoftwareVersions on the
  // System page shows the update itself
  const diffusersRead = resource(() => api.diffusersStatus(), {
    watch: ['models'],
    alsoLive: (): number | null => (updating ? 2000 : null),
  })
  const dwRead = resource(() => api.dwStatus(), {
    watch: ['models'],
    alsoLive: (): number | null => (updating ? 2000 : null),
  })
  const updating = $derived(
    diffusersRead.data?.status === 'running' ||
      dwRead.data?.status === 'running',
  )
</script>

<!-- What this server is doing: the queue across every workspace lives here
     because the queue is the server's, not any one workspace's -->
<PageHeader
  title="Activity"
  secondary={info
    ? `${info.hostname} · v${info.version} · ${info.device}`
    : undefined}
  baseline
/>

<div class="panel">
  <ServerHealthList />
</div>

<ActiveDownloads />

{#if updating}
  <p class="muted updating">
    <a href={serverHref('system')}>dw / diffusers updating…</a>
  </p>
{/if}

<div class="alljobs">
  <JobsPage scope="all" />
</div>

<style>
  .panel {
    margin-bottom: 0.8rem;
  }
  .updating {
    margin: 0 0 0.8rem;
    font-size: 0.9rem;
  }
  .alljobs {
    margin-top: var(--space-5);
    padding-top: var(--space-5);
    border-top: 1px solid var(--line);
  }
</style>
