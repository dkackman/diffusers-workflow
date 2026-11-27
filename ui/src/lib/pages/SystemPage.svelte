<script lang="ts">
  import { api } from '../api'
  import CachesPanel from '../CachesPanel.svelte'
  import CopyButton from '../CopyButton.svelte'
  import SoftwareVersions from '../SoftwareVersions.svelte'
  import { resource } from '../resource.svelte'
  import LoadState from '../ui/LoadState.svelte'
  import PageHeader from '../ui/PageHeader.svelte'
  import {
    jobHistoryLine,
    logsLine,
    runtimeRows,
    runtimeText,
  } from '../serverinfo'
  import type { ServerInfo } from '../types'

  // Static while the server runs; a reconnect or an accepted token raises
  // the epoch, which reads it again, so the page recovers on its own
  const read = resource(() => api.server(), { watch: [] })
  const info = $derived<ServerInfo | null>(read.data ?? null)
</script>

<PageHeader title="System" />

<LoadState of={read} label="the server details" />

{#if info}
  <div class="panel">
    <h2>Server</h2>
    <dl class="kv">
      <dt>Hostname</dt>
      <dd>{info.hostname}</dd>
      <dt>Version</dt>
      <dd>{info.version}</dd>
      <dt>Device</dt>
      <dd>{info.device}</dd>
    </dl>
  </div>

  <div class="panel">
    <h2>Software</h2>
    <SoftwareVersions />
  </div>

  <div class="panel runtime">
    <h2>
      Runtime
      <CopyButton text={runtimeText(info)} title="copy the runtime versions" />
    </h2>
    <dl class="kv">
      {#each runtimeRows(info.runtime) as row (row.name)}
        <dt>{row.name}</dt>
        <dd>
          {#if row.value != null}
            <code>{row.value}</code>
          {:else}
            <span class="muted">{row.absent}</span>
          {/if}
        </dd>
      {/each}
    </dl>
  </div>

  <div class="panel">
    <h2>Directories</h2>
    <dl class="kv">
      <dt>Workspace</dt>
      <dd>
        {#if info.directories.workspace}
          <code>{info.directories.workspace}</code>
        {:else}
          <span class="muted">none — directories set individually</span>
        {/if}
      </dd>
      <dt>Workflows</dt>
      <dd><code>{info.directories.workflows}</code></dd>
      <dt>Outputs</dt>
      <dd><code>{info.directories.outputs}</code></dd>
      <dt>Prompts</dt>
      <dd>
        {#if info.directories.prompts}
          <code>{info.directories.prompts}</code>
        {:else}
          <span class="muted">none configured</span>
        {/if}
      </dd>
      <dt>Assets</dt>
      <dd>
        {#if info.directories.assets}
          <code>{info.directories.assets}</code>
        {:else}
          <span class="muted">none configured</span>
        {/if}
      </dd>
    </dl>
  </div>

  <div class="panel">
    <h2>Storage</h2>
    <dl class="kv">
      <dt>Job history</dt>
      <dd>{jobHistoryLine(info.storage)}</dd>
      <dt>Logs</dt>
      <dd>{logsLine(info.storage)}</dd>
    </dl>
  </div>

  <CachesPanel />
{/if}

<style>
  .panel {
    margin-bottom: 0.8rem;
  }
</style>
