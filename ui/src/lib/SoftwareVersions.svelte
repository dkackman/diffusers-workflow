<script lang="ts">
  import { RefreshCw } from '@lucide/svelte'
  import { api } from './api'
  import { action } from './action.svelte'
  import CopyButton from './CopyButton.svelte'
  import { UPDATE_DIFFUSERS_CONFIRM, UPDATE_DW_CONFIRM } from './destructive'
  import { resource } from './resource.svelte'
  import LoadState from './ui/LoadState.svelte'

  // An update started or finished elsewhere (another tab, an agent) moves
  // models; one in flight also keeps a 2 s poll while it runs
  const diffusersRead = resource(() => api.diffusersStatus(), {
    watch: ['models'],
    alsoLive: (): number | null => (updating ? 2000 : null),
  })
  const diffusers = $derived(diffusersRead.data)
  // Polled while pip runs so the badge flips when it finishes
  const updating = $derived(diffusers?.status === 'running')

  // dw itself: an installed copy updates in place and restarts; any other
  // server shows the command that does it
  const dwRead = resource(() => api.dwStatus(), {
    watch: ['models'],
    alsoLive: (): number | null => (dwUpdating ? 2000 : null),
  })
  const dw = $derived(dwRead.data)
  const dwUpdating = $derived(dw?.status === 'running')
  // The server's PEP 440 compare; null (unknown) offers no update
  const dwNewer = $derived(dw?.newer === true)

  const updateDw = action(
    async () => {
      await api.updateDw()
      await dwRead.reload()
    },
    { doing: 'start the dw update', confirm: () => UPDATE_DW_CONFIRM },
  )

  const update = action(
    async () => {
      await api.updateDiffusers()
      await diffusersRead.reload()
    },
    {
      doing: 'start the diffusers update',
      confirm: () => UPDATE_DIFFUSERS_CONFIRM,
    },
  )
</script>

<LoadState of={diffusersRead} label="the diffusers version" quiet>
  {#if diffusers}
    <div class="engine">
      <span class="enginename">diffusers</span>
      <code>{diffusers.version ?? 'not installed'}</code>
      {#if diffusers.commit}
        <span
          class="muted"
          title="installed from git commit {diffusers.commit}"
        >
          @{diffusers.commit.slice(0, 9)}
        </span>
      {/if}
      {#if updating}
        <span class="muted busy">
          <RefreshCw size={14} class="spin" />updating from GitHub…
        </span>
      {:else}
        <button
          class="quiet withicon"
          onclick={() => update.run()}
          disabled={update.busy}
          title="install the latest diffusers from GitHub - new model pipelines usually land there before a release. The worker restarts when idle so the next job uses it"
        >
          <RefreshCw size={14} />Update from GitHub
        </button>
        {#if diffusers.status === 'failed'}
          <span class="bad" title={diffusers.log ?? ''}>
            update failed: {diffusers.error} (hover for pip output)
          </span>
        {:else if diffusers.status === 'succeeded'}
          <span class="updated">updated - the next job uses it</span>
        {/if}
      {/if}
    </div>
  {/if}
</LoadState>

<LoadState of={dwRead} label="the dw version" quiet>
  {#if dw}
    <div class="engine">
      <span class="enginename">dw</span>
      <code>{dw.version}</code>
      {#if dwUpdating}
        <span class="muted busy">
          <RefreshCw size={14} class="spin" />updating - the server restarts
          when it is done…
        </span>
      {:else if dw.managed}
        {#if dwNewer}
          <button
            class="quiet withicon"
            onclick={() => updateDw.run()}
            disabled={updateDw.busy}
            title="upgrade dw to {dw.latest} and move diffusers to GitHub main, then restart the server"
          >
            <RefreshCw size={14} />Update dw
          </button>
          <span class="muted">{dw.latest} is available</span>
        {:else if dw.newer === false}
          <span class="muted">up to date</span>
        {/if}
        {#if dw.status === 'failed'}
          <span class="bad" title={dw.log ?? ''}>
            update failed: {dw.error} (hover for the output)
          </span>
        {/if}
      {:else}
        <span class="muted">
          {dwNewer ? `${dw.latest} is available - update with` : 'update with'}
        </span>
        <code>{dw.command}</code>
        <CopyButton text={dw.command} title="copy the update command" />
      {/if}
    </div>
  {/if}
</LoadState>

<style>
  .engine {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.6rem;
    margin-bottom: 0.5rem;
    font-size: 0.9rem;
  }
  .enginename {
    color: var(--muted);
  }
  .busy {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
  }
  .engine :global(.spin) {
    animation: spin 1.2s linear infinite;
  }
  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }
  @media (prefers-reduced-motion: reduce) {
    .engine :global(.spin) {
      animation: none;
    }
  }
  .updated {
    color: var(--accent);
  }
  .bad {
    font-size: 0.9rem;
  }
</style>
