<script lang="ts">
  import { action } from '../action.svelte'
  import { api } from '../api'
  import { resource } from '../resource.svelte'
  import type {
    PreferenceEntry,
    PreferenceReset,
    PreferenceSet,
    PreferencesRead,
  } from '../types'
  import LadderCard from '../ladder/LadderCard.svelte'
  import { TIERS, keyProblem, staleEtag } from '../ladder/ladderDraft'
  import LoadState from '../ui/LoadState.svelte'
  import PageHeader from '../ui/PageHeader.svelte'
  import StaleNotice from '../ui/StaleNotice.svelte'

  // Server → Preferences (#1762 stage C): the `preference:` ladders this
  // server resolves, shipped or its own, each editable here, and the tier a
  // reference without one runs at. The routes are the store's only writer
  // (dw/server/routes/system.py); a PUT on one key refreshes the
  // `workflows` change kind, and a resolution reads the host's cards
  const read = resource(() => api.getPreferences({ resolve: true }), {
    watch: ['workflows', 'models'],
  })
  const catalog = resource(() => api.listWorkflows(), { watch: ['workflows'] })
  const names = $derived(catalog.data?.workflows ?? [])

  const stored = $derived(Object.keys(read.data?.preferences ?? {}).sort())
  // keys added here and not saved yet, after the stored ones
  let fresh = $state<string[]>([])
  const shown = $derived([
    ...stored,
    ...fresh.filter((k) => !stored.includes(k)),
  ])

  function withEntry(key: string, entry: PreferenceEntry | null) {
    const data = read.data
    if (!data) return
    const preferences = { ...data.preferences }
    if (entry) preferences[key] = entry
    else delete preferences[key]
    read.adopt({ ...data, preferences })
  }

  function saved({ key, ...entry }: PreferenceSet) {
    withEntry(key, entry)
    fresh = fresh.filter((k) => k !== key)
  }

  function wasReset({
    key,
    removed,
    source,
    etag,
    ladder,
    resolutions,
  }: PreferenceReset) {
    if (removed || !source || !etag || !ladder) withEntry(key, null)
    else withEntry(key, { source, etag, ladder, resolutions })
  }

  let newKey = $state('')
  let keyRefused = $state<string | null>(null)
  function addKey(event: SubmitEvent) {
    event.preventDefault()
    const name = newKey.trim()
    keyRefused = keyProblem(name, shown)
    if (keyRefused) return
    fresh = [...fresh, name]
    newKey = ''
  }

  // the default tier: a 412 asks Reload or Overwrite, as a key's does
  let tierStale = $state<{ tier: string; etag: string | null } | null>(null)
  const setTier = action(
    async (tier: string, ifMatch: string | undefined) => {
      tierStale = null
      try {
        const answer = await api.setDefaultTier(tier, ifMatch)
        if (read.data) read.adopt({ ...read.data, ...answer })
        return answer
      } catch (e) {
        const etag = staleEtag(e)
        if (etag === undefined) throw e
        tierStale = { tier, etag }
        return null
      }
    },
    {
      doing: 'set the default tier',
      success: (r) => `The default tier is ${r.default_tier}`,
    },
  )

  function chooseTier(event: Event, data: PreferencesRead) {
    const tier = (event.currentTarget as HTMLSelectElement).value
    void setTier.run(tier, data.etag)
  }
</script>

<PageHeader title="Preferences" />

<LoadState of={read} label="the preferences" />

{#if read.data}
  {@const data = read.data}
  <p class="muted intro">
    A <code>preference:&lt;key&gt;</code> workflow path runs the first target in its
    tier that fits this host. A key this server saved replaces the shipped one whole;
    Reset to shipped drops it.
  </p>

  <div class="panel top">
    <label for="default-tier">Default tier</label>
    <select
      id="default-tier"
      value={data.default_tier}
      disabled={setTier.busy}
      onchange={(e) => chooseTier(e, data)}
    >
      {#each TIERS as tier (tier)}<option value={tier}>{tier}</option>{/each}
    </select>
    <span class="muted">what a reference with no tier runs at</span>
    {#if tierStale}
      {@const held = tierStale}
      <StaleNotice
        what="The default tier"
        onreload={() => {
          tierStale = null
          void read.refresh()
        }}
        onoverwrite={() => setTier.run(held.tier, held.etag ?? undefined)}
      />
    {/if}
  </div>

  {#each shown as key (key)}
    <LadderCard
      {key}
      entry={data.preferences[key] ?? null}
      catalog={names}
      defaultTier={data.default_tier}
      onsaved={saved}
      onreset={wasReset}
      onreload={() => void read.refresh()}
      ondiscard={() => (fresh = fresh.filter((k) => k !== key))}
    />
  {/each}

  <form class="addkey" onsubmit={addKey}>
    <label for="new-key">New key</label>
    <input
      id="new-key"
      placeholder="image_edit"
      bind:value={newKey}
      oninput={() => (keyRefused = null)}
      aria-invalid={keyRefused ? true : undefined}
      aria-describedby={keyRefused ? 'new-key-problem' : undefined}
    />
    <button type="submit" class="quiet" disabled={!newKey.trim()}>
      Add key
    </button>
    {#if keyRefused}
      <p id="new-key-problem" class="warn" role="alert">{keyRefused}</p>
    {/if}
  </form>
{/if}

<style>
  .intro {
    margin: 0 0 var(--space-3);
  }
  .top,
  .addkey {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: var(--space-2) var(--space-3);
  }
  .top {
    padding: var(--space-3) var(--space-4);
    margin-bottom: var(--space-3);
  }
  .top select,
  .addkey input {
    width: auto;
    min-width: 12rem;
  }
  .addkey p {
    flex-basis: 100%;
    margin: 0;
  }
  .warn {
    color: var(--warn);
  }
</style>
