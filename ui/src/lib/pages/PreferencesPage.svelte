<script lang="ts">
  import { action } from '../action.svelte'
  import { api } from '../api'
  import { resource } from '../resource.svelte'
  import type {
    PreferenceEntry,
    PreferenceReset,
    PreferenceSet,
  } from '../types'
  import PreferenceCard from '../ladder/PreferenceCard.svelte'
  import {
    TIERS,
    keyProblem,
    staleEtag,
    type Ladder,
  } from '../ladder/ladderDraft'
  import Segmented from '../ui/Segmented.svelte'
  import LoadState from '../ui/LoadState.svelte'
  import StaleNotice from '../ui/StaleNotice.svelte'
  import Suggest from '../ui/Suggest.svelte'

  // Server → Preferences (#1762 stage C): the `preference:` ladders this
  // server resolves, shipped or its own, each editable here, and the tier a
  // reference without one runs at. The routes are the store's only writer
  // (dw/server/routes/system.py); a PUT on one key refreshes the
  // `workflows` change kind, and a resolution reads the host's cards
  const read = resource(
    () => api.getPreferences({ resolve: true, full: true }),
    {
      watch: ['workflows', 'models'],
    },
  )
  const catalog = resource(() => api.listWorkflows(), { watch: ['workflows'] })
  const names = $derived(catalog.data?.workflows ?? [])
  // a workflow a shipped library holds opens under Examples, as the
  // Workflows page links it
  const details = $derived(catalog.data?.details ?? {})
  const readOnly = (name: string) => details[name]?.writable === false

  const stored = $derived(Object.keys(read.data?.preferences ?? {}).sort())
  // keys added here and not saved yet, after the stored ones, each with the
  // ladder its edit starts from
  let fresh = $state<Record<string, Ladder>>({})
  const shown = $derived([
    ...stored,
    ...Object.keys(fresh).filter((k) => !stored.includes(k)),
  ])
  // the one card in the chain editor; the others' Edit waits for it
  let editing = $state<string | null>(null)
  const forget = (key: string) => {
    fresh = Object.fromEntries(Object.entries(fresh).filter(([k]) => k !== key))
  }

  function withEntry(key: string, entry: PreferenceEntry | null) {
    const data = read.data
    if (!data) return
    const preferences = { ...data.preferences }
    // a save or reset answers without who uses the key: that has not changed
    if (entry)
      preferences[key] = {
        used_by: data.preferences[key]?.used_by,
        contract: data.preferences[key]?.contract,
        ...entry,
      }
    else delete preferences[key]
    read.adopt({ ...data, preferences })
  }

  function saved({ key, ...entry }: PreferenceSet) {
    withEntry(key, entry)
    forget(key)
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

  // New preference (spec §7): a name and the first model for the default
  // tier make a card in the editor; nothing is saved until its Save
  let creating = $state(false)
  let newKey = $state('')
  let firstModel = $state('')
  let keyRefused = $state<string | null>(null)
  function closeNew() {
    creating = false
    newKey = ''
    firstModel = ''
    keyRefused = null
  }
  function addKey(event: SubmitEvent, tier: string) {
    event.preventDefault()
    const name = newKey.trim()
    keyRefused = keyProblem(name, shown)
    if (keyRefused) return
    fresh = { ...fresh, [name]: { [tier]: [firstModel.trim()] } }
    editing = name
    closeNew()
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
</script>

<LoadState of={read} label="the preferences" />

{#if read.data}
  {@const data = read.data}
  <p class="muted intro">
    When a workflow or agent asks for an image (or another kind of output)
    without naming a model, dw picks one from that preference's list. Each list
    has three tiers — Draft is fastest, Best is highest quality — and dw runs
    the first model in the tier that this machine can run.
  </p>

  <div class="panel top">
    <span class="muted">Requests that don't name a tier run at</span>
    <Segmented
      label="Requests that don't name a tier run at"
      size="sm"
      options={TIERS.map((t) => ({
        value: t,
        label: t.charAt(0).toUpperCase() + t.slice(1),
      }))}
      value={data.default_tier as (typeof TIERS)[number]}
      onselect={(tier) => void setTier.run(tier, data.etag)}
    />
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
    <PreferenceCard
      {key}
      entry={data.preferences[key] ?? null}
      catalog={names}
      defaultTier={data.default_tier}
      bind:editing={
        () => editing === key,
        (open) => {
          if (open) editing = key
          else if (editing === key) editing = null
        }
      }
      locked={editing !== null && editing !== key}
      seed={fresh[key]}
      {readOnly}
      onsaved={saved}
      onreset={wasReset}
      onreload={() => void read.refresh()}
      ondiscard={() => forget(key)}
    />
  {/each}

  {#if creating}
    <form
      class="panel addkey"
      aria-label="New preference"
      onsubmit={(e) => addKey(e, data.default_tier)}
    >
      <label for="new-key">Name</label>
      <input
        id="new-key"
        placeholder="image_edit"
        bind:value={newKey}
        oninput={() => (keyRefused = null)}
        aria-invalid={keyRefused ? true : undefined}
        aria-describedby={keyRefused ? 'new-key-problem' : undefined}
      />
      <p class="muted">
        A workflow uses it by naming preference:&lt;name&gt; as a step's
        workflow path.
      </p>
      <Suggest
        aria-label="First model for the {data.default_tier} tier"
        placeholder="first model for {data.default_tier}…"
        suggestions={names}
        bind:value={firstModel}
      />
      <span class="buttons">
        <button type="button" class="quiet" onclick={closeNew}>Cancel</button>
        <button type="submit" disabled={!newKey.trim() || !firstModel.trim()}>
          Create
        </button>
      </span>
      {#if keyRefused}
        <p id="new-key-problem" class="warn" role="alert">{keyRefused}</p>
      {/if}
    </form>
  {:else}
    <button
      type="button"
      class="quiet"
      disabled={editing !== null}
      title={editing !== null
        ? 'Save or discard the open edit first'
        : undefined}
      onclick={() => (creating = true)}
    >
      New preference
    </button>
  {/if}
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
  .addkey {
    padding: var(--space-3) var(--space-4);
  }
  .addkey input {
    width: auto;
    min-width: 12rem;
  }
  .buttons {
    display: flex;
    gap: var(--space-2);
  }
  .addkey p {
    flex-basis: 100%;
    margin: 0;
  }
  .warn {
    color: var(--warn);
  }
</style>
