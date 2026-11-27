<script lang="ts">
  import { action } from '../action.svelte'
  import { api } from '../api'
  import { rungsFrom } from '../preferences'
  import { wsHref } from '../routes'
  import type {
    PreferenceEntry,
    PreferenceReset,
    PreferenceSet,
  } from '../types'
  import { workspace } from '../workspace.svelte'
  import StaleNotice from '../ui/StaleNotice.svelte'
  import LadderCard from './LadderCard.svelte'
  import TierAnswer from './TierAnswer.svelte'
  import { keyLabel } from './preferenceCopy'
  import { staleEtag, targetName } from './ladderDraft'

  // One preference key on Server → Preferences (#1762): what each tier runs
  // on this machine, in words. Edit opens the ladder editor (LadderCard)
  let {
    key,
    entry,
    defaultTier,
    catalog,
    onsaved,
    onreset,
    onreload,
    ondiscard,
  }: {
    key: string
    /** The key as the server holds it; null for a key added here, not yet saved */
    entry: PreferenceEntry | null
    defaultTier: string
    catalog: readonly string[]
    onsaved: (saved: PreferenceSet) => void
    onreset: (reset: PreferenceReset) => void
    onreload: () => void
    ondiscard?: () => void
  } = $props()

  // a key added here has nothing to read yet: it opens in the editor
  // svelte-ignore state_referenced_locally
  let editing = $state(!entry)
  const rungs = $derived(rungsFrom(entry?.resolutions))
  const tiers = $derived(Object.keys(entry?.ladder ?? {}))
  const used = $derived(entry?.used_by ?? [])
  const namesIn = (tier: string) =>
    ((entry?.ladder[tier] as unknown[] | undefined) ?? []).map((t) =>
      targetName(t as Parameters<typeof targetName>[0]),
    )

  let stale = $state<string | null | undefined>(undefined)
  const reset = action(
    async (ifMatch: string | undefined) => {
      stale = undefined
      try {
        const answer = await api.resetPreference(key, ifMatch)
        onreset(answer)
        return answer
      } catch (e) {
        const etag = staleEtag(e)
        if (etag === undefined) throw e
        stale = etag
        return null
      }
    },
    {
      doing: () => `reset ${key}`,
      success: (r) => `${r.key} is back to what ships`,
      confirm: () => ({
        message: `Restore the shipped list for ${key}? This server's own list is dropped.`,
        confirmLabel: 'Reset',
        danger: true,
      }),
    },
  )
</script>

{#if editing}
  <div class="editing">
    <LadderCard
      {key}
      {entry}
      {catalog}
      {defaultTier}
      onsaved={(s) => {
        onsaved(s)
        editing = false
      }}
      onreset={(r) => {
        onreset(r)
        editing = false
      }}
      {onreload}
      {ondiscard}
    />
    {#if entry}
      <button type="button" class="quiet" onclick={() => (editing = false)}>
        Close editor
      </button>
    {/if}
  </div>
{:else if entry}
  <section class="panel card" data-key={key} aria-label="preference {key}">
    <header>
      <h2>{keyLabel(key)}</h2>
      <code>{key}</code>
      <span class="chip" data-source={entry.source}>
        {entry.source === 'server' ? 'changed on this server' : 'as shipped'}
      </span>
      <span class="actions">
        <button type="button" class="quiet" onclick={() => (editing = true)}>
          Edit
        </button>
        {#if entry.source === 'server'}
          <button
            type="button"
            class="quiet danger"
            disabled={reset.busy}
            onclick={() => reset.run(entry.etag)}
          >
            Reset to shipped
          </button>
        {/if}
      </span>
    </header>

    {#if stale !== undefined}
      <StaleNotice
        what={key}
        onreload={() => {
          stale = undefined
          onreload()
        }}
        onoverwrite={() => reset.run(stale ?? undefined)}
      />
    {/if}

    {#if used.length}
      <details class="used">
        <summary>
          used by {used.length}
          {used.length === 1 ? 'workflow' : 'workflows'}
        </summary>
        <ul>
          {#each used as name (name)}
            <li>
              <a
                href={wsHref(
                  workspace.current,
                  'workflows',
                  ...name.split('/'),
                )}
              >
                {name}
              </a>
            </li>
          {/each}
        </ul>
      </details>
    {/if}

    <div class="tiers">
      {#each tiers as tier (tier)}
        <TierAnswer
          {tier}
          isDefault={tier === defaultTier}
          rung={rungs.find((r) => r.tier === tier)}
          names={namesIn(tier)}
        />
      {/each}
    </div>
  </section>
{/if}

<style>
  .card {
    margin-bottom: var(--space-3);
    display: flex;
    flex-direction: column;
    gap: var(--space-3);
    padding: var(--space-3) var(--space-4);
  }
  .editing {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: var(--space-2);
    margin-bottom: var(--space-3);
  }
  .editing :global(.ladder) {
    align-self: stretch;
    margin-bottom: 0;
  }
  header {
    display: flex;
    flex-wrap: wrap;
    align-items: baseline;
    gap: var(--space-2);
  }
  h2 {
    margin: 0;
    font-size: var(--t-md);
  }
  .actions {
    margin-left: auto;
    display: flex;
    gap: var(--space-2);
  }
  .used {
    font-size: var(--t-sm);
  }
  .used summary {
    cursor: pointer;
    color: var(--muted);
  }
  .used ul {
    margin: var(--space-1) 0 0;
    padding-left: 1.4em;
  }
  .tiers {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(16rem, 1fr));
    gap: var(--space-3) var(--space-4);
  }
</style>
