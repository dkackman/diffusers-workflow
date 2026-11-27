<script lang="ts">
  import { ArrowDown, ArrowUp, X } from '@lucide/svelte'
  import { action } from '../action.svelte'
  import { api } from '../api'
  import { ApiError } from '../apiErrors'
  import { rungsFrom } from '../preferences'
  import type {
    PreferenceEntry,
    PreferenceReset,
    PreferenceSet,
  } from '../types'
  import PreferenceRungs from '../editor/PreferenceRungs.svelte'
  import ErrorLine from '../ui/ErrorLine.svelte'
  import StaleNotice from '../ui/StaleNotice.svelte'
  import Suggest from '../ui/Suggest.svelte'
  import {
    DEVICES,
    TIERS,
    added,
    draftOf,
    moved,
    removed,
    sameLadder,
    sendable,
    staleEtag,
    targetArguments,
    targetDevices,
    targetName,
    type Ladder,
  } from './ladderDraft'

  // One preference key on Server → Preferences (#1762 stage C): its tiers
  // as ordered target lists to reorder, remove from and add to, saved with
  // the key's etag as If-Match. A 422 shows the server's refusal here; a
  // 412 (someone else wrote the key since it was read) asks Reload or
  // Overwrite. Under the tiers, what each one runs on this host, as the
  // last read or save answered it
  let {
    key,
    entry,
    catalog,
    defaultTier,
    onsaved,
    onreset,
    onreload,
    ondiscard,
  }: {
    key: string
    /** The key as the server holds it; null for a key added here, not yet saved */
    entry: PreferenceEntry | null
    /** The workflow names a target may name */
    catalog: readonly string[]
    defaultTier: string
    onsaved: (saved: PreferenceSet) => void
    onreset: (reset: PreferenceReset) => void
    onreload: () => void
    /** Drop a key added here and never saved */
    ondiscard?: () => void
  } = $props()

  // The ladder the draft started from, and the etag it was read at. A read
  // that moves the key replaces an unedited draft, never an edited one; an
  // edited draft still saves against the etag it started from, so a write
  // made meanwhile is a 412, never silently written over
  let base = $state<Ladder>(draftOf(undefined))
  let draft = $state<Ladder>(draftOf(undefined))
  let baseEtag = $state<string | undefined>(undefined)
  let seen = $state<string | null | undefined>(undefined)
  function adopt() {
    base = draftOf(entry?.ladder)
    draft = draftOf(entry?.ladder)
    baseEtag = entry?.etag
  }
  $effect(() => {
    const etag = entry?.etag ?? null
    if (etag === seen) return
    if (seen === undefined || sameLadder(draft, base)) adopt()
    seen = etag
  })
  const dirty = $derived(!entry || !sameLadder(draft, base))
  const empty = $derived(Object.keys(sendable(draft)).length === 0)
  const rungs = $derived(rungsFrom(entry?.resolutions))

  let refused = $state<unknown>(null)
  // A 412: the op held back, and the etag the server holds now
  let stale = $state<{ op: 'save' | 'reset'; etag: string | null } | null>(null)

  /** The write's refusals this card answers itself; any other failure is
   * the action's toast */
  function held(op: 'save' | 'reset', e: unknown): null {
    const etag = staleEtag(e)
    if (etag !== undefined) stale = { op, etag }
    else if (e instanceof ApiError && e.status === 422) refused = e
    else throw e
    return null
  }

  const save = action(
    async (ifMatch: string | undefined) => {
      refused = null
      stale = null
      try {
        const saved = await api.setPreference(key, sendable(draft), ifMatch)
        base = draftOf(saved.ladder)
        baseEtag = saved.etag
        onsaved(saved)
        return saved
      } catch (e) {
        return held('save', e)
      }
    },
    { doing: () => `save ${key}`, success: (r) => `Saved ${r.key}` },
  )

  const reset = action(
    async (ifMatch: string | undefined) => {
      refused = null
      stale = null
      try {
        const answer = await api.resetPreference(key, ifMatch)
        // Reset drops this server's ladder and any edit made on it
        base = draftOf(answer.ladder)
        draft = draftOf(answer.ladder)
        baseEtag = answer.etag
        onreset(answer)
        return answer
      } catch (e) {
        return held('reset', e)
      }
    },
    {
      doing: () => `reset ${key}`,
      success: (r) => `${r.key} is back to what ships`,
      confirm: () => ({
        message: `Drop this server's ladder for ${key} and use the one that ships?`,
        confirmLabel: 'Reset',
        danger: true,
      }),
    },
  )

  function reload() {
    stale = null
    refused = null
    adopt()
    onreload()
  }

  function overwrite() {
    const held = stale
    if (!held) return
    const ifMatch = held.etag ?? undefined
    void (held.op === 'save' ? save.run(ifMatch) : reset.run(ifMatch))
  }

  // One add row per tier: the name typed or picked, and its devices
  type Row = { name: string; devices: string[] }
  const blank = (): Row => ({ name: '', devices: [] })
  let adding = $state<Record<string, Row>>(
    Object.fromEntries(TIERS.map((tier) => [tier, blank()])),
  )
  // a tier the server added later gets its row when the draft has it
  $effect(() => {
    for (const tier of Object.keys(draft)) adding[tier] ??= blank()
  })
  // Bits keeps the typed text in its input when the bound value is
  // cleared, so a field is remade after each add
  let made = $state<Record<string, number>>({})
  function add(tier: string) {
    const r = adding[tier]
    draft[tier] = added(draft[tier], r.name, r.devices)
    adding[tier] = blank()
    made[tier] = (made[tier] ?? 0) + 1
  }
</script>

<section class="panel ladder" data-key={key} aria-label="preference {key}">
  <header>
    <h2><code>{key}</code></h2>
    {#if !entry}
      <span class="chip unsaved">not saved</span>
    {:else}
      <span class="chip" data-source={entry.source}>{entry.source}</span>
    {/if}
    <span class="actions">
      <button
        type="button"
        disabled={!dirty || empty || save.busy}
        onclick={() => save.run(baseEtag)}
      >
        Save
      </button>
      {#if entry?.source === 'server'}
        <button
          type="button"
          class="quiet danger"
          disabled={reset.busy}
          onclick={() => reset.run(baseEtag)}
        >
          Reset to shipped
        </button>
      {/if}
      {#if !entry && ondiscard}
        <button type="button" class="quiet" onclick={ondiscard}>Discard</button>
      {/if}
    </span>
  </header>

  {#if stale}
    <StaleNotice what={key} onreload={reload} onoverwrite={overwrite} />
  {/if}
  {#if refused}<ErrorLine error={refused} doing="save {key}" />{/if}

  <div class="tiers">
    {#each Object.keys(draft).filter((t) => adding[t]) as tier (tier)}
      <div class="tier" data-tier={tier}>
        <h3>
          {tier}{#if tier === defaultTier}
            <span class="muted default">default</span>{/if}
        </h3>
        <ol>
          {#each draft[tier] as target, i (i + targetName(target))}
            {@const name = targetName(target)}
            <li>
              <code>{name}</code>
              {#each targetDevices(target) as device (device)}
                <span class="chip">{device} only</span>
              {/each}
              {#if targetArguments(target).length}
                <span class="muted">
                  with {targetArguments(target).join(', ')}
                </span>
              {/if}
              <span class="moves">
                <button
                  type="button"
                  class="bare icon"
                  aria-label="Move {name} up"
                  title="Move up"
                  disabled={i === 0}
                  onclick={() => (draft[tier] = moved(draft[tier], i, -1))}
                >
                  <ArrowUp size={14} aria-hidden="true" />
                </button>
                <button
                  type="button"
                  class="bare icon"
                  aria-label="Move {name} down"
                  title="Move down"
                  disabled={i === draft[tier].length - 1}
                  onclick={() => (draft[tier] = moved(draft[tier], i, 1))}
                >
                  <ArrowDown size={14} aria-hidden="true" />
                </button>
                <button
                  type="button"
                  class="bare icon"
                  aria-label="Remove {name}"
                  title="Remove"
                  onclick={() => (draft[tier] = removed(draft[tier], i))}
                >
                  <X size={14} aria-hidden="true" />
                </button>
              </span>
            </li>
          {:else}
            <li class="muted none">No target: this tier is left out.</li>
          {/each}
        </ol>
        <div class="add">
          {#key made[tier]}
            <Suggest
              aria-label="workflow to add to {tier}"
              placeholder="add a workflow…"
              suggestions={catalog}
              bind:value={adding[tier].name}
            />
          {/key}
          {#each DEVICES as device (device)}
            <label class="device">
              <input
                type="checkbox"
                value={device}
                bind:group={adding[tier].devices}
              />{device}
            </label>
          {/each}
          <button
            type="button"
            class="quiet"
            disabled={!adding[tier].name.trim()}
            onclick={() => add(tier)}
          >
            Add
          </button>
        </div>
      </div>
    {/each}
  </div>

  {#if rungs.length}
    <div class="picks">
      <h3>On this host</h3>
      <PreferenceRungs {key} {rungs} current={defaultTier} />
    </div>
  {/if}
</section>

<style>
  .ladder {
    margin-bottom: var(--space-3);
    display: flex;
    flex-direction: column;
    gap: var(--space-3);
    padding: var(--space-3) var(--space-4);
  }
  header {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: var(--space-2);
  }
  h2 {
    margin: 0;
    font-size: var(--t-md);
  }
  h3 {
    margin: 0 0 var(--space-1);
    font-size: var(--t-sm);
    font-weight: 600;
  }
  .default {
    margin-left: var(--space-2);
    font-weight: 400;
  }
  .actions {
    margin-left: auto;
    display: flex;
    gap: var(--space-2);
  }
  .tiers {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(16rem, 1fr));
    gap: var(--space-3) var(--space-4);
  }
  ol {
    margin: 0 0 var(--space-2);
    padding-left: 1.4em;
    font-size: var(--t-sm);
  }
  li {
    padding: 0.1rem 0;
  }
  li.none {
    list-style: none;
    margin-left: -1.4em;
  }
  li::after {
    content: '';
    display: block;
    clear: both;
  }
  li .chip {
    margin-left: var(--space-1);
  }
  .moves {
    display: inline-flex;
    float: right;
  }
  .add {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: var(--space-1) var(--space-2);
    font-size: var(--t-sm);
  }
  .device {
    display: inline-flex;
    align-items: center;
    gap: 0.2em;
  }
</style>
