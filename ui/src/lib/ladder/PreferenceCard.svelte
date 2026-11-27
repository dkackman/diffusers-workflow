<script lang="ts">
  import { action } from '../action.svelte'
  import { api } from '../api'
  import { ApiError } from '../apiErrors'
  import { confirmDialog } from '../confirm.svelte'
  import { rungsFrom } from '../preferences'
  import { addLeaveGuard } from '../router.svelte'
  import { sharedHref, wsHref } from '../routes'
  import type {
    PreferenceEntry,
    PreferenceReset,
    PreferenceSet,
  } from '../types'
  import { workspace } from '../workspace.svelte'
  import ErrorLine from '../ui/ErrorLine.svelte'
  import StaleNotice from '../ui/StaleNotice.svelte'
  import ChainEditor from './ChainEditor.svelte'
  import TierAnswer from './TierAnswer.svelte'
  import { dryRun } from './dryRun.svelte'
  import { keyLabel } from './preferenceCopy'
  import {
    blockedFrom,
    draftOf,
    sameLadder,
    sendable,
    staleEtag,
    targetName,
    type Ladder,
  } from './ladderDraft'

  // One preference key on Server → Preferences (#1762): what each tier runs
  // on this machine, in words, and Edit, which turns the tiers into chains
  // (ChainEditor) checked live against this host as they change (dryRun).
  // Save writes the draft with the etag the edit started from: a 412 asks
  // Reload or Overwrite, a 422 shows the server's refusal here. Reset to
  // shipped drops this server's list, and any edit made on it
  let {
    key,
    entry,
    defaultTier,
    catalog,
    onsaved,
    onreset,
    onreload,
    ondiscard,
    editing = $bindable(false),
    locked = false,
    seed,
    readOnly = () => false,
  }: {
    key: string
    /** The key as the server holds it; null for a key added here, not yet saved */
    entry: PreferenceEntry | null
    defaultTier: string
    catalog: readonly string[]
    onsaved: (saved: PreferenceSet) => void
    onreset: (reset: PreferenceReset) => void
    /** Read the page's preferences again */
    onreload: () => void
    /** Drop a key added here and never saved */
    ondiscard?: () => void
    /** The card is in the chain editor */
    editing?: boolean
    /** Another card holds the open edit: this one's Edit waits for it */
    locked?: boolean
    /** A key added here: the ladder its edit starts from */
    seed?: Ladder
    /** A workflow the catalog holds read-only (a shipped example) */
    readOnly?: (name: string) => boolean
  } = $props()

  const rungs = $derived(rungsFrom(entry?.resolutions))
  const tiers = $derived(Object.keys(entry?.ladder ?? {}))
  const used = $derived(entry?.used_by ?? [])
  const namesIn = (tier: string) =>
    ((entry?.ladder[tier] as unknown[] | undefined) ?? []).map((t) =>
      targetName(t as Parameters<typeof targetName>[0]),
    )
  const usedHref = (name: string) =>
    readOnly(name)
      ? sharedHref('examples', ...name.split('/'))
      : wsHref(workspace.current, 'workflows', ...name.split('/'))

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
    draft = draftOf(entry?.ladder ?? seed)
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

  // The live check of the draft: only while the editor is open (an empty
  // ladder sends nothing). Targets the last full read found gated are named,
  // so the check skips them without asking the Hub
  const check = dryRun(
    () => key,
    () => (editing ? draft : {}),
    () => blockedFrom(entry?.resolutions),
  )
  const targetErrors = $derived(
    Object.values(check.answer?.tiers ?? {}).some(
      (t) => Object.keys(t.target_errors ?? {}).length > 0,
    ),
  )

  // Leaving the page with an unsaved draft asks first
  $effect(() => {
    if (!editing || !dirty) return
    return addLeaveGuard(() =>
      confirmDialog(`Leave without saving your changes to ${key}?`, {
        confirmLabel: 'Leave',
        cancelLabel: 'Keep editing',
        danger: true,
      }),
    )
  })

  let refused = $state<unknown>(null)
  // A 412: the op held back, and the etag the server holds now
  let stale = $state<{ op: 'save' | 'reset'; etag: string | null } | null>(null)

  /** The write's refusals this card answers itself; any other failure is
   * the action's toast */
  function held(op: 'save' | 'reset', e: unknown): null {
    const etag = staleEtag(e)
    if (etag !== undefined) stale = { op, etag }
    else if (op === 'save' && e instanceof ApiError && e.status === 422)
      refused = e
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
        editing = false
        onsaved(saved)
        // the answer carries no `used_by` or full check: read them again
        onreload()
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
        editing = false
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
        message: `Restore the shipped list for ${key}? This server's own list is dropped.`,
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

  function discard() {
    refused = null
    stale = null
    draft = draftOf(base)
    editing = false
    if (!entry) ondiscard?.()
  }
</script>

<section class="panel card" data-key={key} aria-label="preference {key}">
  <header>
    <h2>{keyLabel(key)}</h2>
    <code>{key}</code>
    {#if !entry}
      <span class="chip unsaved">not saved</span>
    {:else}
      <span class="chip" data-source={entry.source}>
        {entry.source === 'server' ? 'changed on this server' : 'as shipped'}
      </span>
    {/if}
    <span class="actions">
      {#if entry && !editing}
        <button
          type="button"
          class="quiet"
          disabled={locked}
          title={locked ? 'Save or discard the open edit first' : undefined}
          onclick={() => (editing = true)}
        >
          Edit
        </button>
      {/if}
      {#if entry?.source === 'server'}
        <button
          type="button"
          class="quiet danger"
          disabled={reset.busy}
          onclick={() => reset.run(baseEtag ?? entry.etag)}
        >
          Reset to shipped
        </button>
      {/if}
    </span>
  </header>

  {#if stale}
    <StaleNotice what={key} onreload={reload} onoverwrite={overwrite} />
  {/if}
  {#if refused}<ErrorLine error={refused} doing="save {key}" />{/if}

  {#if used.length}
    <details class="used">
      <summary>
        used by {used.length}
        {used.length === 1 ? 'workflow' : 'workflows'}
      </summary>
      <ul>
        {#each used as name (name)}
          <li><a href={usedHref(name)}>{name}</a></li>
        {/each}
      </ul>
    </details>
  {/if}

  {#if editing}
    <ChainEditor
      bind:draft
      {catalog}
      answer={check.answer}
      contract={entry?.contract}
      base={entry ? base : undefined}
      {defaultTier}
    />
    {#if check.refused}
      <ErrorLine error={check.refused} doing="check {key}" />
    {/if}
    <footer>
      {#if dirty}
        <span class="unsaved-note">
          Unsaved changes · Jobs already queued keep the model they were given
        </span>
        <span class="actions">
          <button type="button" class="quiet" onclick={discard}>
            Discard
          </button>
          <button
            type="button"
            disabled={empty ||
              save.busy ||
              targetErrors ||
              check.refused !== null}
            onclick={() => save.run(baseEtag)}
          >
            Save
          </button>
        </span>
      {:else}
        <span class="actions">
          <button type="button" class="quiet" onclick={discard}>
            Close editor
          </button>
        </span>
      {/if}
    </footer>
  {:else}
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
  {/if}
</section>

<style>
  .card {
    margin-bottom: var(--space-3);
    display: flex;
    flex-direction: column;
    gap: var(--space-3);
    padding: var(--space-3) var(--space-4);
  }
  header,
  footer {
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
  .unsaved-note {
    color: var(--warn);
    font-size: var(--t-sm);
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
