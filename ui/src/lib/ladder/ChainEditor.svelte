<script lang="ts">
  import { tick } from 'svelte'
  import { ArrowDown, ArrowUp, X } from '@lucide/svelte'
  import { serverHref } from '../routes'
  import type { PreferenceCheck, PreferenceTierCheck } from '../types'
  import AddTarget from './AddTarget.svelte'
  import { downloadNote, keyLabel, reasonText } from './preferenceCopy'
  import {
    added,
    moved,
    removed,
    targetArguments,
    targetDevices,
    targetName,
    type Ladder,
  } from './ladderDraft'

  // A preference's tiers as chains being edited (#1762, spec §5): each tier
  // its targets left to right in the order they are tried, to drag, move
  // with Alt+←/→ or the arrow buttons, remove, and add to. Each chip shows
  // what the live dry run (`answer`) says of it on this host. The draft is
  // the caller's (bound); Save and Discard are the card's
  let {
    draft = $bindable(),
    catalog,
    answer,
    contract,
    base,
    defaultTier,
  }: {
    draft: Ladder
    /** The workflow names a target may name */
    catalog: readonly string[]
    /** The latest dry run of the draft; null before one lands */
    answer: PreferenceCheck | null
    /** The arguments every target of the key takes, when the key ships one */
    contract?: { required: string[]; optional: string[] }
    /** The ladder the edit started from: a model not in it is new here */
    base?: Ladder
    defaultTier?: string
  } = $props()

  const HINTS: Record<string, string> = {
    draft: 'fastest',
    best: 'highest quality',
  }

  type Shown = {
    tone: 'good' | 'warn' | 'bad' | null
    struck: boolean
    lines: string[]
    /** A download the pick needs before its first job */
    download: string | null
  }

  const known = $derived(
    new Set(Object.values(base ?? {}).flatMap((list) => list.map(targetName))),
  )

  /** What the dry run says of one target in a tier */
  function shown(check: PreferenceTierCheck | undefined, name: string): Shown {
    const out: Shown = { tone: null, struck: false, lines: [], download: null }
    if (!check) return out
    const skip = (
      (check.skipped ?? []) as { workflow?: string; reason?: string }[]
    ).find((s) => s.workflow === name)
    if (skip) {
      out.struck = true
      out.lines.push(reasonText(skip.reason ?? ''))
    } else if (check.target === name) {
      out.tone = 'good'
      out.lines.push('runs here')
      const note = downloadNote(check.notes ?? [])
      if (note) {
        out.tone = 'warn'
        out.download = note.text
      }
      if (!known.has(name)) out.lines.push('access checked on save')
    } else if (check.after?.includes(name)) {
      out.lines.push('fallback, not checked here')
    }
    const warnings = check.warnings?.[name] ?? []
    if (warnings.length) {
      out.tone = 'warn'
      out.lines.push(...warnings)
    }
    const error = check.target_errors?.[name]
    if (error) {
      out.tone = 'bad'
      out.lines.push(error)
    }
    return out
  }

  // One tier's list element, to put focus back on a chip a key moved
  const lists: Record<string, HTMLElement | null> = {}
  async function move(tier: string, i: number, by: number, refocus = false) {
    const next = moved(draft[tier], i, by)
    if (next === draft[tier]) return
    draft[tier] = next
    if (!refocus) return
    await tick()
    const to = Math.max(0, Math.min(next.length - 1, i + by))
    lists[tier]?.querySelectorAll<HTMLElement>('li.target')[to]?.focus()
  }

  function keydown(event: KeyboardEvent, tier: string, i: number) {
    if (!event.altKey) return
    if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return
    event.preventDefault()
    void move(tier, i, event.key === 'ArrowLeft' ? -1 : 1, true)
  }

  // A drag within one tier: the chip picked up, dropped on another's place
  let dragging = $state<{ tier: string; i: number } | null>(null)
  function drop(event: DragEvent, tier: string, to: number) {
    event.preventDefault()
    const from = dragging
    dragging = null
    if (!from || from.tier !== tier) return
    void move(tier, from.i, to - from.i)
  }
</script>

<div class="chains">
  {#if contract?.required.length}
    <p class="contract muted">
      Every model here must take: {contract.required.join(', ')}
    </p>
  {/if}
  <p class="order muted" aria-hidden="true">try in order →</p>
  {#each Object.keys(draft) as tier (tier)}
    {@const check = answer?.tiers[tier]}
    <div class="tier" data-tier={tier}>
      <h3>
        {keyLabel(tier)}
        {#if tier === defaultTier}<span class="tag">default</span>
        {:else if HINTS[tier]}<span class="tag">{HINTS[tier]}</span>{/if}
      </h3>
      <div class="row">
        {#if draft[tier].length}
          <ol
            class="chain"
            aria-label="{tier}, in the order tried"
            bind:this={lists[tier]}
          >
            {#each draft[tier] as target, i (i + targetName(target))}
              {@const name = targetName(target)}
              {@const state = shown(check, name)}
              <!-- A chip is a list item that takes focus so Alt+←/→ can move
                   it, and that drags; its own buttons are the fallback -->
              <!-- svelte-ignore a11y_no_noninteractive_tabindex, a11y_no_noninteractive_element_interactions -->
              <li
                class="target"
                class:good={state.tone === 'good'}
                class:warn={state.tone === 'warn'}
                class:bad={state.tone === 'bad'}
                class:dragging={dragging?.tier === tier && dragging.i === i}
                tabindex="0"
                draggable="true"
                aria-keyshortcuts="Alt+ArrowLeft Alt+ArrowRight"
                onkeydown={(e) => keydown(e, tier, i)}
                ondragstart={(e) => {
                  dragging = { tier, i }
                  e.dataTransfer?.setData('text/plain', name)
                }}
                ondragend={() => (dragging = null)}
                ondragover={(e) => {
                  if (dragging?.tier === tier) e.preventDefault()
                }}
                ondrop={(e) => drop(e, tier, i)}
              >
                <span class="head">
                  {#if state.struck}<s><code>{name}</code></s>{:else}<code>
                      {name}
                    </code>{/if}
                  <span class="moves">
                    <button
                      type="button"
                      class="bare icon"
                      aria-label="Move {name} up"
                      title="Try earlier"
                      disabled={i === 0}
                      onclick={() => move(tier, i, -1)}
                    >
                      <ArrowUp size={12} aria-hidden="true" />
                    </button>
                    <button
                      type="button"
                      class="bare icon"
                      aria-label="Move {name} down"
                      title="Try later"
                      disabled={i === draft[tier].length - 1}
                      onclick={() => move(tier, i, 1)}
                    >
                      <ArrowDown size={12} aria-hidden="true" />
                    </button>
                    <button
                      type="button"
                      class="bare icon"
                      aria-label="Remove {name}"
                      title="Remove"
                      onclick={() => (draft[tier] = removed(draft[tier], i))}
                    >
                      <X size={12} aria-hidden="true" />
                    </button>
                  </span>
                </span>
                {#each targetDevices(target) as device (device)}
                  <span class="note">{device} only</span>
                {/each}
                {#if targetArguments(target).length}
                  <span class="note">
                    with {targetArguments(target).join(', ')}
                  </span>
                {/if}
                {#each state.lines as line, n (n)}
                  <span class="note state">{line}</span>
                {/each}
                {#if state.download}
                  <a
                    class="note state"
                    href={serverHref('models', 'downloaded')}
                  >
                    {state.download}
                  </a>
                {/if}
              </li>
              {#if i < draft[tier].length - 1}
                <li class="arrow" aria-hidden="true">→</li>
              {/if}
            {/each}
          </ol>
        {:else}
          <p class="none muted">No models: requests at this tier are refused</p>
        {/if}
        <AddTarget
          {tier}
          {catalog}
          onadd={(name, devices) =>
            (draft[tier] = added(draft[tier], name, devices))}
        />
      </div>
      {#if check?.error && draft[tier].length}
        <p class="nothing">Nothing fits this machine: {check.error}</p>
      {/if}
    </div>
  {/each}
</div>

<style>
  .chains {
    display: flex;
    flex-direction: column;
    gap: var(--space-3);
    font-size: var(--t-sm);
  }
  p {
    margin: 0;
  }
  .order {
    margin-bottom: calc(-1 * var(--space-2));
  }
  h3 {
    margin: 0 0 var(--space-1);
    font-size: var(--t-sm);
    font-weight: 600;
  }
  .tag {
    margin-left: var(--space-2);
    color: var(--muted);
    font-weight: 400;
    font-size: var(--t-xs);
  }
  .row {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: var(--space-2);
  }
  .chain {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: var(--space-2);
    margin: 0;
    padding: 0;
    list-style: none;
  }
  .target {
    display: inline-flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 0.1rem;
    padding: var(--space-1) var(--space-2);
    border: 1px solid var(--line);
    border-radius: var(--radius-1);
    background: var(--panel-2);
    cursor: grab;
  }
  .target.dragging {
    opacity: 0.5;
  }
  .target.good {
    border-color: var(--good);
  }
  .target.good .state {
    color: var(--good);
  }
  .target.warn {
    border-color: var(--warn);
  }
  .target.warn .state {
    color: var(--warn);
  }
  .target.bad {
    border-color: var(--bad);
  }
  .target.bad .state {
    color: var(--bad);
  }
  .head {
    display: inline-flex;
    align-items: center;
    gap: var(--space-1);
  }
  .moves {
    display: inline-flex;
  }
  .note {
    color: var(--muted);
    font-size: var(--t-xs);
  }
  .arrow {
    color: var(--muted);
  }
  .nothing {
    margin-top: var(--space-1);
    color: var(--bad);
  }
</style>
