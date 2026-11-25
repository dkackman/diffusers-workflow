<script lang="ts">
  import { untrack } from 'svelte'
  import ErrorLine from '../ui/ErrorLine.svelte'
  import { SlidersHorizontal } from '@lucide/svelte'
  import { api } from '../api'
  import { displayValue } from '../editor'
  import { typedArguments } from '../workflow/typedArguments'
  import { go } from '../router.svelte'
  import { action } from '../action.svelte'
  import Modal from '../ui/Modal.svelte'
  import VariablesForm from '../editor/VariablesForm.svelte'
  import type { JobDetail } from '../types'
  import { promotionTitle, promotionsOf, type Promotion } from './promotions'
  import { resource } from '../resource.svelte'
  import { workflowPresets } from './rerun'

  // A finished job queued again with some of its arguments changed: a form
  // over every variable the run declared, and one button per other value of
  // each preset selector it declares. Both go through the server's rerun,
  // so the new job keeps `rerun_of`, the drawn seed and full admission.
  let {
    job,
    declared,
    seedVariable = null,
    open = $bindable(false),
  }: {
    job: JobDetail
    /** The run's definition's declared variables, or null when the job's
     * definition is not on file - the form then lists its arguments only */
    declared: Record<string, unknown> | null
    /** The variable a new seed is drawn into: its field is locked while
     * New seed is checked, as the server refuses both */
    seedVariable?: string | null
    /** Whether the form is open: the job page's error panel opens it too */
    open?: boolean
  } = $props()

  // What this run used for each variable: its declared default, under the
  // arguments it was queued with. Shown as each field's placeholder.
  const values = $derived<Record<string, unknown>>({
    ...(declared ?? {}),
    ...(job.arguments ?? {}),
  })

  let overrides = $state<Record<string, string>>({})
  let newSeed = $state(false)
  let error = $state<unknown>(null)

  // Only what differs from the run goes: the server stores the overrides
  // exactly as sent, so retyping a value the run already used (30 over 30)
  // would otherwise be named as a change it never was
  const started = (key: string) => displayValue(values[key]).trim()
  // JSON is compared parsed: the same list or object spaced differently is
  // not a change
  const parsed = (text: string): unknown => {
    try {
      return JSON.parse(text)
    } catch {
      return undefined
    }
  }
  const same = (a: string, b: string) => {
    if (a === b) return true
    const [pa, pb] = [parsed(a), parsed(b)]
    return (
      pa !== undefined &&
      typeof pa === 'object' &&
      JSON.stringify(pa) === JSON.stringify(pb)
    )
  }
  const changed = $derived(
    Object.fromEntries(
      Object.entries(overrides).filter(
        ([key, raw]) =>
          !same(raw.trim(), started(key)) &&
          // a blank over a value that was blank is no change
          !(raw.trim() === '' && started(key) === ''),
      ),
    ),
  )
  const changedCount = $derived(Object.keys(changed).length)
  // A blank over a value clears it: the server takes null as "unset"
  const sent = $derived({
    ...typedArguments(changed, values),
    ...Object.fromEntries(
      Object.entries(changed)
        .filter(([, raw]) => raw.trim() === '')
        .map(([key]) => [key, null]),
    ),
  })

  // Opened from here or from outside, the form holds the job's values
  $effect(() => {
    if (!open) return
    // Only opening seeds the form: a job or definition refresh while it is
    // open must not wipe what was typed
    untrack(() => {
      overrides = Object.fromEntries(
        Object.keys(values).map((key) => [key, displayValue(values[key])]),
      )
      newSeed = false
      error = null
    })
  })

  const queueing = action(
    async () => {
      if (!changedCount && !newSeed) return null
      error = null
      let next
      try {
        next = await api.rerunJob(job.id, {
          newSeed,
          arguments: sent,
        })
      } catch (e) {
        // A refused value is the user's to fix: the modal stays open on it
        error = e
        return null
      }
      open = false
      go('ws', job.workspace, 'jobs', next.id)
      return next
    },
    { doing: 'rerun it' },
  )

  // The selectors come from the workflow file as it is now (rerun.ts), read
  // again when a workflow is saved anywhere
  const presetsRead = resource(() => workflowPresets(job.workflow_name), {
    watch: ['workflows'],
  })
  const presets = $derived(presetsRead.data ?? null)

  // The kind of file each variable takes, read once the form is open
  const catalog = resource(
    () => (open ? api.listWorkflows({ inputs: true }) : Promise.resolve(null)),
    { watch: ['workflows'] },
  )
  const kinds = $derived(
    catalog.data?.details?.[job.workflow_name ?? '']?.inputs ?? {},
  )

  const promotions = $derived(
    promotionsOf(presets, declared, job.arguments ?? {}, values),
  )

  const promote = action(
    async (p: Promotion) => {
      const next = await api.rerunJob(job.id, {
        arguments: { [p.selector]: p.value },
      })
      go('ws', job.workspace, 'jobs', next.id)
    },
    { doing: 'run it again' },
  )
</script>

{#if Object.keys(values).length}
  <button
    class="quiet withicon"
    onclick={() => (open = true)}
    title="queue this job again with some of its variables changed - a blank field clears that value"
  >
    <SlidersHorizontal size={14} />Run again with changes…
  </button>
{/if}
{#each promotions as p (p.selector + '\u0000' + p.value)}
  <button
    class="quiet promote"
    data-promotion={p.selector}
    disabled={promote.busy}
    aria-busy={promote.busy}
    onclick={() => void promote.run(p)}
    title={promotionTitle(p)}
  >
    <code>{p.selector}</code>: {p.value}
  </button>
{/each}

<Modal bind:open label="Run again with changes" wide>
  <h2>Run again with changes</h2>
  <p class="hint muted">
    Each field holds what this job used. Change what should differ; a blank
    field clears its value.
  </p>
  <VariablesForm
    mode="override"
    variables={values}
    bind:overrides
    idPrefix="rerun-var-"
    {kinds}
    locked={newSeed && seedVariable ? new Set([seedVariable]) : new Set()}
  />
  <label class="newseed">
    <input type="checkbox" bind:checked={newSeed} />New seed
  </label>
  {#if error}<ErrorLine {error} doing="rerun it" />{/if}
  <div class="actions">
    {#if !changedCount && !newSeed}<span class="muted">
        nothing changed
      </span>{/if}
    <button class="quiet" onclick={() => (open = false)}>Cancel</button>
    <button
      onclick={() => void queueing.run()}
      disabled={(!changedCount && !newSeed) || queueing.busy}
      aria-busy={queueing.busy}
      title={changedCount || newSeed
        ? `queue a rerun${changedCount ? ` changing ${Object.keys(changed).join(', ')}` : ''}${newSeed ? ' with a new seed' : ''}`
        : 'nothing changed - change a value, or choose New seed'}
    >
      {queueing.busy ? 'Queueing…' : 'Queue'}
    </button>
  </div>
</Modal>

<style>
  .promote {
    font-size: 0.85rem;
  }
  .newseed {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    margin-top: var(--space-3);
  }
  .actions {
    display: flex;
    justify-content: flex-end;
    align-items: center;
    gap: var(--space-2);
    margin-top: var(--space-3);
  }
</style>
