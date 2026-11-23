<script lang="ts">
  import ErrorLine from '../ui/ErrorLine.svelte'
  import { SlidersHorizontal } from '@lucide/svelte'
  import { api } from '../api'
  import { displayValue } from '../editor'
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
  }: {
    job: JobDetail
    /** The run's definition's declared variables, or null when the job's
     * definition is not on file - the form then lists its arguments only */
    declared: Record<string, unknown> | null
  } = $props()

  // What this run used for each variable: its declared default, under the
  // arguments it was queued with. Shown as each field's placeholder.
  const values = $derived<Record<string, unknown>>({
    ...(declared ?? {}),
    ...(job.arguments ?? {}),
  })

  let open = $state(false)
  let overrides = $state<Record<string, string>>({})
  let error = $state<unknown>(null)

  // Only what differs from the run goes: the server stores the overrides
  // exactly as sent, so retyping a value the run already used (30 over 30)
  // would otherwise be named as a change it never was
  const changed = $derived(
    Object.fromEntries(
      Object.entries(overrides).filter(([key, raw]) => {
        const text = raw.trim()
        return text !== '' && text !== displayValue(values[key]).trim()
      }),
    ),
  )
  const changedCount = $derived(Object.keys(changed).length)

  function start() {
    overrides = {}
    error = null
    open = true
  }

  const queueing = action(
    async () => {
      if (!changedCount) return null
      error = null
      let next
      try {
        next = await api.rerunJob(job.id, { arguments: { ...changed } })
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
    onclick={start}
    title="queue this job again with some of its variables changed - what you leave blank keeps this run's value"
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
    Each field shows what this run used. Type only what should change - a blank
    field keeps this run's value, the drawn seed included.
  </p>
  <VariablesForm
    mode="override"
    variables={values}
    bind:overrides
    idPrefix="rerun-var-"
  />
  {#if error}<ErrorLine {error} doing="rerun it" />{/if}
  <div class="actions">
    {#if !changedCount}<span class="muted">nothing changed</span>{/if}
    <button class="quiet" onclick={() => (open = false)}>Cancel</button>
    <button
      onclick={() => void queueing.run()}
      disabled={!changedCount || queueing.busy}
      aria-busy={queueing.busy}
      title={changedCount
        ? `queue a rerun changing ${Object.keys(changed).join(', ')}`
        : 'nothing changed - type a value that differs from this run'}
    >
      {queueing.busy ? 'Queueing…' : 'Queue'}
    </button>
  </div>
</Modal>

<style>
  .promote {
    font-size: 0.85rem;
  }
  .actions {
    display: flex;
    justify-content: flex-end;
    align-items: center;
    gap: var(--space-2);
    margin-top: var(--space-3);
  }
</style>
