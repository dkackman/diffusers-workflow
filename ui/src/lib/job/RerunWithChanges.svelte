<script lang="ts">
  import { SlidersHorizontal } from '@lucide/svelte'
  import { api } from '../api'
  import { displayValue } from '../editor'
  import { go } from '../router.svelte'
  import { notify } from '../toast'
  import Modal from '../ui/Modal.svelte'
  import VariablesForm from '../editor/VariablesForm.svelte'
  import type { JobDetail } from '../types'
  import { promotionTitle, promotionsOf, type Promotion } from './promotions'

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
  let error = $state('')
  let submitting = $state(false)

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
    error = ''
    open = true
  }

  async function queue() {
    if (!changedCount || submitting) return
    submitting = true
    error = ''
    try {
      const next = await api.rerunJob(job.id, { arguments: { ...changed } })
      open = false
      go('ws', job.workspace, 'jobs', next.id)
    } catch (e) {
      // A refused value is the user's to fix: the modal stays open on it
      error = e instanceof Error ? e.message : String(e)
    } finally {
      submitting = false
    }
  }

  // The selectors come from the workflow file as it is now - the job's own
  // definition has its bundles folded in and its `presets` stripped. Only a
  // job run by catalog name can be looked up; anything else has none.
  let presets = $state<Record<string, Record<string, unknown>> | null>(null)
  $effect(() => {
    const name = job.workflow_name
    presets = null
    if (!name) return
    let stopped = false
    api
      .getWorkflow(name)
      .then(({ definition }) => {
        if (stopped) return
        const found = (definition as Record<string, unknown>)?.presets
        presets =
          found && typeof found === 'object'
            ? (found as Record<string, Record<string, unknown>>)
            : null
      })
      .catch(() => {
        /* inline or deleted workflow: no promotions */
      })
    return () => {
      stopped = true
    }
  })

  const promotions = $derived(
    promotionsOf(presets, declared, job.arguments ?? {}, values),
  )

  async function promote(p: Promotion) {
    try {
      const next = await api.rerunJob(job.id, {
        arguments: { [p.selector]: p.value },
      })
      go('ws', job.workspace, 'jobs', next.id)
    } catch (e) {
      notify.error(e instanceof Error ? e.message : String(e))
    }
  }
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
    onclick={() => promote(p)}
    title={promotionTitle(p)}><code>{p.selector}</code>: {p.value}</button
  >
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
  {#if error}<p class="error" role="alert">{error}</p>{/if}
  <div class="actions">
    {#if !changedCount}<span class="muted">nothing changed</span>{/if}
    <button class="quiet" onclick={() => (open = false)}>Cancel</button>
    <button
      onclick={queue}
      disabled={!changedCount || submitting}
      title={changedCount
        ? `queue a rerun changing ${Object.keys(changed).join(', ')}`
        : 'nothing changed - type a value that differs from this run'}
      >{submitting ? 'Queueing…' : 'Queue'}</button
    >
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
  .error {
    color: var(--bad);
  }
</style>
