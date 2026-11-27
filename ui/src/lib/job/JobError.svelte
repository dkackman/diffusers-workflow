<script lang="ts">
  // A failed job's lead: the step that failed, the reason in plain words, the
  // one thing to do about it, and the engine's own text folded behind Details
  import { action } from '../action.svelte'
  import ActionButton from '../ui/ActionButton.svelte'
  import Details from '../ui/Details.svelte'
  import { serverHref } from '../routes'
  import type { JobDetail } from '../types'
  import { explainJobError } from './jobError'

  let {
    job,
    step,
    onrerun,
    canRerun,
    live,
  }: {
    job: JobDetail
    step: string | null
    onrerun: () => void
    /** Whether the rerun dialog would have fields to change */
    canRerun: boolean
    /** The job failed while this page was open: only then is it announced */
    live: boolean
  } = $props()

  const explained = $derived(explainJobError(job.error ?? ''))
  // The error and the traceback, whole, whatever the lead says
  const engine = $derived(
    [job.error, job.traceback].filter(Boolean).join('\n\n'),
  )
  const rerun = action(async () => onrerun(), { doing: 'open the rerun form' })
</script>

<div class="panel error-edge" role={live ? 'alert' : undefined}>
  {#if step}<p class="step">Failed in step <code>{step}</code>.</p>{/if}
  <p class="lead">{explained.title}</p>
  {#if explained.remedy}<p class="remedy">{explained.remedy}</p>{/if}
  {#if explained.action === 'rerun' && canRerun}
    <ActionButton act={rerun}>Run again with changes…</ActionButton>
  {:else if explained.action === 'models'}
    <a class="quiet" href={serverHref('models', 'downloaded')}>Open Models</a>
  {/if}
  <Details summary="Engine detail" detail={engine} />
</div>

<style>
  p {
    margin: 0 0 0.25rem;
    overflow-wrap: anywhere;
  }
  .lead {
    color: var(--bad);
    font-weight: 600;
  }
</style>
