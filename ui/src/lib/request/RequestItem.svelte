<script lang="ts">
  import { api, outputUrl } from '../api'
  import { confirmDialog } from '../confirm.svelte'
  import { draftAtRisk, saveFailure } from './draft'
  import { notify } from '../toast'
  import { formatExact, formatMinutes } from '../format'
  import { resultProof } from '../requestResult'
  import { wsHref } from '../routes'
  import type { JobDetail, LedgerItem } from '../types'

  let {
    item,
    workspace,
    onsave,
    onreload,
  }: {
    item: LedgerItem
    workspace: string
    /** Store the comment (null clears it); rejects with the server's error */
    onsave: (text: string | null) => Promise<void>
    /** Read the request again, after a save found it had moved on */
    onreload: () => void
  } = $props()

  // The picture that stands for the result job (resultProof): looked up
  // once per job id, and "this job has none" is remembered like a picture
  // is - a refresh hands over a new, equal `item`, and must not ask again
  let proof = $state<{ job: string; src: string | null } | null>(null)
  const resultJob = $derived(item.result?.job)
  $effect(() => {
    const job = resultJob
    if (!job) {
      proof = null
      return
    }
    if (proof?.job === job) return
    let live = true
    api.getJob(job).then(
      (detail) => {
        if (live) proof = { job, src: proofSrc(detail) }
      },
      () => {
        if (live) proof = { job, src: null }
      },
    )
    return () => (live = false)
  })
  // Until the new result job's own picture is found, the last one's is not
  // shown under a link to this one
  const picture = $derived(proof && proof.job === resultJob ? proof.src : null)
  function proofSrc(detail: JobDetail): string | null {
    const found = resultProof(detail)
    if (!found) return null
    return found.kind === 'image'
      ? outputUrl(found.file, detail.id, detail.workspace)
      : api.galleryThumbnailUrl(found.file, detail.workspace)
  }

  // A draft is the page's only local edit; a refresh replacing `item`
  // leaves it. `refused` is the server's answer to the last save, shown
  // under the draft until the next one
  let editing = $state(false)
  let draft = $state('')
  let saving = $state(false)
  let refused = $state<{ message: string; reload: boolean } | null>(null)

  function edit() {
    draft = item.comment?.text ?? ''
    refused = null
    editing = true
  }
  async function cancel() {
    if (
      draftAtRisk(draft, item.comment?.text ?? null) &&
      !(await confirmDialog('Discard this comment draft?', {
        confirmLabel: 'Discard',
        cancelLabel: 'Keep editing',
      }))
    )
      return
    editing = false
  }
  // The draft stays: it is saved again, against the request as it now is
  function reload() {
    refused = null
    onreload()
  }
  async function save() {
    saving = true
    try {
      await onsave(draft.trim() ? draft : null)
      editing = false
      refused = null
    } catch (e) {
      refused = saveFailure(e)
      if (!refused) notify.error(e instanceof Error ? e.message : String(e))
    } finally {
      saving = false
    }
  }
</script>

<li class="item panel" class:dropped={item.dropped}>
  <div class="head">
    <span
      class="chip status"
      class:running={item.status === 'running'}
      class:queued={item.status === 'queued'}
      class:succeeded={item.status === 'done'}
      class:failed={item.status === 'failed'}
      class:planned={item.status === 'planned'}
      class:gone={item.status === 'dropped'}>{item.status}</span
    >
    <strong class="title">{item.title}</strong>
    <code class="muted">{item.key}</code>
    <span class="flex"></span>
    <span class="muted num" title="estimate"
      >est. {item.estimate_minutes == null
        ? '—'
        : formatMinutes(item.estimate_minutes)}</span
    >
  </div>

  <div class="body">
    {#if picture && item.result}
      <a
        class="plain frame result"
        href={wsHref(workspace, 'jobs', item.result.job)}
        title="open the job"
      >
        <img src={picture} alt="result of {item.title}" loading="lazy" />
      </a>
    {/if}
    <div class="details">
      {#if item.depends_on?.length}
        <p class="muted small">
          depends on {#each item.depends_on as dep, i (dep)}{#if i},
            {/if}<code>{dep}</code>{/each}
        </p>
      {/if}
      {#if item.accepted_missing}
        <p class="warn small">
          accepted run missing — showing the latest succeeded run instead
        </p>
      {/if}
      {#if item.attempts.length}
        <ul class="attempts">
          {#each item.attempts as attempt (attempt.job)}
            <li>
              <a class="plain" href={wsHref(workspace, 'jobs', attempt.job)}
                >{attempt.job}</a
              >
              <span class="chip {attempt.status}">{attempt.status}</span>
              <span class="muted num"
                >{attempt.minutes == null
                  ? '—'
                  : formatMinutes(attempt.minutes)}</span
              >
              {#if attempt.note}<span class="muted">{attempt.note}</span>{/if}
            </li>
          {/each}
        </ul>
      {/if}

      <div class="comment">
        {#if editing}
          <textarea
            bind:value={draft}
            rows="3"
            aria-label="Comment on {item.key}"
            placeholder="Add a comment"></textarea>
          <div class="actions">
            <button type="button" class="quiet" onclick={save} disabled={saving}
              >Save</button
            >
            <button
              type="button"
              class="bare"
              onclick={cancel}
              disabled={saving}>Cancel</button
            >
          </div>
          {#if refused}
            <p class="error" role="alert">
              {refused.message}
              {#if refused.reload}
                <button type="button" class="quiet" onclick={reload}
                  >Reload</button
                >
              {/if}
            </p>
          {/if}
        {:else}
          {#if item.comment}
            <p class="text">{item.comment.text}</p>
            <p class="muted small">
              {formatExact(Date.parse(item.comment.at) / 1000)}
            </p>
          {/if}
          <!-- The ledger keeps a dropped item as stored: a comment
               written on one is not saved, so none is offered -->
          {#if !item.dropped}
            <button type="button" class="quiet" onclick={edit}
              >{item.comment ? 'Edit comment' : 'Add comment'}</button
            >
          {/if}
        {/if}
      </div>
    </div>
  </div>
</li>

<style>
  .item {
    list-style: none;
  }
  .item.dropped .title {
    color: var(--muted);
    text-decoration: line-through;
  }
  .head {
    display: flex;
    flex-wrap: wrap;
    align-items: baseline;
    gap: var(--space-2) var(--space-3);
  }
  .status.planned,
  .status.gone {
    background: var(--panel-2);
    color: var(--muted);
  }
  .body {
    display: flex;
    gap: var(--space-3);
    margin-top: var(--space-2);
  }
  .result {
    flex: none;
    width: 140px;
    aspect-ratio: 4 / 3;
  }
  .result img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
  }
  .details {
    flex: 1;
    min-width: 0;
  }
  .small {
    font-size: var(--t-sm);
    margin: 0 0 var(--space-1);
  }
  .warn {
    color: var(--warn);
  }
  .attempts {
    list-style: none;
    margin: 0 0 var(--space-2);
    padding: 0;
    display: flex;
    flex-direction: column;
    gap: var(--space-1);
    font-family: var(--font-mono);
    font-size: var(--t-sm);
  }
  .attempts li {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: var(--space-2);
  }
  .text {
    white-space: pre-wrap;
    margin: 0;
  }
  textarea {
    width: 100%;
    font: inherit;
    background: var(--panel-2);
    color: var(--ink);
  }
  .actions {
    display: flex;
    gap: var(--space-2);
    margin-top: var(--space-1);
  }
  .error {
    color: var(--bad);
    margin: var(--space-1) 0 0;
  }
</style>
