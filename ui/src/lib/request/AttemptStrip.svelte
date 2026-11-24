<script lang="ts">
  import { Check, TriangleAlert } from '@lucide/svelte'
  import StatusChip from '../ui/StatusChip.svelte'
  import ItemProof from './ItemProof.svelte'
  import { takesOf, type Take } from './itemState'
  import { COMPARE_MAX } from '../compare'
  import { formatSpent } from '../format'
  import { hasPicture, jobMedia, type MediaFile } from '../media'
  import { openInHash } from '../mediaViewer'
  import { compareHref, jobHref } from '../routes'
  import type { LedgerItem } from '../types'
  import MediaViewer from '../ui/MediaViewer.svelte'
  import RecipePanel from '../ui/RecipePanel.svelte'

  // An item's takes - every job linked to it - oldest to newest, as
  // pictures: which one is the result, and whether the user chose it or it
  // is only the latest. With `onaccept`, each good take has the button
  // that makes it the result - or, pressed again on the chosen one, clears
  // the choice. With `onnote`, each take has "Note this take", which adds
  // it to the item's new note for the agent
  let {
    item,
    workspace,
    onaccept,
    onnote,
  }: {
    item: LedgerItem
    workspace: string
    /** Accept this take, or with null clear the one accepted. Left out
     * where the plan would not keep it: a dropped item */
    onaccept?: (take: Take, job: string | null) => void
    /** Add this take to the new note. Left out for a dropped item */
    onnote?: (take: Take) => void
  } = $props()

  const takes = $derived(takesOf(item))
  const chosen = (take: Take) =>
    take.mark === 'accepted' || take.mark === 'accepted-gone'
  const MARK = {
    accepted: 'accepted',
    'accepted-gone': 'accepted, files gone',
    latest: 'latest take',
  }

  // The takes that have a file, oldest to newest: what the viewer steps
  // through. A take's output name is the manifest's own, relative to the
  // output root, and two runs of a workflow can write the same one - so the
  // viewer tells takes apart by `item/job/name`, the key `open=` carries in
  // the hash (its leaf is still the file's): one job linked to two items
  // opens in its own strip. Each file is read through its job
  const files = $derived(
    takes.flatMap((take) => {
      const out = take.attempt.output
      if (!out) return []
      const file = jobMedia(out.name, out.kind, take.attempt.job, workspace)
      return [{ take, file, key: viewerKey(take.attempt.job, out.name) }]
    }),
  )
  const items = $derived(files.map((f) => ({ ...f.file, name: f.key })))
  const fileOf = (shown: MediaFile) => files.find((f) => f.key === shown.name)
  const viewerKey = (job: string, name: string) => `${item.key}/${job}/${name}`
  // A name in the hash that is another item's names no file here, and
  // opens nothing
  let open = $state<string | null>(openInHash())
  const openTake = (job: string, name: string) => () =>
    (open = viewerKey(job, name))
  // Two or more takes with a picture or clip: the newest few side by side.
  // The compare route addresses files by name: each name once, the newest
  // take's place
  const pictured = $derived(files.filter((f) => hasPicture(f.file.kind)))
  const compared = $derived.by(() => {
    const names = pictured.map((f) => f.file.name)
    return names
      .filter((n, i) => names.lastIndexOf(n) === i)
      .slice(-COMPARE_MAX)
  })
  const comparable = $derived(compared.length >= 2)

  // The newest take is the one being looked for: the strip starts at its end
  let strip = $state<HTMLElement>()
  $effect(() => {
    void takes.length
    if (strip) strip.scrollLeft = strip.scrollWidth
  })
</script>

{#if item.accepted_missing}
  <p class="gone">
    <TriangleAlert size={14} aria-hidden="true" />
    The accepted take's files are gone. Showing the latest take instead.
    {#if onaccept}Accept another to settle it.{/if}
  </p>
{/if}
{#if comparable}
  <p class="head">
    <a class="quiet compare" href={compareHref(workspace, compared)}>
      Compare takes
    </a>
  </p>
{/if}
<ul class="strip" bind:this={strip} aria-label="Takes of {item.title}">
  {#each takes as take (take.attempt.job)}
    {@const attempt = take.attempt}
    <li
      class="tile"
      class:result={take.mark === 'accepted' || take.mark === 'latest'}
    >
      <ItemProof
        output={attempt.output}
        state={attempt.status}
        {workspace}
        href={jobHref(attempt.job, workspace)}
        onopen={attempt.output
          ? openTake(attempt.job, attempt.output.name)
          : undefined}
        alt="take {take.n} of {item.title}"
        size="take"
        picked={take.mark === 'accepted'}
      />
      <span class="caption">
        <a
          class="n plain"
          href={jobHref(attempt.job, workspace)}
          title="open the job"
        >
          take {take.n}
        </a>
        {#if attempt.status !== 'succeeded'}
          <StatusChip status={attempt.status} />
        {/if}
        {#if attempt.minutes != null}
          <span class="num muted">{formatSpent(attempt.minutes)}</span>
        {/if}
        <!-- On the caption line, so every take's buttons sit at one height -->
        {#if take.mark}
          <span class="mark" class:muted={take.mark === 'accepted-gone'}>
            {#if take.mark === 'accepted'}<Check
                size={12}
                aria-hidden="true"
              />{/if}
            {MARK[take.mark]}
          </span>
        {/if}
      </span>
      {#if attempt.reason && (attempt.status === 'failed' || attempt.status === 'cancelled')}
        <span class="reason bad" title={attempt.reason}>{attempt.reason}</span>
      {/if}
      {#if attempt.note}
        <span class="note muted" title={attempt.note}>{attempt.note}</span>
      {/if}
      {#if onaccept && attempt.status === 'succeeded'}
        <button
          type="button"
          class="accept"
          class:chosen={chosen(take)}
          aria-pressed={chosen(take)}
          aria-label="Accept take {take.n} for {item.title}"
          title={chosen(take)
            ? 'Accepted as the result. Press to clear the choice'
            : 'Make this take the result'}
          onclick={() => onaccept(take, chosen(take) ? null : attempt.job)}
        >
          {#if chosen(take)}<Check size={12} aria-hidden="true" />{/if}{chosen(
            take,
          )
            ? 'Accepted'
            : 'Accept'}
        </button>
      {/if}
      {#if onnote}
        <button
          type="button"
          class="bare notetake"
          aria-label="Note take {take.n} of {item.title} for the agent"
          onclick={() => onnote(take)}
        >
          Note this take
        </button>
      {/if}
    </li>
  {/each}
</ul>

{#snippet actions(file: MediaFile)}
  {@const take = fileOf(file)?.take}
  {#if take}
    {#if onaccept && take.attempt.status === 'succeeded'}
      <button
        type="button"
        aria-pressed={chosen(take)}
        onclick={() => onaccept(take, chosen(take) ? null : take.attempt.job)}
      >
        {#if chosen(take)}<Check size={12} aria-hidden="true" />{/if}{chosen(
          take,
        )
          ? 'Accepted'
          : 'Accept'}
      </button>
    {/if}
    <a class="quiet" href={jobHref(take.attempt.job, workspace)}>
      Open the job
    </a>
  {/if}
{/snippet}

{#snippet aside(file: MediaFile)}
  {#key file.name}
    <RecipePanel
      name={fileOf(file)?.file.name ?? file.name}
      source="output"
      workspace={file.workspace}
    />
  {/key}
{/snippet}

<MediaViewer {items} bind:open {actions} {aside} />

<style>
  .head {
    margin: 0 0 var(--space-2);
    font-size: var(--t-sm);
  }
  .strip {
    list-style: none;
    margin: 0;
    padding: 0 0 var(--space-1);
    display: flex;
    gap: var(--space-3);
    overflow-x: auto;
  }
  .tile {
    flex: none;
    width: 240px;
    display: flex;
    flex-direction: column;
    gap: 0.15rem;
    font-size: var(--t-xs);
  }
  .caption {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.15rem var(--space-2);
  }
  .n {
    font-weight: 600;
  }
  .mark {
    display: inline-flex;
    align-items: center;
    gap: 0.2rem;
    font-weight: 600;
  }
  .reason,
  .note {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  /* The strip's tiles are one height: the buttons go to the bottom, so
     they line up whether or not a take carries the result mark */
  .accept {
    margin-top: auto;
    padding: 0.2rem 0.5rem;
    font-size: var(--t-xs);
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 0.25rem;
  }
  .notetake {
    align-self: flex-start;
    padding: 0.1rem 0;
    font-size: var(--t-xs);
    text-decoration: underline;
    text-underline-offset: 2px;
  }
  .accept.chosen {
    color: var(--ink);
    border-color: var(--ink);
    font-weight: 600;
  }
  @media (pointer: coarse) {
    .accept,
    .notetake {
      min-height: 44px;
    }
  }
  .gone {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    margin: 0 0 var(--space-2);
    font-size: var(--t-sm);
  }
</style>
