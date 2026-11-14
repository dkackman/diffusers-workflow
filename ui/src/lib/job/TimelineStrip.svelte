<script lang="ts">
  import CopyButton from '../CopyButton.svelte'
  import DownloadLink from '../DownloadLink.svelte'
  import { label, shares, type TimelineSummary } from '../timeline'

  // One row per track, each clip a block whose width is its share of the
  // track and each gap a hatched block. The copy button hands back the
  // clip's `output:` reference, which is what chain_take/chain_output take
  let {
    summary,
    memberHref = () => null,
  }: {
    summary: TimelineSummary
    /** A download URL for a bundle member that is one of the job's files */
    memberHref?: (member: string, source: string | null) => string | null
  } = $props()

  // The `output:` reference a clip recorded for the member it plays, if any
  const sourceOf = (member: string) =>
    summary.tracks
      .flatMap((track) => track.items)
      .find((item) => item.media === member && item.source)?.source ?? null

  // Video above audio, whatever order the bundle lists them in
  const tracks = $derived(
    [...summary.tracks].sort(
      (a, b) => Number(a.kind === 'Audio') - Number(b.kind === 'Audio'),
    ),
  )
  const downloads = $derived(
    summary.members.flatMap((member) => {
      const href = memberHref(member, sourceOf(member))
      return href ? [{ member, href }] : []
    }),
  )
</script>

<div class="timeline" role="group" aria-label="Timeline {summary.name}">
  <div class="scroll">
    {#each tracks as track (track.kind + track.name)}
      {@const widths = shares(track)}
      <div class="track" data-kind={track.kind}>
        <span class="kind muted">{track.kind}</span>
        <div class="row">
          {#each track.items as item, index (index)}
            {#if item.type === 'Gap'}
              <div
                class="block gap"
                style:flex-basis="{widths[index] * 100}%"
                title="gap · {Number((item.seconds ?? 0).toFixed(1))} s"
              ></div>
            {:else}
              <div class="block clip" style:flex-basis="{widths[index] * 100}%">
                <span class="name" title={label(item)}>{label(item)}</span>
                {#if item.source}
                  <CopyButton text={item.source} title="copy {item.source}" />
                {/if}
              </div>
            {/if}
          {/each}
        </div>
      </div>
    {/each}
  </div>
  {#if downloads.length}
    <ul class="members">
      {#each downloads as { member, href } (member)}
        <li>
          <code>{member}</code>
          <DownloadLink {href} label="Download {member}" />
        </li>
      {/each}
    </ul>
  {/if}
</div>

<style>
  .timeline {
    flex-basis: 100%;
    min-width: 0;
    margin-top: var(--space-2);
  }
  .scroll {
    overflow-x: auto;
    padding-bottom: var(--space-1);
  }
  .track {
    display: flex;
    align-items: stretch;
    gap: var(--space-2);
    margin-bottom: var(--space-1);
    min-width: max-content;
  }
  .kind {
    width: 3.2rem;
    flex: none;
    font-size: var(--t-xs);
    align-self: center;
  }
  .row {
    display: flex;
    flex: 1;
    gap: 2px;
    /* Every block holds a readable width; a long timeline scrolls */
    min-width: 100%;
  }
  .block {
    flex-grow: 1;
    flex-shrink: 0;
    min-width: 48px;
    min-height: 2rem;
    box-sizing: border-box;
    border-radius: var(--radius-frame);
  }
  .clip {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-1);
    padding: 0 var(--space-1) 0 var(--space-2);
    background: var(--panel-2);
    border: 1px solid var(--line);
    font-size: var(--t-xs);
  }
  .name {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .gap {
    border: 1px dashed var(--line);
    background: repeating-linear-gradient(
      135deg,
      transparent 0 5px,
      var(--line) 5px 6px
    );
  }
  .members {
    list-style: none;
    margin: var(--space-2) 0 0;
    padding: 0;
    font-size: var(--t-sm);
  }
  .members li {
    display: flex;
    align-items: center;
    gap: var(--space-2);
  }
</style>
