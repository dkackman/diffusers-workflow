<script lang="ts">
  // Above an editor whose document changed on the server under unsaved
  // edits (DocumentEditor.watchRemote): say so, and offer both ways on.
  // Not a dialog - the edits stay in reach, and Keep mine leaves the
  // conflict to Save's stale_write question. Keep mine is the filled
  // button: Load theirs throws the unsaved edits away, so it is the quiet
  // one, and turns the danger colour under the pointer. A file deleted on
  // the server has no theirs to load: the banner says so, and Save (which
  // creates it again) or leaving are the ways on
  let {
    label,
    gone = false,
    ontheirs,
    onmine,
  }: {
    label: string
    gone?: boolean
    ontheirs: () => void
    onmine: () => void
  } = $props()

  const GONE =
    'was deleted on the server. Save to write it back, or leave to discard your edits.'
</script>

<div class="panel remotechanged" role="status">
  {#if gone}
    <p>{`${label} ${GONE}`}</p>
  {:else}
    <p>{label} changed since you opened it, in another tab or by an agent</p>
    <div class="actions">
      <button class="quiet danger" onclick={ontheirs}>Load theirs</button>
      <button onclick={onmine}>Keep mine</button>
    </div>
  {/if}
</div>

<style>
  .remotechanged {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.5rem 1rem;
    margin: 0.5rem 0;
    border-color: var(--warn);
  }
  .remotechanged p {
    margin: 0;
    flex: 1 1 20rem;
    overflow-wrap: anywhere;
  }
  .actions {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
  }
</style>
