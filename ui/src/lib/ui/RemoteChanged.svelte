<script lang="ts">
  // Above an editor whose document changed on the server under unsaved
  // edits (DocumentEditor.watchRemote): say so, and offer both ways on.
  // Not a dialog - the edits stay in reach, and Keep mine leaves the
  // conflict to Save's stale_write question
  let {
    label,
    ontheirs,
    onmine,
  }: {
    label: string
    ontheirs: () => void
    onmine: () => void
  } = $props()
</script>

<div class="panel remotechanged" role="status">
  <p>{label} changed since you opened it, in another tab or by an agent</p>
  <div class="actions">
    <button onclick={ontheirs}>Load theirs</button>
    <button class="quiet" onclick={onmine}>Keep mine</button>
  </div>
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
