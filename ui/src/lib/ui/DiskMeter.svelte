<script lang="ts">
  import { HardDrive } from '@lucide/svelte'
  import { gbFromBytes } from '../format'

  // The room left on a volume: a bar for the share used and the words
  // "x / y GB free" (just "x GB free" when the volume's size is unknown,
  // and then no bar, since a share needs a total)
  let {
    freeBytes,
    totalBytes,
    title,
    icon = false,
  }: {
    freeBytes: number
    totalBytes?: number | null
    title?: string
    icon?: boolean
  } = $props()

  const usedPct = $derived(
    totalBytes
      ? Math.round((100 * (totalBytes - freeBytes)) / totalBytes)
      : null,
  )
</script>

<span class="disk" {title}>
  {#if icon}<HardDrive size={14} />{/if}
  {#if usedPct != null}
    <span class="diskbar"><span style="width: {usedPct}%"></span></span>
  {/if}
  {#if totalBytes}
    {gbFromBytes(freeBytes)} / {gbFromBytes(totalBytes)} GB free
  {:else}
    {gbFromBytes(freeBytes)} GB free
  {/if}
</span>

<style>
  .disk {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    color: var(--muted);
    font-size: 0.85rem;
    white-space: nowrap;
  }
  .diskbar {
    width: 90px;
    height: 6px;
    border-radius: 3px;
    background: var(--line);
    overflow: hidden;
    display: inline-block;
  }
  .diskbar > span {
    display: block;
    height: 100%;
    background: var(--muted);
    opacity: 0.7;
  }
</style>
