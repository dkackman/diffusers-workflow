<script lang="ts" module>
  // "last used" is when a dw job that named the repo last started
  // (last_job_use, #1445). A repo no job has used since tracking began falls
  // back to the files' atime, labelled "(disk)" because copies and backups
  // bump it too
  export const LAST_USED_TIP =
    'When a dw job that named this repo last started. A muted "(disk)" date ' +
    "means no dw job has used it since tracking began, and is the files' " +
    'last access time instead'
  const DW_USE_TIP = 'the last time a dw job that named this repo started'
  const DISK_FALLBACK_TIP =
    'No dw job has used this repo since tracking began. ' +
    "Date is the files' last access time, which copies and backups also bump."
</script>

<script lang="ts">
  import { formatDay } from './format'

  let {
    lastJobUse,
    lastAccessed,
    class: extra = '',
  }: {
    lastJobUse: number | null
    lastAccessed: number | null
    class?: string
  } = $props()
</script>

{#if lastJobUse != null}
  <td class="lastused {extra}" title={DW_USE_TIP}>{formatDay(lastJobUse)}</td>
{:else if lastAccessed != null}
  <td class="muted lastused {extra}" title={DISK_FALLBACK_TIP}>
    {formatDay(lastAccessed)} (disk)
  </td>
{:else}
  <td class="muted lastused {extra}">—</td>
{/if}

<style>
  .lastused {
    font-family: var(--font-mono);
    font-variant-numeric: tabular-nums;
    font-size: var(--t-xs);
  }
</style>
