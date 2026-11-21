<script lang="ts">
  import { formatDuration } from '../format'
  import type { StepTiming } from '../types'

  // Where one pipeline step's time went: load / encode / denoise / decode /
  // save, each drawn only when measured. Widths are each part's share of the
  // parts' sum - not of the node's event span, which carries overhead the
  // parts leave out. first_pass_s / pass_s / passes sit inside denoise_s, so
  // they are in the title only: stacking them would count them twice.
  let { timing }: { timing: StepTiming } = $props()

  const PARTS = [
    ['load', 'load_s'],
    ['encode', 'encode_s'],
    ['denoise', 'denoise_s'],
    ['decode', 'decode_s'],
    ['save', 'save_s'],
  ] as const

  const parts = $derived(
    PARTS.flatMap(([label, key]) => {
      const value = timing[key]
      return typeof value === 'number' && value >= 0 ? [{ label, value }] : []
    }),
  )
  const total = $derived(parts.reduce((sum, part) => sum + part.value, 0))
  const cached = $derived(timing.load_cached === true)

  const title = $derived.by(() => {
    const lines = parts.map(
      (part) =>
        `${part.label} ${formatDuration(part.value)}${
          part.label === 'load' && cached ? ' (cached)' : ''
        }`,
    )
    if (timing.first_pass_s != null)
      lines.push(`first pass ${formatDuration(timing.first_pass_s)}`)
    if (timing.pass_s != null)
      lines.push(`pass ${formatDuration(timing.pass_s)} (median of 2..n)`)
    if (timing.passes != null) lines.push(`${timing.passes} passes`)
    if (timing.segments > 1) lines.push(`${timing.segments} segments`)
    return lines.join('\n')
  })
</script>

{#if parts.length > 0}
  <span class="timing" {title} data-testid="timing-bar">
    <span class="bar" aria-hidden="true">
      {#each parts as part, i (part.label)}
        <span
          class="seg s{i % 5}"
          data-part={part.label}
          style:width={`${total > 0 ? (part.value / total) * 100 : 100 / parts.length}%`}
        ></span>
      {/each}
    </span>
    <span class="figs">
      {parts.map((p) => `${p.label} ${formatDuration(p.value)}`).join(' · ')}
    </span>
    {#if cached}<span class="cached">cached</span>{/if}
  </span>
{/if}

<style>
  .timing {
    display: inline-flex;
    align-items: center;
    gap: var(--space-1);
    margin-left: var(--space-2);
    font-family: var(--font-mono);
    font-weight: normal;
  }
  .bar {
    display: inline-flex;
    width: 6rem;
    height: 0.5rem;
    overflow: hidden;
    border: 1px solid color-mix(in srgb, var(--ink) 35%, transparent);
  }
  .seg {
    display: block;
    height: 100%;
    background: var(--ink);
  }
  .s0 {
    opacity: 0.15;
  }
  .s1 {
    opacity: 0.3;
  }
  .s2 {
    opacity: 0.55;
  }
  .s3 {
    opacity: 0.4;
  }
  .s4 {
    opacity: 0.8;
  }
  .figs,
  .cached {
    font-size: 0.75em;
  }
</style>
