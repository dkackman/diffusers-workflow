<!--
  @component
  A time, the one way the UI shows one (GLOSSARY.md, *Formats*): relative in
  a list ('changed 5 min ago'), so a person sees at a glance what is recent
  and what is stale, and exact in a detail ('Changed Oct 26, 2:03 PM'). The
  hover gives it to the second (formatExact) and `datetime` gives it to a
  machine. A relative time ages on the shared clock (src/lib/clock.svelte.ts),
  so it never needs a reload to stay true.

  `mode: 'day'` is the date alone, for a column where the time of day says
  nothing (a model's last use). `title` replaces the exact-time hover where
  the time needs explaining instead (what "last used" counts).
-->
<script lang="ts">
  import { now } from '../clock.svelte'
  import {
    formatDay,
    formatExact,
    formatMtime,
    formatRelative,
  } from '../format'

  let {
    at,
    mode = 'relative',
    by = null,
    label = null,
    title,
  }: {
    /** Unix seconds */
    at: number
    mode?: 'relative' | 'exact' | 'day'
    /** Who made the change, where the server knows it was the agent */
    by?: 'agent' | null
    label?: 'changed' | 'created' | null
    title?: string
  } = $props()

  const when = $derived(
    mode === 'relative'
      ? formatRelative(at, now.value)
      : mode === 'day'
        ? formatDay(at)
        : formatMtime(at),
  )
  // A list's label leads lower case ('changed 5 min ago'); a detail's is
  // the start of its line ('Created Oct 26, 2:03 PM')
  const lead = $derived(
    label === null
      ? ''
      : (mode === 'relative'
          ? label
          : label[0].toUpperCase() + label.slice(1)) + ' ',
  )
  const text = $derived(
    `${lead}${when}${by === 'agent' ? ' by the agent' : ''}`,
  )
</script>

<time
  datetime={new Date(at * 1000).toISOString()}
  title={title ?? formatExact(at)}
>
  {text}
</time>
