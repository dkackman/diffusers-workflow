<script lang="ts" generics="T">
  // What a resource() shows, so loading never looks like empty. The first
  // that applies: loading with nothing held (a skeleton); an error with
  // nothing held (the failure and Retry); filtered to nothing (Clear
  // filters); nothing at all (Empty); otherwise the content, under a line
  // saying a refresh failed when the data held is the last good read (not
  // for an outage or a lock, which the connection banner says).
  // `quiet` is a secondary panel the page reads fine without: it shows
  // nothing while loading or failed, as the offline banner already speaks.
  // The one place Empty.svelte is imported (eslint.config.js)
  import type { Snippet } from 'svelte'
  import { describeError } from '../apiErrors'
  import type { Resource } from '../resource.svelte'
  import Empty from '../Empty.svelte'
  import Skeleton from './Skeleton.svelte'

  let {
    of,
    label,
    what,
    quiet = false,
    empty = false,
    filtered = false,
    onclear,
    emptyIcon,
    emptyText,
    skeleton,
    children,
  }: {
    of: Resource<T>
    /** What is loaded, as in "Couldn't load <label>" */
    label: string
    /** What a 404 is about, as in "<what> wasn't found"; "That" by default */
    what?: string
    /** A secondary panel: no skeleton, no failure panel, no refresh line */
    quiet?: boolean
    empty?: boolean
    filtered?: boolean
    onclear?: () => void
    emptyIcon?: Snippet
    emptyText?: Snippet
    skeleton?: Snippet
    /** None when the content sits outside, beside the states this shows */
    children?: Snippet<[T]>
  } = $props()

  const held = $derived(of.data !== undefined)
  const described = $derived(
    of.error === undefined ? null : describeError(of.error, what),
  )
  // Out of reach or locked is the connection banner's to say, once, with
  // its own Retry now / Enter token: the line would say it twice
  const bannerSays = $derived(
    described?.kind === 'offline' || described?.kind === 'locked',
  )
  const retry = () => void of.reload()
</script>

{#snippet nothing()}{/snippet}

{#if quiet && !held && of.status !== 'ready'}
  <!-- nothing: loading or failed, the page reads fine without it -->
{:else if of.status === 'loading' && !held}
  {#if skeleton}{@render skeleton()}{:else}<Skeleton
      label="loading {label}"
    />{/if}
{:else if of.status === 'error' && !held}
  <div class="panel loadstate-failed" role="alert">
    <p>Couldn't load {label}: {described?.title}</p>
    {#if described?.remedy}<p>{described.remedy}</p>{/if}
    <button onclick={retry}>Retry</button>
  </div>
{:else if filtered && empty}
  <div class="filtered muted">
    <p>Nothing matches those filters.</p>
    <button class="quiet" onclick={() => onclear?.()}>Clear filters</button>
  </div>
{:else if empty}
  <Empty icon={emptyIcon ?? nothing}>
    {#if emptyText}{@render emptyText()}{/if}
  </Empty>
{:else}
  {#if !quiet && held && described && !bannerSays && (of.stale || of.status === 'error')}
    <div class="refresh-failed" role="alert">
      <span>Couldn't refresh {label}: {described.title}</span>
      <button class="quiet" onclick={retry}>Retry</button>
    </div>
  {/if}
  {@render children?.(of.data as T)}
{/if}

<style>
  .loadstate-failed {
    margin: 1rem 0;
  }
  .loadstate-failed p {
    margin: 0 0 0.6rem;
    overflow-wrap: anywhere;
  }
  .filtered {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: var(--space-2);
    padding: var(--space-5) 0;
    text-align: center;
  }
  .filtered p {
    margin: 0;
  }
  .refresh-failed {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: var(--space-2);
    margin-bottom: var(--space-2);
    overflow-wrap: anywhere;
  }
</style>
