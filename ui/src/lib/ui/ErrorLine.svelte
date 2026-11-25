<script lang="ts">
  // A failure shown inline: the described title in plain words, what to do
  // about it, and the engine's own text folded behind Details
  import { describeError } from '../apiErrors'
  import Details from './Details.svelte'

  let { error, doing }: { error: unknown; doing?: string } = $props()

  const described = $derived(describeError(error))
</script>

<div class="errorline" role="alert">
  <p class="title">
    {doing ? `Couldn't ${doing}: ${described.title}` : described.title}
  </p>
  {#if described.remedy}<p class="remedy">{described.remedy}</p>{/if}
  <Details summary="Engine detail" detail={described.detail ?? ''} />
</div>

<style>
  .errorline {
    color: var(--bad);
    margin: 0.25rem 0 0;
  }
  p {
    margin: 0;
    overflow-wrap: anywhere;
  }
  .remedy {
    color: inherit;
    opacity: 0.85;
  }
</style>
