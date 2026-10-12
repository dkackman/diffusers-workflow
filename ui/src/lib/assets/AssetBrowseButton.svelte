<script lang="ts">
  // The browse button beside a free-text field holding an asset: reference:
  // opens the picker, offering the same kind of media the field holds now
  import { Images } from '@lucide/svelte'
  import AssetPicker from './AssetPicker.svelte'
  import { assetNamed } from '../assetlib.svelte'
  import { ASSET, reference, referenceName } from '../references'

  let {
    value,
    onpick,
  }: {
    /** The field's current value; the button shows for an asset: one */
    value: unknown
    /** The chosen asset, as the reference the field takes */
    onpick: (value: string) => void
  } = $props()

  let open = $state(false)
  const current = $derived(referenceName(value, ASSET))
  const kind = $derived(current ? assetNamed(current)?.kind : undefined)
</script>

{#if current !== null}
  <button
    type="button"
    class="quiet icon"
    title="choose from the asset library"
    aria-label="choose from the asset library"
    onclick={() => (open = true)}
  >
    <Images size={14} />
  </button>
  <AssetPicker
    bind:open
    {kind}
    {current}
    onpick={(name) => onpick(reference(ASSET, name))}
  />
{/if}
