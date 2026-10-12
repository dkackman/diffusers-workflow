<script lang="ts">
  // The browse button beside a free-text field holding an asset: or
  // output: reference: opens the picker on that kind of source, offering
  // the same kind of media the field holds now
  import { Images } from '@lucide/svelte'
  import MediaPicker from './MediaPicker.svelte'
  import { assetNamed } from '../assetlib.svelte'
  import { outputNamed } from '../outputlib.svelte'
  import { ASSET, OUTPUT, referenceName } from '../references'

  let {
    value,
    onpick,
  }: {
    /** The field's current value; the button shows for a file reference */
    value: unknown
    /** The chosen file, as the reference the field takes */
    onpick: (value: string) => void
  } = $props()

  let open = $state(false)
  const asset = $derived(referenceName(value, ASSET))
  const output = $derived(referenceName(value, OUTPUT))
  const kind = $derived(
    asset !== null
      ? assetNamed(asset)?.kind
      : output !== null
        ? outputNamed(output)?.kind
        : undefined,
  )
</script>

{#if asset !== null || output !== null}
  <button
    type="button"
    class="quiet icon"
    title="choose another file"
    aria-label="choose another file"
    onclick={() => (open = true)}
  >
    <Images size={14} />
  </button>
  <MediaPicker bind:open {kind} current={value} {onpick} />
{/if}
