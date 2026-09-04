<script lang="ts">
  import { Download } from '@lucide/svelte'

  let {
    href,
    label = 'Download',
  }: {
    href: string
    label?: string
  } = $props()
</script>

<a class="quiet icon" {href} download aria-label={label} title={label}>
  <Download size={14} />
</a>
