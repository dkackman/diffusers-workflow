<script lang="ts">
  // Test stand-in for the Monaco-backed JsonEditor: renders the value as
  // plain text so a test can assert what the page handed the viewer
  // without booting Monaco in jsdom. Text typed into its field is held, as
  // the real editor's debounce holds it, until the page flushes
  let {
    value,
    onchange,
    register,
  }: {
    value: string
    onchange?: (value: string, options?: { quiet?: boolean }) => void
    register?: (flush: () => void) => () => void
  } = $props()

  let pending: string | null = null

  $effect(() => {
    if (!register) return
    return register(() => {
      if (pending === null) return
      const text = pending
      pending = null
      onchange?.(text)
    })
  })
</script>

<pre data-testid="json-editor">{value}</pre>
<textarea
  data-testid="json-input"
  aria-label="JSON"
  oninput={(e) => (pending = e.currentTarget.value)}></textarea>
