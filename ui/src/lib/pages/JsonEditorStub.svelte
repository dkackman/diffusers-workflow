<script lang="ts">
  // Test stand-in for the Monaco-backed JsonEditor: renders the value as
  // plain text so a test can assert what the page handed the viewer
  // without booting Monaco in jsdom.
  let { value }: { value: string } = $props()
</script>

<pre data-testid="json-editor">{value}</pre>