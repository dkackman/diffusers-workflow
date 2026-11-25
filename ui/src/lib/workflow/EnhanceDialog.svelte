<script lang="ts">
  // Enhance a prompt from where it is typed: the editor's panel in a dialog,
  // over an Enhancer of its own. Presets and cached models load when it
  // first opens, and the idea is the field's text each time it opens.
  import { onDestroy, untrack } from 'svelte'
  import EnhancePanel from '../editor/EnhancePanel.svelte'
  import { Enhancer } from '../enhancer.svelte'
  import Modal from '../ui/Modal.svelte'

  let {
    open = $bindable(false),
    idea,
    onuse,
  }: {
    open?: boolean
    /** The field's text, taken when the dialog opens */
    idea: string
    /** The enhanced prompt, once it is taken; the dialog then closes */
    onuse: (text: string) => void
  } = $props()

  const enhancer = new Enhancer()
  let loaded = false

  $effect(() => {
    if (!open) return
    untrack(() => {
      enhancer.idea = idea
      if (loaded) return
      loaded = true
      enhancer.refreshModels()
      enhancer.load().then(() => enhancer.preselect(undefined))
    })
  })

  onDestroy(() => enhancer.stop())

  // Close stops the server job as well as the page's following of it
  $effect(() => {
    if (open) return
    untrack(() => {
      if (enhancer.busy) void enhancer.cancel()
    })
  })

  function use() {
    onuse(enhancer.takeResult().text)
    open = false
  }
</script>

<Modal bind:open label="Enhance the prompt" wide>
  <EnhancePanel {enhancer} onuse={use} />
</Modal>
