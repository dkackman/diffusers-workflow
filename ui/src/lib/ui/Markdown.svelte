<script lang="ts">
  // Renders parseMarkdown's tree as Svelte nodes - never as HTML - so a note
  // that contains markup shows that markup as text. Headings are styled
  // paragraphs with a heading role two levels below the page's own, so a
  // note's "# RULE" does not compete with the page title in the outline
  import { parseMarkdown, type Inline } from '../markdown'

  let { source }: { source: string } = $props()
  const blocks = $derived(parseMarkdown(source))
</script>

{#snippet inline(nodes: Inline[])}
  {#each nodes as node, i (i)}
    {#if node.kind === 'text'}{node.text}{:else if node.kind === 'br'}<br
      />{:else if node.kind === 'code'}<code>{node.text}</code
      >{:else if node.kind === 'strong'}<strong
        >{@render inline(node.children)}</strong
      >{:else if node.kind === 'em'}<em>{@render inline(node.children)}</em
      >{:else}<a href={node.href} target="_blank" rel="noopener noreferrer"
        >{@render inline(node.children)}</a
      >{/if}
  {/each}
{/snippet}

<div class="md">
  {#each blocks as block, i (i)}
    {#if block.kind === 'heading'}
      <p class="h h{block.level}" role="heading" aria-level={block.level + 2}>
        {@render inline(block.children)}
      </p>
    {:else if block.kind === 'paragraph'}
      <p>{@render inline(block.children)}</p>
    {:else if block.kind === 'list' && block.ordered}
      <ol>
        {#each block.items as item, j (j)}<li>{@render inline(item)}</li>{/each}
      </ol>
    {:else if block.kind === 'list'}
      <ul>
        {#each block.items as item, j (j)}<li>{@render inline(item)}</li>{/each}
      </ul>
    {:else}
      <pre><code>{block.text}</code></pre>
    {/if}
  {/each}
</div>

<style>
  .md > :first-child {
    margin-top: 0;
  }
  .md > :last-child {
    margin-bottom: 0;
  }
  p,
  ul,
  ol,
  pre {
    margin: 0 0 0.5em;
  }
  ul,
  ol {
    padding-left: 1.4em;
  }
  .h {
    font-weight: 600;
    margin-top: 0.75em;
  }
  .h1 {
    font-size: 1.1em;
  }
  pre {
    background: var(--panel-2);
    padding: var(--space-2);
    border-radius: var(--radius-2);
    overflow-x: auto;
    white-space: pre;
  }
  code {
    font-family: var(--font-mono);
    font-size: 0.92em;
  }
</style>
