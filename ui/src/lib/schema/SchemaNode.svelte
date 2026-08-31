<script lang="ts" module>
  export type SchemaValue = Record<string, any>

  export function resolveRef(root: SchemaValue, ref: string): SchemaValue {
    let node: any = root
    for (const part of ref.replace(/^#\//, '').split('/')) node = node?.[part]
    return node ?? {}
  }

  export function refName(ref: string): string {
    return ref.split('/').pop() ?? ref
  }

  export function typeLabel(node: SchemaValue, root: SchemaValue): string {
    if (node.__constraint) return ''
    if (node.$ref) return refName(node.$ref)
    if (node.enum) return 'enum'
    if (node.oneOf) return `one of ${node.oneOf.length}`
    if (node.type === 'array') {
      const items = node.items
      return items ? `array of ${typeLabel(items, root)}` : 'array'
    }
    if (Array.isArray(node.type)) return node.type.join(' | ')
    return node.type ?? (node.properties ? 'object' : 'any')
  }
</script>

<script lang="ts">
  import { ChevronDown, ChevronRight } from '@lucide/svelte'
  import Self from './SchemaNode.svelte'

  let {
    name,
    node,
    root,
    required = false,
    open = false,
    seenRefs = [],
  }: {
    name: string
    node: SchemaValue
    root: SchemaValue
    required?: boolean
    open?: boolean
    seenRefs?: string[]
  } = $props()

  // The prop only seeds the initial state - expansion is then local
  // svelte-ignore state_referenced_locally
  let expanded = $state(open)

  const circular = $derived(!!node.$ref && seenRefs.includes(node.$ref))
  const effective = $derived(
    node.$ref && !circular ? resolveRef(root, node.$ref) : node,
  )
  const childRefs = $derived(node.$ref ? [...seenRefs, node.$ref] : seenRefs)

  type Child = { name: string; node: SchemaValue; required: boolean }
  const children = $derived.by(() => {
    const out: Child[] = []
    const requiredNames: string[] = effective.required ?? []
    for (const [key, child] of Object.entries(effective.properties ?? {})) {
      out.push({
        name: key,
        node: child as SchemaValue,
        required: requiredNames.includes(key),
      })
    }
    if (effective.type === 'array' && effective.items)
      out.push({ name: '(items)', node: effective.items, required: false })
    effective.oneOf?.forEach((option: SchemaValue, index: number) => {
      // A bare required-list option is a shape constraint, not a subtype -
      // say what it requires instead of an opaque "option N"
      const keys = Object.keys(option)
      const constraint = keys.length === 1 && keys[0] === 'required'
      out.push({
        name: constraint
          ? `requires ${option.required.join(' + ')}`
          : `option ${index + 1}`,
        node: constraint ? { __constraint: true } : option,
        required: false,
      })
    })
    if (typeof effective.additionalProperties === 'object')
      out.push({
        name: '(additional properties)',
        node: effective.additionalProperties,
        required: false,
      })
    return out
  })

  const expandable = $derived(!circular && children.length > 0)
  const description = $derived(node.description ?? effective.description)

  // Constraints worth surfacing as chips, mirroring what the schema uses
  const facts = $derived.by(() => {
    const out: string[] = []
    if (node.default !== undefined)
      out.push(`default: ${JSON.stringify(node.default)}`)
    if (effective.minimum !== undefined) out.push(`min: ${effective.minimum}`)
    if (effective.maximum !== undefined) out.push(`max: ${effective.maximum}`)
    if (effective.minItems !== undefined)
      out.push(`at least ${effective.minItems}`)
    if (effective.format) out.push(effective.format)
    if (effective.additionalProperties === false) out.push('no extra keys')
    if (effective.additionalProperties === true) out.push('free-form')
    return out
  })

  function jumpToDef(event: MouseEvent, ref: string) {
    event.stopPropagation()
    document
      .getElementById('def-' + refName(ref))
      ?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }
</script>

<div class="node">
  <button
    class="row"
    class:expandable
    onclick={() => (expanded = expandable && !expanded)}
    disabled={!expandable}
  >
    <span class="caret">
      {#if expandable}
        {#if expanded}<ChevronDown size={13} />{:else}<ChevronRight
            size={13}
          />{/if}
      {/if}
    </span>
    <span class="name">{name}</span>
    {#if required}<span class="req">*</span>{/if}
    {#if node.$ref}
      <span
        class="type reflink"
        role="link"
        tabindex="-1"
        title="jump to definition"
        onclick={(e) => jumpToDef(e, node.$ref)}
        onkeydown={() => {}}
      >
        {typeLabel(node, root)}
      </span>
    {:else}
      <span class="type">{typeLabel(node, root)}</span>
    {/if}
    {#if circular}<span class="muted circ">recursive</span>{/if}
    {#each facts as fact (fact)}<span class="fact">{fact}</span>{/each}
  </button>
  {#if description}
    <div class="desc muted">{description}</div>
  {/if}
  {#if node.enum ?? effective.enum}
    <div class="enums">
      {#each node.enum ?? effective.enum as value (value)}
        <code>{JSON.stringify(value)}</code>
      {/each}
    </div>
  {/if}
  {#if expanded}
    <div class="children">
      {#each children as child (child.name)}
        <Self
          name={child.name}
          node={child.node}
          {root}
          required={child.required}
          seenRefs={childRefs}
        />
      {/each}
    </div>
  {/if}
</div>

<style>
  .node {
    font-size: 0.88rem;
  }
  .row {
    display: flex;
    align-items: baseline;
    flex-wrap: wrap;
    gap: 0.45rem;
    width: 100%;
    text-align: left;
    background: none;
    border: 0;
    padding: 0.15rem 0;
    color: var(--ink);
    cursor: default;
    font-size: inherit;
  }
  .row.expandable {
    cursor: pointer;
  }
  .row:disabled {
    opacity: 1;
  }
  .caret {
    width: 13px;
    display: inline-flex;
    align-self: center;
    color: var(--muted);
    flex-shrink: 0;
  }
  .name {
    font-family: var(--mono, ui-monospace, monospace);
    font-weight: 600;
  }
  .req {
    color: var(--warn);
  }
  .type {
    color: var(--accent);
    font-size: 0.78rem;
  }
  .reflink {
    text-decoration: underline dotted;
    cursor: pointer;
  }
  .circ {
    font-size: 0.75rem;
    font-style: italic;
  }
  .fact {
    font-size: 0.72rem;
    color: var(--muted);
    border: 1px solid var(--line);
    border-radius: 4px;
    padding: 0 0.3rem;
  }
  .desc {
    font-size: 0.78rem;
    margin: 0 0 0.15rem 1.15rem;
    max-width: 75ch;
  }
  .enums {
    display: flex;
    flex-wrap: wrap;
    gap: 0.3rem;
    margin: 0.1rem 0 0.2rem 1.15rem;
  }
  .enums code {
    font-size: 0.75rem;
    border: 1px solid var(--line);
    border-radius: 4px;
    padding: 0 0.3rem;
    background: var(--panel-2);
  }
  .children {
    margin-left: 0.55rem;
    padding-left: 0.75rem;
    border-left: 1px solid var(--line);
  }
  /* The indent compounds per nesting level; a deep schema would walk itself
     off a narrow screen */
  @media (max-width: 640px) {
    .children {
      margin-left: 0;
      padding-left: 0.5rem;
    }
  }
</style>
