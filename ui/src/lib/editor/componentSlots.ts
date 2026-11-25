import type { PipelineDescription } from '../types'

// Python types a constructor takes that are values, not components
const VALUE_TYPES = new Set([
  'bool',
  'int',
  'float',
  'str',
  'bytes',
  'dict',
  'list',
  'tuple',
  'set',
  'none',
  'nonetype',
  'any',
  'typing.any',
  'callable',
])

// `<class 'a.B'>` -> `a.B`; `typing.Optional[X]` and `X | None` -> X; a
// docstring's "[`X`]" -> X; `list[str]` -> list (its head names the kind)
function typeNames(text: string): string[] {
  const cls = /^<class '([^']+)'>$/.exec(text.trim())
  if (cls) return [cls[1]]
  if (text.trim().startsWith('<')) return []
  return text
    .replace(/^(typing\.)?(Optional|Union)\[(.*)\]$/, '$3')
    .split(/[|,]/)
    .map((part) =>
      part
        .replace(/[`*\s]/g, '')
        .replace(/^\[(.*)\]$/, '$1')
        .replace(/\[.*$/, '')
        .replace(/^typing\./, ''),
    )
    .filter(Boolean)
}

function namesClass(text: string | null | undefined): boolean {
  if (!text) return false
  return typeNames(text).some(
    (name) => !VALUE_TYPES.has(name.toLowerCase().replace(/^builtins\./, '')),
  )
}

// An annotation wins; the docstring's type stands in when there is none.
// A doc type reads "[`X`], *optional*", so only its first clause counts
function isComponent(p: PipelineDescription['parameters'][number]): boolean {
  if (p.annotation) return namesClass(p.annotation)
  return namesClass(p.doc_type?.split(/,(?![^[]*\])/)[0])
}

/**
 * The component slots a pipeline class takes: its `__init__` parameters that
 * name a class. Null when there is no description or the class takes
 * **kwargs (a modular pipeline), so the caller falls back to the schema's list.
 */
export function slotsFrom(
  description: PipelineDescription | null,
): string[] | null {
  if (!description || description.accepts_kwargs) return null
  return description.parameters.filter(isComponent).map((p) => p.name)
}
