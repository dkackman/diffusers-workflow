import { api } from './api'
import type { PipelineDescription, PipelineParameter } from './types'
import { ITEM, VARIABLE, isReference, reference } from './references'

export { isReference }

/** Introspection descriptions, cached per class+target for the session. */
const descriptions = new Map<string, Promise<PipelineDescription | null>>()

export function classDescription(
  name: string,
  target: 'call' | 'init' | 'load' | 'task' = 'call',
): Promise<PipelineDescription | null> {
  const key = `${target}:${name}`
  let description = descriptions.get(key)
  if (!description) {
    const fetcher =
      target === 'task'
        ? api.describeTask(name)
        : target === 'call'
          ? api.describePipeline(name)
          : api.describeClass(name, target)
    // a failure is not cached: the next mount asks again (#1640)
    description = fetcher.catch(() => {
      descriptions.delete(key)
      return null
    })
    descriptions.set(key, description)
  }
  return description
}

export type Widget = 'number' | 'boolean' | 'text' | 'textarea' | 'json'

export function widgetFor(
  parameter: PipelineParameter | undefined,
  value: unknown,
): Widget {
  if (value !== null && typeof value === 'object') return 'json'
  if (isReference(value)) return 'text'
  if (typeof value === 'boolean') return 'boolean'
  if (typeof value === 'number') return 'number'
  const annotation =
    (parameter?.annotation ?? '') + ' ' + (parameter?.doc_type ?? '')
  if (/\bbool\b/.test(annotation)) return 'boolean'
  if (/\b(int|float)\b/.test(annotation)) return 'number'
  if (typeof value === 'string' && value.length > 60) return 'textarea'
  if (parameter?.name.includes('prompt')) return 'textarea'
  return 'text'
}

/** Typed text as a boolean when it says true or false, in any case. Any
 * other text stays text, so the engine judges its spelling (dw/variables.py
 * takes yes/no/1/0/on/off and refuses the rest) rather than it silently
 * becoming false */
export function booleanOf(raw: string): boolean | string {
  const word = raw.trim().toLowerCase()
  if (word === 'true') return true
  if (word === 'false') return false
  return raw
}

/** Typed text as a number, or null when it is blank or names no finite
 * one: `Number` reads whitespace as 0, and Infinity serialises to null */
export function finiteNumber(raw: string): number | null {
  const n = Number(raw)
  return raw.trim() !== '' && Number.isFinite(n) ? n : null
}

/** Turn what the input produced back into the JSON value the engine expects.
 * Reference strings and unparseable numbers stay strings - the engine
 * substitutes variables after schema validation. */
export function coerce(widget: Widget, raw: string): unknown {
  if (widget === 'number') return finiteNumber(raw) ?? raw
  if (widget === 'boolean') return booleanOf(raw)
  if (widget === 'json') {
    try {
      return JSON.parse(raw)
    } catch {
      return raw
    }
  }
  return raw
}

export const TORCH_DTYPES = ['torch.bfloat16', 'torch.float16', 'torch.float32']

/** What a step's result can be written as. tests/test_ui_twins.py pins it:
 * each one is accepted by dw/content_types.py, and every audio container
 * the engine writes is reachable from one. */
export const CONTENT_TYPES = [
  'image/png',
  'image/jpeg',
  'image/webp',
  'image/gif',
  'video/mp4',
  'audio/wav',
  'audio/flac',
  'audio/aiff',
  'audio/mpeg',
  'audio/ogg',
  'audio/opus',
  'application/json',
  'text/plain',
]

/** A select's options: a value the list does not hold (written by hand, or
 * an alias) is kept as the last option rather than shown as a blank. */
export function optionsWith(
  listed: readonly string[],
  current?: string | null,
): string[] {
  return current && !listed.includes(current)
    ? [...listed, current]
    : [...listed]
}

/** The result select's options; see optionsWith. */
export function contentTypeOptions(current?: string): string[] {
  return optionsWith(CONTENT_TYPES, current)
}

/** Text that deserves a document-scale editing surface: long enough to
 * be truncated by a single-line input, or already multi-line. */
export function isLongText(value: unknown): boolean {
  return (
    typeof value === 'string' && (value.length > 60 || value.includes('\n'))
  )
}

/** One value-to-input-string rule for every editor field. */
export function displayValue(value: unknown, pretty = false): string {
  if (value === null || value === undefined) return ''
  if (typeof value === 'object') {
    return pretty ? JSON.stringify(value, null, 2) : JSON.stringify(value)
  }
  return String(value)
}

export function emptyWorkflow() {
  return {
    id: 'new_workflow',
    variables: { prompt: 'a scenic landscape' },
    steps: [emptyStep()],
  }
}

export function emptyStep() {
  return {
    name: 'generate',
    pipeline: {
      configuration: { component_type: '', offload: 'model' },
      from_pretrained_arguments: {
        model_name: '',
        torch_dtype: 'torch.bfloat16',
      },
      arguments: { prompt: reference(VARIABLE, 'prompt') },
    },
    result: { content_type: 'image/png' },
  }
}

/** Quantization presets, taken from working example workflows. */
export const QUANT_PRESETS: Record<
  string,
  { config_type: string; arguments: Record<string, unknown> } | null
> = {
  none: null,
  'bnb 4-bit nf4': {
    config_type: 'BitsAndBytesConfig',
    arguments: {
      load_in_4bit: true,
      bnb_4bit_quant_type: '{nf4}',
      bnb_4bit_compute_dtype: 'torch.bfloat16',
    },
  },
  'bnb 8-bit': {
    config_type: 'BitsAndBytesConfig',
    arguments: { load_in_8bit: true },
  },
  'torchao int8 weight-only': {
    config_type: 'TorchAoConfig',
    arguments: { quant_type: 'torchao.quantization.Int8WeightOnlyConfig' },
  },
  'sdnq uint4': {
    config_type: 'sdnq.SDNQConfig',
    arguments: { weights_dtype: '{uint4}', use_quantized_matmul: true },
  },
  'gguf (quantized checkpoint)': {
    config_type: 'GGUFQuantizationConfig',
    arguments: { compute_dtype: 'torch.bfloat16' },
  },
}

/** Component slots a pipeline definition can declare, in display order. */
export const COMPONENT_SLOTS = [
  'transformer',
  'transformer_2',
  'unet',
  'vae',
  'text_encoder',
  'text_encoder_2',
  'text_encoder_3',
  'tokenizer',
  'tokenizer_2',
  'tokenizer_3',
  'controlnet',
  'image_encoder',
  'feature_extractor',
  'prompt_enhancer_head',
  'model',
]

/** The pipeline's own option keys that are not component slots: the schema's
 * `$defs/pipeline` properties less COMPONENT_SLOTS, pinned by
 * tests/test_ui_twins.py. */
export const OPTION_KEYS = [
  'configuration',
  'shared_components',
  'reused_components',
  'scheduler',
  'audio_scheduler',
  'loras',
  'ip_adapter',
  'from_pretrained_arguments',
  'remote_text_encoder',
  'seed',
  'chain',
  'arguments',
]

/** A pipeline's offload modes, the schema's `offload` enum. */
export const OFFLOAD_MODES = ['model', 'sequential']

export const CACHE_TYPES = [
  'first_block',
  'faster',
  'mag',
  'taylorseer',
  'text_kv',
]

/** The variable names that feed an attention_backend, as
 * "attention_backend": "variable:X", anywhere in the steps - pipeline
 * configuration, components, and inline sub-workflow definitions. */
export function attentionBackendVariables(definition: unknown): Set<string> {
  const found = new Set<string>()
  const walk = (node: unknown): void => {
    if (Array.isArray(node)) {
      node.forEach(walk)
    } else if (node !== null && typeof node === 'object') {
      for (const [key, value] of Object.entries(node)) {
        if (
          key === 'attention_backend' &&
          typeof value === 'string' &&
          value.startsWith(VARIABLE)
        ) {
          found.add(value.slice(VARIABLE.length))
        } else {
          walk(value)
        }
      }
    }
  }
  walk((definition as { steps?: unknown } | null)?.steps)
  return found
}

/** The variables a workflow declares `required: true` in its
 * `variable_constraints`: no default, the caller must give a value (#1253). */
export function requiredVariables(definition: unknown): Set<string> {
  const rules = (definition as { variable_constraints?: unknown } | null)
    ?.variable_constraints
  const found = new Set<string>()
  if (rules !== null && typeof rules === 'object') {
    for (const [name, rule] of Object.entries(rules)) {
      if ((rule as { required?: unknown } | null)?.required === true) {
        found.add(name)
      }
    }
  }
  return found
}

/** The keys of a `variable_constraint` (`dw/workflow_schema.json`) that
 * `constraintHints` states in words. Every key the schema declares is here,
 * pinned by `editor.test.ts`: a key with no label would fall back to
 * `key: value`. `required` is the field's existing "required" mark and
 * `reason` follows the rule, so neither is a line of its own. */
export const CONSTRAINT_LABELLED_KEYS = [
  'min_frames',
  'max_frames',
  'modulus',
  'remainder',
  'snap',
  'min_length',
  'max_words_per_second',
  'seconds_from',
  'required',
  'reason',
]

/** The rule one constraint declares, as phrases - the engine's own
 * `_accepted` wording (`dw/variable_constraints.py`) - with every number
 * read from the declaration. `like` is the declared default, which decides
 * whether a `min_length` reads as a string's rule or a list's. */
function constraintPhrases(rule: Record<string, unknown>, like: unknown) {
  const parts = [...lengthPhrases(rule, like), ...framePhrases(rule)]
  if (rule.max_words_per_second !== undefined) {
    const of = rule.seconds_from !== undefined ? ` of ${rule.seconds_from}` : ''
    parts.push(
      `about ${rule.max_words_per_second} words a second${of} (a warning)`,
    )
  }
  for (const [key, value] of Object.entries(rule)) {
    if (!CONSTRAINT_LABELLED_KEYS.includes(key)) {
      parts.push(`${key}: ${displayValue(value)}`)
    }
  }
  return parts
}

/** `min_length`, as a list's rule or a string's. */
function lengthPhrases(rule: Record<string, unknown>, like: unknown): string[] {
  const minimum = rule.min_length
  if (minimum === undefined || minimum === null) return []
  if (Array.isArray(like)) {
    return [minimum === 1 ? 'a non-empty list' : `at least ${minimum} entries`]
  }
  return [
    minimum === 1 ? 'a non-empty string' : `at least ${minimum} characters`,
  ]
}

/** The frame count's range, grid and rounding. */
function framePhrases(rule: Record<string, unknown>): string[] {
  const parts: string[] = []
  const low = rule.min_frames
  const high = rule.max_frames
  if (low !== undefined && high !== undefined) parts.push(`${low} to ${high}`)
  else if (low !== undefined) parts.push(`${low} or more`)
  else if (high !== undefined) parts.push(`${high} or less`)
  if (rule.modulus !== undefined) {
    parts.push(`${rule.modulus} * n + ${rule.remainder ?? 0}`)
  }
  if (rule.snap === 'up') parts.push('an off-grid value rounds up')
  return parts
}

/** The optional item form {"item": "<field>", "default": ...} */
function isOptionalItem(
  obj: Record<string, unknown>,
): obj is Record<string, unknown> & { item: string } {
  return (
    typeof obj.item === 'string' &&
    obj.item !== '' &&
    Object.keys(obj).every((k) => k === 'item' || k === 'default')
  )
}

/** Which list-driven variables carry an entry field, read the way the
 * engine's `list_fields` reads it (`dw/for_each.py`): every step whose
 * `for_each` is `variable:<list>`, and the `item:<field>` references in it. */
function entryFields(definition: unknown): Map<string, Set<string>> {
  const found = new Map<string, Set<string>>()
  const steps = (definition as { steps?: unknown } | null)?.steps
  if (!Array.isArray(steps)) return found
  const walk = (node: unknown, into: Set<string>): void => {
    if (typeof node === 'string') {
      if (node.startsWith(ITEM) && node.length > ITEM.length) {
        into.add(node.slice(ITEM.length))
      }
    } else if (Array.isArray(node)) {
      node.forEach((n) => walk(n, into))
    } else if (node !== null && typeof node === 'object') {
      const obj = node as Record<string, unknown>
      if (isOptionalItem(obj)) {
        into.add(obj.item)
        walk(obj.default, into)
        return
      }
      for (const [key, value] of Object.entries(obj)) {
        if (key !== 'for_each') walk(value, into)
      }
    }
  }
  for (const step of steps) {
    const target = (step as { for_each?: unknown } | null)?.for_each
    if (typeof target !== 'string' || !target.startsWith(VARIABLE)) continue
    const list = target.slice(VARIABLE.length)
    const fields = found.get(list) ?? new Set<string>()
    walk(step, fields)
    found.set(list, fields)
  }
  return found
}

/** Each declared `variable_constraints` rule stated in words, as
 * {variable: lines}, for the form to show under the field. A rule on a
 * list entry's field (dialogue-short's `num_frames` in `shots`) shows under
 * the list, prefixed "each entry's <field>:". It states the rule only:
 * every verdict on a value stays the server's. */
export function constraintHints(definition: unknown): Record<string, string[]> {
  const hints: Record<string, string[]> = {}
  const { rules, variables } = declaredConstraints(definition)
  const add = (name: string, line: string | null, prefix = '') => {
    if (line) (hints[name] ??= []).push(prefix + line)
  }
  const lists = entryFields(definition)
  for (const [name, rule] of rules) {
    if (name in variables) add(name, constraintLine(rule, variables[name]))
    for (const [list, fields] of lists) {
      if (!fields.has(name) || !(list in variables)) continue
      const like = entryValue(variables[list], name)
      add(list, constraintLine(rule, like), `each entry's ${name}: `)
    }
  }
  return hints
}

type ConstraintRule = Record<string, unknown>

/** The definition's `variable_constraints` that are rules, in declaration
 * order, and the variables it declares. */
function declaredConstraints(definition: unknown): {
  rules: [string, ConstraintRule][]
  variables: Record<string, unknown>
} {
  const def = definition as {
    variables?: Record<string, unknown>
    variable_constraints?: unknown
  } | null
  const declared = def?.variable_constraints
  const rules = isPlainObject(declared)
    ? Object.entries(declared).filter(
        (entry): entry is [string, ConstraintRule] =>
          entry[1] !== null && typeof entry[1] === 'object',
      )
    : []
  return { rules, variables: def?.variables ?? {} }
}

/** One rule as a line: its phrases, then its `reason`; null when it has
 * no phrase to state. */
function constraintLine(rule: Record<string, unknown>, like: unknown) {
  const parts = constraintPhrases(rule, like)
  if (!parts.length) return null
  const reason =
    typeof rule.reason === 'string' && rule.reason ? ` - ${rule.reason}` : ''
  return parts.join(', ') + reason
}

/** The first value a list's entries give `field`, which is what its rule
 * is worded like. */
function entryValue(entries: unknown, field: string): unknown {
  if (!Array.isArray(entries)) return undefined
  const entry = entries.find(
    (e) => e !== null && typeof e === 'object' && field in e,
  ) as Record<string, unknown> | undefined
  return entry?.[field]
}

/** One preset selector as the form shows it: its bundle names, exactly the
 * keys of `presets.<selector>`, and the variables its bundles set (their
 * union, the selector itself left out, as `dw/presets.py` skips it). */
export type PresetSelector = { bundles: string[]; sets: string[] }

type Bundles = Record<string, Record<string, unknown>>

/** The definition's `presets` ({selector: {bundle: {variable: value}}}),
 * keeping only the selectors and bundles that are objects - what
 * `apply_presets` would read. */
function presetBundles(definition: unknown): Record<string, Bundles> {
  const raw = (definition as { presets?: unknown } | null)?.presets
  const found: Record<string, Bundles> = {}
  if (!isPlainObject(raw)) return found
  for (const [selector, bundles] of Object.entries(raw)) {
    if (!isPlainObject(bundles)) continue
    const kept: Bundles = {}
    for (const [name, bundle] of Object.entries(bundles)) {
      if (isPlainObject(bundle)) kept[name] = bundle
    }
    found[selector] = kept
  }
  return found
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
}

/** Each preset selector the definition declares: {selector: its bundle
 * names and the variables they set}. */
export function presetView(
  definition: unknown,
): Record<string, PresetSelector> {
  const view: Record<string, PresetSelector> = {}
  for (const [selector, bundles] of Object.entries(presetBundles(definition))) {
    const sets = new Set<string>()
    for (const bundle of Object.values(bundles)) {
      for (const name of Object.keys(bundle)) {
        if (name !== selector) sets.add(name)
      }
    }
    view[selector] = { bundles: Object.keys(bundles), sets: [...sets] }
  }
  return view
}

/** The value the chosen bundle gives `name`, as the placeholder of an empty
 * field; undefined when no chosen bundle sets it, or the field holds a value
 * of its own (an explicit value wins, `dw/presets.py`'s `apply_presets`).
 * The chosen bundle is the selector's override, else its declared default -
 * the engine's `effective_variables` order: declared default, then bundle,
 * then explicit value. Display only: the bundle's values are never sent. */
export function presetPlaceholder(
  definition: unknown,
  overrides: Record<string, string>,
  name: string,
): string | undefined {
  if (overrides[name]) return undefined
  const declared =
    (definition as { variables?: Record<string, unknown> } | null)?.variables ??
    {}
  let found: string | undefined
  // In declaration order, a later selector's bundle over an earlier one's,
  // as apply_presets sets them
  for (const [selector, bundles] of Object.entries(presetBundles(definition))) {
    if (selector === name) continue
    const bundle = chosenBundle(
      bundles,
      overrides[selector] || declared[selector],
    )
    if (bundle && Object.hasOwn(bundle, name)) {
      found = displayValue(bundle[name])
    }
  }
  return found
}

/** The bundle a selector's value names, if it names one. */
function chosenBundle(bundles: Bundles, chosen: unknown) {
  return typeof chosen === 'string' && Object.hasOwn(bundles, chosen)
    ? bundles[chosen]
    : undefined
}

export const ATTENTION_BACKENDS = [
  'native',
  'flash',
  'flash_hub',
  'sage',
  'sage_hub',
  'sparse',
  'xformers',
]

export function emptyComponent() {
  return {
    configuration: { component_type: '' },
    from_pretrained_arguments: { model_name: '' },
  }
}

export function emptyTaskStep() {
  return {
    name: 'process',
    task: { command: '', arguments: {} },
  }
}

export function emptyWorkflowStep() {
  return {
    name: 'delegate',
    workflow: { path: '', arguments: {} },
  }
}
