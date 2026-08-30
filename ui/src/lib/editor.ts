import { api } from './api'
import type { PipelineDescription, PipelineParameter } from './types'

/** Introspection descriptions, cached per class+target for the session. */
const descriptions = new Map<string, Promise<PipelineDescription | null>>()

export function classDescription(
  name: string,
  target: 'call' | 'init' | 'load' | 'task' = 'call',
): Promise<PipelineDescription | null> {
  const key = `${target}:${name}`
  if (!descriptions.has(key)) {
    const fetcher =
      target === 'task'
        ? api.describeTask(name)
        : target === 'call'
          ? api.describePipeline(name)
          : api.describeClass(name, target)
    descriptions.set(
      key,
      fetcher.catch(() => null),
    )
  }
  return descriptions.get(key)!
}

export type Widget = 'number' | 'boolean' | 'text' | 'textarea' | 'json'

/** Reference strings the engine resolves later - always edited as text. */
export function isReference(value: unknown): boolean {
  return (
    typeof value === 'string' &&
    (value.startsWith('variable:') ||
      value.startsWith('previous_result:') ||
      value.startsWith('constant:') ||
      value.startsWith('prompt:'))
  )
}

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

/** Turn what the input produced back into the JSON value the engine expects.
 * Reference strings and unparseable numbers stay strings - the engine
 * substitutes variables after schema validation. */
export function coerce(widget: Widget, raw: string): unknown {
  if (widget === 'number' && raw !== '' && !isReference(raw)) {
    const parsed = Number(raw)
    if (!Number.isNaN(parsed)) return parsed
  }
  if (widget === 'boolean') return raw === 'true'
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

export const CONTENT_TYPES = [
  'image/png',
  'image/jpeg',
  'image/webp',
  'video/mp4',
  'image/gif',
  'audio/wav',
  'audio/mpeg',
  'application/json',
  'text/plain',
]

/** One value-to-input-string rule for every editor field. */
export function displayValue(value: unknown, pretty = false): string {
  if (value === null || value === undefined) return ''
  if (typeof value === 'object') {
    return pretty ? JSON.stringify(value, null, 2) : JSON.stringify(value)
  }
  return String(value)
}

/** Parse a number into target[key], or delete the key on empty input. */
export function setNumber(
  target: Record<string, unknown>,
  key: string,
  raw: string,
) {
  const parsed = Number(raw)
  if (raw === '') delete target[key]
  else if (!Number.isNaN(parsed)) target[key] = parsed
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
      arguments: { prompt: 'variable:prompt' },
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

export const CACHE_TYPES = [
  'first_block',
  'faster',
  'mag',
  'taylorseer',
  'text_kv',
]

export const ATTENTION_BACKENDS = [
  'native',
  'flash',
  'flash_hub',
  'sage',
  'sage_hub',
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

/** Reference completions available to a value input in step `stepIndex`:
 * every declared variable, every EARLIER step's result - with media
 * property suffixes for steps whose result is a video - and every stored
 * prompt the library lists. */
export function referenceSuggestions(
  workflow: Record<string, any>,
  stepIndex: number,
  promptNames: string[] = [],
): string[] {
  const suggestions: string[] = []
  for (const name of Object.keys(workflow.variables ?? {})) {
    suggestions.push(`variable:${name}`)
  }
  for (const name of promptNames) {
    suggestions.push(`prompt:${name}`)
  }
  const steps: Array<Record<string, any>> = workflow.steps ?? []
  for (const step of steps.slice(0, Math.max(0, stepIndex))) {
    if (!step.name) continue
    suggestions.push(`previous_result:${step.name}`)
    const contentType = step.result?.content_type ?? ''
    if (contentType.startsWith('video')) {
      suggestions.push(
        `previous_result:${step.name}.frames`,
        `previous_result:${step.name}.audio`,
      )
    }
  }
  return suggestions
}

/** References the workflow makes that nothing declares: variable: names
 * missing from variables, previous_result: names that match no EARLIER
 * step, and - when the prompt library's listing is supplied - prompt:
 * names it does not hold. The engine would fail these at run time; the
 * editor says so now. */
export function danglingReferences(
  workflow: Record<string, any>,
  promptNames?: string[],
): string[] {
  const problems: string[] = []
  const variables = new Set(Object.keys(workflow.variables ?? {}))
  const prompts = promptNames === undefined ? null : new Set(promptNames)
  const steps: Array<Record<string, any>> = workflow.steps ?? []

  steps.forEach((step, index) => {
    const earlier = new Set(
      steps
        .slice(0, index)
        .map((s) => s.name)
        .filter(Boolean),
    )

    const scan = (value: unknown) => {
      if (typeof value === 'string') {
        if (value.startsWith('variable:')) {
          const name = value.slice('variable:'.length)
          if (!variables.has(name)) {
            problems.push(
              `Step '${step.name}': variable:${name} - no such variable is declared`,
            )
          }
        } else if (value.startsWith('previous_result:')) {
          const name = value.slice('previous_result:'.length).split('.')[0]
          if (!earlier.has(name)) {
            problems.push(
              `Step '${step.name}': previous_result:${name} - no earlier step has that name`,
            )
          }
        } else if (value.startsWith('prompt:')) {
          // Without a listing the server resolves these at run time - only
          // a supplied library can say a name is missing
          const name = value.slice('prompt:'.length)
          if (prompts !== null && !prompts.has(name)) {
            problems.push(
              `Step '${step.name}': prompt:${name} - the prompt library has no such prompt`,
            )
          }
        }
      } else if (Array.isArray(value)) {
        value.forEach(scan)
      } else if (value !== null && typeof value === 'object') {
        Object.values(value).forEach(scan)
      }
    }
    scan(step)
  })
  return problems
}
