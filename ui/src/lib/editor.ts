import { api } from './api'
import type { PipelineDescription, PipelineParameter } from './types'

/** Introspection descriptions, cached per pipeline class for the session. */
const descriptions = new Map<string, Promise<PipelineDescription | null>>()

export function pipelineDescription(name: string): Promise<PipelineDescription | null> {
  if (!descriptions.has(name)) {
    descriptions.set(
      name,
      api.describePipeline(name).catch(() => null),
    )
  }
  return descriptions.get(name)!
}

export type Widget = 'number' | 'boolean' | 'text' | 'textarea' | 'json'

/** Reference strings the engine resolves later - always edited as text. */
export function isReference(value: unknown): boolean {
  return (
    typeof value === 'string' &&
    (value.startsWith('variable:') ||
      value.startsWith('previous_result:') ||
      value.startsWith('constant:'))
  )
}

export function widgetFor(parameter: PipelineParameter | undefined, value: unknown): Widget {
  if (value !== null && typeof value === 'object') return 'json'
  if (isReference(value)) return 'text'
  if (typeof value === 'boolean') return 'boolean'
  if (typeof value === 'number') return 'number'
  const annotation = (parameter?.annotation ?? '') + ' ' + (parameter?.doc_type ?? '')
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

export const TORCH_DTYPES = [
  'torch.bfloat16',
  'torch.float16',
  'torch.float32',
]

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
