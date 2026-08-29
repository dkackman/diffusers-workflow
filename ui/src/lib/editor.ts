import { api } from './api'
import type { PipelineDescription, PipelineParameter } from './types'

/** Introspection descriptions, cached per class+target for the session. */
const descriptions = new Map<string, Promise<PipelineDescription | null>>()

export function classDescription(
  name: string,
  target: 'call' | 'init' | 'load' = 'call',
): Promise<PipelineDescription | null> {
  const key = `${target}:${name}`
  if (!descriptions.has(key)) {
    const fetcher =
      target === 'call' ? api.describePipeline(name) : api.describeClass(name, target)
    descriptions.set(key, fetcher.catch(() => null))
  }
  return descriptions.get(key)!
}

export const pipelineDescription = (name: string) => classDescription(name, 'call')

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
  'controlnet',
  'image_encoder',
  'prompt_enhancer_head',
  'model',
]

export const CACHE_TYPES = ['first_block', 'faster', 'mag', 'taylorseer', 'text_kv']

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
