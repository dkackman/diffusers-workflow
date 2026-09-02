import { api } from './api'
import type { PipelineDescription, PipelineParameter } from './types'
import { danglingReferenceDetails } from './flow'

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

export type MediaKind = 'image' | 'video'

// A pipeline's __call__ signature types media arguments as a union like
// PipelineImageInput (PIL.Image.Image | np.ndarray | torch.Tensor | lists of
// those) - introspection stringifies whatever the annotation or the
// docstring's type column says, and both carry the word "image"/"video".
// No \b word boundary: PyTorch/diffusers type names are often one run of
// camelCase (PipelineImageInput, VaeImageProcessor's output types, ...)
// with no separator before "Image"/"Video" for \b to catch.
const IMAGE_TYPE_PATTERN = /image/i
const VIDEO_TYPE_PATTERN = /video/i

// **kwargs-only callables (from_pretrained) and dw's own tasks (see
// dw/introspection.py's describe_task, e.g. depth_estimator's "image") carry
// no annotation at all - the same argument names workflows/*.json already
// use (img2img's `image`, ControlNet's `control_image`/`conditioning_image`,
// inpainting's `mask_image`) are the only signal left.
const IMAGE_NAME_PATTERN =
  /^(image|mask_image|control_image|conditioning_image|init_image|input_image|ip_adapter_image|reference_image|style_image|depth_map)(_\d+)?$/i
const VIDEO_NAME_PATTERN =
  /^(video|control_video|conditioning_video|input_video)(_\d+)?$/i

/** Whether an argument takes an image/video rather than plain text - and
 * which kind. Reference values (variable:, previous_result:, ...) are the
 * caller's job to exclude first; they resolve to media at run time but are
 * always edited as the reference string itself. */
export function mediaKindFor(
  parameter: PipelineParameter | undefined,
  key: string,
): MediaKind | null {
  const typeText = `${parameter?.annotation ?? ''} ${parameter?.doc_type ?? ''}`
  if (IMAGE_TYPE_PATTERN.test(typeText)) return 'image'
  if (VIDEO_TYPE_PATTERN.test(typeText)) return 'video'
  if (!parameter?.annotation && !parameter?.doc_type) {
    if (IMAGE_NAME_PATTERN.test(key)) return 'image'
    if (VIDEO_NAME_PATTERN.test(key)) return 'video'
  }
  return null
}

/** The file path or URL a media argument's value names, whichever shape it
 * is in: a bare string, or the `{ location: ... }` object the engine also
 * accepts (see dw/arguments.py's FROM_FILE_KEY) - optionally alongside a
 * `media_type`. */
export function mediaLocation(value: unknown): string {
  if (typeof value === 'string') return value
  if (
    value !== null &&
    typeof value === 'object' &&
    !Array.isArray(value) &&
    typeof (value as Record<string, unknown>).location === 'string'
  ) {
    return (value as Record<string, unknown>).location as string
  }
  return ''
}

/** Write a new location into a media argument's value, preserving its shape:
 * an object value keeps its other keys (e.g. `media_type`), anything else
 * becomes a plain path string - both are forms the engine accepts. */
export function withMediaLocation(value: unknown, location: string): unknown {
  if (value !== null && typeof value === 'object' && !Array.isArray(value)) {
    return { ...(value as Record<string, unknown>), location }
  }
  return location
}

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
  return danglingReferenceDetails(workflow, promptNames).map((d) => d.message)
}
