/** Textual digests of a step's JSON. The summary names the step's work
 * on one line for the collapsed bar; the lines are the compact view -
 * one per populated area, each tagged with the full-view section it
 * expands into. Pure: same JSON in, same text out. */

export type DigestSection =
  'main' | 'arguments' | 'components' | 'loras' | 'scheduler' | 'acceleration'

export interface DigestLine {
  text: string
  section: DigestSection
}

export interface StepDigest {
  summary: string
  lines: DigestLine[]
}

const COMPONENT_KEYS = new Set([
  'configuration',
  'from_pretrained_arguments',
  'arguments',
  'loras',
  'scheduler',
  'shared_components',
  'reused_components',
])

function truncate(value: unknown, max = 32): string {
  const s =
    value !== null && typeof value === 'object'
      ? JSON.stringify(value)
      : String(value)
  return s.length > max ? s.slice(0, max - 1) + '…' : s
}

function countText(n: number, noun: string): string {
  return `${n} ${noun}${n === 1 ? '' : 's'}`
}

export function stepDigest(step: Record<string, any>): StepDigest {
  if (step.pipeline) return pipelineDigest(step)
  if (step.task) {
    const argCount = Object.keys(step.task.arguments ?? {}).length
    return {
      summary: `task: ${step.task.command || '?'} · ${countText(argCount, 'arg')}`,
      lines: argsLine(step.task.arguments),
    }
  }
  if (step.workflow) {
    return {
      summary: `workflow: ${step.workflow.path || '?'}`,
      lines: argsLine(step.workflow.arguments),
    }
  }
  if (step.pipeline_reference) {
    return { summary: 'pipeline reference', lines: [] }
  }
  return { summary: 'empty step', lines: [] }
}

function argsLine(args: Record<string, unknown> | undefined): DigestLine[] {
  const entries = Object.entries(args ?? {})
  if (!entries.length) return []
  return [
    {
      section: 'arguments',
      text: entries.map(([k, v]) => `${k} = ${truncate(v)}`).join(' · '),
    },
  ]
}

function pipelineDigest(step: Record<string, any>): StepDigest {
  const pipeline = step.pipeline
  const configuration = pipeline.configuration ?? {}
  const pretrained = pipeline.from_pretrained_arguments ?? {}
  const args = pipeline.arguments ?? {}
  const argCount = Object.keys(args).length

  const summary = [
    configuration.component_type || 'pipeline',
    pretrained.model_name,
    countText(argCount, 'arg'),
  ]
    .filter(Boolean)
    .join(' · ')

  const lines: DigestLine[] = []

  const main = [
    pretrained.model_name,
    pretrained.torch_dtype,
    configuration.offload ? `offload: ${configuration.offload}` : null,
    step.result?.content_type ? `save: ${step.result.content_type}` : null,
  ].filter(Boolean)
  if (main.length) lines.push({ section: 'main', text: main.join(' · ') })

  lines.push(...argsLine(args))

  const components = Object.keys(pipeline).filter(
    (key) =>
      !COMPONENT_KEYS.has(key) &&
      pipeline[key] !== null &&
      typeof pipeline[key] === 'object',
  )
  if (components.length) {
    lines.push({
      section: 'components',
      text: components
        .map((slot) => {
          const quant = pipeline[slot]?.quantization_config?.config_type
          return quant ? `${slot} (${quant})` : slot
        })
        .join(' · '),
    })
  }

  const loras: Array<Record<string, any>> = pipeline.loras ?? []
  if (loras.length) {
    lines.push({
      section: 'loras',
      text: `${countText(loras.length, 'LoRA')}: ${loras
        .map((l) => l.adapter_name || l.model_name || '?')
        .join(', ')}`,
    })
  }

  const schedulerType = pipeline.scheduler?.configuration?.scheduler_type
  if (schedulerType) lines.push({ section: 'scheduler', text: schedulerType })

  const acceleration = [
    configuration.cache ? `cache: ${configuration.cache.type}` : null,
    configuration.attention_backend
      ? `attention: ${configuration.attention_backend}`
      : null,
    configuration.prompt_weighting ? 'prompt weighting' : null,
  ].filter(Boolean)
  if (acceleration.length) {
    lines.push({ section: 'acceleration', text: acceleration.join(' · ') })
  }

  return { summary, lines }
}
