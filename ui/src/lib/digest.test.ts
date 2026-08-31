import { describe, expect, it } from 'vitest'
import { stepDigest } from './digest'

describe('stepDigest', () => {
  it('summarizes a pipeline step: class, model, arg count', () => {
    const d = stepDigest({
      name: 'generate',
      pipeline: {
        configuration: { component_type: 'ZImagePipeline', offload: 'model' },
        from_pretrained_arguments: {
          model_name: 'Tongyi-MAI/Z-Image-Turbo',
          torch_dtype: 'torch.bfloat16',
        },
        arguments: { prompt: 'variable:prompt', num_inference_steps: 9 },
      },
      result: { content_type: 'image/png' },
    })
    expect(d.summary).toBe('ZImagePipeline · Tongyi-MAI/Z-Image-Turbo · 2 args')
    const texts = d.lines.map((l) => l.text)
    expect(texts).toContain(
      'Tongyi-MAI/Z-Image-Turbo · torch.bfloat16 · offload: model · save: image/png',
    )
    expect(texts).toContain(
      'prompt = variable:prompt · num_inference_steps = 9',
    )
  })

  it('digests populated optional sections and skips empty ones', () => {
    const d = stepDigest({
      name: 'generate',
      pipeline: {
        configuration: {
          component_type: 'FluxPipeline',
          cache: { type: 'first_block', threshold: 0.1 },
          attention_backend: 'sage',
        },
        from_pretrained_arguments: { model_name: 'x' },
        arguments: {},
        transformer: {
          configuration: { component_type: 'FluxTransformer2DModel' },
          from_pretrained_arguments: { model_name: 'y' },
          quantization_config: {
            config_type: 'BitsAndBytesConfig',
            arguments: { load_in_4bit: true },
          },
        },
        loras: [{ adapter_name: 'detail' }, { adapter_name: 'motion' }],
        scheduler: {
          configuration: { scheduler_type: 'UniPCMultistepScheduler' },
        },
      },
    })
    const bySection = Object.fromEntries(
      d.lines.map((l) => [l.section, l.text]),
    )
    expect(bySection.components).toBe('transformer (BitsAndBytesConfig)')
    expect(bySection.loras).toBe('2 LoRAs: detail, motion')
    expect(bySection.scheduler).toBe('UniPCMultistepScheduler')
    expect(bySection.acceleration).toBe('cache: first_block · attention: sage')
    expect(d.lines.find((l) => l.section === 'arguments')).toBeUndefined()
  })

  it('truncates long argument values', () => {
    const d = stepDigest({
      name: 'g',
      pipeline: {
        configuration: { component_type: 'P' },
        from_pretrained_arguments: { model_name: 'm' },
        arguments: { prompt: 'x'.repeat(100) },
      },
    })
    const args = d.lines.find((l) => l.section === 'arguments')!
    expect(args.text.length).toBeLessThan(70)
    expect(args.text).toContain('…')
  })

  it('summarizes task and sub-workflow steps', () => {
    expect(
      stepDigest({
        name: 'up',
        task: { command: 'upscale', arguments: { scale: 2 } },
      }).summary,
    ).toBe('task: upscale · 1 arg')
    expect(
      stepDigest({
        name: 'aug',
        workflow: { path: 'builtin:augment_prompt.json', arguments: {} },
      }).summary,
    ).toBe('workflow: builtin:augment_prompt.json')
  })

  it('never throws on a bare or malformed step', () => {
    expect(stepDigest({}).summary).toBe('empty step')
    expect(stepDigest({ name: 'x', pipeline: {} }).summary).toBe(
      'pipeline · 0 args',
    )
  })
})
