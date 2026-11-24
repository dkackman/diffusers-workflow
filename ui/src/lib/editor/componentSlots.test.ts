import { expect, it } from 'vitest'
import { slotsFrom } from './componentSlots'
import type { PipelineDescription } from '../types'

function describe(
  parameters: Array<Record<string, unknown>>,
  accepts_kwargs = false,
): PipelineDescription {
  return {
    name: 'P',
    summary: '',
    accepts_kwargs,
    parameters: parameters.map((p) => ({
      required: false,
      default: null,
      ...p,
    })),
  } as unknown as PipelineDescription
}

it("lists the constructor's component parameters", () => {
  const slots = slotsFrom(
    describe([
      { name: 'vae', annotation: "<class 'diffusers.AutoencoderKL'>" },
      { name: 'scheduler', annotation: "<enum 'KarrasDiffusionSchedulers'>" },
      { name: 'unet', annotation: null, doc_type: '[`UNet2DConditionModel`]' },
      {
        name: 'image_encoder',
        annotation: 'typing.Optional[transformers.CLIPVisionModel]',
      },
    ]),
  )
  expect(slots).toEqual(['vae', 'unet', 'image_encoder'])
})

it('skips flags and numbers', () => {
  const slots = slotsFrom(
    describe([
      { name: 'flag', annotation: "<class 'bool'>" },
      { name: 'count', annotation: 'typing.Optional[int]' },
      {
        name: 'ratio',
        annotation: null,
        doc_type: 'float, *optional*, defaults to 1',
      },
      { name: 'options', annotation: 'typing.Optional[dict]' },
      { name: 'names', annotation: 'list[str] | None' },
      {
        name: 'text_encoder',
        annotation: "<class 'transformers.CLIPTextModel'>",
      },
    ]),
  )
  expect(slots).toEqual(['text_encoder'])
})

it('a modular pipeline falls back', () => {
  expect(slotsFrom(null)).toBeNull()
  expect(
    slotsFrom(describe([{ name: 'vae', annotation: "<class 'x.Vae'>" }], true)),
  ).toBeNull()
})
