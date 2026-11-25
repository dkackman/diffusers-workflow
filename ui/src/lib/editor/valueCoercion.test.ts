// #1656: what a form field commits, through the editors that commit it
import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen } from '@testing-library/svelte'
import ArgumentsEditor from './ArgumentsEditor.svelte'
import MappingEditor from './MappingEditor.svelte'
import VariablesForm from './VariablesForm.svelte'
import { coerce } from '../editor'
import { addedArgument, commitArgument, setNumber } from './commit'
import { api } from '../api'
import type { PipelineDescription, PipelineParameter } from '../types'

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

const parameter = (
  name: string,
  over: Partial<PipelineParameter> = {},
): PipelineParameter => ({
  name,
  required: false,
  default: null,
  annotation: null,
  ...over,
})

function describeWith(...parameters: PipelineParameter[]) {
  vi.spyOn(api, 'describePipeline').mockResolvedValue({
    parameters,
    accepts_kwargs: false,
  } as unknown as PipelineDescription)
}

const enter = async (field: Element, ctrlKey = false) =>
  fireEvent.keyDown(field, { key: 'Enter', ctrlKey })

describe('Enter on a null field', () => {
  it('leaves a step argument null', async () => {
    const args: Record<string, unknown> = { negative_prompt: null }
    render(ArgumentsEditor, { args, componentType: '' })
    const field = document.querySelector('input, textarea')!
    await enter(field)
    await enter(field, true)
    expect(args.negative_prompt).toBeNull()
  })

  it('leaves a mapping value null', async () => {
    const args: Record<string, unknown> = { seed: null }
    render(MappingEditor, { args })
    await enter(document.querySelector('input')!)
    expect(args.seed).toBeNull()
  })

  it('leaves a variable null', async () => {
    const variables: Record<string, unknown> = { style: null }
    render(VariablesForm, { mode: 'define', variables, idPrefix: 'v-' })
    await enter(screen.getByLabelText('style'))
    expect(variables.style).toBeNull()
  })
})

describe('replacing a reference with a literal', () => {
  it('takes the type from the parameter', async () => {
    describeWith(
      parameter('num_inference_steps', { default: 50, annotation: 'int' }),
    )
    const args: Record<string, unknown> = {
      num_inference_steps: 'variable:steps',
    }
    render(ArgumentsEditor, { args, componentType: 'SomePipeline' })
    await screen.findByDisplayValue('variable:steps')
    // the description arrives after the first render
    await vi.waitFor(() => expect(api.describePipeline).toHaveBeenCalled())
    await new Promise((resolve) => setTimeout(resolve))
    const field = screen.getByDisplayValue('variable:steps')
    await fireEvent.input(field, { target: { value: '30' } })
    await fireEvent.change(field, { target: { value: '30' } })
    expect(args.num_inference_steps).toBe(30)
  })

  it('reads the annotation when the default is null', () => {
    const steps = parameter('steps', { annotation: 'Optional[int]' })
    expect(commitArgument(steps, '30', 'variable:steps')).toBe(30)
    expect(commitArgument(undefined, '30', 'variable:steps')).toBe('30')
    expect(commitArgument(steps, 'variable:other', 'variable:steps')).toBe(
      'variable:other',
    )
  })
})

describe('setNumber', () => {
  it('keeps a reference, and shows what the document holds otherwise', () => {
    const target: Record<string, unknown> = { scale: 0.5 }
    expect(setNumber(target, 'scale', 'variable:s')).toBe('variable:s')
    expect(target.scale).toBe('variable:s')
    expect(setNumber(target, 'scale', 'abc')).toBe('variable:s')
    expect(setNumber(target, 'scale', 'Infinity')).toBe('variable:s')
    expect(setNumber(target, 'scale', ' 0.7 ')).toBe('0.7')
    expect(target.scale).toBe(0.7)
    expect(setNumber(target, 'scale', '   ')).toBe('')
    expect('scale' in target).toBe(false)
  })
})

describe('number coercion', () => {
  it('turns neither blank nor non-finite text into a number', () => {
    for (const raw of ['   ', 'Infinity', '1e999', 'NaN']) {
      expect(coerce('number', raw)).toBe(raw)
    }
    expect(coerce('number', '1e3')).toBe(1000)
  })
})

describe('adding a discovered argument', () => {
  it('braces a plain word under a type key and keeps a null default', async () => {
    describeWith(
      parameter('solver_type', { default: 'midpoint' }),
      parameter('revision', { annotation: 'str' }),
    )
    const args: Record<string, unknown> = {}
    render(ArgumentsEditor, { args, componentType: 'OtherPipeline' })
    await fireEvent.click(
      await screen.findByRole('button', { name: /show available arguments/ }),
    )
    for (const name of ['solver_type', 'revision']) {
      await fireEvent.click(
        await screen.findByRole('button', { name: new RegExp(name) }),
      )
    }
    expect(args).toEqual({ solver_type: '{midpoint}', revision: null })
  })

  it('leaves a type name, a braced value and a required argument alone', () => {
    const added = (name: string, over: Partial<PipelineParameter>) =>
      addedArgument(name, parameter(name, over))
    expect(added('torch_dtype', { default: 'torch.float16' })).toBe(
      'torch.float16',
    )
    expect(added('quant_type', { default: '{nf4}' })).toBe('{nf4}')
    expect(added('variant', { default: 'fp16' })).toBe('fp16')
    expect(added('prompt', { required: true })).toBe('')
    expect(addedArgument('mine', undefined)).toBe('')
  })
})
