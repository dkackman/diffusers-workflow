// @vitest-environment jsdom
import { beforeEach, describe, expect, it } from 'vitest'
import {
  draftKey,
  draftValues,
  openingValues,
  essentials,
  isFreeText,
  readDraft,
  seedVariable,
  writeDraft,
} from './runForm'
import type { WorkflowCard, WorkflowDefinition } from '../types'

const card = (extra: object = {}) => extra as WorkflowCard

beforeEach(() => localStorage.clear())

describe('essentials', () => {
  it('required, inputs, cost drivers, selectors and free text are essentials', () => {
    const workflow = {
      id: 'x',
      variables: {
        need: null,
        image: '',
        frames: 24,
        quality: 'high',
        note: 'a cat',
        lookup: 'prompt:cat',
        other: 7,
      },
      variable_constraints: { need: { required: true } },
      presets: { quality: { high: { other: 1 }, low: { other: 2 } } },
    } as unknown as WorkflowDefinition
    const found = essentials(
      workflow,
      card({ inputs: { image: 'image' }, cost_drivers: { frames: [24] } }),
    )
    expect([...found].sort()).toEqual(
      ['frames', 'image', 'lookup', 'need', 'note', 'quality'].sort(),
    )
  })

  it('dtype and backend strings are advanced', () => {
    const workflow = {
      id: 'x',
      variables: { dtype: 'bfloat16', attention_backend: 'sage', seed: 0 },
    } as WorkflowDefinition
    expect(essentials(workflow, undefined).size).toBe(0)
  })

  it('free text is a string that is empty, holds whitespace or is a prompt reference', () => {
    expect(['', ' ', 'a b', 'prompt:x'].map(isFreeText)).toEqual([
      true,
      true,
      true,
      true,
    ])
    expect(['nf4', 3, null].map(isFreeText)).toEqual([false, false, false])
  })
})

it('finds the seed variable', () => {
  const seeded = (seed: unknown) =>
    seedVariable({ id: 'x', seed } as WorkflowDefinition)
  expect(seeded('variable:seed')).toBe('seed')
  expect(seeded(7)).toBeNull()
  expect(seeded(undefined)).toBeNull()
})

describe('drafts', () => {
  it('drops a remembered variable the workflow no longer has', () => {
    const key = draftKey('studio', 'a')
    writeDraft(key, { prompt: 'cat', gone: 'x', blank: '' })
    expect(readDraft(key, ['prompt', 'blank'])).toEqual({ prompt: 'cat' })
  })

  it('are per workspace', () => {
    writeDraft(draftKey('one', 'a'), { prompt: 'cat' })
    expect(readDraft(draftKey('two', 'a'), ['prompt'])).toEqual({})
    expect(readDraft(draftKey('one', 'a'), ['prompt'])).toEqual({
      prompt: 'cat',
    })
  })

  it('an all-blank draft removes the entry', () => {
    const key = draftKey('one', 'a')
    writeDraft(key, { prompt: 'cat' })
    writeDraft(key, { prompt: '' })
    expect(localStorage.getItem('dw-' + key)).toBeNull()
  })
})

describe('a prefill is not a draft', () => {
  it("leaves another workflow's prefill for the workflow it names", () => {
    sessionStorage.setItem(
      'dw-workflow-prefill',
      JSON.stringify({ workflow: 'y', arguments: { b: 2 } }),
    )
    const wf = { id: 'x', variables: { b: 0 } } as WorkflowDefinition
    expect(openingValues('w', 'x', wf).prefill).toEqual({})
    expect(openingValues('w', 'y', wf).prefill).toEqual({ b: '2' })
    expect(sessionStorage.getItem('dw-workflow-prefill')).toBeNull()
  })

  it('opening reports the prefill apart from the draft', () => {
    writeDraft(draftKey('w', 'x'), { a: 'typed' })
    sessionStorage.setItem(
      'dw-workflow-prefill',
      JSON.stringify({ workflow: 'x', arguments: { b: 2 } }),
    )
    const wf = { id: 'x', variables: { a: '', b: 0 } } as WorkflowDefinition
    const opened = openingValues('w', 'x', wf)
    expect(opened.restored).toBe(true)
    expect(opened.draft).toEqual({ a: 'typed' })
    expect(opened.prefill).toEqual({ b: '2' })
    expect(opened.overrides).toEqual({ a: 'typed', b: '2' })
  })

  it('only what was typed beyond the prefill is kept for a draft', () => {
    const opened = { draft: { a: 'old' }, prefill: { a: 'p', b: '2' } }
    // untouched: the prefill is dropped, the draft it replaced stays
    expect(draftValues({ a: 'p', b: '2' }, opened)).toEqual({ a: 'old' })
    // edited: the person's text is stored
    expect(draftValues({ a: 'p', b: '3' }, opened)).toEqual({
      a: 'old',
      b: '3',
    })
  })
})
