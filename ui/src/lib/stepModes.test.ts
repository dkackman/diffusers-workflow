// @vitest-environment jsdom
import { beforeEach, expect, it } from 'vitest'
import { StepModes } from './stepModes.svelte'
import { storageGet } from './storage'

const KEY = 'step-modes:test'
let key = KEY
const modes = () => new StepModes(() => key)

beforeEach(() => {
  localStorage.clear()
  key = KEY
})

it('opens a step it has no word on compact', () => {
  expect(modes().of({ name: 'gen' })).toBe('compact')
})

it('remembers a mode the person picks, under the workflow key', () => {
  const m = modes()
  m.set({ name: 'gen' }, 'full')
  expect(m.of({ name: 'gen' })).toBe('full')
  expect(storageGet(KEY, {})).toEqual({ gen: 'full' })
})

it('marks a new step without persisting it', () => {
  const m = modes()
  m.mark('gen', 'full')
  expect(m.of({ name: 'gen' })).toBe('full')
  expect(storageGet(KEY, {})).toEqual({})
})

it('sets every step at once and persists the lot', () => {
  const m = modes()
  m.setAll([{ name: 'a' }, { name: 'b' }], 'collapsed')
  expect(storageGet(KEY, {})).toEqual({ a: 'collapsed', b: 'collapsed' })
})

it('restores what the current key stored, and resets to a given set', () => {
  const m = modes()
  m.set({ name: 'a' }, 'full')
  key = 'step-modes:other'
  m.restore()
  expect(m.of({ name: 'a' })).toBe('compact')
  m.reset({ a: 'collapsed' })
  expect(m.of({ name: 'a' })).toBe('collapsed')
  key = KEY
  m.restore()
  expect(m.of({ name: 'a' })).toBe('full')
})
