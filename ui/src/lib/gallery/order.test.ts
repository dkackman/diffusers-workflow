import { expect, it } from 'vitest'
import {
  applyEdits,
  gridFiles,
  noEdits,
  roleOf,
  settleEdits,
  withGone,
  withPatch,
} from './order'
import type { GalleryFile } from '../types'

const file = (name: string, over: Partial<GalleryFile> = {}) =>
  ({ name, folder: 'f', run_id: 'r1', starred: false, ...over }) as GalleryFile

it('counts a file with no role as final', () => {
  expect(roleOf(file('a'))).toBe('final')
  expect(roleOf(file('a', { role: 'intermediate' }))).toBe('intermediate')
})

it('filters by name, role and star', () => {
  const files = [
    file('cat.png'),
    file('dog.png', { starred: true }),
    file('cat-step.png', { role: 'intermediate' }),
  ]
  const names = (fs: GalleryFile[]) => fs.map((f) => f.name)
  expect(names(gridFiles(files, 'CAT', ['final'], false))).toEqual(['cat.png'])
  expect(names(gridFiles(files, '', ['final'], true))).toEqual(['dog.png'])
  expect(
    names(gridFiles(files, 'cat', ['final', 'intermediate'], false)),
  ).toEqual(['cat.png', 'cat-step.png'])
})

it('keeps runs where their newest file put them, finals first inside a run', () => {
  const files = [
    file('r2-step', { run_id: 'r2', role: 'intermediate' }),
    file('r1-final', { run_id: 'r1' }),
    file('r2-final', { run_id: 'r2' }),
  ]
  expect(
    gridFiles(files, '', ['final', 'intermediate'], false).map((f) => f.name),
  ).toEqual(['r2-final', 'r2-step', 'r1-final'])
})

// An edit is stamped with a count; a listing carries the count its read
// began at, so a listing that began after an edit has seen it

it('a listing that began before an edit cannot undo it', () => {
  const began = noEdits().made
  const edits = withPatch(noEdits(), 'a', { starred: true })
  const settled = settleEdits([file('a')], began, edits)
  expect(applyEdits([file('a')], settled)[0].starred).toBe(true)
})

it('a listing that began after an edit shows what it carries, even another value', () => {
  const edits = withPatch(noEdits(), 'a', { starred: true })
  // Someone else unstarred it after the star landed
  const settled = settleEdits([file('a')], edits.made, edits)
  expect(settled.patched).toEqual({})
  expect(applyEdits([file('a')], settled)[0].starred).toBe(false)
})

it('a listing that agrees settles the edit whenever it began', () => {
  const edits = withPatch(noEdits(), 'a', { starred: true })
  const settled = settleEdits([file('a', { starred: true })], 0, edits)
  expect(settled.patched).toEqual({})
})

it('a later edit to the same file is not settled by a listing begun between the two', () => {
  const first = withPatch(noEdits(), 'a', { starred: true })
  const between = first.made
  const second = withPatch(first, 'a', { note: 'n' })
  const settled = settleEdits([file('a')], between, second)
  expect(applyEdits([file('a')], settled)[0].note).toBe('n')
})

it('a deleted file stays gone until a listing no longer has it', () => {
  const edits = withGone(noEdits(), ['a'])
  const held = settleEdits([file('a'), file('b')], edits.made, edits)
  expect(applyEdits([file('a'), file('b')], held).map((f) => f.name)).toEqual([
    'b',
  ])
  expect(settleEdits([file('b')], edits.made, held).gone).toEqual([])
})
