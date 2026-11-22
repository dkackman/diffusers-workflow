import { expect, it } from 'vitest'
import { gridFiles, roleOf } from './order'
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
