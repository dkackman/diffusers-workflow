import { expect, it } from 'vitest'
import {
  applyEdits,
  dateGroup,
  gridFiles,
  gridGroups,
  inGroupOrder,
  noEdits,
  roleOf,
  settleEdits,
  withGone,
  withPatch,
} from './order'
import type { GalleryFile, OutputRole } from '../types'

const file = (name: string, over: Partial<GalleryFile> = {}) =>
  ({ name, folder: 'f', run_id: 'r1', starred: false, ...over }) as GalleryFile

const names = (fs: GalleryFile[]) => fs.map((f) => f.name)
const BOTH: OutputRole[] = ['final', 'intermediate']
type Query = Parameters<typeof gridFiles>[1]
const q = (over: Partial<Query> = {}): Query => ({
  text: '',
  roles: ['final'],
  starredOnly: false,
  sort: 'newest',
  prompts: {},
  ...over,
})

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
  expect(names(gridFiles(files, q({ text: 'CAT' })))).toEqual(['cat.png'])
  expect(names(gridFiles(files, q({ starredOnly: true })))).toEqual(['dog.png'])
  expect(names(gridFiles(files, q({ text: 'cat', roles: BOTH })))).toEqual([
    'cat.png',
    'cat-step.png',
  ])
})

it('matches a word of the prompt, the note or the name', () => {
  const files = [
    file('a.png', { run_id: 'r1' }),
    file('b.png', { run_id: 'r2', note: 'Keeper for the poster' }),
    file('c.png', { run_id: 'r3', label: 'lighthouse' }),
    file('d.mp4', { run_id: 'r4' }),
  ]
  const prompts = { r1: 'A Red fox at dawn', r4: 'waves on the shore' }
  const find = (text: string) => names(gridFiles(files, q({ text, prompts })))
  expect(find('fox')).toEqual(['a.png'])
  expect(find('POSTER')).toEqual(['b.png'])
  expect(find('lighthouse')).toEqual(['c.png'])
  // a clip is found through its job's prompt
  expect(find('shore')).toEqual(['d.mp4'])
  expect(find('b.png')).toEqual(['b.png'])
})

it('needs every word', () => {
  const files = [
    file('fox.png', { run_id: 'r1' }),
    file('fox-2.png', { run_id: 'r2' }),
  ]
  const prompts = { r1: 'a red fox', r2: 'a grey fox' }
  // the words may match in different places: one the name, one the prompt
  expect(names(gridFiles(files, q({ text: 'fox  red', prompts })))).toEqual([
    'fox.png',
  ])
  expect(names(gridFiles(files, q({ text: 'fox blue', prompts })))).toEqual([])
})

it('sorts oldest and by name', () => {
  const files = [
    file('r3-b', { run_id: 'r3' }),
    file('r3-step', { run_id: 'r3', role: 'intermediate' }),
    file('r3-a', { run_id: 'r3' }),
    file('r2-z', { run_id: 'r2' }),
    file('r1-m', { run_id: 'r1' }),
  ]
  const sorted = (sort: 'newest' | 'oldest' | 'name') =>
    names(gridFiles(files, q({ roles: BOTH, sort })))
  expect(sorted('newest')).toEqual(['r3-b', 'r3-a', 'r3-step', 'r2-z', 'r1-m'])
  // the runs reverse; inside one the finals still lead
  expect(sorted('oldest')).toEqual(['r1-m', 'r2-z', 'r3-b', 'r3-a', 'r3-step'])
  expect(sorted('name')).toEqual(['r1-m', 'r2-z', 'r3-a', 'r3-b', 'r3-step'])
})

it('dates fall into Today, Yesterday, This week and Earlier', () => {
  const now = new Date(2026, 10, 23, 15, 0)
  const at = (d: number, h = 12) => new Date(2026, 10, d, h).getTime() / 1000
  expect(dateGroup(at(23, 0), now)).toBe('Today')
  expect(dateGroup(at(22, 23), now)).toBe('Yesterday')
  expect(dateGroup(at(22, 0), now)).toBe('Yesterday')
  expect(dateGroup(at(21, 23), now)).toBe('This week')
  // the seven days before yesterday
  expect(dateGroup(at(15, 0), now)).toBe('This week')
  expect(dateGroup(at(14, 23), now)).toBe('Earlier')
})

it('keeps runs where their newest file put them, finals first inside a run', () => {
  const files = [
    file('r2-step', { run_id: 'r2', role: 'intermediate' }),
    file('r1-final', { run_id: 'r1' }),
    file('r2-final', { run_id: 'r2' }),
  ]
  expect(names(gridFiles(files, q({ roles: BOTH })))).toEqual([
    'r2-final',
    'r2-step',
    'r1-final',
  ])
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
  const held = settleEdits([file('a'), file('b')], edits.made - 1, edits)
  expect(applyEdits([file('a'), file('b')], held).map((f) => f.name)).toEqual([
    'b',
  ])
  expect(settleEdits([file('b')], edits.made, held).gone).toEqual({})
})

it('a deleted name is settled by a listing begun after the delete', () => {
  const before = noEdits().made
  const edits = withGone(noEdits(), ['a'])
  // A read begun before the delete still lists it: it stays hidden
  const held = settleEdits([file('a')], before, edits)
  expect(applyEdits([file('a')], held)).toEqual([])
  // A read begun after it lists a real file under the name (another
  // workspace, or a run that wrote it since): it shows
  const settled = settleEdits([file('a')], edits.made, held)
  expect(settled.gone).toEqual({})
  expect(applyEdits([file('a')], settled).map((f) => f.name)).toEqual(['a'])
})

it('leaves out the files of a shut group, heading by heading', () => {
  const files = [
    file('z/1', { folder: 'z' }),
    file('a/1', { folder: 'a' }),
    file('m/1', { folder: 'm' }),
  ]
  const groups = gridGroups(files, 'folder', 'newest', new Date())
  expect(names(inGroupOrder(files, groups))).toEqual(['a/1', 'm/1', 'z/1'])
  expect(names(inGroupOrder(files, groups, (g) => g === 'm'))).toEqual([
    'a/1',
    'z/1',
  ])
})

it('take2 sorts before take10, whatever the case', () => {
  const files = ['take10.png', 'Take3.png', 'take2.png', 'TAKE1.png'].map(
    (name) => file(name),
  )
  expect(names(gridFiles(files, q({ roles: BOTH, sort: 'name' })))).toEqual([
    'TAKE1.png',
    'take2.png',
    'Take3.png',
    'take10.png',
  ])
})
