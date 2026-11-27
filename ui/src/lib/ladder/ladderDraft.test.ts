import { expect, it } from 'vitest'
import { ApiError } from '../apiErrors'
import {
  added,
  draftOf,
  keyProblem,
  moved,
  removed,
  sameLadder,
  sendable,
  staleEtag,
  targetDevices,
  targetName,
} from './ladderDraft'

const CUDA = { workflow: 'models/minimax-h3-image', devices: ['cuda'] }

it('a draft holds every tier in order, empty where the ladder has none', () => {
  const draft = draftOf({ best: [CUDA, 'models/qwen-image-2.1'] })
  expect(Object.keys(draft)).toEqual(['draft', 'standard', 'best'])
  expect(draft.draft).toEqual([])
  expect(draft.best).toEqual([CUDA, 'models/qwen-image-2.1'])
})

it('reads a target as a bare name or one limited to devices', () => {
  expect(targetName('models/z-image')).toBe('models/z-image')
  expect(targetName(CUDA)).toBe('models/minimax-h3-image')
  expect(targetDevices('models/z-image')).toEqual([])
  expect(targetDevices(CUDA)).toEqual(['cuda'])
})

it('moves an entry within its list and no further', () => {
  const list = ['a', 'b', 'c']
  expect(moved(list, 2, -1)).toEqual(['a', 'c', 'b'])
  expect(moved(list, 0, 1)).toEqual(['b', 'a', 'c'])
  expect(moved(list, 0, -1)).toBe(list)
  expect(moved(list, 2, 1)).toBe(list)
  expect(removed(list, 1)).toEqual(['a', 'c'])
})

it('adds a bare name, or an object when devices are picked', () => {
  expect(added(['a'], ' models/z-image ')).toEqual(['a', 'models/z-image'])
  expect(added([], 'models/minimax-h3-image', ['cuda'])).toEqual([CUDA])
  expect(added(['a'], '  ')).toEqual(['a'])
})

it('sends only the tiers that hold a target', () => {
  const draft = draftOf({ standard: ['models/z-image'] })
  expect(sendable(draft)).toEqual({ standard: ['models/z-image'] })
  expect(sameLadder(draft, draftOf({ standard: ['models/z-image'] }))).toBe(
    true,
  )
  expect(
    sameLadder(draft, draftOf({ standard: ['models/z-image-sdnq'] })),
  ).toBe(false)
})

it('refuses a key name the server would, before sending', () => {
  expect(keyProblem('image_edit', [])).toBeNull()
  expect(keyProblem('image-edit2', [])).toBeNull()
  for (const bad of ['', '2fast', 'has space', 'a/b', '../x', 'é'])
    expect(keyProblem(bad, [])).toMatch(/starts with a letter/)
  expect(keyProblem('a'.repeat(101), [])).toMatch(/at most 100/)
  expect(keyProblem('text_to_image', ['text_to_image'])).toMatch(/already/)
})

it('reads the etag a 412 stale_write carries, and nothing else', () => {
  const stale = (etag: unknown) =>
    new ApiError('changed', 412, 'stale_write', { code: 'stale_write', etag })
  expect(staleEtag(stale('"now"'))).toBe('"now"')
  expect(staleEtag(stale(null))).toBeNull()
  expect(staleEtag(new ApiError('changed', 409, 'stale_write'))).toBeUndefined()
  expect(staleEtag(new ApiError('bad', 422))).toBeUndefined()
  expect(staleEtag(new Error('x'))).toBeUndefined()
})
