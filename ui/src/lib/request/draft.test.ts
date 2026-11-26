import { expect, it } from 'vitest'
import { draftKey, refKey, refusalText, saveFailure } from './draft'
import { keepDraft, readDraft } from '../draftStorage'
import { ApiError } from '../apiErrors'

it('a stale revision is the one refusal a reload answers', () => {
  expect(
    saveFailure(new ApiError('at revision 5, not 3', 409, 'stale_revision')),
  ).toEqual({
    message: 'at revision 5, not 3',
    reload: true,
    code: 'stale_revision',
  })
})

it('a note gone, or an item gone under it, is answered by a reload too', () => {
  for (const code of ['unknown_note', 'item_dropped', 'unknown_item'])
    expect(saveFailure(new ApiError('x', 422, code))?.reload).toBe(true)
  for (const code of ['invalid_note_ref', 'too_many_notes', 'too_many_refs'])
    expect(saveFailure(new ApiError('x', 422, code))?.reload).toBe(false)
})

it("says a note-route refusal for the person writing, keeping the server's reason", () => {
  const said = (code: string, message = 'why') =>
    refusalText(saveFailure(new ApiError(message, 422, code))!)
  expect(said('unknown_note')).toMatch(/no longer there.*your words stay here/)
  expect(said('item_dropped')).toMatch(
    /dropped or removed.*not saved.*Copy your words first/,
  )
  expect(said('unknown_item')).toBe(said('item_dropped'))
  expect(said('invalid_note_ref', 'job j9 is not linked to s3')).toBe(
    'A take or file this note refers to does not hold any more (job j9 is not linked to s3). Remove that reference and save again.',
  )
  expect(said('too_many_notes', 'at most 20 notes - delete one first')).toBe(
    'at most 20 notes - delete one first',
  )
})

it("any other refusal shows the server's message, with no reload", () => {
  expect(
    saveFailure(
      new ApiError(
        'this workspace has no root folder',
        409,
        'no_workspace_root',
      ),
    ),
  ).toEqual({
    message: 'this workspace has no root folder',
    reload: false,
    code: 'no_workspace_root',
  })
  expect(saveFailure(new ApiError('note too long', 400))).toEqual({
    message: 'note too long',
    reload: false,
    code: null,
  })
})

it('a failure that is not the server refusing is not shown on the item', () => {
  expect(saveFailure(new TypeError('Failed to fetch'))).toBeNull()
  expect(saveFailure(new ApiError('Internal Server Error', 500))).toBeNull()
})

it("says a stale revision in the writer's words, not the agent's", () => {
  const text = refusalText({
    message: 'read it again and rewrite the plan against revision 5',
    reload: true,
    code: 'stale_revision',
  })
  expect(text).toMatch(/not saved yet/)
  expect(text).not.toMatch(/rewrite the plan/)
  expect(
    refusalText({ message: 'note too long', reload: false, code: null }),
  ).toBe('note too long')
})

it('keeps a draft for the tab under workspace, request and item, and forgets it', () => {
  const key = draftKey('studio', 'r1', 'shot-3')
  expect(key).toBe('dw-request-note:studio/r1/shot-3')
  expect(draftKey('studio', 'r1')).toBe('dw-request-note:studio/r1/')
  expect(readDraft(key)).toBeNull()
  keepDraft(key, 'half a thought')
  expect(readDraft(key)).toBe('half a thought')
  expect(readDraft(draftKey('studio', 'r1', 'other'))).toBeNull()
  keepDraft(key, null)
  expect(readDraft(key)).toBeNull()
  // no key: nothing is kept, nothing read
  keepDraft(null, 'x')
  expect(readDraft(null)).toBeNull()
})

it('keeps the drafts of notes about different takes apart', () => {
  const take2 = [{ item: 's3', job: 'j2', output: 'run/s3.mp4' }]
  const take3 = [{ item: 's3', job: 'j3' }]
  expect(refKey([])).toBe('')
  expect(refKey(take2)).toBe('s3/j2/run/s3.mp4')
  expect(refKey([...take3, { item: 's5' }])).toBe('s3/j3/,s5//')
  const keys = new Set([
    draftKey('studio', 'r1', 's3'),
    draftKey('studio', 'r1', 's3', refKey(take2)),
    draftKey('studio', 'r1', 's3', refKey(take3)),
  ])
  expect(keys.size).toBe(3)
  expect(draftKey('studio', 'r1', 's3', '')).toBe(
    draftKey('studio', 'r1', 's3'),
  )
})

it('says a refusal as describeError reads it, never the raw message', () => {
  const raw = new ApiError('note too long\nTraceback: internals', 400)
  expect(saveFailure(raw)?.message).toBe('note too long')
})
