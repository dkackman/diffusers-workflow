import { expect, it } from 'vitest'
import { draftAtRisk, refusalText, saveFailure } from './draft'
import { ApiError } from '../apiErrors'

it('a draft with text that differs from the saved comment is at risk', () => {
  expect(draftAtRisk('needs more red', null)).toBe(true)
  expect(draftAtRisk('needs more red', 'needs red')).toBe(true)
})

it('an empty, blank or unchanged draft is not', () => {
  expect(draftAtRisk('', null)).toBe(false)
  expect(draftAtRisk('  \n', 'kept')).toBe(false)
  expect(draftAtRisk('kept', 'kept')).toBe(false)
})

it('a stale revision is the one refusal a reload answers', () => {
  expect(
    saveFailure(new ApiError('at revision 5, not 3', 409, 'stale_revision')),
  ).toEqual({ message: 'at revision 5, not 3', reload: true })
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
  ).toEqual({ message: 'this workspace has no root folder', reload: false })
  expect(saveFailure(new ApiError('comment too long', 400))).toEqual({
    message: 'comment too long',
    reload: false,
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
  })
  expect(text).toMatch(/not saved yet/)
  expect(text).not.toMatch(/rewrite the plan/)
  expect(refusalText({ message: 'comment too long', reload: false })).toBe(
    'comment too long',
  )
})
