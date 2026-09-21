import { describe, expect, it } from 'vitest'
import {
  legacyRedirect,
  parseView,
  serverHref,
  sharedHref,
  wsHref,
} from './routes'

describe('parseView', () => {
  it('reads a workspace route with its section and rest', () => {
    expect(parseView(['ws', 'studio', 'jobs', 'abc'])).toEqual({
      kind: 'ws',
      workspace: 'studio',
      section: 'jobs',
      rest: ['abc'],
    })
  })
  it('defaults a bare workspace, or an unknown section, to overview', () => {
    expect(parseView(['ws', 'studio'])).toMatchObject({ section: 'overview' })
    expect(parseView(['ws', 'studio', 'nope'])).toMatchObject({
      section: 'overview',
      rest: [],
    })
  })
  it('keeps a slash-joined name in rest', () => {
    expect(
      parseView(['ws', 'default', 'workflows', 'models', 'z-image']),
    ).toMatchObject({ rest: ['models', 'z-image'] })
  })
  it('reads shared and server routes', () => {
    expect(parseView(['shared', 'prompts'])).toEqual({
      kind: 'shared',
      section: 'prompts',
      rest: [],
    })
    expect(parseView(['server', 'models'])).toEqual({
      kind: 'server',
      section: 'models',
    })
  })
  it('answers null for a legacy or empty hash', () => {
    expect(parseView([])).toBeNull()
    expect(parseView(['gallery'])).toBeNull()
    expect(parseView(['ws'])).toBeNull()
    expect(parseView(['shared', 'nope'])).toBeNull()
    expect(parseView(['server'])).toBeNull()
  })
})

describe('legacyRedirect', () => {
  const last = 'studio'
  it.each([
    [[], ['ws', 'studio', 'overview']],
    [['workflows'], ['ws', 'studio', 'workflows']],
    [
      ['workflows', 'models', 'z'],
      ['ws', 'studio', 'workflows', 'models', 'z'],
    ],
    [['gallery'], ['ws', 'studio', 'gallery']],
    [['assets'], ['ws', 'studio', 'assets']],
    [['edit'], ['ws', 'studio', 'edit']],
    [
      ['edit', 'a', 'b'],
      ['ws', 'studio', 'edit', 'a', 'b'],
    ],
    [['jobs'], ['server', 'status']],
    [
      ['jobs', 'j1'],
      ['ws', 'studio', 'jobs', 'j1'],
    ],
    [['prompts'], ['shared', 'prompts']],
    [
      ['prompt-edit', 'p'],
      ['shared', 'prompt-edit', 'p'],
    ],
    [['models'], ['server', 'models']],
    [['schema'], ['server', 'schema']],
    [['server'], ['server', 'status']],
    [['ws'], ['ws', 'studio', 'overview']],
    [['shared'], ['shared', 'prompts']],
    [
      ['server', 'nope'],
      ['server', 'status'],
    ],
  ])('%j -> %j', (from, to) => {
    expect(legacyRedirect(from, last)).toEqual(to)
  })
  it('leaves a parseable route alone', () => {
    expect(legacyRedirect(['ws', 'x', 'gallery'], last)).toBeNull()
    expect(legacyRedirect(['shared', 'assets'], last)).toBeNull()
  })
  it('sends an unknown top level to the overview', () => {
    expect(legacyRedirect(['whatever'], last)).toEqual([
      'ws',
      'studio',
      'overview',
    ])
  })
})

describe('href builders', () => {
  it('encode every segment', () => {
    expect(wsHref('my ws', 'workflows', 'a/b')).toBe(
      '#/ws/my%20ws/workflows/a%2Fb',
    )
    expect(sharedHref('prompt-edit', 'p')).toBe('#/shared/prompt-edit/p')
    expect(serverHref('models')).toBe('#/server/models')
  })
})
