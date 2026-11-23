import { describe, expect, it } from 'vitest'
import {
  SECTION_LABELS,
  SERVER_SECTIONS,
  SHARED_SECTIONS,
  WS_SECTIONS,
  legacyRedirect,
  resolveView,
  sectionLabel,
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
  it('reads the requests list and one request', () => {
    expect(parseView(['ws', 'studio', 'requests'])).toEqual({
      kind: 'ws',
      workspace: 'studio',
      section: 'requests',
      rest: [],
    })
    expect(parseView(['ws', 'studio', 'requests', 'r-1'])).toEqual({
      kind: 'ws',
      workspace: 'studio',
      section: 'requests',
      rest: ['r-1'],
    })
    expect(wsHref('studio', 'requests', 'r-1')).toBe('#/ws/studio/requests/r-1')
  })
  it('defaults a bare workspace to the overview', () => {
    expect(parseView(['ws', 'studio'])).toMatchObject({ section: 'overview' })
  })
  it('an unknown workspace section is missing, not the overview', () => {
    expect(parseView(['ws', 'studio', 'nope'])).toEqual({
      kind: 'missing',
      path: '/ws/studio/nope',
    })
    expect(legacyRedirect(['ws', 'studio', 'nope'], 'studio')).toBeNull()
  })
  it('an unknown shared or server section is missing', () => {
    expect(parseView(['shared', 'nope'])).toMatchObject({ kind: 'missing' })
    expect(parseView(['server', 'nope'])).toMatchObject({ kind: 'missing' })
  })
  it('an unknown top-level path is missing', () => {
    expect(legacyRedirect(['whatever'], 'studio')).toBeNull()
    expect(resolveView(['whatever'])).toEqual({
      kind: 'missing',
      path: '/whatever',
    })
  })
  it('every section has a label', () => {
    for (const s of WS_SECTIONS) expect(SECTION_LABELS.ws[s]).toBeTruthy()
    for (const s of SHARED_SECTIONS)
      expect(SECTION_LABELS.shared[s]).toBeTruthy()
    for (const s of SERVER_SECTIONS)
      expect(SECTION_LABELS.server[s]).toBeTruthy()
    expect(sectionLabel({ kind: 'missing', path: '/x' })).toBe('Not found')
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
  ])('%j -> %j', (from, to) => {
    expect(legacyRedirect(from, last)).toEqual(to)
  })
  it('leaves a parseable route alone', () => {
    expect(legacyRedirect(['ws', 'x', 'gallery'], last)).toBeNull()
    expect(legacyRedirect(['shared', 'assets'], last)).toBeNull()
  })
  it('leaves an unknown top level alone: it is a missing route', () => {
    expect(legacyRedirect(['whatever'], last)).toBeNull()
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
