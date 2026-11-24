import { describe, expect, it } from 'vitest'
import {
  SECTION_LABELS,
  SERVER_SECTIONS,
  SHARED_SECTIONS,
  WS_SECTIONS,
  compareHref,
  fileHref,
  jobHref,
  legacyRedirect,
  resolveView,
  sectionLabel,
  parseView,
  serverHref,
  sharedHref,
  switchHref,
  wsHref,
  type RouteView,
  type WsSection,
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
  it('a trailing slash opens the overview, as the bare workspace does', () => {
    // '#/ws/studio/' and '#/ws/studio/?q=1' (the router strips the query)
    // both split to an empty third part
    const overview = {
      kind: 'ws',
      workspace: 'studio',
      section: 'overview',
      rest: [],
    }
    expect(parseView(['ws', 'studio', ''])).toEqual(overview)
    expect(parseView(['ws', 'studio'])).toEqual(overview)
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

describe('jobHref', () => {
  it("names the job's workspace when known", () => {
    expect(jobHref('abc', 'studio')).toBe('#/ws/studio/jobs/abc')
  })
  it('keeps the legacy form where the caller has no workspace', () => {
    expect(jobHref('abc')).toBe('#/jobs/abc')
  })
})

describe('fileHref', () => {
  it('fileHref puts the name in open=', () => {
    expect(fileHref('my studio', 'gallery', 'shot/r1/a b.png')).toBe(
      '#/ws/my%20studio/gallery?open=shot%2Fr1%2Fa%20b.png',
    )
    expect(fileHref('studio', 'assets', 'cast/priya.png')).toBe(
      '#/ws/studio/assets?open=cast%2Fpriya.png',
    )
  })
})

describe('compareHref', () => {
  it('round-trips names with slashes', () => {
    const names = ['wf/r1/a b.png', 'wf/v#1.mp4']
    const hash = compareHref('studio', names)
    expect(hash).toBe('#/ws/studio/compare/wf%2Fr1%2Fa%20b.png/wf%2Fv%231.mp4')
    const parts = hash.slice(2).split('/').map(decodeURIComponent)
    expect(parseView(parts)).toEqual({
      kind: 'ws',
      workspace: 'studio',
      section: 'compare',
      rest: names,
    })
  })
})

describe('switchHref', () => {
  const ws = (section: WsSection, ...rest: string[]): RouteView => ({
    kind: 'ws',
    workspace: 'studio',
    section,
    rest,
  })
  it('switching keeps the section', () => {
    expect(switchHref('other', ws('gallery'))).toBe('#/ws/other/gallery')
    expect(switchHref('other', ws('jobs'))).toBe('#/ws/other/jobs')
    expect(switchHref('my ws', ws('assets'))).toBe('#/ws/my%20ws/assets')
  })
  it('switching drops a job id', () => {
    expect(switchHref('other', ws('jobs', 'abc'))).toBe('#/ws/other/jobs')
    expect(switchHref('other', ws('workflows', 'a', 'b'))).toBe(
      '#/ws/other/workflows',
    )
  })
  it('the editor switches to workflows', () => {
    expect(switchHref('other', ws('edit', 'flux'))).toBe('#/ws/other/workflows')
  })
  it('compare switches to the gallery', () => {
    expect(switchHref('other', ws('compare', 'a.png', 'b.png'))).toBe(
      '#/ws/other/gallery',
    )
  })
  it('a shared page switches to the overview', () => {
    const overview = '#/ws/other/overview'
    expect(
      switchHref('other', { kind: 'shared', section: 'prompts', rest: [] }),
    ).toBe(overview)
    expect(switchHref('other', { kind: 'server', section: 'models' })).toBe(
      overview,
    )
    expect(switchHref('other', { kind: 'missing', path: '/nope' })).toBe(
      overview,
    )
  })
})
