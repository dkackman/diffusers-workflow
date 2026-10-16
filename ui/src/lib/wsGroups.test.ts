import { describe, expect, it } from 'vitest'
import { groupWorkspaces } from './wsGroups'

const names = [
  'acorn-wars',
  'regression-co',
  'h3-quality',
  'regression-mod',
  'regression-smoke',
  'sci0fi',
]

describe('groupWorkspaces', () => {
  it('groups three or more names that share a prefix, where the first one stood', () => {
    expect(groupWorkspaces(names, { hidden: [], open: 'h3-quality' })).toEqual({
      entries: [
        { kind: 'ws', name: 'acorn-wars' },
        {
          kind: 'group',
          prefix: 'regression',
          names: ['regression-co', 'regression-mod', 'regression-smoke'],
        },
        { kind: 'ws', name: 'h3-quality' },
        { kind: 'ws', name: 'sci0fi' },
      ],
      hidden: [],
    })
  })
  it('leaves hidden names out, except the open one', () => {
    const r = groupWorkspaces(names, {
      hidden: ['sci0fi', 'h3-quality'],
      open: 'h3-quality',
    })
    expect(r.hidden).toEqual(['sci0fi'])
    expect(r.entries).toContainEqual({ kind: 'ws', name: 'h3-quality' })
    expect(r.entries).not.toContainEqual({ kind: 'ws', name: 'sci0fi' })
  })
  it('does not group a pair', () => {
    const r = groupWorkspaces(['a-1', 'a-2', 'b'], { hidden: [], open: 'b' })
    expect(r.entries.every((e) => e.kind === 'ws')).toBe(true)
  })
  it('hiding names can dissolve a group below the minimum', () => {
    const r = groupWorkspaces(names, {
      hidden: ['regression-co'],
      open: 'h3-quality',
    })
    expect(r.entries.some((e) => e.kind === 'group')).toBe(false)
  })
})
