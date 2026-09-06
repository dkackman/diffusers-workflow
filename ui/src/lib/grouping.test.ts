import { describe, expect, it } from 'vitest'
import { groupNames, groupOf, leafOf } from './grouping'

describe('groupOf', () => {
  it('has no group for a name at the root', () => {
    expect(groupOf('sd15')).toBe('')
  })

  it('groups a one-level name by its folder', () => {
    expect(groupOf('models/flux-dev')).toBe('models')
  })

  it('keeps every folder level, not just the first', () => {
    // The regression this exists to prevent: taking only the first segment put
    // all 64 templates in one group and hid the ltx2 and minimax families
    expect(groupOf('templates/minimax/dialogue-short')).toBe(
      'templates/minimax',
    )
  })

  it('is not how the gallery groups - a gallery name carries a run id', () => {
    // <identity>/<run id>/<file>: the last folder is the run, not the workflow.
    // GalleryPage groups by the server's `folder` (strip_run_id) instead
    expect(groupOf('templates/ltx2/two-stage/20260906-101500-a1b2c3d4/still.png')).toBe(
      'templates/ltx2/two-stage/20260906-101500-a1b2c3d4',
    )
  })
})

describe('leafOf', () => {
  it('returns the whole name when there is no folder', () => {
    expect(leafOf('sd15')).toBe('sd15')
  })

  it('returns the last segment, since the heading shows the rest', () => {
    expect(leafOf('templates/minimax/dialogue-short')).toBe('dialogue-short')
  })
})

describe('groupNames', () => {
  it('buckets names by their group in one pass, groups sorted', () => {
    const grouped = groupNames(['b/two', 'a/one', 'root', 'a/three'])
    expect([...grouped.keys()]).toEqual(['', 'a', 'b'])
    expect(grouped.get('a')).toEqual(['a/one', 'a/three'])
    expect(grouped.get('')).toEqual(['root'])
  })

  it('takes a caller-supplied grouper', () => {
    // The gallery groups by the folder the server computed, which strips the
    // run id - the name alone cannot tell a run id from a folder
    const folderOf = (name: string) => name.split('/')[0]
    const grouped = groupNames(
      ['tti/20260906-101500-a1b2c3d4/a.png', 'tti/20260906-101612-ffffffff/b.png'],
      folderOf,
    )
    expect([...grouped.keys()]).toEqual(['tti'])
  })
})
