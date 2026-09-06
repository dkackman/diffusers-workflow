import { describe, expect, it } from 'vitest'
import { groupOf, leafOf } from './grouping'

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
    expect(groupOf('templates/minimax/dialogue-short')).toBe('templates/minimax')
  })

  it('groups a gallery entry by the workflow that wrote it', () => {
    // A gallery name is <workflow identity>/<file>, and the identity is itself
    // a path now
    expect(groupOf('templates/ltx2/two-stage/still.png')).toBe(
      'templates/ltx2/two-stage',
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
