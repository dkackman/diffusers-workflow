import { render } from '@testing-library/svelte'
import { createRawSnippet } from 'svelte'
import { describe, expect, it } from 'vitest'
import FolderGroups from './FolderGroups.svelte'

const card = createRawSnippet((name: () => string) => ({
  render: () => `<span>${name()}</span>`,
}))

describe('FolderGroups', () => {
  it('groups by the default groupOf when none is supplied', () => {
    const { container } = render(FolderGroups, {
      names: ['a/one', 'a/two', 'b/three'],
      collapseKey: 'test-default-groupof',
      filterActive: false,
      card,
    })
    const headings = [...container.querySelectorAll('.group')].map(
      (el) => el.textContent,
    )
    expect(headings.length).toBe(2)
    expect(headings.some((h) => h?.includes('a/'))).toBe(true)
    expect(headings.some((h) => h?.includes('b/'))).toBe(true)
  })

  it('groups by a caller-supplied groupOf instead, since a gallery name carries a run id the name alone cannot strip', () => {
    // The default groupOf would put each of these under its own heading -
    // the last path segment is a run id, not a folder. The gallery instead
    // passes the server's `folder` lookup.
    const { container } = render(FolderGroups, {
      names: [
        'tti/20260906-101500-a1b2c3d4/a.png',
        'tti/20260906-101612-ffffffff/b.png',
      ],
      collapseKey: 'test-custom-groupof',
      filterActive: false,
      groupOf: (name: string) => name.split('/')[0],
      card,
    })
    const headings = [...container.querySelectorAll('.group')]
    expect(headings.length).toBe(1)
    expect(headings[0].textContent).toContain('tti/')
    expect(headings[0].textContent).toContain('(2)')
  })
})
