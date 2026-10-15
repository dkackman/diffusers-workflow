import { cleanup, render, screen } from '@testing-library/svelte'
import { afterEach, describe, expect, it, vi } from 'vitest'

const file = (name: string, kind = 'image') => ({
  name,
  label: name.split('/').pop(),
  kind,
  url: `/outputs/${name}`,
  folder: 'wf',
  subfolder: '',
  run_id: 'r',
  size: 1,
  mtime: 1,
})

vi.mock('../api', () => ({
  api: {
    gallery: () =>
      Promise.resolve({ files: [file('wf/r1/a.png'), file('wf/r2/b.png')] }),
    galleryMetadata: (name: string) =>
      Promise.resolve({
        metadata: {
          model_name: 'h3',
          seed: name.includes('a') ? 1 : 2,
          arguments: { steps: 12 },
        },
        job: null,
      }),
    galleryThumbnailUrl: (n: string) => `/thumb/${n}`,
    getJob: () => Promise.reject(new Error('none')),
  },
}))

import ComparePage from './ComparePage.svelte'

describe('ComparePage', () => {
  afterEach(cleanup)
  it('shows each file and only the argument that differs', async () => {
    render(ComparePage, { props: { names: ['wf/r1/a.png', 'wf/r2/b.png'] } })
    expect(await screen.findByAltText('wf/r1/a.png')).toBeTruthy()
    expect(await screen.findByRole('rowheader', { name: 'seed' })).toBeTruthy()
    expect(screen.queryByRole('rowheader', { name: 'steps' })).toBeNull()
  })
  it('says when a file is gone', async () => {
    render(ComparePage, { props: { names: ['wf/r1/a.png', 'wf/gone.png'] } })
    expect(await screen.findByText(/no longer in the gallery/)).toBeTruthy()
  })
})
