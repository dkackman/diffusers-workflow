import { cleanup, render, screen, waitFor } from '@testing-library/svelte'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import JobResults from './JobResults.svelte'
import { api } from '../api'
import type { JobDetail } from '../types'

const timeline = {
  kind: 'timeline',
  name: 'cut',
  otio_version: '0.17',
  has_audio_track: false,
  tracks: [
    {
      kind: 'Video',
      name: 'V',
      clip_count: 1,
      seconds: 2,
      items: [
        {
          type: 'Clip',
          name: 'opening',
          start_frame: 0,
          frames: 48,
          rate: 24,
          seconds: 2,
          media: 'media/a.mp4',
          source: 'output:run1/a.mp4',
        },
        {
          type: 'Clip',
          name: 'asset clip',
          start_frame: 0,
          frames: 48,
          rate: 24,
          seconds: 2,
          media: 'media/b.mp4',
          source: null,
        },
      ],
    },
  ],
  members: ['content.otio', 'media/a.mp4', 'media/b.mp4'],
}

vi.mock('../api', () => ({
  api: {
    galleryMetadata: vi.fn((name: string) =>
      Promise.resolve({
        name,
        metadata: null,
        media: timeline,
        inputs: [],
        job: null,
      }),
    ),
    outputDownloadUrl: (name: string) => `/api/gallery/${name}/download`,
  },
  outputUrl: (path: string) => `/outputs/${path}`,
}))

beforeEach(() => vi.mocked(api.galleryMetadata).mockClear())
afterEach(cleanup)

it('fetches an .otioz once and renders its timeline strip', async () => {
  const { container } = render(JobResults, {
    job: { id: 'j1', workspace: 'w' } as JobDetail,
    fileGroups: [
      {
        step: 'join',
        files: ['run1/cut.otioz', 'run1/a.mp4', 'run1/b.mp4'],
        reused: false,
        subfolder: '',
        role: null,
      },
    ],
    unsaved: [],
    kinds: {
      'run1/cut.otioz': null,
      'run1/a.mp4': 'video',
      'run1/b.mp4': 'video',
    },
    seedVariable: null,
  })
  await waitFor(() => expect(screen.getByText('opening · 2 s')).toBeTruthy())
  expect(container.querySelector('[role="group"]')).toBeTruthy()
  const calls = vi.mocked(api.galleryMetadata).mock.calls.map((c) => c[0])
  // Linked by recorded source; b.mp4 shares a job file's name but has none
  expect(
    screen.getByRole('link', { name: 'Download media/a.mp4' }),
  ).toBeTruthy()
  expect(
    screen.queryByRole('link', { name: 'Download media/b.mp4' }),
  ).toBeNull()
  expect(calls).toEqual(['run1/cut.otioz'])
})
