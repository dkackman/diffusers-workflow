import { cleanup, render, screen, within } from '@testing-library/svelte'
import { afterEach, expect, it } from 'vitest'
import ChainSegments from './ChainSegments.svelte'
import type { JobDetail } from '../types'

afterEach(cleanup)

const m = (
  value: number,
  ceiling: number | null,
  extra: Record<string, unknown> = {},
) => ({
  value,
  ceiling,
  bound: 'max',
  warn_only: false,
  failed: false,
  ...extra,
})

const job = (shots: any[]): JobDetail =>
  ({
    id: 'job-1',
    workspace: 'default',
    arguments: {},
    manifest: [
      { step: 'render', files: ['run/still.png'] },
      { step: 'chain', files: ['run/chain.mp4'], shots },
    ],
  }) as unknown as JobDetail

const definition = {
  steps: [
    {
      name: 'chain',
      pipeline: {
        chain: { check: true, retries: 2 },
        arguments: { frame_rate: 24 },
      },
    },
  ],
}

it('renders one row per segment with verdicts, marks and the planned count', () => {
  render(ChainSegments, {
    job: job([
      {
        name: 'segment 1',
        start_frame: 0,
        num_frames: 97,
        planned_segments: 4,
        resumed: true,
      },
      {
        name: 'segment 2',
        start_frame: 97,
        num_frames: 97,
        planned_segments: 4,
        attempts: 2,
        seed: 1000045,
        measures: {
          jump_ratio: m(0.9, 0.6, { failed: true }),
          texture_ratio: m(1.7, 1.5, { warn_only: true, failed: true }),
          mystery: m(1, 2),
        },
      },
      { name: 'segment 3 (repaired)', start_frame: 194, num_frames: 97 },
    ]),
    definition,
  })
  const panel = screen.getByRole('region', { name: 'Chain segments: chain' })
  expect(within(panel).getByText('3 of 4 planned')).toBeTruthy()
  const rows = panel.querySelectorAll('tbody tr')
  expect(rows).toHaveLength(3)

  expect(rows[0].textContent).toContain('resumed')
  expect(rows[0].textContent).toContain(
    'not recorded - resumed from an earlier run',
  )

  const two = rows[1] as HTMLElement
  expect(two.textContent).toContain('2 of 3')
  expect(two.textContent).toContain('kept seed 1000045')
  expect(two.querySelector('.measure.fail')?.textContent).toContain('0.9 ≤ 0.6')
  expect(two.querySelector('.measure.warn')?.textContent).toContain('warn')
  expect(two.querySelector('.measure.pass')?.textContent).toContain('1 ≤ 2')
  expect(
    within(panel).getByRole('columnheader', { name: 'mystery' }),
  ).toBeTruthy()

  expect(rows[2].textContent).toContain('(repaired)')
  expect(rows[2].textContent).toContain('not recorded')

  // The take, seeked to segment 2's start at 24 fps
  const video = two.querySelector('video')!
  expect(video.getAttribute('src')).toMatch(/chain\.mp4.*#t=4\.042$/)
  expect(video.getAttribute('preload')).toBe('metadata')
})

it('renders nothing for a job with no chained step', () => {
  const { container } = render(ChainSegments, {
    job: job([{ name: 'a', start_frame: 0, num_frames: 10 }]),
    definition: null,
  })
  expect(container.querySelector('section')).toBeNull()
})
