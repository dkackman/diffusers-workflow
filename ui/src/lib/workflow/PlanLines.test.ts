import { cleanup, render } from '@testing-library/svelte'
import { afterEach, expect, it } from 'vitest'
import PlanLines from './PlanLines.svelte'
import type { Plan } from '../types'

afterEach(cleanup)

const plan = {
  fingerprint: 'sha256:abc',
  steps: 2,
  list_entries: {},
  cached_steps: null,
  elided_steps: [],
  downloads_required: [
    { repo: 'org/m', gb: 4.2, gated: false, access_blocked: false },
  ],
  estimate: {
    minutes: 3,
    basis: 'catalog',
    device: 'cuda',
    measured_on: null,
    partial: false,
    unpriced: [],
    runs: null,
    cached_minutes: 3,
  },
  workspace: 'default',
} as Plan

it('shows the summary and keeps the lines behind What a job will do', () => {
  const { container } = render(PlanLines, { props: { plan } })
  expect(container.querySelector('p')?.textContent).toBe(
    'about 3 min · downloads 4.2 GB first',
  )
  const details = container.querySelector('details') as HTMLDetailsElement
  expect(details.open).toBe(false)
  expect(details.querySelector('summary')?.textContent).toBe(
    'What a job will do',
  )
  const lines = details.querySelector('pre')?.textContent ?? ''
  expect(lines).toContain('2 steps')
  expect(lines).toContain('needs download: org/m (4.2 GB)')
  expect(lines.split('\n').length).toBeGreaterThan(2)
})
