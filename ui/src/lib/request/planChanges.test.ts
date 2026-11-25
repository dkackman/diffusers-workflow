import { cleanup, fireEvent, render, screen } from '@testing-library/svelte'
import { afterEach, expect, it } from 'vitest'
import PlanChanges from './PlanChanges.svelte'
import { entry } from './fixtures'
import { formatExact, formatMtime } from '../format'

afterEach(cleanup)

const revisions = (container: HTMLElement) =>
  [...container.querySelectorAll('ol > li')].map(
    (li) => li.querySelector('.num')?.textContent,
  )

it('lists the newest first, the first five, and the rest on request', async () => {
  const history = Array.from({ length: 8 }, (_, i) =>
    entry(i + 1, 'agent', `change ${i + 1}`),
  )
  const { container } = render(PlanChanges, { history, revision: 8 })
  expect(screen.getByRole('heading', { name: 'Plan changes (8)' })).toBeTruthy()
  expect(container.querySelector('.now')?.textContent).toBe('now at r8')
  expect(revisions(container)).toEqual(['r8', 'r7', 'r6', 'r5', 'r4'])
  await fireEvent.click(screen.getByRole('button', { name: 'show all 8' }))
  expect(revisions(container)).toHaveLength(8)
})

it("folds a run of the user's writes into one line that opens", async () => {
  const { container } = render(PlanChanges, {
    history: [
      entry(1, 'agent', 'opened', { state: 'open', goal: 'Two posters' }),
      entry(2, 'user', 'comment on a'),
      entry(3, 'user', 'comment on b'),
      entry(4, 'agent', 'retook shot 3 warmer'),
    ],
    revision: 4,
  })
  expect(revisions(container)).toEqual(['r4', 'r2–r3', 'r1'])
  expect(screen.getByText('you, 2 edits')).toBeTruthy()
  expect(screen.queryByText('comment on a')).toBeNull()
  await fireEvent.click(screen.getByRole('button', { name: 'show' }))
  expect(screen.getByText('comment on a')).toBeTruthy()
  expect(screen.getByText('comment on b')).toBeTruthy()
  // opening the request is not a change of goal
  expect(screen.queryByText('goal changed:')).toBeNull()
})

it('says a changed goal, and who closed or reopened the request', () => {
  const { container } = render(PlanChanges, {
    history: [
      entry(1, 'agent', 'opened', { state: 'open', goal: 'Two posters' }),
      entry(2, 'agent', 'the ask grew', { goal: 'Three posters' }),
      entry(3, 'user', 'not needed', { state: 'abandoned' }),
      entry(4, 'user', 'needed after all', { state: 'open' }),
    ],
    revision: 4,
  })
  expect(screen.getByText('goal changed:')).toBeTruthy()
  expect(screen.getByText('Three posters')).toBeTruthy()
  expect(
    [...container.querySelectorAll('.state')].map((e) => e.textContent),
  ).toEqual(['reopened', 'abandoned'])
  expect(
    [...container.querySelectorAll('.by')].map((e) => e.textContent),
  ).toEqual(['you', 'you', 'agent', 'agent'])
})

it('each revision says when, exact, with the exact time to the second on hover', () => {
  const first = entry(1, 'agent', 'opened')
  const { container } = render(PlanChanges, { history: [first], revision: 1 })
  const time = container.querySelector('#plan-change-r1 time') as HTMLElement
  const at = Date.parse(first.at) / 1000
  expect(time.textContent?.trim()).toBe(formatMtime(at))
  expect(time.getAttribute('title')).toBe(formatExact(at))
})

it('a revision whose stored time does not parse still renders', () => {
  const { container } = render(PlanChanges, {
    history: [entry(1, 'agent', 'opened', { at: 'legacy' })],
    revision: 1,
  })
  const head = container.querySelector('#plan-change-r1 .head') as HTMLElement
  expect(head.textContent).toContain('unknown time')
  expect(container.querySelector('#plan-change-r1 .why')?.textContent).toBe(
    'opened',
  )
  expect(head.querySelector('[datetime]')).toBeNull()
})
