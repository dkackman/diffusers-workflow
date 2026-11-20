import { expect, it } from 'vitest'
import { cardTitle } from './workflowCard'
import type { WorkflowCard } from './types'

const card = (fields: Record<string, unknown>) =>
  fields as unknown as WorkflowCard

it('is the name alone for a card with no detail', () => {
  expect(cardTitle('wf', undefined, 'curated')).toBe('wf')
  expect(cardTitle('wf', card({}), 'curated')).toBe('wf')
})

it('adds the description, the traits and the read-only origin, one per line', () => {
  expect(
    cardTitle(
      'sub/wf',
      card({
        description: 'makes a picture',
        traits: ['has-audio', 'chained'],
        writable: false,
        origin: 'examples',
      }),
      '',
    ),
  ).toBe(
    'sub/wf\nmakes a picture\nhas-audio, chained\nread-only, from the examples directory',
  )
})

it('says whose figure the cost is and the settings it holds for', () => {
  const detail = card({
    cost: [{ minutes: 3, vram_gb: 22, name: 'RTX 4090', device: 'cuda' }],
    cost_drivers: { steps: 30 },
  })
  expect(cardTitle('wf', detail, 'curated')).toBe(
    "wf\na maintainer's figure, measured on RTX 4090 at steps=30",
  )
  expect(cardTitle('wf', detail, '')).toBe(
    'wf\nmeasured on RTX 4090 at steps=30',
  )
})

it("adds this server's own measurement when it has one", () => {
  const title = cardTitle(
    'wf',
    card({ observed: { warm_minutes: 2, warm_runs: 3, runs: 3 } }),
    '',
  )
  expect(title.split('\n')[1]).toMatch(/^measured on this server: /)
})
