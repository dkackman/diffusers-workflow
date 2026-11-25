import { expect, it } from 'vitest'
import { FIRST_RUN_WORKFLOW, firstRunLink } from './firstRun'
import type { WorkflowList } from './types'

const listing = (workflows: string[], readOnly: string[] = []): WorkflowList =>
  ({
    libraries: [],
    shadowed: [],
    workflows,
    details: Object.fromEntries(
      workflows.map((w) => [w, { writable: !readOnly.includes(w) }]),
    ),
  }) as unknown as WorkflowList

it('the card finds the template where the listing has it', () => {
  expect(firstRunLink(listing([FIRST_RUN_WORKFLOW]), 'studio')).toEqual({
    href: '#/ws/studio/workflows/templates/text-to-image',
    label: 'Open text-to-image',
  })
  expect(
    firstRunLink(listing([FIRST_RUN_WORKFLOW], [FIRST_RUN_WORKFLOW]), 'studio'),
  ).toEqual({
    href: '#/shared/examples/templates/text-to-image',
    label: 'Open text-to-image',
  })
})

it('no template: the examples', () => {
  expect(firstRunLink(listing(['a', 'b'], ['b']), 'studio')).toEqual({
    href: '#/shared/examples',
    label: 'Browse examples',
  })
})

it('no examples: the workflows', () => {
  expect(firstRunLink(listing(['a']), 'studio')).toEqual({
    href: '#/ws/studio/workflows',
    label: 'Open Workflows',
  })
})

it('a failed listing still orients', () => {
  expect(firstRunLink(null, 'studio')).toEqual({
    href: '#/ws/studio/workflows',
    label: 'Open Workflows',
  })
})
