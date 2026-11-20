import { expect, it } from 'vitest'
import { attempt, doneItem, item, request } from './fixtures'
import { planState } from './itemState'
import { itemMenu, requestMenu, type MenuEntry } from './menuEntries'

const context = {
  total: 3,
  jobsHref: '#jobs',
  outputsHref: '#outputs',
  canRun: true,
}
const labels = (entries: MenuEntry[]) =>
  entries.map((e) => (e.kind === 'rule' ? '-' : e.label))
const menu = (key: string, over = {}) => {
  const plan = planState(
    request([
      doneItem('a'),
      item('b', { ready: true, depends_on: ['a'] }),
      item('c', {
        status: 'failed',
        attempts: [attempt('jc', 'failed')],
        comment: { text: 'n', at: 't' },
      }),
      item('x', {
        dropped: true,
        status: 'dropped',
        attempts: [attempt('jx', 'failed')],
      }),
      item('y', { dropped: true, status: 'dropped' }),
    ]),
  )
  const row = plan.items.find((s) => s.item.key === key)
  if (!row) throw new Error(key)
  return itemMenu(row, { ...context, ...over })
}

it('a ready item can be moved, noted and dropped', () => {
  expect(labels(menu('b'))).toEqual([
    'Move up',
    'Move down',
    'Move to top',
    'Move to bottom',
    '-',
    'Leave a note for the agent',
    '-',
    'Drop from plan',
  ])
  const drop = menu('b').at(-1)
  expect(drop).toMatchObject({ id: 'drop', danger: true })
})

it('a done item leads with another take, then where its jobs and outputs are', () => {
  expect(labels(menu('a')).slice(0, 5)).toEqual([
    'Run another take…',
    '-',
    'Its jobs',
    'Its outputs',
    '-',
  ])
  expect(menu('a')[2]).toMatchObject({ kind: 'link', href: '#jobs' })
  // a page that cannot queue jobs does not offer one
  expect(labels(menu('a', { canRun: false }))[0]).toBe('Its jobs')
})

it('a failed item links its jobs but has no outputs, and says Edit for a note it holds', () => {
  const entries = labels(menu('c'))
  expect(entries[0]).toBe('Its jobs')
  expect(entries).not.toContain('Its outputs')
  expect(entries).toContain('Edit the note')
})

it('the moves that would go nowhere are disabled', () => {
  const off = (key: string) =>
    menu(key)
      .filter((e) => e.kind === 'action' && e.disabled)
      .map((e) => (e.kind === 'action' ? e.id : ''))
  expect(off('a')).toEqual(['up', 'top'])
  expect(off('b')).toEqual([])
  expect(off('c')).toEqual(['down', 'bottom'])
})

it('a dropped item keeps only its links, and one with none has no menu', () => {
  expect(labels(menu('x'))).toEqual(['Its jobs'])
  expect(menu('y')).toEqual([])
})

it('an open request can be closed either way, and a closed one reopened', () => {
  expect(labels(requestMenu('open'))).toEqual(['Mark done', 'Abandon'])
  expect(labels(requestMenu('done'))).toEqual(['Reopen'])
  expect(labels(requestMenu('abandoned'))).toEqual(['Reopen'])
})
