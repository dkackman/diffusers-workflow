import { cleanup, render, screen } from '@testing-library/svelte'
import { afterEach, expect, it } from 'vitest'
import RemoteNotice from './RemoteNotice.svelte'

afterEach(cleanup)

const COPY = 'Updated by an agent or another tab'

it('says nothing until the view says its data changed', () => {
  render(RemoteNotice, { shown: false })
  expect(screen.queryByText(COPY)).toBeNull()
})

it('shows the editors’ copy as a status wearing the changed-row bar', () => {
  render(RemoteNotice, { shown: true })
  const notice = screen.getByRole('status')
  expect(notice.textContent?.trim()).toBe(COPY)
  expect(notice.tagName).toBe('P')
  expect(notice.classList.contains('row-changed')).toBe(true)
})

it('sits inline in a heading row when asked', () => {
  render(RemoteNotice, { shown: true, inline: true })
  expect(screen.getByRole('status').tagName).toBe('SPAN')
})

it('is the one inline notice: no view spells its own', () => {
  const views = import.meta.glob<string>(['../**/*.svelte'], {
    query: '?raw',
    import: 'default',
    eager: true,
  })
  const own = Object.entries(views)
    .filter(([path]) => !path.endsWith('/RemoteNotice.svelte'))
    .filter(([, source]) => /REMOTE_RELOADED|remotenote/.test(source))
    .map(([path]) => path)
  expect(own).toEqual([])
  const using = Object.entries(views)
    .filter(([, source]) => source.includes('<RemoteNotice'))
    .map(([path]) => path)
  expect(using).toEqual([
    '../pages/JobPage.svelte',
    '../pages/WorkflowPage.svelte',
  ])
})
