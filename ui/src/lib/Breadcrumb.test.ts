import { cleanup, render, screen } from '@testing-library/svelte'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import Breadcrumb from './Breadcrumb.svelte'
import { SECTION_LABELS } from './routes'

vi.mock('./api', () => ({
  api: { listWorkspaces: vi.fn(() => new Promise(() => {})) },
}))
vi.mock('./toast', () => ({ notify: { error: vi.fn(), success: vi.fn() } }))

// The component's router parsed the hash at load; drive it the way the
// browser does
function at(hash: string) {
  location.hash = hash
  window.dispatchEvent(new HashChangeEvent('hashchange'))
}

beforeEach(() => localStorage.clear())
afterEach(cleanup)

const links = () =>
  screen
    .getAllByRole('link')
    .map((a) => [a.textContent, a.getAttribute('href')])

it('an overview route renders both crumbs even though they share an href', () => {
  at('#/ws/studio/overview')
  render(Breadcrumb)
  expect(links()).toEqual([
    ['studio', '#/ws/studio/overview'],
    ['Overview', '#/ws/studio/overview'],
  ])
})

it('a workspace section crumbs as workspace / section', () => {
  at('#/ws/studio/jobs/abc')
  render(Breadcrumb)
  expect(links()).toEqual([
    ['studio', '#/ws/studio/overview'],
    ['Jobs', '#/ws/studio/jobs'],
  ])
})

it('a shared route crumbs as shared / section', () => {
  at('#/shared/prompt-edit/x')
  render(Breadcrumb)
  expect(links()).toEqual([
    ['Shared', '#/shared/prompts'],
    ['Prompt editor', '#/shared/prompt-edit'],
  ])
})

it('a server route crumbs as server / section', () => {
  at('#/server/models')
  render(Breadcrumb)
  expect(links()).toEqual([
    ['Server', '#/server/status'],
    ['Models', '#/server/models'],
  ])
})

it("shows the sidebar's labels", () => {
  at('#/shared/prompt-edit/x')
  render(Breadcrumb)
  expect(links().map(([label]) => label)).toEqual(['Shared', 'Prompt editor'])
  expect(SECTION_LABELS.shared['prompt-edit']).toBe('Prompt editor')
})

it('the not-found crumb is not a link', () => {
  at('#/ws/studio/nope')
  render(Breadcrumb)
  expect(screen.queryAllByRole('link')).toEqual([])
  expect(screen.getByText('Page not found').tagName).toBe('SPAN')
})
