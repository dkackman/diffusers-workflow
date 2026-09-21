import { cleanup, render, screen } from '@testing-library/svelte'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import Breadcrumb from './Breadcrumb.svelte'

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
    ['overview', '#/ws/studio/overview'],
  ])
})

it('a workspace section crumbs as workspace / section', () => {
  at('#/ws/studio/jobs/abc')
  render(Breadcrumb)
  expect(links()).toEqual([
    ['studio', '#/ws/studio/overview'],
    ['jobs', '#/ws/studio/jobs'],
  ])
})

it('a shared route crumbs as shared / section', () => {
  at('#/shared/prompt-edit/x')
  render(Breadcrumb)
  expect(links()).toEqual([
    ['shared', '#/shared/prompts'],
    ['prompt-edit', '#/shared/prompt-edit'],
  ])
})

it('a server route crumbs as server / section', () => {
  at('#/server/models')
  render(Breadcrumb)
  expect(links()).toEqual([
    ['server', '#/server/status'],
    ['models', '#/server/models'],
  ])
})
