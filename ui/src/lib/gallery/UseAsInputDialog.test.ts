import { afterEach, expect, it, vi } from 'vitest'
import { flushSync } from 'svelte'
import { cleanup, render, screen, waitFor } from '@testing-library/svelte'
import UseAsInput from './UseAsInput.svelte'
import { api } from '../test/apiMock'
import type { MediaFile } from '../media'

vi.mock('../api', async () => (await import('../test/apiMock')).apiModule())
vi.mock('../changes.svelte', async () => ({
  changes: (await import('../test/feedMock')).feed.changes,
}))

const file = {
  name: 'run/cat.png',
  kind: 'image',
  url: '/x',
  label: 'cat.png',
  workspace: 'default',
} as MediaFile

afterEach(cleanup)

const show = () => {
  const { component } = render(UseAsInput)
  component.show(file)
  flushSync()
}

it('says so plainly when no workflow takes the kind', async () => {
  api.listWorkflows.mockResolvedValue({
    workflows: ['mine/text'],
    details: { 'mine/text': { variable_names: ['prompt'], writable: true } },
  } as never)
  show()
  await waitFor(() =>
    expect(
      screen.getByText(/No workflow in this workspace takes an image\./),
    ).toBeTruthy(),
  )
  expect(
    screen.getByRole('dialog', { name: 'Use cat.png as input' }),
  ).toBeTruthy()
})

it('lists a button per workflow that takes it', async () => {
  api.listWorkflows.mockResolvedValue({
    workflows: ['mine/edit'],
    details: { 'mine/edit': { variable_names: ['image'], writable: true } },
  } as never)
  show()
  await waitFor(() =>
    expect(screen.getByRole('button', { name: 'edit' })).toBeTruthy(),
  )
})

it('two same-leaf workflows are told apart', async () => {
  api.listWorkflows.mockResolvedValue({
    workflows: ['people/edit', 'pets/edit', 'solo'],
    details: {
      'people/edit': { variable_names: ['image'], writable: true },
      'pets/edit': { variable_names: ['image'], writable: true },
      'mine/solo': { variable_names: ['image'], writable: true },
    },
  } as never)
  show()
  await waitFor(() =>
    expect(screen.getByRole('button', { name: 'people/edit' })).toBeTruthy(),
  )
  expect(screen.getByRole('button', { name: 'pets/edit' })).toBeTruthy()
  expect(screen.getByRole('button', { name: 'solo' })).toBeTruthy()
})
