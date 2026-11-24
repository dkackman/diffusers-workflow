import { afterEach, expect, it, vi } from 'vitest'
import { cleanup, render, screen, waitFor } from '@testing-library/svelte'
import UseAsInput from './UseAsInput.svelte'
import { api } from '../api'
import type { MediaFile } from '../media'

vi.mock('../api', () => ({ api: { listWorkflows: vi.fn() } }))
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

it('says so plainly when no workflow takes the kind', async () => {
  vi.mocked(api.listWorkflows).mockResolvedValue({
    workflows: ['mine/text'],
    details: { 'mine/text': { variable_names: ['prompt'], writable: true } },
  } as never)
  render(UseAsInput, { file, open: true })
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
  vi.mocked(api.listWorkflows).mockResolvedValue({
    workflows: ['mine/edit'],
    details: { 'mine/edit': { variable_names: ['image'], writable: true } },
  } as never)
  render(UseAsInput, { file, open: true })
  await waitFor(() =>
    expect(screen.getByRole('button', { name: 'mine/edit' })).toBeTruthy(),
  )
})
