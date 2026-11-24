import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen } from '@testing-library/svelte'
import Harness from './LorasEditorHarness.svelte'
import type { LoraList, LoraRow } from '../types'
import { feed } from '../test/feedMock'
import { workspace } from '../workspaceState.svelte'

// The spy only records calls: vitest's own tracking of a returned rejected
// promise would report it as unhandled, so the result comes from `reply`
const listLoras = vi.hoisted(() =>
  vi.fn<(q: { model: string; workflow?: string }) => void>(),
)
const reply = vi.hoisted(() => ({
  next: (() =>
    Promise.resolve({ libraries: [], loras: [] })) as () => Promise<unknown>,
}))
vi.mock('../api', () => ({
  api: {
    listLoras: (q: { model: string; workflow?: string }) => {
      listLoras(q)
      return reply.next()
    },
  },
}))

vi.mock('../changes.svelte', async () => ({
  changes: (await import('../test/feedMock')).feed.changes,
}))

beforeEach(() => listLoras.mockReset())
afterEach(() => cleanup())

const row = (over: Partial<LoraRow> = {}): LoraRow =>
  ({
    name: 'a-lora',
    origin: 'builtin',
    writable: false,
    model_name: 'org/a-lora',
    base_models: ['org/model'],
    description: 'does a thing',
    use_when: 'you want a thing',
    status: 'proven',
    ...over,
  }) as LoraRow

const respond = (...loras: LoraRow[]) => {
  reply.next = () => Promise.resolve({ libraries: [], loras } as LoraList)
}

const pipe = (
  from: Record<string, any> = { model_name: 'org/model' },
  loras?: Record<string, any>[],
) => ({
  from_pretrained_arguments: from,
  ...(loras ? { loras } : {}),
})

function mount(pipeline: Record<string, any>) {
  const latest = { value: pipeline }
  render(Harness, {
    initial: pipeline,
    onchange: (p: Record<string, any>) => (latest.value = p),
  })
  return latest
}

it('fetches suggestions for a literal repo', async () => {
  respond(row())
  mount(pipe())
  await screen.findByText('a-lora')
  expect(listLoras).toHaveBeenCalledWith({ model: 'org/model' })
})

it('adds a literal workflow partition to the fetch', async () => {
  respond(row())
  mount(pipe({ model_name: 'org/model', workflow: 't2va' }))
  await screen.findByText('a-lora')
  expect(listLoras).toHaveBeenCalledWith({
    model: 'org/model',
    workflow: 't2va',
  })
})

it('does not fetch for a variable base', async () => {
  vi.useFakeTimers()
  try {
    mount(pipe({ model_name: 'variable:base' }))
    await vi.advanceTimersByTimeAsync(1000)
    expect(listLoras).not.toHaveBeenCalled()
    expect(
      screen.getByText('Catalog suggestions need a literal model repo.'),
    ).toBeTruthy()
  } finally {
    vi.useRealTimers()
  }
})

it('hides rejected rows and lists proven before trial', async () => {
  respond(
    row({ name: 'a-trial', status: 'trial' }),
    row({ name: 'b-rejected', status: 'rejected' }),
    row({ name: 'c-proven', status: 'proven' }),
  )
  mount(pipe())
  await screen.findByText('c-proven')
  expect(screen.queryByText('b-rejected')).toBeNull()
  const names = Array.from(document.querySelectorAll('.suggestion .name')).map(
    (n) => n.textContent,
  )
  expect(names).toEqual(['c-proven', 'a-trial'])
})

it('adds a full entry with its default scale', async () => {
  respond(
    row({
      weight_name: 'sub/x.safetensors',
      revision: 'abc123',
      scale: { default: 0.8 },
    }),
  )
  const latest = mount(pipe())
  await fireEvent.click(await screen.findByLabelText('add a-lora'))
  expect(latest.value.loras).toEqual([
    {
      model_name: 'org/a-lora',
      weight_name: 'sub/x.safetensors',
      revision: 'abc123',
      scale: 0.8,
    },
  ])
})

it('adds a bare entry with scale 1 and no weight_name or revision', async () => {
  respond(row())
  const latest = mount(pipe())
  await fireEvent.click(await screen.findByLabelText('add a-lora'))
  expect(latest.value.loras).toEqual([{ model_name: 'org/a-lora', scale: 1 }])
  expect('weight_name' in latest.value.loras[0]).toBe(false)
  expect('revision' in latest.value.loras[0]).toBe(false)
})

it('marks an entry already on the step as added and disables it', async () => {
  respond(row({ weight_name: 'x.safetensors' }))
  mount(
    pipe({ model_name: 'org/model' }, [
      { model_name: 'org/a-lora', weight_name: 'x.safetensors', scale: 1 },
    ]),
  )
  const button = (await screen.findByLabelText(
    'a-lora added',
  )) as HTMLButtonElement
  expect(button.disabled).toBe(true)
  expect(button.textContent).toContain('added')
})

it('shows a muted line when the catalog fails and keeps add LoRA working', async () => {
  reply.next = () => Promise.reject(new Error('boom'))
  const latest = mount(pipe())
  await screen.findByText('Catalog suggestions are unavailable right now.')
  await fireEvent.click(screen.getByRole('button', { name: /add LoRA/ }))
  expect(latest.value.loras).toEqual([{ model_name: '', scale: 1 }])
})

it('shows a loading line while the catalog fetch is pending', async () => {
  reply.next = () => new Promise(() => {})
  mount(pipe())
  expect(await screen.findByText('Loading catalog suggestions…')).toBeTruthy()
})

it('treats a fetch that never settles as unavailable (#1501)', async () => {
  vi.useFakeTimers()
  try {
    reply.next = () => new Promise(() => {})
    mount(pipe())
    await vi.advanceTimersByTimeAsync(1000)
    expect(listLoras).toHaveBeenCalledTimes(1)
    expect(screen.getByText('Loading catalog suggestions…')).toBeTruthy()
    await vi.advanceTimersByTimeAsync(10_000)
    expect(
      screen.getByText('Catalog suggestions are unavailable right now.'),
    ).toBeTruthy()
  } finally {
    vi.useRealTimers()
  }
})

it('shows an entry partition only when the step has none', async () => {
  respond(
    row({ workflow: 't2va' }),
    row({ name: 'b', workflow: ['i2v', 't2v'] }),
  )
  mount(pipe())
  await screen.findByText('workflow: t2va')
  expect(screen.getByText('workflow: i2v, t2v')).toBeTruthy()
  cleanup()

  listLoras.mockClear()
  respond(row({ workflow: 't2va' }))
  mount(pipe({ model_name: 'org/model', workflow: 't2va' }))
  await screen.findByText('a-lora')
  expect(screen.queryByText('workflow: t2va')).toBeNull()
})

it('says so when the catalog has no matches', async () => {
  respond()
  mount(pipe())
  await screen.findByText('No catalog LoRAs for org/model.')
})

it('shows an existing row revision read-only', async () => {
  respond()
  mount(
    pipe({ model_name: 'org/model' }, [
      { model_name: 'org/x', revision: 'deadbeef', scale: 1 },
    ]),
  )
  const code = await screen.findByText('deadbeef')
  expect(code.tagName).toBe('CODE')
  expect(screen.queryByDisplayValue('deadbeef')).toBeNull()
})

it('LorasEditor shows a LoRA saved elsewhere', async () => {
  respond(row())
  mount(pipe())
  await screen.findByText('a-lora')
  expect(feed.watched(workspace.current)).toContain('loras')
  respond(row(), row({ name: 'b-saved', model_name: 'org/b-saved' }))
  feed.move(workspace.current, 'loras')
  await screen.findByText('b-saved')
  expect(screen.getByText('a-lora')).toBeTruthy()
  expect(screen.queryByText('Loading catalog suggestions…')).toBeNull()
})

it('every LoRA field has a label', async () => {
  respond()
  mount(pipe({ model_name: 'org/model' }, [{ model_name: 'org/x', scale: 1 }]))
  expect(
    (await screen.findByLabelText('Model')).getAttribute('placeholder'),
  ).toBe('org/lora-repo or path')
  expect(screen.getByLabelText('Weight file')).toBeTruthy()
  expect(screen.getByLabelText('Scale')).toBeTruthy()
})

it('remove names the LoRA', async () => {
  respond()
  mount(
    pipe({ model_name: 'org/model' }, [
      { model_name: 'org/x', scale: 1 },
      { model_name: '', scale: 1 },
    ]),
  )
  expect(
    await screen.findByRole('button', { name: 'remove org/x' }),
  ).toBeTruthy()
  expect(screen.getByRole('button', { name: 'remove this LoRA' })).toBeTruthy()
})
