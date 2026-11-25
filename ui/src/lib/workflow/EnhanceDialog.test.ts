// @vitest-environment jsdom
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from '@testing-library/svelte'
import EnhanceDialog from './EnhanceDialog.svelte'
import { api } from '../test/apiMock'

vi.mock('../toast', () => ({ notify: { error: vi.fn(), success: vi.fn() } }))
vi.mock('../api', async () =>
  (await import('../test/apiMock')).apiModule({
    fetchOutputText: vi.fn().mockResolvedValue('A long, lit prompt'),
    streamJobEvents: vi.fn(
      (_id: string, _from: number, _on: unknown, done: () => void) => {
        queueMicrotask(done)
        return () => {}
      },
    ),
  }),
)
api.listModels.mockResolvedValue({ repos: [] })

beforeEach(() => {
  api.listEnhancers.mockResolvedValue({
    presets: [
      {
        key: 'p',
        label: 'Preset',
        default_model: 'org/m',
        models: [],
        placeholder: '',
      },
    ],
  } as never)
  api.enhance.mockResolvedValue({
    id: 'j1',
    workspace: 'w',
  } as never)
  api.getJob.mockResolvedValue({
    status: 'succeeded',
    workspace: 'w',
    manifest: [{ files: ['o.txt'] }],
  } as never)
})
afterEach(cleanup)

it("opens with the field's text as the idea", async () => {
  render(EnhanceDialog, {
    props: { open: true, idea: 'a cat', onuse: vi.fn() },
  })
  const dialog = await screen.findByRole('dialog', {
    name: 'Enhance the prompt',
  })
  const box = dialog.querySelector<HTMLTextAreaElement>('#enhance-idea')
  await waitFor(() => expect(box?.value).toBe('a cat'))
})

it('Use this prompt writes the result back and closes', async () => {
  const onuse = vi.fn()
  render(EnhanceDialog, { props: { open: true, idea: 'a cat', onuse } })
  await screen.findByRole('dialog', { name: 'Enhance the prompt' })
  await waitFor(() => expect(api.listEnhancers).toHaveBeenCalled())
  const generate = await screen.findByRole('button', { name: /Generate/ })
  await waitFor(() =>
    expect((generate as HTMLButtonElement).disabled).toBe(false),
  )
  await fireEvent.click(generate)
  await fireEvent.click(
    await screen.findByRole('button', { name: 'Use this prompt' }),
  )
  expect(onuse).toHaveBeenCalledWith('A long, lit prompt')
  await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull())
})

it('Close cancels a job that is still running on the server', async () => {
  const { streamJobEvents } = await import('../api')
  vi.mocked(streamJobEvents).mockImplementationOnce(() => () => {})
  api.cancelJob.mockResolvedValue({} as never)
  render(EnhanceDialog, {
    props: { open: true, idea: 'a cat', onuse: vi.fn() },
  })
  await screen.findByRole('dialog', { name: 'Enhance the prompt' })
  const generate = await screen.findByRole('button', { name: /Generate/ })
  await waitFor(() =>
    expect((generate as HTMLButtonElement).disabled).toBe(false),
  )
  await fireEvent.click(generate)
  await waitFor(() => expect(api.enhance).toHaveBeenCalled())
  await fireEvent.keyDown(document.activeElement ?? document.body, {
    key: 'Escape',
  })
  await waitFor(() => expect(api.cancelJob).toHaveBeenCalledWith('j1'))
})
