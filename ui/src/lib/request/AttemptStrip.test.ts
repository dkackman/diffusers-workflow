import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from '@testing-library/svelte'
import { afterEach, expect, it, vi } from 'vitest'
import AttemptStrip from './AttemptStrip.svelte'
import { attempt, doneItem, item } from './fixtures'

afterEach(cleanup)

const strip = (reason?: string) =>
  render(AttemptStrip, {
    item: item('a', {
      status: 'failed',
      attempts: [attempt('j1', 'failed', reason ? { reason } : {})],
    }),
    workspace: 'studio',
  })

it('a failed tile shows its reason', () => {
  const { container } = strip('CUDA out of memory')
  expect(container.querySelector('.reason')?.textContent).toBe(
    'CUDA out of memory',
  )
})

it('a long reason is one line with the whole in its title', () => {
  const long = 'x'.repeat(300)
  const { container } = strip(long)
  const el = container.querySelector('.reason') as HTMLElement
  expect(el.getAttribute('title')).toBe(long)
  expect(el.classList.contains('bad')).toBe(true)
})

it('a failed take with no reason shows no reason line', () => {
  const { container } = strip()
  expect(container.querySelector('.reason')).toBeNull()
})

const out = (name: string) => ({ name, kind: 'image' as const })
const filed = (key: string, accepted = false) =>
  doneItem(key, {
    title: `Item ${key}`,
    attempts: [attempt('j1', 'succeeded', { output: out('run/a.png') })],
    accepted: accepted ? { job: 'j1', run_id: 'r1' } : null,
  })
const flush = () => new Promise((r) => setTimeout(r, 0))
const hashchange = (change: () => void) =>
  new Promise<void>((resolve) => {
    const on = () => {
      window.removeEventListener('hashchange', on)
      void flush().then(() => resolve())
    }
    window.addEventListener('hashchange', on)
    change()
  })

it('Accept is a default button, Accepted is pressed', () => {
  render(AttemptStrip, {
    item: filed('a'),
    workspace: 'studio',
    onaccept: vi.fn(),
  })
  const button = screen.getByRole('button', { name: /^Accept take 1/ })
  expect(button.classList.contains('quiet')).toBe(false)
  expect(button.getAttribute('aria-pressed')).toBe('false')
})

it('the Accepted state shows the check', async () => {
  const { container } = render(AttemptStrip, {
    item: filed('a', true),
    workspace: 'studio',
    onaccept: vi.fn(),
  })
  const button = container.querySelector('button.accept') as HTMLElement
  expect(button.getAttribute('aria-pressed')).toBe('true')
  expect(button.textContent?.trim()).toBe('Accepted')
  expect(button.querySelector('svg')).not.toBeNull()
})

it('a job linked to two items keeps a viewer per strip', async () => {
  history.replaceState(history.state, '', '#/ws/studio/requests/r1')
  render(AttemptStrip, { item: filed('a'), workspace: 'studio' })
  render(AttemptStrip, { item: filed('b'), workspace: 'studio' })
  await hashchange(
    () =>
      void fireEvent.click(
        screen.getByRole('button', { name: 'take 1 of Item b' }),
      ),
  )
  expect(decodeURIComponent(location.hash)).toContain('open=b/j1/run/a.png')
  // one viewer, not one per strip, though both strips hold the same job
  expect(await screen.findAllByRole('dialog')).toHaveLength(1)
})

it("the viewer's history opens and closes with the hash", async () => {
  history.replaceState(history.state, '', '#/ws/studio/requests/r1')
  render(AttemptStrip, { item: filed('a'), workspace: 'studio' })
  await hashchange(
    () =>
      void fireEvent.click(
        screen.getByRole('button', { name: 'take 1 of Item a' }),
      ),
  )
  await screen.findByRole('dialog')
  await hashchange(() => history.back())
  await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull())
  expect(location.hash).toBe('#/ws/studio/requests/r1')
})
