import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from '@testing-library/svelte'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import FirstRunCard from './FirstRunCard.svelte'
import { FIRST_RUN_KEY, firstRunSession } from './firstRun'
import { storageGet } from './storage'

const link = {
  href: '#/ws/a/workflows/templates/text-to-image',
  label: 'Open text-to-image' as const,
}
const TITLE = 'New here? Make your first picture'

beforeEach(() => {
  localStorage.clear()
  firstRunSession.dismissed = false
})
afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

it('the card says what dw is and how to start', () => {
  render(FirstRunCard, { link })
  const card = screen.getByRole('region', { name: TITLE })
  expect(card.textContent?.replace(/\s+/g, ' ')).toContain(
    "dw runs image, video and audio workflows on this server's GPU. Pick a workflow, press Run, and what it makes lands in the Gallery.",
  )
  const steps = [...card.querySelectorAll('ol li')].map((li) =>
    li.textContent?.replace(/\s+/g, ' ').trim(),
  )
  expect(steps).toEqual([
    'Open text-to-image. It uses a small model that needs no Hugging Face login.',
    'Type a prompt and press Run.',
    'Watch the job, then find the picture in the Gallery.',
    "On the finished job's page, Make a better one opens the same prompt on the model this server prefers.",
  ])
  expect(
    screen
      .getByRole('link', { name: 'Open text-to-image' })
      .getAttribute('href'),
  ).toBe(link.href)
  expect(card.textContent).toContain('Press ? for keyboard shortcuts.')
})

it('Got it hides it and remembers', async () => {
  render(FirstRunCard, { link })
  await fireEvent.click(
    screen.getByRole('button', { name: 'Got it, dismiss this introduction' }),
  )
  expect(screen.queryByRole('region')).toBeNull()
  expect(storageGet(FIRST_RUN_KEY, false)).toBe(true)
})

it('a dismissal that cannot be stored lasts the session', async () => {
  vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => {
    throw new Error('blocked')
  })
  vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
    throw new Error('blocked')
  })
  render(FirstRunCard, { link })
  expect(screen.getByRole('region', { name: TITLE })).toBeTruthy()
  await fireEvent.click(
    screen.getByRole('button', { name: 'Got it, dismiss this introduction' }),
  )
  expect(screen.queryByRole('region')).toBeNull()
  // The Overview remounts the card on every visit: the next one stays hidden
  cleanup()
  render(FirstRunCard, { link })
  expect(screen.queryByRole('region')).toBeNull()
})

it("Got it's accessible name starts with its visible text", () => {
  render(FirstRunCard, { link })
  const button = screen.getByRole('button', { name: /^Got it\b/ })
  expect(button.textContent?.trim()).toBe('Got it')
})

it('a dismissed card stays hidden', () => {
  localStorage.setItem('dw-' + FIRST_RUN_KEY, 'true')
  render(FirstRunCard, { link })
  expect(screen.queryByRole('region')).toBeNull()
})

it('Got it moves focus to the page heading, not the body', async () => {
  const main = document.createElement('main')
  main.innerHTML = '<h1 tabindex="-1">studio</h1>'
  document.body.append(main)
  try {
    render(FirstRunCard, { link })
    const button = screen.getByRole('button', {
      name: 'Got it, dismiss this introduction',
    })
    button.focus()
    await fireEvent.click(button)
    await waitFor(() =>
      expect(document.activeElement).toBe(main.querySelector('h1')),
    )
  } finally {
    main.remove()
  }
})
