import { cleanup, fireEvent, render, screen } from '@testing-library/svelte'
import { afterEach, expect, it, vi } from 'vitest'
import ItemMenu from './ItemMenu.svelte'
import type { MenuEntry } from './menuEntries'

afterEach(() => {
  cleanup()
  document.body.innerHTML = ''
})

const entries: MenuEntry[] = [
  { kind: 'link', id: 'jobs', label: 'Its jobs', href: '#/jobs?req=r1/a' },
  { kind: 'rule' },
  { kind: 'action', id: 'up', label: 'Move up', disabled: true },
  { kind: 'action', id: 'down', label: 'Move down' },
  { kind: 'rule' },
  { kind: 'action', id: 'drop', label: 'Drop from plan', danger: true },
]
const setup = () => {
  const onselect = vi.fn()
  render(ItemMenu, { label: 'More actions for Shot 3', entries, onselect })
  const trigger = screen.getByRole('button', {
    name: 'More actions for Shot 3',
  })
  return { onselect, trigger }
}

it('is a named button that says whether its list is open', async () => {
  const { trigger } = setup()
  expect(trigger.getAttribute('aria-expanded')).toBe('false')
  expect(screen.queryByRole('dialog')).toBeNull()
  await fireEvent.click(trigger)
  expect(trigger.getAttribute('aria-expanded')).toBe('true')
  const list = await screen.findByRole('dialog', {
    name: 'More actions for Shot 3',
  })
  expect(
    [...list.querySelectorAll('button, a')].map((e) => e.textContent?.trim()),
  ).toEqual(['Its jobs', 'Move up', 'Move down', 'Drop from plan'])
  expect(list.querySelectorAll('[role="separator"]')).toHaveLength(2)
})

it('choosing an action hands its id up and closes the list', async () => {
  const { onselect, trigger } = setup()
  await fireEvent.click(trigger)
  await fireEvent.click(
    await screen.findByRole('button', { name: 'Move down' }),
  )
  expect(onselect).toHaveBeenCalledWith('down')
  expect(trigger.getAttribute('aria-expanded')).toBe('false')
})

it('a move that would go nowhere cannot be chosen, and a link is a link', async () => {
  const { onselect, trigger } = setup()
  await fireEvent.click(trigger)
  const up = (await screen.findByRole('button', {
    name: 'Move up',
  })) as HTMLButtonElement
  expect(up.disabled).toBe(true)
  expect(
    screen.getByRole('link', { name: 'Its jobs' }).getAttribute('href'),
  ).toBe('#/jobs?req=r1/a')
  expect(
    screen.getByRole('button', { name: 'Drop from plan' }).className,
  ).toContain('danger')
  expect(onselect).not.toHaveBeenCalled()
})
