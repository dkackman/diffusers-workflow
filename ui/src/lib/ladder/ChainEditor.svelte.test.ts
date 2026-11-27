import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from '@testing-library/svelte'
import { afterEach, expect, it } from 'vitest'
import ChainEditor from './ChainEditor.svelte'
import type { Ladder } from './ladderDraft'
import type { PreferenceCheck } from '../types'

afterEach(() => {
  cleanup()
  document.body.innerHTML = ''
})

/** The editor over a draft the test reads back after each edit */
function edit(
  first: Ladder,
  over: {
    answer?: PreferenceCheck | null
    base?: Ladder
    contract?: { required: string[]; optional: string[] }
  } = {},
) {
  const props = $state({
    draft: first,
    catalog: ['x', 'models/z-image'],
    answer: over.answer ?? null,
    base: over.base,
    contract: over.contract,
  })
  render(ChainEditor, { props })
  return props
}
const tier = (name: string) =>
  document.querySelector<HTMLElement>(`.tier[data-tier="${name}"]`)!
const chip = (tierName: string, name: string) =>
  [...tier(tierName).querySelectorAll<HTMLElement>('li.target')].find(
    (li) => li.querySelector('code')?.textContent === name,
  )!

it('Alt+ArrowRight on the first chip moves it after the second', async () => {
  const props = edit({ draft: [], standard: [], best: ['a', 'b'] })
  await fireEvent.keyDown(chip('best', 'a'), {
    key: 'ArrowRight',
    altKey: true,
  })
  expect(props.draft.best).toEqual(['b', 'a'])
  await fireEvent.keyDown(chip('best', 'a'), { key: 'ArrowLeft', altKey: true })
  expect(props.draft.best).toEqual(['a', 'b'])
  // without Alt the arrow keys are left alone
  await fireEvent.keyDown(chip('best', 'a'), { key: 'ArrowRight' })
  expect(props.draft.best).toEqual(['a', 'b'])
})

it('dragging a chip onto another moves it there', async () => {
  const props = edit({ best: ['a', 'b', 'c'] })
  await fireEvent.dragStart(chip('best', 'c'))
  await fireEvent.dragOver(chip('best', 'a'))
  await fireEvent.drop(chip('best', 'a'))
  expect(props.draft.best).toEqual(['c', 'a', 'b'])
})

it('× removes a chip', async () => {
  const props = edit({ best: ['a', 'b'] })
  await fireEvent.click(screen.getByRole('button', { name: 'Remove a' }))
  expect(props.draft.best).toEqual(['b'])
})

it('the add popover appends a target limited to the devices ticked', async () => {
  const props = edit({ best: ['a'] })
  await fireEvent.click(
    screen.getByRole('button', { name: 'Add a model to best' }),
  )
  const pop = await waitFor(() => {
    const el = document.querySelector<HTMLElement>('[data-popover-content]')
    expect(el).not.toBeNull()
    return el!
  })
  expect(pop.textContent).toContain(
    'Only on these devices — leave empty for any',
  )
  await fireEvent.input(within(pop).getByLabelText('workflow to add to best'), {
    target: { value: 'x' },
  })
  await fireEvent.click(within(pop).getByLabelText('cuda'))
  await fireEvent.click(within(pop).getByText('Add'))
  expect(props.draft.best).toEqual(['a', { workflow: 'x', devices: ['cuda'] }])
})

it("marks each chip with the dry run's answer for it", () => {
  edit(
    { best: ['m/skip', 'a', 'gone/x', 'b', 'w'] },
    {
      base: { best: ['m/skip', 'a', 'gone/x', 'b', 'w'] },
      answer: {
        tiers: {
          best: {
            target: 'a',
            skipped: [{ workflow: 'm/skip', reason: 'cuda-only' }],
            after: ['gone/x', 'b', 'w'],
            target_errors: { 'gone/x': 'not in the catalog' },
            warnings: { w: ['takes no argument prompt'] },
          },
        },
      },
    },
  )
  const skipped = chip('best', 'm/skip')
  expect(skipped.querySelector('s')?.textContent).toBe('m/skip')
  expect(skipped.textContent).toContain('needs CUDA')
  expect(chip('best', 'a').textContent).toContain('runs here')
  expect(chip('best', 'a').classList.contains('good')).toBe(true)
  // a model already in the list is not "checked on save"
  expect(chip('best', 'a').textContent).not.toContain('access checked on save')
  expect(chip('best', 'gone/x').classList.contains('bad')).toBe(true)
  expect(chip('best', 'gone/x').textContent).toContain('not in the catalog')
  expect(chip('best', 'b').textContent).toContain('fallback, not checked here')
  expect(chip('best', 'w').classList.contains('warn')).toBe(true)
  expect(chip('best', 'w').textContent).toContain('takes no argument prompt')
})

it('a model added in this edit that runs here says access is checked on save', () => {
  edit(
    { best: ['new'] },
    {
      base: { best: ['old'] },
      answer: { tiers: { best: { target: 'new', skipped: [] } } },
    },
  )
  expect(chip('best', 'new').textContent).toContain('access checked on save')
})

it('an emptied tier says its requests are refused', async () => {
  const props = edit({ draft: ['a'], best: ['b'] })
  await fireEvent.click(screen.getByRole('button', { name: 'Remove a' }))
  expect(props.draft.draft).toEqual([])
  expect(tier('draft').textContent).toContain(
    'No models: requests at this tier are refused',
  )
})

it("names the key's contract and each chip's arguments", () => {
  edit(
    { best: [{ workflow: 'a', arguments: { steps: 8, cfg: 1 } }] },
    { contract: { required: ['prompt', 'seed'], optional: ['width'] } },
  )
  expect(
    screen.getByText('Every model here must take: prompt, seed'),
  ).toBeTruthy()
  expect(chip('best', 'a').textContent).toContain('with steps, cfg')
})
