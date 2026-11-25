import { afterEach, expect, it } from 'vitest'
import { inputTargets, useAsInput } from './useAsInput'
import { WORKFLOW_PREFILL_KEY } from '../chainActions'
import type { WorkflowCard } from '../types'

const card = (names: string[], writable = true) =>
  ({ variable_names: names, writable }) as unknown as WorkflowCard

afterEach(() => {
  sessionStorage.clear()
  location.hash = ''
})

it("offers workflows with a variable of the file's kind, first variable each", () => {
  const cards = {
    'mine/zoom': card(['prompt', 'input_video']),
    'mine/edit': card(['prompt', 'mask_image', 'image']),
    'mine/both': card(['video', 'image']),
    'mine/text': card(['prompt']),
    'templates/img2img': card(['image'], false),
  }
  expect(inputTargets(cards, 'image')).toEqual([
    { workflow: 'mine/both', variable: 'image' },
    { workflow: 'mine/edit', variable: 'mask_image' },
  ])
  expect(inputTargets(cards, 'video')).toEqual([
    { workflow: 'mine/both', variable: 'video' },
    { workflow: 'mine/zoom', variable: 'input_video' },
  ])
})

it('offers its workflows in natural order: shot2 before shot10', () => {
  const cards = {
    'mine/shot10': card(['image']),
    'mine/shot2': card(['image']),
  }
  expect(inputTargets(cards, 'image').map((t) => t.workflow)).toEqual([
    'mine/shot2',
    'mine/shot10',
  ])
})

it('offers none for audio', () => {
  const cards = { 'mine/edit': card(['image', 'video']) }
  expect(inputTargets(cards, 'audio' as never)).toEqual([])
})

it('writes the prefill and goes to the workflow', () => {
  useAsInput({ workflow: 'mine/edit', variable: 'image' }, 'run/cat.png')
  expect(JSON.parse(sessionStorage.getItem(WORKFLOW_PREFILL_KEY)!)).toEqual({
    workflow: 'mine/edit',
    arguments: { image: 'output:run/cat.png' },
  })
  expect(location.hash).toContain('/workflows/mine/edit')
})
