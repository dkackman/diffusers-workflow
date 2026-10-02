import { afterEach, expect, it, vi } from 'vitest'
import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from '@testing-library/svelte'
import Suggest from './Suggest.svelte'

afterEach(() => cleanup())

const suggestions = ['FluxPipeline', 'ZImagePipeline']
// jsdom has no layout: the listbox stays visibility:hidden, so it and its
// options are found by role with `hidden: true`
const options = () =>
  screen
    .queryAllByRole('option', { hidden: true })
    .map((o) => o.textContent?.trim())

it('offers the suggestions that match what was typed', async () => {
  render(Suggest, { props: { id: 'cls', value: '', suggestions } })
  const input = screen.getByRole('combobox')
  await fireEvent.input(input, { target: { value: 'flux' } })
  await waitFor(() => expect(options()).toEqual(['FluxPipeline']))
})

it('takes a chosen suggestion as the value', async () => {
  const onchange = vi.fn()
  render(Suggest, { props: { id: 'cls', value: '', suggestions, onchange } })
  const input = screen.getByRole('combobox')
  await fireEvent.input(input, { target: { value: 'Flux' } })
  const option = await waitFor(() =>
    screen.getByRole('option', { name: 'FluxPipeline', hidden: true }),
  )
  await fireEvent.pointerUp(option)
  await fireEvent.click(option)
  await waitFor(() =>
    expect((input as HTMLInputElement).value).toBe('FluxPipeline'),
  )
  expect(onchange).toHaveBeenLastCalledWith('FluxPipeline')
})

it('keeps typed text no suggestion matches, and offers nothing', async () => {
  const onchange = vi.fn()
  render(Suggest, { props: { id: 'cls', value: '', suggestions, onchange } })
  const input = screen.getByRole('combobox') as HTMLInputElement
  await fireEvent.input(input, { target: { value: 'MyOwnPipeline' } })
  await fireEvent.change(input, { target: { value: 'MyOwnPipeline' } })
  expect(input.value).toBe('MyOwnPipeline')
  expect(options()).toEqual([])
  expect(onchange).toHaveBeenLastCalledWith('MyOwnPipeline')
})

it('with no suggestions is a plain input', async () => {
  render(Suggest, { props: { id: 'cls', value: 'x', suggestions: [] } })
  const input = screen.getByRole('combobox')
  await fireEvent.input(input, { target: { value: 'xy' } })
  expect(screen.queryByRole('listbox', { hidden: true })).toBeNull()
})

it('keeps its id, so a label names it', () => {
  const { container } = render(Suggest, {
    props: { id: 'cls', value: 'FluxPipeline', suggestions },
  })
  const label = document.createElement('label')
  label.htmlFor = 'cls'
  label.textContent = 'pipeline'
  container.prepend(label)
  expect((screen.getByLabelText('pipeline') as HTMLInputElement).value).toBe(
    'FluxPipeline',
  )
})

it('keeps typed text that a suggestion contains when Enter is pressed', async () => {
  const onchange = vi.fn()
  render(Suggest, {
    props: { id: 'cls', value: '', suggestions: ['variable:prompt'], onchange },
  })
  const input = screen.getByRole('combobox') as HTMLInputElement
  await fireEvent.input(input, { target: { value: 'prompt' } })
  await fireEvent.keyDown(input, { key: 'Enter' })
  expect(input.value).toBe('prompt')
  expect(onchange).not.toHaveBeenCalledWith('variable:prompt')
})

it('keeps typed text on Ctrl+Enter (validate & run), even after arrowing', async () => {
  render(Suggest, {
    props: { id: 'cls', value: '', suggestions: ['variable:prompt'] },
  })
  const input = screen.getByRole('combobox') as HTMLInputElement
  await fireEvent.input(input, { target: { value: 'prompt' } })
  await fireEvent.keyDown(input, { key: 'ArrowDown' })
  await fireEvent.keyDown(input, { key: 'Enter', ctrlKey: true })
  expect(input.value).toBe('prompt')
})

it('takes the suggestion arrowed to on Enter', async () => {
  const onchange = vi.fn()
  render(Suggest, { props: { id: 'cls', value: '', suggestions, onchange } })
  const input = screen.getByRole('combobox') as HTMLInputElement
  await fireEvent.input(input, { target: { value: 'Pipeline' } })
  await fireEvent.keyDown(input, { key: 'ArrowDown' })
  await fireEvent.keyDown(input, { key: 'Enter' })
  await waitFor(() => expect(suggestions).toContain(input.value))
  expect(onchange).toHaveBeenLastCalledWith(input.value)
})

it('takes the same suggestion again after the text was edited', async () => {
  const onchange = vi.fn()
  render(Suggest, { props: { id: 'cls', value: '', suggestions, onchange } })
  const input = screen.getByRole('combobox') as HTMLInputElement
  const pick = async () => {
    const option = await waitFor(() =>
      screen.getByRole('option', { name: 'FluxPipeline', hidden: true }),
    )
    await fireEvent.pointerUp(option)
    await fireEvent.click(option)
  }
  await fireEvent.input(input, { target: { value: 'Flux' } })
  await pick()
  await waitFor(() => expect(input.value).toBe('FluxPipeline'))
  await fireEvent.input(input, { target: { value: 'Flu' } })
  await pick()
  await waitFor(() => expect(input.value).toBe('FluxPipeline'))
  expect(onchange).toHaveBeenCalledTimes(2)
})
