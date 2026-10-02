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
