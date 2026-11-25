import { afterEach, expect, it } from 'vitest'
import { cleanup, render, screen } from '@testing-library/svelte'
import KeyboardHelp from './KeyboardHelp.svelte'

afterEach(cleanup)

it('the help lists slash', async () => {
  render(KeyboardHelp, { props: { open: true } })
  expect(await screen.findByText("search this page's list")).toBeTruthy()
  expect(screen.getByText('/', { selector: 'kbd' })).toBeTruthy()
})
