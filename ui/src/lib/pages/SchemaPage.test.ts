import { cleanup, render, screen } from '@testing-library/svelte'
import { afterEach, expect, it, vi } from 'vitest'
import SchemaPage from './SchemaPage.svelte'
import '../router.svelte'
import { focusPageFilter } from '../pageFrame.svelte'
import { api } from '../test/apiMock'

vi.mock('../api', async () => (await import('../test/apiMock')).apiModule())
api.getSchema.mockImplementation(() =>
  Promise.resolve({
    description: 'd',
    $defs: { Step: { type: 'object' } },
  }),
)

afterEach(cleanup)

it('the definitions filter is the field slash finds', async () => {
  const { container } = render(SchemaPage)
  await screen.findByText('Definitions')
  const field = container.querySelector<HTMLInputElement>('input.filter')
  expect(field).not.toBeNull()
  document.body.innerHTML = `<main>${document.body.innerHTML}</main>`
  expect(focusPageFilter()).toBe(true)
})
