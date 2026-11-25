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

URL.createObjectURL ??= () => 'blob:schema'
URL.revokeObjectURL ??= () => {}

afterEach(cleanup)

it('the definitions filter is the field slash finds', async () => {
  const { container } = render(SchemaPage)
  await screen.findByText('Definitions')
  const field = container.querySelector<HTMLInputElement>('input.filter')
  expect(field).not.toBeNull()
  document.body.innerHTML = `<main>${document.body.innerHTML}</main>`
  expect(focusPageFilter()).toBe(true)
})

// A link to /api/schema sends no token; the schema already read does not need one
it('the raw JSON link serves the schema the page read', async () => {
  const made: Blob[] = []
  URL.createObjectURL = vi.fn((blob: Blob) => {
    made.push(blob)
    return 'blob:schema'
  })
  URL.revokeObjectURL = vi.fn()
  // the test above leaves a copy of its page in the body
  document.body.innerHTML = ''
  const { unmount } = render(SchemaPage)
  const link = await screen.findByRole('link', { name: 'raw JSON' })
  expect(link.getAttribute('href')).toBe('blob:schema')
  expect(JSON.parse(await made.at(-1)!.text()).description).toBe('d')
  unmount()
  expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:schema')
})
