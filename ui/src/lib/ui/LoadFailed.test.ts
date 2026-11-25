import { afterEach, expect, it } from 'vitest'
import { cleanup, render } from '@testing-library/svelte'
import LoadFailed from './LoadFailed.svelte'
import { ApiError } from '../apiErrors'

afterEach(cleanup)

const props = (error: unknown) => ({
  label: 'tuna',
  error,
  onretry: () => {},
  oncreate: () => {},
})

it('offers to start a new one for a missing file, not for a missing workspace', () => {
  const missing = render(LoadFailed, {
    props: props(new ApiError('nope', 404)),
  })
  expect(missing.container.textContent).toContain(
    "Couldn't open tuna: That wasn't found",
  )
  expect(missing.queryByText('Start a new one named tuna')).not.toBeNull()
  cleanup()

  const noWorkspace = render(LoadFailed, {
    props: props(new ApiError('No such workspace: w', 404, 'no_workspace')),
  })
  expect(noWorkspace.container.textContent).toContain(
    "Couldn't open tuna: This workspace doesn't exist",
  )
  expect(noWorkspace.container.textContent).toContain(
    'Pick another one in the sidebar.',
  )
  expect(noWorkspace.queryByText('Start a new one named tuna')).toBeNull()
})

it("offers the engine's message only where asked", () => {
  const error = new ApiError('Traceback: bad step', 500)
  expect(
    render(LoadFailed, { props: props(error) }).queryByText('Engine detail'),
  ).toBeNull()
  cleanup()
  expect(
    render(LoadFailed, {
      props: { ...props(error), detail: true },
    }).queryByText('Engine detail'),
  ).not.toBeNull()
})
