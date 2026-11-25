import { expect, it } from 'vitest'
import { ApiError, NetworkError, describeError } from './apiErrors'

it('describes a fetch that never got a response as offline', () => {
  const d = describeError(new NetworkError('Failed to fetch'))
  expect(d.kind).toBe('offline')
  expect(d.title).toBe("Can't reach the server")
  expect(d.remedy).toBe(
    "Check that dw.serve is running. This page reloads when it's back.",
  )
})

it("a TypeError from the page's own code is not an outage", () => {
  const d = describeError(new TypeError('Cannot read properties of undefined'))
  expect(d.kind).toBe('unknown')
  expect(d.title).toBe('Something went wrong')
})

it('describes a 401 as locked', () => {
  const d = describeError(new ApiError('Unauthorized', 401))
  expect(d.kind).toBe('locked')
  expect(d.title).toBe('The server needs an API token')
  expect(d.remedy).toBe('Enter it with the key button in the header.')
})

it('describes a 403 with the first line of its message', () => {
  const d = describeError(new ApiError('Not allowed here\nmore', 403))
  expect(d.kind).toBe('forbidden')
  expect(d.title).toBe('Not allowed here')
  expect(d.remedy).toBeUndefined()
  expect(d.detail).toBe('Not allowed here\nmore')
})

it('describes a 404 with code no_workspace', () => {
  const d = describeError(
    new ApiError('No such workspace: x', 404, 'no_workspace'),
  )
  expect(d.kind).toBe('no_workspace')
  expect(d.title).toBe("This workspace doesn't exist")
  expect(d.remedy).toBe('Pick another one in the sidebar.')
  expect(d.code).toBe('no_workspace')
  expect(d.status).toBe(404)
})

it('describes a plain 404 as not found, naming what', () => {
  expect(describeError(new ApiError('nope', 404), 'That workflow').title).toBe(
    "That workflow wasn't found",
  )
  const d = describeError(new ApiError('nope', 404))
  expect(d.kind).toBe('not_found')
  expect(d.title).toBe("That wasn't found")
})

it('describes 400, 409 and 422 as refused, with the message', () => {
  for (const status of [400, 409, 422]) {
    const d = describeError(new ApiError('Name is taken', status))
    expect(d.kind).toBe('refused')
    expect(d.title).toBe('Name is taken')
    expect(d.detail).toBeUndefined()
  }
})

it('cuts a long refusal to its first line and keeps the rest as detail', () => {
  const long = `${'word '.repeat(50)}end\nsecond line`
  const d = describeError(new ApiError(long, 422))
  expect(d.title.length).toBeLessThanOrEqual(160)
  expect(d.title.endsWith(' ')).toBe(false)
  expect(d.title).not.toContain('\n')
  expect(d.title.startsWith('word word')).toBe(true)
  expect(d.detail).toBe(long)
})

it('describes a 5xx as a server error', () => {
  const d = describeError(
    new ApiError('Internal failure in the worker pool', 503),
  )
  expect(d.kind).toBe('server')
  expect(d.title).toBe('The server hit an error')
  expect(d.remedy).toBe('Try again. The server log has the detail.')
  expect(d.detail).toBe('Internal failure in the worker pool')
})

it('describes a SyntaxError as json', () => {
  const d = describeError(new SyntaxError('Unexpected token'))
  expect(d.kind).toBe('json')
  expect(d.title).toBe("The JSON doesn't parse")
})

it('describes anything else as unknown', () => {
  const d = describeError('weird')
  expect(d.kind).toBe('unknown')
  expect(d.title).toBe('Something went wrong')
})

it('a short engine message keeps its detail', () => {
  const d = describeError(new Error("KeyError: 'x'"))
  expect(d.title).toBe('Something went wrong')
  expect(d.detail).toBe("KeyError: 'x'")
})

it('a title taken from the message has no detail', () => {
  const d = describeError(new ApiError('Name is taken', 409))
  expect(d.title).toBe('Name is taken')
  expect(d.detail).toBeUndefined()
})
