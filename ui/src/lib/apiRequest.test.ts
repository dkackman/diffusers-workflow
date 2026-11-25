// The one place a request leaves the UI: what it sends, and what each way
// of failing throws. `fetch` is stubbed; everything above it is real.
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { ApiError, NetworkError } from './apiErrors'
import {
  fetchJson,
  fetchOutputText,
  fetchText,
  isValidateRequest,
  jsonInit,
  preconditionHeaders,
  putWithPreconditions,
  request,
} from './apiRequest'
import { connection } from './connection.svelte'
import { setApiToken } from './token'

function answer(status: number, body: unknown, headers: HeadersInit = {}) {
  return Promise.resolve({
    ok: status < 400,
    status,
    statusText: 'Status Text',
    headers: new Headers(headers),
    json: () =>
      body === undefined
        ? Promise.reject(new SyntaxError('no body'))
        : Promise.resolve(body),
    text: () => Promise.resolve(String(body)),
  })
}

const fetch = vi.fn()
const sent = () => fetch.mock.calls[0] as [string, RequestInit]
const header = (name: string) => new Headers(sent()[1].headers).get(name)

beforeEach(() => {
  fetch.mockReset().mockImplementation(() => answer(200, { fine: true }))
  vi.stubGlobal('fetch', fetch)
  setApiToken('')
  connection.state = 'ok'
})
afterEach(() => vi.unstubAllGlobals())

describe('what is sent', () => {
  it('carries the token as a bearer header, and none without one', async () => {
    await request('/api/jobs')
    expect(header('Authorization')).toBeNull()
    fetch.mockClear()
    setApiToken('secret')
    await request('/api/jobs')
    expect(header('Authorization')).toBe('Bearer secret')
  })

  it('serializes a JSON body beside the extra headers', () => {
    expect(jsonInit('PUT', { a: 1 }, { 'If-Match': 'x' })).toEqual({
      method: 'PUT',
      headers: { 'Content-Type': 'application/json', 'If-Match': 'x' },
      body: '{"a":1}',
    })
  })

  it('turns preconditions into If-Match and If-None-Match', () => {
    expect(preconditionHeaders({})).toEqual({})
    expect(preconditionHeaders({ ifMatch: '"v1"' })).toEqual({
      'If-Match': '"v1"',
    })
    expect(preconditionHeaders({ createOnly: true })).toEqual({
      'If-None-Match': '*',
    })
  })
})

describe('what a failure throws', () => {
  it("is an ApiError with the server's detail, status and code", async () => {
    fetch.mockImplementation(() =>
      answer(409, { detail: { message: 'already there', code: 'exists' } }),
    )
    const error = await request('/api/jobs').catch((e: unknown) => e)
    expect(error).toBeInstanceOf(ApiError)
    expect(error).toMatchObject({
      message: 'already there',
      status: 409,
      code: 'exists',
    })
  })

  it('falls back to the status text when the body is not JSON', async () => {
    fetch.mockImplementation(() => answer(500, undefined))
    await expect(request('/api/jobs')).rejects.toThrow('Status Text')
  })

  it('is a NetworkError, and offline, when nothing answers', async () => {
    fetch.mockImplementation(() => Promise.reject(new TypeError('failed')))
    await expect(request('/api/jobs')).rejects.toBeInstanceOf(NetworkError)
    expect(connection.state).toBe('offline')
  })

  it('leaves any other rejection as it was, and says nothing', async () => {
    const aborted = new DOMException('stopped', 'AbortError')
    fetch.mockImplementation(() => Promise.reject(aborted))
    await expect(request('/api/jobs')).rejects.toBe(aborted)
    expect(connection.state).toBe('ok')
  })

  it('a 401 locks the connection', async () => {
    fetch.mockImplementation(() => answer(401, { detail: 'no' }))
    await expect(request('/api/jobs')).rejects.toThrow('no')
    expect(connection.state).toBe('locked')
  })

  it('a 401 for a token since replaced locks nothing', async () => {
    setApiToken('old')
    fetch.mockImplementation(() => {
      setApiToken('new')
      return answer(401, { detail: 'no' })
    })
    await expect(request('/api/jobs')).rejects.toThrow('no')
    expect(connection.state).toBe('ok')
  })

  it('names the output it could not read, keeping the status', async () => {
    fetch.mockImplementation(() => answer(404, { detail: 'gone' }))
    await expect(
      fetchOutputText('run/notes.txt', 'default'),
    ).rejects.toMatchObject({
      message: 'Could not read notes.txt',
      status: 404,
    })
  })
})

describe('what comes back', () => {
  it('the body and the response, or the text revalidated', async () => {
    const { body, response } = await fetchJson<{ fine: boolean }>('/api/x')
    expect(body).toEqual({ fine: true })
    expect(response.status).toBe(200)
    fetch.mockClear()
    fetch.mockImplementation(() => answer(200, 'hello'))
    expect(await fetchText('/outputs/a.txt')).toBe('hello')
    expect(sent()[1].cache).toBe('no-cache')
  })

  it('a conditional write answers what was saved and its new ETag', async () => {
    fetch.mockImplementation(() => answer(200, { id: 'a' }, { ETag: '"v2"' }))
    const written = await putWithPreconditions<{ id: string }>(
      '/api/x',
      { id: 'a' },
      { ifMatch: '"v1"' },
    )
    expect(written).toEqual({ saved: { id: 'a' }, etag: '"v2"' })
    expect(sent()[1].method).toBe('PUT')
    expect(header('If-Match')).toBe('"v1"')
  })

  it('a validate body names a workflow or a path', () => {
    expect(isValidateRequest({ workflow: {} })).toBe(true)
    expect(isValidateRequest({ workflow_path: 'a.json' })).toBe(true)
    expect(isValidateRequest({ id: 'a', steps: [] })).toBe(false)
  })
})
