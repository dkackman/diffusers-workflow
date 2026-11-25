import { afterEach, beforeEach, expect, it, vi } from 'vitest'

type Connection = typeof import('./connection.svelte')
let mod: Connection

beforeEach(async () => {
  vi.resetModules()
  mod = await import('./connection.svelte')
})
afterEach(() => vi.unstubAllGlobals())

it('a 401 locks and the next 2xx raises the epoch', () => {
  mod.reportResponse(200)
  expect(mod.connection.state).toBe('ok')
  mod.reportResponse(401)
  expect(mod.connection.state).toBe('locked')
  expect(mod.connection.epoch).toBe(0)
  mod.reportResponse(200)
  expect(mod.connection.state).toBe('ok')
  expect(mod.connection.epoch).toBe(1)
  // Staying ok does not raise it again
  mod.reportResponse(404)
  expect(mod.connection.epoch).toBe(1)
})

it('leaving unknown does not raise the epoch', () => {
  expect(mod.connection.state).toBe('unknown')
  mod.reportResponse(503)
  expect(mod.connection.state).toBe('ok')
  expect(mod.connection.epoch).toBe(0)
})

it('a fetch TypeError is offline and an AbortError is not', async () => {
  const { api } = await import('./api')
  vi.stubGlobal('fetch', () => Promise.reject(new TypeError('Failed to fetch')))
  const { NetworkError } = await import('./apiErrors')
  await expect(api.health()).rejects.toBeInstanceOf(NetworkError)
  expect(mod.connection.state).toBe('offline')

  mod.reportResponse(200)
  vi.stubGlobal('fetch', () =>
    Promise.reject(new DOMException('The user aborted', 'AbortError')),
  )
  await expect(api.health()).rejects.toThrow('aborted')
  expect(mod.connection.state).toBe('ok')
})

it('a 401 to a token replaced since does not lock the app again', async () => {
  const { api } = await import('./api')
  const { setApiToken } = await import('./token')
  let reply: (r: unknown) => void = () => {}
  vi.stubGlobal('fetch', () => new Promise((r) => (reply = r)))
  setApiToken('old')
  const pending = api.health()
  setApiToken('new')
  mod.reportResponse(200)
  reply({
    ok: false,
    status: 401,
    statusText: 'Unauthorized',
    headers: new Headers(),
    json: () => Promise.resolve({ detail: 'Invalid or missing API token' }),
  })
  await expect(pending).rejects.toMatchObject({ status: 401 })
  expect(mod.connection.state).toBe('ok')
  setApiToken('')
})

it('onReconnect is called once per rise', () => {
  const fn = vi.fn()
  const stop = mod.onReconnect(fn)
  mod.reportUnreachable()
  mod.reportResponse(200)
  mod.reportResponse(200)
  expect(fn).toHaveBeenCalledTimes(1)
  mod.reportResponse(401)
  mod.reportResponse(200)
  expect(fn).toHaveBeenCalledTimes(2)
  stop()
  mod.reportUnreachable()
  mod.reportResponse(200)
  expect(fn).toHaveBeenCalledTimes(2)
})

it('the header says offline or locked in place of idle', () => {
  expect(mod.quietWord('offline')).toBe('offline')
  expect(mod.quietWord('locked')).toBe('locked')
  expect(mod.quietWord('ok')).toBe('idle')
  expect(mod.quietWord('unknown')).toBe('idle')
})
