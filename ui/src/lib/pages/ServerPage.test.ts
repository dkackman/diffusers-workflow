import { render } from '@testing-library/svelte'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
// Hoisted above the component import so its static (hoisted) import sees an
// initialized mock - the same shape the other page tests use
import ServerPage from './ServerPage.svelte'
import type { ServerInfo } from '../types'

const SECRET = 'super-secret-token-value'

const base: ServerInfo = {
  hostname: 'lem',
  version: '0.4.0-alpha.12',
  device: 'cuda',
  bind_host: '0.0.0.0',
  port: 8765,
  wildcard_bind: true,
  auth_required: true,
  mcp: { mounted: true, path: '/mcp' },
  addresses: [{ address: '192.168.1.50', family: 'IPv4', interface: 'enp6s0' }],
  directories: {
    workflows: '/home/don/workflows',
    outputs: '/home/don/outputs',
    prompts: null,
  },
}

const state = vi.hoisted(() => ({
  info: null as unknown,
  fail: '' as string,
}))
const server = vi.hoisted(() =>
  vi.fn(() =>
    state.fail
      ? Promise.reject(new Error(state.fail))
      : Promise.resolve(state.info),
  ),
)
const health = vi.hoisted(() =>
  vi.fn(() =>
    Promise.resolve({
      status: 'ok',
      worker_alive: true,
      current_job: null,
      queued: 0,
    }),
  ),
)
vi.mock('../api', () => ({
  api: { server: () => server(), health: () => health() },
}))

/** Render with a given payload and let the promises settle, returning the
 * page's text with whitespace collapsed - the prose wraps in the markup, so
 * a raw textContent match would depend on where the lines break. There is no
 * automatic cleanup here (vitest globals are off), so each test reads its own
 * container rather than document.body. */
async function show(overrides: Partial<ServerInfo> = {}, fail = '') {
  state.info = { ...base, ...overrides }
  state.fail = fail
  const { container } = render(ServerPage)
  for (let i = 0; i < 5; i++) await new Promise((r) => setTimeout(r, 0))
  return (container.textContent ?? '').replace(/\s+/g, ' ')
}

beforeEach(() => {
  // A token the browser holds for its own requests must never reach the page
  localStorage.setItem('dw-api-token', JSON.stringify(SECRET))
})
afterEach(() => {
  server.mockClear()
  health.mockClear()
  localStorage.clear()
})

it('composes the browser URL and the MCP command from the address', async () => {
  const text = await show()
  expect(text).toContain('http://192.168.1.50:8765')
  expect(text).toContain(
    'claude mcp add --transport http dw http://192.168.1.50:8765/mcp ' +
      '--header "Authorization: Bearer <token>"',
  )
})

it('never renders a real token, only the placeholder', async () => {
  const text = await show()
  expect(text).toContain('<token>')
  expect(text).not.toContain(SECRET)
  expect(text).not.toContain(SECRET.slice(0, 6))
  expect(text).toContain('Token required')
  expect(text).toContain('yes')
})

it('explains that MCP needs --mcp when it is not mounted', async () => {
  const text = await show({ mcp: { mounted: false, path: '/mcp' } })
  expect(text).toContain('dw-serve --mcp')
  expect(text).not.toContain('claude mcp add')
})

it('says so when there is no non-loopback address', async () => {
  const text = await show({ addresses: [] })
  expect(text).toContain('reachable only from itself')
  expect(text).not.toContain('claude mcp add')
  expect(text).not.toContain('http://')
})

it('warns when auth is off on a public bind', async () => {
  const text = await show({ auth_required: false })
  expect(text).toContain(
    'anyone who can reach this address can run workflows on this GPU',
  )
})

it('does not warn when auth is off on a loopback bind', async () => {
  const text = await show({
    auth_required: false,
    bind_host: '127.0.0.1',
    wildcard_bind: false,
    addresses: [],
  })
  expect(text).not.toContain('anyone who can reach this address')
  expect(text).toContain('reachable only from this machine')
})

it('reports a failed request instead of an empty page', async () => {
  const text = await show({}, 'Not authenticated')
  expect(text).toContain('Not authenticated')
  expect(text).toContain('API token')
})
