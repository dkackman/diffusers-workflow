import { cleanup, render, screen, waitFor } from '@testing-library/svelte'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import ConnectPage from './ConnectPage.svelte'
import { connection } from '../connection.svelte'
import { ApiError, NetworkError } from '../apiErrors'
import { api } from '../test/apiMock'
import type { ServerInfo } from '../types'
import { quiet, useQuietClock } from '../test/quiet'

useQuietClock()

const SECRET = 'super-secret-token-value'

const base: ServerInfo = {
  hostname: 'lem',
  version: '0.4.0-alpha.12',
  device: 'cuda',
  bind_host: '0.0.0.0',
  port: 8765,
  wildcard_bind: true,
  auth_required: true,
  trust_workflows: false,
  runtime: {
    python_version: '3.12.0',
    torch_version: '2.9.0',
    cuda_version: '12.8',
    driver_version: null,
    packages: { diffusers: '0.36.0' },
  },
  mcp: { mounted: true, path: '/mcp' },
  addresses: [{ address: '192.168.1.50', family: 'IPv4', interface: 'enp6s0' }],
  directories: {
    workspace: '/home/don/studio',
    workflows: '/home/don/workflows',
    outputs: '/home/don/outputs',
    prompts: null,
    assets: '/home/don/studio/assets',
  },
  storage: {
    job_history: {
      path: '/home/don/.dw/jobs.sqlite',
      size_bytes: 3 * 1024 * 1024,
      rows: 412,
      oldest: 1_760_000_000,
      newest: 1_764_000_000,
    },
    logs: {
      path: '/home/don/.dw/log/dw.log',
      size_bytes: 2048,
      backups: 2,
      total_bytes: 4096,
    },
  },
}

const state = vi.hoisted(() => ({
  info: null as unknown,
  fail: '' as string,
  status: 0,
}))
vi.mock('../api', async () => (await import('../test/apiMock')).apiModule())
api.server.mockImplementation(() =>
  state.fail
    ? Promise.reject(
        state.status
          ? new ApiError(state.fail, state.status)
          : new Error(state.fail),
      )
    : Promise.resolve(state.info),
)
// The page's own location: jsdom serves every test from localhost, so a
// proxied or tunnelled reader is the origin this returns instead
const page = vi.hoisted(() => ({ origin: null as string | null }))
vi.mock('../serverinfo', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../serverinfo')>()),
  reachedOrigin: () => page.origin,
}))

/** Render with a given payload and let the promises settle, returning the
 * page's text with whitespace collapsed - the prose wraps in the markup, so
 * a raw textContent match would depend on where the lines break. There is no
 * automatic cleanup here (vitest globals are off), so each test reads its own
 * container rather than document.body. */
async function show(
  overrides: Partial<ServerInfo> = {},
  fail = '',
  status = 0,
) {
  state.info = { ...base, ...overrides }
  state.fail = fail
  state.status = status
  const { container } = render(ConnectPage)
  await quiet()
  return (container.textContent ?? '').replace(/\s+/g, ' ')
}

beforeEach(() => {
  // A token the browser holds for its own requests must never reach the page
  localStorage.setItem('dw-api-token', JSON.stringify(SECRET))
})
afterEach(() => {
  cleanup()
  api.server.mockClear()
  connection.health = null
  localStorage.clear()
  page.origin = null
})

it('sets the tab title to Connect', async () => {
  await show()
  expect(document.title).toBe('Connect · diffusers-workflow')
  expect(screen.getByRole('heading', { level: 1 }).textContent).toBe('Connect')
})

it('shows its own panels and none of the System or Activity ones', async () => {
  await show()
  const headings = [...document.querySelectorAll('h2')].map((h) =>
    h.textContent?.trim(),
  )
  expect(headings).toEqual([
    'Binding',
    'Addresses',
    'Authentication',
    'MCP',
    'Connect from another machine',
  ])
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

it('an unmounted MCP says python -m dw.serve --mcp', async () => {
  const text = await show({ mcp: { mounted: false, path: '/mcp' } })
  expect(text).toContain(
    'Not mounted. Restart the server with --mcp (python -m dw.serve --mcp) to serve the MCP tools at /mcp.',
  )
  expect(text).toContain(
    'The MCP endpoint is not mounted, so there is no command to run yet. Restart the server with --mcp.',
  )
  expect(text).not.toContain('dw-serve')
  expect(text).not.toContain('claude mcp add')
})

it('a loopback bind offers no network URL', async () => {
  const text = await show({ bind_host: '127.0.0.1', wildcard_bind: false })
  expect(text).toContain(
    'This server is bound to 127.0.0.1, so only this machine can reach it.',
  )
  expect(text).toContain(
    'Only this machine can reach this server. To use it from another, restart it bound to the network, with a token:',
  )
  expect(text).toContain(
    'DW_API_TOKEN=<token> python -m dw.serve --host 0.0.0.0',
  )
  expect(text).toContain('How to use dw from another machine')
  expect(text).not.toContain('192.168.1.50')
  expect(text).not.toContain('claude mcp add')
})

it('a loopback bind read on this machine keeps the restart advice', async () => {
  const text = await show({ bind_host: '127.0.0.1', wildcard_bind: false })
  expect(text).toContain('A loopback bind is reachable only from this machine.')
  expect(text).not.toContain('through a proxy or tunnel')
  expect(text).not.toContain('Endpoint')
})

it('a loopback bind reached through a proxy builds on the page origin', async () => {
  page.origin = 'https://dw.example.com'
  const text = await show({ bind_host: '127.0.0.1', wildcard_bind: false })
  expect(text).toContain(
    'You reached this server at https://dw.example.com, through a proxy or tunnel.',
  )
  expect(text).toContain('https://dw.example.com/mcp')
  expect(text).toContain(
    'claude mcp add --transport http dw https://dw.example.com/mcp ' +
      '--header "Authorization: Bearer <token>"',
  )
  expect(text).toContain("Replace <token> with the server's API token.")
  expect(text).toContain('Paste the token once')
  expect(text).not.toContain('only this machine')
  expect(text).not.toContain('--host 0.0.0.0')
  expect(text).not.toContain('192.168.1.50')
})

it('a tunnel to a tokenless loopback server needs no token header', async () => {
  page.origin = 'http://gpu-tunnel:9000'
  const text = await show({
    bind_host: '127.0.0.1',
    wildcard_bind: false,
    auth_required: false,
  })
  expect(text).toContain(
    'claude mcp add --transport http dw http://gpu-tunnel:9000/mcp',
  )
  expect(text).not.toContain('Authorization: Bearer')
  expect(text).not.toContain('--host 0.0.0.0')
})

it('a wildcard bind lists its addresses', async () => {
  const text = await show()
  expect(text).toContain('http://192.168.1.50:8765')
  expect(text).toContain(
    'claude mcp add --transport http dw http://192.168.1.50:8765/mcp',
  )
  expect(text).not.toContain('Only this machine can reach this server')
})

it('a bind to one address offers that address only', async () => {
  const text = await show({
    bind_host: '10.0.0.7',
    wildcard_bind: false,
    addresses: [
      { address: '192.168.1.50', family: 'IPv4', interface: 'enp6s0' },
      { address: '10.0.0.7', family: 'IPv4', interface: 'eth1' },
    ],
  })
  expect(text).toContain('http://10.0.0.7:8765')
  expect(text).toContain(
    'claude mcp add --transport http dw http://10.0.0.7:8765/mcp',
  )
  expect(text).not.toContain('192.168.1.50')
})

it('a bind to an address no interface lists still offers it', async () => {
  const text = await show({ bind_host: '10.9.9.9', wildcard_bind: false })
  expect(text).toContain('http://10.9.9.9:8765')
  expect(text).not.toContain('192.168.1.50')
})

it('no token required, no token note', async () => {
  const text = await show({ auth_required: false })
  expect(text).toContain(
    'claude mcp add --transport http dw http://192.168.1.50:8765/mcp',
  )
  expect(text).not.toContain('Authorization: Bearer')
  expect(text).not.toContain("Replace <token> with the server's API token.")
  expect(text).not.toContain('Paste the token once')
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

it('reports a 401 with the token remedy', async () => {
  const text = await show({}, 'Not authenticated', 401)
  expect(text).toContain('API token')
})

it('a 500 does not blame the token', async () => {
  const text = await show({}, 'boom', 500)
  expect(text).not.toContain('API token')
  expect(text).toContain('The server hit an error')
})

it('ConnectPage recovers after the server comes back', async () => {
  api.server.mockImplementationOnce(() =>
    Promise.reject(new NetworkError('fetch failed')),
  )
  state.info = base
  state.fail = ''
  const { container } = render(ConnectPage)
  const text = () => (container.textContent ?? '').replace(/\s+/g, ' ')
  await waitFor(() => expect(text()).toContain("Can't reach the server"))
  expect(text()).not.toContain('fetch failed')
  expect(text()).not.toContain('192.168.1.50')
  // The server answering again raises the epoch; the page reads again
  connection.epoch++
  await waitFor(() => expect(text()).toContain('http://192.168.1.50:8765'))
  expect(text()).not.toContain("Can't reach the server")
})

it('a server reached through a proxy is told to stay bound where it is', async () => {
  page.origin = 'https://dw.example.com'
  const text = await show({ bind_host: '127.0.0.1', wildcard_bind: false })
  expect(text).toContain(
    'You reach this server through a tunnel or proxy; keep it bound to 127.0.0.1.',
  )
  expect(text).not.toContain('restart it bound to the network')
})

it('the rebind advice names an ssh tunnel as the safer way', async () => {
  const text = await show({ bind_host: '127.0.0.1', wildcard_bind: false })
  expect(text).toContain(
    'Reaching it through ssh -L is the safer way: see How to use dw from another machine',
  )
})
