import { render, screen, waitFor } from '@testing-library/svelte'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
// Hoisted above the component import so its static (hoisted) import sees an
// initialized mock - the same shape the other page tests use
import ServerPage from './ServerPage.svelte'
import { connection } from '../connection.svelte'
import { ApiError, NetworkError } from '../apiErrors'
import { api } from '../test/apiMock'
import type { HealthInfo, MemoryInfo, ServerInfo } from '../types'
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
}

const state = vi.hoisted(() => ({
  info: null as unknown,
  fail: '' as string,
  status: 0,
}))
const TRITON_COST =
  'The next first step of each SDNQ or compiled shape re-autotunes. ' +
  'This directory is shared with any other Triton user on this machine.'
const cacheList = () => ({
  caches: [
    {
      name: 'inductor',
      root: '/home/don/.cache/dw/torchinductor',
      enabled: true,
      bytes: 273 * 1024 ** 2,
      files: 1200,
      last_used: 1760000000,
      clearable: true,
      cost_of_clearing: 'The next compiled run recompiles its kernels.',
      disk_free: 400 * 1024 ** 3,
      disk_total: 1000 * 1024 ** 3,
    },
    {
      name: 'triton',
      root: '/home/don/.triton/cache',
      enabled: true,
      bytes: Math.round(7.1 * 1024 ** 3),
      files: 19683,
      last_used: 1760000000,
      clearable: true,
      cost_of_clearing: TRITON_COST,
      disk_free: 400 * 1024 ** 3,
      disk_total: 1000 * 1024 ** 3,
    },
  ] as Record<string, unknown>[],
})
const confirmDialog = vi.hoisted(() =>
  vi.fn((...args: [message: string, options?: unknown]): Promise<boolean> =>
    Promise.resolve(args.length > 0),
  ),
)
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
api.listCaches.mockImplementation((): Promise<unknown> =>
  Promise.resolve({ caches: [] }),
)
api.clearCache.mockImplementation((name: string): Promise<unknown> =>
  Promise.resolve({ name, cleared: true, freed: 0, files: 0 }),
)
api.deleteCacheEntry.mockImplementation(
  (name: string, entry: string): Promise<unknown> =>
    Promise.resolve({ name, entry, deleted: true, freed: 0 }),
)
// The page's own location: jsdom serves every test from localhost, so a
// proxied or tunnelled reader is the origin this returns instead
const page = vi.hoisted(() => ({ origin: null as string | null }))
vi.mock('../serverinfo', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../serverinfo')>()),
  reachedOrigin: () => page.origin,
}))
vi.mock('../changes.svelte', async () => ({
  changes: (await import('../test/feedMock')).feed.changes,
}))
vi.mock('../confirm.svelte', () => ({
  confirmDialog: (message: string, options?: unknown) =>
    confirmDialog(message, options),
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
  const { container } = render(ServerPage)
  await quiet()
  return (container.textContent ?? '').replace(/\s+/g, ' ')
}

beforeEach(() => {
  // A token the browser holds for its own requests must never reach the page
  localStorage.setItem('dw-api-token', JSON.stringify(SECRET))
})
afterEach(() => {
  api.server.mockClear()
  connection.health = null
  connection.memory = null
  api.listCaches.mockReset()
  api.listCaches.mockResolvedValue({ caches: [] })
  api.clearCache.mockClear()
  api.deleteCacheEntry.mockClear()
  confirmDialog.mockClear()
  localStorage.clear()
  page.origin = null
})

it('sets the tab title to Status, with the host and version beside it', async () => {
  await show()
  expect(document.title).toBe('Status · diffusers-workflow')
  expect(screen.getByRole('heading', { level: 1 }).textContent).toBe('Status')
  expect(
    screen.getByText(`${base.hostname} · v${base.version}`, {
      selector: '.secondary',
    }),
  ).toBeTruthy()
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

it('one heading says Status', async () => {
  const { container } = render(ServerPage)
  await quiet()
  const headings = [...container.querySelectorAll('h1, h2')].filter(
    (h) => h.textContent?.trim() === 'Status',
  )
  expect(headings).toHaveLength(1)
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

it('lists each worker with its card, VRAM and a link to its job', async () => {
  connection.health = {
    status: 'ok',
    worker_alive: true,
    current_job: 'job-a',
    queued: 0,
    workers: [
      {
        device: 'cuda:0',
        name: 'cuda:0 NVIDIA GeForce RTX 3090',
        vram_gb: 24,
        current_job: 'job-a',
        alive: true,
        host_memory_rss_mb: 2048,
      },
      {
        device: 'cuda:1',
        name: 'cuda:1 NVIDIA GeForce RTX 3090',
        vram_gb: 24,
        current_job: 'job-b',
        alive: true,
      },
    ],
  } as unknown as HealthInfo
  connection.state = 'ok'
  state.info = base
  state.fail = ''
  const { container } = render(ServerPage)
  await quiet()
  const text = (container.textContent ?? '').replace(/\s+/g, ' ')
  expect(text).toContain('cuda:0 NVIDIA GeForce RTX 3090')
  expect(text).toContain('cuda:1 NVIDIA GeForce RTX 3090')
  expect(text).toContain('2.0 GB RSS')
  const hrefs = [...container.querySelectorAll('li a')].map((a) =>
    a.getAttribute('href'),
  )
  expect(hrefs).toEqual(['#/ws/default/jobs/job-a', '#/ws/default/jobs/job-b'])
})

it('shows memory for every card, not just the first', async () => {
  const card = (device: string, allocated: number) => ({
    device,
    live: device === 'cuda:0',
    stale: device !== 'cuda:0',
    reason: device === 'cuda:0' ? null : 'job_running',
    age_seconds: device === 'cuda:0' ? 0 : 12,
    info: {
      gpu_available: true,
      gpu_device_name: 'NVIDIA GeForce RTX 3090',
      gpu_memory_allocated_mb: allocated,
      gpu_memory_total_mb: 24576,
    },
  })
  const first = card('cuda:0', 1024)
  connection.memory = {
    ...first,
    workers: [first, card('cuda:1', 18636.8)],
  } as unknown as MemoryInfo
  const text = await show()
  expect(text).toContain('cuda:0 1.0 / 24.0 GB')
  expect(text).toContain('cuda:1 18.2 / 24.0 GB 12s ago')
})

it('shows the runtime versions, with null as none / not installed', async () => {
  const text = await show({
    runtime: {
      python_version: '3.12.0',
      torch_version: '2.9.0',
      cuda_version: null,
      driver_version: null,
      packages: { diffusers: '0.36.0', xformers: null },
    },
  })
  expect(text).toContain('Runtime')
  expect(text).toContain('Python 3.12.0')
  expect(text).toContain('torch 2.9.0')
  expect(text).toContain('CUDA none')
  expect(text).toContain('driver none')
  expect(text).toContain('diffusers 0.36.0')
  expect(text).toContain('xformers not installed')
})

it('shows each worker pid and per-card free and reserved memory', async () => {
  connection.health = {
    status: 'ok',
    worker_alive: true,
    current_job: null,
    queued: 0,
    workers: [
      { device: 'cuda:0', name: 'cuda:0', alive: true, pid: 4242 },
      { device: 'cuda:1', name: 'cuda:1', alive: false },
    ],
  } as unknown as HealthInfo
  connection.state = 'ok'
  const entry = {
    device: 'cuda:0',
    live: true,
    stale: false,
    reason: null,
    age_seconds: 0,
    info: {
      gpu_available: true,
      gpu_memory_allocated_mb: 1024,
      gpu_memory_total_mb: 24576,
      gpu_memory_free_mb: 20480,
      gpu_memory_reserved_mb: 2048,
    },
  }
  connection.memory = { ...entry, workers: [entry] } as unknown as MemoryInfo
  const text = await show()
  expect(text).toContain('pid 4242')
  expect(text.match(/pid /g)).toHaveLength(1)
  expect(text).toContain('20.0 GB free')
  expect(text).toContain('2.0 GB reserved')
})

/** Render with the cache list given, returning the container. */
async function showCaches(list = cacheList()) {
  api.listCaches.mockResolvedValue(list)
  state.info = base
  state.fail = ''
  const { container } = render(ServerPage)
  await quiet()
  return container
}
const flat = (el: Element) => (el.textContent ?? '').replace(/\s+/g, ' ')
const clearButton = (container: Element, name: string) =>
  [...container.querySelectorAll('.caches tbody tr')]
    .find((tr) => tr.querySelector('td')?.textContent === name)!
    .querySelector('button') as HTMLButtonElement

it('lists each cache with its root, size, files and the disk it is on', async () => {
  const container = await showCaches()
  const text = flat(container.querySelector('.caches')!)
  expect(text).toContain('Caches')
  expect(text).toContain('inductor')
  expect(text).toContain('/home/don/.cache/dw/torchinductor')
  expect(text).toContain('273.0 MB')
  expect(text).toContain('1200')
  expect(text).toContain('triton')
  expect(text).toContain('/home/don/.triton/cache')
  expect(text).toContain('7.10 GB')
  expect(text).toContain('19683')
  expect(text).toContain('400.0 / 1000.0 GB free')
  // the bar is the used share of the volume
  const bar = container.querySelector('.caches .diskbar > span') as HTMLElement
  expect(bar.getAttribute('style')).toContain('width: 60%')
})

it('asks before clearing, quoting the cost and the shared-cache warning', async () => {
  const container = await showCaches()
  clearButton(container, 'triton').click()
  await quiet()
  expect(confirmDialog).toHaveBeenCalledOnce()
  const [message, spec] = confirmDialog.mock.calls[0] as [
    string,
    { title: string },
  ]
  // the title names the cache; the message does not say it again
  expect(spec.title).toContain('triton')
  expect(message).not.toContain('triton')
  expect(message).toContain('7.10 GB, 19683 files.')
  expect(message).toContain('re-autotunes')
  expect(message).toContain('shared with any other Triton user')
  expect(api.clearCache).toHaveBeenCalledWith('triton')
  // the list is read again after the clear, so the size shown drops
  expect(api.listCaches).toHaveBeenCalledTimes(2)
})

it('names an empty cache as 0 bytes in the confirm, not 1 KB', async () => {
  const list = cacheList()
  list.caches[0] = { ...list.caches[0], bytes: 0, files: 0 }
  const container = await showCaches(list)
  clearButton(container, 'inductor').click()
  await quiet()
  const message = confirmDialog.mock.calls[0][0]
  expect(message).toMatch(/^0 bytes, 0 files\. /)
  expect(message).not.toContain('1 KB')
})

it('does not clear when the confirm is declined', async () => {
  confirmDialog.mockResolvedValueOnce(false)
  const container = await showCaches()
  clearButton(container, 'inductor').click()
  await quiet()
  expect(confirmDialog).toHaveBeenCalledOnce()
  expect(api.clearCache).not.toHaveBeenCalled()
})

it('disables Clear with the reason when a cache is not clearable', async () => {
  const list = cacheList()
  list.caches[0] = {
    ...list.caches[0],
    clearable: false,
    not_clearable_reason: 'root not created by dw (no .dw-cache marker)',
  }
  const container = await showCaches(list)
  const button = clearButton(container, 'inductor')
  expect(button.disabled).toBe(true)
  expect(button.title).toBe('root not created by dw (no .dw-cache marker)')
  expect(flat(container)).toContain(
    'root not created by dw (no .dw-cache marker)',
  )
  expect(clearButton(container, 'triton').disabled).toBe(false)
})

it("shows the server's text when a clear is refused as busy", async () => {
  const busy =
    'A job is running or queued - clearing a cache it may be writing ' +
    'would leave it a truncated artifact'
  api.clearCache.mockRejectedValueOnce(
    Object.assign(new Error(busy), { status: 409 }),
  )
  const container = await showCaches()
  clearButton(container, 'inductor').click()
  await quiet()
  expect(flat(container)).toContain(busy)
  // and the sizes are re-read, unchanged
  expect(flat(container)).toContain('273.0 MB')
})

const ID_A = 'a'.repeat(64)
const ID_B = 'b'.repeat(64)
const quantizedList = (extra: Record<string, unknown> = {}) => {
  const list = cacheList()
  list.caches.push({
    name: 'quantized',
    root: '/home/don/.cache/dw/quantized',
    enabled: true,
    bytes: 5 * 1024 ** 3,
    files: 12,
    last_used: 1760000000,
    clearable: true,
    cost_of_clearing: 'The next load re-quantizes the component.',
    disk_free: 400 * 1024 ** 3,
    disk_total: 1000 * 1024 ** 3,
    low_disk: false,
    entries: [
      {
        id: ID_A,
        bytes: 3 * 1024 ** 3,
        files: 8,
        unreadable: false,
        model: 'org/flux',
        subfolder: 'transformer',
        component: 'transformer',
        created: 1760000000,
        last_hit: null,
      },
      {
        id: ID_B,
        bytes: 2 * 1024 ** 3,
        files: 4,
        unreadable: true,
        model: '',
        subfolder: '',
        component: '',
        created: 1760000000,
        last_hit: 1760000500,
      },
    ],
    ...extra,
  })
  return list
}
const toggleButton = (c: Element) =>
  c.querySelector('.entrytoggle') as HTMLButtonElement
const entryDelete = (c: Element, i: number) =>
  c
    .querySelectorAll('.entrytable tbody tr')
    [i].querySelector('button') as HTMLButtonElement

it('shows an entry count that expands to the entry table', async () => {
  const container = await showCaches(quantizedList())
  expect(flat(toggleButton(container))).toContain('2 entries')
  expect(container.querySelector('.entrytable')).toBeNull()
  toggleButton(container).click()
  await quiet()
  const text = flat(container.querySelector('.entrytable')!)
  expect(text).toContain('org/flux')
  expect(text).toContain('transformer')
  expect(text).toContain('3.00 GB')
  expect(text).toContain('unreadable')
  expect(text).toContain('a'.repeat(12))
  expect(text).not.toContain('a'.repeat(13))
  expect(container.querySelector(`code[title="${ID_A}"]`)).not.toBeNull()
  expect(text).toContain('—')
})

it('deletes an entry after the confirm, then re-reads the caches', async () => {
  const container = await showCaches(quantizedList())
  toggleButton(container).click()
  await quiet()
  entryDelete(container, 0).click()
  await quiet()
  const [message, spec] = confirmDialog.mock.calls[0] as [
    string,
    { title: string },
  ]
  // the title names the entry; the message says its size and cost
  expect(spec.title).toBe(
    'Delete transformer of org/flux from the quantized cache?',
  )
  expect(message).toMatch(/^3\.00 GB\. /)
  expect(message).toContain('re-quantizes')
  expect(api.deleteCacheEntry).toHaveBeenCalledWith('quantized', ID_A)
  expect(api.listCaches).toHaveBeenCalledTimes(2)
})

it("shows the server's text when an entry delete is refused", async () => {
  const busy = 'A model download is active - try again when it finishes'
  api.deleteCacheEntry.mockRejectedValueOnce(
    Object.assign(new Error(busy), { status: 409 }),
  )
  const container = await showCaches(quantizedList())
  toggleButton(container).click()
  await quiet()
  entryDelete(container, 1).click()
  await quiet()
  expect(flat(container)).toContain(busy)
})

it('disables entry Delete when the cache is not clearable', async () => {
  const container = await showCaches(
    quantizedList({ clearable: false, not_clearable_reason: 'no marker' }),
  )
  toggleButton(container).click()
  await quiet()
  expect(entryDelete(container, 0).disabled).toBe(true)
  expect(entryDelete(container, 1).disabled).toBe(true)
})

it('warns when the volume is under the free-space floor', async () => {
  const container = await showCaches(quantizedList({ low_disk: true }))
  expect(flat(container)).toContain('free-space floor')
  const ok = await showCaches(quantizedList())
  expect(flat(ok.querySelector('.caches')!)).not.toContain('free-space floor')
})

it('ServerPage recovers after the server comes back', async () => {
  api.server.mockImplementationOnce(() =>
    Promise.reject(new NetworkError('fetch failed')),
  )
  state.info = base
  state.fail = ''
  const { container } = render(ServerPage)
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
