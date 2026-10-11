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
      current_job: null as string | null,
      queued: 0,
      workers: [] as unknown[],
    }),
  ),
)
// the worker answers 503 for memory until it has started
const memory = vi.hoisted(() =>
  vi.fn((): Promise<unknown> => Promise.reject(new Error('503'))),
)
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
const listCaches = vi.hoisted(() =>
  vi.fn((): Promise<unknown> => Promise.resolve({ caches: [] })),
)
const clearCache = vi.hoisted(() =>
  vi.fn((name: string): Promise<unknown> =>
    Promise.resolve({ name, cleared: true, freed: 0, files: 0 }),
  ),
)
const confirmDialog = vi.hoisted(() =>
  vi.fn((...args: [message: string, options?: unknown]): Promise<boolean> =>
    Promise.resolve(args.length > 0),
  ),
)
vi.mock('../api', () => ({
  api: {
    server: () => server(),
    health: () => health(),
    memory: () => memory(),
    listCaches: () => listCaches(),
    clearCache: (name: string) => clearCache(name),
  },
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
  listCaches.mockReset()
  listCaches.mockResolvedValue({ caches: [] })
  clearCache.mockClear()
  confirmDialog.mockClear()
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

it('lists each worker with its card, VRAM and a link to its job', async () => {
  health.mockResolvedValueOnce({
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
  })
  state.info = base
  state.fail = ''
  const { container } = render(ServerPage)
  for (let i = 0; i < 5; i++) await new Promise((r) => setTimeout(r, 0))
  const text = (container.textContent ?? '').replace(/\s+/g, ' ')
  expect(text).toContain('cuda:0 NVIDIA GeForce RTX 3090')
  expect(text).toContain('cuda:1 NVIDIA GeForce RTX 3090')
  expect(text).toContain('2.0 GB RSS')
  const hrefs = [...container.querySelectorAll('li a')].map((a) =>
    a.getAttribute('href'),
  )
  expect(hrefs).toEqual(['#/jobs/job-a', '#/jobs/job-b'])
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
  memory.mockResolvedValueOnce({
    ...first,
    workers: [first, card('cuda:1', 18636.8)],
  })
  const text = await show()
  expect(text).toContain('cuda:0 1.0 / 24.0 GB')
  expect(text).toContain('cuda:1 18.2 / 24.0 GB 12s ago')
})

/** Render with the cache list given, returning the container. */
async function showCaches(list = cacheList()) {
  listCaches.mockResolvedValue(list)
  state.info = base
  state.fail = ''
  const { container } = render(ServerPage)
  for (let i = 0; i < 5; i++) await new Promise((r) => setTimeout(r, 0))
  return container
}
const settle = async () => {
  for (let i = 0; i < 5; i++) await new Promise((r) => setTimeout(r, 0))
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
  await settle()
  expect(confirmDialog).toHaveBeenCalledOnce()
  const message = confirmDialog.mock.calls[0][0]
  expect(message).toContain('triton')
  expect(message).toContain('7.10 GB')
  expect(message).toContain('re-autotunes')
  expect(message).toContain('shared with any other Triton user')
  expect(clearCache).toHaveBeenCalledWith('triton')
  // the list is read again after the clear, so the size shown drops
  expect(listCaches).toHaveBeenCalledTimes(2)
})

it('does not clear when the confirm is declined', async () => {
  confirmDialog.mockResolvedValueOnce(false)
  const container = await showCaches()
  clearButton(container, 'inductor').click()
  await settle()
  expect(confirmDialog).toHaveBeenCalledOnce()
  expect(clearCache).not.toHaveBeenCalled()
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
  clearCache.mockRejectedValueOnce(
    Object.assign(new Error(busy), { status: 409 }),
  )
  const container = await showCaches()
  clearButton(container, 'inductor').click()
  await settle()
  expect(flat(container)).toContain(busy)
  // and the sizes are re-read, unchanged
  expect(flat(container)).toContain('273.0 MB')
})
