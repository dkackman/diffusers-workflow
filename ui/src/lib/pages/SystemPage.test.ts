import { cleanup, render, screen } from '@testing-library/svelte'
import { afterEach, expect, it, vi } from 'vitest'
import SystemPage from './SystemPage.svelte'
import { api } from '../test/apiMock'
import type { ServerInfo } from '../types'
import { quiet, useQuietClock } from '../test/quiet'

useQuietClock()

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
api.server.mockImplementation(() => Promise.resolve(state.info))
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
api.diffusersStatus.mockImplementation(() =>
  Promise.resolve({ version: '0.36.0', commit: null, status: 'idle' }),
)
api.dwStatus.mockImplementation(() => Promise.resolve(dwState({})))
vi.mock('../changes.svelte', async () => ({
  changes: (await import('../test/feedMock')).feed.changes,
}))
vi.mock('../confirm.svelte', () => ({
  confirmDialog: (message: string, options?: unknown) =>
    confirmDialog(message, options),
}))

function dwState(over: Record<string, unknown>) {
  return {
    version: '0.23.0',
    latest: '0.24.0',
    newer: true,
    managed: true,
    command: 'curl -LsSf https://example.test/bootstrap.sh | sh',
    status: 'idle',
    error: null,
    log: null,
    ...over,
  }
}

/** Render with a given payload and let the promises settle, returning the
 * page's text with whitespace collapsed - the prose wraps in the markup, so
 * a raw textContent match would depend on where the lines break. */
async function show(overrides: Partial<ServerInfo> = {}) {
  state.info = { ...base, ...overrides }
  const { container } = render(SystemPage)
  await quiet()
  return (container.textContent ?? '').replace(/\s+/g, ' ')
}

afterEach(() => {
  cleanup()
  api.server.mockClear()
  api.listCaches.mockReset()
  api.listCaches.mockResolvedValue({ caches: [] })
  api.clearCache.mockClear()
  api.deleteCacheEntry.mockClear()
  api.updateDw.mockClear()
  api.dwStatus.mockImplementation(() => Promise.resolve(dwState({})))
  confirmDialog.mockClear()
})

it('sets the tab title to System', async () => {
  await show()
  expect(document.title).toBe('System · diffusers-workflow')
  expect(screen.getByRole('heading', { level: 1 }).textContent).toBe('System')
})

it('shows its own panels and none of the Connect ones', async () => {
  await show()
  const headings = [...document.querySelectorAll('h2')].map((h) =>
    h.textContent?.trim(),
  )
  expect(headings).toEqual([
    'Server',
    'Software',
    'Runtime',
    'Directories',
    'Storage',
    'Caches',
  ])
})

it('names the host, version and device', async () => {
  const text = await show()
  expect(text).toMatch(/Hostname\s*lem/)
  expect(text).toMatch(/Version\s*0\.4\.0-alpha\.12/)
  expect(text).toMatch(/Device\s*cuda/)
})

it('offers the dw update when the server says a newer one exists', async () => {
  await show()
  expect(screen.getByRole('button', { name: /Update dw/ })).toBeTruthy()
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

it('shows the job history and the logs read-only beside the caches', async () => {
  const text = await show()
  expect(text).toContain('Storage')
  expect(text).toMatch(/Job history\s*412 jobs · 3\.0 MB/)
  expect(text).toMatch(/Logs\s*2 KB · 2 backups · 4 KB in all/)
})

/** Render with the cache list given, returning the container. */
async function showCaches(list = cacheList()) {
  api.listCaches.mockResolvedValue(list)
  state.info = base
  state.fail = ''
  const { container } = render(SystemPage)
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
