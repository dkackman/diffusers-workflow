import { describe, expect, it } from 'vitest'
import {
  addressLabel,
  browserUrl,
  isLoopbackBind,
  reachedOrigin,
  isUnauthenticatedPublicBind,
  bindSnippet,
  mcpAddCommand,
  mcpUrl,
  jobHistoryLine,
  logsLine,
  runtimeRows,
  runtimeText,
  TOKEN_PLACEHOLDER,
  urlHost,
} from './serverinfo'
import type { ServerInfo } from './types'

const info = (overrides: Partial<ServerInfo> = {}): ServerInfo => ({
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
    workspace: '/ws',
    workflows: '/w',
    outputs: '/o',
    prompts: null,
    assets: null,
  },
  storage: {
    job_history: {
      path: '/d/jobs.sqlite',
      size_bytes: 3 * 1024 * 1024,
      rows: 412,
      oldest: 1_760_000_000,
      newest: 1_764_000_000,
    },
    logs: { path: '/d/log/dw.log', size_bytes: 2048, backups: 2, total_bytes: 4096 },
  },
  ...overrides,
})

describe('url composition', () => {
  it('builds the browser url from an address and port', () => {
    expect(browserUrl('192.168.1.50', 8765)).toBe('http://192.168.1.50:8765')
  })

  it('brackets an IPv6 literal', () => {
    expect(urlHost('fd00::1')).toBe('[fd00::1]')
    expect(browserUrl('fd00::1', 8765)).toBe('http://[fd00::1]:8765')
  })

  it('appends the reported mcp path', () => {
    expect(mcpUrl('192.168.1.50', 8765, '/mcp')).toBe(
      'http://192.168.1.50:8765/mcp',
    )
    expect(mcpUrl('192.168.1.50', 8765, 'mcp')).toBe(
      'http://192.168.1.50:8765/mcp',
    )
  })
})

describe('mcp add command', () => {
  it('is the documented claude mcp add invocation', () => {
    expect(mcpAddCommand('192.168.1.50', 8765, '/mcp')).toBe(
      'claude mcp add --transport http dw http://192.168.1.50:8765/mcp ' +
        '--header "Authorization: Bearer <token>"',
    )
  })

  it('carries the literal token placeholder, never a value', () => {
    const command = mcpAddCommand('10.0.0.4', 8765, '/mcp')
    expect(command).toContain(TOKEN_PLACEHOLDER)
    expect(TOKEN_PLACEHOLDER).toBe('<token>')
  })
})

describe('mcp add command token header', () => {
  it('the mcp line carries the token header only when a token is required', () => {
    expect(mcpAddCommand('10.0.0.4', 8765, '/mcp', true)).toContain(
      '--header "Authorization: Bearer <token>"',
    )
    expect(mcpAddCommand('10.0.0.4', 8765, '/mcp', false)).toBe(
      'claude mcp add --transport http dw http://10.0.0.4:8765/mcp',
    )
  })

  it('the bind snippet uses python -m dw.serve and the placeholder', () => {
    expect(bindSnippet()).toBe(
      'DW_API_TOKEN=<token> python -m dw.serve --host 0.0.0.0',
    )
  })
})

describe('bind classification', () => {
  it('recognises loopback binds', () => {
    for (const host of [
      '127.0.0.1',
      'localhost',
      '::1',
      '127.0.1.1',
      '[::1]',
      '::ffff:127.0.0.1',
    ])
      expect(isLoopbackBind(host)).toBe(true)
    for (const host of ['0.0.0.0', '192.168.1.50', '::'])
      expect(isLoopbackBind(host)).toBe(false)
  })

  it('flags an unauthenticated public bind only', () => {
    expect(isUnauthenticatedPublicBind(info({ auth_required: false }))).toBe(
      true,
    )
    expect(isUnauthenticatedPublicBind(info({ auth_required: true }))).toBe(
      false,
    )
    expect(
      isUnauthenticatedPublicBind(
        info({ auth_required: false, bind_host: '127.0.0.1' }),
      ),
    ).toBe(false)
  })
})

it('reports the page origin only when the page is not on loopback', () => {
  expect(
    reachedOrigin({ hostname: 'localhost', origin: 'http://localhost:8765' }),
  ).toBeNull()
  expect(
    reachedOrigin({ hostname: '[::1]', origin: 'http://[::1]:8765' }),
  ).toBeNull()
  expect(
    reachedOrigin({
      hostname: 'dw.example.com',
      origin: 'https://dw.example.com',
    }),
  ).toBe('https://dw.example.com')
})

it('labels an address by interface', () => {
  expect(
    addressLabel({ address: '10.0.0.4', family: 'IPv4', interface: 'wlan0' }),
  ).toBe('10.0.0.4 (wlan0)')
})

it('labels an address by itself when the interface is unknown', () => {
  expect(
    addressLabel({ address: '10.0.0.4', family: 'IPv4', interface: null }),
  ).toBe('10.0.0.4')
})

it('copies the runtime as host, version, then one line per row', () => {
  const text = runtimeText(
    info({
      runtime: {
        python_version: '3.12.0',
        torch_version: '2.9.0',
        cuda_version: null,
        driver_version: null,
        packages: { transformers: null, diffusers: '0.36.0' },
      },
    }),
  )
  expect(text.split('\n')).toEqual([
    'host: lem',
    'dw: 0.4.0-alpha.12',
    'Python: 3.12.0',
    'torch: 2.9.0',
    'CUDA: none',
    'driver: none',
    'diffusers: 0.36.0',
    'transformers: not installed',
  ])
})

it('sorts packages by name after the fixed rows', () => {
  const rows = runtimeRows({
    ...info().runtime,
    packages: { b: '1', a: '2' },
  })
  expect(rows.map((r) => r.name).slice(4)).toEqual(['a', 'b'])
})

describe('storage lines', () => {
  it('shows the job count, size and date span', () => {
    const line = jobHistoryLine(info().storage)
    expect(line).toMatch(/^412 jobs · 3\.0 MB · .+ – .+$/)
  })

  it('leaves out a figure the server could not read', () => {
    const storage = info().storage
    expect(
      jobHistoryLine({
        ...storage,
        job_history: { ...storage.job_history!, size_bytes: null, oldest: null },
      }),
    ).toBe('412 jobs')
    expect(jobHistoryLine({ ...storage, job_history: null })).toBe('not open')
  })

  it('shows the log size and its backups', () => {
    expect(logsLine(info().storage)).toBe('2 KB · 2 backups · 4 KB in all')
  })

  it('says when there is no log file', () => {
    const storage = info().storage
    expect(logsLine({ ...storage, logs: null })).toBe('no log file configured')
    expect(
      logsLine({
        ...storage,
        logs: { path: '/l', size_bytes: 10, backups: 0, total_bytes: 10 },
      }),
    ).toBe('1 KB · 0 backups')
  })
})
