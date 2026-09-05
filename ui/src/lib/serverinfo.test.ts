import { describe, expect, it } from 'vitest'
import {
  addressLabel,
  browserUrl,
  isLoopbackBind,
  isUnauthenticatedPublicBind,
  mcpAddCommand,
  mcpUrl,
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
  mcp: { mounted: true, path: '/mcp' },
  addresses: [{ address: '192.168.1.50', family: 'IPv4', interface: 'enp6s0' }],
  directories: { workflows: '/w', outputs: '/o', prompts: null },
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

describe('bind classification', () => {
  it('recognises loopback binds', () => {
    for (const host of ['127.0.0.1', 'localhost', '::1', '127.0.1.1'])
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

it('labels an address by interface', () => {
  expect(
    addressLabel({ address: '10.0.0.4', family: 'IPv4', interface: 'wlan0' }),
  ).toBe('10.0.0.4 (wlan0)')
})
