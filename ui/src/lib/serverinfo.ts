import type { ServerAddress, ServerInfo } from './types'

/** The literal stand-in the copyable snippets carry in place of the API
 * token. The token is never rendered - not the value, not a prefix, not a
 * length - so the user pastes the command and fills this in themselves. */
export const TOKEN_PLACEHOLDER = '<token>'

/** Host part of a URL: an IPv6 literal has to be bracketed, an IPv4
 * address or hostname is used as it is. */
export function urlHost(address: string): string {
  return address.includes(':') ? `[${address}]` : address
}

/** The URL a browser on another machine opens to reach this server. */
export function browserUrl(address: string, port: number): string {
  return `http://${urlHost(address)}:${port}`
}

/** The MCP endpoint URL, given the mount path the server reports. */
export function mcpUrl(
  address: string,
  port: number,
  path: string = '/mcp',
): string {
  const suffix = path.startsWith('/') ? path : `/${path}`
  return `${browserUrl(address, port)}${suffix}`
}

/** The `claude mcp add` command for this server. The token is a literal
 * placeholder - see TOKEN_PLACEHOLDER. */
export function mcpAddCommand(
  address: string,
  port: number,
  path: string = '/mcp',
): string {
  return (
    `claude mcp add --transport http dw ${mcpUrl(address, port, path)} ` +
    `--header "Authorization: Bearer ${TOKEN_PLACEHOLDER}"`
  )
}

/** A bind that only this machine can reach. */
export function isLoopbackBind(host: string): boolean {
  return (
    host === '127.0.0.1' ||
    host === 'localhost' ||
    host === '::1' ||
    host.startsWith('127.')
  )
}

/** True when the server is reachable from other machines and asks nothing
 * of them - the case worth a warning, since reaching it is running
 * workflows on this GPU. */
export function isUnauthenticatedPublicBind(info: ServerInfo): boolean {
  return !info.auth_required && !isLoopbackBind(info.bind_host)
}

/** Label for one interface address in the picker. Falls back to the bare
 * address when psutil isn't available server-side to name the interface. */
export function addressLabel(entry: ServerAddress): string {
  return entry.interface
    ? `${entry.address} (${entry.interface})`
    : entry.address
}
