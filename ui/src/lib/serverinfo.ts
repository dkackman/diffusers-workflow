import type { ServerAddress, ServerInfo } from './types'
import { formatBytes, formatDay } from './format'

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

/** The MCP endpoint URL under a base URL (`http://host:port`, or the origin
 * a proxied page was loaded from), given the mount path the server reports. */
export function mcpUrlAt(base: string, path: string = '/mcp'): string {
  return `${base}${path.startsWith('/') ? path : `/${path}`}`
}

/** The MCP endpoint URL at an address and port. */
export function mcpUrl(
  address: string,
  port: number,
  path: string = '/mcp',
): string {
  return mcpUrlAt(browserUrl(address, port), path)
}

/** The `claude mcp add` command for the server at a base URL. The token
 * header is part of it only when the server requires one; the token itself
 * is a literal placeholder - see TOKEN_PLACEHOLDER. */
export function mcpAddCommandAt(
  base: string,
  path: string = '/mcp',
  withToken: boolean = true,
): string {
  const command = `claude mcp add --transport http dw ${mcpUrlAt(base, path)}`
  return withToken
    ? `${command} --header "Authorization: Bearer ${TOKEN_PLACEHOLDER}"`
    : command
}

/** The `claude mcp add` command for the server at an address and port. */
export function mcpAddCommand(
  address: string,
  port: number,
  path: string = '/mcp',
  withToken: boolean = true,
): string {
  return mcpAddCommandAt(browserUrl(address, port), path, withToken)
}

/** The guide to using a server from another machine. */
export const REMOTE_GUIDE =
  'https://github.com/dkackman/diffusers-workflow/blob/master/docs/REMOTE.md'

/** The command that restarts the server reachable from the network, with a
 * token. The token is a placeholder - see TOKEN_PLACEHOLDER. */
export function bindSnippet(): string {
  return `DW_API_TOKEN=${TOKEN_PLACEHOLDER} python -m dw.serve --host 0.0.0.0`
}

/** A bind that only this machine can reach. */
export function isLoopbackBind(host: string): boolean {
  return (
    host === '127.0.0.1' ||
    host === 'localhost' ||
    host === '::1' ||
    host === '[::1]' ||
    host.startsWith('127.') ||
    host.startsWith('::ffff:127.')
  )
}

/** The origin this page was loaded from, when its hostname is not this
 * machine's loopback: the reader reached the server through something - a
 * reverse proxy (docs/REMOTE.md, *Beyond your LAN*) or a tunnel - so that
 * origin is the address that works for them, and a loopback bind is no
 * reason to advise rebinding. Null on a loopback page. */
export function reachedOrigin(
  page: Pick<Location, 'hostname' | 'origin'> = window.location,
): string | null {
  return isLoopbackBind(page.hostname) ? null : page.origin
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

/** The rows of the Runtime panel, in display order. A null value is a
 * component that is not there: `none` for torch/CUDA/driver, `not installed`
 * for a package. */
export function runtimeRows(
  runtime: ServerInfo['runtime'],
): { name: string; value: string | null; absent: string }[] {
  return [
    { name: 'Python', value: runtime.python_version, absent: 'none' },
    { name: 'torch', value: runtime.torch_version, absent: 'none' },
    { name: 'CUDA', value: runtime.cuda_version, absent: 'none' },
    { name: 'driver', value: runtime.driver_version, absent: 'none' },
    ...Object.keys(runtime.packages)
      .sort()
      .map((name) => ({
        name,
        value: runtime.packages[name],
        absent: 'not installed',
      })),
  ]
}

/** Plain text for the Runtime panel's copy button: the hostname and dw
 * version first, then one `name: value` line per row. */
export function runtimeText(info: ServerInfo): string {
  return [
    `host: ${info.hostname}`,
    `dw: ${info.version}`,
    ...runtimeRows(info.runtime).map(
      (r) => `${r.name}: ${r.value ?? r.absent}`,
    ),
  ].join('\n')
}

/** The job history's storage line for the Server page: jobs, size, and the
 * span of dates they were submitted over. A figure the server could not
 * read is left out rather than shown as zero. */
export function jobHistoryLine(storage: ServerInfo['storage']): string {
  const history = storage.job_history
  if (!history) return 'not open'
  const parts: string[] = []
  if (history.rows != null) {
    parts.push(`${history.rows} ${history.rows === 1 ? 'job' : 'jobs'}`)
  }
  if (history.size_bytes != null) parts.push(formatBytes(history.size_bytes))
  if (history.oldest != null && history.newest != null) {
    parts.push(`${formatDay(history.oldest)} – ${formatDay(history.newest)}`)
  }
  return parts.length ? parts.join(' · ') : 'unreadable'
}

/** The log's storage line for the Server page: the active file's size and
 * how many rotated backups sit beside it (with their total). */
export function logsLine(storage: ServerInfo['storage']): string {
  const logs = storage.logs
  if (!logs) return 'no log file configured'
  if (logs.size_bytes == null) return 'unreadable'
  const parts = [formatBytes(logs.size_bytes)]
  if (logs.backups != null) {
    parts.push(`${logs.backups} ${logs.backups === 1 ? 'backup' : 'backups'}`)
  }
  if (logs.backups && logs.total_bytes != null) {
    parts.push(`${formatBytes(logs.total_bytes)} in all`)
  }
  return parts.join(' · ')
}
