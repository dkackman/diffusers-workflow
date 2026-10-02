import { getApiToken } from './token'
import { DEFAULT_WORKSPACE, workspace as picker } from './workspaceState.svelte'

/** Encode a workflow name for a URL, keeping its folder separators. */
export const encodePath = (name: string) =>
  name.split('/').map(encodeURIComponent).join('/')

/** Append one query parameter to a URL, keeping whatever query it already
 * has. Shared by every place that tacks a selector onto a path - the
 * workspace scope, the download token - so there is one rule for `?` vs
 * `&` instead of a hand-rolled check at each call site. */
export function appendQuery(url: string, key: string, value: string): string {
  const separator = url.includes('?') ? '&' : '?'
  return `${url}${separator}${key}=${encodeURIComponent(value)}`
}

/** Scope a path to a workspace. The server defaults to 'default' when no
 * selector is sent, so that one sends nothing and the request looks
 * exactly as it did before workspaces existed. Routes that are not
 * workspace-scoped (prompts, models, system) ignore an unknown query
 * parameter, which is what lets this live in one place instead of being
 * threaded through every call site. */
export function scopeTo(path: string, ws: string): string {
  return ws === DEFAULT_WORKSPACE ? path : appendQuery(path, 'workspace', ws)
}

/** Scope a path to the picker's current workspace. */
export const scoped = (path: string): string => scopeTo(path, picker.current)

/** Append the configured API token as a query parameter. Only for the
 * routes a browser loads without being able to set headers - EventSource,
 * <img> tags and <a download> navigations - which the server accepts it
 * on; see docs/SERVER.md. */
export function addToken(url: string): string {
  const token = getApiToken()
  return token ? appendQuery(url, 'token', token) : url
}

/** A browser-loadable URL: scoped to a workspace (the picker's unless one
 * is named - a job's own files must resolve to where they were written),
 * with the token. */
export function withToken(url: string, ws: string = picker.current): string {
  return addToken(scopeTo(url, ws))
}

/** The URL an output file is served from. Jobs report files by their name
 * relative to the output directory - a workflow under a subfolder writes
 * to '<sub>/<file>' - so the whole relative path is kept. A job recorded
 * before that change carries an absolute path, for which the basename is
 * the best available guess. `version` busts the browser cache: two runs of
 * one workflow write the same file names. `workspace`, when given, names
 * the job's own workspace and wins over whatever is currently selected in
 * the picker - a job page must load its files from where they were written,
 * not from wherever the user has since navigated to. */
export function outputUrl(
  path: string,
  version?: string,
  workspace?: string,
): string {
  const name = path.startsWith('/') ? (path.split('/').pop() ?? '') : path
  const url = `/outputs/${encodePath(name)}`
  const versioned = version === undefined ? url : appendQuery(url, 'v', version)
  return scopeTo(versioned, workspace ?? picker.current)
}
