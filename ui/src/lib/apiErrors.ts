/** What a failed API call carries back: the server's detail as a message a
 * person can read, and the status a caller may branch on. */

const BYTES_PER_MB = 1024 * 1024

/** The per-folder file counts a workspace delete answers with, as one line.
 * Folders holding nothing are left out - the point of the count is what
 * would actually be lost. */
function describeContents(contents: unknown): string {
  if (!contents || typeof contents !== 'object') return ''
  const parts: string[] = []
  for (const [folder, value] of Object.entries(
    contents as Record<string, { files?: number; bytes?: number }>,
  )) {
    const files = Number(value?.files ?? 0)
    if (!files) continue
    const bytes = Number(value?.bytes ?? 0)
    const size =
      bytes >= BYTES_PER_MB ? ` (${(bytes / BYTES_PER_MB).toFixed(1)} MB)` : ''
    parts.push(`${folder}: ${files} file${files === 1 ? '' : 's'}${size}`)
  }
  return parts.length ? parts.join(', ') : 'nothing'
}

/** The message inside an error response's `detail`. Most routes answer with
 * a plain string, but the ones that have to say what they would do send an
 * object instead - the workspace delete's `{message, contents}`, the
 * not-a-workspace refusal's `{message, entries}` - and handing that straight
 * to `new Error` shows the user '[object Object]' rather than the very
 * numbers the confirmation exists to present. FastAPI's 422 list of
 * validation errors gets the same treatment. */
export function errorDetail(payload: unknown, fallback: string): string {
  const detail = (payload as { detail?: unknown } | null)?.detail
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail)) {
    const messages = detail
      .map((entry) =>
        entry && typeof entry === 'object'
          ? String((entry as { msg?: unknown }).msg ?? JSON.stringify(entry))
          : String(entry),
      )
      .filter(Boolean)
    return messages.length ? messages.join('. ') : fallback
  }
  if (detail && typeof detail === 'object') {
    const record = detail as Record<string, unknown>
    const message =
      typeof record.message === 'string' ? record.message : fallback
    const contents = describeContents(record.contents)
    if (contents) return `${message}\n\n${contents}`
    if (Array.isArray(record.entries) && record.entries.length) {
      return `${message}\n\n${record.entries.join(', ')}`
    }
    return message
  }
  return fallback
}

/** The stable name a refusal carries beside its message - the requests
 * ledger's `stale_revision`, `no_workspace_root` - when `detail` is an
 * object that has one. */
export function errorCode(payload: unknown): string | undefined {
  const detail = (payload as { detail?: unknown } | null)?.detail
  const code = (detail as { code?: unknown } | null)?.code
  return typeof code === 'string' ? code : undefined
}

/** The `detail` object of a refusal, for a caller that needs more than its
 * message: the plan and the acknowledgement to resend that a 409 on a
 * bound cost acknowledgement carries, the `jobs` holding a delete back. */
export function errorPayload(
  payload: unknown,
): Record<string, unknown> | undefined {
  const detail = (payload as { detail?: unknown } | null)?.detail
  return detail && typeof detail === 'object' && !Array.isArray(detail)
    ? (detail as Record<string, unknown>)
    : undefined
}

/** An error response, with the status a caller may need to branch on -
 * the export's 409 is "already exported, overwrite?" rather than a
 * failure, and the message alone cannot say which - the detail's `code`
 * where the server names its refusals, and the detail itself when it is an
 * object. */
export class ApiError extends Error {
  status: number
  code?: string
  detail?: Record<string, unknown>
  constructor(
    message: string,
    status: number,
    code?: string,
    detail?: Record<string, unknown>,
  ) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.code = code
    this.detail = detail
  }
}

/** fetch's own TypeError - the request never reached the server - as
 * `send` rethrows it. Only this is an outage: a TypeError from a bug in a
 * page's own code is not, and must not send the user to restart a healthy
 * server. Still a TypeError, for callers that only ask that. */
export class NetworkError extends TypeError {
  constructor(message: string) {
    super(message)
    this.name = 'NetworkError'
  }
}

export type ErrorKind =
  | 'offline'
  | 'locked'
  | 'forbidden'
  | 'no_workspace'
  | 'not_found'
  | 'refused'
  | 'server'
  | 'json'
  | 'unknown'

export type Described = {
  kind: ErrorKind
  title: string
  remedy?: string
  detail?: string
  code?: string
  status?: number
}

const REFUSAL_TITLE_MAX = 160

function firstLine(message: string): string {
  return message.split('\n')[0].trim()
}

/** A line cut to `max` characters at a word boundary */
function cutAtWord(line: string, max: number): string {
  if (line.length <= max) return line
  const cut = line.slice(0, max)
  const space = cut.lastIndexOf(' ')
  return (space > 0 ? cut.slice(0, space) : cut).trimEnd()
}

type Base = Pick<Described, 'kind' | 'title' | 'remedy'>

function titled(kind: ErrorKind, title: string, remedy?: string): Base {
  return { kind, title, remedy }
}

function byStatus(
  status: number,
  code: string | undefined,
  message: string,
  what: string,
): Base | undefined {
  if (status === 401) {
    return titled(
      'locked',
      'The server needs an API token',
      'Enter it with the key button in the header.',
    )
  }
  if (status === 403) return titled('forbidden', firstLine(message))
  if (status === 404 && code === 'no_workspace') {
    return titled(
      'no_workspace',
      "This workspace doesn't exist",
      'Pick another one in the sidebar.',
    )
  }
  if (status === 404) return titled('not_found', `${what} wasn't found`)
  if (status === 400 || status === 409 || status === 422) {
    return titled('refused', cutAtWord(firstLine(message), REFUSAL_TITLE_MAX))
  }
  if (status >= 500) {
    return titled(
      'server',
      'The server hit an error',
      'Try again. The server log has the detail.',
    )
  }
  return undefined
}

function classify(e: unknown, message: string, what: string): Base {
  if (e instanceof NetworkError) {
    return titled(
      'offline',
      "Can't reach the server",
      "Check that dw.serve is running. This page reloads when it's back.",
    )
  }
  if (e instanceof SyntaxError) {
    return titled('json', "The JSON doesn't parse")
  }
  const known =
    e instanceof ApiError ? byStatus(e.status, e.code, message, what) : null
  return known ?? titled('unknown', 'Something went wrong')
}

/** The one mapping from any failure to a title a person can read and, where
 * there is one, what to do about it. `what` names the thing a 404 is about. */
export function describeError(e: unknown, what?: string): Described {
  const message = e instanceof Error ? e.message : String(e)
  const { kind, title, remedy } = classify(e, message, what ?? 'That')
  return {
    kind,
    title,
    ...(remedy ? { remedy } : {}),
    ...(message && message !== title ? { detail: message } : {}),
    ...(e instanceof ApiError
      ? { status: e.status, ...(e.code ? { code: e.code } : {}) }
      : {}),
  }
}
