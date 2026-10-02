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

/** An error response, with the status a caller may need to branch on -
 * the export's 409 is "already exported, overwrite?" rather than a
 * failure, and the message alone cannot say which. */
export class ApiError extends Error {
  status: number
  constructor(message: string, status: number) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}
