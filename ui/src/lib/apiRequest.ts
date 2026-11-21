// What is under every call in api.ts: the request with its token, scope and
// ApiError, the file-download variants, and the bodies a call sends that
// the generated response types do not name.
import type {
  LedgerItem,
  LedgerRequest,
  Plan,
  WorkflowDefinition,
} from './types'
import { getApiToken } from './token'
import { ApiError, errorCode, errorDetail, errorPayload } from './apiErrors'
import { scoped } from './apiUrls'

/** A JSON body sent with `method` */
export const jsonInit = (method: string, body: unknown): RequestInit => ({
  method,
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body),
})

/** A request with the bearer token, scoped to the picker's workspace
 * unless `scope` is false, failing with an ApiError that carries the
 * server's detail and status. Under every JSON call and file download. */
async function send(
  path: string,
  init: RequestInit = {},
  scope = true,
): Promise<Response> {
  const token = getApiToken()
  const headers = new Headers(init.headers)
  if (token) headers.set('Authorization', `Bearer ${token}`)
  const response = await fetch(scope ? scoped(path) : path, {
    ...init,
    headers,
  })
  if (!response.ok) {
    let detail = response.statusText
    let code: string | undefined
    let held: Record<string, unknown> | undefined
    try {
      const payload = await response.json()
      detail = errorDetail(payload, detail)
      code = errorCode(payload)
      held = errorPayload(payload)
    } catch {
      /* not json */
    }
    throw new ApiError(detail, response.status, code, held)
  }
  return response
}

/** A JSON response together with its headers, for the rare endpoint whose
 * result depends on both - `getWorkflow` reads its origin/writable from
 * headers rather than the body. */
export async function fetchJson<T>(
  path: string,
  init?: RequestInit,
  options?: { scope?: boolean },
): Promise<{ body: T; response: Response }> {
  const response = await send(path, init, options?.scope !== false)
  return { body: await response.json(), response }
}

export async function request<T>(
  path: string,
  init?: RequestInit,
  options?: { scope?: boolean },
): Promise<T> {
  return (await fetchJson<T>(path, init, options)).body
}

/** Fetch a file-response endpoint and hand the body to the browser as a
 * save, using the name the server put in Content-Disposition. Separate
 * from `request` because the body is a file, not JSON, and separate from
 * the `<a download>` URLs because the request needs a method and a body. */
async function downloadResponse(
  path: string,
  init: RequestInit,
): Promise<void> {
  const response = await send(path, init)
  const disposition = response.headers.get('content-disposition') ?? ''
  const filename = /filename="?([^"]+)"?/.exec(disposition)?.[1] ?? 'download'
  const url = URL.createObjectURL(await response.blob())
  try {
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = filename
    anchor.click()
  } finally {
    URL.revokeObjectURL(url)
  }
}

/** A bulk-download endpoint: POST the selection, save the zip that comes
 * back. The gallery and the asset library each have one. */
export function archiveFrom(path: string) {
  return (names: string[]) =>
    downloadResponse(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ names }),
    })
}

/** What the validate route is asked: exactly one of an inline `workflow` or a
 * `workflow_path` (a catalog name), the `arguments` a run would carry, and
 * `sizes: false` to skip asking the hub how large a missing model is. */
export type ValidateRequest = {
  workflow?: WorkflowDefinition
  workflow_path?: string
  arguments?: Record<string, unknown>
  /** The workspace it resolves in, when not the open one (body first) */
  workspace?: string
  sizes?: boolean
}

/** A definition never carries `workflow` or `workflow_path` at its root -
 * the schema refuses unknown root keys - so either names a request. */
export function isValidateRequest(
  input: WorkflowDefinition | ValidateRequest,
): input is ValidateRequest {
  return (
    Object.hasOwn(input, 'workflow') || Object.hasOwn(input, 'workflow_path')
  )
}

/** An item as a plan write sends it: as read, less its notes, which only
 * the request-notes routes write */
export type WrittenItem = Omit<LedgerItem, 'notes'>

/** A request's plan as PUT sends it: the read view's items go back as they
 * came (the server ignores their derived fields) but for their notes,
 * `prompt` is left out because it is write-once. */
export type RequestWriteBody = {
  title: string
  goal: string
  items: WrittenItem[]
  revision: number
  why: string
  state?: LedgerRequest['state']
}

/** A cost acknowledgement bound to a plan: the `{fingerprint, minutes,
 * downloads}` a 409 hands back to resend, or the validate plan itself */
export type BoundAcknowledgement =
  Plan | { fingerprint: string; minutes?: number | null; downloads?: string[] }

/** What a job submit sends. `request` and `item` (both or neither) link
 * the job to an item of the workspace's requests ledger; with
 * `acknowledged_cost` bound to a plan, the server refuses (409) a run
 * whose plan is no longer that one. */
export type JobSubmitBody = {
  workflow_path?: string
  workflow?: WorkflowDefinition
  arguments?: Record<string, unknown>
  base_dir?: string
  request?: string
  item?: string
  acknowledged_cost?: true | BoundAcknowledgement
}
