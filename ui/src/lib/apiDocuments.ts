// The library documents the editors open and save - workflows and prompts -
// spread into `api` by api.ts. A read carries the ETag of the bytes it
// returned and a save takes WritePreconditions, so an editor can refuse to
// overwrite a file that changed since it was opened (409 `stale_write`) or
// one that already exists under a new name (409 `exists`).
import type {
  PromptDefinition,
  PromptSaved,
  StoredPrompt,
  WorkflowDefinition,
  WorkflowSaved,
  WorkflowWithOrigin,
} from './types'
import {
  fetchJson,
  putWithPreconditions,
  type WritePreconditions,
} from './apiRequest'
import { encodePath } from './apiUrls'

export const documentRoutes = {
  /** The workflow plus where it came from, read off the response headers
   * rather than a separate `listWorkflows` lookup. Beside the definition,
   * the way `getPrompt` keeps a prompt's: the object the caller holds is
   * what validate, save and run send back, and the schema refuses unknown
   * root keys, so the transport metadata must not ride inside it. */
  getWorkflow: (name: string) =>
    fetchJson<WorkflowDefinition>(`/api/workflows/${encodePath(name)}`).then(
      ({ body, response }): WorkflowWithOrigin => ({
        definition: body,
        origin: response.headers.get('X-Workflow-Origin') ?? '',
        writable: response.headers.get('X-Workflow-Writable') !== 'false',
        etag: response.headers.get('ETag'),
      }),
    ),

  saveWorkflow: (
    name: string,
    workflow: WorkflowDefinition,
    pre?: WritePreconditions,
  ) =>
    putWithPreconditions<WorkflowSaved>(
      `/api/workflows/${encodePath(name)}`,
      { workflow },
      pre,
    ),

  /** The prompt plus which library it came from, read off the response
   * headers rather than a separate `listPrompts` lookup. */
  getPrompt: (name: string) =>
    fetchJson<PromptDefinition>(`/api/prompts/${encodePath(name)}`).then(
      ({ body, response }): StoredPrompt => ({
        prompt: body,
        origin: response.headers.get('X-Prompt-Origin') ?? '',
        writable: response.headers.get('X-Prompt-Writable') !== 'false',
        etag: response.headers.get('ETag'),
      }),
    ),
  savePrompt: (
    name: string,
    prompt: PromptDefinition,
    pre?: WritePreconditions,
  ) =>
    putWithPreconditions<PromptSaved>(
      `/api/prompts/${encodePath(name)}`,
      { prompt },
      pre,
    ),
}
