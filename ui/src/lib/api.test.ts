import { afterEach, describe, expect, it, vi } from 'vitest'
import { api, fetchOutputText, setNote, streamJobEvents } from './api'
import { galleryThumbnailUrl } from './apiUrls'

type Stub = {
  ok: boolean
  status?: number
  body?: unknown
  text?: string
  headers?: Record<string, string>
}

/** Install a fetch stub and record every URL and init it was called with. */
function stubFetch(response: Stub) {
  const calls: Array<[string, RequestInit | undefined]> = []
  vi.stubGlobal('fetch', (url: string, init?: RequestInit) => {
    calls.push([url, init])
    return Promise.resolve({
      ok: response.ok,
      status: response.status ?? (response.ok ? 200 : 500),
      statusText: 'Internal Server Error',
      headers: new Headers(response.headers),
      json: () =>
        response.body === undefined
          ? Promise.reject(new Error('not json'))
          : Promise.resolve(response.body),
      text: () => Promise.resolve(response.text ?? ''),
    })
  })
  return calls
}

afterEach(() => vi.unstubAllGlobals())

describe('request error handling', () => {
  it('surfaces the detail FastAPI puts in the body', async () => {
    stubFetch({ ok: false, status: 400, body: { detail: 'Unknown variable' } })
    await expect(api.listJobs()).rejects.toThrow('Unknown variable')
  })

  it("carries the detail's code, for a caller that branches on it", async () => {
    stubFetch({
      ok: false,
      status: 409,
      body: {
        detail: {
          code: 'stale_revision',
          message: 'request is at revision 5, not 3',
          current_revision: 5,
        },
      },
    })
    const failure = await api.getRequest('default', 'r1').catch((e) => e)
    expect(failure.message).toBe('request is at revision 5, not 3')
    expect(failure.status).toBe(409)
    expect(failure.code).toBe('stale_revision')
    stubFetch({ ok: false, status: 400, body: { detail: 'Unknown variable' } })
    expect((await api.listJobs().catch((e) => e)).code).toBeUndefined()
  })

  it('reads the message and counts out of an object-shaped detail', async () => {
    stubFetch({
      ok: false,
      status: 409,
      body: {
        detail: {
          message: 'Deleting workspace shots removes these files permanently.',
          contents: {
            workflows: { files: 2, bytes: 400 },
            outputs: { files: 3, bytes: 4 * 1024 * 1024 },
            assets: { files: 0, bytes: 0 },
          },
        },
      },
    })
    await expect(api.deleteWorkspace('shots')).rejects.toThrow(
      /Deleting workspace shots[\s\S]*workflows: 2 files, outputs: 3 files \(4\.0 MB\)/,
    )
  })

  it('reads the message and entries out of a not-a-workspace refusal', async () => {
    stubFetch({
      ok: false,
      status: 409,
      body: { detail: { message: "'shots' holds more", entries: ['notes'] } },
    })
    await expect(api.deleteWorkspace('shots', true)).rejects.toThrow(
      /holds more[\s\S]*notes/,
    )
  })

  it('falls back to the status text when the body is not JSON', async () => {
    stubFetch({ ok: false, status: 500 })
    await expect(api.listJobs()).rejects.toThrow('Internal Server Error')
  })

  it('falls back when the JSON body carries no detail', async () => {
    stubFetch({ ok: false, status: 500, body: { message: 'nope' } })
    await expect(api.listJobs()).rejects.toThrow('Internal Server Error')
  })
})

describe('file downloads', () => {
  it('fail with the same ApiError, detail and status, as every request', async () => {
    stubFetch({ ok: false, status: 404, body: { detail: 'Unknown file' } })
    const failure = await api.archiveOutputs(['a.png']).catch((e) => e)
    expect(failure.name).toBe('ApiError')
    expect(failure.message).toBe('Unknown file')
    expect(failure.status).toBe(404)
  })
})

describe('name encoding', () => {
  it('keeps folder separators in a workflow name but escapes the segments', async () => {
    const calls = stubFetch({ ok: true, body: {} })
    await api.getWorkflow('video/my workflow')
    expect(calls[0][0]).toBe('/api/workflows/video/my%20workflow')
  })

  it('escapes a name that would otherwise change the path', async () => {
    const calls = stubFetch({ ok: true, body: {} })
    await api.getPrompt('minimax/fox?dawn')
    expect(calls[0][0]).toBe('/api/prompts/minimax/fox%3Fdawn')
  })

  it('escapes a repo id in the delete query string', async () => {
    const calls = stubFetch({ ok: true, body: {} })
    await api.deleteModel('acme/tiny-model')
    expect(calls[0][0]).toBe('/api/models?repo=acme%2Ftiny-model')
    expect(calls[0][1]?.method).toBe('DELETE')
  })

  it('escapes a gallery file name in both metadata and delete', async () => {
    const calls = stubFetch({ ok: true, body: {} })
    await api.galleryMetadata('wf-gen.0-0.0 copy.png')
    await api.deleteOutput('wf-gen.0-0.0 copy.png')
    expect(calls[0][0]).toBe('/api/gallery/wf-gen.0-0.0%20copy.png/metadata')
    expect(calls[1][0]).toBe('/api/gallery/wf-gen.0-0.0%20copy.png')
  })

  it('keeps the workflow-folder separator in a gallery name but escapes the segments', async () => {
    const calls = stubFetch({ ok: true, body: {} })
    await api.galleryMetadata('ltx/wf-gen.0-0.0 copy.png')
    await api.deleteOutput('ltx/wf-gen.0-0.0 copy.png')
    expect(calls[0][0]).toBe(
      '/api/gallery/ltx/wf-gen.0-0.0%20copy.png/metadata',
    )
    expect(calls[1][0]).toBe('/api/gallery/ltx/wf-gen.0-0.0%20copy.png')
  })

  it("GETs a file's assess route, scoped to the picker's workspace", async () => {
    const { workspace, DEFAULT_WORKSPACE } = await import('./workspace.svelte')
    const calls = stubFetch({ ok: true, body: {} })
    await api.assess('ltx/run-1/final/clip one.mp4')
    expect(calls[0][0]).toBe(
      '/api/gallery/ltx/run-1/final/clip%20one.mp4/assess',
    )
    expect(calls[0][1]?.method ?? 'GET').toBe('GET')
    workspace.current = 'studio'
    try {
      await api.assess('a.mp4')
      expect(calls[1][0]).toBe('/api/gallery/a.mp4/assess?workspace=studio')
    } finally {
      workspace.current = DEFAULT_WORKSPACE
    }
  })
})

describe('job event log', () => {
  it('sends after, limit and each kind as a repeated parameter', async () => {
    const calls = stubFetch({ ok: true, body: { events: [] } })
    await api.eventLog('j1', {
      after: 41,
      limit: 500,
      kinds: ['log', 'warning'],
    })
    expect(calls[0][0]).toBe(
      '/api/jobs/j1/journal?after=41&limit=500&kinds=log&kinds=warning',
    )
    expect(calls[0][1]?.method ?? 'GET').toBe('GET')
  })

  it('sends no query when nothing narrows it, and keeps after=-1', async () => {
    const calls = stubFetch({ ok: true, body: { events: [] } })
    await api.eventLog('j1')
    await api.eventLog('j1', { after: -1 })
    expect(calls[0][0]).toBe('/api/jobs/j1/journal')
    expect(calls[1][0]).toBe('/api/jobs/j1/journal?after=-1')
  })

  it('fails with the 404 status for an unknown job', async () => {
    stubFetch({ ok: false, status: 404, body: { detail: 'Unknown job' } })
    const failure = await api.eventLog('gone').catch((e) => e)
    expect(failure.status).toBe(404)
  })
})

describe('job listing', () => {
  it('passes workspace, status set and limit as query parameters', async () => {
    const calls = stubFetch({ ok: true, body: { jobs: [], total: 0 } })
    await api.listJobs('studio', 5, 'succeeded,failed,cancelled')
    expect(calls[0][0]).toBe(
      '/api/jobs?workspace=studio&status=succeeded%2Cfailed%2Ccancelled&limit=5',
    )
  })

  it('asks for every job when nothing narrows it', async () => {
    const calls = stubFetch({ ok: true, body: { jobs: [], total: 0 } })
    await api.listJobs()
    expect(calls[0][0]).toBe('/api/jobs')
  })

  it('asks for prompts only when told to', async () => {
    const calls = stubFetch({ ok: true, body: { jobs: [], total: 0 } })
    await api.listJobs('studio', undefined, undefined, undefined, undefined, {
      prompts: true,
    })
    expect(calls[0][0]).toBe('/api/jobs?workspace=studio&prompts=true')
  })
})

describe('workflow definition fetch', () => {
  it('keeps the definition exactly as served, with origin and writable beside it', async () => {
    const body = { id: 'z-image', steps: [] }
    stubFetch({
      ok: true,
      body,
      headers: {
        'X-Workflow-Origin': 'workspace',
        'X-Workflow-Writable': 'true',
      },
    })
    const result = await api.getWorkflow('models/z-image')
    expect(result.definition).toEqual(body)
    // The transport metadata rides beside the definition, not inside it -
    // a workflow opened by name must validate and save as the file it
    // came from, and the schema refuses unknown root keys
    expect(result.definition).not.toHaveProperty('origin')
    expect(result.definition).not.toHaveProperty('writable')
    expect(result.origin).toBe('workspace')
    expect(result.writable).toBe(true)
  })
})

describe('gallery listing and thumbnails', () => {
  it('fetches the whole listing in one request', async () => {
    const calls = stubFetch({ ok: true, body: {} })
    await api.gallery()
    expect(calls[0][0]).toBe('/api/gallery?limit=100000')
  })

  it('narrows to a request item, and sends no item without a request', async () => {
    const calls = stubFetch({ ok: true, body: {} })
    await api.gallery(undefined, 'r1', 'alpha')
    await api.gallery(undefined, undefined, 'alpha')
    expect(calls[0][0]).toBe('/api/gallery?limit=100000&request=r1&item=alpha')
    expect(calls[1][0]).toBe('/api/gallery?limit=100000')
  })

  it('asks for prompts and durations only when told to', async () => {
    const calls = stubFetch({ ok: true, body: {} })
    await api.gallery(undefined, undefined, undefined, {
      prompts: true,
      media: true,
    })
    await api.gallery(undefined, undefined, undefined, {})
    expect(calls[0][0]).toBe(
      '/api/gallery?limit=100000&prompts=true&media=true',
    )
    expect(calls[1][0]).toBe('/api/gallery?limit=100000')
  })

  it('narrows to one folder, and to the newest few when given a limit', async () => {
    const calls = stubFetch({ ok: true, body: {} })
    await api.gallery(undefined, undefined, undefined, {
      folder: 'templates/text-to-image',
      limit: 6,
    })
    await api.gallery(undefined, undefined, undefined, { folder: 'ltx/Gyre' })
    await api.gallery(undefined, undefined, undefined, { folder: '' })
    expect(calls[0][0]).toBe(
      '/api/gallery?folder=templates%2Ftext-to-image&limit=6',
    )
    expect(calls[1][0]).toBe('/api/gallery?folder=ltx%2FGyre&limit=100000')
    // '' is the output root's own files, still a folder to narrow to
    expect(calls[2][0]).toBe('/api/gallery?folder=&limit=100000')
  })

  it("scopes a thumbnail URL to the file's own workspace when one is named", () => {
    expect(galleryThumbnailUrl('run/clip.mp4', 'studio')).toBe(
      '/api/gallery/run/clip.mp4/thumbnail?workspace=studio',
    )
  })

  it('builds a thumbnail URL that keeps folder separators', () => {
    expect(galleryThumbnailUrl('ltx/wf-gen.0-0.0.png')).toBe(
      '/api/gallery/ltx/wf-gen.0-0.0.png/thumbnail',
    )
  })
})

describe('fetchOutputText', () => {
  it('reads the basename of a manifest path from the outputs mount', async () => {
    const calls = stubFetch({ ok: true, text: 'an enhanced prompt' })
    expect(await fetchOutputText('/abs/outputs/enhance-0.0.txt')).toBe(
      'an enhanced prompt',
    )
    expect(calls[0][0]).toBe('/outputs/enhance-0.0.txt')
  })

  it("reads from the job's own workspace, not the selected one", async () => {
    const calls = stubFetch({ ok: true, text: 'an enhanced prompt' })
    await fetchOutputText('enhance-0.0.txt', 'shots')
    expect(calls[0][0]).toBe('/outputs/enhance-0.0.txt?workspace=shots')
  })

  it('rejects with the file name when the read fails', async () => {
    stubFetch({ ok: false, status: 404 })
    await expect(fetchOutputText('/abs/outputs/gone.txt')).rejects.toThrow(
      'Could not read gone.txt',
    )
  })
})

describe('the event log read', () => {
  it("keeps clear of content blockers' /event-log? rule", async () => {
    const calls = stubFetch({ ok: true, body: { events: [] } })
    await api.eventLog('j1', { after: -1, limit: 500 })
    expect(calls[0][0]).toContain('/api/jobs/j1/journal?after=-1&limit=500')
    expect(calls[0][0]).not.toContain('/event-log?')
  })
})

describe('text reads through the shared request path', () => {
  const unreachable = () =>
    vi.stubGlobal('fetch', () =>
      Promise.reject(new TypeError('Failed to fetch')),
    )

  it('reads an output text that never arrived as the offline copy', async () => {
    const { connection } = await import('./connection.svelte')
    const { describeError } = await import('./apiErrors')
    unreachable()
    const failed = await fetchOutputText('enhance-0.0.txt').catch((e) => e)
    expect(describeError(failed).title).toBe("Can't reach the server")
    expect(connection.state).toBe('offline')
  })

  it("reads a gallery file's text that never arrived as the offline copy", async () => {
    const { describeError } = await import('./apiErrors')
    unreachable()
    const failed = await api.fileText('/outputs/a.txt').catch((e) => e)
    expect(describeError(failed).title).toBe("Can't reach the server")
  })

  it("reads a gallery file's text with the token, not re-scoped", async () => {
    const calls = stubFetch({ ok: true, text: 'hello' })
    expect(await api.fileText('/outputs/a.txt?workspace=shots')).toBe('hello')
    expect(calls[0][0]).toBe('/outputs/a.txt?workspace=shots')
  })
})

describe('output URLs', () => {
  it('keeps the subfolder of an output-dir-relative name', async () => {
    const { outputUrl } = await import('./api')
    expect(outputUrl('flux/FluxDev-gen.0.png', 'job1')).toBe(
      '/outputs/flux/FluxDev-gen.0.png?v=job1',
    )
  })

  it('falls back to the basename for a legacy absolute path', async () => {
    const { outputUrl } = await import('./api')
    expect(outputUrl('/abs/outputs/a.png', 'job1')).toBe(
      '/outputs/a.png?v=job1',
    )
  })

  it('reads a nested text output from its subfolder', async () => {
    const calls = stubFetch({ ok: true, text: 'x' })
    await fetchOutputText('enhance/enhance-0.0.txt')
    expect(calls[0][0]).toBe('/outputs/enhance/enhance-0.0.txt')
  })

  it('attaches the API token to thumbnail URLs, which load via <img>', async () => {
    const { setApiToken } = await import('./token')
    setApiToken('s3cr3t')
    try {
      expect(galleryThumbnailUrl('a.png')).toBe(
        '/api/gallery/a.png/thumbnail?token=s3cr3t',
      )
    } finally {
      setApiToken('')
    }
    expect(galleryThumbnailUrl('a.png')).toBe('/api/gallery/a.png/thumbnail')
  })
})

describe('rerunJob', () => {
  const body = (calls: Array<[string, RequestInit | undefined]>) =>
    JSON.parse(String(calls[0][1]?.body))

  it('sends new_seed alone for a plain Run again', async () => {
    const calls = stubFetch({ ok: true, body: { id: 'j2' } })
    await api.rerunJob('j1')
    expect(calls[0][0]).toContain('/api/jobs/j1/rerun')
    expect(body(calls)).toEqual({ new_seed: false })
  })

  it('sends new_seed true for New seed', async () => {
    const calls = stubFetch({ ok: true, body: { id: 'j2' } })
    await api.rerunJob('j1', { newSeed: true })
    expect(body(calls)).toEqual({ new_seed: true })
  })

  it('leaves arguments out when there are none', async () => {
    const calls = stubFetch({ ok: true, body: { id: 'j2' } })
    await api.rerunJob('j1', { arguments: {} })
    expect(body(calls)).toEqual({ new_seed: false })
  })

  it('sends the changed arguments beside new_seed', async () => {
    const calls = stubFetch({ ok: true, body: { id: 'j2' } })
    await api.rerunJob('j1', { arguments: { tier: 'standard' } })
    expect(body(calls)).toEqual({
      new_seed: false,
      arguments: { tier: 'standard' },
    })
  })
})

describe('job export', () => {
  it("posts to the job's own workspace, not the selected one", async () => {
    const calls = stubFetch({
      ok: true,
      status: 201,
      body: {
        directory: '/ws/shots/exports/j1',
        zip_url: '/exports/j1.zip?workspace=shots',
        files: [],
        total_bytes: 0,
        missing: [],
      },
    })
    await api.exportJob('j1', 'shots')
    expect(calls[0][0]).toBe(
      '/api/jobs/j1/export?workspace=shots&overwrite=false',
    )
    expect(calls[0][1]?.method).toBe('POST')
  })

  it('sends no selector for the default workspace and passes overwrite through', async () => {
    const calls = stubFetch({ ok: true, status: 201, body: {} })
    await api.exportJob('j1', 'default', true)
    expect(calls[0][0]).toBe('/api/jobs/j1/export?overwrite=true')
  })

  it('surfaces the 409 status so the page can offer to replace', async () => {
    const { ApiError } = await import('./api')
    stubFetch({ ok: false, status: 409, body: { detail: 'already exists' } })
    const error = await api.exportJob('j1', 'default').catch((e) => e)
    expect(error).toBeInstanceOf(ApiError)
    expect(error.status).toBe(409)
    expect(error.message).toBe('already exists')
  })

  it("adds the token to the server's zip URL without scoping it again", async () => {
    const { setApiToken } = await import('./token')
    setApiToken('s3cr3t')
    try {
      expect(api.exportZipUrl('/exports/j1.zip?workspace=shots')).toBe(
        '/exports/j1.zip?workspace=shots&token=s3cr3t',
      )
    } finally {
      setApiToken('')
    }
    expect(api.exportZipUrl('/exports/j1.zip')).toBe('/exports/j1.zip')
  })
})

describe('setNote', () => {
  it('scopes to a named workspace, not the picker', async () => {
    const calls = stubFetch({ ok: true, body: {} })
    await setNote('wf/run-1', 'n', 'studio')
    expect(calls[0][0]).toBe('/api/notes?workspace=studio')
    expect(calls[0][1]?.method).toBe('PUT')
    expect(JSON.parse(String(calls[0][1]?.body))).toEqual({
      target: 'wf/run-1',
      note: 'n',
    })
  })

  it("uses the picker's workspace when none is named", async () => {
    const calls = stubFetch({ ok: true, body: {} })
    await setNote('wf/run-1', null)
    expect(calls[0][0]).toBe('/api/notes')
  })

  it('sends If-Match only when given an etag', async () => {
    const calls = stubFetch({ ok: true, body: {} })
    await setNote('wf/run-1', 'n', 'studio', '"e1"')
    await setNote('wf/run-1', 'n', 'studio')
    const sent = (i: number) => new Headers(calls[i][1]?.headers)
    expect(sent(0).get('If-Match')).toBe('"e1"')
    expect(sent(1).get('If-Match')).toBeNull()
  })

  it('reads a note and its etag, scoped like the write', async () => {
    const calls = stubFetch({
      ok: true,
      body: { target: 'wf/run 1', note: null, etag: '"e"' },
    })
    expect((await api.noteForSave('wf/run 1', 'studio')).etag).toBe('"e"')
    expect(calls[0][0]).toBe('/api/notes?target=wf%2Frun%201&workspace=studio')
  })
})

describe('request notes', () => {
  it('adds, edits and deletes through the request-notes route, unscoped', async () => {
    const calls = stubFetch({ ok: true, body: { revision: 4 } })
    await api.addNote('my ws', {
      request: 'r1',
      item: 's3',
      text: 'warmer',
      refs: [{ item: 's3', job: 'j2' }],
    })
    await api.editNote('my ws', { request: 'r1', note: 'n2', text: 'cooler' })
    await api.deleteNote('my ws', 'r1', 'n2')
    await api.notesChannel('my ws')
    expect(calls.map(([url, init]) => [init?.method ?? 'GET', url])).toEqual([
      ['POST', '/api/workspaces/my%20ws/request-notes'],
      ['PUT', '/api/workspaces/my%20ws/request-notes'],
      ['DELETE', '/api/workspaces/my%20ws/request-notes?request=r1&note=n2'],
      ['GET', '/api/workspaces/my%20ws/notes-channel'],
    ])
    expect(JSON.parse(String(calls[0][1]?.body))).toEqual({
      request: 'r1',
      item: 's3',
      text: 'warmer',
      refs: [{ item: 's3', job: 'j2' }],
    })
    expect(JSON.parse(String(calls[1][1]?.body))).toEqual({
      request: 'r1',
      note: 'n2',
      text: 'cooler',
    })
  })
})

describe('write preconditions', () => {
  it('saveWorkflow sends If-Match and reads the ETag', async () => {
    const calls = stubFetch({
      ok: true,
      body: { saved: 'hero' },
      headers: { ETag: '"new"' },
    })
    const result = await api.saveWorkflow(
      'hero',
      { id: 'hero', steps: [] } as never,
      { ifMatch: '"old"' },
    )
    expect(result).toEqual({ saved: { saved: 'hero' }, etag: '"new"' })
    const headers = new Headers(calls[0][1]?.headers)
    expect(calls[0][1]?.method).toBe('PUT')
    expect(headers.get('If-Match')).toBe('"old"')
    expect(headers.get('If-None-Match')).toBeNull()
    expect(headers.get('Content-Type')).toBe('application/json')
  })

  it('createOnly sends If-None-Match: *', async () => {
    const calls = stubFetch({ ok: true, body: {} })
    const result = await api.savePrompt('fox', { text: 'a fox' } as never, {
      createOnly: true,
    })
    expect(result.etag).toBeNull()
    expect(new Headers(calls[0][1]?.headers).get('If-None-Match')).toBe('*')
    await api.uploadMedia(new File(['x'], 'photo.jpg'), 'photo.jpg', false, {
      createOnly: true,
    })
    expect(new Headers(calls[1][1]?.headers).get('If-None-Match')).toBe('*')
  })

  it('sends no precondition header without one', async () => {
    const calls = stubFetch({ ok: true, body: {} })
    await api.saveWorkflow('hero', { id: 'hero', steps: [] } as never)
    const headers = new Headers(calls[0][1]?.headers)
    expect(headers.get('If-Match')).toBeNull()
    expect(headers.get('If-None-Match')).toBeNull()
  })

  it('reads the ETag a get sends', async () => {
    stubFetch({ ok: true, body: { text: 'a' }, headers: { ETag: '"p"' } })
    expect((await api.getPrompt('fox')).etag).toBe('"p"')
    stubFetch({ ok: true, body: {}, headers: { ETag: '"w"' } })
    expect((await api.getWorkflow('hero')).etag).toBe('"w"')
  })
})

describe('job event stream', () => {
  /** A stand-in EventSource that records itself, so a test can fail it */
  class FakeSource {
    static last: FakeSource | null = null
    onmessage: ((message: { data: string }) => void) | null = null
    onerror: (() => void) | null = null
    closed = false
    constructor(readonly url: string) {
      FakeSource.last = this
    }
    close() {
      this.closed = true
    }
  }

  it('onDrop is called on a stream error and the stream is left open', () => {
    vi.stubGlobal('EventSource', FakeSource)
    const onDrop = vi.fn()
    streamJobEvents(
      'j1',
      -1,
      () => {},
      () => {},
      onDrop,
    )
    const source = FakeSource.last!
    source.onerror?.()
    expect(onDrop).toHaveBeenCalledTimes(1)
    // EventSource repairs a dropped connection itself; closing would stop it
    expect(source.closed).toBe(false)
  })
})

describe('an id is one path segment', () => {
  // a job id reaches these from the page's hash, already percent-decoded
  const id = 'a/../b?x=1#y'
  const encoded = 'a%2F..%2Fb%3Fx%3D1%23y'

  it.each([
    ['getJob', () => api.getJob(id), `/api/jobs/${encoded}`],
    ['getJobWorkflow', () => api.getJobWorkflow(id), `/workflow`],
    ['eventLog', () => api.eventLog(id), `/journal`],
    ['rerunJob', () => api.rerunJob(id), `/rerun`],
    ['rerunRestartBatch', () => api.rerunRestartBatch(id), `-batch`],
    ['restartBatch', () => api.restartBatch(id), `/restart-batch`],
    ['moveJob', () => api.moveJob(id, 'up'), `/move`],
    ['exportJob', () => api.exportJob(id, 'default'), `/export`],
    ['cancelJob', () => api.cancelJob(id), `/cancel`],
    ['cancelDownload', () => api.cancelDownload(id), `/cancel`],
    ['describePipeline', () => api.describePipeline(id), encoded],
  ])('%s', async (_, call, tail) => {
    const calls = stubFetch({ ok: true, body: {} })
    await call()
    const url = new URL(calls[0][0], 'http://dw')
    expect(url.pathname).toContain(encoded)
    expect(url.pathname.endsWith(tail.split('/').pop()!)).toBe(true)
    expect(url.searchParams.has('x')).toBe(false)
    expect(url.hash).toBe('')
  })

  it.each(['.', '..'])('%s is not walked as a dot segment', async (dots) => {
    const calls = stubFetch({ ok: true, body: {} })
    await api.cancelJob(dots)
    const { pathname } = new URL(calls[0][0], 'http://dw')
    expect(pathname).toMatch(/^\/api\/jobs\/(%252E){1,2}\/cancel$/)
  })

  it('in the event stream URL', () => {
    const urls: string[] = []
    vi.stubGlobal(
      'EventSource',
      class {
        constructor(url: string) {
          urls.push(url)
        }
        close() {}
      },
    )
    streamJobEvents(
      id,
      0,
      () => {},
      () => {},
    )
    expect(urls[0]).toContain(`/api/jobs/${encoded}/events?after=`)
  })

  it('and the class kind is one query value', async () => {
    const calls = stubFetch({ ok: true, body: {} })
    await api.listClasses('a&b=1')
    expect(calls[0][0]).toContain('/api/classes?kind=a%26b%3D1')
  })
})
