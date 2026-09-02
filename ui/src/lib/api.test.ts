import { afterEach, describe, expect, it, vi } from 'vitest'
import { api, fetchOutputText } from './api'

type Stub = { ok: boolean; status?: number; body?: unknown; text?: string }

/** Install a fetch stub and record every URL and init it was called with. */
function stubFetch(response: Stub) {
  const calls: Array<[string, RequestInit | undefined]> = []
  vi.stubGlobal('fetch', (url: string, init?: RequestInit) => {
    calls.push([url, init])
    return Promise.resolve({
      ok: response.ok,
      status: response.status ?? (response.ok ? 200 : 500),
      statusText: 'Internal Server Error',
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

  it('falls back to the status text when the body is not JSON', async () => {
    stubFetch({ ok: false, status: 500 })
    await expect(api.listJobs()).rejects.toThrow('Internal Server Error')
  })

  it('falls back when the JSON body carries no detail', async () => {
    stubFetch({ ok: false, status: 500, body: { message: 'nope' } })
    await expect(api.listJobs()).rejects.toThrow('Internal Server Error')
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
    expect(calls[0][0]).toBe('/api/gallery/ltx/wf-gen.0-0.0%20copy.png/metadata')
    expect(calls[1][0]).toBe('/api/gallery/ltx/wf-gen.0-0.0%20copy.png')
  })
})

describe('gallery pagination and thumbnails', () => {
  it('sends limit and offset, and omits folder when unset', async () => {
    const calls = stubFetch({ ok: true, body: {} })
    await api.gallery(120, 240)
    expect(calls[0][0]).toBe('/api/gallery?limit=120&offset=240')
  })

  it('sends an explicit folder filter, including the root folder as an empty string', async () => {
    const calls = stubFetch({ ok: true, body: {} })
    await api.gallery(50, 0, 'ltx')
    await api.gallery(50, 0, '')
    expect(calls[0][0]).toBe('/api/gallery?limit=50&offset=0&folder=ltx')
    expect(calls[1][0]).toBe('/api/gallery?limit=50&offset=0&folder=')
  })

  it('builds a thumbnail URL that keeps folder separators', () => {
    expect(api.galleryThumbnailUrl('ltx/wf-gen.0-0.0.png')).toBe(
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

  it('rejects with the file name when the read fails', async () => {
    stubFetch({ ok: false, status: 404 })
    await expect(fetchOutputText('/abs/outputs/gone.txt')).rejects.toThrow(
      'Could not read gone.txt',
    )
  })
})
