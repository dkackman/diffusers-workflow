import { beforeEach, expect, it, vi } from 'vitest'
import { editorLists, loadEditorLists } from './editorLists.svelte'
import { api, resetApi } from './test/apiMock'

vi.mock('./api', async () => (await import('./test/apiMock')).apiModule())

beforeEach(resetApi)

// A rejection nothing handles fails the run on its own: vitest reports it
it('keeps what a failed list had and still fills the others', async () => {
  editorLists.pipelines = ['Kept']
  const refused = () => Promise.reject(new Error('refused'))
  api.listPipelines.mockImplementation(refused)
  api.listClasses.mockImplementation(refused)
  api.listBuiltinWorkflows.mockImplementation(refused)
  api.listTasks.mockResolvedValue({
    commands: ['b'],
    image_processors: ['a'],
    video_processors: [],
  })
  loadEditorLists()
  await vi.waitFor(() => expect(editorLists.taskCommands).toEqual(['a', 'b']))
  // A macrotask, so a rejection left unhandled is reported inside this test
  await new Promise((done) => setTimeout(done, 10))
  expect(editorLists.pipelines).toEqual(['Kept'])
})
