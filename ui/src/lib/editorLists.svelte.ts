import { api } from './api'

/** The class and file names the workflow editor suggests, fetched once by
 * the editor page and read by the inputs nested under it - one shared
 * source rather than a <datalist> id threaded through the markup. */
export const editorLists = $state({
  pipelines: [] as string[],
  modelClasses: [] as string[],
  schedulerClasses: [] as string[],
  quantizationClasses: [] as string[],
  taskCommands: [] as string[],
  workflowFiles: [] as string[],
  /** The packaged fragments a sub-workflow step names as builtin: */
  builtinWorkflows: [] as string[],
})

/** Fetch the class lists and task commands the editor suggests. The
 * workflow files arrive with the listing the editor reads anyway, so the
 * page fills those itself. A list whose read fails keeps what it had: the
 * inputs it feeds just suggest less. */
export function loadEditorLists(): void {
  const keep = () => {}
  api
    .listPipelines()
    .then((r) => {
      editorLists.pipelines = r.pipelines
    })
    .catch(keep)
  api
    .listClasses('models')
    .then((r) => {
      editorLists.modelClasses = r.classes
    })
    .catch(keep)
  api
    .listClasses('schedulers')
    .then((r) => {
      editorLists.schedulerClasses = r.classes
    })
    .catch(keep)
  api
    .listClasses('quantization')
    .then((r) => {
      editorLists.quantizationClasses = r.classes
    })
    .catch(keep)
  api
    .listTasks()
    .then((r) => {
      editorLists.taskCommands = [
        ...r.commands,
        ...r.image_processors,
        ...r.video_processors,
      ].sort()
    })
    .catch(keep)
  // An older server has no such route, and the call may not even exist -
  // the path field just offers the library
  Promise.resolve()
    .then(() => api.listBuiltinWorkflows())
    .then((r) => {
      editorLists.builtinWorkflows = r.workflows
    })
    .catch(keep)
}
