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
})

/** Fetch the class lists and task commands the editor suggests. The
 * workflow files arrive with the listing the editor reads anyway, so the
 * page fills those itself. */
export function loadEditorLists(): void {
  api.listPipelines().then((r) => (editorLists.pipelines = r.pipelines))
  api.listClasses('models').then((r) => (editorLists.modelClasses = r.classes))
  api
    .listClasses('schedulers')
    .then((r) => (editorLists.schedulerClasses = r.classes))
  api
    .listClasses('quantization')
    .then((r) => (editorLists.quantizationClasses = r.classes))
  api
    .listTasks()
    .then(
      (r) =>
        (editorLists.taskCommands = [
          ...r.commands,
          ...r.image_processors,
          ...r.video_processors,
        ].sort()),
    )
}
