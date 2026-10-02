// The prompt editor's "Enhance with AI" panel, as state: the presets the
// server offers, the model and idea picked, a model download in flight, and
// the enhancement job that expands the idea. It lives on the page rather
// than in the panel, so switching the editor's view - which unmounts the
// panel - keeps a download or a job going.
import { api, fetchOutputText, streamJobEvents } from './api'
import { sleep } from './poll'
import { phaseLabel } from './progress'
import {
  knownIntendedModels,
  manifestTextFile,
  presetForIntendedModel,
} from './prompts'
import type { EnhancerPreset, ModelRepo, PromptDetail } from './types'

const message = (e: unknown) => (e instanceof Error ? e.message : String(e))

export class Enhancer {
  presets = $state<EnhancerPreset[]>([])
  presetKey = $state('')
  model = $state('')
  idea = $state('')
  device = $state('')
  models = $state<ModelRepo[]>([])
  downloading = $state(false)
  busy = $state(false)
  status = $state('')
  error = $state('')
  result = $state('')
  jobId = $state('')
  // The server reported no presets, or could not be asked
  down = $state(false)
  #stopStream: (() => void) | null = null

  readonly preset = $derived(this.presets.find((p) => p.key === this.presetKey))
  readonly modelCached = $derived(
    this.models.some((repo) => repo.repo_id === this.model),
  )

  /** Every intended model worth suggesting: the presets' own, then the
   * ones prompts in the library already name. */
  intendedModels(details: Record<string, PromptDetail>): string[] {
    return knownIntendedModels(
      this.presets,
      Object.values(details).map((detail) => detail.intended_model),
    )
  }

  async load() {
    try {
      this.presets = (await api.listEnhancers()).presets
      this.down = this.presets.length === 0
    } catch {
      // Editing still works without the enhancer - but say so, rather
      // than leaving a permanently disabled Generate button unexplained
      this.down = true
    }
  }

  // Switching preset resets the model to that preset's default; re-picking
  // the current one leaves a hand-typed model alone
  pickPreset(key: string) {
    const picked = this.presets.find((p) => p.key === key)
    if (!picked || key === this.presetKey) return
    this.presetKey = key
    this.model = picked.default_model
  }

  /** Pick the preset `intended` names. With no match the current selection
   * stands, so an ltx-2 prompt doesn't get the H3 enhancer. */
  preselect(intended: string | undefined) {
    if (!this.presets.length) return
    const picked = presetForIntendedModel(this.presets, intended)
    if (picked) this.pickPreset(picked.key)
    else if (!this.presetKey) this.pickPreset(this.presets[0].key)
  }

  async refreshModels() {
    try {
      this.models = (await api.listModels()).repos
    } catch {
      /* cache indicator stays pessimistic */
    }
  }

  async downloadModel() {
    if (!this.model) return
    this.downloading = true
    this.error = ''
    try {
      await api.startDownload(this.model)
      // Poll until this repo's download leaves the active list
      while (this.downloading) {
        await sleep(2000)
        const { downloads } = await api.listDownloads()
        const mine = downloads.find((d) => d.repo_id === this.model)
        if (!mine || mine.status !== 'downloading') {
          if (mine?.status === 'failed')
            this.error = mine.error ?? 'download failed'
          break
        }
      }
    } catch (e) {
      this.error = message(e)
    } finally {
      this.downloading = false
      this.refreshModels()
    }
  }

  async generate() {
    if (!this.idea.trim()) {
      this.error = 'Describe the idea to expand first'
      return
    }
    this.busy = true
    this.error = ''
    this.result = ''
    this.status = 'queueing…'
    try {
      const job = await api.enhance({
        idea: this.idea,
        preset: this.presetKey,
        model_name: this.model || undefined,
        device: this.device || undefined,
      })
      this.jobId = job.id
      if (job.queue_position !== undefined) {
        this.status = `queued · #${job.queue_position + 1} in line`
      }
      this.#stopStream = streamJobEvents(
        job.id,
        -1,
        (event) => {
          if (event.event === 'log') this.status = String(event.message)
          // The enhancer is one task step - its phase is the whole story,
          // and 'loading' is most of the wait on a cold model
          else if (event.event === 'phase')
            this.status = phaseLabel(event) + '…'
          else if (event.event === 'job_status')
            this.status = String(event.status)
        },
        () => this.#finish(job.id),
      )
    } catch (e) {
      this.error = message(e)
      this.busy = false
      this.status = ''
    }
  }

  async #finish(jobId: string) {
    this.#stopStream = null
    try {
      const detail = await api.getJob(jobId)
      if (detail.status !== 'succeeded') {
        this.error = detail.error ?? `enhancement ${detail.status}`
        return
      }
      const file = manifestTextFile(detail.manifest)
      if (!file) {
        this.error = 'The enhancement produced no text'
        return
      }
      this.result = (await fetchOutputText(file, detail.workspace)).trim()
    } catch (e) {
      this.error = message(e)
    } finally {
      this.busy = false
      this.status = ''
      this.jobId = ''
    }
  }

  async cancel() {
    if (!this.jobId) return
    try {
      await api.cancelJob(this.jobId)
    } catch {
      /* already finished */
    }
  }

  /** Stop following the job and end a download's polling - the page is
   * going away. */
  stop() {
    this.#stopStream?.()
    this.#stopStream = null
    this.downloading = false
  }

  /** The result, and the provenance a prompt records for it; the result
   * is cleared, since it now lives in the prompt. */
  takeResult() {
    const taken = {
      text: this.result,
      enhanced: { model: this.model, idea: this.idea },
    }
    this.result = ''
    return taken
  }
}
