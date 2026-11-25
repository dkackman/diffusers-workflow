// The prompt editor's "Enhance with AI" panel, as state: the presets the
// server offers, the model and idea picked, a model download in flight, and
// the enhancement job that expands the idea. It lives on the page rather
// than in the panel, so switching the editor's view - which unmounts the
// panel - keeps a download or a job going.
import { TERMINAL_STATUSES, api, fetchOutputText, streamJobEvents } from './api'
import { ApiError, describeError } from './apiErrors'
import { onReconnect } from './connection.svelte'
import { sleep } from './poll'
import { phaseLabel } from './progress'
import {
  knownIntendedModels,
  manifestTextFile,
  presetForIntendedModel,
} from './prompts'
import type {
  EnhancerPreset,
  JobDetail,
  JobEvent,
  ModelRepo,
  PromptDetail,
} from './types'

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
  /** What went wrong, in plain words */
  error = $state('')
  /** The server's own text behind `error`, for Details */
  errorDetail = $state('')
  result = $state('')
  jobId = $state('')
  /** The workspace the job runs in, which its link names */
  jobWorkspace = $state('')
  // The server reported no presets, or could not be asked
  down = $state(false)
  // Ends the following of the job in flight
  #unfollow: (() => void) | null = null
  // Raised by stop(), so a queueing it overtook opens no stream
  #stops = 0

  /** Say what went wrong: our words for it, the engine's kept beside */
  #say(words: string, detail = '') {
    this.error = words
    this.errorDetail = detail
  }
  /** A caught failure while `doing`, through describeError; `what` names
   * the thing a 404 is about */
  #fail(doing: string, e: unknown, what?: string) {
    const described = describeError(e, what)
    this.#say(`Couldn't ${doing}: ${described.title}`, described.detail)
  }

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
    this.#say('')
    try {
      await api.startDownload(this.model)
      // Poll until this repo's download leaves the active list
      while (this.downloading) {
        await sleep(2000)
        const { downloads } = await api.listDownloads()
        const mine = downloads.find((d) => d.repo_id === this.model)
        if (!mine || mine.status !== 'downloading') {
          if (mine?.status === 'failed')
            this.#say('The model download failed', mine.error ?? '')
          break
        }
      }
    } catch (e) {
      this.#fail('download the model', e)
    } finally {
      this.downloading = false
      this.refreshModels()
    }
  }

  async generate() {
    if (!this.idea.trim()) {
      this.#say('Describe the idea to expand first')
      return
    }
    this.busy = true
    this.#say('')
    this.result = ''
    this.status = 'queueing…'
    const stops = this.#stops
    try {
      const job = await api.enhance({
        idea: this.idea,
        preset: this.presetKey,
        model_name: this.model || undefined,
        device: this.device || undefined,
      })
      // The page went away while the job was being queued
      if (stops !== this.#stops) {
        this.busy = false
        this.status = ''
        return
      }
      this.jobId = job.id
      this.jobWorkspace = job.workspace
      if (job.queue_position !== undefined) {
        this.status = `queued · #${job.queue_position + 1} in line`
      }
      this.#follow(job.id)
    } catch (e) {
      this.#fail('start the enhancement', e)
      this.busy = false
      this.status = ''
    }
  }

  #progress(event: JobEvent) {
    if (event.event === 'log') this.status = String(event.message)
    // The enhancer is one task step - its phase is the whole story,
    // and 'loading' is most of the wait on a cold model
    else if (event.event === 'phase') this.status = phaseLabel(event) + '…'
    else if (event.event === 'job_status') this.status = String(event.status)
  }

  /** Follow the job until it ends. EventSource repairs a dropped connection
   * itself but gives up on an HTTP error (a restarted server no longer has
   * the job), and then no event says so: each drop reads the job, and a
   * reconnect reopens the stream from the last event it delivered. */
  #follow(jobId: string) {
    let lastSeq = -1
    let close: (() => void) | null = null
    const unfollow = () => {
      close?.()
      unwatch()
      this.#unfollow = null
    }
    const end = () => {
      // A drop's read and the stream can both see the end; one acts
      if (this.#unfollow !== unfollow) return
      unfollow()
      void this.#finish(jobId)
    }
    const dropped = async () => {
      let job: JobDetail
      try {
        job = await api.getJob(jobId)
      } catch (e) {
        // Gone ends it; any other failure waits for the next drop or the
        // reconnect
        if (e instanceof ApiError && e.status === 404) end()
        return
      }
      if (job.historical || TERMINAL_STATUSES.includes(job.status)) end()
    }
    const open = () => {
      close?.()
      close = streamJobEvents(
        jobId,
        lastSeq,
        (event) => {
          lastSeq = Math.max(lastSeq, event.seq)
          this.#progress(event)
        },
        end,
        () => void dropped(),
      )
    }
    const unwatch = onReconnect(open)
    this.#unfollow = unfollow
    open()
  }

  async #finish(jobId: string) {
    try {
      const detail = await api.getJob(jobId)
      if (detail.status !== 'succeeded') {
        this.#say(`The enhancement ${detail.status}`, detail.error ?? '')
        return
      }
      const file = manifestTextFile(detail.manifest)
      if (!file) {
        this.#say('The enhancement produced no text')
        return
      }
      this.result = (await fetchOutputText(file, detail.workspace)).trim()
    } catch (e) {
      this.#fail('read the enhancement', e, 'The job')
    } finally {
      this.busy = false
      this.status = ''
      this.jobId = ''
      this.jobWorkspace = ''
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
    this.#stops++
    this.#unfollow?.()
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
