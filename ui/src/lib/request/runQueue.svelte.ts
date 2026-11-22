// Pricing and queuing the items of a run: every candidate is validated
// before the dialog offers to spend (three at a time, dropped when the
// dialog closes), and on confirm each is submitted in plan order with a
// cost acknowledgement bound to the plan it was quoted. Nothing here is
// optimistic: a row is queued only when the server has said so.
import { ApiError } from '../apiErrors'
import type { JobSubmitBody, ValidateRequest } from '../apiRequest'
import type { JobDetail, Plan, ValidationResult } from '../types'
import { RUN_REFUSALS, type Candidate } from './runPlan'

/** How many validations are out at once */
export const PRICING_LANES = 3

export type RunApi = {
  validate: (
    body: ValidateRequest,
    signal?: AbortSignal,
  ) => Promise<ValidationResult>
  submitJob: (body: JobSubmitBody) => Promise<JobDetail>
}

export type RowState =
  /** The validation is out */
  | 'pricing'
  /** Quoted: it can be run */
  | 'priced'
  /** Validation refused it, or could not be made: it will not run */
  | 'refused'
  | 'sending'
  /** The server queued a job for it */
  | 'queued'
  /** The plan moved since the quote: `plan` is the new one, to accept */
  | 'repriced'
  /** The submit was refused: `error` says why */
  | 'failed'

export type RunRow = {
  candidate: Candidate
  state: RowState
  ticked: boolean
  /** The plan quoted; null when validation answered with none */
  plan: Plan | null
  /** What to resend after a re-price, as the server gave it */
  acknowledge: JobSubmitBody['acknowledged_cost'] | null
  error: string | null
  job: string | null
}

const message = (error: unknown) =>
  error instanceof Error ? error.message : String(error)

/** A submit the server did not take, in words for the row */
function refusal(error: unknown): string {
  if (!(error instanceof ApiError)) return message(error)
  const plain = error.code ? RUN_REFUSALS[error.code] : undefined
  if (plain) return plain
  if (error.status === 400)
    return `The item's stored workflow or arguments were refused: ${error.message}`
  return error.message
}

/** A validation that did not answer. The server refusing the question
 * itself - a workflow it cannot reach, arguments it will not take - is
 * what a submit would meet too; anything else is not about the item. */
function unchecked(error: unknown): string {
  const refused =
    error instanceof ApiError && error.status >= 400 && error.status < 500
  return `${refused ? 'would be refused' : 'could not be checked'}: ${message(error)}`
}

/** The current plan a 409 on a bound acknowledgement carries */
function repriced(error: unknown): Record<string, unknown> | null {
  if (!(error instanceof ApiError) || error.status !== 409) return null
  const detail = error.detail
  return detail?.plan && detail.acknowledge ? detail : null
}

export class RunQueue {
  rows = $state<RunRow[]>([])
  /** A submit is out */
  busy = $state(false)
  #api: RunApi
  #workspace: string
  #request: string
  #abort = new AbortController()

  constructor(
    candidates: Candidate[],
    options: { workspace: string; requestId: string; api: RunApi },
  ) {
    this.#api = options.api
    this.#workspace = options.workspace
    this.#request = options.requestId
    this.rows = candidates.map((candidate) => ({
      candidate,
      state: 'pricing',
      ticked: true,
      plan: null,
      acknowledge: null,
      error: null,
      job: null,
    }))
  }

  /** Every row has an answer: only then may the dialog offer to spend */
  get settled() {
    return this.rows.every((row) => row.state !== 'pricing')
  }
  /** The rows the Run button would queue */
  get chosen() {
    return this.rows.filter((row) => row.ticked && row.state === 'priced')
  }
  /** Something was sent: the choice of rows is closed */
  get started() {
    return this.rows.some((row) =>
      ['sending', 'queued', 'repriced', 'failed'].includes(row.state),
    )
  }
  get queued() {
    return this.rows.filter((row) => row.state === 'queued')
  }
  /** Every row the user chose is queued, and there is at least one */
  get finished() {
    const asked = this.rows.filter(
      (row) => row.ticked && row.state !== 'refused',
    )
    return asked.length > 0 && asked.every((row) => row.state === 'queued')
  }

  /** Validate every candidate, `PRICING_LANES` at a time, in plan order */
  async price(): Promise<void> {
    let next = 0
    const lane = async () => {
      while (next < this.rows.length && !this.#abort.signal.aborted)
        await this.#priceRow(this.rows[next++])
    }
    await Promise.all(Array.from({ length: PRICING_LANES }, lane))
  }

  async #priceRow(row: RunRow) {
    const { workflow, arguments: args } = row.candidate
    try {
      const result = await this.#api.validate(
        {
          workflow_path: workflow,
          arguments: args,
          workspace: this.#workspace,
        },
        this.#abort.signal,
      )
      if (this.#abort.signal.aborted) return
      if (result.valid) {
        row.plan = result.plan ?? null
        row.state = 'priced'
        return
      }
      const first = result.error ?? result.errors[0]?.message ?? 'not valid'
      row.error = `would be refused: ${first}`
    } catch (error) {
      if (this.#abort.signal.aborted) return
      row.error = unchecked(error)
    }
    row.state = 'refused'
    row.ticked = false
  }

  /** The dialog closed: drop the validations still out. A submit already
   * sent is the server's now and is not taken back. */
  close() {
    this.#abort.abort()
  }

  /** Queue the chosen rows in plan order, each to its own outcome; a row
   * that fails or re-prices does not stop the ones after it */
  async submit(): Promise<void> {
    if (this.busy || !this.settled) return
    this.busy = true
    try {
      for (const row of this.chosen) await this.#send(row, row.plan ?? true)
    } finally {
      this.busy = false
    }
  }

  /** Queue a re-priced row at the price it now shows */
  async accept(key: string): Promise<void> {
    const row = this.rows.find((r) => r.candidate.key === key)
    if (this.busy || row?.state !== 'repriced' || !row.acknowledge) return
    this.busy = true
    try {
      await this.#send(row, row.acknowledge)
    } finally {
      this.busy = false
    }
  }

  async #send(row: RunRow, acknowledged: JobSubmitBody['acknowledged_cost']) {
    const { key, workflow, arguments: args } = row.candidate
    row.state = 'sending'
    row.error = null
    try {
      const job = await this.#api.submitJob({
        workflow_path: workflow,
        arguments: args,
        request: this.#request,
        item: key,
        acknowledged_cost: acknowledged,
      })
      row.job = job.id
      row.state = 'queued'
    } catch (error) {
      const moved = repriced(error)
      if (moved) {
        row.plan = moved.plan as Plan
        row.acknowledge = moved.acknowledge as RunRow['acknowledge']
        row.state = 'repriced'
      } else {
        row.error = refusal(error)
        row.state = 'failed'
      }
    }
  }
}
