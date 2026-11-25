/** A chained step's segments as table rows, read off its manifest entry (#1475).
 *
 * A chain is one step: its segments reach the record only as the shots its
 * manifest entry carries at `step_end` (`dw/shots.py` lists every key). A
 * step counts as chained when a shot carries `planned_segments`. Each shot
 * is one row: its thumbnail (the segment's own clip under `keep_segments`,
 * else the take seeked to the segment's start), how many renders its check
 * tried, each measure its kept attempt computed against the ceiling in
 * force, and whether it was copied from an earlier run (`resumed`) or
 * spliced in by repair-segment (`(repaired)`).
 *
 * The verdict of a measure is the engine's `failed`, never value against
 * ceiling: `texture_ratio` is bounded both ways, so only the check knows.
 * A failed warn-only measure (re-rolling never fixes it) reads `warn`.
 *
 * A shot with no `measures` key is one of three things, told apart by the
 * rest of the record: a resumed segment (copied, not re-checked), a checked
 * chain recorded before measures existed (`attempts` present - "not
 * recorded"), or a chain run with `check` off (no `attempts` - no measure
 * cells at all). Pure, so the rules are tested without a DOM. */

import type { ManifestEntry, WorkflowDefinition } from './types'
import { MEMBER_SEPARATOR, VARIABLE, referenceName } from './references'

type Shot = NonNullable<ManifestEntry['shots']>[number]

/** The measures the engine names, in the order its check names a failure
 * (`MEASURES`, dw/pipeline_processors/chain_check.py). Any other name the
 * record carries is shown after these, as it is named. */
const KNOWN_ORDER = [
  'jump_ratio',
  'look_ratio',
  'texture_ratio',
  'cut_ratio',
  'audio_dbfs',
  'words',
  'face_drop',
]

export type Verdict = 'pass' | 'warn' | 'fail'

export interface MeasureCell {
  name: string
  value: number
  ceiling: number | null
  bound: 'max' | 'min' | null
  warnOnly: boolean
  verdict: Verdict
}

/** What the row's measure cells show:
 * - `recorded`: one cell per measure (possibly none: a segment 1 the check
 *   had nothing to hold to)
 * - `not-recorded`: a checked shot from before the record kept measures
 * - `resumed`: copied from an earlier run, not checked again
 * - `unchecked`: the chain ran with `check` off; no cells */
export type MeasureState = 'recorded' | 'not-recorded' | 'resumed' | 'unchecked'

export interface Thumb {
  /** The output path of the file the thumbnail plays */
  file: string
  /** Seconds into it, when the frame rate is known; null otherwise */
  seconds: number | null
  /** The same place as a fraction of the file's length, for a player to
   * seek once it knows the duration */
  fraction: number
}

export interface ChainRow {
  /** 1-based, in take order */
  index: number
  name: string
  startFrame: number
  numFrames: number
  thumb: Thumb | null
  attempts: number | null
  /** Renders the check could try: 1 + the chain's `retries`, when known */
  budget: number | null
  /** The kept re-roll's seed: only when a re-roll (not the first render) was kept */
  keptSeed: number | null
  measureState: MeasureState
  measures: Record<string, MeasureCell>
  resumed: boolean
  repaired: boolean
  /** The output path of the segment's own clip, when the run kept it */
  segmentFile: string | null
}

export interface ChainTable {
  step: string
  rows: ChainRow[]
  /** The output path of the take the chain wrote, when the entry lists it */
  take: string | null
  /** The plan's segment count, which a `stop_after` take falls short of */
  planned: number | null
  /** Every measure name any row carries, known names first in the engine's
   * order, then any other alphabetically - the table's measure columns */
  measureNames: string[]
}

export interface ChainContext {
  /** The run's frame rate, for a `#t=` seek into the take */
  fps?: number | null
  /** The chain's `retries`: a checked segment may render 1 + retries times */
  retries?: number | null
}

const SEGMENT_FILE = /\.segment-\d{3}(-\d+)?\.mp4$/i

const isNumber = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value)

const basename = (path: string) => path.slice(path.lastIndexOf('/') + 1)
const dirname = (path: string) => path.slice(0, path.lastIndexOf('/') + 1)

/** Whether a manifest entry is a chain's: a shot carries `planned_segments` */
export function isChained(entry: ManifestEntry | null | undefined): boolean {
  return Boolean(entry?.shots?.some((shot) => isNumber(shot.planned_segments)))
}

/** The take: the entry's video that is not one of its segment files */
function takeFile(entry: ManifestEntry): string | null {
  const segments = new Set(
    (entry.shots ?? [])
      .map((shot) => shot.segment_file)
      .filter((f): f is string => typeof f === 'string'),
  )
  return (
    entry.files.find(
      (file) =>
        /\.(mp4|webm|mov|mkv)$/i.test(file) &&
        !SEGMENT_FILE.test(file) &&
        !segments.has(basename(file)),
    ) ?? null
  )
}

/** A segment file's output path: the entry's own file of that name when it
 * lists one, else beside the take - the chain writes them side by side */
function segmentPath(
  entry: ManifestEntry,
  take: string | null,
  name: string,
): string {
  const listed = entry.files.find((file) => basename(file) === name)
  if (listed) return listed
  return take ? dirname(take) + name : name
}

function measureCell(name: string, raw: unknown): MeasureCell | null {
  if (raw === null || typeof raw !== 'object') return null
  const m = raw as Record<string, unknown>
  if (!isNumber(m.value)) return null
  const failed = m.failed === true
  const warnOnly = m.warn_only === true
  return {
    name,
    value: m.value,
    ceiling: isNumber(m.ceiling) ? m.ceiling : null,
    bound: m.bound === 'min' || m.bound === 'max' ? m.bound : null,
    warnOnly,
    verdict: failed ? (warnOnly ? 'warn' : 'fail') : 'pass',
  }
}

function orderNames(names: Iterable<string>): string[] {
  const all = [...new Set(names)]
  const known = KNOWN_ORDER.filter((name) => all.includes(name))
  const other = all.filter((name) => !KNOWN_ORDER.includes(name)).sort()
  return [...known, ...other]
}

/** What one chain's rows share: where its take is, how long it runs, and
 * what the definition says of its frame rate and attempt budget */
interface RowContext {
  entry: ManifestEntry
  take: string | null
  totalFrames: number
  fps: number | null
  budget: number | null
  /** A chain with the check on records `attempts` on every shot it ran */
  checked: boolean
}

const orZero = (value: unknown) => (isNumber(value) ? value : 0)

function shotSegmentFile(shot: Shot, ctx: RowContext): string | null {
  return typeof shot.segment_file === 'string' && shot.segment_file
    ? segmentPath(ctx.entry, ctx.take, shot.segment_file)
    : null
}

function shotThumb(shot: Shot, ctx: RowContext): Thumb | null {
  const file = shotSegmentFile(shot, ctx)
  if (file) return { file, seconds: 0, fraction: 0 }
  if (!ctx.take) return null
  const start = orZero(shot.start_frame)
  return {
    file: ctx.take,
    seconds: ctx.fps ? start / ctx.fps : null,
    fraction: ctx.totalFrames ? start / ctx.totalFrames : 0,
  }
}

function shotMeasures(
  shot: Shot,
  ctx: RowContext,
): Pick<ChainRow, 'measureState' | 'measures'> {
  const measures: Record<string, MeasureCell> = {}
  if (shot.measures && typeof shot.measures === 'object') {
    for (const name of orderNames(Object.keys(shot.measures))) {
      const cell = measureCell(name, shot.measures[name])
      if (cell) measures[name] = cell
    }
    return { measureState: 'recorded', measures }
  }
  if (shot.resumed === true) return { measureState: 'resumed', measures }
  if (ctx.checked) return { measureState: 'not-recorded', measures }
  return { measureState: 'unchecked', measures }
}

function shotRow(shot: Shot, i: number, ctx: RowContext): ChainRow {
  const attempts = isNumber(shot.attempts) ? shot.attempts : null
  const reRolled = attempts !== null && attempts > 1
  return {
    index: i + 1,
    name: String(shot.name ?? `segment ${i + 1}`),
    startFrame: orZero(shot.start_frame),
    numFrames: orZero(shot.num_frames),
    thumb: shotThumb(shot, ctx),
    attempts,
    budget: attempts !== null ? ctx.budget : null,
    keptSeed: reRolled && isNumber(shot.seed) ? shot.seed : null,
    ...shotMeasures(shot, ctx),
    resumed: shot.resumed === true,
    repaired: String(shot.name ?? '').includes('(repaired)'),
    segmentFile: shotSegmentFile(shot, ctx),
  }
}

/** A chained step's rows; null for a step that is not a chain's */
export function chainRows(
  entry: ManifestEntry | null | undefined,
  context: ChainContext = {},
): ChainTable | null {
  if (!entry || !isChained(entry)) return null
  const shots: Shot[] = entry.shots ?? []
  const { fps, retries } = context
  const ctx: RowContext = {
    entry,
    take: takeFile(entry),
    totalFrames: shots.reduce((sum, shot) => sum + orZero(shot.num_frames), 0),
    fps: isNumber(fps) && fps > 0 ? fps : null,
    budget: isNumber(retries) && retries >= 0 ? 1 + Math.floor(retries) : null,
    checked: shots.some(
      (shot) => isNumber(shot.attempts) || shot.measures !== undefined,
    ),
  }
  const rows = shots.map((shot, i) => shotRow(shot, i, ctx))
  const planned = shots.map((shot) => shot.planned_segments).filter(isNumber)
  return {
    step: entry.step,
    rows,
    take: ctx.take,
    planned: planned.length ? Math.max(...planned) : null,
    measureNames: orderNames(rows.flatMap((row) => Object.keys(row.measures))),
  }
}

/** The thumbnail's src: a `#t=` fragment seeks the browser's first paint */
export function thumbSrc(url: string, thumb: Thumb): string {
  if (thumb.seconds === null) return url
  // A frame just past zero, so a segment clip paints its first frame too
  const seconds = Math.max(thumb.seconds, 0.001)
  return `${url}#t=${Number(seconds.toFixed(3))}`
}

/** "2 of 3", "2 of 3 tried", or "-" for a shot whose check never ran */
export function attemptsText(row: ChainRow): string {
  if (row.attempts === null) return '-'
  if (row.budget !== null) return `${row.attempts} of ${row.budget}`
  return row.attempts === 1 ? '1 attempt' : `${row.attempts} attempts`
}

/** A measure value, 3 significant figures: ratios and dBFS alike */
export function formatMeasure(value: number): string {
  if (value === 0) return '0'
  const abs = Math.abs(value)
  if (abs >= 100) return value.toFixed(0)
  return Number(value.toPrecision(3)).toString()
}

/** Value against its ceiling, as the bound reads: `0.42 ≤ 0.6`, `-18 ≥ -40` */
export function measureText(cell: MeasureCell): string {
  const value = formatMeasure(cell.value)
  if (cell.ceiling === null) return `${value} (no ceiling)`
  const op = cell.bound === 'min' ? '≥' : '≤'
  return `${value} ${op} ${formatMeasure(cell.ceiling)}`
}

// ---- reading the chain's settings off the definition the job ran ----

type Definition = WorkflowDefinition | null | undefined

/** A value as the run resolved it: a literal, or a `variable:` the job's
 * arguments (or the definition's own default) supplied. Anything else -
 * a reference the UI cannot follow - is unknown */
export function resolved(
  value: unknown,
  definition: Definition,
  args: Record<string, unknown> | null | undefined,
): unknown {
  const name = referenceName(value, VARIABLE)
  if (name === null) return value
  if (args && name in args) return args[name]
  return definition?.variables?.[name]
}

export function definitionStep(definition: Definition, step: string) {
  const at = step.indexOf(MEMBER_SEPARATOR)
  const name = at === -1 ? step : step.slice(0, at)
  const steps = Array.isArray(definition?.steps) ? definition.steps : []
  return steps.find((s) => s?.name === name) ?? null
}

/** The step's frame rate and `retries`, best effort: a step the definition
 * does not hold at its top level (a sub-workflow's) gives neither */
export function chainContext(
  definition: Definition,
  step: string,
  args?: Record<string, unknown> | null,
): ChainContext {
  const found = definitionStep(definition, step)
  const pipeline = found?.pipeline ?? found?.pipeline_reference
  const chain = pipeline?.chain ?? {}
  const call = pipeline?.arguments ?? {}
  const number = (value: unknown) => {
    const v = resolved(value, definition, args)
    return isNumber(v) ? v : null
  }
  // The schema's default is 0; a reference the UI cannot follow is unknown
  const retries = () =>
    chain.retries === undefined ? 0 : number(chain.retries)
  return {
    fps: number(call.frame_rate) ?? number(call.fps),
    retries: found ? retries() : null,
  }
}
