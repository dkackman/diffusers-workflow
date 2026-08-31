# Editor Step Model Implementation Plan (UI/UX pass, stages 1–2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the workflow editor a legible step model: rich collapsed summaries, three-state density (collapsed/compact/full), an ordinal rail with producer/consumer chips driven by a real flow graph, and on-demand knob discovery.

**Architecture:** Two pure modules (`flow.ts` for the `previous_result:` edge graph, `digest.ts` for textual step summaries) feed presentation changes in `StepEditor.svelte` and `EditorPage.svelte`. A tiny `storage.ts` wrapper handles persistence. No backend/API changes — everything is under `ui/`.

**Tech Stack:** Svelte 5 (runes), TypeScript, Vitest (unit), Playwright (e2e), plain CSS with custom-property tokens.

**Spec:** `docs/superpowers/specs/2026-08-31-ui-ux-optimization-design.md` (this plan implements build-order stages 1–2; the spec's toast store is deferred to the stage-3 plan, which is its first consumer).

## Global Constraints

- All changes under `ui/` only. No API, schema, or Python changes.
- Svelte 5 runes syntax (`$state`, `$derived`, `$props`, `$bindable`) — no legacy stores/`$:` syntax.
- Execution order is linear but data flow is a fan-in/fan-out DAG (3 image steps → 1 video step). Nothing may assume a chain.
- Container-query breakpoint scale is 400 / 640 / 900 px (comment-enforced; `@container` cannot read custom properties).
- All commands run from `ui/`: `npm test` (vitest), `npm run check`, `npm run lint`, `npx playwright test` (starts its own server via `e2e/serve_fixture.py`).
- Format with `npm run format` before each commit (prettier owns style).
- Reuse over rewrite where sensible (user preference) — but the modules here are small, domain-specific pure functions; no library earns its place in this plan's scope.

---

### Task 1: Storage wrapper

**Files:**
- Create: `ui/src/lib/storage.ts`
- Test: `ui/src/lib/storage.test.ts`

**Interfaces:**
- Consumes: nothing.
- Produces: `storageGet<T>(key: string, fallback: T): T` and `storageSet(key: string, value: unknown): void`. Keys are auto-prefixed `dw-`; values are JSON round-tripped; every failure (private mode, quota, bad JSON) returns the fallback / no-ops. Later tasks persist step view state through these.

- [ ] **Step 1: Write the failing test**

```ts
// ui/src/lib/storage.test.ts
import { afterEach, describe, expect, it, vi } from 'vitest'
import { storageGet, storageSet } from './storage'

function stubStorage(backing: Record<string, string>) {
  vi.stubGlobal('localStorage', {
    getItem: (k: string) => (k in backing ? backing[k] : null),
    setItem: (k: string, v: string) => {
      backing[k] = v
    },
  })
}

afterEach(() => vi.unstubAllGlobals())

describe('storage', () => {
  it('round-trips JSON values under the dw- prefix', () => {
    const backing: Record<string, string> = {}
    stubStorage(backing)
    storageSet('step-modes:ZImage', { generate: 'compact' })
    expect(backing['dw-step-modes:ZImage']).toBe('{"generate":"compact"}')
    expect(storageGet('step-modes:ZImage', {})).toEqual({ generate: 'compact' })
  })

  it('returns the fallback for missing keys and corrupt JSON', () => {
    stubStorage({ 'dw-broken': '{not json' })
    expect(storageGet('missing', 'fb')).toBe('fb')
    expect(storageGet('broken', 'fb')).toBe('fb')
  })

  it('survives storage that throws (private mode)', () => {
    vi.stubGlobal('localStorage', {
      getItem: () => {
        throw new Error('denied')
      },
      setItem: () => {
        throw new Error('denied')
      },
    })
    expect(storageGet('anything', 42)).toBe(42)
    expect(() => storageSet('anything', 1)).not.toThrow()
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ui && npx vitest run src/lib/storage.test.ts`
Expected: FAIL — cannot resolve `./storage`

- [ ] **Step 3: Implement**

```ts
// ui/src/lib/storage.ts
/** One home for the app's persisted UI state. Keys are namespaced,
 * values are JSON, and storage that is missing, full, or forbidden
 * degrades to "state just doesn't persist". */
const PREFIX = 'dw-'

export function storageGet<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(PREFIX + key)
    return raw === null ? fallback : (JSON.parse(raw) as T)
  } catch {
    return fallback
  }
}

export function storageSet(key: string, value: unknown): void {
  try {
    localStorage.setItem(PREFIX + key, JSON.stringify(value))
  } catch {
    /* private mode or quota - session only */
  }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ui && npx vitest run src/lib/storage.test.ts`
Expected: PASS (3 tests)

- [ ] **Step 5: Format and commit**

```bash
cd ui && npm run format && cd ..
git add ui/src/lib/storage.ts ui/src/lib/storage.test.ts
git commit -m "Add namespaced localStorage wrapper for UI state"
```

---

### Task 2: Design tokens

**Files:**
- Modify: `ui/src/app.css` (token block at the top, after the theme variables)
- Modify: `ui/src/lib/editor/StepEditor.svelte:507`, `ui/src/lib/editor/ArgumentsEditor.svelte:220`, `ui/src/lib/pages/EditorPage.svelte:810,854` (breakpoint 420 → 400)

**Interfaces:**
- Consumes: nothing.
- Produces: CSS custom properties `--space-1` (0.25rem), `--space-2` (0.5rem), `--space-3` (0.7rem), `--space-4` (1rem), `--space-5` (1.5rem), `--radius-1` (6px), `--radius-2` (8px), available app-wide. Later tasks' new CSS uses tokens; existing CSS migrates only in files a task already touches.

- [ ] **Step 1: Add the token block to app.css**

In `ui/src/app.css`, after the `:root[data-theme='light']` block (line 66), add:

```css
/* Layout tokens. Spacing steps and radii for all new CSS; existing rules
   migrate as their files are touched. Container-query breakpoints cannot
   read custom properties, so the scale is a convention instead:
   @container at 400px (a panel too narrow for a label column) and 640px;
   viewport @media at 640px / 900px for page-level layout. */
:root {
  --space-1: 0.25rem;
  --space-2: 0.5rem;
  --space-3: 0.7rem;
  --space-4: 1rem;
  --space-5: 1.5rem;
  --radius-1: 6px;
  --radius-2: 8px;
}
```

- [ ] **Step 2: Align the stray container breakpoints**

The audit found ad-hoc thresholds (380/420/460). In the three files this plan touches anyway, change every `@container (max-width: 420px)` to `@container (max-width: 400px)`:
- `ui/src/lib/editor/StepEditor.svelte` (`.grid`/`.grid2` rule)
- `ui/src/lib/editor/ArgumentsEditor.svelte` (`.row` rule)
- `ui/src/lib/pages/EditorPage.svelte` (`.filegrid` and `.vars` rules)

Leave other files' thresholds for the stage-4 consistency sweep.

- [ ] **Step 3: Verify nothing broke**

Run: `cd ui && npm run check && npm run lint && npm test`
Expected: all pass (tokens are additive; breakpoint shift is 20px)

- [ ] **Step 4: Commit**

```bash
cd ui && npm run format && cd ..
git add ui/src/app.css ui/src/lib/editor/StepEditor.svelte ui/src/lib/editor/ArgumentsEditor.svelte ui/src/lib/pages/EditorPage.svelte
git commit -m "Add spacing/radius tokens and align editor container breakpoints to 400px"
```

---

### Task 3: Flow graph module

**Files:**
- Create: `ui/src/lib/flow.ts`
- Test: `ui/src/lib/flow.test.ts`
- Modify: `ui/src/lib/editor.ts:258-310` (`danglingReferences` delegates to the new module)

**Interfaces:**
- Consumes: nothing (pure functions over the workflow JSON).
- Produces:

```ts
export interface StepFlow {
  name: string
  inputs: string[] // earlier step names this step references (unique, in encounter order)
  consumers: string[] // later step names that reference this step
  resolvedRefs: number // total previous_result references that resolve (cartesian signal)
}
export interface DanglingReference {
  stepIndex: number
  message: string
}
export function flowGraph(workflow: Record<string, any>): StepFlow[]
export function danglingReferenceDetails(
  workflow: Record<string, any>,
  promptNames?: string[],
): DanglingReference[]
```

`editor.ts`'s existing `danglingReferences(workflow, promptNames): string[]` keeps its exact signature and messages (existing tests in `editor.test.ts` must stay green) by mapping over `danglingReferenceDetails`.

- [ ] **Step 1: Write the failing tests**

```ts
// ui/src/lib/flow.test.ts
import { describe, expect, it } from 'vitest'
import { danglingReferenceDetails, flowGraph } from './flow'

const step = (name: string, args: Record<string, unknown>) => ({
  name,
  task: { command: 'noop', arguments: args },
})

describe('flowGraph', () => {
  it('builds fan-in: three generators feeding one combiner, not each other', () => {
    const wf = {
      steps: [
        step('gen1', {}),
        step('gen2', {}),
        step('gen3', {}),
        step('video', {
          a: 'previous_result:gen1',
          b: 'previous_result:gen2',
          c: 'previous_result:gen3',
        }),
      ],
    }
    const graph = flowGraph(wf)
    expect(graph[0]).toEqual({
      name: 'gen1',
      inputs: [],
      consumers: ['video'],
      resolvedRefs: 0,
    })
    expect(graph[1].consumers).toEqual(['video'])
    expect(graph[3].inputs).toEqual(['gen1', 'gen2', 'gen3'])
    expect(graph[3].resolvedRefs).toBe(3)
  })

  it('builds fan-out: one producer with two consumers', () => {
    const wf = {
      steps: [
        step('gen', {}),
        step('up', { image: 'previous_result:gen' }),
        step('caption', { image: 'previous_result:gen' }),
      ],
    }
    const graph = flowGraph(wf)
    expect(graph[0].consumers).toEqual(['up', 'caption'])
  })

  it('resolves media suffixes to the base step and finds refs in nested values', () => {
    const wf = {
      steps: [
        step('gen', {}),
        step('mux', {
          nested: { list: ['previous_result:gen.frames'] },
          audio: 'previous_result:gen.audio',
        }),
      ],
    }
    const graph = flowGraph(wf)
    expect(graph[1].inputs).toEqual(['gen'])
    expect(graph[1].resolvedRefs).toBe(2)
  })

  it('only earlier steps are producers - a later or missing name is no edge', () => {
    const wf = {
      steps: [step('a', { x: 'previous_result:b' }), step('b', {})],
    }
    const graph = flowGraph(wf)
    expect(graph[0].inputs).toEqual([])
    expect(graph[1].consumers).toEqual([])
  })
})

describe('danglingReferenceDetails', () => {
  it('attributes each problem to its step index', () => {
    const wf = {
      variables: {},
      steps: [
        step('a', { x: 'variable:missing' }),
        step('b', { y: 'previous_result:nope' }),
      ],
    }
    const details = danglingReferenceDetails(wf)
    expect(details).toHaveLength(2)
    expect(details[0].stepIndex).toBe(0)
    expect(details[0].message).toContain('variable:missing')
    expect(details[1].stepIndex).toBe(1)
    expect(details[1].message).toContain('previous_result:nope')
  })
})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ui && npx vitest run src/lib/flow.test.ts`
Expected: FAIL — cannot resolve `./flow`

- [ ] **Step 3: Implement flow.ts**

```ts
// ui/src/lib/flow.ts
/** The workflow's data-flow graph. Execution is linear (steps run in
 * order) but data flow is a DAG: several steps can feed one consumer
 * (3 images -> 1 video) and one step can feed several. Everything the
 * editor says about flow - chips, highlights, reorder warnings, the
 * cartesian-product note - derives from here, and only from here. */

export interface StepFlow {
  name: string
  inputs: string[]
  consumers: string[]
  resolvedRefs: number
}

export interface DanglingReference {
  stepIndex: number
  message: string
}

/** Visit every string anywhere inside a JSON value. */
function scanStrings(value: unknown, visit: (s: string) => void): void {
  if (typeof value === 'string') visit(value)
  else if (Array.isArray(value)) value.forEach((v) => scanStrings(v, visit))
  else if (value !== null && typeof value === 'object')
    Object.values(value).forEach((v) => scanStrings(v, visit))
}

/** previous_result: target step name, media suffix stripped. */
function refTarget(s: string): string | null {
  if (!s.startsWith('previous_result:')) return null
  return s.slice('previous_result:'.length).split('.')[0]
}

export function flowGraph(workflow: Record<string, any>): StepFlow[] {
  const steps: Array<Record<string, any>> = workflow.steps ?? []
  const graph: StepFlow[] = steps.map((s) => ({
    name: s.name ?? '',
    inputs: [],
    consumers: [],
    resolvedRefs: 0,
  }))

  steps.forEach((step, index) => {
    // Only EARLIER steps are producers - the engine resolves in order
    const earlier = new Map<string, number>()
    steps.slice(0, index).forEach((s, i) => {
      if (s.name) earlier.set(s.name, i)
    })
    scanStrings(step, (s) => {
      const target = refTarget(s)
      if (target === null || !earlier.has(target)) return
      graph[index].resolvedRefs += 1
      if (!graph[index].inputs.includes(target)) {
        graph[index].inputs.push(target)
        graph[earlier.get(target)!].consumers.push(step.name ?? '')
      }
    })
  })
  return graph
}

export function danglingReferenceDetails(
  workflow: Record<string, any>,
  promptNames?: string[],
): DanglingReference[] {
  const problems: DanglingReference[] = []
  const variables = new Set(Object.keys(workflow.variables ?? {}))
  const prompts = promptNames === undefined ? null : new Set(promptNames)
  const steps: Array<Record<string, any>> = workflow.steps ?? []

  steps.forEach((step, index) => {
    const earlier = new Set(
      steps
        .slice(0, index)
        .map((s) => s.name)
        .filter(Boolean),
    )
    scanStrings(step, (value) => {
      if (value.startsWith('variable:')) {
        const name = value.slice('variable:'.length)
        if (!variables.has(name)) {
          problems.push({
            stepIndex: index,
            message: `Step '${step.name}': variable:${name} - no such variable is declared`,
          })
        }
      } else if (value.startsWith('previous_result:')) {
        const name = refTarget(value)!
        if (!earlier.has(name)) {
          problems.push({
            stepIndex: index,
            message: `Step '${step.name}': previous_result:${name} - no earlier step has that name`,
          })
        }
      } else if (value.startsWith('prompt:')) {
        // Without a listing the server resolves these at run time - only
        // a supplied library can say a name is missing
        const name = value.slice('prompt:'.length)
        if (prompts !== null && !prompts.has(name)) {
          problems.push({
            stepIndex: index,
            message: `Step '${step.name}': prompt:${name} - the prompt library has no such prompt`,
          })
        }
      }
    })
  })
  return problems
}
```

- [ ] **Step 4: Delegate editor.ts's danglingReferences**

In `ui/src/lib/editor.ts`, replace the whole body of `danglingReferences` (lines 258–310) with a delegation, keeping the exported signature and doc comment:

```ts
import { danglingReferenceDetails } from './flow'

/** References the workflow makes that nothing declares ... (keep the
 * existing doc comment). */
export function danglingReferences(
  workflow: Record<string, any>,
  promptNames?: string[],
): string[] {
  return danglingReferenceDetails(workflow, promptNames).map((d) => d.message)
}
```

(Add the import at the top of the file; delete the now-unused inline scan code.)

- [ ] **Step 5: Run the full unit suite**

Run: `cd ui && npm test`
Expected: PASS — including the pre-existing `danglingReferences` tests in `editor.test.ts`, unchanged

- [ ] **Step 6: Format and commit**

```bash
cd ui && npm run format && cd ..
git add ui/src/lib/flow.ts ui/src/lib/flow.test.ts ui/src/lib/editor.ts
git commit -m "Add flow graph module - single source of truth for previous_result edges"
```

---

### Task 4: Step digest module

**Files:**
- Create: `ui/src/lib/digest.ts`
- Test: `ui/src/lib/digest.test.ts`

**Interfaces:**
- Consumes: nothing (pure function over a step's JSON).
- Produces:

```ts
export type DigestSection =
  | 'main'
  | 'arguments'
  | 'components'
  | 'loras'
  | 'scheduler'
  | 'acceleration'
export interface DigestLine {
  text: string
  section: DigestSection
}
export interface StepDigest {
  summary: string // one line for the collapsed title bar
  lines: DigestLine[] // the compact view, one line per populated area
}
export function stepDigest(step: Record<string, any>): StepDigest
```

Task 5's StepEditor renders `summary` in the collapsed bar and `lines` in compact view; clicking a line switches to full view and opens `section`.

- [ ] **Step 1: Write the failing tests**

```ts
// ui/src/lib/digest.test.ts
import { describe, expect, it } from 'vitest'
import { stepDigest } from './digest'

describe('stepDigest', () => {
  it('summarizes a pipeline step: class, model, arg count', () => {
    const d = stepDigest({
      name: 'generate',
      pipeline: {
        configuration: { component_type: 'ZImagePipeline', offload: 'model' },
        from_pretrained_arguments: {
          model_name: 'Tongyi-MAI/Z-Image-Turbo',
          torch_dtype: 'torch.bfloat16',
        },
        arguments: { prompt: 'variable:prompt', num_inference_steps: 9 },
      },
      result: { content_type: 'image/png' },
    })
    expect(d.summary).toBe('ZImagePipeline · Tongyi-MAI/Z-Image-Turbo · 2 args')
    const texts = d.lines.map((l) => l.text)
    expect(texts).toContain(
      'Tongyi-MAI/Z-Image-Turbo · torch.bfloat16 · offload: model · save: image/png',
    )
    expect(texts).toContain('prompt = variable:prompt · num_inference_steps = 9')
  })

  it('digests populated optional sections and skips empty ones', () => {
    const d = stepDigest({
      name: 'generate',
      pipeline: {
        configuration: {
          component_type: 'FluxPipeline',
          cache: { type: 'first_block', threshold: 0.1 },
          attention_backend: 'sage',
        },
        from_pretrained_arguments: { model_name: 'x' },
        arguments: {},
        transformer: {
          configuration: { component_type: 'FluxTransformer2DModel' },
          from_pretrained_arguments: { model_name: 'y' },
          quantization_config: {
            config_type: 'BitsAndBytesConfig',
            arguments: { load_in_4bit: true },
          },
        },
        loras: [{ adapter_name: 'detail' }, { adapter_name: 'motion' }],
        scheduler: {
          configuration: { scheduler_type: 'UniPCMultistepScheduler' },
        },
      },
    })
    const bySection = Object.fromEntries(
      d.lines.map((l) => [l.section, l.text]),
    )
    expect(bySection.components).toBe('transformer (BitsAndBytesConfig)')
    expect(bySection.loras).toBe('2 LoRAs: detail, motion')
    expect(bySection.scheduler).toBe('UniPCMultistepScheduler')
    expect(bySection.acceleration).toBe(
      'cache: first_block · attention: sage',
    )
    expect(d.lines.find((l) => l.section === 'arguments')).toBeUndefined()
  })

  it('truncates long argument values', () => {
    const d = stepDigest({
      name: 'g',
      pipeline: {
        configuration: { component_type: 'P' },
        from_pretrained_arguments: { model_name: 'm' },
        arguments: { prompt: 'x'.repeat(100) },
      },
    })
    const args = d.lines.find((l) => l.section === 'arguments')!
    expect(args.text.length).toBeLessThan(70)
    expect(args.text).toContain('…')
  })

  it('summarizes task and sub-workflow steps', () => {
    expect(
      stepDigest({
        name: 'up',
        task: { command: 'upscale', arguments: { scale: 2 } },
      }).summary,
    ).toBe('task: upscale · 1 arg')
    expect(
      stepDigest({
        name: 'aug',
        workflow: { path: 'builtin:augment_prompt.json', arguments: {} },
      }).summary,
    ).toBe('workflow: builtin:augment_prompt.json')
  })

  it('never throws on a bare or malformed step', () => {
    expect(stepDigest({}).summary).toBe('empty step')
    expect(stepDigest({ name: 'x', pipeline: {} }).summary).toBe(
      'pipeline · 0 args',
    )
  })
})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ui && npx vitest run src/lib/digest.test.ts`
Expected: FAIL — cannot resolve `./digest`

- [ ] **Step 3: Implement digest.ts**

```ts
// ui/src/lib/digest.ts
/** Textual digests of a step's JSON. The summary names the step's work
 * on one line for the collapsed bar; the lines are the compact view -
 * one per populated area, each tagged with the full-view section it
 * expands into. Pure: same JSON in, same text out. */

export type DigestSection =
  | 'main'
  | 'arguments'
  | 'components'
  | 'loras'
  | 'scheduler'
  | 'acceleration'

export interface DigestLine {
  text: string
  section: DigestSection
}

export interface StepDigest {
  summary: string
  lines: DigestLine[]
}

const COMPONENT_KEYS = new Set([
  'configuration',
  'from_pretrained_arguments',
  'arguments',
  'loras',
  'scheduler',
  'shared_components',
  'reused_components',
])

function truncate(value: unknown, max = 32): string {
  const s =
    value !== null && typeof value === 'object'
      ? JSON.stringify(value)
      : String(value)
  return s.length > max ? s.slice(0, max - 1) + '…' : s
}

function countText(n: number, noun: string): string {
  return `${n} ${noun}${n === 1 ? '' : 's'}`
}

export function stepDigest(step: Record<string, any>): StepDigest {
  if (step.pipeline) return pipelineDigest(step)
  if (step.task) {
    const argCount = Object.keys(step.task.arguments ?? {}).length
    return {
      summary: `task: ${step.task.command || '?'} · ${countText(argCount, 'arg')}`,
      lines: argsLine(step.task.arguments),
    }
  }
  if (step.workflow) {
    return {
      summary: `workflow: ${step.workflow.path || '?'}`,
      lines: argsLine(step.workflow.arguments),
    }
  }
  if (step.pipeline_reference) {
    return { summary: 'pipeline reference', lines: [] }
  }
  return { summary: 'empty step', lines: [] }
}

function argsLine(args: Record<string, unknown> | undefined): DigestLine[] {
  const entries = Object.entries(args ?? {})
  if (!entries.length) return []
  return [
    {
      section: 'arguments',
      text: entries
        .map(([k, v]) => `${k} = ${truncate(v)}`)
        .join(' · '),
    },
  ]
}

function pipelineDigest(step: Record<string, any>): StepDigest {
  const pipeline = step.pipeline
  const configuration = pipeline.configuration ?? {}
  const pretrained = pipeline.from_pretrained_arguments ?? {}
  const args = pipeline.arguments ?? {}
  const argCount = Object.keys(args).length

  const summary = [
    configuration.component_type || 'pipeline',
    pretrained.model_name,
    countText(argCount, 'arg'),
  ]
    .filter(Boolean)
    .join(' · ')

  const lines: DigestLine[] = []

  const main = [
    pretrained.model_name,
    pretrained.torch_dtype,
    configuration.offload ? `offload: ${configuration.offload}` : null,
    step.result?.content_type ? `save: ${step.result.content_type}` : null,
  ].filter(Boolean)
  if (main.length) lines.push({ section: 'main', text: main.join(' · ') })

  lines.push(...argsLine(args))

  const components = Object.keys(pipeline).filter(
    (key) =>
      !COMPONENT_KEYS.has(key) &&
      pipeline[key] !== null &&
      typeof pipeline[key] === 'object',
  )
  if (components.length) {
    lines.push({
      section: 'components',
      text: components
        .map((slot) => {
          const quant = pipeline[slot]?.quantization_config?.config_type
          return quant ? `${slot} (${quant})` : slot
        })
        .join(' · '),
    })
  }

  const loras: Array<Record<string, any>> = pipeline.loras ?? []
  if (loras.length) {
    lines.push({
      section: 'loras',
      text: `${countText(loras.length, 'LoRA')}: ${loras
        .map((l) => l.adapter_name || l.model_name || '?')
        .join(', ')}`,
    })
  }

  const schedulerType = pipeline.scheduler?.configuration?.scheduler_type
  if (schedulerType) lines.push({ section: 'scheduler', text: schedulerType })

  const acceleration = [
    configuration.cache ? `cache: ${configuration.cache.type}` : null,
    configuration.attention_backend
      ? `attention: ${configuration.attention_backend}`
      : null,
    configuration.prompt_weighting ? 'prompt weighting' : null,
  ].filter(Boolean)
  if (acceleration.length) {
    lines.push({ section: 'acceleration', text: acceleration.join(' · ') })
  }

  return { summary, lines }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ui && npx vitest run src/lib/digest.test.ts`
Expected: PASS (5 tests)

- [ ] **Step 5: Format and commit**

```bash
cd ui && npm run format && cd ..
git add ui/src/lib/digest.ts ui/src/lib/digest.test.ts
git commit -m "Add step digest module for collapsed summaries and compact view"
```

---

### Task 5: Three-state steps in the editor

**Files:**
- Modify: `ui/src/lib/editor/StepEditor.svelte` (replace `open` boolean with a three-state `mode`)
- Modify: `ui/src/lib/pages/EditorPage.svelte` (own and persist per-step modes; expand/collapse-all row)

**Interfaces:**
- Consumes: `stepDigest` from Task 4, `storageGet`/`storageSet` from Task 1.
- Produces: `StepEditor` prop contract changes — `open` is gone; new props `mode: 'collapsed' | 'compact' | 'full'` (owned by the parent, since EditorPage persists modes by step name) and `onmodechange: (mode) => void`. Task 6 builds on this exact prop set.

- [ ] **Step 1: Rework StepEditor's state and bar**

In `ui/src/lib/editor/StepEditor.svelte`:

1. Replace `let open = $state(true)` with parent-owned mode props and helpers (add `mode`/`onmodechange` to the destructured `$props()` and its type):

```ts
let {
  step = $bindable(),
  index,
  count,
  references = [],
  baseFolder = '',
  mode = 'full',
  onmodechange = undefined,
  onremove,
  onmove,
}: {
  step: Record<string, any>
  index: number
  count: number
  references?: string[]
  baseFolder?: string
  mode?: 'collapsed' | 'compact' | 'full'
  onmodechange?: (mode: 'collapsed' | 'compact' | 'full') => void
  onremove: () => void
  onmove: (delta: number) => void
} = $props()

import { stepDigest } from '../digest'
const digest = $derived(stepDigest($state.snapshot(step)))

// Mode is parent-owned (EditorPage persists it per step name) - every
// internal change routes through the callback and flows back down
function setModeInternal(next: 'collapsed' | 'compact' | 'full') {
  onmodechange?.(next)
}

// The chevron re-opens to whichever expanded state the step last had
let lastExpanded = $state<'compact' | 'full'>('compact')
$effect(() => {
  if (mode !== 'collapsed') lastExpanded = mode
})
function toggleCollapsed() {
  setModeInternal(mode === 'collapsed' ? lastExpanded : 'collapsed')
}

// A compact line clicked open jumps straight to its section in full view
let openSection = $state('')
function openFull(section: string) {
  openSection = section
  setModeInternal('full')
}
```

The bar's compact/full buttons likewise call `setModeInternal('compact')` / `setModeInternal('full')` instead of assigning `mode`.

2. In the bar markup: the chevron button calls `toggleCollapsed()` and shows `ChevronRight` when `mode === 'collapsed'`, else `ChevronDown`. After the `kind` span, add the collapsed summary and (when expanded) a compact/full switch:

```svelte
{#if mode === 'collapsed'}
  <span class="muted summary" title={digest.summary}>{digest.summary}</span>
{/if}
<span class="flex"></span>
{#if mode !== 'collapsed'}
  <div class="modeswitch" role="group" aria-label="step detail level">
    <button
      class="quiet"
      class:activebtn={mode === 'compact'}
      onclick={() => (mode = 'compact')}
      title="one-line-per-area digest of what this step sets">compact</button
    >
    <button
      class="quiet"
      class:activebtn={mode === 'full'}
      onclick={() => (mode = 'full')}
      title="every field, editable">full</button
    >
  </div>
{/if}
```

3. Replace `{#if open}` around the body with `{#if mode === 'full'}` and add a compact branch before it:

```svelte
{#if mode === 'compact'}
  <div class="digest">
    {#each digest.lines as line (line.section)}
      <button
        class="digestline"
        onclick={() => openFull(line.section)}
        title="edit in full view"
      >
        <span class="digestsection muted">{line.section}</span>
        <span class="digesttext">{line.text}</span>
      </button>
    {:else}
      <div class="muted hint">nothing set yet - switch to full to edit</div>
    {/each}
  </div>
{:else if mode === 'full'}
  ... (existing body, unchanged)
{/if}
```

4. Wire `openSection` into the four `<details>` elements so a digest click lands open, e.g. components becomes `open={activeSlots.length > 0 || openSection === 'components'}` — same pattern for `loras`, `scheduler`, `acceleration`. (`main` and `arguments` are always visible in full view; no change needed.)

5. Styles (scoped, using tokens):

```css
.summary {
  font-size: 0.8rem;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  min-width: 0;
  flex: 1;
}
.modeswitch {
  display: inline-flex;
}
.modeswitch button {
  border-radius: 0;
  padding: 0.2rem 0.55rem;
  font-size: 0.75rem;
}
.modeswitch button:first-child {
  border-radius: var(--radius-1) 0 0 var(--radius-1);
}
.modeswitch button:last-child {
  border-radius: 0 var(--radius-1) var(--radius-1) 0;
  margin-left: -1px;
}
.digest {
  display: flex;
  flex-direction: column;
  gap: var(--space-1);
  margin-top: var(--space-2);
}
.digestline {
  display: flex;
  gap: var(--space-2);
  align-items: baseline;
  background: transparent;
  border: 0;
  color: var(--ink);
  font-weight: 400;
  text-align: left;
  padding: 0.2rem 0.3rem;
  border-radius: var(--radius-1);
  font-size: 0.85rem;
}
.digestline:hover {
  background: var(--panel-2);
  filter: none;
}
.digestsection {
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  flex: none;
  width: 90px;
}
.digesttext {
  overflow-wrap: anywhere;
}
```

Reuse the existing `.activebtn` pattern from EditorPage by adding a scoped copy in StepEditor's styles:

```css
.activebtn {
  border-color: var(--accent);
  color: var(--accent);
  position: relative;
  z-index: 1;
}
```

- [ ] **Step 2: Own the modes in EditorPage**

In `ui/src/lib/pages/EditorPage.svelte`:

1. Imports: add `storageGet, storageSet` from `../storage`.

2. State and persistence (near the other `let` declarations):

```ts
type StepMode = 'collapsed' | 'compact' | 'full'
// Keyed by step name; existing steps open compact, new ones full
let stepModes = $state<Record<string, StepMode>>({})
const modesKey = $derived(`step-modes:${name || '(unsaved)'}`)

function modeOf(step: Record<string, any>): StepMode {
  return stepModes[step.name] ?? 'compact'
}
function setMode(step: Record<string, any>, mode: StepMode) {
  stepModes[step.name] = mode
  storageSet(modesKey, $state.snapshot(stepModes))
}
function setAllModes(mode: StepMode) {
  for (const step of workflow.steps ?? []) stepModes[step.name] = mode
  storageSet(modesKey, $state.snapshot(stepModes))
}
```

3. In the load `$effect`, after a workflow arrives (both the `api.getWorkflow` `.then` branch and the new/imported branch), restore: `stepModes = storageGet(modesKey, {})` — except a brand-new blank workflow, whose single starter step should open full: `stepModes = { [fresh.steps[0].name]: 'full' }` when there was no import.

4. In `addStep`, after pushing the step: `stepModes[step.name] = 'full'` (a just-added step is being worked on).

5. Bind into the step list (replacing the current `<StepEditor ...>` invocation):

```svelte
<StepEditor
  bind:step={workflow.steps[index]}
  {index}
  count={workflow.steps.length}
  references={stepReferences[index] ?? []}
  baseFolder={folder === '__new__' ? '' : folder}
  mode={modeOf(step)}
  onmodechange={(m) => setMode(step, m)}
  onremove={() => removeStep(index)}
  onmove={(delta) => moveStep(index, delta)}
/>
```

Note: since the parent stores modes by name, use a callback rather than `bind:mode`. In StepEditor, change `mode = $bindable('full')` to a plain prop plus `onmodechange?: (mode: 'collapsed' | 'compact' | 'full') => void`, and route every internal assignment through a local `function setModeInternal(next)` that calls `onmodechange?.(next)` (the parent's state change flows back down). Adjust the Step 1 snippets accordingly — `toggleCollapsed`, the modeswitch buttons, and `openFull` all call `setModeInternal`.

6. Above the `{#each workflow.steps ...}` loop, add the density controls row:

```svelte
{#if (workflow.steps ?? []).length > 1}
  <div class="densityrow">
    <span class="muted">steps</span>
    <span class="flex"></span>
    <button class="quiet" onclick={() => setAllModes('collapsed')}
      >collapse all</button
    >
    <button class="quiet" onclick={() => setAllModes('compact')}
      >compact all</button
    >
    <button class="quiet" onclick={() => setAllModes('full')}
      >expand all</button
    >
  </div>
{/if}
```

```css
.densityrow {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  margin-bottom: var(--space-2);
}
.densityrow button {
  font-size: 0.75rem;
  padding: 0.2rem 0.55rem;
}
```

7. A renamed step orphans its stored mode — acceptable (falls back to compact); no migration code.

- [ ] **Step 3: Verify by hand and by gates**

Run: `cd ui && npm run check && npm run lint && npm test`
Expected: all pass. Then run the e2e smoke to catch regressions in the editor flows:

Run: `cd ui && npx playwright test e2e/smoke.spec.ts`
Expected: `editor opens a workflow with introspected arguments` FAILS if the loaded step defaults to compact (the test expects `#ct-0` visible). Fix the test — it now describes the new behavior: after `page.goto('/#/edit/flux/FluxDev')`, add

```ts
await page.getByRole('button', { name: 'full' }).click()
```

before the `#ct-0` assertion. Re-run until smoke passes.

- [ ] **Step 4: Commit**

```bash
cd ui && npm run format && cd ..
git add ui/src/lib/editor/StepEditor.svelte ui/src/lib/pages/EditorPage.svelte ui/e2e/smoke.spec.ts
git commit -m "Three-state step density: collapsed summary, compact digest, full edit"
```

---

### Task 6: Ordinal rail, flow chips, inline warnings

**Files:**
- Modify: `ui/src/lib/pages/EditorPage.svelte` (rail layout, flow graph wiring, hover state; remove the `refproblems` banner)
- Modify: `ui/src/lib/editor/StepEditor.svelte` (chips, cartesian note, inline problems)
- Modify: `ui/e2e/smoke.spec.ts` (the `.refproblems` assertion moves inline)

**Interfaces:**
- Consumes: `flowGraph`, `danglingReferenceDetails`, `StepFlow` from Task 3; Task 5's StepEditor prop shape.
- Produces: StepEditor props gain `flow?: StepFlow`, `problems?: string[]`, `onhover?: (stepName: string | null) => void`. The `referenceProblems` derived and its banner in EditorPage are deleted.

- [ ] **Step 1: Wire the graph and hover state in EditorPage**

1. Imports: `import { danglingReferenceDetails, flowGraph } from '../flow'` (drop the now-unused `danglingReferences` import from `../editor`).

2. Replace the `referenceProblems` derived (lines 131–133) with:

```ts
const flow = $derived(flowGraph($state.snapshot(workflow)))
const problemsByStep = $derived.by(() => {
  const details = danglingReferenceDetails(
    $state.snapshot(workflow),
    promptLibrary.names,
  )
  const map = new Map<number, string[]>()
  for (const d of details) {
    map.set(d.stepIndex, [...(map.get(d.stepIndex) ?? []), d.message])
  }
  return map
})
let hovered = $state<string | null>(null)
```

3. Delete the `refproblems` banner block (the `{#if referenceProblems.length}` markup) and its `.refproblems` styles.

4. Wrap the step loop in a rail:

```svelte
<div class="steps">
  {#each workflow.steps ?? [] as step, index (step)}
    <div
      class="steprow"
      class:flowlit={hovered !== null && step.name === hovered}
    >
      <div class="railcell">
        <span class="ordinal" title={`step ${index + 1} of ${workflow.steps.length}`}
          >{index + 1}</span
        >
      </div>
      <StepEditor
        bind:step={workflow.steps[index]}
        {index}
        count={workflow.steps.length}
        references={stepReferences[index] ?? []}
        baseFolder={folder === '__new__' ? '' : folder}
        mode={modeOf(step)}
        flow={flow[index]}
        problems={problemsByStep.get(index) ?? []}
        onmodechange={(m) => setMode(step, m)}
        onhover={(n) => (hovered = n)}
        onremove={() => removeStep(index)}
        onmove={(delta) => moveStep(index, delta)}
      />
    </div>
  {/each}
</div>
```

5. Rail styles (StepEditor's own `.step { margin-bottom }` becomes the row gap):

```css
.steps {
  display: flex;
  flex-direction: column;
  gap: var(--space-3);
  margin-bottom: var(--space-4);
}
.steprow {
  display: grid;
  grid-template-columns: 28px minmax(0, 1fr);
  gap: 0 var(--space-2);
}
.railcell {
  position: relative;
  display: flex;
  justify-content: center;
}
/* the connecting line - drawn per row so it spans the gaps too */
.steprow:not(:last-child) .railcell::before {
  content: '';
  position: absolute;
  top: 26px;
  bottom: calc(-1 * var(--space-3));
  width: 2px;
  background: color-mix(in srgb, var(--accent) 35%, transparent);
}
.ordinal {
  width: 22px;
  height: 22px;
  border-radius: 50%;
  background: var(--accent);
  color: var(--accent-ink);
  font-size: 0.75rem;
  font-weight: 700;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  margin-top: var(--space-2);
  z-index: 1;
}
.steprow.flowlit :global(.panel.step) {
  border-color: var(--accent);
}
```

In `StepEditor.svelte`, remove the `.step { margin-bottom: 0.7rem }` rule (the parent gap owns spacing now).

- [ ] **Step 2: Chips, cartesian note and inline problems in StepEditor**

1. Extend props (after Task 5's shape):

```ts
import type { StepFlow } from '../flow'
// in the $props() destructuring and type:
flow = undefined as StepFlow | undefined,
problems = [] as string[],
onhover = undefined as ((stepName: string | null) => void) | undefined,
```

(Written in the established `name = default` prop style with the matching type entries `flow?: StepFlow`, `problems?: string[]`, `onhover?: (stepName: string | null) => void`.)

2. In the bar, after the `kind` span (before the collapsed summary), render the flow chips — visible in every mode, since they are the step's identity in the sequence:

```svelte
{#each flow?.inputs ?? [] as producer (producer)}
  <span
    class="flowchip in"
    role="note"
    onmouseenter={() => onhover?.(producer)}
    onmouseleave={() => onhover?.(null)}
    title={`consumes previous_result:${producer}`}>← {producer}</span
  >
{/each}
{#each flow?.consumers ?? [] as consumer (consumer)}
  <span
    class="flowchip out"
    role="note"
    onmouseenter={() => onhover?.(consumer)}
    onmouseleave={() => onhover?.(null)}
    title={`step '${consumer}' consumes this step's result`}
    >→ {consumer}</span
  >
{/each}
```

3. Directly under the bar (inside the panel, before the `{#if mode === ...}` branches), the warnings that used to live in the page banner, plus the cartesian note:

```svelte
{#each problems as problem (problem)}
  <div class="stepwarn"><TriangleAlert size={13} /> {problem}</div>
{/each}
{#if (flow?.resolvedRefs ?? 0) > 1}
  <div class="muted hint cartesian">
    {flow!.resolvedRefs} previous_result inputs - iterations multiply
    (every combination runs)
  </div>
{/if}
```

Add `TriangleAlert` to the lucide import.

4. Styles:

```css
.flowchip {
  font-family: ui-monospace, 'Cascadia Code', monospace;
  font-size: 0.72rem;
  padding: 0.05rem 0.5rem;
  border-radius: 999px;
  border: 1px solid;
  white-space: nowrap;
  cursor: default;
}
.flowchip.in {
  color: var(--accent);
  border-color: color-mix(in srgb, var(--accent) 45%, transparent);
  background: color-mix(in srgb, var(--accent) 12%, transparent);
}
.flowchip.out {
  color: var(--good);
  border-color: color-mix(in srgb, var(--good) 45%, transparent);
  background: color-mix(in srgb, var(--good) 12%, transparent);
}
.stepwarn {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  color: var(--warn);
  font-size: 0.85rem;
  margin-top: var(--space-2);
}
.cartesian {
  margin-top: var(--space-1);
}
```

- [ ] **Step 3: Update the dangling-reference e2e test**

In `ui/e2e/smoke.spec.ts`, the test `editor flags a dangling reference without asking the server` asserts the old banner. Replace its two `.refproblems` assertions:

```ts
await expect(page.locator('.stepwarn')).toHaveCount(0)
// removing the variable the step's prompt argument points at
await page
  .locator('#wfvar-prompt')
  .locator('xpath=following-sibling::button[1]')
  .click()
await expect(page.locator('.stepwarn')).toContainText(
  'variable:prompt - no such variable is declared',
)
```

- [ ] **Step 4: Run the gates**

Run: `cd ui && npm run check && npm run lint && npm test && npx playwright test e2e/smoke.spec.ts`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
cd ui && npm run format && cd ..
git add ui/src/lib/pages/EditorPage.svelte ui/src/lib/editor/StepEditor.svelte ui/e2e/smoke.spec.ts
git commit -m "Ordinal rail, producer/consumer chips, and inline reference warnings"
```

---

### Task 7: Knob discovery on demand

**Files:**
- Modify: `ui/src/lib/editor/ArgumentsEditor.svelte` (replace the add-argument select with a discovery disclosure)
- Modify: `ui/e2e/smoke.spec.ts` (the `add argument…` select assertion)

**Interfaces:**
- Consumes: existing `description`/`unused` deriveds in ArgumentsEditor.
- Produces: no API change — same `args` binding contract. The "add argument…" `<select>` is gone; its replacement is a `button.discover` toggling a `.available` list.

- [ ] **Step 1: Replace the unused-parameter select**

In `ui/src/lib/editor/ArgumentsEditor.svelte`, delete the `{#if unused.length}` block (the select + Plus button, lines 157–177) and the now-unused `adding` state and `add()` function. In their place:

```svelte
{#if unused.length}
  <button
    class="quiet discover"
    onclick={() => (showAll = !showAll)}
    title="every argument this callable accepts, discovered from its signature"
  >
    {showAll ? 'hide' : 'show'} available arguments ({unused.length})
  </button>
  {#if showAll}
    <div class="available">
      {#each unused as parameter (parameter.name)}
        <div class="availrow">
          <button
            class="quiet icon"
            onclick={() => {
              args[parameter.name] = parameter.default ?? ''
            }}
            title={`add ${parameter.name}`}
            aria-label={`add ${parameter.name}`}
          >
            <Plus size={14} />
          </button>
          <span class="availname"
            >{parameter.name}{parameter.required ? ' *' : ''}</span
          >
          {#if parameter.description}
            <span class="muted availdesc">{parameter.description}</span>
          {/if}
        </div>
      {/each}
    </div>
  {/if}
{/if}
```

With state `let showAll = $state(false)` and styles:

```css
.discover {
  align-self: start;
  font-size: 0.8rem;
}
.available {
  display: flex;
  flex-direction: column;
  gap: var(--space-1);
  border-left: 2px solid var(--line);
  padding-left: var(--space-2);
}
.availrow {
  display: flex;
  align-items: baseline;
  gap: var(--space-2);
}
.availrow .icon {
  align-self: center;
  flex: none;
}
.availname {
  font-weight: 600;
  flex: none;
}
.availdesc {
  font-size: 0.75rem;
  overflow-wrap: anywhere;
}
```

Adding a parameter removes it from `unused` reactively (it's now `in args`), so the row disappears and the populated field appears above — the list stays honest with no extra bookkeeping.

- [ ] **Step 2: Update the introspection e2e assertion**

In `ui/e2e/smoke.spec.ts`, test `editor opens a workflow with introspected arguments`: replace the `add argument…` select assertion with the disclosure, and exercise it:

```ts
// the arguments editor discovered real __call__ parameters - the
// discovery disclosure renders once the description arrives
const discover = page
  .getByRole('button', { name: /show available arguments/ })
  .first()
await expect(discover).toBeVisible({ timeout: 90_000 })
await discover.click()
await expect(page.locator('.availdesc').first()).toBeVisible()
```

- [ ] **Step 3: Run the gates**

Run: `cd ui && npm run check && npm run lint && npm test && npx playwright test e2e/smoke.spec.ts`
Expected: all pass

- [ ] **Step 4: Commit**

```bash
cd ui && npm run format && cd ..
git add ui/src/lib/editor/ArgumentsEditor.svelte ui/e2e/smoke.spec.ts
git commit -m "Arguments editor: on-demand knob discovery replaces the add-argument select"
```

---

### Task 8: End-to-end coverage of the step model

**Files:**
- Create: `ui/e2e/step-model.spec.ts`

**Interfaces:**
- Consumes: everything shipped in Tasks 5–7, plus the existing Playwright fixture (`e2e/serve_fixture.py` — the config starts it; new spec files need no setup).

- [ ] **Step 1: Write the spec**

```ts
// ui/e2e/step-model.spec.ts
import { expect, test } from '@playwright/test'

// A two-step task workflow typed through the JSON view - task steps need
// no server-side pipeline imports, so these tests stay fast
const TWO_STEP = JSON.stringify({
  id: 'flow_check',
  variables: {},
  steps: [
    { name: 'first', task: { command: 'noop', arguments: { x: 1 } } },
    {
      name: 'second',
      task: { command: 'noop', arguments: { y: 'previous_result:first' } },
    },
  ],
})

async function loadTwoStep(page: import('@playwright/test').Page) {
  await page.goto('/#/edit')
  await page.getByRole('button', { name: /JSON/ }).click()
  await expect(page.locator('.monaco-editor').first()).toBeVisible({
    timeout: 20_000,
  })
  await page.locator('.view-lines').first().click()
  await page.keyboard.press('ControlOrMeta+a')
  await page.keyboard.press('Backspace')
  await page.keyboard.insertText(TWO_STEP)
  await page.getByRole('button', { name: /form/ }).click()
}

test('steps carry ordinals and producer/consumer chips', async ({ page }) => {
  await loadTwoStep(page)
  // the rail numbers the sequence
  await expect(page.locator('.ordinal').nth(0)).toHaveText('1')
  await expect(page.locator('.ordinal').nth(1)).toHaveText('2')
  // fan edges render on both ends
  await expect(page.locator('.flowchip.out', { hasText: 'second' })).toBeVisible()
  await expect(page.locator('.flowchip.in', { hasText: 'first' })).toBeVisible()
  // hovering the consumer's input chip lights the producer's row
  await page.locator('.flowchip.in', { hasText: 'first' }).hover()
  await expect(page.locator('.steprow.flowlit')).toHaveCount(1)
})

test('step density cycles collapsed / compact / full and persists intent', async ({
  page,
}) => {
  await loadTwoStep(page)
  const firstStep = page.locator('.panel.step').first()
  // compact digest shows what is set, as text
  await firstStep.getByRole('button', { name: 'compact' }).click()
  await expect(firstStep.locator('.digestline')).toContainText('x = 1')
  // clicking a digest line jumps to full view
  await firstStep.locator('.digestline').first().click()
  await expect(firstStep.getByLabel('x')).toBeVisible()
  // collapsing leaves a one-line summary, not nothing
  await firstStep.getByRole('button', { name: /collapse this step/ }).click()
  await expect(firstStep.locator('.summary')).toContainText('task: noop')
})

test('collapse all / expand all sweep every step', async ({ page }) => {
  await loadTwoStep(page)
  await page.getByRole('button', { name: 'collapse all' }).click()
  await expect(page.locator('.summary')).toHaveCount(2)
  await page.getByRole('button', { name: 'expand all' }).click()
  await expect(page.locator('.summary')).toHaveCount(0)
})

test('reordering that breaks a reference warns on the step itself', async ({
  page,
}) => {
  await loadTwoStep(page)
  await expect(page.locator('.stepwarn')).toHaveCount(0)
  // move the consumer above its producer
  await page
    .locator('.panel.step')
    .nth(1)
    .getByRole('button', { name: 'move up' })
    .click()
  await expect(page.locator('.stepwarn')).toContainText(
    'no earlier step has that name',
  )
})
```

Note: `loadTwoStep` assumes Task 5's default of compact for loaded steps does not hide the bar controls (it doesn't — the bar always renders). If the `x` label assertion is ambiguous, scope it to `firstStep` as written.

- [ ] **Step 2: Run the new spec**

Run: `cd ui && npx playwright test e2e/step-model.spec.ts`
Expected: PASS (4 tests). Debug selectors against the implementation if any drift — the implementation is authoritative; adjust the spec's selectors, not its assertions' intent.

- [ ] **Step 3: Run every gate**

Run: `cd ui && npm run check && npm run lint && npm test && npx playwright test`
Expected: all pass (unit, smoke, responsive, step-model)

- [ ] **Step 4: Commit**

```bash
cd ui && npm run format && cd ..
git add ui/e2e/step-model.spec.ts
git commit -m "E2E coverage for the editor step model: rail, chips, density, inline warnings"
```

---

## After this plan

Stages 3–5 of the spec (feedback layer with toasts, consistency sweep with shared components, chrome & keyboard) get their own plan once this one lands, written against the post-stage-2 code. New "even better if" ideas found during execution go to the deferred list in `docs/superpowers/scope/UI-UX.md`.
