# UI Phase 0: Baseline and Correctness Fixes - Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Commit a measured UI baseline that CI enforces, make the gate reports count the UI's `.svelte` code, and fix the five places where the UI and the engine disagree today (U1-U5), so later phases restructure from a correct, measured starting point.

**Architecture:** One new developer script, `ui/scripts/arch-metrics.mjs`, measures `ui/src` the way `scripts/arch_metrics.py` measures the Python packages; its baseline sits in `docs/stabilization/ui/baseline.json`. One new UI module, `ui/src/lib/references.ts`, becomes the UI's only speller of the engine's reference prefixes; it is the owner the ratchet's `prefix_literals` names. A copy the UI must keep is pinned to its Python owner by `tests/test_ui_twins.py`, which reads the TypeScript source; a rule tested in both languages reads one shared case file under `tests/fixtures/`. One server change: `GET /api/jobs/{id}` gains `output_kinds`, so the job page renders by the server's media kinds instead of its own extension list.

**Tech Stack:** Svelte 5 (runes), TypeScript 6, Vite 8, Vitest 4 (jsdom) with @testing-library/svelte, ESLint 10 (Node API), Python 3.12, pytest, FastAPI.

**Spec:** [ASSESSMENT.md](ASSESSMENT.md) and [ROADMAP.md](ROADMAP.md) in this directory.

## Global Constraints

- Freeze (ROADMAP *Decisions*, Don 2026-10-01): no new UI features. Change only what a task names.
- The only new modules are `ui/scripts/arch-metrics.mjs`, `ui/src/lib/references.ts`, test files and `tests/fixtures/*.json`.
- Every new test fails on the code before its fix. Run it and see the failure before writing the fix.
- No count-pinning assertions. Baseline numbers live in `docs/stabilization/ui/baseline.json`, never in a test.
- Comments explain why the code is the way it is. Do not narrate ticket history in new comments.
- A task may lower a ratchet and must not raise one. Run `node scripts/arch-metrics.mjs --check ../docs/stabilization/ui/baseline.json` from `ui/` before every commit from Task 1 on.
- UI commands run from `ui/`: `npx vitest run <file>`, `npm run lint`, `npm run check`, `npx prettier --write <files>`. Python from the repo root: `venv/bin/python -m pytest <path> -q`, `venv/bin/ruff format <files>`, `venv/bin/ruff check <files>`.
- Commits: conventional prefix (`fix(ui)`, `test(ui)`, `chore(ui)`), and end each message with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Work on branch `ui-stabilization/phase-0` from develop; never commit to `master`.

## Review Focus

1. A `for_each` member reference with a property (`previous_result:g@a.mask`) is not dangling, and a member reference to a step *without* `for_each` (`previous_result:g@a` where `g` has none) still is (Task 4).
2. A step name containing a dot (`x.y`) referenced whole is not dangling; the old check split on the first dot (Task 4).
3. A workflow already holding a content type outside the list (`audio/x-flac`) shows that value selected, not a blank select, and saving without touching it keeps it (Task 7).
4. A job detail from history, whose `manifest` is `null`, still renders and gets `output_kinds: {}` rather than a 500 (Task 6).
5. A workspace name using a non-ASCII letter (`café`) is accepted by the UI exactly when the engine accepts it (Task 5).

---

### Task 1: UI ratchet and baseline

The script was drafted while measuring for the ROADMAP and is committed with
this plan at `ui/scripts/arch-metrics.mjs`; its header comment holds the
counting rules. This task tests it, records the baseline and wires the check
into CI and preflight.

**Files:**
- Test: `ui/scripts/arch-metrics.test.ts`
- Modify: `ui/eslint.config.js` (Node globals for `scripts/`)
- Modify: `ui/package.json` (`metrics` script; `format` and `preflight` cover it)
- Modify: `.github/workflows/ci.yml` (`ui` job: prettier over `scripts`, a ratchet step)
- Create: `docs/stabilization/ui/baseline.json` (generated)

**Interfaces:**
- Consumes: `ui/scripts/arch-metrics.mjs` exports `sourceFiles(root?) -> string[]`, `referencePrefixes(text?) -> string[]`, `cycles(graph: Record<string, string[]>) -> string[][]`, `measure(files?, prefixes?) -> Promise<{ metrics, detail }>`, `regressions(current, baseline) -> string[]`, and the constants `SIZE_CEILING` (600), `COMPLEXITY_LIMIT` (10), `FUNCTION_LINE_LIMIT` (60), `PREFIX_OWNERS` (`src/lib/references.ts`).
- Produces: `npm run metrics -- --check ../docs/stabilization/ui/baseline.json`, which exits 1 and prints one line per regressed metric. Every later task runs it; the harness (Phase 4) will too.

- [ ] **Step 1: Write the tests**

```ts
// @vitest-environment node
import { describe, expect, it } from 'vitest'
import {
  cycles,
  measure,
  referencePrefixes,
  regressions,
  sourceFiles,
} from './arch-metrics.mjs'

describe('cycles', () => {
  it('finds a two-module cycle and ignores a chain', () => {
    expect(cycles({ a: ['b'], b: ['a'], c: ['a'] })).toEqual([['a', 'b']])
    expect(cycles({ a: ['b'], b: ['c'], c: [] })).toEqual([])
  })

  it('reports a three-module knot as one component', () => {
    expect(cycles({ a: ['b'], b: ['c'], c: ['a'] })).toEqual([['a', 'b', 'c']])
  })
})

describe('referencePrefixes', () => {
  it('reads the prefix constants and nothing else', () => {
    const text = 'ASSET = "asset:"\nFROM_FILE_KEY = "from_file"\nITEM = "item:"\n'
    expect(referencePrefixes(text)).toEqual(['asset:', 'item:'])
  })

  it('refuses a file with no prefixes rather than measuring 0', () => {
    expect(() => referencePrefixes('nothing here')).toThrow()
  })
})

describe('regressions', () => {
  it('names a metric that rose and ignores one that fell or is new', () => {
    expect(
      regressions({ a: 3, b: 1, c: 9 }, { a: 2, b: 2 }),
    ).toEqual(['a: 3 > baseline 2'])
  })
})

describe('the real tree', () => {
  it('excludes tests and measures every metric', async () => {
    const files = sourceFiles()
    expect(files.some((f) => f.endsWith('.test.ts'))).toBe(false)
    expect(files.some((f) => f.endsWith('App.svelte'))).toBe(true)
    const { metrics } = await measure()
    expect(Object.keys(metrics).sort()).toEqual([
      'a11y_suppressions',
      'complex_functions',
      'files_over_size_ceiling',
      'import_cycles',
      'long_functions',
      'modules_in_import_cycles',
      'prefix_literals',
    ])
    for (const value of Object.values(metrics))
      expect(Number.isInteger(value)).toBe(true)
  }, 60_000)
})
```

- [ ] **Step 2: Run them**

Run: `cd ui && npx vitest run scripts/arch-metrics.test.ts`
Expected: PASS (the script exists). If a `cycles` case fails, fix the script, not the test.

- [ ] **Step 3: Give `scripts/` Node globals, so `npm run lint` passes**

In `ui/eslint.config.js`, add after the `globals.browser` block:

```js
  {
    files: ['scripts/**'],
    languageOptions: { globals: { ...globals.node } },
  },
```

Run: `cd ui && npm run lint`
Expected: no errors.

- [ ] **Step 4: Add the npm script and cover `scripts/` in format and preflight**

In `ui/package.json` `scripts`:

```json
    "format": "prettier --write src e2e scripts *.ts *.js",
    "metrics": "node scripts/arch-metrics.mjs",
    "preflight": "npm run check && npm run lint && npm run format && npm run metrics -- --check ../docs/stabilization/ui/baseline.json && npm run build && npm run test && npm run e2e"
```

Run: `cd ui && npx prettier --write scripts && npm run lint`

- [ ] **Step 5: Record the baseline**

Run: `cd ui && npm run metrics -- --write ../docs/stabilization/ui/baseline.json && cat ../docs/stabilization/ui/baseline.json`
Expected: the seven metrics, matching the ROADMAP's *Today* column (7, 14, 2, 1, 2, 29, 7) unless develop has moved. If it has, the ROADMAP table takes the measured numbers in this commit.

- [ ] **Step 6: Check it in CI**

In `.github/workflows/ci.yml`'s `ui` job, change the format step to `npx prettier --check src e2e scripts *.ts *.js`, and add after `Lint`:

```yaml
      - name: Architecture ratchet
        run: npm run metrics -- --check ../docs/stabilization/ui/baseline.json
```

- [ ] **Step 7: Prove the check bites, then restore**

Run: `cd ui && node -e "const f='../docs/stabilization/ui/baseline.json';const b=JSON.parse(require('fs').readFileSync(f));b.complex_functions--;require('fs').writeFileSync(f,JSON.stringify(b,null,2)+'\n')" && npm run metrics -- --check ../docs/stabilization/ui/baseline.json; echo "exit $?"; git checkout ../docs/stabilization/ui/baseline.json`
Expected: `complex_functions: 14 > baseline 13` and `exit 1`.

- [ ] **Step 8: Commit**

```bash
git add ui/scripts ui/eslint.config.js ui/package.json .github/workflows/ci.yml docs/stabilization/ui/baseline.json
git commit -m "chore(ui): architecture ratchet over ui/src, baseline and CI check"
```

---

### Task 2: The gate report counts `.svelte`

pygount has no lexer for `.svelte` and reports each file as 0 lines, so every
engine gate report's UI row counted only `.ts`.

**Files:**
- Modify: `scripts/arch_report.py` (`sloc`)
- Test: `tests/test_arch_report.py`
- Modify: `docs/stabilization/ROADMAP.md` (*After gate 4* note)

**Interfaces:**
- Produces: `svelte_code_lines(path: pathlib.Path) -> int` in `scripts/arch_report.py`: lines that are neither blank nor a whole comment (`<!-- -->`, `//`, `/* */` on one line).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_arch_report.py`:

```python
def test_a_svelte_file_counts_its_code_lines(tmp_path):
    component = tmp_path / "ui" / "src" / "Card.svelte"
    component.parent.mkdir(parents=True)
    component.write_text(
        '<script lang="ts">\n'
        "  // a comment\n"
        "  let { title } = $props()\n"
        "</script>\n"
        "\n"
        "<!-- markup -->\n"
        "<h2>{title}</h2>\n"
        "<style>\n"
        "  /* a style comment */\n"
        "  h2 { margin: 0; }\n"
        "</style>\n"
    )
    counts = _load().sloc(tmp_path)
    assert counts["UI (ui/src, tests excluded)"] == 7
```

The seven: `<script lang="ts">`, `let ...`, `</script>`, `<h2>...`, `<style>`, `h2 {...}`, `</style>`.

- [ ] **Step 2: Run it to see it fail**

Run: `venv/bin/python -m pytest tests/test_arch_report.py -q -k svelte`
Expected: FAIL, `0 == 7`.

- [ ] **Step 3: Count `.svelte` by hand**

In `scripts/arch_report.py`, add above `sloc`:

```python
# A line that is only a comment. A multi-line /* */ comment's inner lines
# count as code: matching them by a leading '*' would also match a CSS '*'
# selector, and an undercount hides growth where an overcount does not.
SVELTE_COMMENT = re.compile(r"^(<!--.*-->|//.*|/\*.*\*/)$")


def svelte_code_lines(path):
    """Code lines in a .svelte file, which pygount reads as 0: every line
    that is neither blank nor a comment on its own. Script, markup and style
    all count, since all three are what a reader holds."""
    return sum(
        1
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not SVELTE_COMMENT.match(line.strip())
    )
```

and in `sloc`, replace the `counts[layer] += ...` line with:

```python
if path.suffix == ".svelte":
    counts[layer] += svelte_code_lines(path)
else:
    counts[layer] += SourceAnalysis.from_file(str(path), layer).code_count
```

Add `import re` if the module does not already import it.

- [ ] **Step 4: Run the test and the file**

Run: `venv/bin/python -m pytest tests/test_arch_report.py -q`
Expected: PASS.

- [ ] **Step 5: Record the corrected number**

Run: `venv/bin/python scripts/arch_report.py 2>/dev/null | grep -A4 "SLOC by layer"` (or the script's documented invocation for the current tree). In `docs/stabilization/ROADMAP.md`'s *After gate 4* section, add one sentence giving the corrected UI code-line count from this run, measured the same way the API row is.

- [ ] **Step 6: Commit**

```bash
git add scripts/arch_report.py tests/test_arch_report.py docs/stabilization/ROADMAP.md
git commit -m "fix(stabilization): the gate report counts .svelte code lines"
```

---

### Task 3: One speller of reference prefixes; `isReference` knows them all (U2)

`isReference` knows 4 of the engine's prefixes. A value like `item:flag` on
a parameter annotated `bool` gets a checkbox, and toggling it replaces the
reference with `true`/`false`.

**Files:**
- Create: `ui/src/lib/references.ts`
- Modify: `ui/src/lib/editor.ts` (`isReference` moves out; re-exported)
- Modify: `ui/src/lib/runstate.ts` (imports `MEMBER_SEPARATOR`)
- Test: `ui/src/lib/editor.test.ts`
- Create: `tests/test_ui_twins.py`

**Interfaces:**
- Produces: from `ui/src/lib/references.ts`: `ASSET`, `OUTPUT`, `PROMPT`, `VARIABLE`, `PREVIOUS_RESULT`, `CONSTANT`, `ITEM`, `GATHER`, `BUILTIN`, `CONSTRAINT` (each `'<name>:'`), `PREFIXES` (all ten), `MEMBER_SEPARATOR` (`'@'`), `FROM_PREVIOUS_RESULT_KEY` (`'from_previous_result'`), `isReference(value: unknown): value is string`. `editor.ts` keeps exporting `isReference` (re-export) so existing imports hold.
- Produces: `tests/test_ui_twins.py` helpers `ts_constants(path) -> dict[str, str]` and `ts_string_array(path, name) -> list[str]`, used by Tasks 5, 7 and 8.

- [ ] **Step 1: Write the failing UI tests**

Append to `ui/src/lib/editor.test.ts`:

```ts
describe('a reference is always edited as text', () => {
  const boolParam = param({ annotation: 'bool' })
  const intParam = param({ annotation: 'int' })
  it.each([
    'item:flag',
    'gather:shots',
    'asset:cast/priya.png',
    'output:run/v2/final.mp4',
  ])('%s', (value) => {
    expect(isReference(value)).toBe(true)
    expect(widgetFor(boolParam, value)).toBe('text')
    expect(widgetFor(intParam, value)).toBe('text')
  })
})
```

- [ ] **Step 2: Write the failing twin test**

Create `tests/test_ui_twins.py`:

```python
"""The UI's copies of rules the engine owns, pinned to their owners.

The UI cannot import Python, so a rule it must know is a copy. These tests
read the TypeScript source and compare each copy with its owner: a change
to one side fails until the other follows. A copy that can be read from the
server instead is deleted, not pinned.
"""

import pathlib
import re

from dw import references

REPO = pathlib.Path(__file__).resolve().parent.parent
UI_LIB = REPO / "ui" / "src" / "lib"


def ts_constants(path):
    """`export const NAME = 'value'` string constants in a TS file."""
    return dict(
        re.findall(r"^export const ([A-Z_]+) = '([^']*)'", path.read_text(), re.M)
    )


def ts_string_array(path, name):
    """The quoted strings of `export const NAME = [ ... ]` in a TS file.
    Exported only: a module-private copy is not the one other modules use."""
    found = re.search(
        rf"^export const {name}\b[^=]*= \[(.*?)\]", path.read_text(), re.M | re.S
    )
    assert found, f"no array {name} in {path}"
    return re.findall(r"'([^']*)'", found.group(1))


def test_the_ui_spells_every_reference_prefix_the_engine_does():
    engine = {
        name: value
        for name, value in vars(references).items()
        if name.isupper()
        and isinstance(value, str)
        and re.fullmatch(r"[a-z_]+:", value)
    }
    ui = {
        name: value
        for name, value in ts_constants(UI_LIB / "references.ts").items()
        if value.endswith(":")
    }
    assert ui == engine


def test_the_member_separator_and_reference_key_are_the_engines():
    ui = ts_constants(UI_LIB / "references.ts")
    assert ui["MEMBER_SEPARATOR"] == references.MEMBER_SEPARATOR
    assert ui["FROM_PREVIOUS_RESULT_KEY"] == references.FROM_PREVIOUS_RESULT_KEY
```

- [ ] **Step 3: Run both to see them fail**

Run: `cd ui && npx vitest run src/lib/editor.test.ts` then `venv/bin/python -m pytest tests/test_ui_twins.py -q`
Expected: the four `item:`/`gather:`/`asset:`/`output:` cases FAIL; the twin tests FAIL (no `references.ts`).

- [ ] **Step 4: Create the owner**

`ui/src/lib/references.ts`:

```ts
/** The engine's reference prefixes and the names that go with them.
 * dw/references.py owns them; tests/test_ui_twins.py pins this copy. The
 * one module in ui/src that spells a prefix: the UI ratchet's
 * prefix_literals counts any other. */

export const ASSET = 'asset:'
export const OUTPUT = 'output:'
export const PROMPT = 'prompt:'
export const VARIABLE = 'variable:'
export const PREVIOUS_RESULT = 'previous_result:'
export const CONSTANT = 'constant:'
export const ITEM = 'item:'
export const GATHER = 'gather:'
export const BUILTIN = 'builtin:'
export const CONSTRAINT = 'constraint:'

export const PREFIXES = [
  ASSET,
  OUTPUT,
  PROMPT,
  VARIABLE,
  PREVIOUS_RESULT,
  CONSTANT,
  ITEM,
  GATHER,
  BUILTIN,
  CONSTRAINT,
] as const

/** Joins a for_each step's name to an entry's: `<step>@<entry>`. */
export const MEMBER_SEPARATOR = '@'

/** The key a reference object names an earlier step under, unprefixed. */
export const FROM_PREVIOUS_RESULT_KEY = 'from_previous_result'

/** A value the engine resolves later, so it is always edited as text:
 * a widget that coerces it (a checkbox, a number) would replace it. */
export function isReference(value: unknown): value is string {
  return (
    typeof value === 'string' && PREFIXES.some((p) => value.startsWith(p))
  )
}
```

- [ ] **Step 5: Point the old homes at it**

In `ui/src/lib/editor.ts`, delete the `isReference` function and its doc comment, and add near the top imports:

```ts
import { isReference } from './references'
export { isReference }
```

In `ui/src/lib/runstate.ts`, replace `const MEMBER_SEPARATOR = '@'` with `import { MEMBER_SEPARATOR } from './references'` (keep the doc comment above it, which explains the separator's use there).

- [ ] **Step 6: Run the tests, the ratchet and the checks**

Run: `cd ui && npx vitest run && npm run check && npm run lint && npm run metrics -- --check ../docs/stabilization/ui/baseline.json` then `venv/bin/python -m pytest tests/test_ui_twins.py -q`
Expected: all PASS; `prefix_literals` falls by 4.

- [ ] **Step 7: Lower the baseline and commit**

Run: `cd ui && npm run metrics -- --write ../docs/stabilization/ui/baseline.json`

```bash
git add ui/src/lib/references.ts ui/src/lib/editor.ts ui/src/lib/runstate.ts ui/src/lib/editor.test.ts tests/test_ui_twins.py docs/stabilization/ui/baseline.json
git commit -m "fix(ui): every reference prefix is edited as text; references.ts owns them"
```

---

### Task 4: The live reference check agrees with the engine (U1)

`danglingReferenceDetails` (`ui/src/lib/flow.ts`) is a second copy of
`dw/previous_results.py`'s `previous_result_reference_errors`. It flags a
`for_each` member reference the engine accepts, splits a name on its first
`.` where the engine matches the whole name or the name plus a property, and
never looks at a `from_previous_result` key the engine checks. The engine
checks the expanded definition, where a `for_each` step `g` becomes members
`g@<entry>`; the editor holds `g` unexpanded, so it accepts any
`g@...` reference to a `for_each` step. Whether this check should become the
server's validate instead is Phase 1's call; here it only stops disagreeing.

**Files:**
- Modify: `ui/src/lib/flow.ts` (`danglingReferenceDetails`, a new `resolvesTo`)
- Test: `ui/src/lib/flow.test.ts`

**Interfaces:**
- Consumes: `PREVIOUS_RESULT`, `VARIABLE`, `GATHER`, `PROMPT`, `MEMBER_SEPARATOR`, `FROM_PREVIOUS_RESULT_KEY` from `./references` (Task 3).
- Produces: `danglingReferenceDetails(workflow, promptNames?)` keeps its signature and `DanglingReference` shape.

- [ ] **Step 1: Write the failing tests**

Append inside `describe('danglingReferenceDetails', ...)` in `ui/src/lib/flow.test.ts`:

```ts
  const forEachStep = (name: string) => ({
    ...step(name, { x: 'item:prompt' }),
    for_each: [{ name: 'a', prompt: 'p' }],
  })

  it('accepts a member reference to a for_each step, with or without a property', () => {
    const wf = {
      variables: {},
      steps: [
        forEachStep('g'),
        step('b', { one: 'previous_result:g@a', two: 'previous_result:g@a.mask' }),
      ],
    }
    expect(danglingReferenceDetails(wf)).toEqual([])
  })

  it('flags a member reference to a step that is not for_each', () => {
    const wf = {
      variables: {},
      steps: [step('g', {}), step('b', { y: 'previous_result:g@a' })],
    }
    expect(danglingReferenceDetails(wf)).toHaveLength(1)
  })

  it('matches a dotted step name whole, and a property of a plain one', () => {
    const wf = {
      variables: {},
      steps: [
        step('x.y', {}),
        step('seg', {}),
        step('b', { v: 'previous_result:x.y', m: 'previous_result:seg.mask' }),
      ],
    }
    expect(danglingReferenceDetails(wf)).toEqual([])
  })

  it('flags a from_previous_result that names no earlier step', () => {
    const wf = {
      variables: {},
      steps: [step('b', { image: { from_previous_result: 'nope' } })],
    }
    const details = danglingReferenceDetails(wf)
    expect(details).toHaveLength(1)
    expect(details[0].message).toContain('nope')
  })
```

- [ ] **Step 2: Run them to see three fail**

Run: `cd ui && npx vitest run src/lib/flow.test.ts`
Expected: the member, dotted-name and `from_previous_result` cases FAIL; "flags a member reference to a step that is not for_each" passes already.

- [ ] **Step 3: Rewrite the check**

In `ui/src/lib/flow.ts`, import from `./references` and replace `danglingReferenceDetails` with:

```ts
import {
  FROM_PREVIOUS_RESULT_KEY,
  GATHER,
  MEMBER_SEPARATOR,
  PREVIOUS_RESULT,
  PROMPT,
  VARIABLE,
} from './references'

interface EarlierStep {
  name: string
  forEach: boolean
}

/** Whether a previous_result reference resolves to an earlier step: the
 * name itself or the name plus a property (`seg.mask`), as the engine's
 * reference_resolves_to decides - or, for a for_each step, one of the
 * `<step>@<entry>` members the engine expands it into, which the editor
 * cannot list because it holds the step unexpanded. */
function resolvesTo(reference: string, step: EarlierStep): boolean {
  if (reference === step.name || reference.startsWith(step.name + '.'))
    return true
  return step.forEach && reference.startsWith(step.name + MEMBER_SEPARATOR)
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
    const earlier: EarlierStep[] = steps
      .slice(0, index)
      .filter((s) => typeof s.name === 'string' && s.name)
      .map((s) => ({ name: s.name, forEach: s.for_each !== undefined }))
    const report = (message: string) =>
      problems.push({ stepIndex: index, message: `Step '${step.name}': ${message}` })

    scanStringsWithPath(step, [], (value, path) => {
      if (path[path.length - 1] === FROM_PREVIOUS_RESULT_KEY) {
        if (!earlier.some((s) => resolvesTo(value, s)))
          report(`${FROM_PREVIOUS_RESULT_KEY} '${value}' - no earlier step has that name`)
      } else if (value.startsWith(VARIABLE)) {
        const name = value.slice(VARIABLE.length)
        if (!variables.has(name))
          report(`${VARIABLE}${name} - no such variable is declared`)
      } else if (value.startsWith(PREVIOUS_RESULT)) {
        const reference = value.slice(PREVIOUS_RESULT.length)
        if (!earlier.some((s) => resolvesTo(reference, s)))
          report(`${PREVIOUS_RESULT}${reference} - no earlier step has that name`)
      } else if (value.startsWith(GATHER)) {
        const name = value.slice(GATHER.length)
        if (!earlier.some((s) => s.name === name))
          report(`${GATHER}${name} - no earlier step has that name`)
      } else if (value.startsWith(PROMPT)) {
        // Without a listing the server resolves these at run time - only
        // a supplied library can say a name is missing
        const name = value.slice(PROMPT.length)
        if (prompts !== null && !prompts.has(name))
          report(`${PROMPT}${name} - the prompt library has no such prompt`)
      }
    })
  })
  return problems
}
```

If `scanStrings` is now unused, delete it; `npm run lint` says so.

- [ ] **Step 4: Run the tests, checks and ratchet**

Run: `cd ui && npx vitest run && npm run check && npm run lint && npm run metrics -- --check ../docs/stabilization/ui/baseline.json`
Expected: all PASS. The existing messages keep their wording, so the existing flow and EditorPage tests still pass; if one asserted the exact old message for a `previous_result:` with a property, update it to the full reference, which is what the engine names.

- [ ] **Step 5: Lower the baseline if it fell, and commit**

```bash
cd ui && npm run metrics -- --write ../docs/stabilization/ui/baseline.json && cd ..
git add ui/src/lib/flow.ts ui/src/lib/flow.test.ts docs/stabilization/ui/baseline.json
git commit -m "fix(ui): the live reference check resolves names as the engine does"
```

---

### Task 5: Workspace names follow the engine's rule (U3)

The UI allows `[A-Za-z0-9_][A-Za-z0-9_-]*`; the engine's
`WORKSPACE_NAME_PATTERN` is `^[\w][\w.-]*\Z` with Python's Unicode `\w`
(letters, digits, underscore) and a 100-character cap. One case file is read
by a pytest and a vitest, so the two rules cannot drift without a test
failing.

**Files:**
- Create: `tests/fixtures/workspace_names.json`
- Modify: `ui/src/lib/workspaceActions.ts` (`NAME`, a length cap)
- Test: `ui/src/lib/workspaceActions.test.ts`
- Test: `tests/test_ui_twins.py`

**Interfaces:**
- Produces: `workspaceNameError(name: string): string | null` keeps its signature; `MAX_WORKSPACE_NAME_LENGTH` (100) exported from `workspaceActions.ts`.

- [ ] **Step 1: Write the shared cases**

`tests/fixtures/workspace_names.json` (the 100- and 101-character names written out in full):

```json
[
  { "name": "ep4", "valid": true },
  { "name": "ep4.v2", "valid": true },
  { "name": "under_score-dash", "valid": true },
  { "name": "café", "valid": true },
  { "name": "_lead", "valid": true },
  { "name": ".hidden", "valid": false },
  { "name": "-lead", "valid": false },
  { "name": "a/b", "valid": false },
  { "name": "a b", "valid": false },
  { "name": "outputs", "valid": false },
  { "name": "", "valid": false },
  { "name": "<100 x characters>", "valid": true },
  { "name": "<101 x characters>", "valid": false }
]
```

- [ ] **Step 2: Write both tests**

Append to `tests/test_ui_twins.py`:

```python
import json

import pytest

from dw.security import InvalidInputError, SecurityError, validate_workspace_name
from dw.workspace import RESERVED_WORKSPACE_NAMES

WORKSPACE_NAMES = json.loads(
    (REPO / "tests" / "fixtures" / "workspace_names.json").read_text()
)


@pytest.mark.parametrize(
    "case", WORKSPACE_NAMES, ids=lambda c: c["name"][:12] or "empty"
)
def test_the_engine_decides_each_shared_workspace_name_case(case):
    if case["valid"]:
        validate_workspace_name(case["name"], reserved=RESERVED_WORKSPACE_NAMES)
    else:
        with pytest.raises((InvalidInputError, SecurityError)):
            validate_workspace_name(case["name"], reserved=RESERVED_WORKSPACE_NAMES)


def test_the_ui_reserves_the_engines_workspace_names():
    assert ts_string_array(
        UI_LIB / "workspaceActions.ts", "RESERVED_WORKSPACE_NAMES"
    ) == list(RESERVED_WORKSPACE_NAMES)
```

Append to `ui/src/lib/workspaceActions.test.ts`:

```ts
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'

const cases: { name: string; valid: boolean }[] = JSON.parse(
  readFileSync(
    fileURLToPath(
      new URL('../../../tests/fixtures/workspace_names.json', import.meta.url),
    ),
    'utf8',
  ),
)

it.each(cases)('decides $name as the engine does', ({ name, valid }) => {
  expect(workspaceNameError(name) === null).toBe(valid)
})
```

(`workspaceNameError` is imported at the top of the file with the module's other exports; add it if it is not.)

- [ ] **Step 3: Run both to see them fail**

Run: `venv/bin/python -m pytest tests/test_ui_twins.py -q` then `cd ui && npx vitest run src/lib/workspaceActions.test.ts`
Expected: pytest PASSES (the engine is the owner; if a case fails here, the case is wrong). vitest FAILS on `ep4.v2`, `café` and the 101-character name.

- [ ] **Step 4: Match the rule**

In `ui/src/lib/workspaceActions.ts`, replace `const NAME = ...` and the pattern check:

```ts
/** dw/security.py's WORKSPACE_NAME_PATTERN, `^[\w][\w.-]*\Z`: Python's \w
 * is Unicode letters and digits plus underscore, so \p{L}\p{N}_ here.
 * tests/fixtures/workspace_names.json is read by both sides' tests. */
const NAME = /^[\p{L}\p{N}_][\p{L}\p{N}_.-]*$/u
export const MAX_WORKSPACE_NAME_LENGTH = 100
```

and in `workspaceNameError`, before the pattern test:

```ts
  if (name.length > MAX_WORKSPACE_NAME_LENGTH)
    return `Use at most ${MAX_WORKSPACE_NAME_LENGTH} characters`
  if (!NAME.test(name))
    return 'Start with a letter, digit or _; then letters, digits, _, - and .'
```

- [ ] **Step 5: Run the tests and checks**

Run: `cd ui && npx vitest run && npm run check && npm run lint && npm run metrics -- --check ../docs/stabilization/ui/baseline.json` then `venv/bin/python -m pytest tests/test_ui_twins.py -q`
Expected: all PASS. If an existing test asserted the old message text, update it to the new one.

- [ ] **Step 6: Commit**

```bash
git add tests/fixtures/workspace_names.json tests/test_ui_twins.py ui/src/lib/workspaceActions.ts ui/src/lib/workspaceActions.test.ts
git commit -m "fix(ui): workspace names follow the engine's pattern and length cap"
```

---

### Task 6: The job page renders by the server's media kinds (U4)

`JobPage.svelte` decides image or video from its own extension regexes, which
miss `.bmp`, `.mov`, `.mkv`, `.avi` and every audio file; an audio output is
shown as a bare link. `dw/server/outputs.py`'s `MEDIA_KINDS` is the owner.
The job detail gains one top-level map, `output_kinds`, so no manifest entry
changes shape and no existing manifest assertion moves.

**Files:**
- Modify: `dw/server/outputs.py` (`output_kinds`)
- Modify: `dw/server/routes/jobs.py` (`get_job`)
- Test: `tests/test_server.py`
- Modify: `ui/src/lib/types.ts` (`JobDetail.output_kinds`)
- Modify: `ui/src/lib/pages/JobPage.svelte`
- Test: `ui/src/lib/pages/JobPage.test.ts`

**Interfaces:**
- Produces: `output_kinds(manifest: list | None) -> dict[str, str | None]` in `dw/server/outputs.py`: every file a manifest entry lists, mapped to its `MEDIA_KINDS` value, or `None` for a kind the gallery does not show. `GET /api/jobs/{id}` carries it as `output_kinds`.
- Produces: `JobDetail.output_kinds?: Record<string, MediaKind | 'text' | null>` in `types.ts`.

- [ ] **Step 1: Write the failing server test**

Append to `tests/test_server.py`, after `test_job_files_are_reported_relative_to_the_output_dir`:

```python
def test_a_job_reports_each_output_files_media_kind(server, tmp_path):
    """The job page renders by kind, so the kind comes from MEDIA_KINDS
    rather than an extension list in the browser."""
    outputs = tmp_path / "outputs"
    files = [str(outputs / name) for name in ("a.bmp", "b.mov", "c.flac", "d.bin")]

    def script(command):
        yield {
            "type": "success",
            "message": "ok",
            "run_count": 1,
            "manifest": [{"step": "gen", "files": files}],
        }

    with server(script) as client:
        job = client.post("/api/jobs", json={"workflow": valid_workflow()}).json()
        detail = wait_for_status(client, job["id"], TERMINAL_STATES)
        assert detail["output_kinds"] == {
            "a.bmp": "image",
            "b.mov": "video",
            "c.flac": "audio",
            "d.bin": None,
        }


def test_output_kinds_tolerates_a_job_with_no_manifest():
    from dw.server.outputs import output_kinds

    assert output_kinds(None) == {}
    assert output_kinds([{"step": "s"}, "not an entry"]) == {}
```

- [ ] **Step 2: Run them to see them fail**

Run: `venv/bin/python -m pytest tests/test_server.py -q -k "media_kind or output_kinds"`
Expected: FAIL (`KeyError: 'output_kinds'`, `ImportError`).

- [ ] **Step 3: Classify on the server**

In `dw/server/outputs.py`, after `MEDIA_KINDS`:

```python
def output_kinds(manifest):
    """Each file a job manifest lists, mapped to its MEDIA_KINDS entry, or
    None for a kind the gallery does not show. A client renders an output
    by this rather than keeping its own extension list."""
    kinds = {}
    for entry in manifest or []:
        if not isinstance(entry, dict):
            continue
        for name in entry.get("files") or []:
            kinds[name] = MEDIA_KINDS.get(os.path.splitext(name)[1].lower())
    return kinds
```

(`os` is already imported there; add it if not.) In `dw/server/routes/jobs.py`'s `get_job`, replace the return:

```python
    # a historical job is already a detail dict; a live one renders itself
    detail = job if isinstance(job, dict) else manager.describe(job)
    return {**detail, "output_kinds": output_kinds(detail.get("manifest"))}
```

with `from ..outputs import output_kinds` among the module's imports.

- [ ] **Step 4: Run the server tests**

Run: `venv/bin/python -m pytest tests/test_server.py tests/test_server_jobs.py tests/test_mcp_*.py -q`
Expected: PASS.

- [ ] **Step 5: Write the failing UI test**

In `ui/src/lib/pages/JobPage.test.ts`, add:

```ts
it('renders each output by the kind the server reports', async () => {
  detail.job = {
    ...job([{ step: 'generate', files: ['a.bmp', 'b.mov', 'c.flac'] }]),
    output_kinds: { 'a.bmp': 'image', 'b.mov': 'video', 'c.flac': 'audio' },
  }
  const { container } = render(JobPage, { jobId: 'j1' })
  await waitFor(() => expect(container.querySelector('audio')).toBeTruthy())
  expect(container.querySelector('img')?.getAttribute('src')).toContain('a.bmp')
  expect(container.querySelector('video')?.getAttribute('src')).toContain('b.mov')
  expect(container.querySelector('audio')?.getAttribute('src')).toContain('c.flac')
})
```

In the existing tests that render media and read metadata (the one around the `'a.png', 'clip.mp4'` manifest, and any other that expects an `<img>` or a `galleryMetadata` call), add `output_kinds` to the job they build, for example `{ ...job([...]), output_kinds: { 'a.png': 'image', 'clip.mp4': 'video' } }`, because the page no longer guesses from the extension.

- [ ] **Step 6: Run it to see it fail**

Run: `cd ui && npx vitest run src/lib/pages/JobPage.test.ts`
Expected: the new test FAILS (no `<audio>`, `.bmp` not shown as an image).

- [ ] **Step 7: Render by kind**

In `ui/src/lib/types.ts`, add to `JobDetail`:

```ts
  /** Each output file's kind, from the server's MEDIA_KINDS; null for a
   * kind the gallery does not show. */
  output_kinds?: Record<string, 'image' | 'video' | 'audio' | 'text' | null>
```

In `ui/src/lib/pages/JobPage.svelte`, replace the two extension helpers:

```ts
  const kindOf = (path: string) => job?.output_kinds?.[path] ?? null
```

and replace `isImage(file)` with `kindOf(file) === 'image'` at both uses (the metadata filter and the markup), `isVideo(file)` with `kindOf(file) === 'video'`, and add an audio branch before `{:else}`:

```svelte
              {:else if kindOf(file) === 'audio'}
                <span class="frame">
                  <audio src={fileUrl(file)} controls></audio>
                </span>
```

- [ ] **Step 8: Run the tests, checks and ratchet**

Run: `cd ui && npx vitest run && npm run check && npm run lint && npm run metrics -- --check ../docs/stabilization/ui/baseline.json`
Expected: all PASS.

- [ ] **Step 9: Commit**

```bash
git add dw/server/outputs.py dw/server/routes/jobs.py tests/test_server.py ui/src/lib/types.ts ui/src/lib/pages/JobPage.svelte ui/src/lib/pages/JobPage.test.ts
git commit -m "fix(ui): the job page renders outputs by the server's media kinds, audio included"
```

---

### Task 7: Every writable content type is offered, and an unlisted one is kept (U5)

`CONTENT_TYPES` (`ui/src/lib/editor.ts`) is a closed `<select>` without
`audio/flac`, `audio/ogg`, `audio/opus` or `audio/aiff`, all of which the
engine writes; a step holding one shows a blank select.

**Files:**
- Modify: `ui/src/lib/editor.ts` (`CONTENT_TYPES`, `contentTypeOptions`)
- Modify: `ui/src/lib/editor/StepEditor.svelte` (the result select)
- Test: `ui/src/lib/editor.test.ts`
- Test: `tests/test_ui_twins.py`

**Interfaces:**
- Produces: `contentTypeOptions(current?: string): string[]` in `editor.ts`: `CONTENT_TYPES`, plus `current` at the end when it is set and not already listed.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_ui_twins.py`:

```python
from dw.content_types import AUDIO_FORMATS, MUXED_VIDEO_CONTENT_TYPE, content_type_fault


def test_every_content_type_the_ui_offers_is_one_the_engine_writes():
    offered = ts_string_array(UI_LIB / "editor.ts", "CONTENT_TYPES")
    assert [ct for ct in offered if content_type_fault(ct)] == []


def test_the_ui_offers_every_audio_container_and_the_video_one():
    offered = set(ts_string_array(UI_LIB / "editor.ts", "CONTENT_TYPES"))
    written = {extension for extension, _ in AUDIO_FORMATS.values()}
    reachable = {AUDIO_FORMATS[ct][0] for ct in offered if ct in AUDIO_FORMATS}
    assert reachable == written
    assert MUXED_VIDEO_CONTENT_TYPE in offered
```

Append to `ui/src/lib/editor.test.ts` (import `contentTypeOptions` and `CONTENT_TYPES` from `./editor`):

```ts
describe('contentTypeOptions', () => {
  it('keeps a value the list does not hold, so the select shows it', () => {
    expect(contentTypeOptions('audio/x-flac')).toEqual([
      ...CONTENT_TYPES,
      'audio/x-flac',
    ])
  })
  it('adds nothing for a listed value or none', () => {
    expect(contentTypeOptions('image/png')).toEqual([...CONTENT_TYPES])
    expect(contentTypeOptions(undefined)).toEqual([...CONTENT_TYPES])
  })
})
```

- [ ] **Step 2: Run them to see them fail**

Run: `venv/bin/python -m pytest tests/test_ui_twins.py -q -k content_type` then `cd ui && npx vitest run src/lib/editor.test.ts`
Expected: the audio-container test FAILS (`.flac`, `.ogg`, `.aiff` unreachable); `contentTypeOptions` is not defined.

- [ ] **Step 3: Complete the list and keep an unlisted value**

In `ui/src/lib/editor.ts`, replace `CONTENT_TYPES` and add the helper:

```ts
/** What a step's result can be written as. tests/test_ui_twins.py pins it:
 * each one is accepted by dw/content_types.py, and every audio container
 * the engine writes is reachable from one. */
export const CONTENT_TYPES = [
  'image/png',
  'image/jpeg',
  'image/webp',
  'image/gif',
  'video/mp4',
  'audio/wav',
  'audio/flac',
  'audio/aiff',
  'audio/mpeg',
  'audio/ogg',
  'audio/opus',
  'application/json',
  'text/plain',
]

/** The select's options: a value the list does not hold (an alias like
 * audio/x-flac, written by hand) is kept as the last option rather than
 * shown as a blank select. */
export function contentTypeOptions(current?: string): string[] {
  return current && !CONTENT_TYPES.includes(current)
    ? [...CONTENT_TYPES, current]
    : [...CONTENT_TYPES]
}
```

In `ui/src/lib/editor/StepEditor.svelte`, import `contentTypeOptions` with the other `../editor` imports and change the result select's loop to:

```svelte
          {#each contentTypeOptions(step.result?.content_type) as contentType (contentType)}<option
              >{contentType}</option
            >{/each}
```

Remove `CONTENT_TYPES` from that file's imports if it is no longer used there.

- [ ] **Step 4: Run the tests, checks and ratchet**

Run: `cd ui && npx vitest run && npm run check && npm run lint && npm run metrics -- --check ../docs/stabilization/ui/baseline.json` then `venv/bin/python -m pytest tests/test_ui_twins.py -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add ui/src/lib/editor.ts ui/src/lib/editor/StepEditor.svelte ui/src/lib/editor.test.ts tests/test_ui_twins.py
git commit -m "fix(ui): offer every content type the engine writes; keep an unlisted one"
```

---

### Task 8: Pin the copies that agree today

These agree with their owners now and nothing would notice a drift. The
two terminal-state lists in the UI become one first.

**Files:**
- Modify: `ui/src/lib/api.ts` (export `TERMINAL_STATUSES`)
- Modify: `ui/src/lib/pages/JobPage.svelte` (import it; delete `TERMINAL`)
- Test: `tests/test_ui_twins.py`

**Interfaces:**
- Produces: `TERMINAL_STATUSES` exported from `ui/src/lib/api.ts`.

- [ ] **Step 1: Write the pins**

Append to `tests/test_ui_twins.py`:

```python
import json as _json

from dw.server.job_record import TERMINAL_STATES
from dw.workspace import DEFAULT_WORKSPACE_NAME


def test_the_ui_knows_the_servers_terminal_job_states():
    assert set(ts_string_array(UI_LIB / "api.ts", "TERMINAL_STATUSES")) == set(
        TERMINAL_STATES
    )


def test_the_job_page_has_no_terminal_states_of_its_own():
    assert "const TERMINAL =" not in (UI_LIB / "pages" / "JobPage.svelte").read_text()


def test_the_ui_default_workspace_is_the_servers():
    assert (
        ts_constants(UI_LIB / "workspace.svelte.ts")["DEFAULT_WORKSPACE"]
        == DEFAULT_WORKSPACE_NAME
    )


def _cache_type_enum():
    schema = _json.loads((REPO / "dw" / "workflow_schema.json").read_text())

    def walk(node):
        if isinstance(node, dict):
            cache = node.get("cache")
            if isinstance(cache, dict):
                found = cache.get("properties", {}).get("type", {}).get("enum")
                if found:
                    return found
            for child in node.values():
                found = walk(child)
                if found:
                    return found
        elif isinstance(node, list):
            for child in node:
                found = walk(child)
                if found:
                    return found
        return None

    found = walk(schema)
    assert found, "no cache type enum in the workflow schema"
    return found


def test_the_ui_cache_types_are_the_schemas():
    assert set(ts_string_array(UI_LIB / "editor.ts", "CACHE_TYPES")) == set(
        _cache_type_enum()
    )
```

- [ ] **Step 2: Run them to see the two expected failures**

Run: `venv/bin/python -m pytest tests/test_ui_twins.py -q`
Expected: `test_the_ui_knows_the_servers_terminal_job_states` FAILS (`TERMINAL_STATUSES` is not exported, and `ts_string_array` reads exported arrays only) and `test_the_job_page_has_no_terminal_states_of_its_own` FAILS. The cache-type and default-workspace pins PASS; if one fails, the copies disagree and that is a finding to fix in the UI.

- [ ] **Step 3: One list in the UI**

In `ui/src/lib/api.ts`, change `const TERMINAL_STATUSES = [...]` to `export const TERMINAL_STATUSES = [...]`. In `ui/src/lib/pages/JobPage.svelte`, delete `const TERMINAL = [...]`, import `TERMINAL_STATUSES` from `../api` (it already imports `api` from there), and replace each `TERMINAL` use with `TERMINAL_STATUSES`.

- [ ] **Step 4: Run everything**

Run: `cd ui && npx vitest run && npm run check && npm run lint && npm run metrics -- --check ../docs/stabilization/ui/baseline.json` then `venv/bin/python -m pytest tests/test_ui_twins.py -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add ui/src/lib/api.ts ui/src/lib/pages/JobPage.svelte tests/test_ui_twins.py
git commit -m "test(ui): pin the UI's copies of engine vocabularies to their owners"
```

---

### Task 9: Seam-map rows

**Files:**
- Modify: `docs/ARCHITECTURE.md` (`## UI` section)

- [ ] **Step 1: Add the rows**

Read the `## UI` section's existing rows first and keep their format. Add:

| Concept | Owner | Rule | Enforced by |
| --- | --- | --- | --- |
| Reference prefixes in the UI | `ui/src/lib/references.ts` | The only module in `ui/src` that spells a reference prefix; each equals `dw/references.py`'s. | `tests/test_ui_twins.py::test_the_ui_spells_every_reference_prefix_the_engine_does`; `prefix_literals` in `ui/scripts/arch-metrics.mjs` |
| Output kinds on the job page | `dw/server/outputs.py`: `MEDIA_KINDS`, `output_kinds` | The page renders an output by the `output_kinds` the job detail carries, never by its extension. | `tests/test_server.py::test_a_job_reports_each_output_files_media_kind`, `ui/src/lib/pages/JobPage.test.ts` |
| UI copies of engine rules | the constants `tests/test_ui_twins.py` reads | A list the UI must hold equals its owner, read from the TS source; a rule tested on both sides reads one case file in `tests/fixtures/`. | `tests/test_ui_twins.py` |
| UI architecture ratchet | `ui/scripts/arch-metrics.mjs` | No metric rises above `docs/stabilization/ui/baseline.json`. | the `ui` job in `.github/workflows/ci.yml`; `npm run preflight` |

- [ ] **Step 2: Commit**

```bash
git add docs/ARCHITECTURE.md
git commit -m "docs: seam-map rows for the UI's owners and its ratchet"
```

---

### Task 10: Gate 0

**Files:**
- Modify: `docs/stabilization/ui/ROADMAP.md` (gate report, Phase 0 status)

- [ ] **Step 1: Full checks**

Run: `scripts/preflight.sh`
Expected: every step passes, including `ui preflight` (check, lint, format, ratchet, build, unit, e2e). On an MPS Mac, run pytest with `DW_DEVICE=cpu` (the template-validation tests depend on host capacity; see the engine ROADMAP's Gate 4 section).

- [ ] **Step 2: lem smoke**

Merge to develop, deploy to lem (`scripts/deploy.sh`, which rsyncs the built UI), and in the browser: open a finished job with an audio output and confirm the player; open a job with a `.mov` or `.bmp` output if one exists; in the editor, reference a `for_each` member (`previous_result:<step>@<entry>`) and confirm no inline warning; create a workspace named with a dot and confirm the server accepts it.

- [ ] **Step 3: Write the gate report**

In `docs/stabilization/ui/ROADMAP.md` under *Gate reports*, add `### Gate 0 (<date>, develop <sha>)` with: the ratchet table (before Phase 0 and gate 0 columns), the ten largest files, the ten most complex functions (from `npm run metrics`'s ESLint pass, or ESLint's `complexity` output), SLOC by `.svelte` block, and what the lem smoke showed. Set Phase 0's status to `done <date> (ui-stabilization-gate-0)`.

- [ ] **Step 4: Tag and commit**

```bash
git add docs/stabilization/ui/ROADMAP.md
git commit -m "docs(ui-stabilization): gate 0 report"
git tag ui-stabilization-gate-0
```

Push the tag only with Don's go-ahead.
