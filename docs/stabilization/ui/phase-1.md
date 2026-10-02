# UI Phase 1: One Owner per Rule - Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every rule the UI or `dw_mcp` knows that the engine owns is either read from the server or pinned to its owner by a test, and the reference prefixes are spelled in exactly one UI module; the server takes over the logic `dw_mcp` computes client-side, and `dw_mcp`'s internal duplicates collapse once that logic is gone.

**Architecture:** Three stages, each ending green. 1a works in `ui/` and the shared case files: `references.ts` gains the helpers that let every other module stop spelling prefixes; one name-segment rule replaces four copies; closed selects keep a value they do not list; the editor's live reference check and the engine are both run against one case file, with the engine as the oracle. 1b adds or widens server routes (`dw/server/`) so `dw_mcp` becomes call-and-reshape (`docs/stabilization/mcp-assessment.md` stage 2). 1c consolidates `dw_mcp` (stage 3).

**Tech Stack:** Svelte 5, TypeScript, Vitest, Python 3.12, pytest, FastAPI, Pillow.

**Spec:** [ROADMAP.md](ROADMAP.md) Phase 1 row and Gate 0 report (its deferred findings); [ASSESSMENT.md](ASSESSMENT.md); [../mcp-assessment.md](../mcp-assessment.md) M3, M4, M7, M8.

## Global Constraints

- Freeze (Don, 2026-10-01): no new UI features until gate 2. A new server route is in scope only where it replaces logic a client computes today.
- No metric in either ratchet rises: `cd ui && npm run metrics -- --check ../docs/stabilization/ui/baseline.json` and `venv/bin/python scripts/arch_metrics.py --check docs/stabilization/baseline.json` before every commit. A task that lowers one rewrites that baseline in its commit.
- Every new test fails on the code before its fix; run it and see the failure first.
- No count-pinning assertions. Comments say why, not ticket history.
- The MCP surface budget (`tests/test_mcp_server.py`: `SURFACE_BUDGET`, `CLIENT_TEXT_LIMIT`) holds; a tool description that changes is re-measured by that file.
- A route a released client calls keeps answering as it did; new behaviour is opt-in by parameter or a new route.
- Commands: UI from `ui/` (`npx vitest run <file>`, `npm run check`, `npm run lint`, `npx prettier --write <files>`); Python from the repo root (`venv/bin/python -m pytest <path> -q`, `venv/bin/ruff format <files>`, `venv/bin/ruff check <files>`). On an MPS Mac the full pytest suite runs with `DW_DEVICE=cpu`; integration tests run without it.
- Imports at the top of a Python module (ruff E402).
- Commits: conventional prefix; end each message with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Branch `ui-stabilization/phase-1` from develop.

## Review Focus

1. A `prompt:` picked from the picker for a prompt whose name has a folder (`cast/priya`) round-trips: the reference written is `prompt:cast/priya` and the tooltip finds its text (Task 1a-1).
2. A folder or prompt name with a non-ASCII letter is accepted in the editor exactly when the workspace rule accepts it, and an astral-plane name of 100 code points is accepted (Task 1a-2).
3. A workflow with `torch_dtype: "torch.float8_e4m3fn"` opens with that dtype selected, and saving without touching it keeps it (Task 1a-3).
4. A `previous_result:` naming a `for_each` group from a step that iterates the *same* list is accepted (the engine rewrites it to the member), while the same reference from a step outside it is flagged (Task 1a-4).
5. An older MCP client that never sends the new frames budget parameter gets exactly today's `/frames` answer (Stage 1b).

---

## Stage 1a: the UI's own copies

### Task 1a-1: Prefixes only in `references.ts`

Fifteen string literals outside `ui/src/lib/references.ts` still spell a
reference prefix (`npm run metrics` lists them under `detail.prefix_literals`).
Each builds a reference, tests for one, or strips one; `references.ts` gains
one helper per job and every site calls it.

**Files:**
- Modify: `ui/src/lib/references.ts`
- Modify: `ui/src/lib/editor.ts` (`emptyStep`, `referenceSuggestions`)
- Modify: `ui/src/lib/prompts.ts` (`promptTooltip`, `promptListId`)
- Modify: `ui/src/lib/flow.ts` (`VARIABLE_PREFIX`)
- Modify: `ui/src/lib/editor/ArgumentsEditor.svelte:137`, `ui/src/lib/editor/VariablesForm.svelte:81`, `ui/src/lib/pages/EditorPage.svelte:445`, `ui/src/lib/pages/PromptEditorPage.svelte:554`, `ui/src/lib/pages/WorkflowPage.svelte:173`
- Test: `ui/src/lib/references.test.ts` (new), `ui/src/lib/prompts.test.ts`
- Modify: `docs/stabilization/ui/baseline.json`

**Interfaces:**
- Produces, from `references.ts`: `reference(prefix: Prefix, name: string): string` (`prefix + name`); `referenceName(value: unknown, prefix: Prefix): string | null` (the name after `prefix`, trimmed, or null when `value` is not a string starting with it); `type Prefix = (typeof PREFIXES)[number]`.

- [ ] **Step 1: Write the failing tests**

`ui/src/lib/references.test.ts`:

```ts
import { describe, expect, it } from 'vitest'
import { PROMPT, VARIABLE, reference, referenceName } from './references'

describe('reference and referenceName', () => {
  it('build and read back a name, folders included', () => {
    expect(reference(PROMPT, 'cast/priya')).toBe('prompt:cast/priya')
    expect(referenceName('prompt:cast/priya', PROMPT)).toBe('cast/priya')
  })
  it('trim the name, as the engine does', () => {
    expect(referenceName('prompt: scenic ', PROMPT)).toBe('scenic')
  })
  it('answer null for another prefix or a non-string', () => {
    expect(referenceName('variable:x', PROMPT)).toBeNull()
    expect(referenceName(42, VARIABLE)).toBeNull()
  })
})
```

Append to `ui/src/lib/prompts.test.ts` (import `promptTooltip` if the file does not already):

```ts
it('finds the text of a foldered prompt picked from the picker', () => {
  expect(promptTooltip('prompt:cast/priya', { 'cast/priya': 'a portrait' })).toBe(
    'a portrait',
  )
})
```

- [ ] **Step 2: Run them**

Run: `cd ui && npx vitest run src/lib/references.test.ts src/lib/prompts.test.ts`
Expected: `references.test.ts` FAILS (`reference` is not exported). The prompts case passes today (it pins behaviour the refactor must keep).

- [ ] **Step 3: Add the helpers**

Append to `ui/src/lib/references.ts`:

```ts
export type Prefix = (typeof PREFIXES)[number]

/** A reference to `name` under `prefix`: how every module writes one. */
export function reference(prefix: Prefix, name: string): string {
  return prefix + name
}

/** The name a reference names under `prefix`, trimmed as the engine trims
 * it, or null when the value is not that kind of reference. */
export function referenceName(value: unknown, prefix: Prefix): string | null {
  return typeof value === 'string' && value.startsWith(prefix)
    ? value.slice(prefix.length).trim()
    : null
}
```

- [ ] **Step 4: Replace every site**

- `editor.ts` `emptyStep`: `arguments: { prompt: reference(VARIABLE, 'prompt') }`.
- `editor.ts` `referenceSuggestions`: `reference(VARIABLE, name)`, `reference(PROMPT, name)`, `reference(PREVIOUS_RESULT, step.name)`, and for a video step `reference(PREVIOUS_RESULT, `${step.name}.frames`)` and the same with `.audio`.
- `prompts.ts` `promptTooltip`: `const name = referenceName(value, PROMPT); return name === null ? undefined : texts[name] || undefined`. `promptListId`: `referenceName(value, PROMPT) === null ? undefined : PROMPT_LIST_ID`.
- `flow.ts`: delete `const VARIABLE_PREFIX = 'variable:'`; in `forEachMembers` use `const declaredName = referenceName(value, VARIABLE)` and `workflow.variables?.[declaredName]` when it is not null.
- The five `.svelte` sites: `reference(PROMPT, name)` (ArgumentsEditor, VariablesForm), `reference(PROMPT, promptName)` (EditorPage, WorkflowPage), `reference(PROMPT, savePath())` (PromptEditorPage), each importing what it uses from `../references` or `../../references`.

- [ ] **Step 5: Run everything and the ratchet**

Run: `cd ui && npx vitest run && npm run check && npm run lint && npm run metrics`
Expected: all pass; `prefix_literals` is 0. If a site remains in `detail.prefix_literals`, convert it.

- [ ] **Step 6: Lower the baseline and commit**

```bash
cd ui && npm run metrics -- --write ../docs/stabilization/ui/baseline.json && cd ..
git add ui/src docs/stabilization/ui/baseline.json
git commit -m "refactor(ui): reference prefixes are spelled only in references.ts"
```

---

### Task 1a-2: One name-segment rule

`workspaceActions.ts` has the engine's workspace rule (Unicode letters,
digits, `_`, then `.` and `-`); `EditorPage.svelte:328` and
`PromptEditorPage.svelte:366,373,377` repeat an ASCII-only `^[\w][\w.-]*$` for
folder and prompt names. The server has no pattern of its own for those (only
path validation), so the UI's rule is a convention - but one convention, not
four, and the same one the workspace uses. The length check counts code
points, as Python's `len` does.

**Files:**
- Create: `ui/src/lib/names.ts`
- Modify: `ui/src/lib/workspaceActions.ts`, `ui/src/lib/pages/EditorPage.svelte`, `ui/src/lib/pages/PromptEditorPage.svelte`
- Test: `ui/src/lib/names.test.ts`
- Modify: `tests/fixtures/workspace_names.json` (an astral case)

**Interfaces:**
- Produces: `isNameSegment(name: string): boolean` and `MAX_NAME_LENGTH` (100), from `names.ts`. `workspaceActions.ts` keeps exporting `MAX_WORKSPACE_NAME_LENGTH` as an alias so its tests hold.

- [ ] **Step 1: Write the failing tests**

`ui/src/lib/names.test.ts`:

```ts
import { describe, expect, it } from 'vitest'
import { isNameSegment } from './names'

describe('isNameSegment', () => {
  it.each(['ep4', 'ep4.v2', 'café', '_lead', 'cast-2'])('accepts %s', (n) =>
    expect(isNameSegment(n)).toBe(true),
  )
  it.each(['', '.hidden', '-lead', 'a/b', 'a b'])('refuses %s', (n) =>
    expect(isNameSegment(n)).toBe(false),
  )
  it('counts code points, as the engine does', () => {
    expect(isNameSegment('𝐀'.repeat(100))).toBe(true)
    expect(isNameSegment('𝐀'.repeat(101))).toBe(false)
  })
})
```

Add to `tests/fixtures/workspace_names.json` two cases: `"𝐀"` repeated 100 times (`valid: true`) and 101 times (`valid: false`), written out in full.

- [ ] **Step 2: Run them**

Run: `cd ui && npx vitest run src/lib/names.test.ts src/lib/workspaceActions.test.ts` then `venv/bin/python -m pytest tests/test_ui_twins.py -q`
Expected: `names.test.ts` FAILS (no module); the workspace vitest FAILS on the 100-code-point case (UTF-16 length 200); pytest PASSES (the engine is the oracle - if it fails, the case is wrong).

- [ ] **Step 3: The rule, once**

`ui/src/lib/names.ts`:

```ts
/** One name segment - a workspace, a folder, a stored workflow or prompt:
 * dw/security.py's WORKSPACE_NAME_PATTERN, `^[\w][\w.-]*\Z`, where Python's
 * \w is Unicode letters and digits plus underscore. Length is in code
 * points, as Python's len counts. tests/fixtures/workspace_names.json is
 * read by both sides' tests. */
const SEGMENT = /^[\p{L}\p{N}_][\p{L}\p{N}_.-]*$/u
export const MAX_NAME_LENGTH = 100

export function isNameSegment(name: string): boolean {
  return [...name].length <= MAX_NAME_LENGTH && SEGMENT.test(name)
}
```

In `workspaceActions.ts`, delete `NAME` and its comment, export `MAX_WORKSPACE_NAME_LENGTH = MAX_NAME_LENGTH`, keep the length message (now `[...name].length > MAX_NAME_LENGTH`) ahead of `if (!isNameSegment(name))`. In `EditorPage.svelte` and `PromptEditorPage.svelte`, replace each `/^[\w][\w.-]*$/.test(x)` with `isNameSegment(x)`.

- [ ] **Step 4: Run, check, ratchet, commit**

Run: `cd ui && npx vitest run && npm run check && npm run lint && npm run metrics -- --check ../docs/stabilization/ui/baseline.json` then `venv/bin/python -m pytest tests/test_ui_twins.py -q`
Expected: all pass.

```bash
git add ui/src tests/fixtures/workspace_names.json
git commit -m "fix(ui): one name-segment rule for workspaces, folders and prompts"
```

---

### Task 1a-3: A closed select keeps a value it does not list

`contentTypeOptions` (Phase 0) kept an unlisted content type selected. The
dtype select (`StepEditor.svelte:340`, `TORCH_DTYPES`) has the same defect:
the schema allows any string, so `torch.float8_e4m3fn` shows a blank select.
One helper serves both, and a rendered test pins both (the Phase 0 review
found only a unit test for the content type).

**Files:**
- Modify: `ui/src/lib/editor.ts` (`optionsWith`; `contentTypeOptions` delegates)
- Modify: `ui/src/lib/editor/StepEditor.svelte` (dtype select)
- Test: `ui/src/lib/editor.test.ts`, `ui/src/lib/editor/StepEditor.test.ts` (create if absent)

**Interfaces:**
- Produces: `optionsWith(listed: readonly string[], current?: string | null): string[]`.

- [ ] **Step 1: Write the failing tests**

Append to `editor.test.ts`:

```ts
describe('optionsWith', () => {
  it('appends a current value the list lacks', () => {
    expect(optionsWith(['a', 'b'], 'c')).toEqual(['a', 'b', 'c'])
  })
  it('leaves the list alone for a listed or empty value', () => {
    expect(optionsWith(['a', 'b'], 'a')).toEqual(['a', 'b'])
    expect(optionsWith(['a', 'b'], null)).toEqual(['a', 'b'])
  })
})
```

Create or extend `ui/src/lib/editor/StepEditor.test.ts` with one rendered case. Render `StepEditor` the way the existing `EditorPage.test.ts` renders a step (read it for the props and mocks the component needs), with a pipeline step whose `from_pretrained_arguments.torch_dtype` is `'torch.float8_e4m3fn'` and whose `result.content_type` is `'audio/x-flac'`, then:

```ts
expect((screen.getByLabelText('dtype') as HTMLSelectElement).value).toBe(
  'torch.float8_e4m3fn',
)
expect(
  (container.querySelector('select[id^="result-"]') as HTMLSelectElement).value,
).toBe('audio/x-flac')
```

- [ ] **Step 2: Run them**

Run: `cd ui && npx vitest run src/lib/editor.test.ts src/lib/editor/StepEditor.test.ts`
Expected: `optionsWith` FAILS (not exported); the dtype assertion FAILS (blank); the content-type assertion passes.

- [ ] **Step 3: Implement**

In `editor.ts`:

```ts
/** A select's options: a value the list does not hold (written by hand, or
 * an alias) is kept as the last option rather than shown as a blank. */
export function optionsWith(
  listed: readonly string[],
  current?: string | null,
): string[] {
  return current && !listed.includes(current) ? [...listed, current] : [...listed]
}
```

and make `contentTypeOptions(current)` return `optionsWith(CONTENT_TYPES, current)`. In `StepEditor.svelte`, the dtype select iterates `optionsWith(TORCH_DTYPES, pretrained.torch_dtype)`.

- [ ] **Step 4: Run, check, ratchet, commit**

Run: `cd ui && npx vitest run && npm run check && npm run lint && npm run metrics -- --check ../docs/stabilization/ui/baseline.json`

```bash
git add ui/src
git commit -m "fix(ui): the dtype select keeps a dtype it does not list"
```

---

### Task 1a-4: The editor's reference check and the engine read one case file

The live check in `flow.ts` is a second copy of the engine's reference
rules, kept because the editor warns as the author types. Phase 0 aligned
the cases it knew; the review found more it accepts and the engine refuses.
Rather than chase them one at a time, both sides run one case file:
`tests/fixtures/reference_cases.json`. pytest runs each case through
`Workflow.validation_errors()` - the engine is the oracle, so a case labelled
wrong fails there first - and vitest runs it through
`danglingReferenceDetails`.

**Files:**
- Create: `tests/fixtures/reference_cases.json`
- Test: `tests/test_ui_twins.py`, `ui/src/lib/flow.test.ts`
- Modify: `ui/src/lib/flow.ts`

**Interfaces:**
- Case shape: `{ "name": str, "workflow": {...}, "flagged": [str] }`. `flagged` lists, for each reference the engine refuses, a substring of it (`"previous_result:g"`); an empty list means the engine accepts every reference in the workflow. Both sides assert: each `flagged` substring appears in some reported problem, and no problem is reported when `flagged` is empty.

- [ ] **Step 1: Write the cases**

`tests/fixtures/reference_cases.json`, every step a cheap task so the engine
validates it without models (`{"name": ..., "task": {"command": "slice_audio", "arguments": {...}}}` - use any task whose arguments accept an arbitrary string; read `tests/test_previous_results.py` for the step shape it uses and copy it). Cases:

1. `plain`: step `a`, step `b` reading `previous_result:a` - flagged `[]`.
2. `property`: `previous_result:a.mask` - `[]`.
3. `dotted_name`: step `x.y`, `previous_result:x.y` - `[]`.
4. `member_with_property`: `for_each` step `g` over `[{"name": "a"}]`, `previous_result:g@a.mask` from a later step - `[]`.
5. `unknown_member`: same `g`, `previous_result:g@zzz` - `["g@zzz"]`.
6. `group_from_outside`: same `g`, `previous_result:g` from a plain later step - `["previous_result:g"]`.
7. `group_from_same_list`: `g` over `variable:shots`, then a `for_each` step over `variable:shots` reading `previous_result:g` - `[]`, with `variables.shots` declared as a two-entry list.
8. `gather_plain_step`: plain step `a`, `gather:a` - `["gather:a"]`.
9. `gather_for_each`: `g` as above, `gather:g` - `[]`.
10. `from_key_variable`: `{"from_previous_result": "variable:source"}` with `source` declared - `[]`.
11. `from_key_missing`: `{"from_previous_result": "nope"}` - `["nope"]`.

- [ ] **Step 2: Write both tests**

Append to `tests/test_ui_twins.py` (imports at the top: `from dw.workflow import workflow_from_definition`):

```python
REFERENCE_CASES = json.loads(
    (REPO / "tests" / "fixtures" / "reference_cases.json").read_text()
)


@pytest.mark.parametrize("case", REFERENCE_CASES, ids=lambda c: c["name"])
def test_the_engine_decides_each_shared_reference_case(case, tmp_path):
    workflow = workflow_from_definition(case["workflow"], str(tmp_path))
    problems = " | ".join(
        str(e.get("message", e)) for e in workflow.validation_errors()
    )
    for fragment in case["flagged"]:
        assert fragment in problems, problems
    if not case["flagged"]:
        assert "previous_result" not in problems and "gather" not in problems, problems
```

Append to `ui/src/lib/flow.test.ts`:

```ts
import referenceCases from '../../../tests/fixtures/reference_cases.json'

describe('the shared reference cases', () => {
  it.each(referenceCases)('$name', ({ workflow, flagged }) => {
    const messages = danglingReferenceDetails(workflow).map((d) => d.message)
    for (const fragment of flagged)
      expect(messages.some((m) => m.includes(fragment))).toBe(true)
    if (flagged.length === 0) expect(messages).toEqual([])
  })
})
```

- [ ] **Step 3: Run them**

Run: `venv/bin/python -m pytest tests/test_ui_twins.py -q -k reference_case` then `cd ui && npx vitest run src/lib/flow.test.ts`
Expected: pytest PASSES for every case. A case that fails there is mislabelled - correct the case from the engine's message, not the engine. vitest FAILS on `unknown_member`, `group_from_outside` and `gather_plain_step` (and `group_from_same_list` if it flags the group); the rest pass.

- [ ] **Step 4: Make the editor agree**

In `flow.ts`, extend `EarlierStep` with `forEachSource: unknown` (the step's `for_each` value) and `members: string[] | null` (`forEachMembers(workflow, step)`), so `earlierSteps` takes the workflow. Then:

- `resolvesTo(reference, step, consumer)`: a plain name match against a `for_each` step is accepted only when `consumer[FOR_EACH_KEY]` deep-equals `step.forEachSource` (the engine rewrites it to the member); a member reference `g@x[.prop]` is accepted when `step.members` is null (the list is not known until run time) or includes `x`.
- The `GATHER` check accepts only an earlier step with `forEach` true.
- `refTarget` keeps resolving for the graph as before (a group reference still draws an edge to the group), so pass a flag or a separate predicate: the graph asks "which step", the check asks "is this allowed".
- Correct `resolvesTo`'s doc comment: it now states the cases above rather than claiming to be the engine's rule.

Keep `complex_functions` from rising: split predicates rather than growing one function.

- [ ] **Step 5: Run, check, ratchet, commit**

Run: `cd ui && npx vitest run && npm run check && npm run lint && npm run metrics -- --check ../docs/stabilization/ui/baseline.json` then `venv/bin/python -m pytest tests/test_ui_twins.py -q`

```bash
git add tests/fixtures/reference_cases.json tests/test_ui_twins.py ui/src/lib/flow.ts ui/src/lib/flow.test.ts
git commit -m "fix(ui): the live reference check and the engine run one case file"
```

---

### Task 1a-5: Pin what remains, and strengthen the weak pins

**Files:**
- Test: `tests/test_ui_twins.py`, `tests/test_server.py`
- Modify: `ui/src/lib/editor.ts` (export `OFFLOAD_MODES` if the offload select's options are inline)

- [ ] **Step 1: Write the pins**

Append to `tests/test_ui_twins.py` (schema helpers may reuse `_cache_type_enum`'s walk; factor a `_schema_find(predicate)` if a third walker would appear):

- `WORKFLOW_SHAPES` and `WORKFLOW_TRAITS` (`ui/src/lib/types.ts`) equal the schema's `shape` and `traits` enums (`ts_string_array` reads `export const NAME = [...] as const`).
- `COMPONENT_SLOTS` (`editor.ts`) equals the keys of the schema's pipeline definition whose value is `{"$ref": "#/$defs/pipeline_component"}`.
- The offload select's options equal the schema's `offload` enum (`model`, `sequential`); if they are inline in `StepEditor.svelte`, move them to `export const OFFLOAD_MODES` in `editor.ts` first.
- Replace `test_the_ui_offers_every_audio_container_and_the_video_one` with a pin over distinct writer settings: `{(ext, tuple(sorted(args.items()))) for ext, args in AUDIO_FORMATS.values()}` must each be reachable from an offered content type, so dropping `audio/opus` (same `.ogg`, different subtype) fails.
- Replace `test_the_job_page_has_no_terminal_states_of_its_own` with a scan: no `.ts` or `.svelte` file under `ui/src` other than `lib/api.ts` (tests excluded) contains all three of `'succeeded'`, `'failed'`, `'cancelled'` inside one array literal (`re.search(r"\[[^\]]*'succeeded'[^\]]*'failed'[^\]]*'cancelled'[^\]]*\]", text)`).

Append to `tests/test_server.py`: a historical job whose stored manifest is `null` answers `GET /api/jobs/{id}` with `output_kinds == {}` (write the row through the job history the `server` fixture uses; read `tests/test_jobs_listing.py` for how a historical row is seeded).

- [ ] **Step 2: Run them**

Run: `venv/bin/python -m pytest tests/test_ui_twins.py tests/test_server.py -q -k "shape or trait or slot or offload or writer or terminal or null_manifest"`
Expected: each new pin PASSES on today's code, because these copies agree (that is why they are pins, not fixes). The strengthened audio pin is the one to prove: temporarily delete `'audio/opus'` from `CONTENT_TYPES`, see it FAIL, restore.

- [ ] **Step 3: Commit**

```bash
git add tests ui/src/lib/editor.ts ui/src/lib/editor/StepEditor.svelte
git commit -m "test(ui): pin the remaining engine vocabularies; strengthen two weak pins"
```

---

## Stage 1b: the server takes over what dw_mcp computes

`mcp-assessment.md` stage 2. Each task adds a route or an optional parameter
(no released client breaks), then deletes the client copy. The order is the
design survey's: the smallest client-only deletion first, the image work
last.

Rulings recorded with this plan (2026-10-01):
- `PATCH` is made atomic by a lock shared with `PUT`. An `ETag`/`If-Match`
  guard for the editor's whole-document saves is a new UI-facing behaviour
  and waits for the freeze to lift.
- `dw_mcp` stops clamping `max_dimension` to 64 before calling `/frames`; the
  server's rule (`max(1, max_dimension)`, sub-tiles floored at
  `FRAME_MIN_DIMENSION`, pinned by `tests/test_server.py`'s 32px-tile test)
  is the only one.
- The `hear` excerpt budget stays in `dw_mcp`: the UI never asks `/frames`
  for audio. Once the server budgets the tiles, the client's part is one
  subtraction.
- No `dw_mcp/consent.py` and no split of `diagnose.py`: each gate is two
  lines with its own refusal text, already pinned by
  `tests/test_mcp_twins.py::test_every_tool_that_takes_acknowledged_cost_refuses_without_it`.
- A stdio `dw-mcp` newer than a remote `dw.serve` loses the new routes; the
  release note says to upgrade the server first.

### Task 1b-1: `server_info` reads `/api/server` alone (M7.5)

**Files:** Modify `dw_mcp/workspaces.py` (`server_info`, about lines 183-212); Test `tests/test_mcp_workspaces.py`.

- [ ] **Step 1:** Rewrite `test_mcp_workspaces.py`'s named-workspace `server_info` test to assert the one request is `GET /api/server?workspace=shots` and that `/api/workspaces` is never requested (record the paths the `MockTransport` sees). Run it: FAIL (the client also requests `/api/workspaces`).
- [ ] **Step 2:** Delete the re-derivation; `server_info` returns `client.get_json("/api/server")` reshaped as today, its directories now the server's own (workspace-scoped since #389). Run `tests/test_mcp_workspaces.py tests/test_mcp_server.py`: PASS.
- [ ] **Step 3:** Commit `refactor(mcp): get_server_info reads the workspace-scoped /api/server alone`.

### Task 1b-2: The server tolerates a null download and sends the acknowledgement (M7.3, M7.4)

**Files:** Modify `dw/server/admission.py` (`AcknowledgedCost.downloads` about line 302; `refuse()` about lines 383-426); Modify `dw_mcp/diagnose.py` (`_acknowledgement_body`), `dw_mcp/client.py` (`_format_detail`); Test `tests/test_server.py` (`TestBoundAcknowledgement`), `tests/test_mcp_diagnose.py`, `tests/test_mcp_client.py`.

**Interfaces:** Produces: a 409 cost refusal whose `detail` carries `"acknowledge": {"fingerprint": str, "minutes": float | None, "downloads": [str]}` whenever a plan exists - the exact `acknowledged_cost` to resend.

- [ ] **Step 1: Failing server tests.** In `TestBoundAcknowledgement`: `test_a_null_download_entry_is_tolerated` (submit with `acknowledged_cost.downloads: [null, "<repo>"]` matching the plan - expect 201, not 422); `test_the_409_carries_a_ready_acknowledgement` (submit unacknowledged; the 409 `detail["acknowledge"]` resent as `acknowledged_cost` is accepted). Run: both FAIL.
- [ ] **Step 2: Server.** `downloads: List[Optional[str]]` with a `field_validator("downloads", mode="before")` that drops `None`; in `refuse()`, when `plan is not None`, add `detail["acknowledge"] = {"fingerprint": ..., "minutes": ..., "downloads": [repo for repo in ... if repo]}` built from the same plan fields the refusal already reports. Run: PASS.
- [ ] **Step 3: Client.** Delete the null-stripping in `_acknowledgement_body` (rewrite the test that asserted stripping to assert the body is sent as given); in `_format_detail`, print `json.dumps(detail["acknowledge"])` when present and omit the sentence when absent (an older server), instead of rebuilding it; rewrite the `test_mcp_client.py` 409 cases to put `acknowledge` in the stub. Run `tests/test_mcp_*.py tests/test_server.py -q`: PASS; `ruff check --select C901` shows `_format_detail` below 13.
- [ ] **Step 4:** Commit `refactor(mcp): the server tolerates null downloads and sends the re-acknowledgement`.

### Task 1b-3: `DELETE /api/jobs/{id}/run` (M7.2)

**Files:** Modify `dw/server/jobs.py` (`JobManager.run_location`, factored from `realized` about lines 264-295), `dw/server/routes/gallery.py` (extract `delete_run_directory(path, root)` from the gallery DELETE about lines 353-386), `dw/server/routes/jobs.py` (new route), `dw_mcp/media.py` (`delete_output(job_id=)`); Test `tests/test_server_jobs.py`, `tests/test_mcp_media.py`.

**Interfaces:** Produces `JobManager.run_location(job_id) -> tuple[str, str] | None` (output root, run dir, confined as `realized` confines); `DELETE /api/jobs/{job_id}/run` -> `{job_id, run_dir, deleted, run_swept}`; 404 for an unknown job or one with no run dir (today's message); 409 for a queued or running job.

- [ ] **Step 1: Failing tests** in `tests/test_server_jobs.py`: `test_deleting_a_jobs_run_removes_its_run_directory`, `test_it_uses_the_jobs_own_workspace`, `test_a_job_without_a_run_dir_is_404`, `test_a_running_job_is_409`, `test_a_run_dir_escaping_the_root_is_refused`. Run: FAIL (405/404 - no route).
- [ ] **Step 2: Server.** Factor `run_location` out of `realized` (keep `realized` using it); move the run-directory removal (`record_run_versions`, `rmtree`, `_remove_empty_identity_folders`) into `delete_run_directory` and have the gallery DELETE call it; add the route. Run: PASS, and the existing gallery DELETE tests still pass.
- [ ] **Step 3: Client.** `delete_output(job_id=)` becomes one `client.delete_json(api_path("api", "jobs", job_id, "run"))`; the job carries its own workspace, so the `workspace` argument no longer applies to the `job_id` path - drop it there, and its test. Rewrite the `delete_output(job_id=)` tests to one call each. Run `tests/test_mcp_media.py tests/test_mcp_server.py`: PASS (surface budget included).
- [ ] **Step 4:** Commit `feat(server): DELETE /api/jobs/{id}/run; delete_output(job_id=) calls it`.

### Task 1b-4: `PATCH /api/workflows/{name}` (M7.1)

**Files:** Modify `dw/library.py` (`merge_patch`, moved from `dw_mcp/authoring.py`'s `_merge_patch`), `dw/server/routes/library.py` (extract `_write_validated(state, ws, name, definition)` from `PUT`; a module-level `threading.Lock` both take; the `PATCH` route), `dw_mcp/authoring.py` (`save_workflow` patch mode); Test `tests/test_server_library_path.py`, `tests/test_mcp_authoring.py`.

**Interfaces:** Produces `merge_patch(target: dict, patch: dict) -> dict` (RFC 7396) in `dw/library.py`; `PATCH /api/workflows/{name:path}` taking the raw patch object (`application/merge-patch+json` or `application/json`), answering as `PUT` does.

- [ ] **Step 1: Failing tests** in `tests/test_server_library_path.py`: `test_patch_merges_onto_the_stored_definition`, `test_patch_null_deletes_a_key`, `test_patch_of_an_example_writes_a_writable_copy`, `test_an_invalid_patch_result_is_a_400_and_writes_nothing`. Run: FAIL (405).
- [ ] **Step 2: Server.** Move `merge_patch` (with its tests, if `test_mcp_authoring.py` has unit tests of `_merge_patch`, to `tests/test_library.py`); extract `_write_validated`; `PATCH` under the lock: read the current definition the way `GET` resolves it, merge, `_write_validated`. `PUT` takes the same lock. Run: PASS, existing PUT tests still pass.
- [ ] **Step 3: Client.** `save_workflow` patch mode sends one `PATCH` with the patch; delete `_merge_patch` and the GET. Rewrite the two patch-mode tests to assert a single `("PATCH", "/api/workflows/<name>")` carrying the patch. Run `tests/test_mcp_authoring.py tests/test_mcp_server.py`: PASS.
- [ ] **Step 4:** Commit `feat(server): PATCH a stored workflow atomically; save_workflow patch mode calls it`.

### Task 1b-5: The metadata route reports level findings (M4)

**Files:** Modify `dw/server/assess.py` (`level_findings`), `dw/server/routes/media.py` (`gallery_metadata` about lines 56-138), `dw_mcp/catalog.py` (`get_gallery_metadata` about lines 290-361); Test `tests/test_server_assess.py`, `tests/test_mcp_catalog.py`.

**Interfaces:** Produces `level_findings(media: dict | None) -> list[dict]` in `dw/server/assess.py`, findings in `dw/assessment_rules.finding()` shape, reading `CLIPPED_WARN_DBFS`, `NEAR_SILENT_WARN_DBFS` and `NEAR_SILENT_QUIET_NOT_EMPTY_DBFS` from `dw/audio_qc.py`; the metadata response gains `"findings"`.

- [ ] **Step 1: Failing tests** in `tests/test_server_assess.py`: `test_metadata_reports_a_full_scale_finding` (peak at 0 dBFS -> a `warn` finding naming `normalize_audio`), `test_metadata_reports_near_silent_as_info_when_peaks_are_real`, `test_metadata_findings_are_empty_for_an_image`, `test_level_findings_read_audio_qc_thresholds` (monkeypatch `audio_qc.CLIPPED_WARN_DBFS` and see the finding move). Run: FAIL.
- [ ] **Step 2: Server.** `level_findings` mirrors `warn_if_written_near_silent`'s severities; the route adds `"findings": level_findings(media) if media else []`. Run: PASS.
- [ ] **Step 3: Client.** `next` is built from structured fields only: the job id (get_job_workflow), an asset with no job, an asset with media, shots (assess_output), and "see `findings`" when any are present. Delete the dBFS numbers and the Music 3 sentence (the skill owns it, `plugins/dw/skills/minimax-music3/SKILL.md`). Rewrite `test_mcp_catalog.py`'s metadata tests: no `audio_duration` text for any kind; the `loop_audio` asset hint stays. Run `tests/test_mcp_catalog.py tests/test_mcp_server.py`: PASS.
- [ ] **Step 4:** Commit `refactor(server): the metadata route reports level findings; dw_mcp stops restating the thresholds`.

### Task 1b-6: Inline images and frame budgets on the server (M3)

**Files:** Create `dw/server/inline_media.py`; Modify `dw/server/routes/media.py` (thumbnail onto the helper; new `/image` route registered beside `/thumbnail`; `/frames` `max_total_bytes`), `dw/media.py` (`base64_size` moves), `dw_mcp/media.py`; Test `tests/test_server_gallery_image.py` (new), `tests/test_server.py`, `tests/test_security_decoder_bombs.py`, `tests/test_mcp_media.py`, `tests/test_mcp_twins.py`.

**Interfaces:**
- `dw/server/inline_media.py`: `base64_size(n: int) -> int`; `fit_longest(image, limit) -> Image` (never upscales); `open_bounded(path) -> Image` (refuses over `MAX_DECODE_PIXELS` with 413 before `load()`); `encode_within_budget(image, limit, fmt, max_base64_bytes, floor=FRAME_MIN_DIMENSION) -> tuple[bytes, Image]`; `fit_tiles_within_budget(images, limit, max_base64_bytes) -> tuple[list[Image], int | None]`.
- `GET /api/gallery/{name:path}/image?max_dimension=768&crop=x,y,w,h&max_bytes=N&format=auto|png|jpeg`, `query_token_ok`, `asset:` names accepted; returns the image bytes with `X-DW-Original-Size`, `X-DW-Returned-Size`, and when they apply `X-DW-Crop` and `X-DW-Downscaled-To`; 404 "not an image" for another kind; 400 for a malformed crop.
- `/frames` gains `max_total_bytes: Optional[int]`; its answer gains `"downscaled_to"` (null when nothing shrank). Without the parameter the answer is unchanged.

- [ ] **Step 1: The helper, no behaviour change.** Write `tests/test_server_gallery_image.py::test_fit_longest_never_upscales` and `::test_encode_within_budget_halves_until_it_fits` against `inline_media` (FAIL: no module). Create the module; move `base64_size` (re-export from `dw/media.py` so `projected_wav_base64_size` keeps working); put the thumbnail route and `_encoded_tile` onto `fit_longest`. Run the new tests and every existing thumbnail and frames test: PASS. Commit `refactor(server): one inline-media helper for thumbnails and frames`.
- [ ] **Step 2: `/image`.** Failing route tests: `test_a_large_image_is_downscaled_to_max_dimension`, `test_a_crop_returns_that_region_at_full_resolution`, `test_a_crop_is_clamped_and_echoed`, `test_a_malformed_crop_is_a_400`, `test_max_bytes_halves_until_the_base64_fits`, `test_a_jpeg_source_comes_back_jpeg`, `test_a_video_is_404`, `test_an_asset_reference_is_read`, `test_the_query_token_is_accepted`; and in `test_security_decoder_bombs.py` a `TestGalleryImage` class carrying the decoder-bomb cases now in `TestGetOutputImage`. Implement the route on `open_bounded`, `resolve_crop_box` and `encode_within_budget`. Run: PASS. Commit `feat(server): GET /api/gallery/{name}/image - cropped, fitted, within a byte budget`.
- [ ] **Step 3: `/frames` budget.** Failing tests beside the 32px-tile test: `test_gallery_frames_shrinks_tiles_together_under_max_total_bytes`, `test_gallery_frames_without_a_budget_is_unchanged`. Keep the PIL tiles until the budget pass; encode once. Run: PASS. Commit `feat(server): /frames shrinks tiles together under max_total_bytes`.
- [ ] **Step 4: `dw_mcp` becomes call-and-reshape.** `get_output_image` calls `/image` with `max_dimension`, `crop`, `max_bytes=MAX_RETURNED_BYTES` and reshapes the headers into today's answer fields; `get_output_frames` passes `max_total_bytes=MAX_RETURNED_BYTES`, echoes `downscaled_to`, and gives `hear` excerpts `MAX_RETURNED_BYTES - sum(len(tile["data"]))`. Delete `_crop_box`, `_fit`, `_encode_within_budget`, `_fit_tiles_within_budget`, `MIN_DIMENSION`, `MAX_DECODE_PIXELS`, the client-side selector/`boundaries`/`names`/`hear` refusals (the server's 400s reach the caller), and `from PIL import Image`. Rewrite `tests/test_mcp_media.py`'s decode/downscale/crop/budget tests as forwarding-and-reshaping tests, its client-side refusals as "the server's 400 reaches the caller", and drop the pins for deleted constants from `tests/test_mcp_twins.py` and `test_security_decoder_bombs.py::test_the_mcp_limit_is_the_engines`. Add `TestStartupWeight`'s check that `dw_mcp` imports no `PIL`. Run `tests/test_mcp_*.py tests/test_server*.py tests/test_security_decoder_bombs.py`: PASS, surface budget included. Commit `refactor(mcp): images and frames are fitted on the server; dw_mcp drops Pillow`.

### Task 1b-7: Release note and map rows

**Files:** `docs/RELEASING.md` (0.8.0 notes: the new routes; a stdio `dw-mcp` needs a server at least as new), `docs/MCP.md` (any tool whose behaviour text changed), `docs/SERVER.md` (the new routes and parameters), `docs/ARCHITECTURE.md` (rows: inline media owner `dw/server/inline_media.py`; workflow patch owner `dw/library.py: merge_patch` and the route's lock; run deletion owner `JobManager.run_location` + `delete_run_directory`; level findings owner `dw/server/assess.py: level_findings`).

- [ ] Write them; run `venv/bin/python -m pytest tests/test_architecture_map.py -q` (the map's named tests must exist); commit `docs: stage 2 routes - release note, server and MCP guides, seam-map rows`.

## Stage 1c: dw_mcp consolidates

`mcp-assessment.md` stage 3, after 1b removed the code some duplicates
guarded.

### Task 1c-1: One confinement module

**Files:** Create `dw_mcp/confine.py`; Modify `dw_mcp/media.py` (`_remote_root`, `_confine`), `dw_mcp/assets.py` (`_remote_roots`, `_confine_source`); Test `tests/test_mcp_confine.py` (new), existing `tests/test_mcp_media.py`, `tests/test_mcp_assets.py`, `tests/test_security_symlinks.py`.

**Interfaces:** `remote_roots(client, workspace, keys, writable_libraries=False) -> list[str] | None` (None for a stdio client; raises `DwApiError` when a mounted server names no roots - each caller passes its own message) and `contains(path, roots) -> bool` (realpath of the nearest existing ancestor).

- [ ] Write `tests/test_mcp_confine.py` first (a symlink out of a root is outside; a not-yet-existing file under a root is inside; a stdio client gets None; the per-call workspace reaches `/api/server` and `/api/assets`) - FAIL (no module); implement; switch both callers, each keeping its own refusal text; run the four test files: PASS; commit `refactor(mcp): one confinement module for reads and writes`.

### Task 1c-2: Small duplicates

**Files:** `dw_mcp/client.py` (`base64_size`, `project`, `workflow_source`, `UNSHAREABLE_HOSTS`; delete `get_bytes`), `dw_mcp/media.py`, `dw_mcp/catalog.py`, `dw_mcp/assets.py`, `dw_mcp/workspaces.py`, `dw_mcp/authoring.py`, `dw_mcp/diagnose.py`; Test `tests/test_mcp_client.py` and the callers' files.

- [ ] For each helper, a unit test in `tests/test_mcp_client.py` first (`base64_size(3) == 4`, `base64_size(4) == 8`; `project` on a list and on a dict; `workflow_source` for each of the four spellings and for none; `UNSHAREABLE_HOSTS` includes the loopback set plus `0.0.0.0` and `::`), FAIL, then implement and switch the callers. `UNSHAREABLE_HOSTS = LOOPBACK_HOSTS | {"0.0.0.0", "::"}` keeps the `127.` prefix test beside it; the two sets stay two because they answer different questions. Delete `DwClient.get_bytes` once nothing but tests calls it, pointing those tests at `get_bytes_if`. Run `tests/test_mcp_*.py`: PASS; both ratchets clean. Commit `refactor(mcp): one home for the base64 size, projection, workflow-source parsing and unshareable hosts`.
- [ ] Update `docs/stabilization/mcp-assessment.md`: stages 2 and 3 done, M-item by M-item, with what was deliberately not done (the consent module, the `diagnose.py` split) and why.

## Gate 1

- [ ] `prefix_literals` 0 in the UI baseline.
- [ ] Every UI and `dw_mcp` copy of an engine rule is gone or pinned (`tests/test_ui_twins.py`, `tests/test_mcp_twins.py`) and has a seam-map row.
- [ ] `mcp-assessment.md` marks stages 2 and 3 done, M-item by M-item.
- [ ] `scripts/preflight.sh` green (pytest with `DW_DEVICE=cpu` on an MPS Mac; integration without it); both ratchets clean.
- [ ] One fresh-reviewer pass over the phase branch; Critical and Important fixed test-first.
- [ ] Merge, deploy to lem, smoke: in the browser, the editor's live warnings on the shared reference cases; over MCP, `get_output_image` with a crop, `get_output_frames` with `hear`, `get_gallery_metadata` on an audio output, `save_workflow` in patch mode, `delete_output(job_id=)` on a scratch run.
- [ ] Gate report in `ROADMAP.md`; tag `ui-stabilization-gate-1` (annotated); push the tag on Don's word.
