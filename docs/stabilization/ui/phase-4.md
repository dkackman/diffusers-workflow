# UI Phase 4: Contract and Guardrails - Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A server response change that breaks the UI fails CI, and the harness refuses a UI ratchet rise without `arch-approved`: the routes the UI reads declare Pydantic response models, the UI's response types are generated from the OpenAPI document those models produce, e2e runs before develop moves, and `ui/CLAUDE.md`'s guardrail rules are checks.

**Architecture:** Four stages, each ending green and merged on its own.

- **4a** builds the contract machinery and converts the system routes as the pilot:
  - `dw/server/api_models.py` holds an `ApiModel` base that is lenient at runtime and strict under `DW_STRICT_RESPONSES=1`.
  - `scripts/dump_openapi.py` writes the strict OpenAPI document to `ui/src/lib/generated/openapi.json`, and a pytest test fails when the committed copy is stale.
  - `openapi-typescript` turns that document into `ui/src/lib/generated/api-schema.ts`, and CI fails when that file is stale.
  - `types.ts` re-exports the generated types in place of its hand-written ones.
- **4b** converts the job and validation routes, using 4a's recipe.
- **4c** converts the library, gallery, asset and introspection routes. A twin pin then ties every `/api/` path in `api.ts` to a declared response model.
- **4d** adds the guardrails:
  - e2e before develop moves;
  - the UI metrics script's `--compare`;
  - the harness's UI ratchet (in harnest);
  - `ui/CLAUDE.md`'s guardrail rules turned into checks;
  - the gate.

How a break reaches CI: a server change alters the strict OpenAPI document. `tests/test_api_contract.py` then fails until `openapi.json` is regenerated. Once it is, CI's `ui` job regenerates `api-schema.ts`, and the diff fails until that is committed. With the regenerated types, `npm run check` fails wherever the UI reads a field the server no longer sends.

**Tech Stack:** Python 3.12, FastAPI 0.142, Pydantic 2.13, pytest; Svelte 5, TypeScript, svelte-check, Vitest, Playwright; openapi-typescript 7; the harness (`~/testing/harnest`, Python 3 + bash tests).

**Spec:** [ROADMAP.md](ROADMAP.md), the Phase 4 row and gate; [ASSESSMENT.md](ASSESSMENT.md) item 4 ("Contract"). The engine's own Phase 4 harness stage is in [../harness/stage-c-guardrails.md](../harness/stage-c-guardrails.md).

## Global Constraints

- **The payload does not change.** Every converted route answers exactly what it answered before for the same state:
  - the same keys;
  - an absent key stays absent, never `null`;
  - an int stays an int.
  - `dw_mcp`, older UIs and scripts read these routes too.
- Every converted route declares `response_model=` and `response_model_exclude_unset=True`. That is what keeps an absent key absent.
- The `ApiModel` config is `extra="allow"` at runtime, so an undeclared key passes through and nothing turns into a 500 in production. It is `extra="forbid"` when `DW_STRICT_RESPONSES=1`, so an undeclared key is a 500. Strict mode is set by:
  - `tests/conftest.py`, so every route test checks it;
  - the e2e fixture server;
  - `scripts/dump_openapi.py`, so the generated types have no index signature.
- Field rules:
  - A key the handler always emits is required.
  - A key it sometimes omits gets a default of `None` and is never sent as `null` (because of `exclude_unset`).
  - A key that can be `null` is typed `X | None`.
  - Numbers: `int` only where the value is always an int, `float` where it is always a float, and `int | float` otherwise. Pydantic turns an int in a `float` field into `1.0`.
- A model is documented once: types.ts's doc comments move onto the fields as `Field(description=...)`, and the generated TS carries them.
- There is one new Python module, `dw/server/api_models.py`, with one section per router. Adding it raises `modules` 167 → 168 in `docs/stabilization/baseline.json`, in the commit that adds it, which names the rise and why. If the module would pass 900 lines, 4c splits it into `dw/server/api_models/` by router, and that commit names that rise too.
- `ui/src/lib/generated/` is never edited by hand. It is excluded from ESLint, Prettier and `ui/scripts/arch-metrics.mjs`.
- `types.ts` keeps only what is not a server response: request bodies (`WorkflowDefinition`, `PromptDefinition`), the SSE `JobEvent`, and UI-only unions. Every other export is a re-export from `generated/api-schema.ts`, under its existing name, so no importer changes.
- Out of scope:
  - `/api/jobs/{id}/events` (SSE; `JobEvent` stays hand-written);
  - the binary file routes (`/outputs`, `/inputs`, `/exports`);
  - `/api/schema` and `/api/prompt-schema`, which return JSON Schema documents typed `dict[str, Any]`.
- Commands:
  - pytest: `DW_DEVICE=cpu venv/bin/python -m pytest -q`
  - ruff: `ruff format dw dw_mcp tests scripts && ruff check dw dw_mcp tests scripts`
  - engine ratchet: `python scripts/arch_metrics.py --check docs/stabilization/baseline.json`
  - UI, from `ui/`: `npx vitest run`, `npm run check`, `npm run lint`, `npx prettier --check src e2e scripts *.ts *.js`, `npm run metrics -- --check ../docs/stabilization/ui/baseline.json`, `npm run build && DW_E2E_PYTHON=../venv/bin/python npx playwright test`.
- Branches and commits:
  - Each stage has its own branch from develop: `ui-stabilization/phase-4a` … `-4d`.
  - Commits use a conventional prefix and end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
  - Harness commits go on harnest's `main`, local until Don says push.

## Review Focus

1. A key a route emits only in some states must neither 500 in production nor appear as `null` when absent. Examples: `queue_position` only while queued, `traceback` only on failure, `info: null` with no worker. Pinned per route family, in 4a-3, 4b-1 and 4c-1, by one test per family that asserts a key's absence in the state that omits it.
2. A job row from `job_history`, written by an older server and missing the newer fields (`run_id`, `run_version`, `acknowledged`, `output_kinds`), must still list and open. Pinned in 4b-1.
3. An integer must stay an integer: a seed, a count, a `size_on_disk`, a `queue_position`. It must not come back as `3.0`, which `dw_mcp` would print and a UI `===` would miss. Pinned in 4a-3 (`size_on_disk`) and 4b-1 (`queue_position`, `created_at` stays whatever the handler emits).
4. The committed OpenAPI document must not depend on the machine or the release: no `info.version`, no hostname, no device default, no path from the environment. Otherwise the freshness test fails on another box or after a version bump. Pinned in 4a-1, which dumps twice under different `DW_DEVICE` and versions and gets the same bytes.
5. A route a test never exercises with a given key can still 500 in strict mode in e2e, and in production must stay lenient. Pinned in 4a-1 by a test that runtime mode passes an undeclared key through, and by the e2e fixture server running strict.

---

## Stage 4a: the contract machinery, and the system routes as pilot

Branch `ui-stabilization/phase-4a` from develop.

### Task 4a-1: `ApiModel`, strict mode and the OpenAPI dump

**Files:**
- Create: `dw/server/api_models.py`, `scripts/dump_openapi.py`, `tests/test_api_contract.py`
- Modify: `tests/conftest.py` (top, before any `dw` import), `ui/playwright.config.ts` (the `webServer` env), `docs/stabilization/baseline.json` (`modules` 167 → 168)
- Generated: `ui/src/lib/generated/openapi.json`

**Interfaces:**
- Produces:
  - `dw.server.api_models.ApiModel`, a `pydantic.BaseModel` whose `model_config` is `ConfigDict(extra="forbid" if STRICT else "allow")`;
  - `dw.server.api_models.STRICT: bool`, read once from `DW_STRICT_RESPONSES` at import;
  - `scripts/dump_openapi.py`'s `openapi_document() -> dict` (the strict, normalized document) and CLI `python scripts/dump_openapi.py [--check]`;
  - the constant `OPENAPI_PATH = "ui/src/lib/generated/openapi.json"`.

- [ ] **Step 1: Write the failing tests** in `tests/test_api_contract.py`:

```python
"""The UI's response contract: the routes the UI reads declare response
models, and the OpenAPI document generated from them is committed where the
UI generates its types from (ui/src/lib/generated/). See docs/ARCHITECTURE.md,
"The UI's response contract"."""

import json
import os
import subprocess
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

REPO = Path(__file__).resolve().parent.parent
DUMP = REPO / "scripts" / "dump_openapi.py"


def _model_in(mode: str):
    """ApiModel as a fresh interpreter builds it with the switch set or not."""
    env = {k: v for k, v in os.environ.items() if k != "DW_STRICT_RESPONSES"}
    if mode == "strict":
        env["DW_STRICT_RESPONSES"] = "1"
    code = "from dw.server.api_models import ApiModel; print(ApiModel.model_config['extra'])"
    return subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
    ).stdout.strip()


def test_the_suite_runs_strict():
    from dw.server.api_models import STRICT

    assert STRICT, "tests/conftest.py sets DW_STRICT_RESPONSES before dw is imported"


def test_runtime_is_lenient_and_strict_mode_forbids():
    assert _model_in("runtime") == "allow"
    assert _model_in("strict") == "forbid"


def test_a_lenient_model_passes_an_undeclared_key_through():
    # What production does with a key a model has not declared yet: send it
    from pydantic import ConfigDict

    from dw.server.api_models import ApiModel

    class Probe(ApiModel):
        model_config = ConfigDict(extra="allow")
        x: int

    app = FastAPI()

    @app.get("/p", response_model=Probe, response_model_exclude_unset=True)
    def p():
        return {"x": 1, "later": {"y": 2}}

    assert TestClient(app).get("/p").json() == {"x": 1, "later": {"y": 2}}


def _dump(extra_env):
    env = {**os.environ, **extra_env}
    return subprocess.run(
        [sys.executable, str(DUMP), "--stdout"], env=env, cwd=REPO,
        capture_output=True, text=True, check=True,
    ).stdout


def test_the_document_does_not_depend_on_the_machine():
    assert _dump({"DW_DEVICE": "cpu"}) == _dump({"DW_DEVICE": "mps"})


def test_the_document_carries_no_release_version():
    assert json.loads(_dump({}))["info"]["version"] == "0"


def test_the_committed_document_is_current():
    committed = (REPO / "ui" / "src" / "lib" / "generated" / "openapi.json").read_text()
    assert committed == _dump({}), (
        "the server's response contract changed: run `python scripts/dump_openapi.py`, "
        "then `cd ui && npm run gen:api`, and commit both"
    )
```

- [ ] **Step 2: Run them.** Run: `DW_DEVICE=cpu venv/bin/python -m pytest tests/test_api_contract.py -q -n0`. Expected: FAIL with `ModuleNotFoundError: dw.server.api_models` (and the dump script missing).

- [ ] **Step 3: Write `dw/server/api_models.py`:**

```python
"""Response models for the routes the web UI reads - the server half of the
UI's response contract. The UI's types are generated from the OpenAPI
document these produce (scripts/dump_openapi.py -> ui/src/lib/generated/),
so a change here that the UI depends on fails its type check.

Lenient at runtime: a key a model has not declared is still sent, so a
handler that grows a field never turns into a 500 for dw_mcp or a script.
Strict under DW_STRICT_RESPONSES=1 - the test suite, the e2e fixture server
and the OpenAPI dump - so an undeclared key fails a test, and the generated
types carry no index signature that would let a removed field type-check.

Routes declare `response_model_exclude_unset=True`: a key the handler did
not emit stays absent rather than arriving as null.
"""

import os

from pydantic import BaseModel, ConfigDict

STRICT = os.environ.get("DW_STRICT_RESPONSES") == "1"


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid" if STRICT else "allow")
```

- [ ] **Step 4: Write `scripts/dump_openapi.py`:**

```python
"""Write the server's OpenAPI document, strict and normalized, to
ui/src/lib/generated/openapi.json - what `npm run gen:api` turns into the UI's
response types. `--check` exits 1 when the committed copy is stale;
`--stdout` prints instead of writing.

Normalized so it is the same on every machine and release: `info.version`
is "0", and the app is built with fixed directories and no token."""

import json
import os
import sys
import tempfile
from pathlib import Path

os.environ["DW_STRICT_RESPONSES"] = "1"  # before dw is imported: no index signatures

REPO = Path(__file__).resolve().parent.parent
OPENAPI_PATH = REPO / "ui" / "src" / "lib" / "generated" / "openapi.json"


def openapi_document() -> dict:
    from dw.server.app import create_app

    with tempfile.TemporaryDirectory() as tmp:
        app = create_app(
            workflow_dir=os.path.join(tmp, "workflows"),
            output_dir=os.path.join(tmp, "outputs"),
            prompt_dir=os.path.join(tmp, "prompts"),
        )
        document = app.openapi()
    document["info"]["version"] = "0"
    return document


def render() -> str:
    return json.dumps(openapi_document(), indent=2, sort_keys=True) + "\n"


def main(argv: list[str]) -> int:
    text = render()
    if "--stdout" in argv:
        sys.stdout.write(text)
        return 0
    if "--check" in argv:
        current = OPENAPI_PATH.read_text() if OPENAPI_PATH.exists() else ""
        if current != text:
            print(f"{OPENAPI_PATH.relative_to(REPO)} is stale: run python scripts/dump_openapi.py")
            return 1
        return 0
    OPENAPI_PATH.parent.mkdir(parents=True, exist_ok=True)
    OPENAPI_PATH.write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
```

If `create_app` with these arguments starts a worker, touches `~` or reads the network, inject a stub `JobManager` (as `tests/conftest.py`'s app fixture does) and ledger the ruling. If the document still varies between machines (a `Query` default computed from the environment), find the default in the diff and fix it at the route (a literal default, resolved in the body). Each such fix is a Review Focus 4 finding.

- [ ] **Step 5: Turn strict mode on for the suite and e2e.**
  - Add as the first statements of `tests/conftest.py` (after the module docstring, if any):

    ```python
    # The UI's response contract runs strict under test: an undeclared key a
    # route emits is a 500 here, not a silent pass (dw/server/api_models.py)
    import os as _os

    _os.environ.setdefault("DW_STRICT_RESPONSES", "1")
    ```

  - In `ui/playwright.config.ts`'s `webServer`, add `env: { ...process.env, DW_STRICT_RESPONSES: '1' }`, merging with any `env` already there.

- [ ] **Step 6: Generate and run.** Run: `venv/bin/python scripts/dump_openapi.py && DW_DEVICE=cpu venv/bin/python -m pytest tests/test_api_contract.py -q -n0`. Expected: 6 passed.

- [ ] **Step 7: The whole suite and the ratchet.**
  - Run: `DW_DEVICE=cpu venv/bin/python -m pytest -q` and `python scripts/arch_metrics.py --check docs/stabilization/baseline.json`. Expected: pytest green. The ratchet reports `modules: 167 -> 168`.
  - Raise `modules` to 168 in `docs/stabilization/baseline.json`.

- [ ] **Step 8: Commit.** Message: `feat(server): ApiModel and the strict OpenAPI dump for the UI's response contract`. The body names the rise: "modules 167 -> 168: dw/server/api_models.py, the response models the UI's types are generated from".

### Task 4a-2: Generated types in the UI, checked in CI

**Files:**
- Modify: `ui/package.json` (devDependency `openapi-typescript@^7`, script `gen:api`), `ui/eslint.config.js` (ignore `src/lib/generated/**`), `ui/scripts/arch-metrics.mjs` (`sourceFiles` skips `generated/`), `ui/scripts/arch-metrics.test.ts`, `.github/workflows/ci.yml` (`ui` job)
- Create: `ui/.prettierignore`
- Generated: `ui/src/lib/generated/api-schema.ts`

**Interfaces:**
- Produces:
  - `ui/src/lib/generated/api-schema.ts`, exporting `paths` and `components` (openapi-typescript's shape);
  - `npm run gen:api`;
  - in the CI `ui` job, a step "Response contract" that runs `npm run gen:api` then `git diff --exit-code src/lib/generated`.

- [ ] **Step 1: Write the failing metrics test.** Append to `ui/scripts/arch-metrics.test.ts`:

```ts
it('does not measure generated code', () => {
  const files = sourceFiles().map((f) => f.replaceAll('\\', '/'))
  expect(files.some((f) => f.includes('/src/lib/generated/'))).toBe(false)
})
```

  For the test to bite, a `.ts` file must exist under `src/lib/generated/` first, so do Step 3 before running it.

- [ ] **Step 2: Add the dependency and script.**
  - Run `cd ui && npm install -D openapi-typescript@^7`.
  - Add the script `"gen:api": "openapi-typescript src/lib/generated/openapi.json -o src/lib/generated/api-schema.ts"`.

- [ ] **Step 3: Generate.** Run `npm run gen:api`. Expected: `src/lib/generated/api-schema.ts` exists.

- [ ] **Step 4: Run the metrics test.** Run: `npx vitest run scripts/arch-metrics.test.ts`. Expected: FAIL. `sourceFiles` lists `api-schema.ts`.

- [ ] **Step 5: Exclude generated code.**
  - In `sourceFiles`, skip a directory named `generated`:

    ```js
    if (statSync(path).isDirectory()) {
      // Generated from the server's OpenAPI document - measured on the server side
      if (name !== 'generated') found.push(...sourceFiles(path))
    }
    ```

  - Add `{ ignores: ['src/lib/generated/**', 'dist/**'] }` as the first element of the ESLint config, keeping any `ignores` already present.
  - Create `ui/.prettierignore` containing `src/lib/generated/`.

- [ ] **Step 6: Run.** Run: `npx vitest run scripts/arch-metrics.test.ts && npm run lint && npx prettier --check src e2e scripts *.ts *.js && npm run metrics -- --check ../docs/stabilization/ui/baseline.json && npm run check`. Expected: all pass, and metrics unchanged.

- [ ] **Step 7: CI.** In `.github/workflows/ci.yml`'s `ui` job, after `npm ci`, add:

```yaml
      # The UI's response types are generated from the server's OpenAPI
      # document (scripts/dump_openapi.py writes it; the backend job's
      # tests/test_api_contract.py fails when it is stale). A stale
      # api-schema.ts fails here; a field the UI reads that the server no
      # longer sends fails the type check below
      - name: Response contract
        run: npm run gen:api && git diff --exit-code src/lib/generated
```

- [ ] **Step 8: Commit.** Message: `build(ui): generate response types from the server's OpenAPI document`.

### Task 4a-3: The system routes declare their responses

The pilot sets the recipe that 4b and 4c repeat for every route family.

**Routes:**

| Route | Model |
| --- | --- |
| `GET /api/health` | `HealthInfo` |
| `GET /api/server` | `ServerInfo`, with `ServerAddress`, `McpMount`, `ServerDirectories` |
| `GET /api/memory` | `MemoryInfo`, with `MemoryDetail` |
| `POST /api/memory/clear` | `MemoryCleared` |
| `GET /api/models` | `ModelCache`, with `ModelRepo`, `ModelRevision` |
| `POST /api/models/download`, `POST /api/models/downloads/{id}/cancel` | `ModelDownload` |
| `GET /api/models/downloads` | `ModelDownloads` |
| `DELETE /api/models` | `ModelDeleted` |
| `GET /api/system/diffusers`, `POST /api/system/diffusers/update` | `DiffusersStatus` |

**Files:** Modify `dw/server/api_models.py`, `dw/server/routes/system.py`, `ui/src/lib/types.ts`, `tests/test_api_contract.py`; regenerate both generated files.

**Interfaces:**
- Produces the models above in `dw.server.api_models`. In `types.ts`, `HealthInfo`, `ServerAddress`, `ServerInfo`, `MemoryInfo`, `ModelRevision`, `ModelRepo`, `ModelCache`, `ModelDownload` and `DiffusersStatus` become re-exports of `components['schemas'][...]` under the same names.
- Produces `UI_READ_ROUTES` in `tests/test_api_contract.py`, a list of `(method, path)` that grows in 4b and 4c.

- [ ] **Step 1: Write the failing tests.** Add to `tests/test_api_contract.py`:

```python
UI_READ_ROUTES = [
    ("get", "/api/health"),
    ("get", "/api/server"),
    ("get", "/api/memory"),
    ("post", "/api/memory/clear"),
    ("get", "/api/models"),
    ("post", "/api/models/download"),
    ("get", "/api/models/downloads"),
    ("post", "/api/models/downloads/{download_id}/cancel"),
    ("delete", "/api/models"),
    ("get", "/api/system/diffusers"),
    ("post", "/api/system/diffusers/update"),
]


def _success_schema(document, method, path):
    responses = document["paths"][path][method]["responses"]
    ok = next(code for code in responses if code.startswith("2"))
    return responses[ok].get("content", {}).get("application/json", {}).get("schema", {})


import pytest  # noqa: E402  (beside the routes it parametrizes over)


@pytest.mark.parametrize("method,path", UI_READ_ROUTES)
def test_a_route_the_ui_reads_declares_its_response(method, path):
    document = json.loads(_dump({}))
    schema = _success_schema(document, method, path)
    assert "$ref" in schema or schema.get("type") == "array", (
        f"{method.upper()} {path} declares no response model: the UI's type for it "
        "is not generated from the server"
    )
```

  Add one state-dependent-key test for the family (Review Focus 1 and 3), using the suite's existing app/client fixture for system routes. Find it with `grep -n "def client\|def app" tests/conftest.py tests/test_server*.py`.
  - With no worker resident, `GET /api/memory` answers `info: null` and `live: false`, exactly the keys it answered before.
  - `GET /api/models` on a fixture cache returns `size_on_disk` as a JSON integer (`isinstance(body["size_on_disk"], int)`).

- [ ] **Step 2: Run.** Run: `DW_DEVICE=cpu venv/bin/python -m pytest tests/test_api_contract.py -q -n0`. Expected: the 11 parametrized cases FAIL ("declares no response model"). The state tests pass, since today's payload is right. They are the pin that it stays so.

- [ ] **Step 3: Write the models.**
  - Start from `types.ts`'s interfaces, which are what the UI believes today. Copy each into `api_models.py` as a class following the field rules in Global Constraints, and move each doc comment onto its field as `Field(description=...)`.
  - TS literal unions become `Literal[...]`. For example, `ModelDownload.status: Literal["downloading", "completed", "cancelled", "failed"]`.
  - Then read each handler (and the builder it calls: `manager.memory_status()`, `download_manager`, `diffusers_updater`, `model_cache` listing) for keys `types.ts` never declared, and declare them. The strict suite reports any you miss as a 500 naming the key.
  - `HealthInfo`, written out as the pattern:

```python
class HealthInfo(ApiModel):
    status: str
    version: str
    worker_alive: bool = Field(
        description="An on-demand worker: false on an idle server that has not run a job "
        "yet, or after a memory clear, is normal - no model process is resident."
    )
    current_job: str | None
    queued: int
    hostname: str = Field(description="Which machine answered.")
    device: str
    mcp: bool
```

  Here `types.ts` says `version?`, `hostname?`, `device?`, `mcp?`, `queued?` (optional), but the handler always emits them. A field is required when the handler always emits it, so the generated type tightens. Any UI code that guarded against their absence is now dead. Leave it, and note it for the triage in 4d.

- [ ] **Step 4: Declare them on the routes.** For example:

```python
@router.get("/api/health", response_model=HealthInfo, response_model_exclude_unset=True)
```

  `/api/models/downloads` returns `{"downloads": [...]}`, so it gets a wrapper model, `class ModelDownloads(ApiModel): downloads: list[ModelDownload]`.

- [ ] **Step 5: Regenerate and re-export.**
  - Run `venv/bin/python scripts/dump_openapi.py && (cd ui && npm run gen:api)`.
  - In `types.ts`, delete the nine interfaces and add:

```ts
import type { components } from './generated/api-schema'

type Schemas = components['schemas']

// Server responses: generated from the server's response models
// (dw/server/api_models.py), so a field the server stops sending fails the
// type check rather than reading undefined at runtime
export type HealthInfo = Schemas['HealthInfo']
export type ServerAddress = Schemas['ServerAddress']
export type ServerInfo = Schemas['ServerInfo']
export type MemoryInfo = Schemas['MemoryInfo']
export type ModelRevision = Schemas['ModelRevision']
export type ModelRepo = Schemas['ModelRepo']
export type ModelCache = Schemas['ModelCache']
export type ModelDownload = Schemas['ModelDownload']
export type DiffusersStatus = Schemas['DiffusersStatus']
```

  If FastAPI names a schema `X-Output` (a model used as both request and response), use that key and ledger it.

- [ ] **Step 6: Run everything.** Run pytest (whole suite, strict), ruff, the engine ratchet, then from `ui/`: `npm run check`, `npx vitest run`, `npm run lint`, prettier, metrics, build, e2e.
  - Expected: all green.
  - `npm run check` errors are the contract working. Each is a place where `types.ts` and the server disagreed. Fix each at the right end, and ledger it:
    - Where the server is right, change the UI code.
    - Where the model is wrong, fix the model, and the strict tests confirm it.
  - Test fixtures and mocks that build these objects (`vi.mock('../api')` return values) must now satisfy the stricter types. Add the missing required fields to the fixtures; do not loosen the types.

- [ ] **Step 7: Commit.** Message: `feat(server): the system routes declare their responses; the UI's system types are generated`.

### Stage 4a close

Merge `ui-stabilization/phase-4a` into develop only on Don's OK. Deploy to lem the same way (`ssh lem '~/diffusers-workflow/scripts/deploy.sh develop'`). Smoke the status popover, the Models page (cache, a download's progress, delete confirm), and `/api/health` over MCP (`get_health`), with `dw_mcp` output unchanged.

---

## Stage 4b: job and validation routes

Branch `ui-stabilization/phase-4b`. Every task follows 4a-3's steps 1-7 with its own routes:
- append the routes to `UI_READ_ROUTES`;
- add the family's state test or tests;
- write the models from `types.ts` and the handlers;
- declare them on the routes with `exclude_unset`;
- regenerate, re-export, and run everything.

### Task 4b-1: Jobs

**Routes:**

| Route | Model |
| --- | --- |
| `GET /api/jobs` | `JobList` (`jobs: list[JobSummary]`, `total: int \| None`) |
| `GET /api/jobs/{job_id}` | `JobDetail` |
| `POST /api/jobs` | `JobDetail` |
| `POST /api/jobs/{job_id}/rerun` | `JobDetail` |
| `POST /api/enhance` | `JobDetail` |
| `POST /api/jobs/{job_id}/move` | `JobMoved` (`id`, `queue: list[str]`) |
| `POST /api/jobs/{job_id}/cancel` | `JobCancelled` |
| `POST /api/jobs/{job_id}/export` | `JobExport` |
| `GET /api/jobs/{job_id}/workflow` | `JobWorkflow` |
| `DELETE /api/jobs/{job_id}/run` | `RunDeleted` |

**Files:** `dw/server/api_models.py`, `dw/server/routes/jobs.py`, `dw/server/routes/library.py` (`/api/enhance`), `ui/src/lib/types.ts` (`JobSummary`, `ManifestEntry`, `JobDetail`, `AcknowledgedCost`, `StepEndEvent` stays), `ui/src/lib/api.ts` (inline response types become named imports), `tests/test_api_contract.py`.

**Notes:**
- `AcknowledgedCost` already exists as a request model in `dw/server/admission.py`. The response reuses that class; do not write a second one. If it is used as both input and output, the schema may be split as `AcknowledgedCost-Input`/`-Output`. Re-export whichever the response uses.
- `JobDetail` extends `JobSummary`, so in Python `class JobDetail(JobSummary)`.
- `status` is `Literal["queued", "running", "succeeded", "failed", "cancelled"]`.

**State tests:**
- Review Focus 1: a queued job's JSON has `queue_position` and no `traceback` key.
- Review Focus 2: a `job_history` row written without `run_id`, `run_version`, `acknowledged` or `output_kinds` lists (`GET /api/jobs`) and opens (`GET /api/jobs/{id}`) with those keys absent. Use `tests/test_server_jobs.py`'s history fixture; find it with `grep -n "job_history\|historical" tests/test_server_jobs.py`.
- Review Focus 3: `queue_position` is an `int` in the JSON.

### Task 4b-2: Validation

**Routes:** `POST /api/validate` → `ValidationResult`, with `Plan` and its nested estimate and finding models.

**Files:**
- `dw/server/api_models.py`, `dw/server/routes/jobs.py`.
- `ui/src/lib/types.ts`: `ValidationResult`, `Plan`, and the finding shape.
- `tests/test_api_contract.py`.

**Notes:**
- `Plan` is built by `dw/plan.py`. Read its builder for every key, including the estimate's `minutes`, `vram`, `downloads` and `warnings`.
- A finding is `dw/assessment_rules.finding()`'s shape. Declare it once and reuse it in 4c's gallery metadata (`findings`).

**State test:** an invalid workflow answers `valid: false` with its errors and no `plan` key, exactly as before.

### Stage 4b close

Merge and deploy on Don's OK, then smoke on lem:
- the Jobs page (a queued job and a historical one);
- a job page;
- validate in the editor (a valid and an invalid workflow);
- `get_job` and `validate_workflow` over MCP, with output unchanged.

---

## Stage 4c: library, gallery, assets and introspection; the coverage pin

Branch `ui-stabilization/phase-4c`. Tasks 4c-1 to 4c-3 follow 4a-3's steps.

### Task 4c-1: Library

**Routes:**

| Route | Model |
| --- | --- |
| `GET /api/workspaces` | `WorkspaceList` |
| `POST /api/workspaces` | `WorkspaceCreated` |
| `DELETE /api/workspaces/{name}` | `WorkspaceDeleted` |
| `GET /api/workflows` | `WorkflowList`, with `LibraryRoot` and the shape/trait/cost entries |
| `GET /api/workflows/{name}` | `WorkflowWithOrigin` |
| `PUT` and `PATCH /api/workflows/{name}` | `WorkflowSaved` |
| `DELETE /api/workflows/{name}` | `Deleted` (`name`, `deleted`) |
| `GET /api/prompts` | `PromptList` |
| `GET /api/prompts/{name}` | `StoredPrompt` |
| `PUT /api/prompts/{name}` | `PromptSaved` |
| `DELETE /api/prompts/{name}` | `Deleted` |
| `GET /api/enhancers` | `EnhancerPresets` (`presets: list[EnhancerPreset]`) |

**Notes:**
- A workflow or prompt *definition* inside a response is `dict[str, Any]`. It is open JSON by design (`types.ts`'s `WorkflowDefinition` stays hand-written, and `ui/eslint.config.js` allows `any` for the same reason).
- `GET /api/workflows/{name}` serves the raw file verbatim, per the seam map's "raw workflow GET" row. Its model wraps only the envelope keys. A test asserts the definition comes back byte-identical to the file.

**State test:** a workspace list on a server with only `default` keeps today's exact keys.

### Task 4c-2: Gallery, assets and media

**Routes:**

| Route | Model |
| --- | --- |
| `GET /api/gallery` | `GalleryList` (`files: list[GalleryFile]`) |
| `DELETE /api/gallery/{name}` | `Deleted` |
| `POST /api/gallery/archive` | `Archived` |
| `GET /api/gallery/{name}/metadata` | `GalleryMetadata`, with 4b-2's finding model for `findings` |
| `GET /api/assets` | `AssetLibrary`, with `AssetFile`, `ShadowedAsset`, `ShadowedEntry` |
| `POST /api/uploads` | `Uploaded` |
| `POST /api/assets/keep` | `Kept` |
| `POST /api/assets/archive` | `Archived` |
| `DELETE /api/assets/{name}` | `AssetDeleted` (`name`, `deleted`, `origin`) |

The other `media.py` routes (`/thumbnail`, `/image`, `/frames`, `/audio`) return media or MCP-shaped JSON. Convert the JSON ones only if `api.ts` calls them. The coverage pin in 4c-4 decides.

**State test:** the metadata of an image with no embedded workflow has no `workflow` key, and its `findings` is `[]`.

### Task 4c-3: Introspection

**Routes:**

| Route | Model |
| --- | --- |
| `GET /api/pipelines` | `PipelineNames` |
| `GET /api/pipelines/{name}`, `GET /api/tasks/{command}`, `GET /api/classes/{name}` | `PipelineDescription`, with `PipelineParameter` |
| `GET /api/tasks` | `TaskList` |
| `GET /api/classes` | `ClassList` (`kind`, `classes`) |

`PipelineParameter.default` is `Any`: a parameter's default is whatever the signature holds.

**State test:** a parameter with no default has no `default` key (not `null`), since the UI's argument widgets tell the two apart.

### Task 4c-4: Every route `api.ts` calls is in the contract

**Files:** Modify `tests/test_ui_twins.py` and `tests/test_api_contract.py`.

- [ ] **Step 1: Write the failing test.** Add to `tests/test_ui_twins.py`:

```python
API_TS = UI_LIB / "api.ts"
# Paths api.ts reaches that are not JSON responses the contract covers: the
# event stream, JSON Schema documents served as-is, and file/media URLs
NOT_IN_CONTRACT = {
    "/api/jobs/{}/events",
    "/api/schema",
    "/api/prompt-schema",
}


def _api_ts_paths():
    """Every /api/ path api.ts builds, with each ${...} as {}."""
    text = API_TS.read_text()
    found = set()
    for literal in re.findall(r"[`'\"](/api/[^`'\"?]*)", text):
        found.add(re.sub(r"\$\{[^}]*\}", "{}", literal).rstrip("/"))
    return found


def test_every_json_route_the_ui_calls_declares_its_response():
    from tests.test_api_contract import UI_READ_ROUTES

    covered = {re.sub(r"\{[^}]*\}", "{}", path) for _, path in UI_READ_ROUTES}
    missing = sorted(p for p in _api_ts_paths() - NOT_IN_CONTRACT if p not in covered)
    assert missing == [], f"api.ts calls routes with no declared response model: {missing}"
```

  If `tests` is not importable as a package, move `UI_READ_ROUTES` into `tests/api_contract_routes.py` and import it from both tests. Ledger that.

- [ ] **Step 2: Run.** Run: `DW_DEVICE=cpu venv/bin/python -m pytest tests/test_ui_twins.py -q -n0`. Expected: PASS if 4c-1 to 4c-3 covered everything. Otherwise it FAILS naming the paths. Each named path is a route to convert by 4a-3's recipe in this task, or one to add to `NOT_IN_CONTRACT` with a reason in the comment, if it is not a JSON response.
  - To see the pin bite, temporarily delete one entry from `UI_READ_ROUTES`, watch the test fail naming it, and restore it.

- [ ] **Step 3: The size check.** If `dw/server/api_models.py` is over 900 lines, split it into the package `dw/server/api_models/` (`__init__.py` re-exporting, plus `system.py`, `jobs.py`, `library.py`, `media.py`). Raise `modules` by the count added, in that commit, naming why.

- [ ] **Step 4: Commit.** Message: `test: every JSON route api.ts calls declares its response model`.

### Stage 4c close

Merge and deploy on Don's OK, then smoke on lem:
- Workflows, Prompts, Gallery, Assets, the editor's pipeline suggestions and a step's parameter list;
- `list_workflows`, `get_prompt`, `list_assets` and `get_gallery_metadata` over MCP, with output unchanged.

---

## Stage 4d: guardrails and the gate

Branch `ui-stabilization/phase-4d`.

### Task 4d-1: e2e before develop moves

**Files:** Modify `.github/workflows/ci.yml` (`e2e` job's `if:`).

- [ ] Change the `e2e` job's condition to:

```yaml
    # Playwright against a real dw.serve before develop moves - on PRs into
    # develop or master, and on pushes to develop, since the agent loop
    # pushes develop directly with no PR
    if: >-
      (github.event_name == 'pull_request' &&
       (github.base_ref == 'develop' || github.base_ref == 'master')) ||
      (github.event_name == 'push' && github.ref == 'refs/heads/develop')
```

  The roadmap names only PRs into develop. A push to develop is added because the harness's loop never opens one, so PR-only would not cover most of what reaches develop. Ledger that as a ruling, with its cost: CI minutes for each push to develop.

- [ ] Verify the expression by pushing the branch and opening a draft PR into develop, which shows the `e2e` job queued. Close the draft afterwards. Commit `ci: run e2e before develop moves`.

### Task 4d-2: The UI metrics script speaks the harness's language

**Files:** Modify `ui/scripts/arch-metrics.mjs` (`regressions` line format; `--compare`), `ui/scripts/arch-metrics.test.ts`.

**Interfaces:**
- Produces: `regressions(current, baseline) -> string[]`, with lines in the form `key: before -> after`. This is the engine script's format, which harnest's `waiver_problems` parses.
- Produces: `node scripts/arch-metrics.mjs --compare CURRENT.json BASELINE.json`, which prints one regression line per metric that rose, exits 1 when any rose and exits 0 otherwise. It does not measure.

- [ ] **Step 1: Write the failing tests:**

```ts
it('reports a rise as the engine ratchet does', () => {
  expect(regressions({ a: 3, b: 1 }, { a: 2, b: 1 })).toEqual(['a: 2 -> 3'])
})

it('compares two measurements without measuring', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'uimetrics-'))
  writeFileSync(join(dir, 'cur.json'), JSON.stringify({ a: 3, b: 1 }))
  writeFileSync(join(dir, 'base.json'), JSON.stringify({ a: 2, b: 1, c: 0 }))
  const run = spawnSync(
    'node',
    ['scripts/arch-metrics.mjs', '--compare', join(dir, 'cur.json'), join(dir, 'base.json')],
    { encoding: 'utf8' },
  )
  expect(run.status).toBe(1)
  expect(run.stdout.trim()).toBe('a: 2 -> 3')
})
```

  Use imports `mkdtempSync`/`writeFileSync` from `node:fs`, `tmpdir` from `node:os`, `join` from `node:path` and `spawnSync` from `node:child_process`.

- [ ] **Step 2: Run.** Run: `npx vitest run scripts/arch-metrics.test.ts`. Expected: FAIL (old format `a: 3 > baseline 2`; `--compare` unknown).

- [ ] **Step 3: Implement.**
  - `regressions` maps to `` `${key}: ${baseline[key]} -> ${value}` ``.
  - `main` handles `--compare` before measuring:

```js
async function main(argv) {
  if (argv[0] === '--compare') {
    const [current, baseline] = argv.slice(1, 3).map((p) => JSON.parse(readFileSync(p, 'utf8')))
    const problems = regressions(current, baseline)
    for (const line of problems) console.log(line)
    return problems.length ? 1 : 0
  }
  const { metrics, detail } = await measure()
  // ...unchanged
```

- [ ] **Step 4: Run.** Run: `npx vitest run scripts/arch-metrics.test.ts && npm run metrics -- --check ../docs/stabilization/ui/baseline.json`. Expected: PASS. Commit `build(ui): the metrics script reports rises in the engine ratchet's format, and compares without measuring`.

### Task 4d-3: The harness ratchets the UI (harnest)

Work in `~/testing/harnest`, on `main`, committed locally. Pushing waits for Don.

**Files:** Modify `lib/arch_ratchet.py`, `agent-settings/hooks/guard.py` (`arch_gate`, `merge_gated`), `tests/test-ratchet.sh`, `run-loop.sh` (`ensure_arch_tools`), `README.md` and `HARNESS-ROADMAP.md` (one line each).

**Interfaces:**
- Produces:
  - in `lib/arch_ratchet.py`: `UI_SCRIPT = "ui/scripts/arch-metrics.mjs"`, `UI_BASELINE = "docs/stabilization/ui/baseline.json"`;
  - `check_ui(checkout, src="HEAD") -> list[str]` (lines `key: before -> after`, `[]` when off or no rise; raises `ToolError` when it cannot say);
  - `ui_active(checkout, fetch=True) -> bool` (develop carries `UI_SCRIPT`);
  - `waiver_problems(checkout, worse, src="HEAD", baseline=BASELINE)`, which gains the `baseline` parameter;
  - `refusal(worse, problems=None, baseline=BASELINE, what="dw's architecture metrics")`.
- `guard.arch_gate` runs both ratchets. Each one's waiver is judged against its own baseline file. The UI baseline file joins the engine baseline file as the hot-zone exception in an approved rise.

**How `check_ui` measures**, mirroring `check`:
- The two trees are the merge base and src, extracted with `_archive`.
- In each extracted tree:
  - symlink `ui/node_modules` to `<checkout top>/ui/node_modules`;
  - write develop's `UI_SCRIPT` over `ui/scripts/arch-metrics.mjs`;
  - run `node ui/scripts/arch-metrics.mjs` with `cwd` = the tree's `ui/`;
  - parse the first JSON value and drop `detail`.
- Compare with develop's script: `node <develop script> --compare cur.json base.json`.
  - Output that starts with `{` means the script predates `--compare`: raise `ToolError` (fail closed).
- Cache key: the tree, the script's sha256, `node --version`, and `ui/node_modules/eslint/package.json`'s version.
- Readiness: `node` on PATH and `<top>/ui/node_modules/eslint` present. Otherwise `ToolError("... run `npm ci` in ui/ ...")`. `ensure_arch_tools` in `run-loop.sh` runs `npm ci --prefix ui` when it is missing.

- [ ] **Step 1: Write the failing tests** in `tests/test-ratchet.sh`. They use a fixture repo whose develop carries a dependency-free stand-in `ui/scripts/arch-metrics.mjs`. The stand-in counts `// complex` markers under `ui/src`, implements `--compare` with the `key: before -> after` format, and imports nothing. The fixture has a dummy `ui/node_modules/eslint/package.json` with `{"version":"0.0.0"}`. Cases:
  - a branch adding one `// complex` marker under `ui/src` is refused at hand-off, with a message naming `complex_functions: 1 -> 2` and the UI baseline;
  - the same branch on issue #7 (`arch-approved`), whose commits raise `docs/stabilization/ui/baseline.json` by exactly that, in a commit naming it with a body, passes;
  - #7 with the baseline raised by 2 is refused, and the message says it should raise it to 2;
  - a branch touching only `dw/` measures no UI rise (the UI trees are equal, so it returns early);
  - a checkout with no `ui/node_modules/eslint` fails closed, naming `npm ci`;
  - a develop whose UI script lacks `--compare` (it prints JSON) fails closed;
  - with `ui/scripts/arch-metrics.mjs` absent on develop, the UI ratchet is off and the engine ratchet still runs.

- [ ] **Step 2: Run.** Run: `bash tests/test-ratchet.sh`. Expected: the new cases FAIL and the existing cases pass.

- [ ] **Step 3: Implement** `check_ui`, `ui_active`, the `baseline` parameters and the guard changes, as described above.
  - In `arch_gate`, after the engine check, add `worse_ui = arch_ratchet.check_ui(checkout, src)`, under the same `ToolError` → `deny` handling.
  - `problems_ui = arch_ratchet.waiver_problems(checkout, worse_ui, src, arch_ratchet.UI_BASELINE) if worse_ui and is_approved() else None`.
  - The hot-zone filter exempts `UI_BASELINE` when `problems_ui is not None`.
  - The deny message for a UI rise is `"dw's UI architecture ratchet: " + arch_ratchet.refusal(worse_ui, problems_ui, arch_ratchet.UI_BASELINE, "the UI's architecture metrics")`.
  - `merge_gated` also counts `ui_active`.

- [ ] **Step 4: Run.** Run: `bash tests/test-ratchet.sh && bash tests/test-guard.sh`, then the harness's full test list (`ls tests/test-*.sh`). Expected: all pass.

- [ ] **Step 5: A live dry run on dw.** In a scratch worktree of dw cut from develop:
  - add a function over the complexity limit to `ui/src/lib/format.ts`;
  - commit;
  - run `python3 ~/testing/harnest/lib/arch_ratchet.py check <worktree>` and confirm it fails closed or reports `complex_functions: 12 -> 13`. Add a `check-ui` subcommand to the CLI for this.
  - Delete the worktree.
  - Record the output for the gate report.

- [ ] **Step 6: Commit** in harnest: `ratchet: the UI's architecture metrics, same rule and waiver as the engine's`.

### Task 4d-4: `ui/CLAUDE.md`'s guardrail rules become checks

**Files:**
- Create `ui/src/lib/designRules.test.ts`.
- Modify `ui/CLAUDE.md` and `docs/ARCHITECTURE.md` (seam-map rows).

**Rules, line by line:**

| `ui/CLAUDE.md` rule | Becomes |
| --- | --- |
| "The UI reads engine-derived fields … never computes or orders them" | The response contract plus the seam-map row "The UI's response contract". The line is replaced by that row. |
| "Show the proof" | Judgment; stays. |
| "Every token pair passes WCAG AA in both themes" | Stays as guidance: a nice-to-have, not a guardrail (Don, 2026-10-02). No check. For the record, at develop `36c4628e` the light theme's `--live`/`--warn` on `--panel-2` measure 4.31. |
| "`--live` appears in these places and nowhere else" | A test: the files that use `var(--live)` are exactly an allowlist. The places themselves stay listed in that test's comment. |

- [ ] **Step 1: Write the test** in `designRules.test.ts`, which runs in Vitest's default node environment, reading files:

```ts
// Where the --live state colour may appear, held mechanically (ui/CLAUDE.md;
// app.css's header says why --live means machine state and nothing else)
import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { expect, it } from 'vitest'
import { sourceFiles } from '../../scripts/arch-metrics.mjs'

const UI = join(__dirname, '..', '..')

// The places --live may appear: the header's running link and VRAM
// pressure, a running job's row and status chip, the job page's progress
// bar and current step, the flow view's active step, a model download in
// flight, and the focus ring
const LIVE_FILES = [
  'src/App.svelte',
  'src/app.css',
  'src/lib/editor/FlowView.svelte',
  'src/lib/job/JobProgress.svelte',
  'src/lib/pages/JobsPage.svelte',
  'src/lib/pages/ModelsPage.svelte',
  'src/lib/pages/OverviewPage.svelte',
]

it('uses --live only where machine state is shown', () => {
  const using = sourceFiles()
    .filter((f) => readFileSync(f, 'utf8').includes('var(--live)'))
    .map((f) => f.slice(UI.length + 1).replaceAll('\\', '/'))
  expect(using).toEqual(LIVE_FILES)
})
```

  If the import of the `.mjs` needs a declaration, add `scripts/arch-metrics.d.mts` with `export function sourceFiles(root?: string): string[]`.

- [ ] **Step 2: Run.** Run: `npx vitest run src/lib/designRules.test.ts`. Expected: PASS, since it pins today's 7 files. The test is a pin, not a RED. Show that it bites by adding `var(--live)` to `GalleryPage.svelte` and watching it fail, then reverting.

- [ ] **Step 3: Rewrite `ui/CLAUDE.md`.**
  - Keep the build/check pointer, "Show the proof", the AA line as guidance, and one line for `--live` naming `src/lib/designRules.test.ts` as its holder.
  - Replace the engine-derived-fields line with a pointer to the seam-map row.
  - The file must end shorter than its 12 lines. The engine's `claude_md_lines` ratchet counts it, so lower `docs/stabilization/baseline.json`'s `claude_md_lines` in the same commit.
  - Add seam-map rows to `docs/ARCHITECTURE.md`:
    - "The UI's response contract": owner `dw/server/api_models.py`; rule: the payload does not change, strict under test; enforced by `tests/test_api_contract.py`, `tests/test_ui_twins.py::test_every_json_route_the_ui_calls_declares_its_response` and CI's "Response contract" step;
    - "Where `--live` appears": owner `ui/src/app.css`; enforced by `ui/src/lib/designRules.test.ts`.

- [ ] **Step 4: Run** vitest, check, lint, prettier, UI metrics, pytest, and the engine ratchet. Expected: all green, with `claude_md_lines` lowered. Commit `test(ui): where --live appears is a check`.

### Task 4d-5: The gate

- [ ] **Demonstrate the contract.** In a scratch worktree from develop, rename `HealthInfo.worker_alive` to `worker_up` in `api_models.py` and in the route. Record each result:
  - `pytest tests/test_api_contract.py`: it fails on the stale document.
  - `python scripts/dump_openapi.py && cd ui && npm run gen:api && npm run check`: it fails where `App.svelte`/`StatusPopover.svelte` read `worker_alive`.
  - Delete the worktree.
- [ ] **Demonstrate the harness.** Use 4d-3 Step 5's output.
- [ ] **Run everything.** Run `scripts/preflight.sh`. Run unit pytest with `DW_DEVICE=cpu` and integration on MPS, as at gate 3.
- [ ] **Final review.** A fresh reviewer on the most capable model reviews the 4d branch, plus harnest's commits since `b7e7e47`. Fix Critical and Important findings test-first, and ledger the minors.
- [ ] **On Don's OK:**
  - merge, push, and deploy to lem;
  - smoke the status popover, Models, Jobs, a job page, the editor (validate), Gallery, Assets and the prompt editor's enhancer, plus `get_health`, `get_job` and `list_workflows` over MCP, with output unchanged;
  - push harnest.
- [ ] **Write the gate report** into `ROADMAP.md`. It includes:
  - the criteria and evidence;
  - suite results;
  - the ratchet table with a Gate 4 column;
  - the two demonstrations;
  - the route count under contract;
  - deferred minors.
  - Mark the Phase 4 row done.
  - On Don's word, tag `ui-stabilization-gate-4` and push it.
