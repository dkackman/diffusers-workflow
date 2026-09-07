# Guides and Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The engine serves its own guides over `GET /api/guides`, the MCP `list_guides`/`get_guide` tools proxy them; `/api/validate` and `Workflow.validate()` report every schema error at once; `WORKFLOW_GUIDE.md` gains a section written for an agent composing a workflow.

**Architecture:** A pure module `dw/server/guides.py` takes over the `GUIDES` table and section extraction from `dw_mcp/guides.py`, which shrinks to two proxy calls like every other MCP handler. `dw/schema.py` gains `validate_data_all` over `iter_errors`; `Workflow.validate()` and the validate route join its output. The docs work is prose only. No engine behaviour changes beyond the error message.

**Tech Stack:** Python 3.10+, FastAPI, jsonschema 4.x, httpx MockTransport for MCP tests, pytest. Tests run with `python -m pytest tests/<file> -v` from the repo root with the venv active (`source ./activate`).

**Spec:** `docs/superpowers/specs/2026-09-06-agent-catalog-legibility-design.md` (plan 2, sections 2.1–2.3). Read it first. Plan 1 (catalog metadata) is merged on master as of 2026-09-07.

**Agent assignment:** each task is tagged `[model: haiku|sonnet|opus]`. Haiku for mechanical edits with exact code given, sonnet for integration work with judgement about existing code, opus for prose that has to be right for a reader who cannot ask. The orchestrator passes the tag as the `model` parameter when dispatching.

**Ledger:** `docs/proposals/agent-catalog-legibility.md` ends with a `## Ledger` table. The last step of every task updates the row(s) it lands, changing `designed` to `done (task N)` and adding anything learned to the notes column. Keep the row on one line.

## Rulings against the spec text

Two details in spec §2.1/§2.2 conflict with contracts that already exist. The rulings, so no implementer re-litigates them:

1. **Guide payloads keep the key `content`, not `text`.** The MCP tools return `{name, section, content}` today and spec §2.1 says they "pass through unchanged"; the routes therefore return exactly what the MCP handlers returned, `{"guides": [...]}` for the listing and `{name, section, content}` for one guide. An agent's contract does not change.
2. **`/api/validate` keeps `error` and adds `errors`.** The spec calls the joined field `message`; the route, the UI's `ValidationResult` type and the MCP tests all read `error`. `error` stays the joined string (old clients unchanged), `errors` is the new list. Nothing named `message` is added.

## Global Constraints

- Never `eval`/`exec`/`shell=True`; all paths through `dw/security.py`. Guide names are looked up in the closed `GUIDES` dict, never joined into a path from the request.
- `dw_mcp/` must not import anything under `dw.*` (`tests/test_mcp_server.py` guards the torch boundary). After this plan `dw_mcp/guides.py` imports only `dw_mcp.client`.
- `dw/server/guides.py` imports only the standard library.
- The no-param `/api/validate` response is today's response plus one field (`errors`) — nothing removed or renamed.
- `Workflow.validate()`'s message carries the text `Validation error` exactly once (`tests/test_workflow.py`, `tests/test_cli.py` count it); `dw/validate.py` prints the message verbatim when it starts with `Validation error`.
- The multi-error cap is 25 (`MAX_VALIDATION_ERRORS`).
- No model-family names in Python (spec "Principle carried from Part 3"). The authoring section names prefixes and tasks, not checkpoints.
- Docs that mention an example workflow path must point at a file that exists (`tests/test_docs_links.py` runs over every `.md`).
- Commit messages end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

---

## File map

| file | responsibility |
|---|---|
| `dw/server/guides.py` (new) | `GUIDES`, `GuideError`, `read_guide`, `list_guides`, `get_guide`, section extraction |
| `dw/server/app.py` | `GET /api/guides`, `GET /api/guides/{name}`; `/api/validate` returns `errors` |
| `dw_mcp/guides.py` | two proxy functions `(client, ...)` |
| `dw_mcp/server.py` | passes `client` to the guide handlers |
| `dw/schema.py` | `validate_data_all`, `format_validation_errors`, `MAX_VALIDATION_ERRORS` |
| `dw/workflow.py` | `validate()` raises the joined message |
| `scripts/build_dist.sh`, `MANIFEST.in` | `dw/docs/` is a `dw.server` build product and ships in the wheel |
| `docs/WORKFLOW_GUIDE.md` | `## Authoring a workflow from an agent` |
| `CLAUDE.md`, `dw_mcp/CLAUDE.md`, `dw/server/CLAUDE.md`, `docs/MCP.md`, `docs/SERVER.md` | pointers and route docs |
| `tests/test_server_guides.py` (new) | module and route tests |
| `tests/test_mcp_guides.py` | rewritten as proxy tests |
| `tests/test_schema.py`, `tests/test_workflow.py`, `tests/test_server.py`, `tests/test_mcp_authoring.py` | multi-error tests |

---

### Task 1: `dw/server/guides.py` — the guides module on the server `[model: sonnet]`

**Files:**
- Create: `dw/server/guides.py`
- Create: `tests/test_server_guides.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `GUIDES: dict[str, tuple[str, str]]`; `class GuideError(LookupError)`; `read_guide(name) -> str`; `list_guides() -> {"guides": [{name, file, summary, sections}]}`; `get_guide(name, section=None) -> {name, section, content}`. Task 2 wraps these in routes; Task 3 deletes the originals from `dw_mcp/guides.py`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_server_guides.py`:

```python
"""The guides the engine serves to an agent that has to pick a capability.

They live on the server rather than in the MCP client's install so the guides
an agent reads are the guides for the engine it is about to drive: an MCP at
one version against a server at another would otherwise index sections the
server does not have. What matters: the index is complete enough to route a
request by, a guide can be fetched a section at a time, and a wrong name says
what the right ones are.
"""

import pytest

from dw.server import guides
from dw.server.guides import GuideError

TASKS_TEXT = (
    "# Tasks\n\n## Speech Generation\n\ngenerate_speech\n\n"
    "## Frame Interpolation\n\ninterpolate_frames\n"
)


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    """A checkout-shaped tree holding two small guides, with the module
    resolving files against it and the table trimmed to those two."""
    root = tmp_path / "checkout"
    (root / "docs").mkdir(parents=True)
    (root / "docs" / "TASKS.md").write_text(TASKS_TEXT)
    (root / "docs" / "WORKFLOW_GUIDE.md").write_text("## Structure\n\nsteps\n")
    monkeypatch.setattr(guides, "__file__", str(root / "dw" / "server" / "guides.py"))
    monkeypatch.setattr(
        guides,
        "GUIDES",
        {"tasks": guides.GUIDES["tasks"], "workflows": guides.GUIDES["workflows"]},
    )
    return root


class TestListing:
    def test_every_curated_guide_is_listed(self, checkout):
        listed = {guide["name"] for guide in guides.list_guides()["guides"]}

        assert listed == {"tasks", "workflows"}

    def test_each_guide_carries_a_summary_and_its_sections(self, checkout):
        # The summary is what an agent matches a request against; the
        # section headings are the routing table
        tasks = next(
            g for g in guides.list_guides()["guides"] if g["name"] == "tasks"
        )

        assert tasks["summary"].strip()
        assert tasks["file"] == "TASKS.md"
        assert tasks["sections"] == ["Speech Generation", "Frame Interpolation"]


class TestWhereTheyComeFrom:
    def test_a_checkout_is_preferred_over_a_stale_packaged_copy(
        self, tmp_path, monkeypatch
    ):
        """build_dist.sh leaves dw/docs/ behind (gitignored). If that copy won,
        every later edit to docs/ would be invisible to the server - so the
        repo's docs/ wins whenever it is there, and the packaged copy is only
        for an install, which has no docs/ beside the package."""
        root = tmp_path / "checkout"
        (root / "dw" / "docs").mkdir(parents=True)
        (root / "docs").mkdir()
        (root / "dw" / "docs" / "TASKS.md").write_text("## Packaged\n")
        (root / "docs" / "TASKS.md").write_text("## Checkout\n")
        monkeypatch.setattr(
            guides, "__file__", str(root / "dw" / "server" / "guides.py")
        )

        assert guides.read_guide("tasks") == "## Checkout\n"

    def test_an_install_reads_the_packaged_copy(self, tmp_path, monkeypatch):
        root = tmp_path / "site-packages"
        (root / "dw" / "docs").mkdir(parents=True)
        (root / "dw" / "docs" / "TASKS.md").write_text("## Packaged\n")
        monkeypatch.setattr(
            guides, "__file__", str(root / "dw" / "server" / "guides.py")
        )

        assert guides.read_guide("tasks") == "## Packaged\n"

    def test_a_guide_missing_from_the_install_says_so(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            guides, "__file__", str(tmp_path / "dw" / "server" / "guides.py")
        )

        with pytest.raises(FileNotFoundError, match="missing from this install"):
            guides.read_guide("tasks")


class TestFetching:
    def test_a_guide_comes_back_whole_by_default(self, checkout):
        guide = guides.get_guide("tasks")

        assert guide == {"name": "tasks", "section": None, "content": TASKS_TEXT}

    def test_one_section_comes_back_alone_with_its_heading(self, checkout):
        guide = guides.get_guide("tasks", section="Speech Generation")

        assert guide["section"] == "Speech Generation"
        assert guide["content"] == "## Speech Generation\n\ngenerate_speech\n"

    def test_the_last_section_runs_to_the_end_of_the_file(self, checkout):
        guide = guides.get_guide("tasks", section="Frame Interpolation")

        assert guide["content"] == "## Frame Interpolation\n\ninterpolate_frames\n"

    def test_a_section_name_need_not_match_case_or_spacing(self, checkout):
        # An agent reproduces a heading from the listing loosely -
        # "speech-generation" for "Speech Generation" - and being strict
        # about it just costs a round trip
        guide = guides.get_guide("tasks", section="speech-generation")

        assert guide["section"] == "Speech Generation"


class TestErrors:
    def test_an_unknown_guide_names_the_ones_that_exist(self, checkout):
        with pytest.raises(GuideError, match="tasks, workflows"):
            guides.get_guide("nonexistent")

    def test_an_unknown_section_names_the_sections_that_exist(self, checkout):
        with pytest.raises(GuideError, match="Speech Generation, Frame Interpolation"):
            guides.get_guide("tasks", section="nonexistent")


class TestTheRealDocs:
    """Over the repo's own docs/, so a renamed file or heading fails here."""

    def test_every_curated_guide_file_exists(self):
        for name in guides.GUIDES:
            assert guides.read_guide(name).strip()

    def test_the_tasks_guide_indexes_speech_generation(self):
        tasks = next(
            g for g in guides.list_guides()["guides"] if g["name"] == "tasks"
        )

        assert "Speech Generation" in tasks["sections"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_server_guides.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dw.server.guides'`

- [ ] **Step 3: Create the module**

Create `dw/server/guides.py`. The `GUIDES` table is copied verbatim from `dw_mcp/guides.py` (nine entries, `workflows` through `workspaces`); do not retype it, copy it.

```python
"""The prose guides, served by the engine they describe.

The gap they close is that a catalog entry describes what a workflow *is*,
and an open-ended request ("a lego movie trailer set in the marvel
universe") names nothing that appears in any of them. What has to be
matched is shape - multi-shot video, cuts, a consistent cast, narration -
and that is what the guides are written in. An agent that reads one before
choosing composes from what exists instead of authoring a fresh workflow
badly.

They are served from here, not read from the MCP client's install, because
the guides an agent reads have to be the guides for the engine it is about
to drive: an MCP at one version against a server at another would
otherwise index sections the server does not have.

Two things keep this cheap. The listing carries each guide's section
headings, so the index is itself the routing table and is small enough to
read every time. And a guide can be fetched one section at a time, because
handing over three thousand lines of TASKS.md is how an agent ends up
reading none of it carefully.
"""

import re
from pathlib import Path

# The docs that bear on choosing a capability. Deliberately not all of them -
# testing, releasing, security and dependency notes are for people working on
# dw, and listing them would dilute an index whose whole value is that it is
# short enough to read in full.
GUIDES = {
    # ... copied verbatim from dw_mcp/guides.py ...
}

# Only the top level. Sub-headings would triple the index for detail that is
# better reached by reading the section they sit in
SECTION_PATTERN = re.compile(r"^## (.+)$", re.M)


class GuideError(LookupError):
    """A guide or section that does not exist. The message names what does,
    so the route can hand it straight back as a 404 detail."""


def _guide_file(file_name):
    """Where a guide lives: the repo's docs/ in a checkout, else the copy
    build_dist.sh puts under dw/docs/ for an install.

    The checkout wins because the build leaves dw/docs/ behind; if that copy
    took precedence, editing docs/ would change nothing an agent reads. The
    same rule default_ui_dir applies to the SPA.
    """
    root = Path(__file__).resolve().parent.parent.parent
    checkout = root / "docs" / file_name
    if checkout.is_file():
        return checkout
    return root / "dw" / "docs" / file_name


def read_guide(name):
    """One guide's full markdown text.

    Raises:
        GuideError: If the name is not a guide.
        FileNotFoundError: If the guide's file is missing from this install.
    """
    if name not in GUIDES:
        raise GuideError(
            f"No guide named '{name}'. The guides are: {', '.join(sorted(GUIDES))}."
        )
    path = _guide_file(GUIDES[name][0])
    if not path.is_file():
        raise FileNotFoundError(
            f"The '{name}' guide is missing from this install "
            f"({GUIDES[name][0]} was not found)."
        )
    return path.read_text(encoding="utf-8")


def _sections(text):
    """The top-level section headings of a guide, in the order they appear."""
    return SECTION_PATTERN.findall(text)


def _normalized(heading):
    """A heading reduced to what a loose match compares.

    An agent reproduces a heading from the listing approximately -
    "speech-generation" for "Speech Generation" - and refusing that costs a
    round trip to say something it already knew.
    """
    return re.sub(r"[^a-z0-9]+", "", heading.lower())


def _extract_section(text, section):
    """One section's text, heading included, or None if there is no such one."""
    wanted = _normalized(section)
    matches = list(SECTION_PATTERN.finditer(text))
    for index, match in enumerate(matches):
        if _normalized(match.group(1)) != wanted:
            continue
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        return match.group(1), text[match.start() : end].rstrip() + "\n"
    return None


def list_guides():
    """Every guide, with what it covers and the sections it holds."""
    listed = []
    for name, (file_name, summary) in GUIDES.items():
        listed.append(
            {
                "name": name,
                "file": file_name,
                "summary": summary,
                "sections": _sections(read_guide(name)),
            }
        )
    return {"guides": listed}


def get_guide(name, section=None):
    """One guide, whole or one section of it."""
    text = read_guide(name)
    if section is None:
        return {"name": name, "section": None, "content": text}

    found = _extract_section(text, section)
    if found is None:
        raise GuideError(
            f"The '{name}' guide has no section '{section}'. Its sections are: "
            f"{', '.join(_sections(text))}."
        )
    heading, content = found
    return {"name": name, "section": heading, "content": content}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_server_guides.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add dw/server/guides.py tests/test_server_guides.py
git commit -m "server: guides module, the GUIDES table and section extraction moved from dw_mcp

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

No ledger change yet — row 4 lands with Task 3.

---

### Task 2: `GET /api/guides` routes, packaging `[model: sonnet]`

**Files:**
- Modify: `dw/server/app.py` (import near line 88; routes after the `/api/schema` route at ~line 1005)
- Modify: `tests/test_server_guides.py` (append a route class)
- Modify: `scripts/build_dist.sh`, `MANIFEST.in`
- Modify: `docs/SERVER.md` (~line 185), `dw/server/CLAUDE.md`

**Interfaces:**
- Consumes: Task 1's `list_guides`, `get_guide`, `GuideError`.
- Produces: `GET /api/guides` → `{"guides": [...]}`; `GET /api/guides/{name}?section=` → `{name, section, content}`; 404 with `detail` for an unknown name or section. Task 3's proxy calls these paths.

- [ ] **Step 1: Write the failing route tests**

Append to `tests/test_server_guides.py`:

```python
from fastapi.testclient import TestClient

from dw.server.app import create_app
from dw.server.jobs import JobManager
from tests.test_server import ScriptedWorkerManager, success_script


@pytest.fixture
def client(tmp_path, checkout):
    """An app over the two-guide checkout; nothing here queues a job."""
    workflow_dir = tmp_path / "workflows"
    workflow_dir.mkdir()
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()
    manager = JobManager(
        str(tmp_path / "outputs"),
        worker_manager=ScriptedWorkerManager(success_script),
        history_path=str(tmp_path / "jobs.sqlite"),
    )
    app = create_app(
        workflow_dir=str(workflow_dir),
        output_dir=str(tmp_path / "outputs"),
        prompt_dir=str(prompt_dir),
        job_manager=manager,
    )
    with TestClient(app, base_url="http://localhost") as c:
        yield c


class TestRoutes:
    def test_the_listing_is_the_index(self, client):
        body = client.get("/api/guides").json()

        names = [g["name"] for g in body["guides"]]
        assert names == ["tasks", "workflows"]
        tasks = body["guides"][0]
        assert tasks["sections"] == ["Speech Generation", "Frame Interpolation"]
        assert tasks["summary"].strip()

    def test_a_whole_guide(self, client):
        body = client.get("/api/guides/tasks").json()

        assert body == {"name": "tasks", "section": None, "content": TASKS_TEXT}

    def test_one_section_matched_loosely(self, client):
        body = client.get(
            "/api/guides/tasks", params={"section": "speech-generation"}
        ).json()

        assert body["section"] == "Speech Generation"
        assert body["content"] == "## Speech Generation\n\ngenerate_speech\n"

    def test_an_unknown_guide_is_a_404_naming_the_guides(self, client):
        response = client.get("/api/guides/nonexistent")

        assert response.status_code == 404
        assert "tasks, workflows" in response.json()["detail"]

    def test_an_unknown_section_is_a_404_naming_the_sections(self, client):
        response = client.get("/api/guides/tasks", params={"section": "nope"})

        assert response.status_code == 404
        assert "Speech Generation, Frame Interpolation" in response.json()["detail"]
```

Move the `import pytest` and the new `from fastapi...`/`from dw.server...` imports to the top of the file with the others (black will complain otherwise).

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_server_guides.py::TestRoutes -v`
Expected: FAIL, 404 on `/api/guides` with no `guides` key (the route does not exist; the SPA fallback or FastAPI's own 404 answers).

- [ ] **Step 3: Add the routes**

In `dw/server/app.py`, beside the other `from .` imports (~line 88):

```python
from . import guides
from .guides import GuideError
```

Directly after the `/api/schema` route (the `workflow_schema` function, ~line 1005), before `@app.post("/api/validate")`:

```python
    # ------------------------------------------------------------ guides

    @app.get("/api/guides")
    def list_guides():
        """The documentation that bears on choosing a capability: each
        guide's name, what it covers, and its section headings. Served by
        the engine rather than read from an MCP client's install, so the
        guides an agent reads are the guides for the engine it drives."""
        return guides.list_guides()

    @app.get("/api/guides/{name}")
    def get_guide(name: str, section: Optional[str] = None):
        """One guide from /api/guides, whole or one section of it. A
        section name is matched loosely - case and punctuation dropped -
        so a heading copied approximately still resolves. An unknown name
        or section is a 404 whose detail lists what exists."""
        try:
            return guides.get_guide(name, section=section)
        except GuideError as e:
            raise HTTPException(status_code=404, detail=str(e))
```

A missing guide *file* (an install with no `dw/docs/`) raises `FileNotFoundError` and surfaces as a 500 — that is a broken install, not a client error, and the log carries the message.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_server_guides.py -v`
Expected: all PASS

- [ ] **Step 5: Packaging**

In `scripts/build_dist.sh`, replace the comment block above `rm -rf dw/docs` (the paragraph beginning `# The guides dw_mcp serves.`) with:

```bash
# The guides dw.server serves at /api/guides. A checkout reads the repo's
# docs/ directly (dw/server/guides.py prefers it), so this copy only matters
# to an install - but without it an installed server has no guides at all.
# Every doc goes in: read_guide only ever opens the names in
# dw.server.guides.GUIDES, so the extras are inert.
```

Also change the header comment's `the guides dw_mcp serves are copied into dw/docs/` to `the guides dw.server serves are copied into dw/docs/`.

In `MANIFEST.in`, add a line so the wheel carries the copy:

```
recursive-include dw/docs *.md
```

The `.gitignore` entry for `dw/docs/` stays.

- [ ] **Step 6: Docs**

In `docs/SERVER.md`, after the `GET /api/schema` bullet (~line 185), add:

```markdown
- `GET /api/guides` — the documentation that bears on choosing a
  capability: each guide's name, what it covers, and its section headings
- `GET /api/guides/{name}?section=` — one guide whole, or one section of
  it; section names match loosely. Served by the engine so an MCP client
  at another version reads the guides for the server it is driving, not
  its own. A checkout serves the repo's `docs/`; an install the copy
  `build_dist.sh` puts under `dw/docs/`
```

In `dw/server/CLAUDE.md`, add a paragraph before `See docs/SERVER.md.`:

```markdown
`guides.py` serves the prose guides (`GET /api/guides`) an agent reads before
choosing a capability. The `GUIDES` table there is the closed set of names; a
guide file resolves to the checkout's `docs/` first, else the packaged
`dw/docs/` copy `scripts/build_dist.sh` makes, the same rule `default_ui_dir`
uses for the SPA. `dw_mcp/guides.py` is a proxy of these routes.
```

- [ ] **Step 7: Run the server suite and commit**

Run: `python -m pytest tests/test_server_guides.py tests/test_server.py -q`
Expected: all PASS

```bash
git add dw/server/app.py tests/test_server_guides.py scripts/build_dist.sh MANIFEST.in docs/SERVER.md dw/server/CLAUDE.md
git commit -m "server: GET /api/guides and /api/guides/{name}; dw/docs is a dw.server build product

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: `dw_mcp/guides.py` becomes a proxy `[model: sonnet]`

**Files:**
- Modify: `dw_mcp/guides.py` (replace whole file)
- Modify: `dw_mcp/server.py` (lines 286–304, the two guide tools)
- Modify: `tests/test_mcp_guides.py` (replace whole file)
- Modify: `dw_mcp/CLAUDE.md` (the `guides.py` sentences, ~lines 24–29), `docs/MCP.md` (~lines 196–203)
- Modify: `docs/proposals/agent-catalog-legibility.md` (ledger row 4)

**Interfaces:**
- Consumes: Task 2's routes.
- Produces: `list_guides(client) -> dict`, `get_guide(client, name, section=None) -> dict`.

- [ ] **Step 1: Write the failing tests**

Replace `tests/test_mcp_guides.py` with:

```python
"""The guide tools are proxies: the guides an agent reads have to be the
guides for the engine it is about to drive, so nothing is read locally."""

import httpx
import pytest

from dw_mcp import guides
from dw_mcp.client import DwApiError, DwClient


def scripted(routes):
    """routes: {(method, path): (status, json_body)}; records params and wire path."""
    seen = []

    def handler(request):
        key = (request.method, request.url.path)
        seen.append(
            {
                "key": key,
                "params": dict(request.url.params),
                "raw_path": request.url.raw_path,
            }
        )
        if key not in routes:
            return httpx.Response(404, json={"detail": f"unrouted {key}"})
        status, body = routes[key]
        return httpx.Response(status, json=body)

    return DwClient(transport=httpx.MockTransport(handler)), seen


LISTING = {
    "guides": [
        {
            "name": "tasks",
            "file": "TASKS.md",
            "summary": "The utility task commands.",
            "sections": ["Speech Generation"],
        }
    ]
}


def test_list_guides_passes_the_listing_through():
    client, seen = scripted({("GET", "/api/guides"): (200, LISTING)})

    assert guides.list_guides(client) == LISTING
    assert seen[0]["key"] == ("GET", "/api/guides")


def test_get_guide_whole_sends_no_section():
    body = {"name": "tasks", "section": None, "content": "## Speech Generation\n"}
    client, seen = scripted({("GET", "/api/guides/tasks"): (200, body)})

    assert guides.get_guide(client, "tasks") == body
    assert seen[0]["params"] == {}


def test_get_guide_section_is_a_query_parameter():
    body = {"name": "tasks", "section": "Speech Generation", "content": "..."}
    client, seen = scripted({("GET", "/api/guides/tasks"): (200, body)})

    assert guides.get_guide(client, "tasks", section="speech-generation") == body
    assert seen[0]["params"] == {"section": "speech-generation"}


def test_a_404_detail_reaches_the_model_as_the_message():
    # The server writes "No guide named 'x'. The guides are: ..." - that text
    # is the answer, and it must not be replaced by an HTTP status
    client, _seen = scripted(
        {
            ("GET", "/api/guides/nonexistent"): (
                404,
                {"detail": "No guide named 'nonexistent'. The guides are: tasks."},
            )
        }
    )

    with pytest.raises(DwApiError, match="The guides are: tasks"):
        guides.get_guide(client, "nonexistent")


def test_a_guide_name_is_path_encoded():
    """A name with '..' must reach the server intact, so its own validation
    - not httpx's dot-segment normalisation - decides what it means.
    httpx.URL.path decodes escapes for display; only raw_path shows the
    wire bytes."""
    client, seen = scripted({})

    with pytest.raises(DwApiError):
        guides.get_guide(client, "../escape")

    assert seen[0]["raw_path"] == b"/api/guides/..%2Fescape"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_mcp_guides.py -v`
Expected: FAIL with `TypeError: list_guides() takes 0 positional arguments but 1 was given`

- [ ] **Step 3: Replace the module**

Replace `dw_mcp/guides.py` in full:

```python
"""The prose guides, read from the engine an agent is about to drive.

These proxy `GET /api/guides` like every other handler proxies its route.
They used to read the markdown shipped in this package, on the theory that
documentation works the same against a remote engine as a local one. It
does not: an MCP at one version against a `dw.serve` at another confidently
indexed sections - `Speech Generation`, `templates/` - the server did not
have, and the guides an agent reads have to describe the engine that will
run what it authors. The one thing given up is answering `list_guides`
while the server is down, which is not a real use.

The `GUIDES` table, section extraction and the checkout-else-packaged file
resolution now live in `dw/server/guides.py`.
"""

from dw_mcp.client import api_path


def list_guides(client):
    """Every guide the engine serves, with what it covers and the sections
    it holds. The section headings are the routing table."""
    return client.get_json("/api/guides")


def get_guide(client, name, section=None):
    """One guide, whole or one section of it. A section name is matched
    loosely on the server, so a heading copied approximately resolves."""
    params = {"section": section} if section is not None else None
    return client.get_json(api_path("api", "guides", name), params=params)
```

- [ ] **Step 4: Pass the client from the tools**

In `dw_mcp/server.py`, the two tool bodies become:

```python
        return guides.list_guides(client)
```

and

```python
        return guides.get_guide(client, name, section=section)
```

Change `list_guides`'s docstring first clause from "List the documentation shipped with this engine" to "List the documentation the engine serves" — it is now literally true.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/test_mcp_guides.py tests/test_mcp_server.py -v`
Expected: all PASS, including the torch-boundary guard in `test_mcp_server.py`.

- [ ] **Step 6: Docs**

`dw_mcp/CLAUDE.md`: replace the sentences from "`guides.py` is the one handler that talks to no server at all" through "authors a fresh workflow instead of composing one." with:

```markdown
`guides.py` proxies `GET /api/guides`: the guides an agent reads are the guides
for the engine it is about to drive, so nothing is read from this package's
own install (the `GUIDES` table lives in `dw/server/guides.py`). They exist
because a request names a subject and the catalog is written in shapes, and an
agent with nowhere to look up the shape authors a fresh workflow instead of
composing one.
```

`docs/MCP.md`: in the `list_guides()` table row, change "List the documentation shipped with the engine" to "List the documentation the engine serves". In the prose paragraph above the table (~line 196), change "indexes the shipped documentation by section" to "indexes the engine's documentation by section".

- [ ] **Step 7: Ledger**

In `docs/proposals/agent-catalog-legibility.md`, ledger row 4 becomes:

```
| 4 server-side guides | done (plan 2, tasks 1–3) | spec §2.1 | `dw/server/guides.py` owns `GUIDES`; `GET /api/guides`, `GET /api/guides/{name}?section=`; `dw_mcp/guides.py` is two proxy calls; payload keys unchanged (`guides`, `content`) so an agent's contract did not move; `dw/docs/` is a `dw.server` build product now in `MANIFEST.in`; supersedes `guides.py`'s "works with the server down" rationale |
```

- [ ] **Step 8: Commit**

```bash
git add dw_mcp/guides.py dw_mcp/server.py tests/test_mcp_guides.py dw_mcp/CLAUDE.md docs/MCP.md docs/proposals/agent-catalog-legibility.md
git commit -m "mcp: list_guides and get_guide proxy the engine's /api/guides

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: `validate_data_all` in `dw/schema.py` `[model: sonnet]`

**Files:**
- Modify: `dw/schema.py`
- Modify: `tests/test_schema.py` (append)

**Interfaces:**
- Consumes: nothing.
- Produces: `MAX_VALIDATION_ERRORS = 25`; `validate_data_all(data, schema) -> list[dict]` with entries `{"path": str | None, "message": str}`; `format_validation_errors(errors) -> str`. Task 5 calls both.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_schema.py`:

```python
from dw.schema import (
    MAX_VALIDATION_ERRORS,
    format_validation_errors,
    validate_data_all,
)


class TestEveryError:
    """An agent iterating on a draft should get every schema violation in one
    round trip, each with the JSON path it sits at."""

    def _three_violations(self):
        workflow = _pipeline_step({}, seed="not-a-number")
        workflow["steps"][0]["release_models"] = "yes"
        workflow["variables"] = "not-an-object"
        return workflow

    def test_independent_violations_are_all_reported_with_paths(self):
        errors = validate_data_all(self._three_violations(), load_schema("workflow"))

        paths = [e["path"] for e in errors]
        assert "steps[0].seed" in paths
        assert "steps[0].release_models" in paths
        assert "variables" in paths
        assert all(e["message"] for e in errors)

    def test_errors_are_sorted_by_path(self):
        errors = validate_data_all(self._three_violations(), load_schema("workflow"))

        assert [e["path"] for e in errors] == sorted(e["path"] for e in errors)

    def test_a_single_error_is_reported_exactly_as_validate_data_does(self):
        # The one-error case is the common one and must not change shape
        # or wording: the CLI, the REPL and the editor all show it
        workflow = _pipeline_step({}, seed="not-a-number")
        schema = load_schema("workflow")

        errors = validate_data_all(workflow, schema)
        _status, message = validate_data(workflow, schema)

        assert len(errors) == 1
        assert message == f"Validation error at {errors[0]['path']}: {errors[0]['message']}"

    def test_a_valid_definition_yields_no_errors(self):
        assert validate_data_all(_pipeline_step({}), load_schema("workflow")) == []

    def test_the_list_is_capped(self):
        # anyOf branches produce dozens of near-identical entries; 25 is
        # more than an agent fixes in one pass
        workflow = _pipeline_step({})
        workflow["steps"] = [
            {"name": f"s{i}", "seed": "x", "pipeline": "nope"} for i in range(40)
        ]

        errors = validate_data_all(workflow, load_schema("workflow"))

        assert len(errors) == MAX_VALIDATION_ERRORS

    def test_duplicates_on_path_and_message_collapse(self):
        errors = validate_data_all(self._three_violations(), load_schema("workflow"))

        assert len(errors) == len({(e["path"], e["message"]) for e in errors})

    def test_a_root_error_has_no_path(self):
        errors = validate_data_all({"id": "x"}, load_schema("workflow"))

        assert errors[0]["path"] is None
        assert "steps" in errors[0]["message"]


class TestFormatting:
    def test_one_error_is_the_familiar_line(self):
        text = format_validation_errors(
            [{"path": "steps[0].seed", "message": "'x' is not of type 'integer'"}]
        )

        assert text == "Validation error at steps[0].seed: 'x' is not of type 'integer'"

    def test_one_root_error_has_no_location(self):
        text = format_validation_errors([{"path": None, "message": "'steps' is a required property"}])

        assert text == "Validation error: 'steps' is a required property"

    def test_several_errors_are_one_per_line_under_one_heading(self):
        text = format_validation_errors(
            [
                {"path": None, "message": "'steps' is a required property"},
                {"path": "variables", "message": "'x' is not of type 'object'"},
            ]
        )

        assert text == (
            "Validation errors (2):\n"
            "  at root: 'steps' is a required property\n"
            "  at variables: 'x' is not of type 'object'"
        )
        # The CLI and the REPL count this prefix once per failure
        assert text.count("Validation error") == 1

    def test_a_capped_list_says_so(self):
        errors = [{"path": f"steps[{i}]", "message": "bad"} for i in range(MAX_VALIDATION_ERRORS)]

        text = format_validation_errors(errors)

        assert text.startswith(f"Validation errors (first {MAX_VALIDATION_ERRORS}):\n")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_schema.py -k "EveryError or Formatting" -v`
Expected: FAIL with `ImportError: cannot import name 'validate_data_all'`

- [ ] **Step 3: Implement**

In `dw/schema.py`, change the import line to:

```python
from jsonschema import validate, ValidationError
from jsonschema.exceptions import best_match
from jsonschema.validators import validator_for
```

and add, after `validate_data`:

```python
# anyOf branches produce dozens of near-identical entries; this is more than
# an agent fixes in one pass, and enough that nothing real is hidden
MAX_VALIDATION_ERRORS = 25


def validate_data_all(data, schema):
    """Every schema violation in `data`, as [{path, message}].

    Sorted by path, deduplicated on (path, message), capped at
    MAX_VALIDATION_ERRORS. Each top-level error is reduced with best_match
    - the same descent into anyOf branches jsonschema.validate performs to
    pick the one exception it raises - so a definition with a single
    violation is reported exactly as validate_data reports it.
    """
    validator = validator_for(schema)(schema)
    seen = {}
    for error in validator.iter_errors(data):
        chosen = best_match([error])
        key = (json_path(chosen.absolute_path), chosen.message)
        seen.setdefault(key, None)
    ordered = sorted(seen, key=lambda key: (key[0] or "", key[1]))
    return [
        {"path": path, "message": message}
        for path, message in ordered[:MAX_VALIDATION_ERRORS]
    ]


def format_validation_errors(errors):
    """The message a raised validation failure carries.

    One error keeps the line every caller already shows -
    'Validation error at <path>: <message>'. Several are listed one per
    line under a single heading, so the text 'Validation error' still
    appears once per failure (the CLI and the REPL count on that).
    """
    if len(errors) == 1:
        path, message = errors[0]["path"], errors[0]["message"]
        location = f" at {path}" if path else ""
        return f"Validation error{location}: {message}"
    count = (
        f"first {MAX_VALIDATION_ERRORS}"
        if len(errors) >= MAX_VALIDATION_ERRORS
        else str(len(errors))
    )
    lines = [f"Validation errors ({count}):"]
    for error in errors:
        lines.append(f"  at {error['path'] or 'root'}: {error['message']}")
    return "\n".join(lines)
```

`validator_for` honours the schema's `$schema` (draft 2020-12), the same validator `jsonschema.validate` picks.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_schema.py -v`
Expected: all PASS. If `test_the_list_is_capped` yields fewer than 25 entries, raise the loop to 80 steps; the point is that the cap is hit, not the exact count that hits it.

- [ ] **Step 5: Commit**

```bash
git add dw/schema.py tests/test_schema.py
git commit -m "schema: validate_data_all reports every violation, sorted, deduplicated, capped at 25

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: every error through `Workflow.validate()` and `/api/validate` `[model: sonnet]`

**Files:**
- Modify: `dw/workflow.py` (`validate`, ~line 278)
- Modify: `dw/server/app.py` (validate route, ~lines 1062–1070; import at line 52)
- Modify: `dw_mcp/server.py` (`validate_workflow` docstring, ~line 477)
- Modify: `ui/src/lib/types.ts` (`ValidationResult`, line 153)
- Modify: `docs/SERVER.md` (~line 186), `docs/MCP.md` (the `validate_workflow` row)
- Modify: `tests/test_workflow.py` (append), `tests/test_server.py` (append), `tests/test_mcp_authoring.py` (one scripted body)
- Modify: `docs/proposals/agent-catalog-legibility.md` (ledger row 8)

**Interfaces:**
- Consumes: Task 4's `validate_data_all`, `format_validation_errors`.
- Produces: `/api/validate` → `{"valid": false, "error": "<joined>", "errors": [{path, message}], "warnings": []}`; the valid response gains `"errors": []`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_workflow.py`:

```python
def test_validate_reports_every_schema_error_at_once(tmp_path):
    """An agent iterating on a draft fixes all of them in one round trip."""
    from dw.workflow import Workflow

    definition = {
        "id": "bad",
        "variables": "not-an-object",
        "steps": [
            {
                "name": "gen",
                "seed": "not-a-number",
                "pipeline": {
                    "configuration": {"component_type": "{Fake}"},
                    "from_pretrained_arguments": {"model_name": "m"},
                    "arguments": {},
                },
            }
        ],
    }
    with pytest.raises(Exception) as exc_info:
        Workflow(definition, str(tmp_path), "").validate()
    message = str(exc_info.value)
    assert message.startswith("Validation errors (2):")
    assert "  at steps[0].seed:" in message
    assert "  at variables:" in message
    assert message.count("Validation error") == 1
```

Append to `tests/test_server.py`:

```python
def test_validate_endpoint_lists_every_schema_error(server):
    with server(success_script) as client:
        workflow = valid_workflow()
        workflow["variables"] = "not-an-object"
        workflow["steps"][0]["seed"] = "not-a-number"

        result = client.post("/api/validate", json={"workflow": workflow}).json()

        assert result["valid"] is False
        paths = [e["path"] for e in result["errors"]]
        assert "variables" in paths and "steps[0].seed" in paths
        # The joined string is what older clients read
        assert result["error"].startswith("Validation errors (")

        result = client.post(
            "/api/validate", json={"workflow": valid_workflow()}
        ).json()
        assert result["valid"] is True and result["errors"] == []
```

In `tests/test_mcp_authoring.py`, change the scripted body in `test_validate_returns_an_invalid_verdict_rather_than_raising` to the new shape (the test's assertions stay):

```python
                {
                    "valid": False,
                    "error": "Validation error: steps must not be empty",
                    "errors": [{"path": None, "message": "steps must not be empty"}],
                    "warnings": [],
                },
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_workflow.py::test_validate_reports_every_schema_error_at_once tests/test_server.py::test_validate_endpoint_lists_every_schema_error -v`
Expected: both FAIL (`message` starts with `Validation error at`; `KeyError: 'errors'`)

- [ ] **Step 3: `Workflow.validate()`**

In `dw/workflow.py`, change the import of `validate_data` to import `validate_data_all, format_validation_errors, load_schema` (keep `load_schema`; drop `validate_data` if nothing else in the file uses it — check with grep), and replace the body of `validate`:

```python
    def validate(self):
        """Validates workflow definition against JSON schema.

        Every violation is reported, one per line, so the CLI, the REPL
        and an agent iterating on a draft fix them in one pass rather than
        one per round trip.
        """
        logger.debug(f"Validating workflow: {self.name}")
        errors = validate_data_all(self.workflow_definition, load_schema("workflow"))
        if errors:
            # message already carries the 'Validation error' prefix
            message = format_validation_errors(errors)
            logger.error(message)
            raise Exception(message)
        logger.debug(f"Workflow {self.name} validated successfully")
```

Then add `validation_errors(self) -> list` right above it, so the route does not re-parse the message:

```python
    def validation_errors(self):
        """Every schema violation in the definition, as [{path, message}];
        empty when it validates."""
        return validate_data_all(self.workflow_definition, load_schema("workflow"))
```

and have `validate()` call `self.validation_errors()` instead of `validate_data_all` directly.

- [ ] **Step 4: The route**

In `dw/server/app.py`, replace the tail of `validate_workflow` (the `try: candidate.validate() ...` through the return) with:

```python
        errors = candidate.validation_errors()
        if errors:
            return {
                "valid": False,
                "error": format_validation_errors(errors),
                "errors": errors,
                "warnings": [],
            }
        return {
            "valid": True,
            "error": None,
            "errors": [],
            "warnings": workflow_argument_warnings(definition),
        }
```

and extend the line-52 import: `from ..schema import load_schema, validate_data, format_validation_errors`. The save route (`PUT /api/workflows`) keeps calling `candidate.validate()`; its 400 detail now carries every line, which is the behaviour wanted there.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/test_workflow.py tests/test_server.py tests/test_mcp_authoring.py tests/test_cli.py tests/test_schema.py -q`
Expected: all PASS, including the existing `count("Validation error") == 1` assertions.

- [ ] **Step 6: Types and docs**

`ui/src/lib/types.ts`, `ValidationResult`:

```ts
export interface ValidationResult {
  valid: boolean
  error: string | null
  /** Every schema violation with its JSON path; empty when valid. */
  errors: { path: string | null; message: string }[]
  warnings: string[]
}
```

Run `cd ui && npx tsc --noEmit -p .` (or `npm run check` if the project defines it) to confirm nothing constructs a `ValidationResult` literal without the field; fix any that does by adding `errors: []`.

`docs/SERVER.md` `POST /api/validate` bullet: append the sentence "Every schema violation is returned in `errors` (`[{path, message}]`, sorted by path, capped at 25), and joined one per line in `error`."

`docs/MCP.md` `validate_workflow` row: append "Returns every schema violation in `errors`, each with the JSON path it sits at, so a draft is fixed in one pass."

`dw_mcp/server.py` `validate_workflow` docstring: append "Every schema error comes back at once, each with its JSON path."

- [ ] **Step 7: Ledger**

Row 8 becomes:

```
| 8 multi-error validation | done (plan 2, tasks 4–5) | spec §2.2 | `validate_data_all` over `iter_errors`, each reduced with `best_match` so the one-error case reads exactly as before; sorted, deduplicated, capped at 25; `Workflow.validate()` joins one per line under one `Validation errors (N):` heading (the CLI/REPL count the prefix once); `/api/validate` adds `errors` beside the existing `error` — the spec's `message` name was not used because `error` is what every client already reads |
```

- [ ] **Step 8: Commit**

```bash
git add dw/workflow.py dw/server/app.py dw_mcp/server.py ui/src/lib/types.ts docs/SERVER.md docs/MCP.md tests/test_workflow.py tests/test_server.py tests/test_mcp_authoring.py docs/proposals/agent-catalog-legibility.md
git commit -m "validate: report every schema error at once, from the CLI to /api/validate

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: `## Authoring a workflow from an agent` `[model: opus]`

**Files:**
- Modify: `docs/WORKFLOW_GUIDE.md` (new top-level section, placed after `## Cross-Step Data Flow`, before `## Result Configuration`)
- Modify: `CLAUDE.md` (one pointer line under `### Type System`)
- Modify: `tests/test_server_guides.py` (append to `TestTheRealDocs`)
- Modify: `docs/proposals/agent-catalog-legibility.md` (ledger rows 7 and 9)

**Interfaces:**
- Consumes: Task 1's `get_guide` over the real docs.
- Produces: the section, reachable as `get_guide("workflows", section="Authoring a workflow from an agent")`.

- [ ] **Step 1: Write the failing tests**

Append to `TestTheRealDocs` in `tests/test_server_guides.py`:

```python
    def test_the_authoring_section_is_reachable_by_name(self):
        guide = guides.get_guide("workflows", section="authoring-a-workflow-from-an-agent")

        assert guide["section"] == "Authoring a workflow from an agent"

    def test_the_authoring_section_names_every_reference_prefix(self):
        """The prefixes the engine reserves are the ones the section has to
        explain; a new prefix added to the engine fails here until it is
        written up."""
        from dw.prompts import RESERVED_TEXT_PREFIXES

        content = guides.get_guide(
            "workflows", section="Authoring a workflow from an agent"
        )["content"]

        for prefix in RESERVED_TEXT_PREFIXES:
            assert f"`{prefix}`" in content, prefix

    def test_the_authoring_section_states_the_cartesian_rule_and_the_loop(self):
        content = guides.get_guide(
            "workflows", section="Authoring a workflow from an agent"
        )["content"]

        assert "cartesian" in content.lower()
        for tool in ("validate_workflow", "save_workflow", "run_workflow", "wait_for_job", "get_output_image"):
            assert f"`{tool}`" in content, tool
        for shape in ("image", "image-set", "image-edit", "shot", "sequence", "audio", "text", "utility"):
            assert f"`{shape}`" in content, shape
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_server_guides.py::TestTheRealDocs -v`
Expected: the three new tests FAIL with `GuideError` (no such section).

- [ ] **Step 3: Write the section**

Insert into `docs/WORKFLOW_GUIDE.md` immediately before `## Result Configuration`. The heading must be exactly `## Authoring a workflow from an agent`. It is written for something composing a draft that will be validated by a machine, not for a person onboarding; every rule states what happens if it is broken. Cover, in this order, with the wording adjusted to read well but the facts kept exactly:

**Opening paragraph.** Who this is for: an agent that has read the catalog (`list_workflows`), found nothing that produces the shape it needs, and is about to write JSON. The schema (`get_schema`) says what is *well-formed*; this section says what the engine *does* with it.

**The reference prefixes.** A subsection `### References`, one bullet each, stating what it resolves to and where it is rooted:

- `variable:name` — the workflow's own `variables` entry, overridden by the caller's `arguments`. A variable declared `null` is optional. Schema validation runs before substitution, so a default must already have the JSON type the field expects (`25`, not `"25"`).
- `previous_result:step_name` — the outputs of an earlier step, by that step's `name`; iterates (see the cartesian rule). A `.field` suffix picks one result field.
- `constant:module.path.NAME` — a value declared in Python, read by import rather than copied into JSON; anything callable is refused.
- `asset:name` — a file in the asset library, rooted at the library, never at the workflow file, confined to it; `upload_asset` returns one of these.
- `output:<workflow identity>/<run id>/<file>` — a file an earlier run wrote; `latest` in the run-id position picks the newest run that holds the file.
- `prompt:name` or `prompt:folder/name` — a stored prompt's `text`; that text may not itself begin with any of these prefixes (the engine rejects it).

**Types and escaping.** A subsection `### Types and escaping`: any key ending in `_type` or `_dtype`, or named `dtype`, is loaded as a Python object — `"FluxPipeline"` from `diffusers`, `"torch.bfloat16"` by full dotted path; wrapping a value in braces, `"{nf4}"`, keeps it a plain string. Getting this wrong fails at load time, after validation passed.

**The cartesian rule.** A subsection `### Several `previous_result` references multiply`: when one step references two or more earlier results, the engine runs the step once for every combination — four images and three masks is twelve iterations. This is by design; it is how one prompt fans out over a set. It means a pairing (shot *i* with speaker *i*, prompt *i* with portrait *i*) is **not** expressible with two references on one step. Write that as one step per pair, each referencing exactly the two things it pairs, or gather the pairs upstream into single results. A step that seems to need a "zip" is the signal to restructure, not to add a reference.

**The loop.** A subsection `### The loop`: `validate_workflow` (free; every schema error at once, with paths, plus argument-name warnings against real signatures) → fix everything reported → `save_workflow` (validates again on the way in, returns the catalog metadata the draft will carry) → `run_workflow` with `acknowledged_cost=true` after saying what it costs → `wait_for_job` rather than polling → `get_output_image` to look at what was made and say whether it answers the request. Validation passing does not mean the model accepts the arguments; the warnings are where a typo against a pipeline's real `__call__` shows up.

**Being found next time.** A subsection `### Being found next time`: the catalog derives a `shape` (one of `image`, `image-set`, `image-edit`, `shot`, `sequence`, `audio`, `text`, `utility`) and `traits` (`has-audio`, `chained`, `image-conditioned`, `identity-referenced`, `needs-input-media`, `composes-workflows`) from the definition's structure, and a `summary` from the first sentence of `description`. Write that first sentence to say what the workflow *makes* and what it *needs supplied*, in under 120 characters — "H3 video with audio between two supplied stills" — not what technique it demonstrates. Declare `shape`, `traits` or `summary` at the top level only when derivation gets it wrong; a declaration that matches the derivation is refused by the repo's tests as noise. `cost` is never derived; leave it absent until a run has been measured.

Keep the whole section under roughly 120 lines. Do not name any model checkpoint in it; the one example summary above names a model family because the summary vocabulary is the author's, not the engine's — keep it to that one.

- [ ] **Step 4: `CLAUDE.md` pointer**

In `CLAUDE.md`, at the end of the `### Type System` section (after the `prompt:` bullet), add:

```markdown
The same conventions, written for an agent composing a workflow over MCP, are
the `Authoring a workflow from an agent` section of docs/WORKFLOW_GUIDE.md;
change both when one changes.
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/test_server_guides.py tests/test_docs_links.py -q`
Expected: all PASS. `test_docs_links.py` fails if the section names an example workflow path that does not exist; it names none.

- [ ] **Step 6: Ledger**

Rows 7 and 9 become:

```
| 7 authoring guide | done (plan 2, task 6) | spec §2.3 | `## Authoring a workflow from an agent` in `WORKFLOW_GUIDE.md`, reachable as one `get_guide` section; `CLAUDE.md` points at it; a test checks every reserved prefix is explained there |
| 9 composition rules | done (plan 2, task 6) | spec §2.3 | the cartesian rule and the one-step-per-pair form are stated in the authoring section, the general case of `scripted-dialogue-and-tts.md`'s reasoning |
```

- [ ] **Step 7: Commit**

```bash
git add docs/WORKFLOW_GUIDE.md CLAUDE.md tests/test_server_guides.py docs/proposals/agent-catalog-legibility.md
git commit -m "docs: authoring a workflow from an agent - references, types, the cartesian rule, the loop

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

## Self-review

**Spec coverage.** §2.1 routes, resolution, proxy, docstring rewrite, packaging, tests → Tasks 1–3. §2.2 `validate_data_all`, sort/dedupe/cap, `validate()` joins, additive route shape, three tests → Tasks 4–5. §2.3 the six bullets → Task 6, with `CLAUDE.md` pointer. The two deviations from spec wording (`content` not `text`; `error` not `message`) are ruled above and recorded in the ledger rows.

**Not in this plan.** The cold-session probe (ledger, 2026-09-07) asked for two new guides — "choosing a video model" and "keeping shots consistent". They are model knowledge and belong to Part 4's packaging question, not to the spec's §2.1, which moves the existing guides and adds none. Left in the ledger as-is.

**Type consistency.** `GuideError` is defined in Task 1 and caught in Task 2. `validate_data_all` / `format_validation_errors` / `MAX_VALIDATION_ERRORS` are defined in Task 4 and imported in Task 5 (`dw/workflow.py`, `dw/server/app.py`). `validation_errors()` is added and used in Task 5 only. The MCP handlers take `client` first, as every other handler does.
