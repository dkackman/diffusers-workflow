# Agent discoverability: the prompt library, the trait vocabulary, and the surfaces that enumerate themselves

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the eight gaps a review of the agent-facing surface found, the
first of which makes the prompt library uncallable and the second of which
leaves it unmentioned by anything an agent reads.

**Architecture:** Each task touches one layer and ends with a test that would
have caught the drift. Two tasks add a *budget* test in the shape
`tests/test_catalog_structure.py::test_the_compact_listing_fits_the_budget`
already uses — a measured ceiling with a comment logging why it moved — because
every finding here is a thing that grew until it broke rather than a thing that
was designed wrong.

**Tech Stack:** Python 3 / FastAPI (`dw/server/`), the MCP SDK (`dw_mcp/`),
pytest, markdown skills under `plugins/dw/skills/`.

**Spec:** the review findings in this session. There is no separate spec file;
the Findings and Global Constraints below are the binding text.

---

## Findings this plan closes

| # | Finding | Task |
| - | ------- | ---- |
| 1 | `list_prompts` returns the full text of all 44 prompts (87,373 chars) and is rejected by the client result cap | 1, 2 |
| 2 | No plugin skill and no guide names the prompt library, so the exemplars for the hardest input are unreachable | 5 |
| 3 | `intended_model` is free text: `minimax-music` (4) and `minimax-music3` (4) are one family under two spellings | 3 |
| 4 | `has-audio` gates on `_kind(step) == "video"`, so `templates/minimax/music` does not carry it | 4 |
| 5 | `series-episodes` is absent from the plugin README table, `plugin.json`, `marketplace.json` and root CLAUDE.md | 6 |
| 6 | The surface costs 13,225 tokens to connect and nothing guards it | 9 |
| 7 | README says "55 tools"; so does `get_guide`'s docstring, where the number carries an argument; there are 57 | 7 |
| 8 | 114 repo-relative links in the nine served guides are dead ends for a reader with no checkout | 8 |

## Global Constraints

- Python: never `eval`/`exec`/`shell=True`; every disk read of a
  request-named path goes through `validate_path(path, base)` with a
  non-`None` base (CLAUDE.md *Security Rules* — CodeQL models the validators
  as barriers, and a read that reaches the disk another way is a real alert).
- Run tests with `venv/bin/python -m pytest` from the repo root. `venv` has
  torch; do not create another.
- **Existing response fields keep their names and meaning.** `GET
  /api/prompts` must keep answering `prompt_dir`, `prompt_dirs`, `prompts`,
  `origins` and `details`, and each `details` entry must keep
  `description`, `intended_model`, `tags` and `text`, for a caller that
  passes no new parameter. `ui/src/lib/pages/PromptsPage.svelte` reads
  `details[name].text` as the card subtitle fallback and calls
  `/api/prompts` with no query string; it must not change.
- `plugins/dw/skills/*/SKILL.md` must stay **≤ 12,288 bytes**
  (`SKILL_SIZE_LIMIT` in `tests/test_plugin_skills.py`). `ltx-2.5` has 29
  bytes of headroom and `minimax-h3` has 12, so every edit to those two is
  byte-accounted in its task and verified with `wc -c`.
- Comments explain *why*, in the voice of the surrounding code — no narration
  of what a line does. Prose in guides and skills is written in the repo's
  register: a sentence says what breaks, not that something is important.
- Commit per task. Imperative subject, a body saying why, ending with:
  `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`
- Do not touch `dw/server/observed_cost.py`, `dw/plan.py` or anything under
  `ui/` beyond what a task names.

---

### Task 1: `GET /api/prompts` takes filters and can leave the text out

The route serves both the web UI (which needs `text` as a card fallback) and
MCP (which cannot afford it). Add narrowing, keep the default byte-identical.

**Files:**
- Modify: `dw/server/app.py:2160-2181` (the `list_prompts` route)
- Test: `tests/test_server.py` (class `TestPromptLibrary`, around line 2924)

**Interfaces:**
- Consumes: `prompt_details(paths)` (`dw/server/app.py:485`), `_prompt_roots()`,
  `workflow_names`, `referenceable`, `WORKSPACE_ORIGIN`, `EXAMPLES_ORIGIN` — all unchanged.
- Produces: `GET /api/prompts?tag=<t>&intended_model=<m>&include_text=<bool>`.
  `include_text` defaults to `True`. When false, every `details` entry drops
  `text` and carries `text_chars: int` instead. `tag` matches one entry of
  `tags` exactly; `intended_model` matches `intended_model` exactly; both are
  case-insensitive; both narrow `prompts`, `origins` and `details` together.
  Task 2 calls this with `include_text=false`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_server.py`, inside `class TestPromptLibrary`:

```python
def _store(self, client, name, prompt):
    assert (
        client.put(f"/api/prompts/{name}", json={"prompt": prompt}).status_code == 200
    )


def test_the_listing_narrows_by_tag_and_model_and_can_omit_the_text(
    self, server, tmp_path
):
    with server(success_script) as client:
        self._store(
            client,
            "minimax/Song",
            {
                "text": "Global Metadata\nbpm is 58.",
                "description": "a song",
                "intended_model": "minimax-music3",
                "tags": ["music", "score"],
            },
        )
        self._store(
            client,
            "minimax/Fox",
            {
                "text": "a red fox at dawn",
                "description": "a fox",
                "intended_model": "minimax-h3",
                "tags": ["wildlife"],
            },
        )

        # A caller that passes nothing gets what it always got
        plain = client.get("/api/prompts").json()
        assert "minimax/Song" in plain["prompts"]
        assert plain["details"]["minimax/Fox"]["text"] == "a red fox at dawn"
        assert "text_chars" not in plain["details"]["minimax/Fox"]

        # include_text=false swaps the payload for its size
        slim = client.get("/api/prompts?include_text=false").json()
        entry = slim["details"]["minimax/Fox"]
        assert "text" not in entry
        assert entry["text_chars"] == len("a red fox at dawn")
        assert entry["description"] == "a fox"

        # A filter narrows the names, the origins and the details together
        by_model = client.get("/api/prompts?intended_model=MINIMAX-MUSIC3").json()
        assert by_model["prompts"] == ["minimax/Song"]
        assert list(by_model["details"]) == ["minimax/Song"]
        assert list(by_model["origins"]) == ["minimax/Song"]

        by_tag = client.get("/api/prompts?tag=Wildlife").json()
        assert by_tag["prompts"] == ["minimax/Fox"]

        # Both at once, and a miss is an empty listing rather than a 404
        assert (
            client.get("/api/prompts?tag=music&intended_model=minimax-h3").json()[
                "prompts"
            ]
            == []
        )

        # The writable directory is reported whatever the filter
        assert client.get("/api/prompts?tag=music").json()["prompt_dir"]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `venv/bin/python -m pytest "tests/test_server.py::TestPromptLibrary::test_the_listing_narrows_by_tag_and_model_and_can_omit_the_text" -v`

Expected: FAIL — `assert 'text' not in entry` (the route ignores the query
string, so `include_text=false` still returns the text).

- [ ] **Step 3: Implement the route change**

Replace `dw/server/app.py:2160-2181` with:

```python
    @app.get("/api/prompts")
    def list_prompts(
        tag: str | None = None,
        intended_model: str | None = None,
        include_text: bool = True,
    ):
        # A stray file too deep or oddly named can sit in the directory, but
        # no workflow could reference it - listing it would only invite that
        paths = {}
        origins = {}
        roots = _prompt_roots()
        for index, root in enumerate(roots):
            for name in workflow_names(root):
                if referenceable(name) and name not in paths:
                    paths[name] = os.path.join(root, f"{name}.json")
                    origins[name] = WORKSPACE_ORIGIN if index == 0 else EXAMPLES_ORIGIN
        details = prompt_details(paths)

        # Narrowing happens after the details are read, since that is where a
        # prompt says what it is for, and it narrows every parallel key at
        # once: a `prompts` list and a `details` map that disagree is worse
        # than no filter at all
        wanted = _matching_prompts(details, tag, intended_model)
        if wanted is not None:
            details = {name: detail for name, detail in details.items() if name in wanted}
        # The three parallel keys agree by construction, filter or no filter.
        # `prompt_details` drops a path whose mtime it cannot read - the file
        # went away between the walk and the read - and listing a name that
        # carries no detail only tells a caller to go and get a 404.
        origins = {name: origin for name, origin in origins.items() if name in details}

        # The MCP listing cannot carry 44 prompt bodies - it exceeds a client's
        # result cap and the listing becomes uncallable - but the editors read
        # `text` as the card fallback, so the omission is opt-in and the size
        # is reported in its place
        if not include_text:
            details = {
                name: {
                    **{key: value for key, value in detail.items() if key != "text"},
                    "text_chars": len(detail.get("text") or ""),
                }
                for name, detail in details.items()
            }

        return {
            # The writable library, unchanged: what a save is written to,
            # and what a client that predates the search path expects
            "prompt_dir": app.state.prompt_dir,
            "prompt_dirs": roots,
            "prompts": sorted(details),
            "origins": origins,
            "details": details,
        }
```

And add, immediately above the route (module level, beside `prompt_details`
at `dw/server/app.py:485` is also fine — put it beside `prompt_details`):

```python
def _matching_prompts(details, tag, intended_model):
    """The prompt names matching the filters, or None when there are none.

    Case-insensitive and exact per value: a `tags` entry or the whole
    `intended_model`, never a substring - `minimax-music` must not match
    `minimax-music3`, which is the confusion the one-spelling-per-family
    rule exists to prevent.
    """
    if tag is None and intended_model is None:
        return None
    wanted_tag = tag.lower() if tag is not None else None
    wanted_model = intended_model.lower() if intended_model is not None else None
    matches = set()
    for name, detail in details.items():
        if wanted_tag is not None and wanted_tag not in {
            str(each).lower() for each in detail.get("tags") or []
        }:
            continue
        if (
            wanted_model is not None
            and str(detail.get("intended_model") or "").lower() != wanted_model
        ):
            continue
        matches.add(name)
    return matches
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `venv/bin/python -m pytest tests/test_server.py -k Prompt -q`

Expected: PASS, including the pre-existing
`test_save_prompt_roundtrip_and_confinement` which asserts `detail["text"]`
on an unfiltered listing.

- [ ] **Step 5: Commit**

```bash
git add dw/server/app.py tests/test_server.py
git commit -m "feat(server): narrow the prompt listing, and let a caller drop the text

The listing carried every prompt's whole body, which the editors need as a
card fallback and an MCP client cannot afford - 44 prompts is 87 KB, past a
client result cap, so the tool could not be called at all. include_text=false
swaps each body for its length, and tag/intended_model narrow the names, the
origins and the details together.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: MCP `list_prompts` stops carrying the bodies

**Files:**
- Modify: `dw_mcp/prompts.py:26-29` (the `list_prompts` handler)
- Modify: `dw_mcp/server.py:897-901` (the tool wrapper and its docstring)
- Modify: `dw_mcp/CLAUDE.md` (the `list_jobs` paragraph gains its sibling)
- Test: `tests/test_mcp_prompts.py`

**Interfaces:**
- Consumes: Task 1's `GET /api/prompts?tag=&intended_model=&include_text=`.
- Produces: `list_prompts(tag: str | None = None, intended_model: str | None = None, include_text: bool = False) -> dict`.
  The handler signature is `list_prompts(client, tag=None, intended_model=None, include_text=False)`.
  Task 5 quotes this signature in the guide and the skills.

- [ ] **Step 1: Write the failing tests**

Note that `scripted()` in this file keys on the request *path* only, so a
query-string assertion reads `request.url.params`. Add a recorder that keeps
them, and the tests, to `tests/test_mcp_prompts.py` after
`test_list_prompts_returns_the_library`:

```python
def scripted_with_params(routes):
    """`scripted`, keeping each request's query parameters."""
    seen = []

    def handler(request):
        key = (request.method, request.url.path)
        seen.append((key, dict(request.url.params)))
        if key not in routes:
            return httpx.Response(404, json={"detail": f"unrouted {key}"})
        status, body = routes[key]
        return httpx.Response(status, json=body)

    return DwClient(transport=httpx.MockTransport(handler)), seen


def test_list_prompts_asks_the_server_to_leave_the_bodies_out():
    client, seen = scripted_with_params(
        {("GET", "/api/prompts"): (200, {"prompts": [], "details": {}})}
    )

    prompts.list_prompts(client)

    # The default is the cheap listing: a prompt body belongs in get_prompt,
    # and 44 of them do not fit a client's result cap
    assert seen[0][1]["include_text"] == "false"
    assert "tag" not in seen[0][1]
    assert "intended_model" not in seen[0][1]


def test_list_prompts_forwards_the_filters_and_can_ask_for_the_text():
    client, seen = scripted_with_params(
        {("GET", "/api/prompts"): (200, {"prompts": [], "details": {}})}
    )

    prompts.list_prompts(
        client, tag="ic-lora", intended_model="ltx-2.5", include_text=True
    )

    assert seen[0][1] == {
        "tag": "ic-lora",
        "intended_model": "ltx-2.5",
        "include_text": "true",
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `venv/bin/python -m pytest tests/test_mcp_prompts.py -k list_prompts -v`

Expected: FAIL — `KeyError: 'include_text'`; the handler takes no parameters
and sends no query string.

- [ ] **Step 3: Implement the handler**

Replace `dw_mcp/prompts.py:26-29` with:

```python
def list_prompts(client, tag=None, intended_model=None, include_text=False):
    """The library as a routing table: what each prompt is for, not what it
    says.

    The bodies are left out by default, which is the same lesson `list_jobs`
    learned - the whole library is 44 prompts and 87 KB, past a client's
    result cap, so a listing that carried them could not be called at all.
    `get_prompt` is where a body comes from once one has been chosen.
    """
    params = {"include_text": "true" if include_text else "false"}
    if tag is not None:
        params["tag"] = tag
    if intended_model is not None:
        params["intended_model"] = intended_model
    return client.get_json("/api/prompts", params=params)
```

- [ ] **Step 4: Rewrite the tool wrapper**

Replace `dw_mcp/server.py:897-901` with:

```python
    def list_prompts(
        tag: Optional[str] = None,
        intended_model: Optional[str] = None,
        include_text: bool = False,
    ) -> dict:
        """List the stored prompts - the worked examples a workflow reaches
        by writing "prompt:name" or "prompt:folder/name". Each entry carries
        its `description`, `intended_model`, `tags` and the size of its text;
        `get_prompt` returns the text itself. This is where the caption a
        model was trained on is already written out, so read the exemplar
        for the family you are about to run rather than inventing the
        format: `intended_model` narrows to one family (`minimax-h3`,
        `minimax-music3`, `ltx-2.5`, `z-image`, `flux`) and `tag` to one
        label. `include_text=true` returns every body, which for the whole
        library is more than a client will accept - filter first."""
        return prompts.list_prompts(
            client,
            tag=tag,
            intended_model=intended_model,
            include_text=include_text,
        )
```

- [ ] **Step 5: Run the MCP tests to verify they pass**

Run: `venv/bin/python -m pytest tests/test_mcp_prompts.py tests/test_mcp_server.py -q`

Expected: PASS. `test_mcp_server.py` asserts every tool has a non-empty
description and that `list_prompts` is annotated read-only; both still hold.

- [ ] **Step 6: Record it beside the `list_jobs` paragraph**

In `dw_mcp/CLAUDE.md`, immediately after the sentence ending `...and it takes
a `workspace` of its own to narrow by.`, add:

```
`list_prompts` had the same disease and the same cure: the library's 44
prompts are 87 KB of prompt bodies, which no client will accept, so the
listing asks for `include_text=false` and carries each prompt's
`description`, `intended_model`, `tags` and `text_chars` instead - the
routing table, with `get_prompt` for the one body that was chosen. It also
forwards `tag` and `intended_model`, because the library is where the
trained caption format for a family is already written out and the reason to
read it is to find that one.
```

- [ ] **Step 7: Commit**

```bash
git add dw_mcp/prompts.py dw_mcp/server.py dw_mcp/CLAUDE.md tests/test_mcp_prompts.py
git commit -m "fix(mcp): list_prompts is callable again

It returned every prompt's full text - 87 KB across 44 prompts - and was
rejected outright by a client's result cap, so the library that holds the
worked caption examples could not be listed at all. Same cure as list_jobs:
ask the server for the routing table, with tag/intended_model to narrow and
get_prompt for the body.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: one spelling per family in `intended_model`, swept

Two repo prompts say `minimax-music` where the family is `minimax-music3`,
so Task 2's `intended_model` filter would return half the Music 3 library.
Three prompts declare nothing, which the schema allows and this task keeps
allowing — the rule is *one spelling per family*, not *every prompt declares
one*.

**Files:**
- Modify: `prompts/minimax/acoustic_pop_song.json`, `prompts/minimax/otter_soul_song.json`
- Modify: `dw/prompt_schema.json:19-22` (the `intended_model` description)
- Create: `tests/test_prompt_library.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `INTENDED_MODELS` in `tests/test_prompt_library.py`, the vocabulary
  the repo's own library is held to. Task 2's docstring already quotes the
  same five values; keep them identical.

- [ ] **Step 1: Write the failing test**

Create `tests/test_prompt_library.py`:

```python
"""Conventions the whole prompt library is held to.

`intended_model` is documented as informational, and it is - the engine
ignores it. What it is not is free: `list_prompts(intended_model=...)`
narrows by an exact match, so `minimax-music` beside `minimax-music3` is a
filter that returns half a family and looks like it returned all of it. The
sweep is the same shape `tests/test_observed_cost.py` uses for
`cost_drivers`: a value that buckets on nothing looks exactly like a value
that works.

An empty `intended_model` is allowed. Some stored text is not written for a
model at all - an enhancer's system prompt, a generic landscape - and
inventing a family for it would be worse than saying nothing.
"""

import glob
import json
import os

import pytest

from tests.test_examples import REPO_ROOT

PROMPT_DIR = os.path.join(REPO_ROOT, "prompts")

# One spelling per family. Every value here is either a key the enhancer
# presets preselect for (`intended_models` in dw/server/enhancers.py) or a
# model family the catalog ships templates for.
INTENDED_MODELS = frozenset(
    {
        "minimax-h3",
        "minimax-music3",
        "ltx-2.5",
        "z-image",
        "flux",
    }
)


def prompt_files():
    return sorted(glob.glob(os.path.join(PROMPT_DIR, "**", "*.json"), recursive=True))


@pytest.mark.parametrize(
    "path", prompt_files(), ids=lambda p: os.path.relpath(p, PROMPT_DIR)
)
def test_intended_model_is_one_of_the_known_families(path):
    with open(path) as file:
        definition = json.load(file)
    declared = definition.get("intended_model")
    if not declared:
        return
    assert declared in INTENDED_MODELS, (
        f"{os.path.relpath(path, REPO_ROOT)} declares intended_model "
        f"{declared!r}; list_prompts filters on an exact match, so a variant "
        f"spelling hides the prompt from the family it belongs to. "
        f"Known: {sorted(INTENDED_MODELS)}"
    )


def test_every_enhancer_preset_targets_a_known_family():
    """The presets and the library share one vocabulary, or the editor
    preselects a preset no prompt will ever match."""
    from dw.server.enhancers import PRESETS

    for key, preset in PRESETS.items():
        for model in preset["intended_models"]:
            assert model in INTENDED_MODELS, (
                f"enhancer preset {key!r} targets {model!r}, which no prompt "
                f"may declare"
            )
```

- [ ] **Step 2: Run it to verify it fails**

Run: `venv/bin/python -m pytest tests/test_prompt_library.py -q`

Expected: 2 failures — `minimax/acoustic_pop_song.json` and
`minimax/otter_soul_song.json` declare `'minimax-music'`.

- [ ] **Step 3: Fix the two prompts**

```bash
cd /Users/don/src/dkackman/diffusers-workflow
for f in prompts/minimax/acoustic_pop_song.json prompts/minimax/otter_soul_song.json; do
  venv/bin/python - "$f" <<'PY'
import json, sys
path = sys.argv[1]
with open(path) as file:
    prompt = json.load(file)
assert prompt["intended_model"] == "minimax-music", prompt["intended_model"]
prompt["intended_model"] = "minimax-music3"
with open(path, "w") as file:
    json.dump(prompt, file, indent=2)
    file.write("\n")
PY
done
git diff --stat
```

- [ ] **Step 4: Document the rule where a prompt is written**

In `dw/prompt_schema.json`, replace the `intended_model` description with:

```json
      "description": "The model or model family the prompt was written for, such as 'minimax-h3'. Informational - the engine ignores it - but one spelling per family: 'list_prompts' narrows by an exact match, so a variant hides the prompt from its own family. The repo's library is swept by tests/test_prompt_library.py. Leave it out for text written for no particular model."
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `venv/bin/python -m pytest tests/test_prompt_library.py tests/test_prompts.py tests/test_ltx_prompt_library.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add prompts/minimax/acoustic_pop_song.json prompts/minimax/otter_soul_song.json \
        dw/prompt_schema.json tests/test_prompt_library.py
git commit -m "fix(prompts): one spelling per family in intended_model

Two Music 3 captions said minimax-music and four said minimax-music3, which
list_prompts' exact-match filter turns into a listing that returns half a
family and looks complete. Swept, the way cost_drivers is: a value that
buckets on nothing looks exactly like one that works. An absent value stays
legal - an enhancer's system prompt is written for no family.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

**Note for the executor:** the live server's workspace holds ~20 more prompts
(`iron-bloom/*`, `tool-musical/*`, `slow-light/*`, `parhelion/*`) that are not
in this repo, and two of them declare `minimax-music`. They are the user's
data, not repo content; the sweep cannot reach them. Say so when reporting the
task rather than trying to edit the box.

---

### Task 4: `has-audio` means a generated audio track, as it says

`_derive_traits` promises "a speech task, a video pipeline asked for audio, or
one carrying a component that exists only to synthesise a waveform", but the
last two checks gate on `_kind(step) == "video"`, so `templates/minimax/music`
— which generates and normalizes a 44.1 kHz track — does not carry the trait.
`list_workflows(shape="audio", traits="has-audio")` returns only
`templates/generate-speech`.

**Files:**
- Modify: `dw/server/catalog_shape.py:230-263` (`_derive_traits`)
- Test: `tests/test_catalog_shape.py`

**Interfaces:**
- Consumes: `_generates(step)` and `_kind(step)` (`dw/server/catalog_shape.py:98`, `:91`), unchanged.
- Produces: no signature change. Exactly one catalog entry gains a trait
  (`templates/minimax/music`); verified in Step 5.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_catalog_shape.py`:

```python
def test_a_generated_audio_track_carries_has_audio():
    """A pipeline whose result is audio generates audio, whether or not a
    video step is anywhere near it. The checks used to gate on video, so
    Music 3's own template answered a `traits=has-audio` filter with
    nothing - the one filter that names the thing it makes."""
    definition = {
        "id": "Song",
        "steps": [
            {
                "name": "generate_music",
                "pipeline": {
                    "configuration": {"component_type": "ModularPipeline"},
                    "from_pretrained_arguments": {
                        "model_name": "MiniMaxAI/MiniMax-Music3"
                    },
                    "arguments": {"prompt": "variable:prompt"},
                },
                "result": {"content_type": "audio/mpeg"},
            }
        ],
    }

    metadata = derive_catalog_metadata(definition)

    assert metadata["shape"] == "audio"
    assert "has-audio" in metadata["traits"]


def test_processing_a_supplied_track_does_not_claim_to_generate_one():
    """`has-audio` says the workflow *makes* a track. Trimming one it was
    handed is what `needs-input-media` is for."""
    definition = {
        "id": "Level",
        "steps": [
            {
                "name": "balanced",
                "task": {
                    "command": "normalize_audio",
                    "arguments": {"audio": "asset:song.mp3"},
                },
                "result": {"content_type": "audio/mpeg"},
            }
        ],
    }

    metadata = derive_catalog_metadata(definition)

    assert "has-audio" not in metadata["traits"]
    assert "needs-input-media" in metadata["traits"]
```

`derive_catalog_metadata` is already imported at the top of that file.

- [ ] **Step 2: Run the tests to verify the first fails**

Run: `venv/bin/python -m pytest tests/test_catalog_shape.py -k has_audio -v`

Expected: `test_a_generated_audio_track_carries_has_audio` FAILS
(`'has-audio' not in []`); `test_processing_a_supplied_track...` PASSES
already, and is there to pin that the fix does not over-fire.

- [ ] **Step 3: Fix the derivation**

In `dw/server/catalog_shape.py`, in `_derive_traits`, add one clause inside
the existing `for step in steps:` loop, immediately after the
`generate_speech` clause:

```python
        # A pipeline whose own result is a waveform generates one. The three
        # checks below it all ask about a *video* step that also carries
        # audio, which left Music 3's template - the one entry in the catalog
        # whose whole output is a track - answering a `has-audio` filter with
        # nothing.
        if _generates(step) and _kind(step) == "audio":
            traits.add("has-audio")
```

Then correct the docstring's first line so it describes the code:

```python
    """The independent facts about how the output is made or what it needs.

    `has-audio` says the workflow emits a generated audio track - a step
    whose own result is a waveform, a speech task, a video pipeline asked
    for audio, or one carrying a component that exists only to synthesise
    one. Not specifically dialogue, and not a track the workflow was handed.
    """
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `venv/bin/python -m pytest tests/test_catalog_shape.py tests/test_catalog_structure.py -q`

Expected: PASS. `test_the_compact_listing_fits_the_budget` still holds — one
extra trait string on one entry.

- [ ] **Step 5: Verify the blast radius is one entry**

Run:

```bash
cd /Users/don/src/dkackman/diffusers-workflow && venv/bin/python -c "
import json, glob
from dw.server.catalog_shape import derive_catalog_metadata as derive
for path in sorted(glob.glob('workflows/**/*.json', recursive=True)):
    with open(path) as file:
        traits = derive(json.load(file))['traits']
    if 'has-audio' in traits:
        print(path)
"
```

Expected: the 40 entries that already carried it, **plus exactly one new line**
`workflows/templates/minimax/music.json`. If any other entry appears, stop and
report it — the clause is over-firing.

- [ ] **Step 6: Commit**

```bash
git add dw/server/catalog_shape.py tests/test_catalog_shape.py
git commit -m "fix(catalog): has-audio covers a workflow whose output is the track

The three checks all gated on a video step, so templates/minimax/music -
the one entry whose whole deliverable is a 44.1 kHz track - answered a
traits=has-audio filter with nothing, while the docstring claimed it would
match a component that exists only to synthesise a waveform. A step whose own
result is audio now says so; a step that trims a supplied track still does not.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: the prompt library is named where every agent reads

The library is the fallback for the one thing an MCP-only client cannot get
from the catalog — the trained caption format. It belongs in the two
engine-generic places every agent reads (the tool docstring, done in Task 2,
and the authoring guide), with a one-line pointer in each skill.

**Files:**
- Modify: `docs/WORKFLOW_GUIDE.md` (the `asset:`/`prompt:` reference list in
  *Authoring a workflow from an agent*)
- Modify: `plugins/dw/skills/ltx-2.5/SKILL.md` (byte-accounted)
- Modify: `plugins/dw/skills/minimax-h3/SKILL.md` (byte-accounted)
- Modify: `plugins/dw/skills/minimax-music3/SKILL.md`
- Test: `tests/test_plugin_skills.py`

**Interfaces:**
- Consumes: Task 2's `list_prompts(tag=, intended_model=, include_text=)` and
  the unchanged `get_prompt(name)`. Quote those spellings exactly.
- Produces: nothing later tasks consume.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_plugin_skills.py`:

```python
@pytest.mark.parametrize(
    "path", SKILLS, ids=lambda p: os.path.basename(os.path.dirname(p))
)
def test_a_skill_points_at_the_stored_exemplars(path):
    """Every family's hardest input is its prompt format, and the library
    already holds captions written to it. A skill that names the vendor's
    spec and not the worked example leaves an agent inventing prose it
    could have read - and `prompts/ltx2/` as a directory is unreachable
    from a client that has the MCP and no checkout."""
    text = skill_text(path)
    assert "list_prompts" in text or "get_prompt" in text, (
        f"{path} never names the prompt library; a filesystem path is not a "
        f"call an MCP client can make"
    )
```

- [ ] **Step 2: Run it to verify it fails**

Run: `venv/bin/python -m pytest tests/test_plugin_skills.py -k stored_exemplars -v`

Expected: 4 FAIL, one per skill.

- [ ] **Step 3: Add the convention to the authoring guide**

In `docs/WORKFLOW_GUIDE.md`, in *Authoring a workflow from an agent* →
*References*, replace the `prompt:` bullet:

```markdown
- `prompt:` — `prompt:name` or `prompt:folder/name` is a stored prompt's
  `text`, rooted at the prompt library. That text may not itself begin with any of these
  prefixes; the engine rejects such a prompt rather than resolving twice.
```

with:

```markdown
- `prompt:` — `prompt:name` or `prompt:folder/name` is a stored prompt's
  `text`, rooted at the prompt library. That text may not itself begin with any of these
  prefixes; the engine rejects such a prompt rather than resolving twice.
  The library is also the worked-example shelf: a template's default prompt
  is usually a `prompt:` reference, and the text behind it is a caption
  written to whatever spec that model was trained on. Before writing a
  prompt for a family, read the one that is already there —
  `list_prompts(intended_model="ltx-2.5")` for the shelf,
  `get_prompt("ltx2/fox_dawn_choir")` for a body. The listing leaves the
  bodies out by default and reports each one's `text_chars`; asking for all
  of them at once is more than a client will accept.
```

- [ ] **Step 4: Point the LTX skill at the call instead of the directory (byte-accounted)**

`plugins/dw/skills/ltx-2.5/SKILL.md` is 12,259 bytes — 29 to spare. In the
`## Prompts` section, replace this sentence (59 bytes):

```
The stored prompts under
`prompts/ltx2/` are written to it.
```

with (53 bytes):

```
The `ltx2/`
stored prompts (`list_prompts`) follow it.
```

Net −6 bytes.

- [ ] **Step 5: Point the H3 skill at the call (byte-accounted)**

`plugins/dw/skills/minimax-h3/SKILL.md` is 12,276 bytes — 12 to spare, so
this edit pays for itself twice over before it spends.

First free 20 bytes in `## Prompts` item 2. Replace:

```
   for text- and frame-conditioned generation, and
```

with:

```
   for text and frame conditioning, and
```

(−12) and replace:

```
   for reference-conditioned generation.
```

with:

```
   for reference conditioning.
```

(−10). Then spend 14 on the warning, which is more useful pointing somewhere.
Replace:

```
H3 wants Context-IR, MiniMax's own format. Do not invent it and do not
paraphrase it from examples:
```

with:

```
H3 wants Context-IR, MiniMax's own format. Do not invent it and do not
paraphrase it from `get_prompt` examples:
```

Net −8 bytes.

- [ ] **Step 6: Point the Music 3 skill at the call**

`plugins/dw/skills/minimax-music3/SKILL.md` is 10,572 bytes — 1,716 to spare.
In `## Prompts`, replace item 3:

```
3. The concise one-paragraph form the templates' stored prompts use (genre,
   BPM, key, emotional progression, listening scenario, production profile,
   vocals, arrangement) is the model card's own example and works; the
   three-heading form is for precise control.
```

with:

```
3. The concise one-paragraph form the templates' stored prompts use (genre,
   BPM, key, emotional progression, listening scenario, production profile,
   vocals, arrangement) is the model card's own example and works; the
   three-heading form is for precise control. Both are on the shelf:
   `list_prompts(intended_model="minimax-music3")` names them and
   `get_prompt` reads one. Read the exemplar before writing a caption - it
   is what the format looks like when it is right, which is not the same as
   a source to paraphrase the rules from.
```

- [ ] **Step 7: Point the series skill at the cast's prompts**

`plugins/dw/skills/series-episodes/SKILL.md` is 6,605 bytes — ample. At the end
of `## 0. Draw the cast once, before episode 1`, add:

```
Save each portrait prompt with `save_prompt` under one folder for the series
and reference it as `prompt:<series>/<character>` from every episode.
`list_prompts(tag="<series>")` is then the cast list, and a character
described the same way in episode 6 as in episode 1 is a reference rather
than a paragraph retyped - which is the drift this skill exists to stop.
```

- [ ] **Step 8: Run the tests to verify they pass, and check the byte caps**

```bash
cd /Users/don/src/dkackman/diffusers-workflow
venv/bin/python -m pytest tests/test_plugin_skills.py tests/test_docs_links.py -q
for f in plugins/dw/skills/*/SKILL.md; do printf "%-46s %6d\n" "$f" $(wc -c < "$f"); done
```

Expected: PASS, and every skill ≤ 12288 bytes — `ltx-2.5` at 12,253 and
`minimax-h3` at 12,268.

- [ ] **Step 9: Commit**

```bash
git add docs/WORKFLOW_GUIDE.md plugins/dw/skills tests/test_plugin_skills.py
git commit -m "docs: name the prompt library where an agent will read it

The library holds captions written to each family's trained spec, and
nothing an agent reads mentioned it: the guide described 'prompt:' as a
reference convention without saying the shelf behind it is worth reading,
and the LTX skill pointed at prompts/ltx2/ - a directory an MCP client with
no checkout cannot list. Now the call is named in the guide and in every
skill, and a test keeps it named.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: the plugin enumerates all four of its skills, pinned

`series-episodes` ships, is tested and is invocable, and appears in none of the
four places that list what the plugin teaches.

**Files:**
- Modify: `plugins/dw/README.md` (the skill table)
- Modify: `plugins/dw/.claude-plugin/plugin.json` (`description`)
- Modify: `.claude-plugin/marketplace.json` (the `dw` entry's `description`)
- Modify: `CLAUDE.md:50-52`
- Test: `tests/test_plugin_skills.py`

**Interfaces:**
- Consumes: `SKILLS` and `PLUGIN_DIR` (`tests/test_plugin_skills.py:21`, `:20`).
- Produces: nothing later tasks consume.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_plugin_skills.py`:

```python
@pytest.mark.parametrize(
    "path", SKILLS, ids=lambda p: os.path.basename(os.path.dirname(p))
)
def test_a_skill_is_enumerated_where_the_plugin_describes_itself(path):
    """A skill nobody lists is a capability nobody installs for. The four
    documents that enumerate them drifted the moment a fourth skill shipped,
    and the size and catalog tests glob the directory, so nothing noticed."""
    name = os.path.basename(os.path.dirname(path))
    for document in (
        os.path.join(PLUGIN_DIR, "README.md"),
        os.path.join(REPO_ROOT, "CLAUDE.md"),
    ):
        with open(document) as file:
            assert name in file.read(), (
                f"skill {name!r} is not named in {os.path.relpath(document, REPO_ROOT)}"
            )
```

- [ ] **Step 2: Run it to verify it fails**

Run: `venv/bin/python -m pytest tests/test_plugin_skills.py -k enumerated -v`

Expected: 1 FAIL — `skill 'series-episodes' is not named in
plugins/dw/README.md` (3 pass).

- [ ] **Step 3: Add the table row**

In `plugins/dw/README.md`, add after the `ltx-2.5` row:

```markdown
| `series-episodes` | A series rather than one generation: several episodes over one cast, each cut and scored from its own template runs. The five-beat recut-bed-match_levels-normalize-pair procedure, and the one step that must not be skipped or the cast drifts between episodes. |
```

- [ ] **Step 4: Widen the two descriptions**

In `plugins/dw/.claude-plugin/plugin.json` and the `dw` entry of
`.claude-plugin/marketplace.json`, replace the `description` value (it is the
same string in both) with:

```
Compose MiniMax H3 video, MiniMax Music 3 and LTX-2.5 workflows over a dw MCP server, and cut a multi-episode series from them: which template fits which shape, the hard rules, cost, and how to judge the output. Prompt format comes from the vendors' own guides.
```

Also update `marketplace.json`'s top-level `description` from
`"one composition skill per model family."` to
`"a composition skill per model family, and one for cutting a series from them."`

- [ ] **Step 5: Correct CLAUDE.md**

Replace `CLAUDE.md:50-52`:

```
`.claude-plugin/marketplace.json` publishes the `dw` plugin in `plugins/dw/`: one
composition skill per model family (`minimax-h3`, `minimax-music3`, `ltx-2.5`) that
chooses a template for a request's shape and states the family's hard rules. Model
```

with:

```
`.claude-plugin/marketplace.json` publishes the `dw` plugin in `plugins/dw/`: one
composition skill per model family (`minimax-h3`, `minimax-music3`, `ltx-2.5`) that
chooses a template for a request's shape and states the family's hard rules, plus
`series-episodes`, which is a shape above them - several episodes over one cast,
each cut and scored from those templates' runs. Every skill the directory holds is
named in `plugins/dw/README.md` and here, pinned by the same test. Model
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `venv/bin/python -m pytest tests/test_plugin_skills.py -q && venv/bin/python -c "import json; [json.load(open(p)) for p in ('plugins/dw/.claude-plugin/plugin.json', '.claude-plugin/marketplace.json')]"`

Expected: PASS, and both JSON files parse.

- [ ] **Step 7: Commit**

```bash
git add plugins/dw/README.md plugins/dw/.claude-plugin/plugin.json \
        .claude-plugin/marketplace.json CLAUDE.md tests/test_plugin_skills.py
git commit -m "docs(plugin): list series-episodes where the plugin describes itself

It shipped, is tested and is invocable, and the README table, both plugin
descriptions and CLAUDE.md all still said three skills - the size and catalog
tests glob the directory, so nothing noticed a fourth. Pinned now: every skill
directory must be named in the README and in CLAUDE.md.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: the tool count is the surface's, in both places that state it

The README says 55. So does `get_guide`'s own docstring, where the number
carries an argument — that a whole guide costs more to read than the surface
costs to connect. Both drift every time a tool lands.

**Files:**
- Modify: `README.md:86`
- Modify: `dw/server/guides.py` (the `get_guide` docstring, ~line 172)
- Test: `tests/test_mcp_server.py`

**Interfaces:**
- Consumes: `EXPECTED_TOOLS` (`tests/test_mcp_server.py`), currently 57 names.
- Produces: nothing later tasks consume.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_mcp_server.py`:

```python
def test_the_stated_tool_count_is_the_registered_one():
    """Two documents state the size of the surface: the README, where it is
    the first claim made about it, and get_guide's docstring, where it carries
    the argument that a whole guide costs more than connecting does. Both said
    55 while 57 were registered, and nothing was checking either."""
    import re

    from tests.test_examples import REPO_ROOT

    stated = {}
    with open(os.path.join(REPO_ROOT, "README.md")) as file:
        stated["README.md"] = re.search(r"The agent has (\d+) tools", file.read())
    with open(os.path.join(REPO_ROOT, "dw", "server", "guides.py")) as file:
        stated["dw/server/guides.py"] = re.search(
            r"(\d+)-tool MCP surface", file.read()
        )

    for where, found in stated.items():
        assert found, f"{where} no longer states a tool count in the expected form"
        assert int(found.group(1)) == len(EXPECTED_TOOLS), (
            f"{where} says {found.group(1)} tools; {len(EXPECTED_TOOLS)} are registered"
        )
```

- [ ] **Step 2: Run it to verify it fails**

Run: `venv/bin/python -m pytest tests/test_mcp_server.py -k stated_tool_count -v`

Expected: FAIL — `README.md says 55 tools; 57 are registered`.

- [ ] **Step 3: Correct both**

In `README.md:86`, replace `The agent has 55 tools covering the whole surface — the`
with `The agent has 57 tools covering the whole surface — the`.

In `dw/server/guides.py`, in the `get_guide` docstring, replace
`more in one call than the entire 55-tool MCP surface costs to connect,`
with
`more in one call than the entire 57-tool MCP surface costs to connect,`.

The claim itself still holds and is worth keeping: Task 9 measures the surface
at 13,225 tokens against WORKFLOW_GUIDE's ~19.6k.

- [ ] **Step 4: Run it to verify it passes**

Run: `venv/bin/python -m pytest tests/test_mcp_server.py tests/test_server_guides.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add README.md dw/server/guides.py tests/test_mcp_server.py
git commit -m "docs: pin the stated tool count to the registered tools

The README said 55 and so did get_guide's docstring, where the number carries
the argument for serving a guide a section at a time; 57 are registered. Both
are checked against EXPECTED_TOOLS now rather than maintained by hand.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: a served guide says how to turn a repo path into a catalog name

The nine served guides hold 114 links of the form
`[image-to-video.json](../workflows/templates/minimax/image-to-video.json)`
(TASKS 48, RECIPES_24GB 22, WORKFLOW_GUIDE 21, the rest 23). They resolve for a
reader with a checkout and dead-end for the MCP-only reader that
`docs/AGENT_LOOP.md` defines as the tester role. Rewriting 114 links in prose
people also read is the wrong fix; saying the rule once, on the payload that
contains one, is the right one.

**Files:**
- Modify: `dw/server/guides.py`
- Test: `tests/test_server_guides.py`

**Interfaces:**
- Consumes: the existing `get_guide` section extraction in `dw/server/guides.py`.
- Produces: an extra `catalog_paths: str` key on a `GET /api/guides/{name}`
  response whose served content contains a `workflows/....json` path. Absent
  otherwise. `list_guides` is unchanged — the index carries no prose.

- [ ] **Step 1: Write the failing tests**

`tests/test_server_guides.py` tests the mechanism against the synthetic
`checkout` fixture (`tmp_path` holding two tiny guides, `GUIDES` monkeypatched
down to them) and the shipped text separately in `class TestTheRealDocs`. Both
get a test.

First give the fixture a section that links an example. Replace `TASKS_TEXT`
at `tests/test_server_guides.py:20-23`:

```python
TASKS_TEXT = (
    "# Tasks\n\n## Speech Generation\n\ngenerate_speech\n\n"
    "## Frame Interpolation\n\ninterpolate_frames\n"
)
```

with:

```python
TASKS_TEXT = (
    "# Tasks\n\n## Speech Generation\n\ngenerate_speech\n\n"
    "## Frame Interpolation\n\n"
    "interpolate_frames - see [it](../workflows/templates/interpolate-frames.json)\n"
)
```

Then add to `class TestFetching`:

```python
def test_a_section_linking_an_example_says_how_to_reach_it(self, checkout):
    """A guide's link to an example is a repo path. The reader these
    guides exist for has no checkout - it has the MCP and nothing else -
    so a payload holding such a path carries the translation rule, rather
    than 114 links being rewritten in prose people read too."""
    body = guides.get_guide("tasks", "Frame Interpolation")

    # The note is one fixed sentence - it states the rule, it does not
    # name the paths it saw
    assert body["catalog_paths"] == guides.CATALOG_PATH_NOTE
    assert "get_workflow" in body["catalog_paths"]


def test_a_section_linking_nothing_carries_no_note(self, checkout):
    body = guides.get_guide("tasks", "Speech Generation")

    assert "catalog_paths" not in body


def test_an_index_holding_a_path_is_noted_too(self, checkout):
    """The index carries the guide's opening and its whole first section,
    which is as able to link an example as any other."""
    assert "catalog_paths" not in guides.get_guide("tasks")
    assert "catalog_paths" in guides.get_guide("workflows")
```

The last assertion needs the fixture's `WORKFLOW_GUIDE.md` to link one. In the
`checkout` fixture, replace:

```python
    (root / "docs" / "WORKFLOW_GUIDE.md").write_text("## Structure\n\nsteps\n")
```

with:

```python
    (root / "docs" / "WORKFLOW_GUIDE.md").write_text(
        "## Structure\n\nsteps - as in workflows/templates/text-to-image.json\n"
    )
```

And add to `class TestTheRealDocs`:

```python
    def test_the_tasks_guide_carries_the_catalog_path_note(self):
        """TASKS.md links 48 example workflows by repo path - the most of any
        served guide."""
        body = guides.get_guide("tasks", "Examples")

        assert "catalog_paths" in body
```

`## Examples` in the shipped `TASKS.md` holds 20 such paths as of this plan, so
the assertion is sound; if it has been rewritten by then, pick another section
`grep -n '^## ' docs/TASKS.md` shows that contains one, and name it in the test.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `venv/bin/python -m pytest tests/test_server_guides.py -k "catalog_path or reach_it or noted" -v`

Expected: the three positive assertions FAIL with `KeyError: 'catalog_paths'`;
`test_a_section_linking_nothing_carries_no_note` passes already and is there to
pin that the note does not attach unconditionally.

- [ ] **Step 3: Implement the note**

In `dw/server/guides.py`, add at module level beside `GUIDES`:

```python
# A doc's link to an example is a repo-relative path, which resolves for a
# reader with a checkout and dead-ends for the one these guides are served
# to: an agent holding the MCP and nothing else. Said once, on a payload
# that contains such a path, rather than rewritten into 114 links in prose
# that people read too.
CATALOG_PATH_NOTE = (
    "Paths like `workflows/templates/minimax/image-to-video.json` in this "
    "text are repo-relative and are not served by this machine. The catalog "
    "name is the part after `workflows/` with `.json` dropped - "
    "`templates/minimax/image-to-video` - which is what `get_workflow`, "
    "`validate_workflow`, `run_workflow` and a sub-workflow step's `path` "
    "all take. Paths under `dw/workflows/` are the packaged built-ins a "
    "sub-workflow names as `builtin:<file>.json`."
)

# The same shape tests/test_docs_links.py checks the targets of: the
# lookbehind keeps `workflows/` as the start of the path, so `dw/workflows/`
# and `tests/test_data/workflows/` do not match.
_EXAMPLE_PATH = re.compile(r"(?<![\w/-])(?:\.\./)?workflows/[A-Za-z0-9_.\-/]+\.json")
```

Add the helper beside `_extract_section`:

```python
def _noted(body):
    """The payload, with the repo-path rule attached when it holds one.

    Attached to the body rather than woven into the guides because the same
    markdown is read on GitHub and in an editor, where the links resolve.
    """
    if _EXAMPLE_PATH.search(body.get("content") or ""):
        body["catalog_paths"] = CATALOG_PATH_NOTE
    return body
```

`get_guide` has two return points — the index and a named section — and either
can hold a path, so wrap both. Replace the body of `get_guide` after its
docstring:

```python
    text = read_guide(name)
    if section is None:
        return _index(name, text)

    found = _extract_section(text, section)
    if found is None:
        raise GuideError(
            f"The '{name}' guide has no section '{section}'. Its sections are: "
            f"{', '.join(_sections(text))}."
        )
    heading, content = found
    return {"name": name, "section": heading, "content": content}
```

with:

```python
    text = read_guide(name)
    if section is None:
        return _noted(_index(name, text))

    found = _extract_section(text, section)
    if found is None:
        raise GuideError(
            f"The '{name}' guide has no section '{section}'. Its sections are: "
            f"{', '.join(_sections(text))}."
        )
    heading, content = found
    return _noted({"name": name, "section": heading, "content": content})
```

The route in `dw/server/app.py` returns `get_guide`'s dict as-is and needs no
change; `dw_mcp/guides.py` proxies the route and needs none either.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `venv/bin/python -m pytest tests/test_server_guides.py tests/test_mcp_guides.py tests/test_docs_links.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add dw/server/guides.py tests/test_server_guides.py
git commit -m "feat(guides): say how a repo path in a guide becomes a catalog name

The nine served guides hold 114 links to example workflows as repo-relative
paths. They resolve for a reader with a checkout; the reader these guides
exist for has the MCP and nothing else, and follows one nowhere. A section
that contains such a path now carries the translation rule, which is one
sentence rather than 114 edits to prose people also read.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: the tool surface has a budget

Connecting costs 13,225 tokens before any call: descriptions 9,013, input
schemas 3,212, instructions 999. Nothing measures it, while the *catalog*
listing — read once per session, not held for all of it — has had a documented
ceiling since #101. This task adds the guard, not a reduction; the number to
beat is the one measured here.

**Files:**
- Test: `tests/test_mcp_server.py`

**Interfaces:**
- Consumes: `tools_of(server_over(ok({})))` (`tests/test_mcp_server.py:147`).
  Each `Tool` exposes `.description` and `.input_schema` (snake case — not
  `inputSchema`).
- Produces: `SURFACE_BUDGET` in `tests/test_mcp_server.py`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_mcp_server.py`:

```python
# Chars / 4, the way COMPACT_BUDGET in tests/test_catalog_structure.py is
# measured. This is not a listing an agent chooses to read - it is resident
# for the whole session before a single call, which is why it gets a ceiling
# at all and why the catalog listing, read once, has had one since #101.
# Measured 2026-09-19 at 13_225: descriptions 9_013, input schemas 3_212,
# instructions 999. Set at 13_800, which is room for a tool or two and not
# room for a second validate_workflow.
# Worth knowing before raising it: four tools are a quarter of the
# descriptions (validate_workflow 872, wait_for_job 583, list_workflows 540,
# list_gallery 501), and validate_workflow's `plan.basis` taxonomy and
# wait_for_job's stall-diagnosis paragraph are both restated in
# WORKFLOW_GUIDE's "The loop" - which an agent fetches on demand. The
# question to ask first is whether the second copy has to be the resident one.
SURFACE_BUDGET = 13_800


@pytest.mark.asyncio
async def test_the_tool_surface_fits_the_budget():
    tools = await tools_of(server_over(ok({})))
    server = server_over(ok({}))

    descriptions = sum(len(tool.description or "") for tool in tools.values())
    schemas = sum(len(json.dumps(tool.input_schema or {})) for tool in tools.values())
    instructions = len(getattr(server, "instructions", "") or "")
    total = (descriptions + schemas + instructions) / 4

    assert total <= SURFACE_BUDGET, (
        f"the surface is {total:.0f} tokens resident before any call "
        f"(descriptions {descriptions / 4:.0f}, schemas {schemas / 4:.0f}, "
        f"instructions {instructions / 4:.0f})"
    )
```

`json` and `os` are already imported in that file, and `@pytest.mark.asyncio`
is the decorator its other async tests use. `Tool` exposes `.input_schema`, in
snake case — `.inputSchema` raises `AttributeError` under the installed
pydantic.

- [ ] **Step 2: Run it and read the measurement**

Run: `venv/bin/python -m pytest tests/test_mcp_server.py -k fits_the_budget -v -s`

Expected: PASS at roughly 13,225 tokens. This test is a guard, not a red-green
cycle — if it *fails* on the first run, Tasks 1–8 grew the surface past 13,800
and the comment's log line must be updated with the new measurement and the
reason, in the shape `COMPACT_BUDGET`'s comment uses.

- [ ] **Step 3: Confirm the whole suite is green**

Run: `venv/bin/python -m pytest tests/ -q -x`

Expected: PASS. This is the first run of the full suite since Task 1; the
changes to `dw/server/app.py`, `dw/server/catalog_shape.py` and the two JSON
manifests are the ones most likely to have a consumer this plan did not name.

- [ ] **Step 4: Commit**

```bash
git add tests/test_mcp_server.py
git commit -m "test(mcp): put a ceiling on the resident tool surface

Connecting costs 13,225 tokens before any call - 9,013 of descriptions,
3,212 of input schemas, 999 of instructions - and nothing measured it, while
the catalog listing an agent reads once has had a documented budget since
#101. A guard at 13,800 with the measurement logged, so the next tool that
lands states what it cost.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Deliberately out of scope

- **Rewriting the 114 guide links.** Task 8 states the rule once instead. The
  links resolve on GitHub and in an editor, which is the other half of their
  audience.
- **`knownIntendedModels` in `ui/src/lib/prompts.ts:87`** extends the
  suggestion list with whatever values are *in use*, which is how a variant
  spelling becomes a legitimate-looking suggestion. Task 3 removes the variant
  from the repo but not the mechanism. Worth an issue, not this plan — it is a
  UI affordance, and the filter that matters now has a swept vocabulary behind it.
- **The live workspace's prompts.** ~20 prompts on `lem` are not in this repo;
  two declare `minimax-music`. Task 3's note says to report rather than edit.
- **Shrinking any tool description.** Task 9 measures and guards. Every line in
  the four largest encodes a defect that actually landed, and deciding which
  belongs in the resident copy versus the on-demand guide is a design question,
  not a cleanup.
- **A skill for Z-Image or Flux.** The catalog's nine `models/` entries are all
  Flux and Z-Image, neither of which has a composition skill. Both are
  single-step image shapes the catalog describes adequately; a skill would be
  restating `list_workflows`.

## Deploy

Nothing here touches the worker, the engine's generation path or `ui/`, so the
box needs a pull and a server restart, not a UI build:

```bash
ssh lem 'cd ~/diffusers-workflow && git pull && pkill -KILL -f dw.serve; \
  nohup venv/bin/python -m dw.serve > ~/dw-serve.log 2>&1 & disown'
```

Then verify from a fresh shell that the two findings a client can see are
closed:

```bash
# list_prompts answers at all, and narrows
curl -sH "Authorization: Bearer $DW_API_TOKEN" \
  'http://lem:8765/api/prompts?include_text=false&intended_model=minimax-music3' \
  | venv/bin/python -c 'import json,sys; b=json.load(sys.stdin); print(len(json.dumps(b)), b["prompts"])'

# has-audio finds the song
curl -sH "Authorization: Bearer $DW_API_TOKEN" \
  'http://lem:8765/api/workflows?shape=audio&traits=has-audio&view=compact' \
  | venv/bin/python -c 'import json,sys; print(json.load(sys.stdin)["workflows"])'
```

Expected: the first prints a payload of a few kB and the four Music 3 prompt
names; the second lists `templates/minimax/music` alongside
`templates/generate-speech`.
