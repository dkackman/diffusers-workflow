# Develop Review Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the defects the 2026-10-07 review of `develop` confirmed (guide-chain cap, admission ceiling, outbound-request SSRF, cancel bookkeeping, held-audio output, chain label, `plan_cuts` gap, skill drift) without changing any behaviour the review found sound.

**Architecture:** Each fix lands where the architecture map (`docs/ARCHITECTURE.md`) already puts the rule: a validation gap is closed in the validator that owns it and pinned by the test that enforces that row; the outbound-request policy grows one shared `_safe_request` in `dw/locations.py` that GET and POST both go through; the dispatcher's cancel path gains a flag on the job record rather than a new lock. No new modules except one test file. The structural refactors the review named (h3_blocks split, task-layer consolidation, jobs.py split, VRAM accounting for guides, streaming windows) are **out of scope** here and listed at the end as follow-on plans.

**Tech Stack:** Python 3.14, pytest (+xdist; `-p no:xdist` to run one file in-process), requests 2.34 / urllib3 2.8, FastAPI server under `dw/server/`, `venv/bin/python` locally.

**Spec:** the review itself, saved as the memory note `develop-review-2026-10-07.md` and the "Confirmed defects" table in the 2026-10-07 session; the finding numbers below (#1–#9) are that table's.

## Execution model

Don chose this over the implementer/tester issue loop so one session (Fable) holds the architecture context. Per task:

- **Implementer:** a fresh subagent per task, in a worktree. Tier by task: Tasks 1, 5, 6, 7, 8 → `sonnet` (local, single-file, pattern-following). Tasks 2, 3, 4 → `opus` (cross-module, concurrency or security).
- **Reviewer:** the orchestrating session reads every diff against this plan before the next task starts; a second `opus` reviewer on Task 3 only.
- **Parallelism:** Tasks 1, 3, 5, 6, 7, 8 touch disjoint files and may run concurrently. Tasks 2 and 4 both edit `dw/server/jobs.py` → run 2 then 4, after 1/5/6/7/8 so merges stay trivial.
- **Branch:** `fix/review-2026-10-07` off `develop`; one commit per task; the orchestrator merges with `--no-ff`.

## Global Constraints

- Every test file runs with `venv/bin/python -m pytest <file> -q -p no:xdist` while iterating; the whole suite (`venv/bin/python -m pytest tests/ -q`) must stay at 0 failures before each commit.
- Filesystem/URL/subprocess access goes through `dw/security.py` / `dw/locations.py` validators (CLAUDE.md "Security Rules"); never `shell=True`, `eval`, `exec`.
- No model-family knowledge is added to engine code; this plan touches H3 only in modules the map already names as H3-specific.
- Docstrings follow the repo's voice: a sentence that says *why*, not a restatement of the signature; comments explain a non-obvious choice. Match surrounding density.
- `docs/ARCHITECTURE.md` rows must name real modules and real tests; `tests/test_architecture_map.py` checks that.
- The minimax-h3 skill is at 11,866 of 12,288 bytes; Task 8 adds a test, not skill text.
- No `TBD`/`TODO` in shipped code. No new runtime dependencies.

## Review Focus

Inputs the spec implies but that no task's tests exercise, most likely to bite first. Each got a test added to the owning task.

1. A workflow whose `guides` is itself an unresolved `variable:` reference, plus a guide chain — validation must stay silent and leave it to the run (Task 1, `test_an_unresolved_guides_value_is_left_to_the_run`).
2. A single-card server whose worker has not yet reported its capacity (lazy start) — admission must fall back to the process device's capacity, not refuse everything (Task 2, `test_an_unread_pool_falls_back_to_the_device`).
3. A media URL whose host does not resolve — must still fail as a fetch error, not a policy refusal, and must not crash the pinning code (Task 3, `test_an_unresolvable_host_is_fetched_unpinned`).
4. `--trust-workflows` — the pinning and the byte cap must not change what a trusted operator can fetch from their own LAN (Task 3, `test_trust_lifts_pinning_but_keeps_the_cap`).
5. A cancel that races the worker's *own* end-of-job — `cancel_requested` set after `_consume_results` returned must not turn a `succeeded` into a `cancelled` (Task 4, `test_a_cancel_after_the_job_finished_is_a_no_op`).

---

### Task 1: A guide chain counts the guide it adds (#1)

**Files:**
- Modify: `dw/guides.py:226-271` (`guide_chain_errors`)
- Modify: `docs/WORKFLOW_GUIDE.md:1731-1752` (one sentence)
- Test: `tests/test_guide_chain_validation.py`

**Interfaces:**
- Consumes: `GUIDE_LIMIT`, `GUIDES_INPUT`, `GUIDE_CHAIN_RULE`, `GUIDE_CONTINUITY` from `dw.pipeline_processors.h3_blocks` (already imported in `dw/guides.py`); `references.is_ref(references.UNRESOLVED, value)`.
- Produces: nothing new; `guide_chain_errors(workflow_definition, source_indices=None) -> [{path, message}]` unchanged in shape.

Context: `dw/pipeline_processors/chain.py:189` appends one guide at frame 0 to the step's own `guides` for every segment after the first; `dw/pipeline_processors/pipeline.py:688` then refuses more than `GUIDE_LIMIT` (4). `guides_errors` checks only the written list, so 4 own guides + a guide chain validate clean and fail on segment 2.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_guide_chain_validation.py`:

```python
class TestOwnGuidesPlusTheChain:
    """The chain adds a guide of its own at frame 0 (chain.py GuideContinuity),
    so a step's written guides have one fewer slot than GUIDE_LIMIT."""

    @staticmethod
    def guides(count):
        return [{"video": f"g{i}.mp4", "frame": 17 * (i + 1)} for i in range(count)]

    def test_four_own_guides_leave_no_room_for_the_chain_guide(self):
        error = one(definition(prompt="a cat", guides=self.guides(4)))
        assert error["path"].endswith("continuity")
        assert "the chain adds one" in error["message"]
        assert "4" in error["message"]

    def test_three_own_guides_leave_room(self):
        assert guide_chain_errors(definition(prompt="a cat", guides=self.guides(3))) == []

    def test_an_unresolved_guides_value_is_left_to_the_run(self):
        assert (
            guide_chain_errors(definition(prompt="a cat", guides="variable:guides"))
            == []
        )

    def test_last_frame_continuity_takes_the_full_four(self):
        chain = {"segments": 2, "continuity": "last_frame"}
        assert (
            guide_chain_errors(definition(chain=chain, prompt="a cat", guides=self.guides(4)))
            == []
        )
```

- [ ] **Step 2: Run them to verify they fail**

Run: `venv/bin/python -m pytest tests/test_guide_chain_validation.py -q -p no:xdist -k OwnGuides`
Expected: `test_four_own_guides_leave_no_room_for_the_chain_guide` FAILS (`assert len(errors) == 1` sees 0); the other three pass already.

- [ ] **Step 3: Count the step's own guides in `guide_chain_errors`**

In `dw/guides.py`, inside the `for index, step in enumerate(steps):` loop of `guide_chain_errors`, after the `_takes_guides_problem` block and before `for key, message in guide_chain_problems(chain):`, add:

```python
        # The chain adds a guide of its own at frame 0 for every segment
        # after the first (chain.py GuideContinuity), and the run's cap
        # counts it - so the step's written guides get one slot fewer. An
        # unresolved value is the run's to count.
        arguments = pipeline.get("arguments")
        own = arguments.get(GUIDES_INPUT) if isinstance(arguments, dict) else None
        if isinstance(own, (list, tuple)) and len(own) + 1 > GUIDE_LIMIT:
            errors.append(
                {
                    "path": render_path(base + ("chain", "continuity")),
                    "message": (
                        f"{GUIDE_CHAIN_RULE}: guides takes at most {GUIDE_LIMIT} "
                        f"clips and the chain adds one of its own, so a guide "
                        f"chain leaves room for {GUIDE_LIMIT - 1} - got {len(own)}"
                    ),
                }
            )
```

- [ ] **Step 4: Run the file to verify it passes**

Run: `venv/bin/python -m pytest tests/test_guide_chain_validation.py tests/test_guides_validation.py -q -p no:xdist`
Expected: all PASS.

- [ ] **Step 5: Document the slot in the guide**

In `docs/WORKFLOW_GUIDE.md`, in the section `#### Chaining with a guide: \`continuity: "guide"\`` (line ~1731), after the sentence that introduces `guide_frames`, add:

```markdown
The chain's carried clip is itself a guide, so a step that also writes its own
`guides` may list at most three of them with `continuity: "guide"` - validation
says so before the run rather than after the first segment.
```

- [ ] **Step 6: Commit**

```bash
git add dw/guides.py tests/test_guide_chain_validation.py docs/WORKFLOW_GUIDE.md
git commit -m "fix(guides): a guide chain counts the guide it adds against GUIDE_LIMIT

Four written guides plus continuity: \"guide\" validated clean and failed
on segment 2 after a full H3 segment of GPU time (review 2026-10-07 #1).

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: Admission holds a declared VRAM need to the pool's largest card (#2)

**Files:**
- Modify: `dw/validation.py:656-690` (`workflow_context`)
- Modify: `dw/server/admission.py:100-115, 173-175, 378-388` (`admit`, `admit_for`)
- Modify: `dw/server/jobs.py:323-331` (add `largest_ceiling_gb` beside `_capacities`)
- Modify: `docs/ARCHITECTURE.md:145` ("Worker pool dispatch" row)
- Test: `tests/test_worker_pool.py` (class `TestFitAtSubmit`), create `tests/test_admission_capacity.py`

**Interfaces:**
- Produces: `validation.workflow_context(workflow, arguments=None, composing=(), ceiling_index=None, capacity_gb=None)` — `capacity_gb=None` means "this process's device", as today.
- Produces: `admission.admit(..., capacity_gb=None)` keyword; `admission.pool_capacity_gb(state) -> float | None`.
- Produces: `JobManager.largest_ceiling_gb() -> float | None` — the largest `ceiling_gb()` among slots whose size is readable, else `None`.

Context: `workflow_context` sets `capacity_gb=device_capacity_gb()` — the process default device, which `dw/serve.py:192` pins to `devices[0]`. `vram_estimate._entries_for` then refuses a declared `vram_estimate` above card 0 even when card 1 is larger; `JobManager._unfit_message` (which does judge against the largest card) never runs because admission already said no.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_admission_capacity.py`:

```python
"""Review 2026-10-07 #2: a declared vram_estimate is held to the pool's
largest card, not the process's default device (card 0 under --devices)."""

from types import SimpleNamespace

from dw.server import admission
from dw.validation import workflow_context
from dw.workflow import workflow_from_definition


def empty_workflow(tmp_path):
    return workflow_from_definition({"id": "w", "steps": []}, str(tmp_path / "out"))


def test_the_context_defaults_to_the_devices_capacity(tmp_path, monkeypatch):
    monkeypatch.setattr("dw.validation.device_capacity_gb", lambda: 12.0)
    assert workflow_context(empty_workflow(tmp_path)).capacity_gb == 12.0


def test_the_context_takes_a_pool_ceiling_instead(tmp_path, monkeypatch):
    monkeypatch.setattr("dw.validation.device_capacity_gb", lambda: 12.0)
    assert workflow_context(empty_workflow(tmp_path), capacity_gb=24.0).capacity_gb == 24.0


def _patched_admit_for(monkeypatch, state):
    seen = {}
    monkeypatch.setattr(admission, "admit", lambda **kw: seen.update(kw))
    monkeypatch.setattr(admission, "ceiling_index", lambda state, workspace: None)
    monkeypatch.setattr(admission, "resolution_library", lambda state, workspace: None)
    monkeypatch.setattr(admission, "server_prompt_library", lambda state: None)
    admission.admit_for(state, "workspace")
    return seen


def test_admit_for_hands_the_pools_largest_ceiling_to_admit(monkeypatch):
    manager = SimpleNamespace(largest_ceiling_gb=lambda: 24.0)
    seen = _patched_admit_for(monkeypatch, SimpleNamespace(job_manager=manager))
    assert seen["capacity_gb"] == 24.0


def test_an_unread_pool_falls_back_to_the_device(monkeypatch):
    """A lazily started worker has not read its card yet: None here means
    workflow_context's own default, not a refusal of everything."""
    manager = SimpleNamespace(largest_ceiling_gb=lambda: None)
    seen = _patched_admit_for(monkeypatch, SimpleNamespace(job_manager=manager))
    assert seen["capacity_gb"] is None


def test_a_state_without_a_manager_falls_back_to_the_device(monkeypatch):
    seen = _patched_admit_for(monkeypatch, SimpleNamespace())
    assert seen["capacity_gb"] is None
```

Append to `class TestFitAtSubmit` in `tests/test_worker_pool.py`:

```python
    def test_the_largest_card_is_the_ceiling_admission_checks(self, pool):
        manager = pool(
            card(success_script, "cuda:0", 12, name="Small GPU"),
            card(success_script, "cuda:1", 24, name="Large GPU"),
        )
        assert manager.largest_ceiling_gb() == 24

    def test_an_unread_card_has_no_ceiling(self, pool):
        worker = ScriptedWorkerManager(success_script)
        worker.device = "cuda:0"
        manager = pool(worker)
        assert manager.largest_ceiling_gb() is None
```

Note for the implementer: `card()` sets `_capacity_read = True` and `_capacity_gb`; a bare `ScriptedWorkerManager` leaves them unread, and `ceiling_gb()` falls back to `capacity_gb()`, which is `None` until read. If `ScriptedWorkerManager` reads a real device on first call, patch `dw.worker_manager.device_capacity_gb` to return `None` in the second test, as `test_a_declared_need_is_held_to_the_ceiling_admission_uses` patches it.

- [ ] **Step 2: Run them to verify they fail**

Run: `venv/bin/python -m pytest tests/test_admission_capacity.py tests/test_worker_pool.py -q -p no:xdist -k "capacity or largest or unread"`
Expected: `test_the_context_takes_a_pool_ceiling_instead` fails with `TypeError: unexpected keyword 'capacity_gb'`; the `admit_for` tests fail with `KeyError: 'capacity_gb'`; the pool tests fail with `AttributeError: largest_ceiling_gb`.

- [ ] **Step 3: Thread `capacity_gb` through `workflow_context`**

In `dw/validation.py`, change the signature and the one line:

```python
def workflow_context(
    workflow, arguments=None, composing=(), ceiling_index=None, capacity_gb=None
):
```

Add to the docstring, after "the device the request is checked against": `(its own capacity, or \`capacity_gb\` when the caller knows a larger card the job may land on - the server's worker pool)`.

Replace `capacity_gb=device_capacity_gb(),` with:

```python
        capacity_gb=device_capacity_gb() if capacity_gb is None else capacity_gb,
```

- [ ] **Step 4: Add `JobManager.largest_ceiling_gb`**

In `dw/server/jobs.py`, directly after `_capacities`:

```python
    def largest_ceiling_gb(self):
        """The most VRAM any card here can be held to (WorkerSlot.ceiling_gb),
        or None when no card has reported its size yet. Admission checks a
        declared vram_estimate against this rather than the process's own
        device, which under --devices is only the first card."""
        capacities = self._capacities(hard=True)
        return max(capacity for _, capacity in capacities) if capacities else None
```

- [ ] **Step 5: Thread it through `admit` and `admit_for`**

In `dw/server/admission.py`, add `capacity_gb=None,` to `admit`'s keyword list after `plan_for=None,`, and change the `workflow_context` call at line ~173 to:

```python
            context = validation.workflow_context(
                candidate, checked, ceiling_index=ceiling_index, capacity_gb=capacity_gb
            )
```

Add before `admit_for`:

```python
def pool_capacity_gb(state):
    """The pool's largest card, for a declared vram_estimate's ceiling - or
    None, which leaves workflow_context on the process's own device: a
    state without a manager (validate-only embeddings) or a pool whose
    workers have not read their cards yet."""
    manager = getattr(state, "job_manager", None)
    if manager is None or not hasattr(manager, "largest_ceiling_gb"):
        return None
    return manager.largest_ceiling_gb()
```

And in `admit_for` add `capacity_gb=pool_capacity_gb(state),` to the `admit(...)` call.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `venv/bin/python -m pytest tests/test_admission_capacity.py tests/test_worker_pool.py tests/test_vram_estimate.py tests/test_server.py -q -p no:xdist`
Expected: all PASS - nothing that patches `dw.validation.device_capacity_gb` changes behaviour, since `None` still means that call.

- [ ] **Step 7: Update the architecture map**

In `docs/ARCHITECTURE.md`, the row beginning `| Worker pool dispatch |`: append to its module column `, \`largest_ceiling_gb\`; \`dw/server/admission.py\`: \`pool_capacity_gb\``, and to its rule column the sentence `A declared \`vram_estimate\` is held at admission to the pool's largest card, so a job the first card cannot hold is not refused for a larger one.` Add to its test column `, \`tests/test_admission_capacity.py::test_admit_for_hands_the_pools_largest_ceiling_to_admit\``. Then run `venv/bin/python -m pytest tests/test_architecture_map.py -q -p no:xdist` — PASS.

- [ ] **Step 8: Commit**

```bash
git add dw/validation.py dw/server/admission.py dw/server/jobs.py docs/ARCHITECTURE.md tests/test_admission_capacity.py tests/test_worker_pool.py
git commit -m "fix(server): admission holds a vram_estimate to the pool's largest card

Under --devices the process device is card 0, so a job that fit only the
larger card was refused at submit before the fit check that knows the
pool could run (review 2026-10-07 #2).

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: One outbound-request path - pinned address, byte cap, timeout, POST too (#3, #4)

**Files:**
- Modify: `dw/locations.py:377-418` (`MAX_MEDIA_REDIRECTS`, `safe_get`) and add `MAX_MEDIA_BYTES`, `_PinnedAdapter`, `_pinned_session`, `_read_capped`, `_safe_request`, `safe_post`
- Modify: `dw/pipeline_processors/remote.py:1-45`
- Modify: `docs/SECURITY.md:168-180` (the "Every redirect is re-checked" bullet), `docs/ARCHITECTURE.md:131` ("Locations from a workflow" row)
- Test: `tests/test_locations.py` (modify `test_an_untrusted_run_withholds_the_token_from_a_third_party`; add `class TestSafeRequest`)

**Interfaces:**
- Produces: `safe_get(url, what="a media argument", timeout=60, max_bytes=MAX_MEDIA_BYTES) -> requests.Response` (same contract as today: `.content` is the whole body, status already checked).
- Produces: `safe_post(url, what, timeout=120, max_bytes=MAX_MEDIA_BYTES, validate=validate_media_url, **kwargs) -> requests.Response` where `kwargs` are passed to `Session.request` (`json=`, `headers=`).
- Produces: `MAX_MEDIA_BYTES = 1024**3` (1 GiB - Don's call; a URL-fetched video asset larger than this is refused, and `upload_asset` is the route for one).
- Consumes: `_resolved_addresses`, `_is_internal`, `validate_media_url`, `validate_remote_encoder_url`, `workflows_are_trusted` (all in `dw/locations.py` today).

Context: `validate_media_url` resolves the host and refuses internal addresses, then `requests` resolves it *again* when it dials (DNS rebinding); `safe_get` reads the whole body into memory with no cap; `remote.py` POSTs with `requests.post` directly - redirects followed unchecked, no timeout. The design, prototyped 2026-10-07 against a local HTTP server and `https://example.com`: a `requests` adapter that dials the address the policy validated while keeping TLS SNI and certificate checks on the hostname (`server_hostname` / `assert_hostname` pool kwargs; the unpinned IP-literal fetch fails its cert check, proving the kwargs are honoured).

- [ ] **Step 1: Write the failing tests**

In `tests/test_locations.py` add the imports `import http.server`, `import socketserver`, `import threading` at the top (keep the existing ones), then append:

```python
class _EchoHandler(http.server.BaseHTTPRequestHandler):
    """/ok answers 2 bytes; /host echoes the Host header; /big answers 1000
    bytes; /nolength streams 1000 bytes with no Content-Length; /hop
    redirects to /ok; /loop redirects to itself; /internal redirects to
    127.0.0.1."""

    def do_GET(self):
        self._answer()

    def do_POST(self):
        self._answer()

    def _answer(self):
        path = self.path
        if path.startswith("/hop"):
            return self._redirect(f"http://{self.headers['Host']}/ok")
        if path.startswith("/loop"):
            return self._redirect(f"http://{self.headers['Host']}/loop")
        if path.startswith("/internal"):
            return self._redirect("http://127.0.0.1:1/x")
        if path.startswith("/host"):
            body = self.headers["Host"].encode()
        elif path.startswith("/big") or path.startswith("/nolength"):
            body = b"x" * 1000
        else:
            body = b"ok"
        self.send_response(200)
        if not path.startswith("/nolength"):
            self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _redirect(self, target):
        self.send_response(302)
        self.send_header("Location", target)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *args):
        pass


@pytest.fixture
def local_server():
    server = socketserver.TCPServer(("127.0.0.1", 0), _EchoHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def _public(monkeypatch):
    """Every name resolves to a public address for the policy; the pin
    then decides what is actually dialed."""
    monkeypatch.setattr(
        "dw.locations.socket.getaddrinfo",
        lambda host, *a, **k: [(None, None, None, "", ("93.184.216.34", 80))],
    )


class TestSafeRequest:
    """Review 2026-10-07 #3/#4: one path for every outbound request - the
    address dialed is the one the policy checked, the body is capped, and
    POST goes the same way as GET."""

    def test_the_pin_dials_the_validated_address_not_a_second_lookup(
        self, trusted, local_server
    ):
        from dw.locations import _pinned_session

        host, port = local_server.split(":")
        url = f"http://pinned.example:{port}/host"
        # pinned.example resolves nowhere; only the pin can reach the server
        response = _pinned_session(url, host).get(url, timeout=5, allow_redirects=False)
        assert response.content == f"pinned.example:{port}".encode()

    def test_trust_lifts_pinning_but_keeps_the_cap(self, trusted, local_server):
        from dw.locations import safe_get

        assert safe_get(f"http://{local_server}/ok", timeout=5).content == b"ok"
        with pytest.raises(InvalidInputError, match="larger than"):
            safe_get(f"http://{local_server}/big", timeout=5, max_bytes=100)
        with pytest.raises(InvalidInputError, match="larger than"):
            safe_get(f"http://{local_server}/nolength", timeout=5, max_bytes=100)

    def test_a_declared_length_over_the_cap_is_refused_before_the_body(
        self, trusted, local_server, monkeypatch
    ):
        from dw import locations

        read = []
        monkeypatch.setattr(
            locations, "_read_capped", lambda *a, **k: read.append(a) or b""
        )
        with pytest.raises(InvalidInputError, match="larger than"):
            locations.safe_get(f"http://{local_server}/big", timeout=5, max_bytes=100)
        assert read == []

    def test_a_redirect_to_an_internal_address_is_refused(
        self, untrusted, local_server, monkeypatch
    ):
        """The first hop is a public name (pinned here to the local server
        so the test can answer it); its 302 names 127.0.0.1, which the
        policy refuses before anything is dialed."""
        from dw import locations

        _public(monkeypatch)
        host, port = local_server.split(":")
        real_pinned = locations._pinned_session
        monkeypatch.setattr(
            locations, "_pinned_session", lambda url, address: real_pinned(url, host)
        )
        with pytest.raises(InvalidInputError, match="inside this deployment"):
            locations.safe_get(f"http://public.example:{port}/internal", timeout=5)

    def test_a_redirect_loop_is_refused(self, trusted, local_server):
        from dw.locations import MAX_MEDIA_REDIRECTS, safe_get

        with pytest.raises(InvalidInputError, match=f"{MAX_MEDIA_REDIRECTS} times"):
            safe_get(f"http://{local_server}/loop", timeout=5)

    def test_a_redirect_is_followed_to_a_good_target(self, trusted, local_server):
        from dw.locations import safe_get

        assert safe_get(f"http://{local_server}/hop", timeout=5).content == b"ok"

    def test_post_goes_through_the_same_path(self, trusted, local_server):
        from dw.locations import safe_post

        response = safe_post(
            f"http://{local_server}/ok", "a test endpoint", timeout=5, json={"a": 1}
        )
        assert response.content == b"ok"

    def test_an_unresolvable_host_is_fetched_unpinned(self, untrusted, monkeypatch):
        """A typo stays the fetch's error to report (TestHostPolicy), and
        the pin must not crash on an empty resolution."""
        import socket

        from dw import locations

        monkeypatch.setattr(
            "dw.locations.socket.getaddrinfo", Mock(side_effect=socket.gaierror)
        )
        dialed = {}

        class _Session:
            def request(self, method, url, **kwargs):
                dialed["url"] = url
                raise requests.ConnectionError("no such host")

            def close(self):
                pass

        def _never_pinned(url, address):
            raise AssertionError("an unresolved name must not be pinned")

        monkeypatch.setattr(locations, "_pinned_session", _never_pinned)
        monkeypatch.setattr(locations, "_plain_session", lambda: _Session())
        with pytest.raises(requests.ConnectionError):
            locations.safe_get("https://nope.invalid/x.png", timeout=5)
        assert dialed["url"] == "https://nope.invalid/x.png"
```

`requests` and `Mock` must be imported at the top of the test file if they are not already (`import requests`; `from unittest.mock import Mock, patch`).

Then change `test_an_untrusted_run_withholds_the_token_from_a_third_party` so it patches the shared path instead of `remote.requests.post`. Replace its `_Response`/`_post`/`with` block with:

```python
        sent = {}

        class _Response:
            ok = True
            is_redirect = False
            headers = {"Content-Type": "application/octet-stream"}
            status_code = 200
            _content = b""

            def raise_for_status(self):
                pass

            def iter_content(self, chunk_size):
                return iter([b""])

            def close(self):
                pass

        class _Session:
            def request(self, method, url, **kwargs):
                sent["method"] = method
                sent["headers"] = kwargs.get("headers")
                sent["timeout"] = kwargs.get("timeout")
                return _Response()

            def close(self):
                pass

        with (
            patch("dw.locations._pinned_session", lambda url, address: _Session()),
            patch.object(remote.torch, "load", return_value=_Embeds()),
            patch(
                "dw.locations.socket.getaddrinfo",
                return_value=[(None, None, None, "", ("93.184.216.34", 443))],
            ),
        ):
            remote.remote_text_encoder(["a"], "https://evil.example.com/encode", "cpu")

        assert "Authorization" not in sent["headers"]
        assert sent["method"] == "POST"
        assert sent["timeout"] == remote.REMOTE_ENCODER_TIMEOUT
```

- [ ] **Step 2: Run them to verify they fail**

Run: `venv/bin/python -m pytest tests/test_locations.py -q -p no:xdist -k "SafeRequest or withholds"`
Expected: `ImportError`/`AttributeError` on `_pinned_session`, `safe_post`, `_read_capped`, `REMOTE_ENCODER_TIMEOUT`; `test_trust_lifts_pinning_but_keeps_the_cap` fails on the missing `max_bytes` keyword.

- [ ] **Step 3: Implement the shared request path in `dw/locations.py`**

Add `from urllib.parse import urlsplit, urlunsplit` beside the existing `urlparse` import if not present, and `import requests` at module level is **not** wanted (the module imports it lazily inside `safe_get` today - keep that: do `import requests` and `from requests.adapters import HTTPAdapter` inside the functions that need them, or add a lazy module-level accessor). Replace everything from `MAX_MEDIA_REDIRECTS = 5` through the end of `safe_get` with:

```python
# How many redirects a media fetch follows before giving up. requests' own
# default is 30; a CDN needs one or two
MAX_MEDIA_REDIRECTS = 5

# The most a fetched body may hold - the worker reads it whole into host RAM
# before the media loader sees it, and a workflow names the URL
MAX_MEDIA_BYTES = 1024**3

_READ_CHUNK = 1 << 16


def _pinned_adapter_class():
    from requests.adapters import HTTPAdapter

    class _PinnedAdapter(HTTPAdapter):
        """Dials `address` for every request to `host`, with TLS still
        negotiated and checked against `host`. The address is the one the
        host policy resolved and passed, so a name that answers a public
        address to the lookup and an internal one to the connect (DNS
        rebinding) reaches only the address that was checked."""

        def __init__(self, host, address, **kwargs):
            super().__init__(**kwargs)
            self.host = host
            self.address = address

        def _pinned(self, url):
            parts = urlsplit(url)
            literal = f"[{self.address}]" if ":" in self.address else self.address
            netloc = literal + (f":{parts.port}" if parts.port else "")
            return urlunsplit(parts._replace(netloc=netloc))

        def build_connection_pool_key_attributes(self, request, verify, cert=None):
            host_params, pool_kwargs = super().build_connection_pool_key_attributes(
                request, verify, cert
            )
            if request.url.lower().startswith("https"):
                pool_kwargs["server_hostname"] = self.host
                pool_kwargs["assert_hostname"] = self.host
            return host_params, pool_kwargs

        def send(self, request, **kwargs):
            request.headers.setdefault("Host", self.host)
            request.url = self._pinned(request.url)
            return super().send(request, **kwargs)

    return _PinnedAdapter


def _pinned_session(url, address):
    """A requests Session that dials `address` for `url`'s host."""
    import requests

    session = requests.Session()
    host = (urlsplit(url).hostname or "").strip("[]")
    session.mount(url, _pinned_adapter_class()(host, address))
    return session


def _plain_session():
    """A requests Session that resolves for itself - the one a trusted run
    or an unresolved name is fetched with. A seam the tests replace."""
    import requests

    return requests.Session()


def _session_for(url):
    """The session a validated URL is fetched with: pinned to the first
    address the policy resolved, or plain when the policy is lifted
    (--trust-workflows) or the name resolved to nothing (the fetch reports
    that itself)."""
    if workflows_are_trusted():
        return _plain_session()
    host = (urlsplit(url).hostname or "").strip("[]")
    addresses = _resolved_addresses(host)
    if not addresses:
        return _plain_session()
    return _pinned_session(url, str(addresses[0]))


def _validated(validate, target, what):
    """`validate` applied the way each policy takes its arguments:
    validate_media_url names the argument for its message, and
    validate_remote_encoder_url takes the URL alone."""
    if validate is validate_media_url:
        return validate_media_url(target, what)
    return validate(target)


def _read_capped(response, max_bytes, what, url):
    """The body, read in chunks and refused past `max_bytes`."""
    chunks, size = [], 0
    for chunk in response.iter_content(_READ_CHUNK):
        size += len(chunk)
        if size > max_bytes:
            response.close()
            raise InvalidInputError(
                f"Refusing {what} from '{url}': larger than {max_bytes} bytes"
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _safe_request(
    method, url, what, timeout, validate=validate_media_url, max_bytes=MAX_MEDIA_BYTES, **kwargs
):
    """One outbound request on the workflow's behalf: `validate` on the URL
    and on every redirect target, each hop dialed at the address the policy
    resolved, the body capped at `max_bytes`, a timeout always set.

    Returns the final requests.Response with its `.content` read, status
    already checked. Raises InvalidInputError for a refused hop, a redirect
    chain past MAX_MEDIA_REDIRECTS, or a body over the cap, and
    requests.HTTPError for an error status.
    """
    current = _validated(validate, url, what)
    for _ in range(MAX_MEDIA_REDIRECTS + 1):
        session = _session_for(current)
        try:
            response = session.request(
                method,
                current,
                timeout=timeout,
                allow_redirects=False,
                stream=True,
                **kwargs,
            )
            if response.is_redirect:
                target = urljoin(current, response.headers["Location"])
                response.close()
                logger.debug(f"{current} redirects to {target}")
                current = _validated(
                    validate, target, f"{what} (redirected from '{url}')"
                )
                continue
            response.raise_for_status()
            declared = response.headers.get("Content-Length")
            if declared and declared.isdigit() and int(declared) > max_bytes:
                response.close()
                raise InvalidInputError(
                    f"Refusing {what} from '{url}': larger than {max_bytes} bytes "
                    f"({declared} declared)"
                )
            response._content = _read_capped(response, max_bytes, what, url)
            response._content_consumed = True
            return response
        finally:
            session.close()
    raise InvalidInputError(
        f"Refusing to fetch {what} from '{url}': it redirects more than "
        f"{MAX_MEDIA_REDIRECTS} times"
    )


def safe_get(url, what="a media argument", timeout=60, max_bytes=MAX_MEDIA_BYTES):
    """GET a media URL for a workflow (`_safe_request`)."""
    return _safe_request("GET", url, what, timeout, max_bytes=max_bytes)


def safe_post(
    url, what, timeout=120, max_bytes=MAX_MEDIA_BYTES, validate=validate_media_url, **kwargs
):
    """POST on a workflow's behalf, the way safe_get fetches: `kwargs` are
    requests' own (`json=`, `headers=`)."""
    return _safe_request(
        "POST", url, what, timeout, validate=validate, max_bytes=max_bytes, **kwargs
    )
```

`InvalidInputError`, `urljoin`, `logger`, `workflows_are_trusted`, `_resolved_addresses` already exist in the module. `urlsplit`/`urlunsplit` join the existing `from urllib.parse import ...` line. `requests` stays a function-local import, as `safe_get` has it today.

- [ ] **Step 4: Route the remote text encoder through `safe_post`**

Replace `dw/pipeline_processors/remote.py` lines 1–45 with:

```python
import io
import logging
from urllib.parse import urlparse

import requests
import torch
from huggingface_hub import get_token

from ..locations import (
    HF_TOKEN_HOST_SUFFIXES,
    safe_post,
    token_host_allowed,
    validate_remote_encoder_url,
)
from ..trust import workflows_are_trusted

logger = logging.getLogger("dw")

# A text encoder answering a long prompt list takes seconds, not minutes;
# without a bound a dead endpoint holds the card's worker forever
REMOTE_ENCODER_TIMEOUT = 120


def remote_text_encoder(prompts, url, device):
    url = validate_remote_encoder_url(url)
    headers = {"Content-Type": "application/json"}
    host = urlparse(url).hostname
    if token_host_allowed(host) or workflows_are_trusted():
        headers["Authorization"] = f"Bearer {get_token()}"
    else:
        logger.warning(
            f"Not sending the HuggingFace token to {host}: it is outside "
            f"{', '.join(HF_TOKEN_HOST_SUFFIXES)}. If the endpoint needs the "
            f"token, run with --trust-workflows."
        )

    # Every redirect re-validated and dialed at the checked address, like a
    # media fetch - a 307 to an internal address otherwise re-sends this
    # POST, token and all, inside the deployment
    try:
        response = safe_post(
            url,
            "the remote text encoder",
            timeout=REMOTE_ENCODER_TIMEOUT,
            validate=validate_remote_encoder_url,
            json={"prompt": prompts},
            headers=headers,
        )
    except requests.HTTPError as e:
        # An error status is still answered below, by its status and type
        response = e.response
    content_type = response.headers.get("Content-Type", "")
    # An endpoint that has moved or been retired answers with an HTML page,
    # and torch.load's unpickling error about it names nothing a reader
    # could act on
    if not response.ok or "text/html" in content_type:
        raise RuntimeError(
            f"The remote text encoder at {url} did not return embeddings "
            f"(HTTP {response.status_code}, {content_type or 'no content type'}). "
            "The endpoint may have moved or been retired; drop "
            "'remote_text_encoder' to load the text encoder locally."
        )
    prompt_embeds = torch.load(io.BytesIO(response.content), weights_only=True)

    return prompt_embeds.to(device)
```

The `except` keeps today's behaviour for an error status: `_safe_request` raises `requests.HTTPError` (which carries `.response`), and the message below still names the status. `weights_only=True` is made explicit (it is already the default on the pinned torch).

- [ ] **Step 5: Run the tests to verify they pass**

Run: `venv/bin/python -m pytest tests/test_locations.py tests/test_arguments.py tests/test_security.py -q -p no:xdist`
Expected: all PASS. `tests/test_arguments.py` patches `dw.argument_media.safe_get` and must be untouched by this change.

- [ ] **Step 6: Manual TLS check (not a unit test)**

Run once, from the repo, and paste the output into the commit body:

```bash
venv/bin/python - <<'EOF'
from dw.locations import safe_get
r = safe_get("https://example.com/", "a check", timeout=15)
print(r.status_code, len(r.content))
EOF
```

Expected: `200 <some bytes>` - the pinned HTTPS dial negotiates SNI and verifies the certificate for `example.com`. (The prototype also showed that the same fetch with the IP literal and *no* pin fails its certificate check, which is what proves the pool kwargs are honoured.)

- [ ] **Step 7: Docs**

`docs/SECURITY.md`, the bullet beginning `- **Every redirect is re-checked.**` (line ~168): append `Each hop is dialed at the address the policy resolved, so a name that answers differently to the lookup and the connect (DNS rebinding) reaches only what was checked; the body is capped at \`MAX_MEDIA_BYTES\` (1 GiB); and \`remote_text_encoder\` POSTs through the same path with a timeout.`

`docs/ARCHITECTURE.md`, the row beginning `| Locations from a workflow |`: append to the rule column `Every outbound request (\`safe_get\`, \`safe_post\`) is validated per hop, dialed at the resolved address, capped and timed.`; add to the test column `, \`tests/test_locations.py::TestSafeRequest::test_the_pin_dials_the_validated_address_not_a_second_lookup\``. Run `venv/bin/python -m pytest tests/test_architecture_map.py -q -p no:xdist` - PASS.

- [ ] **Step 8: Commit**

```bash
git add dw/locations.py dw/pipeline_processors/remote.py docs/SECURITY.md docs/ARCHITECTURE.md tests/test_locations.py
git commit -m "fix(security): one outbound-request path - pinned address, byte cap, POST too

safe_get checked the host at lookup and let requests resolve it again at
the dial; the body was read whole with no cap; remote_text_encoder POSTed
with requests.post, following redirects unchecked and with no timeout
(review 2026-10-07 #3/#4). Both pre-date develop.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: A cancel always lands, and a cancelled queued job leaves nothing behind (#5, #6)

**Files:**
- Modify: `dw/server/job_record.py:53-60` (`Job.__init__`)
- Modify: `dw/server/jobs.py:782-803` (`cancel`), `943-985` (`_run_on`)
- Test: `tests/test_worker_pool.py` (class `TestCancel`)

**Interfaces:**
- Produces: `Job.cancel_requested: bool` (default `False`), not persisted.
- Consumes: `JobManager.slots[i].lock`, `Gate`, `card`, `pool`, `submit`, `wait_until`, `wait_for`, `success_script` from the test module.

Context: `cancel()` on a RUNNING job calls `slot.manager.cancel()` immediately. The dispatcher marks RUNNING *before* the job's thread takes `slot.lock`, starts the worker and sends `Execute`; a probe can hold `slot.lock` for seconds. In that window `manager.cancel()` either raises ("not active", logged and swallowed) or reaches an idle worker that ignores it (`dw/worker.py:200`), and the job then runs to completion though the caller was told the cancel was taken. Separately, `cancel()` of a QUEUED job removes `_pending` but not `_needs`/`_preferred`.

- [ ] **Step 1: Write the failing tests**

Append to `class TestCancel` in `tests/test_worker_pool.py`:

```python
    def test_a_cancel_before_the_execute_is_sent_still_lands(self, pool):
        """The dispatcher marks RUNNING before the job's thread takes the
        slot; a cancel in that window used to reach an idle worker, which
        ignores it, and the job ran anyway."""
        gate = Gate()
        manager = pool(card(gate, "cuda:0", 24))
        slot = manager.slots[0]
        with slot.lock:  # a cache probe holds the card between dispatch and Execute
            job = submit(manager, "late-cancel")
            assert wait_until(lambda: job.status == "running")
            assert manager.cancel(job.id) == "running"
        wait_for(job, {"cancelled"})
        assert not gate.started.is_set(), "the worker was still sent the job"

    def test_a_cancel_after_the_job_finished_is_a_no_op(self, pool):
        manager = pool(card(success_script, "cuda:0", 24))
        job = submit(manager, "done")
        wait_for(job, {"succeeded"})
        assert manager.cancel(job.id) == "succeeded"
        assert job.status == "succeeded"

    def test_cancelling_a_queued_job_drops_its_dispatch_bookkeeping(self, pool):
        gate = Gate()
        manager = pool(card(gate, "cuda:0", 24))
        first = submit(manager, "first")
        gate.wait_started()
        second = submit(manager, "second", vram_need=(8, True))
        assert second.id in manager._needs
        assert manager.cancel(second.id) == "cancelled"
        assert second.id not in manager._needs
        assert second.id not in manager._preferred
        gate.release.set()
        wait_for(first, {"succeeded"})
```

- [ ] **Step 2: Run them to verify they fail**

Run: `venv/bin/python -m pytest tests/test_worker_pool.py -q -p no:xdist -k TestCancel`
Expected: `test_a_cancel_before_the_execute_is_sent_still_lands` fails (job reaches `succeeded`, or `gate.started` is set); `test_cancelling_a_queued_job_drops_its_dispatch_bookkeeping` fails on `second.id not in manager._needs`; the no-op test passes already.

- [ ] **Step 3: Add the flag to the job record**

In `dw/server/job_record.py`, in `Job.__init__` after `self.status = QUEUED`:

```python
        # Set by JobManager.cancel on a running job: the job's thread
        # checks it before sending Execute, since a cancel that reaches an
        # idle worker is ignored
        self.cancel_requested = False
```

- [ ] **Step 4: Make `cancel()` record the request and clean up a queued job**

In `dw/server/jobs.py`, replace the body of `cancel` from `if job.status == QUEUED:` to the end of the method with:

```python
            if job.status == QUEUED:
                if job.id in self._pending:
                    self._pending.remove(job.id)
                self._needs.pop(job.id, None)
                self._preferred.pop(job.id, None)
                self._finish(job, CANCELLED)
                return job.status
            slot = self._slot_running(job.id)
            if job.status == RUNNING and slot is not None:
                # Recorded first: the job's thread may not have sent Execute
                # yet, and a cancel reaching an idle worker is ignored
                job.cancel_requested = True
                try:
                    # The card running this job, and only that one
                    slot.manager.cancel()
                except Exception as e:
                    logger.warning(f"Could not send cancel for job {job_id}: {e}")
        return job.status
```

- [ ] **Step 5: Honour the flag in `_run_on`**

In `dw/server/jobs.py` `_run_on`, replace the `try:` block's contents up to and including `outcome = self._consume_results(job, manager, slot)` with:

```python
            try:
                if job.status != RUNNING or job.cancel_requested:
                    # Cancelled between dispatch and here: nothing was sent,
                    # so the outcome is the cancel itself
                    outcome = (CANCELLED, None, None)
                else:
                    manager.ensure_worker(self.log_level)
                    self._rank_for_oom(slot)
                    command = Execute(
                        # The snapshot admission checked, which the worker runs
                        # as it is rather than reading the file again
                        definition=job.spec.get("definition"),
                        file_spec=job.spec.get("file_spec"),
                        source=job.spec.get("source"),
                        workflow_dir=job.spec.get("workflow_dir"),
                        # The job's own roots, so a job queued for one workspace
                        # still runs in it after the manager has served another
                        output_dir=job.spec.get("output_dir") or self.output_dir,
                        arguments=job.spec["arguments"],
                        log_level=self.log_level,
                        asset_dir=job.spec.get("asset_dir") or None,
                    )
                    manager.send_command(command.to_wire())
                    if job.cancel_requested:
                        # Asked while the worker was still idle; now it has
                        # the job, and a second cancel lands behind it
                        manager.cancel()
                    outcome = self._consume_results(job, manager, slot)
```

Leave the `except`/`finally` and the trailing `if job.status not in TERMINAL_STATES:` as they are.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `venv/bin/python -m pytest tests/test_worker_pool.py tests/test_server.py -q -p no:xdist`
Expected: all PASS. If `test_a_cancel_before_the_execute_is_sent_still_lands` hangs, the `ScriptedWorkerManager.cancel()` path raised inside `cancel()` and was swallowed - that is expected and fine; the hang would instead mean the `with slot.lock:` in the test was taken *after* the job thread took it. Add `assert slot.current_job_id is None` before the `with` to confirm the order, and `time.sleep(0)` after `submit` if needed - do **not** widen `wait_until`'s timeout.

- [ ] **Step 7: Commit**

```bash
git add dw/server/job_record.py dw/server/jobs.py tests/test_worker_pool.py
git commit -m "fix(server): a cancel before Execute lands, and a cancelled queued job is forgotten

A cancel between the dispatcher's RUNNING mark and the job thread's
Execute reached an idle worker, which ignores it; cancel() of a queued
job left its _needs/_preferred entries (review 2026-10-07 #5/#6).

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: `hold_audio` attaches its outputs to a string `output` too (#7)

**Files:**
- Modify: `dw/pipeline_processors/pipeline.py:645-653` (`_with_held_audio`)
- Test: `tests/test_h3_hold_audio.py` (class `TestWithHeldAudio`)

**Interfaces:**
- Consumes: `ns`, `insert_audio_hold`, `minimax`, `MiniMaxH3AudioReference`, `HELD_AUDIO_OUTPUT`, `HELD_AUDIO_RATE_OUTPUT` from the test module's existing imports.

Context: the held-track outputs are added only when `output` is a list containing `"audio"`. The catalog writes string `output` specs (`"output": "audios"` in `music.json`), and `hold_audio` is an argument an agent adds from the skill, so a step with `"output": "audio"` silently returns the VAE round trip instead of the caller's track.

- [ ] **Step 1: Write the failing test**

Append to `class TestWithHeldAudio` in `tests/test_h3_hold_audio.py`:

```python
    def test_a_string_output_naming_audio_gets_the_held_keys_too(self):
        pipeline = minimax.MiniMaxH3Blocks().get_workflow("t2va").init_pipeline()
        insert_audio_hold(pipeline)
        reference = MiniMaxH3AudioReference(audio=torch.zeros(2, 10), sample_rate=8000)

        result = Pipeline._with_held_audio(
            ns(pipeline), {"hold_audio": reference, "output": "audio"}
        )

        assert result["output"] == ["audio", HELD_AUDIO_OUTPUT, HELD_AUDIO_RATE_OUTPUT]

    def test_a_string_output_not_naming_audio_is_left_as_written(self):
        pipeline = minimax.MiniMaxH3Blocks().get_workflow("t2va").init_pipeline()
        insert_audio_hold(pipeline)
        reference = MiniMaxH3AudioReference(audio=torch.zeros(2, 10), sample_rate=8000)

        result = Pipeline._with_held_audio(
            ns(pipeline), {"hold_audio": reference, "output": "videos"}
        )

        assert result["output"] == "videos"
```

- [ ] **Step 2: Run them to verify they fail**

Run: `venv/bin/python -m pytest tests/test_h3_hold_audio.py -q -p no:xdist -k string_output`
Expected: the first fails with `assert 'audio' == ['audio', ...]`; the second passes.

- [ ] **Step 3: Normalise the output spec**

In `dw/pipeline_processors/pipeline.py` `_with_held_audio`, replace:

```python
        output = arguments.get("output")
        if isinstance(output, (list, tuple)) and "audio" in output:
            arguments["output"] = list(output) + [
                HELD_AUDIO_OUTPUT,
                HELD_AUDIO_RATE_OUTPUT,
            ]
        return arguments
```

with:

```python
        output = arguments.get("output")
        # A single output is written as a string; the held keys ride along
        # only when the track itself is asked for
        names = [output] if isinstance(output, str) else list(output or [])
        if "audio" in names:
            arguments["output"] = names + [HELD_AUDIO_OUTPUT, HELD_AUDIO_RATE_OUTPUT]
        return arguments
```

- [ ] **Step 4: Run the file to verify it passes**

Run: `venv/bin/python -m pytest tests/test_h3_hold_audio.py tests/test_result.py -q -p no:xdist`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add dw/pipeline_processors/pipeline.py tests/test_h3_hold_audio.py
git commit -m "fix(h3): hold_audio attaches its held outputs to a string output spec

\"output\": \"audio\" with hold_audio returned the VAE round trip of the
held track, silently (review 2026-10-07 #7).

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: A chain clears its segment label on every exit (#8)

**Files:**
- Modify: `dw/pipeline_processors/chain.py:447-552` (the `for segment in config.plan:` loop in `run_chain`)
- Test: `tests/test_phase_events.py`

**Interfaces:**
- Consumes: `ChainedFakePipeline` in the test module; `run_chain(pipeline, chain_definition, arguments)`.

Context: `pipeline.segment_label` is set per segment and cleared only after the loop. The wrapper is rebuilt per step today so nothing leaks across jobs, but the invariant "the label belongs to the run that set it" is only true on the happy path.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_phase_events.py` after `test_a_chain_labels_each_segment_for_the_restarting_counter`:

```python
def test_a_chain_that_fails_mid_run_clears_its_label():
    from dw.pipeline_processors.chain import run_chain

    class FailsOnSecond(ChainedFakePipeline):
        def _run_once(self, arguments):
            if len(self.labels) == 1:
                raise RuntimeError("segment 2 blew up")
            return super()._run_once(arguments)

    pipeline = FailsOnSecond()
    with patch("dw.pipeline_processors.chain.empty_device_cache"):
        with pytest.raises(RuntimeError, match="segment 2"):
            run_chain(pipeline, {"segments": 3}, {"prompt": "p"})

    assert pipeline.segment_label is None
```

(`pytest` and `patch` are already imported in that file.)

- [ ] **Step 2: Run it to verify it fails**

Run: `venv/bin/python -m pytest tests/test_phase_events.py -q -p no:xdist -k clears_its_label`
Expected: FAIL with `assert 'segment 2/3' is None`.

- [ ] **Step 3: Wrap the loop**

In `dw/pipeline_processors/chain.py` `run_chain`, indent the whole `for segment in config.plan:` loop body one level under a `try:` and move the clearing line into its `finally:`:

```python
    try:
        for segment in config.plan:
            ...  # the existing loop body, unchanged, indented one level
    finally:
        # The label belongs to the run that set it - on a raise or a cancel
        # as much as on the way out
        pipeline.segment_label = None
```

Delete the now-duplicate `pipeline.segment_label = None` that followed the loop.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `venv/bin/python -m pytest tests/test_phase_events.py tests/test_chain.py -q -p no:xdist`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add dw/pipeline_processors/chain.py tests/test_phase_events.py
git commit -m "fix(chain): clear the segment label on a failed or cancelled run

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 7: `plan_cuts` honours `min_gap_seconds=0` (#9)

**Files:**
- Modify: `dw/tasks/cuts.py:801`
- Test: `tests/test_plan_cuts.py`

**Interfaces:**
- Consumes: `plan(**kwargs)` helper and `TRANSCRIPT` in the test module; `_transcript_lines(chunks, stanza_gap)`.

Context: `args.min_gap_seconds or 2.0` turns an explicit `0` into the default, so a caller asking that every silence start a stanza gets 2.0 s instead. The parameter defaults to `2.0` in the signature, so `None` is the only "unset" value to guard.

- [ ] **Step 1: Write the failing test**

Append to the class in `tests/test_plan_cuts.py` that holds `test_the_transcripts_own_gaps_make_stanzas` (line ~331):

```python
    def test_a_zero_gap_starts_a_stanza_at_every_silence(self):
        """`min_gap_seconds=0` is a value, not an absence: every pause in the
        transcript begins a stanza, so the fixture's four chunks make four
        one-line stanzas rather than the default gap's two."""
        result = plan(segment_by="stanza", min_gap_seconds=0)
        lyrics = [s["lyric"] for s in result["shots"] if s["lyric"]]
        assert len(lyrics) == 4
        assert not any("\n" in lyric for lyric in lyrics)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `venv/bin/python -m pytest tests/test_plan_cuts.py -q -p no:xdist -k zero_gap`
Expected: FAIL - the two plans are identical.

- [ ] **Step 3: Fix the default**

In `dw/tasks/cuts.py` line ~801 replace:

```python
        lines = _transcript_lines(chunks, args.min_gap_seconds or 2.0)
```

with:

```python
        gap = 2.0 if args.min_gap_seconds is None else args.min_gap_seconds
        lines = _transcript_lines(chunks, gap)
```

- [ ] **Step 4: Run the file to verify it passes**

Run: `venv/bin/python -m pytest tests/test_plan_cuts.py -q -p no:xdist`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add dw/tasks/cuts.py tests/test_plan_cuts.py
git commit -m "fix(cuts): plan_cuts honours min_gap_seconds=0 for stanzas

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 8: The skill's guide-chain lengths are the engine's (drift pin)

**Files:**
- Test: `tests/test_plugin_skills.py` (beside `test_the_guides_argument_is_offered_and_its_numbers_are_the_engine_s`, line ~392)

**Interfaces:**
- Consumes: `skill_body(H3_SKILL)` in the test module; `GUIDE_CHAIN_FRAMES` from `dw.pipeline_processors.h3_blocks`.

Context: `GUIDE_FRAMES_PER_CHUNK` is pinned against the skill text; `GUIDE_CHAIN_FRAMES = (22, 39)` is repeated by hand at `plugins/dw/skills/minimax-h3/SKILL.md:51` (`guide_frames` 22 or 39) with nothing holding the two together. The skill is at its byte cap, so this task adds a test, not text.

- [ ] **Step 1: Write the test**

Add after `test_the_guides_argument_is_offered_and_its_numbers_are_the_engine_s`:

```python
    def test_the_guide_chain_lengths_the_skill_offers_are_the_engines(self):
        """Review 2026-10-07: `guide_frames` 22 or 39 is hand-written in the
        skill; the engine's GUIDE_CHAIN_FRAMES is the rule it must match."""
        from dw.pipeline_processors.h3_blocks import GUIDE_CHAIN_FRAMES

        body = " ".join(skill_body(H3_SKILL).split())
        offered = " or ".join(str(frames) for frames in GUIDE_CHAIN_FRAMES)
        assert f"`guide_frames` {offered}" in body
```

- [ ] **Step 2: Run it to verify it passes against the current text**

Run: `venv/bin/python -m pytest tests/test_plugin_skills.py -q -p no:xdist -k guide_chain_lengths`
Expected: PASS (the text already says `\`guide_frames\` 22 or 39`). Then temporarily change `GUIDE_CHAIN_FRAMES` to `(22, 40)` in `h3_blocks.py`, rerun, confirm it FAILS, and revert - the pin has to be shown to bite.

- [ ] **Step 3: Commit**

```bash
git add tests/test_plugin_skills.py
git commit -m "test(skills): pin the H3 skill's guide_frames choices to GUIDE_CHAIN_FRAMES

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 9: Whole-branch verification and merge

**Files:** none new.

- [ ] **Step 1: Full suite**

Run: `venv/bin/python -m pytest tests/ -q`
Expected: 0 failures (baseline 2026-10-07: 10,105 passed, 15 skipped, 1 xfailed).

- [ ] **Step 2: UI tests untouched but run anyway** (Task 2 changed no API model, so no OpenAPI regen is needed)

Run: `cd ui && npx vitest run`
Expected: PASS.

- [ ] **Step 3: Validate the catalog**

Run: `for f in workflows/templates/minimax/chained-segments.json workflows/templates/minimax/music.json; do venv/bin/python -m dw.validate "$f" || true; done`
Expected: `chained-segments.json` clean; `music.json` clean (or only this Mac's MPS ceiling, as before).

- [ ] **Step 4: Merge**

```bash
git checkout develop && git merge --no-ff fix/review-2026-10-07 -m "Merge fix/review-2026-10-07: review fixes #1-#9

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

Deploy to lem is a separate decision (needs a server restart; a change to engine code is not seen by the persistent worker until then).

---

## Follow-on plans (out of scope here; each gets its own plan)

Listed so the orchestrator can schedule them after this branch lands. Order is a recommendation.

1. **`h3_blocks.py` split** - `h3_rules.py` (constants + pure checks that `guides.py`/`hold_audio.py`/`chain.py` import for validation; make `_not_h3` a public `not_h3` there), `h3_hold.py`, `h3_guides.py`. Also move `GUIDE_CHAIN_FRAMES`/`guide_frames` into `chained-segments.json`'s `variable_constraints` so the catalog, not `dw/`, owns the 22/39 rule - then Task 8's pin reads the catalog. Add the drift test the review asked for: refine's timestep plan equals stock's at `strength=1`, and each `LAYOUT_ANCHORS` block builds against the installed diffusers.
2. **Task-layer consolidation** - one registration pattern (`register_command(domains=..., static_check=...)`), the `extra` dict in `task_domains.py:1036` derived from the registry and pinned; one `whole_number(value, name, command)` replacing the five coercion idioms; `image_ops.py` for the three luma-weight copies and alpha split/join; `locations.load_json_record` for `_read_fit`/`_read_track`; per-task rule sets moved out of `task_domains.py` next to their tasks. `crop_face_track` takes its frame grid as arguments like `plan_cuts` does.
3. **`jobs.py` split** - `pool.py` (WorkerSlot, `route`, `_choose_slot`, `_capacities`, `largest_ceiling_gb`, `_unfit_message`) and `job_results.py` (`_consume_results`, manifest/progress recording); device identity as `(ordinal, name)` fields rather than a display string re-parsed by `split(" ", 1)`; retire the `slots[0]` shims after migrating `routes/system.py:413`.
4. **VRAM accounting for guides, held audio and refine** - a `gb_per_guide_frame` term in the `vram_estimate` schema or an explicit warning that these sit outside the estimate.
5. **Streaming `window_video` / `join_windows`** - a frame-range reader so N windows do not decode the source N times, and a uint8 accumulator in the join.
6. **Smaller leftovers** - `enhancers.PRESETS` family wiring (decide whether it is UI registry or model knowledge); the frame-0 collision between an own guide and the chain's guide (confirm the run refuses it, then validate it); `from_file` routed through `validate_media_path`; chunked uploads sized before the body is read.
