# Installer follow-ups, stages 1 and 2 (#1736)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out,
2026-11-26). Plan v4 was approved by Don (2026-11-26 13:58Z, "approve with
delegate recommendations"). The verdict was "build smaller": two of four stages
were approved and built, #1742 and #1743. Stages 3 and 4 were deferred, and Don
asked for them to come back as a new plan once stage 2 closed. They are carried
forward in #1768 (see *Deferred*).

## The issue

The installer branch (`feat/installer`, spec
`docs/superpowers/specs/2026-11-25-installer-design.md`) merged with a list of
follow-ups its review had deferred: real-machine checks, setup CLI bugs,
updater gaps, bootstrap and release pins, and one docs line. The design sweep
(develop @ e23deb6e) confirmed that 15 of the 17 still held. The bootstrap pin
held only in part: an explicit `--version` already pins. The bare `dw agent`
lines in `INSTALL.md` follow that doc's own rule (it writes plain `dw`). The
one misleading place was the runtime message at `dw/setup_cli.py:437`.

## Decisions

- **Q1 [trust], setup token:** default, not built. Reopen trigger: a second
  login user, or a non-loopback bind.
- **Q2 [scope], `MCP_REGISTRY_PUBLISH` gate:** default, it stays. Don flips the
  repo variable after his Desktop/Cowork pass succeeds.
- **Q8 [none]:** Windows `RestartOnFailure` stays at 3, the service gets no
  PATH, and the bare `dw agent` lines in `INSTALL.md` stay.
- **Q9 [loosen], verifiers:** default. The tester reaches `dw` only over MCP
  and the web UI, so the v2 verifier ("the tester runs the named tests on lem
  and reads the diff") couldn't run (tester spec question, 13:46Z). Stage 1
  became `docs-review` and took every item with no MCP or REST surface. That
  was the first use of `docs-review` on code, and Don approved it as a
  precedent. Stage 2 was cut down to the lock fix and verified by `C-F1443`.
  Don's three real `dw setup` flag calls moved to his manual pass.
- **Q3 to Q7:** still open, and deferred with stages 3 and 4.

## What was built

### Stage 1, #1742: setup CLI fixes and build hygiene

`feat/1736-1-setup-cli-hygiene` (1f32c25d, 586232e7, a4e5ab26, aab7ed9a,
c9d34350), merged to `develop` as 59f5061d.

Setup CLI:
- `_our_service_installed` requires the service to be installed **and
  running**. If the service is stopped and some other server answers on the
  port, setup leaves that server alone.
- `DW_INSTALL_ROOT` (`setup_state.install_root`) and `dw-serve --install-root`
  go through `validate_path` with no base. A traversal raises `SetupError` or
  exits 2.
- On Windows, "running" comes from `(Get-ScheduledTask).State`, which is not
  localized, instead of matching `schtasks /V` text. PowerShell runs only when
  the task exists.
- `code_duplicates` takes the executable as everything up to the first ` -`
  argument, so a path with spaces matches.
- `_listed` reads only the first section. A listing counts as empty only when
  it is blank or a line of its own starts with `No `. Anything else
  unparseable gives `None`, and register then never removes anything.
- `claude mcp list` and `claude plugin list` each run once per status and once
  per install. `register_code(..., servers=)` and `code_duplicates(servers)`
  take the parsed listing.
- `--status` and `--uninstall` are mutually exclusive, and `--json` without
  `--status` is a `parser.error`. Both exit 2.
- The `dw agent` hint prints the full path to `dw`.

Hygiene:
- New tests for exit 75 vs a plain interrupt (`tests/test_serve_main.py`).
- The constraints dir in `self_update._run_commands` is a `TemporaryDirectory`.
- `build_mcpb._name` uses `packaging.requirements.Requirement`.
- `test_mcpb.py` reads the `mcp` floor from `pyproject.toml`.
- `DW_BOOTSTRAP_TTY` is honoured only under `DW_BOOTSTRAP_DRY_RUN`, and the
  script header documents it as test-only.
- `setup-uv` is pinned to `version: "0.9.5"`.

Every Tests bullet in the plan has a test that fails on the old code. The
docs reviewer listed them on #1742.

Deviations:
- The bootstrap has no `--dry-run` flag. The gate is the existing
  `DW_BOOTSTRAP_DRY_RUN` env var.
- The repo had no uv version on record, so the uv pin was chosen by the
  implementer.
- `code_duplicates` and `register_code` changed signature. These are internal
  and have no MCP surface.

### Stage 2, #1743: version-check lock

4f3c051c on `feat/1736-2-latest-lock`, merged as 15fffadc and pushed in
`develop` @ 9692f92f.

- In `self_update.latest_version()`, `_latest_lock` now guards only the cache
  read and the cache write. The PyPI GET (`_fetch_pypi`) runs outside the lock.
  Two cold reads at the same time may both fetch, and the later write wins.
- Tests: a second `GET /api/system/dw` returns within 1 s while the first fetch
  is blocked (route level and function level), and an offline fetch answers
  `latest: null`.
- No change to the reply shape or to MCP.

## Verification

- **#1742:** the architecture review passed. The docs review passed the
  Builds/Tests checklist and found nothing that touches the service manager
  outside `dw setup`. It sent the served `docs/INSTALL.md` text back to the
  tester, who checked it over `get_guide` and passed it (lem, develop @
  2eb5263a). The reviewer's wording point ("reports it" should be "reports it
  running") was fixed separately in #1760. It now reads "installed and
  running".
- **#1743:** the architecture review passed. The tester ran `C-F1443` on lem
  @ 9692f92f: four back-to-back reads all answered 200 with the same seven keys,
  and the Models page showed the version within 3 s with no errors. The slow
  and offline paths rest on the unit tests.
- Both build sessions ran the full suite: one failure, the same before and
  after (`test_docs_links.py` on `qwen-image-transparent-complete.md`). This
  feature didn't cause it.

**Bounces:** none on either stage. **Cost:** no stage comment recorded a
`usage:` figure. The plan estimated $5 for stage 1 and $2 for stage 2.

## Deferred

- **Stage 3, update replies** (Q3 to Q5, about $4):
  - `DwUpdateState.newer: bool | null` computed with PEP 440, which replaces
    the heuristic in `ModelsPage.svelte`;
  - POST restart-batch answers 409 during an update;
  - POST `/api/system/dw` answers 400 for a short-sha pin when git isn't
    usable.

  Its test-only items need a verifier other than the tester.
- **Stage 4, release pins** (Q6 and Q7, about $2):
  - the release job stamps its version into `bootstrap.sh`/`bootstrap.ps1`;
  - a committed npm lockfile pins `@anthropic-ai/mcpb`.

  Verifier: `docs-review`, and Don's next release.
- Don expects to take every default for both stages. Neither is approved yet.
  Both are carried in #1768, filed at this close-out, which points to plan
  v4's full stage text.
- **Don's manual pass** (no agent can run it):
  - Desktop/Cowork: the shim and the `.mcpb` (`uv` type, `--frozen`);
  - Windows: `${HOME}` in `.mcp.json`, the `dw-mcp.exe` shim, SIGINT giving
    exit 75 and a restart, uv `-U` with the task ended;
  - Windows with an older NVIDIA driver: a cu126/cu128 index;
  - lem in a scratch user: the lingering prompt and detection of an existing
    `dw-serve`;
  - on lem: `dw setup --status --uninstall` and `dw setup --json` each exit 2,
    and `dw setup --status --json` exits 0 with its old shape.
