# Installer follow-ups, part 2: update replies and release pins (#1768)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out,
2026-11-26). Plan v1 approved by Don on 2026-11-27 ("approved with delegate
recommendations", every default Q3-Q10), three stages: #1770 (`newer`),
#1771 (update-refusal replies) and #1772 (release pins), all built, verified
and closed.

## The idea

#1736 (installer follow-ups) plan v4 deferred its stages 3 and 4, and Don
ruled that they come back as a new plan once stage 2 closed. Each was a latent
bug on `develop`:

- the Models page decided "update available" with a string heuristic;
- a batch rerun during an update answered 200 and quietly queued nothing;
- a short commit pin on a host without git got a 202, then failed in the
  updater thread;
- a bootstrap script fetched from a versioned release URL installed the
  newest release;
- the `.mcpb` packer's transitive npm dependencies were unlocked
  (`npx -y @anthropic-ai/mcpb@2.1.2`).

## Corrections the sweep made to #1736 plan v4 (develop 8cfd914d)

1. The Q5 route is POST `/api/system/diffusers/update`, not `/api/system/dw`.
2. The restart-batch rerun has no `dry_run`. A call without
   `acknowledged_cost` is the preview, and its reply model
   `RestartBatchRerun` had no field that could name a refusal, so "the
   preview names the refusal" became a new optional field.
3. A root `package-lock.json` was already tracked, as an empty `packages:{}`
   stub, with no root `package.json`.

## Decisions

- **Verdict: build**, split so each stage has one verifier.
- **Q3 [scope]:** `DwUpdateState.newer: bool | null` on GET `/api/system/dw`
  and the POST update reply; the UI heuristic is deleted.
- **Q4 [default]:** the acknowledged batch rerun answers 409 during an update;
  the preview stays 200 and names it in a new `refusal` field.
- **Q5 [default]:** 400 for a commit under 40 hex characters when git isn't
  usable.
- **Q6 [default]:** the release job stamps its version into both bootstrap
  scripts.
- **Q7 [trust]:** a committed root `package.json` and lockfile pin the packer,
  installed with `npm ci`. No Dependabot entry.
- **Q9 [loosen]:** stages B and C are verified by `docs-review` reading code
  and tests, as Don approved for #1736 stage 1.
- **Q10 [none]:** a 40-character uppercase commit is lowercased before the
  updater.

## What was built

### Stage A, #1770: `newer` on the dw state (6e86bbde, merged 6891095a)
- `DwUpdateState.newer: bool | None` (`dw/server/api_models.py`), required
  and nullable like `latest`.
- `_is_newer(latest, __version__)` in `dw/server/routes/system.py` compares
  with `packaging.version.Version`; None when `latest` is None or either side
  raises `InvalidVersion`. GET and POST `/api/system/dw/update` both carry it.
- `ModelsPage.svelte`: `isNewer()` deleted; the offer shows only when
  `newer === true`.
- **Deviation:** "up to date" now shows only when `newer` is false. A null
  `newer` (offline, unparseable) shows the version alone, neither offering an
  update nor claiming the install is current.
- Tests: `TestDwNewer` in `tests/test_server.py`, the `9.9.9` stub asserting
  `newer is True`, and ModelsPage vitest on true, false and null.
- Docs: `SERVER.md` (Update dw, the reply list), `INSTALL.md` (Update button).

### Stage B, #1771: update-refusal replies (e3e6e0d1, merged b41cb119)
- **Q4:** `rerun_restart_batch` (`dw/server/routes/jobs.py`) reads
  `manager.submission_refusal()` after the 404/bound-ack checks; an
  acknowledged call gets 409 with the refusal text. The preview carries
  `RestartBatchRerun.refusal: str | None` (null when none is set) and
  `would_queue: []`.
- **Q5:** `update_diffusers` (`dw/server/routes/system.py`) answers 400 for a
  non-40-character commit when `updater.git_usable()` is false.
- **Q10:** a 40-character commit is lowercased before `updater.start`.
- **Deviations:** with a refusal set, preview members that would have been
  admitted are listed in `skipped` with the refusal as the reason, so every
  member stays accounted for. Lowercasing applies with git too (git accepts
  either case).
- Tests in `tests/test_restart_tombstones.py` and
  `tests/test_server.py::TestDiffusersUpdate`; five of the six failed against
  the pre-fix source (the full-hash 202 test is a guard).
- Docs: `SERVER.md` (restart batch, diffusers update). `ARCHITECTURE.md`
  needed no change: its rows don't list the refusal seam's routes.

### Stage C, #1772: release pins (b2d124d2, merged f8db2084)
- **Q6:** the release step runs `scripts/stamp_bootstrap.py
  "${GITHUB_REF_NAME#v}" dist/bootstrap.sh dist/bootstrap.ps1` between the
  copy and `sha256sum`. It fails unless each placeholder appears exactly once
  before and its stamped form exactly once after, and refuses a non-semver
  version. `--version`/`-Version` and `DW_BOOTSTRAP_WHEEL` still override.
- **Q7:** root `package.json` (private, no scripts, `@anthropic-ai/mcpb`
  `2.1.2` exact) and a real lockfile (v3, 55 packages, each with an integrity
  hash, none with install scripts). The wheel job runs a root `npm ci` and
  caches both locks. `build_mcpb.py` runs `node_modules/.bin/mcpb`, or exits
  saying to run `npm ci` at the repo root. The mcp-registry job needs no node.
- **Deviation:** the stamp is a small script rather than inline `sed`, so the
  code a release runs is the code the tests run
  (`tests/test_stamp_bootstrap.py`, and
  `test_bootstrap.py::test_a_release_stamped_script_installs_its_release`).
  That closed the plan's main risk, a stamp only exercised by a tag push.
- Docs: `INSTALL.md` (script options: a versioned URL installs that release),
  `RELEASING.md` (the root `npm ci`, the stamp step, moving the mcpb pin).

## Deferred

- Showing the preview's `refusal` in the UI's restart-batch dialog. The 409 on
  the real call already surfaces as a failure toast. A follow-up if wanted.
- A Dependabot entry for the root npm lock (Q7 default: none).
- The end-to-end proof of stage C is Don's next release:
  `curl …/releases/download/vX/bootstrap.sh | bash -s -- --dry-run` names
  `diffusers-workflow==X`, and the `.mcpb` builds from the locked packer.

## Cost and bounces

Planned at about $2.5 + $2 + $2 ≈ $6.5 of agent work and no GPU. The stage
comments name no `usage:` figures, so actual cost is not recorded.

- **#1770: none.** C-F1544 and C-F1545 passed on lem at b41cb119.
- **#1771: none.** Tester found nothing observable over MCP; `docs-review`
  passed.
- **#1772: none.** `docs-review` routed it back to the tester because the
  change touched the served `INSTALL.md` (a routing step, not a failure); the
  tester verified the served text with `get_guide("install")`.
