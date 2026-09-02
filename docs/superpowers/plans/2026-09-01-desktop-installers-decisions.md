# Desktop Installers — Decisions Taken Overnight

Companion to [the plan](2026-09-01-desktop-installers.md) and
[the spec](../specs/2026-09-01-desktop-installers-design.md). Everything
here was decided without you; the ones marked **needs your call** are
where I'd most like to be overruled.

## Deviations from the plan

**1. Two crates instead of one.** The plan put every module in the Tauri
crate. That makes `cargo test` build the whole GUI toolchain. I split it
into `desktop/core` (zero Tauri dependencies — paths, GPU detection,
ports, provisioning, config merging) and `desktop/src-tauri` (the thin
app). Core tests now compile and run in about 6 seconds, which is what
made the next two decisions affordable. This serves the spec's own
"unit-testable without a running Tauri app" goal better than the plan did.

**2. A `desktop-core` CI job on every push.** The plan only had a
tags-only `desktop` job, which would mean 45 tests that never run until
release day. The core job is seconds of Ubuntu time and runs `cargo fmt
--check`, `clippy -D warnings` and the tests on every push.

**3. A cross-language contract test.** Not in the plan.
`desktop/core/tests/python_contract.rs` writes `server.json` with the real
Rust and reads it back with the real Python (`resolve_base_url` and
`running_server_url`). The port handshake is the one place a silent
mismatch between the two languages was possible, and each side's unit
tests only proved its own half. It skips cleanly when a checkout has no
development venv.

**4. `seed_dir` lives in `provision.rs`,** not `paths.rs` as the plan
said. Seeding is a provisioning step; it belongs with the marker and the
install sequence.

## Judgment calls

**5. The driver → CUDA index table — needs your call.**

| Driver | Index |
| --- | --- |
| ≥ 580 (Linux) / ≥ 581 (Windows) | cu130 |
| ≥ 525 (Linux) / ≥ 528 (Windows) | cu128 |
| below that | CPU, with an explanation |

PyTorch's CUDA wheels bundle their runtime, so only the driver floor
matters. I erred toward the older, wider-compatibility wheel — installing
cu128 on a machine that could run cu130 costs a little performance;
installing cu130 on a machine that cannot run it produces an app that
does not work. You have real hardware and I don't, so this table is worth
your eyes.

**6. A malformed MCP config is an error, not an overwrite.**
`merge_mcp_config` returns `Err` rather than replacing a file it could not
parse. That file is the user's and other tools write to it; discarding it
to insert our entry would be the wrong trade.

**7. The health probe insists on a 200.** It writes a raw
`GET /api/health` over `std::net` rather than adding an HTTP client crate
for one localhost check. A TCP connect alone would have been simpler but
reports "ready" while the engine is still importing torch.

**8. Runtime versions are pinned to real, verified hashes.** uv 0.12.9
and CPython 3.12.14 (python-build-standalone 20260901). I fetched the
SHA256s from upstream rather than inventing them, and then **actually ran
`fetch-runtime.sh`** — it downloaded both, verified both, and the
extracted binaries report `uv 0.12.9` and `Python 3.12.14`. 3.12 because
that is what CI tests on; the project floor is 3.10.

## What I could not verify, and two real gaps

**9. The updater public key is a literal placeholder.**
`tauri.conf.json` contains `REPLACE_WITH_TAURI_SIGNER_PUBLIC_KEY`. Run
`npm run tauri signer generate`, put the public half there and the private
half in the `TAURI_SIGNING_PRIVATE_KEY` secret. **Auto-update does not
work until this is done**, and I could not do it — it generates a
credential.

**10. Version syncing — found as a gap, then fixed.** The shell pins the
wheel to its own version, but `scripts/release.sh` bumped only
`pyproject.toml`, so a release would have built a shell installing
`diffusers-workflow[server]==0.0.0`. Now `release.sh` bumps
`desktop/Cargo.toml` in the same path-limited commit, refreshes
`Cargo.lock`, and the CI release job refuses a tag that does not match
both. I removed the duplicate `version` from `tauri.conf.json` so Cargo
is the single source.

Verified rather than assumed: bumping the workspace version propagates
to both crates (`cargo metadata` reports `0.4.0-alpha.11` for each), and
pip's own parser confirms the semver pin resolves —
`Requirement("diffusers-workflow[server]==0.4.0-alpha.10")` matches
version `0.4.0a10`. So no second spelling of the version is needed
anywhere.

The shell is still at `0.0.0` on this branch; the next `release.sh` run
sets it.

**11. Placeholder app icons.** Generated programmatically — a blue
rounded square with a faint concentric sweep. They satisfy the build;
they are not a design.

**12. Only macOS arm64 was built here.** Windows and Linux bundles are
unverified — no runner. The Rust compiles for this host only; the
`cfg!(windows)` branches are exercised by logic tests, not by a Windows
compiler.

**13. `capabilities/default.json` is unverified at runtime.** The
permission list is my best reading of what the shell's screens need. A
missing grant surfaces as an `invoke` being denied, which will show up
the first time the app actually runs.

**14. `ubuntu-22.04` runners are on GitHub's deprecation path.** Chosen
for the oldest glibc/webkit2gtk, which gives the AppImage the widest
reach. Worth revisiting before it is retired.

## Built and verified here

`npm run tauri build --target aarch64-apple-darwin` produced a real
**59 MB `diffusers-workflow_0.0.0_aarch64.dmg`** — within the 60–100 MB
the spec predicted. `fetch-runtime.sh` was run for real: it downloaded
uv and CPython, both checksums verified, and the extracted binaries
report `uv 0.12.9` and `Python 3.12.14`. The frontend builds all four
entry points. Unsigned, since the Apple secrets are not on this machine.

## Deliberately not included

HTTP MCP transport. `dw_mcp` is already an HTTP client of `dw.serve` and
already reaches a remote engine via `--url`, so the transport is not what
makes a split front end and back end possible — authentication is, and
there is none today. Scoped separately in
[remote-backend.md](../scope/remote-backend.md).

## What running CI locally through `act` found

Run against a remote linux/amd64 Docker host, so these are native results
rather than emulated ones. Three real defects, none of which the macOS
build had any way to reveal:

**The `desktop-core` job would have failed outright.** It installed only
`httpx`, but the contract test imports `dw_mcp` and `dw`. It passed on my
machine solely because the dev venv has the package installed. Fixed by
following the boundary that already exists: `dw_mcp` imports nothing from
`dw`, so its half runs for real with `httpx` plus a `PYTHONPATH`, and the
REPL half skips where torch is absent.

**A green job that proved nothing.** After that fix the job passed - but a
skipped Rust test still prints `ok`, so "green" was not evidence. Added
`DW_CONTRACT_REQUIRE`, which turns a skip of a named half into a loud
failure; CI names `dw_mcp.client`. Verified in both directions.

**The AppImage could never have built.** `linuxdeploy` walks every ELF in
the AppDir and resolves its shared libraries. The bundled interpreter's
`_tkinter.so` links against a Tcl/Tk that python-build-standalone ships in
a layout it cannot resolve, so bundling failed with
`Could not find dependency: libtcl9tk9.0.so`. `fetch-runtime.sh` now
strips Tcl/Tk recursively - the nested copies in `lib/itcl4.3.8/` and
`lib/thread3.0.6/` matter too, since linuxdeploy scans every ELF it finds,
not only ones something links to. The macOS and Windows bundlers do no ELF
resolution, which is exactly why building the `.dmg` locally said nothing
about this.

Also fixed: the AppImage bundler needs `xdg-open`, so the job installs
`xdg-utils`. That one may be act-specific - GitHub's runner images are far
fatter - but the bundler genuinely requires it either way.

### What `act` still does not prove

- It maps every `ubuntu-*` runner to one image (a cache key showed
  `24.04-Ubuntu`), so the deliberate `ubuntu-22.04` choice - older
  glibc/webkit2gtk for the widest AppImage reach - is untested.
- Windows and macOS runners cannot run under it at all.
- `actions/upload-artifact` fails for want of `ACTIONS_RUNTIME_TOKEN`, so
  the release attachment path is unverified.
- Docker has no `/dev/fuse`, so verifying the AppImage needed
  `APPIMAGE_EXTRACT_AND_RUN=1` passed to `act` only. It is deliberately
  **not** in `ci.yml` - production CI should not be reshaped to suit a
  local emulator.

## macOS signing, resolved against a real certificate

The Developer ID cert now exists (team `86TDY6D9V2`) and all eight secrets
are registered, so the two risks flagged in item 9 could be tested rather
than guessed at. Both were real.

**The bundled binaries were ad-hoc signed.** A signed build showed
`Signature=adhoc` on `uv`, `python3.12` and `libpython3.12.dylib` -
how python-build-standalone and uv ship on arm64. Tauri signs the app
binary and seals `Contents/Resources` by hash, but does not re-sign what
lives there, and `codesign --deep --strict` passes anyway because those
files are sealed as resources rather than nested code. Notarization is
stricter and rejects ad-hoc signatures. `scripts/sign-macos-runtime.sh`
now signs all five Mach-O files deepest-first before the bundler copies
them, and fails loudly if any ad-hoc signature survives.

**Library validation would have broken every install.** Under the
Hardened Runtime a process may only load libraries signed by its own
team, and a provisioned venv loads pip-built extensions signed by nobody.
`python.entitlements` grants `disable-library-validation` (plus JIT and
dyld-environment entitlements the worker's `spawn` re-exec needs).
Verified end to end: numpy 2.5.2 - whose `_multiarray_umath.so` is
`adhoc, linker-signed` - imports and computes under the signed
interpreter. Without the entitlement the app would have installed
cleanly and failed on the first import.

After both fixes the bundle reports `Authority=Developer ID Application`
on every nested binary and `spctl` gives `source=Unnotarized Developer
ID`, which is the correct state for a signed build that has not been
through notarization yet.

**Two updater gaps found alongside.** `bundle.createUpdaterArtifacts` was
absent, so no update archives or `.sig` files were produced at all; and
nothing generated `latest.json`, which the endpoint in `tauri.conf.json`
fetches. Both are fixed - the manifest is built in the release job by
`scripts/make-latest-json.sh`, tested against fixtures including a
platform whose `.sig` is missing.

### Still unverified

Notarization itself has not run - it needs the app-specific password,
which only CI holds. The first tagged build is the first real test of
`APPLE_ID` / `APPLE_PASSWORD` / `APPLE_TEAM_ID` and of
`createUpdaterArtifacts`, which could not be exercised locally because
the Tauri signing key is password-protected.

## State

Branch `desktop-installers`, 17 commits. 1,984 Python tests pass
(4 skipped), 45 Rust tests pass, `black --check`, `cargo fmt --check` and
`clippy -D warnings` all clean. Nothing merged to master except the spec
and the plan.
