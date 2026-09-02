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

## State

Branch `desktop-installers`, 13 commits. 1,984 Python tests pass
(4 skipped), 45 Rust tests pass, `black --check`, `cargo fmt --check` and
`clippy -D warnings` all clean. Nothing merged to master except the spec
and the plan.
