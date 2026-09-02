# Desktop Installers — What's Left

> Status: PR [#34](https://github.com/dkackman/diffusers-workflow/pull/34)
> open on branch `desktop-installers`, 19 commits, deliberately unmerged as
> of 2026-09-02. Design:
> [spec](../specs/2026-09-01-desktop-installers-design.md) ·
> [plan](../plans/2026-09-01-desktop-installers.md) ·
> [decisions](../plans/2026-09-01-desktop-installers-decisions.md).

## The one thing to know first

**The app has never been launched.** Every check so far is static: it
compiles, it bundles, it signs, its logic is unit-tested. Nobody has
double-clicked the `.dmg` and watched it provision a venv, start
`dw.serve`, and navigate the webview to the SPA.

So the whole runtime path is unverified: the provisioning screen, the
accelerator dropdown, the progress events, the health poll and navigation,
the Connect/Developer/Logs windows, the menu, and whether
`capabilities/default.json` grants what those screens actually invoke. A
missing permission surfaces as an `invoke` being denied at runtime, which
no build-time check can catch.

Cheapest way in: open the built `.app` from
`desktop/target/aarch64-apple-darwin/release/bundle/`, or run
`cd desktop && npm run tauri dev`.

## Done and verified

- 45 Rust tests, 1,994 Python tests, `black`, `cargo fmt`, `clippy -D warnings`.
- A real 59 MB `.dmg` builds on macOS arm64; the Linux AppImage builds
  under `act` on amd64.
- The cross-language `server.json` handshake, proven by running the real
  Python against a file the real Rust wrote.
- macOS code signing, against the real Developer ID cert (team
  `86TDY6D9V2`): all five bundled Mach-O binaries carry
  `Authority=Developer ID Application`, and `spctl` reports
  `source=Unnotarized Developer ID`.
- Hardened Runtime entitlements: numpy's `adhoc, linker-signed` C
  extension imports under the signed interpreter, which is the case
  `disable-library-validation` exists to allow.
- All eight CI secrets registered (six `APPLE_*`, two `TAURI_SIGNING_*`).

## Untested, in rough order of risk

1. **The app at runtime** — see above.
2. **Notarization.** Never run; it needs the app-specific password, which
   only CI holds. The first tagged build is the first real test of
   `APPLE_ID` / `APPLE_PASSWORD` / `APPLE_TEAM_ID`.
3. **`createUpdaterArtifacts` and `latest.json`.** Enabled and the manifest
   builder is fixture-tested, but no real update archive or `.sig` has ever
   been produced - the Tauri signing key is password-protected, correctly,
   so it could not be exercised locally.
4. **Windows entirely.** No runner available locally; `act` cannot run
   Windows or macOS images. The `.msi`, the CUDA detection path, and every
   `cfg!(windows)` branch have only ever been compiled for macOS/Linux.
5. **`ubuntu-22.04` specifically.** `act` maps every `ubuntu-*` runner to
   one image (a cache key showed `24.04-Ubuntu`), so the deliberate choice
   of 22.04 - older glibc/webkit2gtk for the widest AppImage reach - is
   untested.
6. **`actions/upload-artifact`.** Fails under `act` for want of
   `ACTIONS_RUNTIME_TOKEN`, so the release attachment path is unproven.

## Suggested next step

Cut a throwaway pre-release tag - `v0.4.0-alpha.11` - rather than
discovering a notarization rejection on a release meant for people. That
single tag exercises items 2 through 6 at once, on real runners. Note that
`scripts/release.sh` now bumps `desktop/Cargo.toml` alongside
`pyproject.toml`, and the release job refuses a tag that does not match
both; the shell is at `0.0.0` until a release sets it.

Launching the app locally (item 1) is worth doing before that tag, since
it needs no CI and would catch the most embarrassing class of problem.

## Known-deliberate gaps, not oversights

- **Windows is unsigned.** Documented SmartScreen click-through. The CI job
  is written so enabling it later means adding secrets and a step, not
  restructuring.
- **Placeholder icons.** Generated programmatically; a blue rounded square.
  They satisfy the build and are not a design.
- **No ROCm or Intel XPU detection** - CPU fallback, documented.
- **No offline installer, no `.deb`/`.rpm`, no x86-64 macOS.**
- **No HTTP MCP transport.** Not what unblocks a split front end and back
  end - authentication is, and there is none. Scoped separately in
  [remote-backend.md](remote-backend.md).
