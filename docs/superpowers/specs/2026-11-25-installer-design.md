# Installer and agent hookup - design

2026-11-25. Status: approved.

## Intent

dw is growing past "an agent harness you build from a checkout". The part we
hope sets it apart is the conversation: an agent drives jobs, the person reacts
in the web UI through notes, the agent reads the notes and goes again. Today a
newcomer reaches that only by cloning the repo, running `install.sh`, starting
`dw.serve` by hand, and hand-editing MCP registrations - and the live notes
push (the `dw-notes` channel) needs a stdio server in `.mcp.json` plus a
`claude --dangerously-load-development-channels` flag nobody will discover.

The installer makes "dw installed, running, and connected to my Claude" one
step. The agent stays outside dw: Claude Desktop and Claude Code, over the MCP
surface that already exists. dw does not grow a chat panel or a second agent.

Success: on a clean Mac, Linux box or Windows machine, a person pastes one
line, waits through the torch download, and lands in the web UI with dw
already listed as connected in Claude Desktop and/or Claude Code - and, in
Claude Code, a note saved in the UI reaches the session live.

## Goals

- One-line bootstrap per OS that installs uv, a managed Python, dw from PyPI
  with the right torch build, and diffusers from git HEAD, then runs
  `dw setup`.
- `dw setup`: a Python command, safe to re-run, that
  1. starts `dw-serve` at login as a per-user service,
  2. registers dw with Claude Desktop (macOS, Windows) if present,
  3. registers dw with Claude Code if the `claude` CLI is present: the `dw`
     plugin from this repo's marketplace (which now carries the `dw` tool
     server, below) and the `dw-notes` channel server,
  4. opens the web UI.
- `dw agent`: starts Claude Code with the notes channel loaded
  (`claude --dangerously-load-development-channels server:dw-notes`), passing
  through any further arguments.
- `dw setup --status` (human and `--json`): what is installed, running and
  registered, and what is missing.
- `dw setup --uninstall`: removes the service and the registrations it made;
  leaves the venv, models, workspace and caches unless asked.
- The diffusers update button on the Models page keeps working on an
  installed copy.
- **Upgrading dw is re-running the one-liner.** The bootstrap is safe to run
  over an existing install. It keeps the torch build chosen the first time
  (recorded at install, re-detected only if the record is missing), keeps
  diffusers on `main`, upgrades dw to the latest release, then re-runs
  `dw setup`, which refreshes the registered paths and the plugin pin and
  restarts the service.
- **An "Update dw" button in the web UI, in v1.** It sits beside the diffusers
  update and follows its pattern (a check for a newer release, a confirm, a
  progress log, a restart), and it runs the same upgrade as the one-liner.
  The server cannot replace itself mid-run, so the upgrade runs detached and
  the service manager restarts the server. When the server was not started by
  setup's service (a checkout, the systemd unit on lem, mini-ai's
  LaunchDaemon, Docker), the button does not run anything. It shows the
  command to run instead.

## The MCP bundle and the plugin (v1)

Anthropic's directory no longer takes Desktop extensions (`.mcpb`); a local
MCP server is distributed inside a plugin. The same plugin installs in Claude
Desktop chat, Cowork and Claude Code, and each loads a different part of it:
skills load in all three, a local MCP server loads in Cowork and Claude Code,
and Desktop chat ignores it. So v1 ships the server two ways from one source.

**The bundle: `dw-mcp-<version>.mcpb`, built every release.**

- A `.mcpb` is a zip of a `manifest.json` and a local MCP server. It carries
  `dw_mcp` alone. That package imports no `dw` (only `mcp`, `httpx`, `anyio`,
  `packaging`), so the bundle is small and pulls no torch. A test pins that
  `dw_mcp` stays free of `dw` imports.
- It uses the `uv` server type (experimental since manifest v0.4): the host
  provisions the Python environment from the bundle's `pyproject.toml`. The
  fallback, if `uv` proves unreliable in either host, is the `python` type
  with the pure-Python dependencies vendored into `lib/`.
- `user_config`: server URL (default `http://127.0.0.1:8765`) and an optional
  API token, marked sensitive. Those two fields are also what a laptop needs to
  reach a remote GPU box, so the bundle is the future remote-client path.
- The release job builds it with the `mcpb` CLI, attaches it and its SHA-256
  to the GitHub release beside the bootstrap scripts, and smoke-tests that it
  starts and lists tools against the CPU server the docker job already uses.
- The same job publishes a `server.json` to the official MCP Registry
  (`registryType: "mcpb"`, the release-asset URL, `fileSha256`), authenticated
  by GitHub OIDC under the `io.github.dkackman` namespace. The registry is in
  preview, so a failed publish does not fail the release.

**The plugin carries the server.** `plugins/dw` gains an MCP server entry that
launches the bundle, so one plugin install gives skills and tools in Claude
Code and Cowork. setup no longer runs `claude mcp add dw` separately.

- **Verify first:** that Claude Code and Cowork both start a uv-type `.mcpb`
  from a plugin, and how a plugin's server entry passes `user_config`
  (Claude Code prompts for `${user_config.*}`, Cowork does not; the loopback
  default must work with no prompt). If a host can't start the bundle, the
  plugin's server entry launches the installed `dw-mcp` through a stable path
  that setup writes, and the bundle stays a release artifact and registry
  entry only.
- **Desktop chat** still needs `claude_desktop_config.json`, which setup
  writes. Skills reach Desktop chat and Cowork only through a plugin installed
  on the person's claude.ai account: setup cannot do that from the command
  line, so it prints the step (add the GitHub marketplace in Desktop, install
  `dw`). A plugin installed on the account also syncs into Claude Code; setup
  detects that and does not install a second copy.
- **Still no channels in Desktop:** notes reach a Desktop agent when it next
  reads the request.

## Pinokio hooks (designed in, not built now)

- Every `dw setup` step is selectable and non-interactive
  (`--yes`, `--no-service`, `--no-desktop`, `--no-code`, `--no-open`), so a
  Pinokio script can drive it without a TTY.
- `--status --json` is the machine-readable contract a wrapper polls.
- The bootstrap's torch selection is a separate, documented step a Pinokio
  script can call or replicate.

## Non-goals

- A native installer, tray app, or code-signed bundle (option 3). Deferred.
- A chat panel or agent inside the dw UI.
- The Pinokio script itself (hooks only).
- Submitting the `dw` plugin to Anthropic's directory. It is a reviewed
  listing; the marketplace in this repo works without it. A follow-up.
- Client-only setup on a laptop pointing at a remote GPU box. `REMOTE.md`
  covers it by hand today; `dw setup --url` is a natural follow-up.
- Changing how lem or mini-ai are deployed (systemd unit, LaunchDaemon,
  Docker). Those stay as they are.
- Hugging Face login inside the web UI. Setup offers `hf auth login` as an
  optional prompt; a UI flow is a follow-up.

## Anti-goals

- Setup that clobbers a person's other MCP servers, Claude settings, or an
  existing dw service. Every config edit merges, backs up first, and only
  removes what dw itself added.
- A second, silent dw server on the same port (e.g. on a box that already runs
  the systemd unit or LaunchDaemon). Setup detects an existing server and
  registers against it instead of starting another.
- An installed copy frozen on released diffusers. dw follows diffusers HEAD.

## Constraints (from the code today)

- **The wheel already ships the SPA and docs** (`scripts/build_dist.sh`,
  `dw/server/app.py` finds `dw/server/ui`). No npm on the user's machine.
- **The server needs the `server` extra**: install `diffusers-workflow[server,mcp]` (`dw-mcp`, which every Claude registration launches, needs the `mcp` extra).
- **Torch must come from the right index.** `install.sh` picks cu130 for a
  driver >= 580, PyPI's cu12 build for older drivers, CPU without a driver;
  macOS takes PyPI's (pinned `<2.15` in `pyproject.toml`) plus
  `fp4-fp8-for-torch-mps`. torchaudio comes from the same index.
- **diffusers HEAD**: `install.sh` force-reinstalls it from git with
  `--no-deps`, then a plain install fills dependencies.
- **The updater runs `sys.executable -m pip`** (`dw/server/updater.py`). A uv
  tool environment has no pip unless one is added, so the install must include
  pip in the environment (or the updater must learn uv - builder's choice,
  but the button must work).
- **Claude Desktop has no channels.** Notes reach a Desktop agent only when it
  next reads the request. That is accepted for v1; the UI already labels the
  button "Save note" when no session listens.
- **Claude Desktop needs an absolute command path** (no shell `PATH`).
  Config lives at `~/Library/Application Support/Claude/claude_desktop_config.json`
  (macOS) and `%APPDATA%\Claude\claude_desktop_config.json` (Windows). There is
  no Claude Desktop on Linux.
- **Security rules** (`CLAUDE.md`): paths through `dw/security.py`
  validators, subprocess args through `sanitize_command_args()`, no
  `shell=True`. The service binds loopback only; `--mcp` on loopback needs no
  token.
- **Architecture ratchet**: new modules raise the modules baseline, which
  needs Don's approval (precedent: #1687). Count them in the plan.
- **A running Claude session does not pick up a registration change**: setup
  says so at the end.
- **The "Update dw" endpoint is a new API route**, so it follows
  `dw/server/CLAUDE.md` and the OpenAPI pins. Its UI follows `ui/CLAUDE.md`
  and sits beside the existing diffusers update on the Models page.

## Approaches considered

1. **`dw setup` plus a one-line bootstrap script** - chosen. Setup is Python,
   testable and re-runnable; the scripts only get uv, Python and the right
   torch onto the machine. Cost: the person pastes one line into a terminal.
2. **Option 1 plus Pinokio and a `.mcpb` bundle** - chosen with option 1.
   The bundle and the plugin that carries it are built in v1 (above); the
   Pinokio script is deferred with its hooks built now.
3. **Native installer / tray app** - a true double-click, but signing,
   notarization and a torch/CUDA matrix to maintain, and torch still downloads
   at first launch. Deferred until Pinokio shows demand.

Agent side: option "connect to Claude Desktop and Claude Code" chosen over a
chat panel inside the UI (a second agent to build and maintain) and over
leaving the agent out of v1.

## Decided shape

**Bootstrap** - `scripts/install/bootstrap.sh` (macOS, Linux) and
`scripts/install/bootstrap.ps1` (Windows), attached to each GitHub release and
pinned to that release's version, with a published SHA-256. Each:

1. installs uv if missing (uv's own installer),
2. detects the torch build exactly as `install.sh` does,
3. installs `diffusers-workflow[server,mcp]` plus pip into a uv-managed
   environment on Python 3.12+, torch from the detected index,
4. installs diffusers from git HEAD the way `install.sh` does, and the macOS
   extra,
5. runs `dw setup` with any flags passed to the bootstrap.

**`dw setup`** - registrations it makes, each detected first and skipped with a
reason if its target is absent:

| Target | What is written |
|---|---|
| Service, macOS | `~/Library/LaunchAgents/<label>.plist` running `dw-serve --mcp` on 127.0.0.1:8765 |
| Service, Linux | `~/.config/systemd/user/dw-serve.service`, enabled. Setup then asks whether the server should keep running after logout (needed on a headless box reached over ssh). Yes runs `loginctl enable-linger`, with sudo if the distro requires it. `--yes` answers yes. No, or a refused sudo, is reported, and `--status` says the server stops at logout |
| Service, Windows | a per-user logon scheduled task |
| Claude Desktop | `mcpServers.dw` -> absolute `dw-mcp`, merged into the config |
| Claude Code plugin | `claude plugin marketplace add dkackman/diffusers-workflow`, `claude plugin install dw@diffusers-workflow`; the plugin brings the `dw` tool server |
| Claude Desktop plugin | not scriptable: setup prints the steps to add the marketplace and install `dw` in Desktop |
| Claude Code notes | `claude mcp add --scope user dw-notes -- <abs>/dw-mcp --notes-channel` |

Setup records what it added (a small manifest beside dw's settings) so
`--uninstall` removes exactly that and `--status` can tell "ours" from "the
person's own".

**Existing server**: if something already answers `/api/health` on the port,
setup does not install a service; it registers the clients against that server
and says so.

## Documentation

The docs updates are part of the work, not a follow-up. Each lands in the same
change as the code it describes.

- **New `docs/INSTALL.md`**: the full reference for an installed copy. The
  one-liners, what the bootstrap installs and where, the torch choice per
  platform, every `dw setup` flag, the table of what setup writes per OS and
  client, `--status`/`--json` (documented as the wrapper contract),
  `--uninstall` and what it keeps, `dw agent`, how to update (dw itself and
  diffusers), and troubleshooting (port taken, existing server, Desktop not
  found, `claude` not on `PATH`, a session that predates the registration).
  Added to the guide index in `dw/server/guides.py` so `list_guides` and
  `get_guide` serve it: an agent helping someone through a broken setup should
  be able to read it.
- **`README.md`**: the quickstart's three steps (clone + `install.sh`, start
  `dw-serve`, `claude mcp add` + `/plugin install`) become: paste the
  one-liner, open the UI, start `dw agent` or Claude Desktop. The checkout path
  moves under a "From a checkout (development)" heading. The prerequisites list
  drops what uv now provides (system Python, Node) and keeps GPU driver and
  disk. Links to `INSTALL.md`.
- **`docs/DEPENDENCIES.md`**: the installed copy as the first installation
  path; `install.sh`/`install.ps1` described as the checkout path.
- **`docs/MCP.md`**: *Clients* / *Claude Desktop* and the notes channel's *Set
  it up* lead with "`dw setup` does this" and keep the hand-written JSON as the
  manual route. The notes-channel section says that Claude Desktop has no
  channels, so a Desktop agent sees a note when it next reads the request.
  `dw agent` is named as the way to start a listening Claude Code session.
- **`docs/SERVER.md`**: the per-user service setup writes, where its logs go,
  and how to stop or restart it on each OS, including lingering on Linux. The
  "Update dw" button, and when it shows a command instead.
- **`docs/REMOTE.md`**: one line saying `dw setup` is same-machine only for
  now, with the manual remote steps unchanged.
- **`docs/WORKSPACES.md`**: where an installed copy keeps its workspace,
  library and outputs, since there is no checkout `workflows/` folder.
- **`plugins/dw/README.md`**: says `dw setup` installs the plugin, that it now
  carries the `dw` tool server, and how to install it in Claude Desktop (add
  the marketplace, install `dw`) with what loads where (skills everywhere, the
  server in Cowork and Claude Code, not Desktop chat). The in-Claude `/plugin`
  lines stay as the manual route.
- **`docs/MCP.md`** (also): *Clients* gains the `.mcpb` bundle and the plugin's
  server, and which Claude surface uses which registration.
- **`docs/RELEASING.md`**: the release job's new steps (bootstrap scripts,
  the `.mcpb`, their checksums, the MCP Registry publish and its one-time
  setup), and a release-note entry for the installer under *Unreleased*.
- **`docs/ARCHITECTURE.md`**: a row for the setup module (concept, module, the
  rule that holds: setup only removes what its manifest records, and every
  config write merges and backs up) and the test that enforces it.
- **`CLAUDE.md`** *Common Commands*: `dw setup` and `dw agent` beside the
  checkout commands, and that the plugin now carries an MCP server (the
  plugin paragraph).
- **Doc-pinning tests**: wherever the suite pins docs (guide index, README
  links, plugin README), it is updated to match.

## Calls made in design (Claude's, open to correction)

- **diffusers without git.** A clean Mac or Windows machine has no git, and
  asking for it on a Mac triggers the Command Line Tools installer. The
  bootstrap resolves diffusers' current `main` commit and installs that
  commit's GitHub archive, so no git is needed. The updater learns to read
  the commit from that archive install as well as from a git install.
- **Plugin version follows the install.** Setup adds the marketplace at the
  installed release's tag, not the default branch, so the skills match the
  engine. Whether the `claude plugin marketplace add` command takes a ref has
  to be checked first. If it doesn't, setup adds the default branch and
  `--status` reports a version mismatch.
- **Checksums.** The README one-liner fetches
  `releases/latest/download/bootstrap.sh` (or `.ps1`). `INSTALL.md` gives the
  download, verify and run steps for anyone who wants to check the SHA-256
  first. The hashes are attached to every release.

## Left to the builder

- Module layout for `dw setup` / `dw agent` (within the ratchet).
- Exact uv invocation (`uv tool install` with `--with pip`, or a dedicated
  `uv venv --seed`), so long as the updater works and `dw-serve`, `dw-mcp`,
  `dw` land on a stable absolute path.
- Service label names, log locations, restart policy.
- Whether `dw` is a new console script or `dw setup` lives under an existing
  one.
- Test strategy: config merging, manifest round-trip and command building are
  unit-testable with fakes; one manual clean-machine run per OS before release.
