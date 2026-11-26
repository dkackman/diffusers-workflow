# Installing dw

The installed copy is dw without a checkout: one line installs it, `dw setup`
starts the server at login and connects your Claude, and re-running the same
line upgrades it. This page is the reference for that copy. To work on dw
itself, use a checkout instead (the repository README's *From a checkout*).

## Install

macOS and Linux:

```bash
curl -LsSf https://github.com/dkackman/diffusers-workflow/releases/latest/download/bootstrap.sh | bash
```

Windows (PowerShell):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://github.com/dkackman/diffusers-workflow/releases/latest/download/bootstrap.ps1 | iex"
```

You need a GPU driver (an NVIDIA driver 580 or newer gets the CUDA 13 torch
build; Apple Silicon needs nothing extra) and disk: torch and the venv take
several GB, and most models are tens of GB each. You do not need git, Python or
Node: the script brings uv, and uv brings Python. Claude Code and Claude Desktop
are optional; setup connects whichever it finds.

The script ends by running `dw setup`, which asks a question or two on the
terminal (lingering on Linux, a Hugging Face login), opens the web UI at
`http://127.0.0.1:8765`, and says to start a new Claude session.

### Passing options

Anything the script does not recognise goes to `dw setup`
([its flags](#dw-setup)). On macOS and Linux, pass them after `bash -s --`:

```bash
curl -LsSf https://github.com/dkackman/diffusers-workflow/releases/latest/download/bootstrap.sh | bash -s -- --yes --no-desktop
```

On Windows, `irm | iex` takes no arguments, so run the downloaded text as a
script block:

```powershell
& ([scriptblock]::Create((irm https://github.com/dkackman/diffusers-workflow/releases/latest/download/bootstrap.ps1))) -TorchIndex cpu --yes
```

The script's own options:

| macOS, Linux | Windows | Meaning |
| --- | --- | --- |
| `--torch-index URL` | `-TorchIndex URL` | Where torch comes from: an index URL, `pypi`, `cpu` or `cu130` (the short names expand to `https://download.pytorch.org/whl/cpu` and `.../cu130`). Skips the detection below and is recorded |
| `--version X` | `-Version X` | Install dw `X` from PyPI instead of the newest release |

`DW_INSTALL_ROOT` moves the install root (below).

### Check the download first

Every release carries `SHA256SUMS` beside the scripts. To check a script before
running it:

```bash
curl -LO https://github.com/dkackman/diffusers-workflow/releases/latest/download/bootstrap.sh
curl -LO https://github.com/dkackman/diffusers-workflow/releases/latest/download/SHA256SUMS
shasum -a 256 -c --ignore-missing SHA256SUMS     # Linux: sha256sum -c --ignore-missing SHA256SUMS
bash bootstrap.sh
```

On Windows, download `bootstrap.ps1` and `SHA256SUMS` from the release page,
compare `(Get-FileHash .\bootstrap.ps1).Hash` with the `bootstrap.ps1` line
(case does not matter), then run
`powershell -ExecutionPolicy ByPass -File .\bootstrap.ps1`.

## What the bootstrap installs, and where

1. uv, with uv's own installer, when `uv` is not on `PATH` (it lands in
   `~/.local/bin`).
2. A Python 3.12 venv made with `uv venv --seed`, so it has pip (the update
   buttons use it).
3. torch, torchvision and torchaudio, from the index chosen below.
4. `diffusers-workflow[server,mcp]`, with torch pinned to the build just
   installed so the resolver cannot swap it.
5. diffusers `main`, from the GitHub archive of its current commit (no git
   needed), then a plain `diffusers` install for any dependency that commit
   adds. When GitHub's API cannot be read it warns and installs released
   diffusers; the Models page moves it to `main` later.
6. The platform extras: on macOS `fp4-fp8-for-torch-mps` and `fluidtop`, on
   Windows `bitsandbytes` and `kernels`.
7. The install record, then `dw setup`.

Everything goes under the install root:

| | macOS, Linux | Windows |
| --- | --- | --- |
| Install root | `~/.local/share/diffusers-workflow` | `%LOCALAPPDATA%\diffusers-workflow` |
| The venv | `<root>/venv` | `<root>\venv` |
| The `dw` command | `<root>/venv/bin/dw` | `<root>\venv\Scripts\dw.exe` |
| Install record | `<root>/install.json` | `<root>\install.json` |

The install record is `{"torch_index": ..., "version": ..., "diffusers_commit": ...}`:
the torch index (a URL or `pypi`), the dw version installed and the diffusers
commit (or `null`). An upgrade reads it to rebuild the venv the same way.

**`dw` is not put on your `PATH`.** Run it by its full path
(`~/.local/share/diffusers-workflow/venv/bin/dw setup --status`), or add the
venv's `bin` (`Scripts` on Windows) to your `PATH`. The commands below write
plain `dw`.

### The torch build

| Machine | torch from |
| --- | --- |
| macOS | PyPI, pinned below 2.15 (2.15 with the fp4-fp8 MPS autoload breaks `scalar ** mps_tensor`) |
| NVIDIA driver reporting CUDA 13 or newer (driver 580+) | `https://download.pytorch.org/whl/cu130` |
| An older NVIDIA driver | PyPI (its CUDA 12 build) |
| No working `nvidia-smi` | `https://download.pytorch.org/whl/cpu` |

This is the same detection `install.sh` uses. It runs once: the choice is
recorded in `install.json`, and a re-run reuses it unless `--torch-index` is
given. To change it later, re-run the one-liner with `--torch-index`. A wrapper
that wants to make the choice itself (a Pinokio script, say) passes
`--torch-index` and the detection never runs.

## dw setup

`dw setup` is safe to re-run. In order, it:

1. reports the dw version and the install record;
2. asks `GET /api/health` on `127.0.0.1:8765` whether a server is running. A
   server setup's own service started (the manifest records the service and
   the service manager reports it) is treated like no server: step 3 rewrites
   the service and restarts it, which is how a re-run of the one-liner starts
   the new version. Any other server is left alone and gets no service;
   Claude is registered as usual, and the clients reach whatever answers on
   8765;
3. installs the per-user service and starts it (or restarts it). When the
   running server has a job running or queued, setup warns that a restart
   stops it and asks first; without a terminal it does not restart, and under
   `--yes` it restarts after the warning;
4. registers dw with Claude Desktop (macOS, Windows) and prints the steps to
   add the plugin there;
5. writes the `dw-mcp` shim, which both Claude Code and Cowork start the
   plugin's tool server through (skipped only under both `--no-code` and
   `--no-desktop`);
6. registers dw with Claude Code, when it finds `claude` (on `PATH`, or in
   `~/.local/bin`, `~/.claude/local`, `/opt/homebrew/bin`, `/usr/local/bin`,
   or `%APPDATA%\npm` on Windows), and warns about a duplicate `dw` server;
7. offers `hf auth login` when no Hugging Face token is found;
8. opens the web UI, and says to start a new Claude session.

Each step whose target is absent is skipped with the reason. A step that fails
prints why, the rest still run, and setup exits 1.

| Flag | Meaning |
| --- | --- |
| `--yes` | Take every default without asking, and answer yes to lingering. Skips the Hugging Face prompt |
| `--no-service` | Install no service |
| `--no-desktop` | Leave Claude Desktop alone |
| `--no-code` | Leave Claude Code alone |
| `--no-open` | Do not open the web UI |
| `--status` | Report what is installed, running and registered; change nothing |
| `--json` | With `--status`, print the report as JSON |
| `--uninstall` | Remove what setup added |
| `--port N` | The port setup checks for a running server, reports and opens (default 8765). It changes nothing else: the service always listens on 8765 and the registrations are not port-specific |

`--port` exists for a server you run yourself on another port: `dw setup
--no-service --port 8800` finds it there and opens its web UI. The Claude
registrations still reach the server by the plugin and `dw-mcp` defaults
(port 8765); point them elsewhere by hand ([MCP.md](MCP.md)).

Prompts are read only from a terminal, and never under `--yes`; without a
terminal each takes its default. Under `curl | bash` the script hands setup
`/dev/tty`, so the prompts still reach you.

### What setup writes

| Target | What is written |
| --- | --- |
| Service, macOS | the launchd agent `~/Library/LaunchAgents/com.github.dkackman.dw-serve.plist`, started at login and kept alive. Logs: `~/Library/Logs/diffusers-workflow/dw-serve.log` and `dw-serve.err.log` |
| Service, Linux | the systemd user unit `~/.config/systemd/user/dw-serve.service` (`Restart=always`), enabled and started. Logs: `journalctl --user -u dw-serve`. Setup then asks whether to enable lingering (below) |
| Service, Windows | the scheduled task `dw-serve`, started at your logon, restarted on failure up to 3 times a minute apart |
| Claude Desktop | `mcpServers.dw` = `{"command": "<root>/venv/bin/dw-mcp", "args": []}`, merged into `claude_desktop_config.json` |
| Claude Desktop plugin | nothing: setup prints the steps (below) |
| The shim | `~/.local/share/diffusers-workflow/bin/dw-mcp` (below) |
| Claude Code | `claude plugin marketplace add dkackman/diffusers-workflow@v<version>`, `claude plugin install dw@diffusers-workflow`, and `claude mcp add --scope user dw-notes -- <root>/venv/bin/dw-mcp --notes-channel` |

Every executable setup registers is the one beside the `dw` that runs it:
`<root>/venv/bin` for an installed copy (`<root>\venv\Scripts` on Windows),
the checkout's `venv/bin` when a checkout runs `venv/bin/dw setup`.

Each service runs `dw-serve --host 127.0.0.1 --port 8765 --mcp --managed`
from there, with the install root as its working directory (so the server
never takes your home directory for a workspace) and, when you set
`DW_INSTALL_ROOT`, that variable too (the Windows task passes it as
`--install-root`, since a task has no environment of its own). `--managed` is
what lets the web UI's **Update dw** button restart it ([Updating](#updating)).

Claude Desktop's config is
`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS and
`%APPDATA%\Claude\claude_desktop_config.json` on Windows. The Microsoft Store
build keeps it under
`%LOCALAPPDATA%\Packages\Claude_*\LocalCache\Roaming\Claude\`; setup uses
whichever of those holds a config, else whichever directory exists. There is no
Claude Desktop on Linux.

The shim is the stable path the plugin's tool server launches (the plugin
names it as `${HOME}/.local/share/diffusers-workflow/bin/dw-mcp`, the same on
every OS). It is a symlink to the venv's `dw-mcp`; on Windows it is a copy,
`dw-mcp.exe`, which each `dw setup` refreshes when the venv's copy has
changed. A running Claude holds that file open: setup then says to close
Claude Code and Claude Desktop and run it again.

The plugin's marketplace is added at the installed release's tag
(`v0.23.0-alpha.1` for dw `0.23.0a1`), so the skills match the engine. A
version with no tag yet (a checkout between releases) falls back to the
default branch, and `--status` reports the plugin version mismatch. Each
Claude Code step is skipped when `claude plugin marketplace list`, `claude
plugin list` or `claude mcp list` already shows it, except that a re-run at a
new version moves the marketplace setup added: it removes it, adds it at the
new tag and reinstalls the plugin. A marketplace you added yourself is never
moved; setup reports the ref it is pinned at.

**What setup never does.** It changes only what its manifest records. Every
config write merges into the file and leaves your other MCP servers and
settings alone; the first change backs the original up to `<file>.dw-backup`,
and that backup is never overwritten. A config file that does not parse is
refused and left as it is. A `dw` entry in Claude Desktop, a launchd plist,
systemd unit or scheduled task, or a file at the shim's path that setup did not
write (or one changed since it wrote it) is left alone and reported. A
Claude Code MCP server named `dw`, or one
that launches `dw-mcp` (other than the notes channel), duplicates the plugin's
tools: setup prints `claude mcp remove <name>` for it and never removes it.

The manifest is `setup-manifest.json` in dw's settings directory
(`~/.diffusers_helper/`, or `DIFFUSERS_HELPER_ROOT`).

### Lingering on Linux

A systemd user service stops when you log out, unless lingering is enabled for
your user. A headless box you reach over ssh needs it. Setup asks (default no;
`--yes` answers yes) and runs `sudo loginctl enable-linger <you>`. A no, or a
refused sudo, is reported, and `--status` then says the server stops at
logout. Run `sudo loginctl enable-linger $USER` yourself at any time.

## Which Claude uses what

| Where | Tools | Skills | Notes channel |
| --- | --- | --- | --- |
| Claude Code | the plugin's `dw` server, through the shim | the plugin | `dw-notes`, in a session started with `dw agent` |
| Cowork | the plugin's `dw` server | the plugin, installed on your claude.ai account | none |
| Claude Desktop chat | the `dw` entry in `claude_desktop_config.json` (Desktop chat ignores a plugin's server) | the plugin, installed on your claude.ai account | none |

Claude Desktop chat has no channels: a note you save in the web UI reaches a
Desktop agent when it next reads the request.

**The plugin in Claude Desktop** is a manual step. Setup prints it:

1. Open Customize > Plugins (or Settings > Plugins) and choose Add marketplace.
2. Enter `dkackman/diffusers-workflow`.
3. Install the `dw` plugin from that marketplace.

A plugin installed on your account also syncs into Claude Code; when `claude
plugin list` already shows `dw@diffusers-workflow`, setup does not install a
second copy.

**The `.mcpb` bundle.** Each release also carries `dw-mcp-<version>.mcpb`, the
MCP server alone, for a Claude Desktop double-click install and the
[MCP Registry](https://registry.modelcontextprotocol.io) (`io.github.dkackman/dw`).
It asks for the server URL (default `http://127.0.0.1:8765`) and an optional
token. Claude Code does not run it, and setup does not use it. Its launch is
`uv run --frozen`, so the dependencies locked at release are used when the
host runs the manifest's launch command; that is unverified for Desktop.

### The plugin without dw setup

The plugin now starts a `dw` tool server through the shim. A machine that has
the plugin but never ran `dw setup` - a checkout, or a laptop that uses a
remote GPU box - has no shim, so that server fails to start (`/mcp` lists it as
failed). Either:

- from a checkout, run `venv/bin/dw setup --no-service --no-desktop`, which
  writes the shim pointing at the checkout's venv; or
- keep using the `dw` registration you added by hand (an HTTP one to a remote
  box, say) and ignore the failed plugin server. Setup, when run, reports that
  hand-added `dw` beside the plugin's as a duplicate and never removes it.

## dw setup --status

`dw setup --status` prints one line per part:

```
dw 0.23.0 installed at /home/you/.local/share/diffusers-workflow
install record: {'torch_index': 'https://download.pytorch.org/whl/cu130', 'version': '0.23.0', 'diffusers_commit': '...'}
server http://127.0.0.1:8765: answering, managed by the service
service: installed, running
Claude Desktop: absent, not registered
Claude Code: /home/you/.local/bin/claude; plugin 0.23.0; notes channel present
missing: nothing
```

`--status --json` is the machine-readable contract: a wrapper (a Pinokio
script, say) polls it. It always exits 0. The keys:

| Key | Value |
| --- | --- |
| `version` | the dw version setup runs as |
| `install_root` | the install root |
| `install_record` | `install.json` as an object, or `null` |
| `server` | `url`; `answering` (`/api/health` answered within 2 s); `managed` (answering and the service is running) |
| `service` | `installed`; `running`; `stops_at_logout` (`true` on macOS and Windows, from lingering on Linux, `null` when it cannot be told) |
| `desktop` | `present` (Claude Desktop's config directory exists); `registered` (a `dw` entry is in the config); `ours` (setup wrote it) |
| `code` | `cli` (the `claude` path, or `null`); `plugin`; `plugin_version`; `notes_channel`; `duplicates` (Claude Code MCP servers that duplicate the plugin's tools, other than ones setup added) |
| `plugin_version_mismatch` | `true` when the installed plugin's version is not dw's, compared as versions (`0.23.0-alpha.1` matches `0.23.0a1`) |
| `missing` | a list of what is absent: `install record`, `server not answering`, `service`, `Claude Desktop registration`, `Claude Code plugin`, `dw-notes channel` |

A Claude Desktop or Claude Code item is listed in `missing` only when that
client is present.

## dw setup --uninstall

`dw setup --uninstall` asks once (default no; `--yes` skips the question), then
removes what the manifest records, in this order: the Claude Code plugin, its
marketplace and the `dw-notes` server; the shim (only if it is still the link
or copy setup made); the Claude Desktop `dw` entry (only if its command is
still the one setup wrote); the service. One you already removed yourself is
forgotten. With no manifest entries it says so and changes nothing. When
`claude` cannot be found, the Claude Code entries stay recorded for a later
run.

It keeps the install root and its venv, the models in the Hugging Face cache,
your workspace, dw's caches and settings, and the `.dw-backup` files. Lingering
is left enabled; setup prints `sudo loginctl disable-linger $USER`. To remove dw
completely, uninstall first, then delete the install root.

## dw agent

```bash
dw agent [claude arguments...]
```

starts Claude Code with the notes channel loaded:
`claude --dangerously-load-development-channels server:dw-notes`, followed by
any arguments you pass. It finds `claude` where setup does; without one it
says where to get it. The
channel is a Claude Code research preview; [the notes channel](MCP.md#the-notes-channel)
says what it does.

## Updating

**dw.** Re-run the one-liner. Over an existing install it keeps the recorded
torch index, pins torch to the installed build, upgrades dw to the newest
release, moves diffusers to `main`, rewrites `install.json` and re-runs
`dw setup`, which refreshes the registered paths, rewrites the service and
restarts it ([step 2](#dw-setup)), and moves the plugin's marketplace to the
new release's tag and reinstalls the plugin. On Windows the script first ends
the `dw-serve` task, because a running `dw-serve.exe` cannot be replaced;
setup starts it again. On macOS and Linux the files are replaced while the old
server runs, and the restart picks them up.

`plugin_version_mismatch` stays `true` when the marketplace could not follow
(the version has no tag yet, or you added the marketplace yourself); it clears
at the next release that has one, or after you remove your own marketplace
and run `dw setup` again.

**The Update dw button.** The Models page shows the installed dw version and,
when PyPI has a newer release, an **Update dw** button. It runs the same
upgrade inside the server (dw with torch held, diffusers to `main`, then
`dw setup --yes --no-open --no-service`) and exits so the service manager
restarts the server. The `dw setup` refresh is best effort: once pip has
installed the new packages the server restarts even if the refresh fails,
and its output is in the update log; run `dw setup` again then. It works only on a server setup's service started
(`--managed`); any other server (a checkout, the systemd unit in
`contrib/systemd`, a LaunchDaemon, Docker) shows the one-liner to run instead.
[SERVER.md](SERVER.md#the-service-dw-setup-installs) has the rules.

**diffusers alone.** The Models page's **Update from GitHub** moves diffusers to
`main`. On an installed copy, which has no git, it installs the GitHub archive
of `main`'s commit.

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `A dw server is already running at ... that setup's service did not start; no service installed.` | Something else answers on port 8765: an earlier `dw-serve`, the `contrib/systemd` unit, a LaunchDaemon or Docker. Setup registers Claude, which reaches it. Stop it first if you want setup's service instead |
| `service: not restarted; the running server keeps the old version ...` | The server had a job running or queued and the restart was declined (or there was no terminal to ask). Run `dw setup` again when it is idle |
| The service is installed but `--status` says `server not answering` | Another program holds port 8765 and does not answer `/api/health`, or `dw-serve` failed to start. Read the service log ([What setup writes](#what-setup-writes)); on Windows, run `dw-serve` from the venv by hand to see the error |
| `Claude Desktop is not installed; skipped` | Setup looks for Desktop's config directory. Start Claude Desktop once, so it creates it, then run `dw setup` again |
| `Claude Code: skipped, Claude Code not found` | `claude` is neither on `PATH` nor in the places setup looks ([dw setup](#dw-setup), step 6). Install Claude Code, open a new terminal, and run `dw setup` again |
| `Argument contains dangerous characters: ...` | Setup refuses to pass an argument holding `` $ & ; \| < > `` or a backtick to any program, and your home or user path (or `DW_INSTALL_ROOT`) contains one. Set `DW_INSTALL_ROOT` to a path without them and re-run the one-liner |
| `cannot replace ...dw-mcp.exe ... Close Claude Code` (Windows) | A running Claude holds the shim open. Close Claude Code and Claude Desktop, then run `dw setup` again |
| `... is not valid JSON` | A Claude config does not parse. Setup changed nothing; fix the file, then run setup again |
| `... already exists and setup did not create it` | A service definition or task setup did not write is in the way. Remove or rename it, or pass `--no-service` |
| Claude does not list dw after setup | A running session does not pick up a registration change. Start a new session (quit and reopen Claude Desktop) |
| The plugin's `dw` server failed in `/mcp` | The shim is missing: see [The plugin without dw setup](#the-plugin-without-dw-setup) |
| The server stops when you log out (Linux) | Lingering is off: `sudo loginctl enable-linger $USER` |
| `Warning: Claude Code already has an MCP server named ...` | A hand-added `dw` registration duplicates the plugin's tools. Remove it with the `claude mcp remove` line setup printed |

## Known limits

- **Same machine only.** Setup connects Claude on the machine it runs on.
  A laptop using a GPU box elsewhere is set up by hand ([REMOTE.md](REMOTE.md)).
- **The examples are read-only.** The package carries the repository's
  `workflows/`, `prompts/` and `loras/` and the server offers them beside your
  workspace's own; saving an edited one makes your own copy in the workspace
  ([WORKSPACES.md](WORKSPACES.md#an-installed-copy)).
- **`${HOME}` on Windows.** The plugin names the shim through `${HOME}`, which
  is verified in Claude Code on macOS and unverified on Windows.
- **Windows restarts.** The scheduled task restarts a failed server 3 times per
  login; after that it waits for the next logon (or `schtasks /Run /TN dw-serve`).
- **No token on loopback.** The service binds `127.0.0.1` with no token, as
  `dw-serve` does by default. On a machine shared by several user accounts, the
  other local accounts can reach it.

## For maintainers

The bootstrap scripts read a few environment variables that exist for tests and
pre-release checks, not for an install:

- `DW_BOOTSTRAP_DRY_RUN` - print each command (prefixed `+ `) instead of running
  it; nothing is written.
- `DW_BOOTSTRAP_WHEEL` - install this wheel instead of the PyPI release.
- `DW_BOOTSTRAP_DIFFUSERS_SHA` - use this diffusers commit instead of asking
  GitHub for `main`'s (40 lowercase hex characters, or it is ignored).
- `DW_BOOTSTRAP_TTY` (`bootstrap.sh` only, test-only) - with
  `DW_BOOTSTRAP_DRY_RUN`, `1` or `0` says whether `/dev/tty` is available instead
  of probing it. Without a dry run it is ignored.

The scripts ship inside the package (`dw/install_scripts/`) and are attached to
each GitHub release with `SHA256SUMS`. `docs/RELEASING.md` covers the release
job.
