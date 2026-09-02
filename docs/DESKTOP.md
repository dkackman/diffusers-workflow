# Desktop App

`diffusers-workflow` ships as a native app for macOS (Apple Silicon),
Windows and Linux. It is the same engine, server and web UI as the
command line install — the app adds an installer, a window, and a
supervisor that starts and stops the server for you.

## What the installer actually contains

The download is small (roughly 60–100 MB). It carries a private Python
interpreter and [uv](https://github.com/astral-sh/uv), both pinned by
version and SHA256, and nothing else.

Everything else is fetched on first launch: PyTorch and the ~4 GB of
Python packages the engine needs. Models are downloaded later still, the
first time you run a workflow that needs one.

Nothing is frozen into a single binary. The app builds a real, writable
virtual environment, because three things depend on that being real:

- the engine resolves pipeline and quantization classes by name at
  runtime, so a frozen bundle would have to enumerate every diffusers
  pipeline ahead of time and would break the "new backends work
  automatically" property;
- the persistent worker re-executes the interpreter (`spawn`);
- the Models page upgrades `diffusers` by running pip against its own
  environment.

## First launch

The app detects your hardware and offers a choice before installing.

| Platform | What it installs |
| --- | --- |
| macOS (Apple Silicon) | The standard PyTorch wheel, which is the MPS-accelerated one |
| Windows / Linux with a recent NVIDIA driver | A CUDA build of PyTorch |
| Everything else | A CPU build |

The CUDA choice is made from the **driver** version — PyTorch's CUDA
wheels bundle their own runtime, so no CUDA toolkit install is needed.
Driver 580 or newer gets CUDA 13; 525 or newer gets CUDA 12; anything
older falls back to CPU and says so. You can always override the
detected choice to CPU in the dropdown.

Installing takes roughly 5–15 minutes depending on your connection. If
it is interrupted, the next launch notices and resumes — completion is
recorded only after the whole sequence succeeds.

## Where things go

| What | macOS | Windows | Linux |
| --- | --- | --- | --- |
| Python environment, logs | `~/Library/Application Support/diffusers-workflow` | `%LOCALAPPDATA%\diffusers-workflow` | `~/.local/share/diffusers-workflow` |
| Workflows, prompts, outputs | `~/Documents/diffusers-workflow` | `Documents\diffusers-workflow` | `~/Documents/diffusers-workflow` |
| Settings, job history | `~/.diffusers_helper` | `~/.diffusers_helper` | `~/.diffusers_helper` |
| Models | the standard Hugging Face cache | same | same |

Your workflows and prompts are seeded with the examples on first run.
After that they are yours: an app update never overwrites a file that
already exists.

## Using the command line and the REPL

The installer does **not** modify your `PATH` or your shell
configuration. The tools are all installed in the app's environment —
**Developer…** in the app menu shows the exact path and has an *Open
terminal here* button.

Available there: `dw-run`, `dw-validate`, `dw-repl`, `dw-serve`,
`dw-mcp`.

One caution: `dw-repl` starts its **own** persistent worker, so running
it while the app is open puts two processes on one GPU. The REPL warns
you when it detects a running server. Running both is still reasonable
when you mean to — pinning a step to the CPU while the app works is a
normal way to isolate a GPU problem.

## Connecting an MCP client

**Connect to Claude…** in the app menu shows a ready-made configuration
block. It names the `dw-mcp` in the app's environment and carries no
port: the server publishes whichever port it bound to
`~/.diffusers_helper/server.json`, and the client reads it from there.
A configuration you save today keeps working even if the app later has
to fall back to a different port.

The app prefers port **8765** and only moves if something else holds it.

## Updates

The app updates itself; on the first launch after an update it brings
the Python environment to the matching version.

`diffusers` is separate. The app installs the released version, and the
**Models** page upgrades it to the development version when you want a
pipeline that has not been released yet — see
[ACCELERATION.md](ACCELERATION.md) and the Models page itself.

## Troubleshooting

**Windows shows a SmartScreen warning.** The Windows build is not code
signed yet. Choose *More info* → *Run anyway*. The macOS build is signed
and notarized.

**The install failed partway.** Open **Developer…** and choose *Repair
on next launch*, then restart the app. This rebuilds the Python
environment and leaves your workflows, prompts, outputs and downloaded
models alone.

**The server did not start.** **Server Logs…** in the app menu shows the
engine's own output, which is where the reason will be.

**It installed a CPU build but I have an NVIDIA GPU.** Update your
graphics driver, then use *Repair on next launch* — the hardware check
runs again during the rebuild.
