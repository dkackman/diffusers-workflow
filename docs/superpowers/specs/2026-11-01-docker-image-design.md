# Docker image - design

Status: built on feat/docker-image (2026-11-01); lem GPU test pending. The guide is docs/DOCKER.md. Tauri desktop notes at the end; they are a
follow-on, not part of this spec's scope.

## Intent

Make dw installable on a Linux/NVIDIA box with one `docker run` (or
`docker compose up`): no Python, no venv, no torch-index guesswork. This is
the first step toward packaging dw for end users. The second is a Tauri
desktop app (notes below).

Don's constraint: **diffusers tracks its GitHub main HEAD**, because the newest
model pipelines land there before a release. That means diffusers can't be
frozen into the image. It lives on a volume where it can be updated and where
updates survive restarts, container re-creation and image upgrades.

Docker is the Linux/NVIDIA (and remote-server, `docs/REMOTE.md`) channel only.
Docker Desktop on a Mac runs a Linux VM with no MPS, so Apple Silicon users
would get CPU only. Macs are the Tauri app's job.

## How ComfyUI does it (for reference)

- Git clone plus pip is the base install. The Windows portable 7z (embedded
  Python, torch preinstalled) and comfy-cli wrap it.
- ComfyUI Desktop (Electron) bundles uv and a standalone Python, **installs
  torch on first run** for the detected GPU, pins the ComfyUI core inside the
  app, and keeps models and user data in a folder outside the app.
- The frontend ships as its own pip package (`comfyui-frontend-package`).
  dw already does the equivalent: `scripts/build_dist.sh` copies the SPA
  into `dw.server`.
- There is no official first-party Docker image. Community images
  (yanwk/comfyui-boot, ai-dock) fill the gap.

Lessons taken: torch is chosen per machine; user data lives outside the
thing that gets replaced; the engine version is pinned by the package and
not by whatever HEAD was on the day of install. diffusers is the one
deliberate exception, and this design gives it its own volume.

## Layering

```
image (immutable, rebuilt per dw release)
  /usr/local/bin/python        python:3.13-slim system interpreter
  site-packages                torch (cu130 or cu12), dw wheel [server,mcp],
                               every pinned dependency, and a *baked* diffusers
                               (a git commit chosen at build time)
  /opt/dw/examples             the repo's workflows/ + prompts/ (read-only catalog)

volume /data (persistent, owned by the user)
  env/venv                     the diffusers overlay: a venv made with
                               --system-site-packages over the image's Python
  env/overlay.json             marker: image build id, Python version,
                               installed diffusers commit
  workspace/                   DW_WORKSPACE - workflows, prompts, assets, outputs
  hf/                          HF_HOME - model weights, incl. dw-quantized/
  cache/                       XDG_CACHE_HOME (torchinductor), TRITON_CACHE_DIR
  settings/                    DIFFUSERS_HELPER_ROOT - settings.json
  home/                        HOME for the non-root user
```

### Why an overlay venv, not the whole venv on the volume

With the whole venv on the volume, the image would be nothing more than a
bootstrap script, and an upgrade would mean reinstalling torch (several GB)
into the volume. The overlay holds only what differs from the image:

- `python -m venv --system-site-packages /data/env/venv` sees every image
  package. Its own site-packages comes first on `sys.path`, so a diffusers
  installed there shadows the baked one.
- pip run from the overlay leaves the image's copy alone ("Not uninstalling
  diffusers ... outside environment") and installs into the overlay. It adds
  only the dependencies the new diffusers commit needs that the image doesn't
  already satisfy.
- **The existing updater works unchanged.** `build_pip_commands` runs
  `sys.executable -m pip`. The server runs under the overlay's Python, so the
  Models page's HEAD / pinned-commit / revert actions land on the volume, and
  the worker restart after an update behaves as it does today.

A volume holding a git checkout of diffusers (editable install) was rejected.
pip's `git+` install is what install.sh and the updater already do, and
a checkout adds a second update path to keep consistent.

### Overlay lifecycle (the entrypoint)

The entrypoint is `docker/entrypoint.py`, run by the image's interpreter.
It's a script outside `dw/`, so it adds no engine module to the ratchet. It
imports `dw.security` (`sanitize_command_args`, `validate_commit_hash`) and
`dw.server.updater` (`build_pip_commands`, which now takes the interpreter
to install into); no `shell=True`. tests/test_docker.py loads it by path. On every start it:

1. **Creates** the overlay if `/data/env/venv` is missing, then installs the
   diffusers that `DW_DIFFUSERS_REF` names. The two install commands are the
   same ones `build_pip_commands` builds.
2. **Rebuilds** it if `overlay.json`'s image build id or Python minor version
   doesn't match the running image. An old overlay can hold dependency
   versions that would now shadow newer ones the image ships. The rebuild
   reinstalls the *recorded* diffusers commit, not HEAD, so an image upgrade
   never silently moves diffusers.
3. **Smoke-tests** the overlay with `python -c "import diffusers, dw.server"`.
   If that fails (a broken HEAD; transformers regressions have broken HEAD
   before), it logs the error and serves from the image's Python with the
   baked diffusers instead, so the server still comes up and the UI can be
   used to pin a working commit.
4. **Execs** `<chosen python> -m dw.serve --host 0.0.0.0 --examples-dir
   /opt/dw/examples ...`. User flags pass through: `docker run ... image
   --devices cuda:0,cuda:1 --mcp`.

`DW_DIFFUSERS_REF` (validated):

| value | meaning |
|---|---|
| `head` (default) | install main HEAD **when the overlay is created**; afterwards update from the Models page or `entrypoint.py update-diffusers` |
| a 40-hex commit | install that commit at creation |
| `image` | no overlay diffusers; use the baked one |

**A restart never updates diffusers.** HEAD moves only when the user asks,
so restarting a container is reproducible. Automatic updates on start
(`DW_DIFFUSERS_UPDATE=on-start`) are a possible later addition.

`python /opt/dw/entrypoint.py update-diffusers [--commit <sha> | --revert]`
(through `docker compose exec`) runs the same `build_pip_commands` the UI
button runs.

If the first start has no network, the overlay install fails, the
entrypoint logs it, and the server falls back to the baked diffusers. The
image works offline.

## The image

Multi-stage `Dockerfile` at the repo root:

1. **ui stage** (`node:<pinned>`): `npm ci && npm run build`, then the
   `build_dist.sh` steps (copy the SPA into `dw/server/ui`, the docs into
   `dw/docs`), then `python -m build` produces the wheel.
2. **runtime stage** (`python:3.13-slim-bookworm`):
   - apt: `git` (for pip's git installs), `build-essential` (Inductor and
     Triton compile kernels at run time), `tini`.
   - torch from `ARG TORCH_INDEX`, in install.sh's order: torch first, then
     the wheel with `[server,mcp]`, then
     `pip install --force-reinstall --no-deps git+...diffusers@${DIFFUSERS_REF}`
     and a plain `pip install diffusers` for its dependencies.
   - Install pins come from a constraints file generated per release, so a
     rebuild of the same tag resolves the same versions. Today only the
     pyproject floors exist; the constraints file is new work. The
     `<0.3` cap on comfy-kitchen is already a pin.
   - No CUDA base image is needed. torch wheels bundle the CUDA runtime
     (`nvidia-*` packages), and the host's driver comes in through the NVIDIA
     Container Toolkit.
   - A non-root `dw` user, `HOME=/data/home`, `WORKDIR /data`.
   - Labels: dw version, the baked diffusers commit, the torch build, and the
     image build id the overlay marker compares against.
   - `HEALTHCHECK` against `GET /api/health`.

**Decided: cu130 only, and build it yourself for now.** Nothing is
published to a registry yet; `docker compose build` (or `docker build -t
dw .`) on the GPU box is the install. The requirements are an NVIDIA driver
>= 580 and a Turing or newer card: CUDA 13 dropped Maxwell, Pascal and
Volta. The only reason to add a cu12 variant would be a user stuck on
an older driver branch, or on a V100 or a 10-series card. `ARG TORCH_INDEX`
keeps that a one-line build override (`--build-arg
TORCH_INDEX=https://pypi.org/simple`), with no published tag.
linux/amd64 only at first; arm64 (GH200, DGX Spark) comes later if someone
asks.

Expected size: about 8-10 GB, mostly torch and the CUDA libraries. Model
weights are never baked in; they go to `/data/hf`.

## Running it

**As built: shared with a native install.** `docker/init-env.sh` writes
`docker/.env` from the host's native layout (workspace, HF_HOME, the
Inductor and Triton caches, the settings directory). compose mounts each one
**at the same absolute path** and sets the variable dw reads. Manifests,
settings and quantized-cache entries record absolute paths, so the
container and a systemd dw.serve read each other's records.
`docker/switch.sh docker|native|status` stops one and starts the other.
Only `DW_ENV_DIR` (the overlay) is the container's own. The single-`/data`
layout above is still the image's default when it runs without compose.
The sketch below is the original; `docker/compose.yaml` is what shipped.

`docker-compose.yml` in the repo:

```yaml
services:
  dw:
    build: .
    image: diffusers-workflow:local
    ports: ["127.0.0.1:8765:8765"]
    volumes:
      - ./dw-data:/data            # or split: ~/.cache/huggingface:/data/hf
    environment:
      DW_API_TOKEN: ${DW_API_TOKEN:-}
      HF_TOKEN: ${HF_TOKEN:-}      # gated models
    deploy:
      resources:
        reservations:
          devices: [{driver: nvidia, count: all, capabilities: [gpu]}]
    restart: unless-stopped
```

- Every `/data/*` subpath can be its own mount, and each env var can be
  overridden, so an existing host HF cache can be shared instead of
  downloading models twice.
- Multi-GPU works the way it does on lem: `--devices cuda:0,cuda:1`. The
  worker manager's `CUDA_VISIBLE_DEVICES` pinning works inside the
  container.
- `--user $(id -u):$(id -g)` with a bind mount keeps output files owned by
  the host user. The entrypoint must not need root.

## Security

- The server has to bind `0.0.0.0` inside the container, and it cannot see
  whether the port is published on localhost only or everywhere.
  `dw.serve` warns on a non-loopback host without a token, and refuses
  `--mcp` there. **Decided: Jupyter's pattern.** If `DW_API_TOKEN` is unset
  on first start, the entrypoint generates a token, saves it to
  `/data/settings/token` (mode 0600). Unlike Jupyter, it never logs the
  token: CodeQL flagged that (alert 85 on PR #1156), and because this token
  outlives the container, a `docker logs` paste in an issue would leak a
  working credential. `entrypoint.py login-url` prints the
  `http://localhost:8765/?token=<token>` link when asked. Also unlike Jupyter,
  the token persists across restarts, so an MCP client configured with the
  header keeps working. Setting `DW_API_TOKEN` overrides it.
  `DW_INSECURE_NO_TOKEN=1` is the explicit opt-out, for a port published on
  localhost only.
  - **UI work:** the SPA has a token prompt and appends the token to media
    URLs (`ui/src/lib/apiUrls.ts`), but it does not read `?token=` from the
    page URL. It needs to: save the token the way the prompt does, then
    remove it from the address bar with `history.replaceState` so it stays
    out of history and screenshots.
- `--trust-workflows` stays off by default; the container changes nothing
  about the trust model. The usual `docs/SECURITY.md` rules apply.
- The overlay venv lives on a volume the container writes to, so anything
  that can write to the volume can change the code the server runs. That's
  no different from a venv on disk, but the docs should say it.
- Entrypoint subprocesses use fixed argument lists through
  `sanitize_command_args`. `DW_DIFFUSERS_REF` goes through
  `validate_commit_hash` or the two literal keywords.

## Known wrinkles

- `release_floor()` reads `pyproject.toml` relative to the source tree. From
  a wheel it falls back to `FALLBACK_RELEASE_FLOOR`. That already works, but
  in the container "revert" could more usefully mean "drop the overlay's
  diffusers and use the baked one". That is a small updater change, and
  optional.
- If the smoke test failed and the server is running on the image's Python,
  the UI updater would pip into the container's writable layer, and that
  install is lost when the container is re-created. In that mode the
  entrypoint should make the updater rebuild the overlay instead. Exactly
  how that hand-off works still needs designing.
- flash_attn needs nvcc, and the image doesn't have it. Out of scope. The
  backends that ship as Hub kernels (`kernels`, sage_hub) and comfy-kitchen's
  prebuilt Linux kernels do work.
- lem and mini-ai keep their venv installs. Docker is an additional channel
  and replaces nothing.

## Testing

- pytest for `docker/entrypoint.py` (tests/test_docker.py): marker mismatch leads to a rebuild; the
  recorded commit (not HEAD) is reinstalled on a rebuild; `DW_DIFFUSERS_REF`
  validation; a failed smoke test falls back to the image's Python; the
  argument lists are what `build_pip_commands` builds. Subprocess calls are
  mocked.
- CI (no GPU): build the image; run the container with `DW_DIFFUSERS_REF=image`;
  check that `/api/health` answers, and that `dw.validate` passes on
  `workflows/templates/text-to-image.json` inside the container.
- GPU smoke on lem: `docker compose up`, then `python -m dw.test` inside the
  container; then create an overlay at HEAD, update from the UI, re-create
  the container, and confirm the commit survived.

## Stages

1. `docker/entrypoint.py` + tests (no Docker needed).
2. Dockerfile, compose file, `.dockerignore`, CI build and health check.
3. A per-release constraints file, generated by `release.sh`, that the
   Dockerfile installs against. Publishing to a registry (GHCR) is
   deferred; it would be a later `release.sh` step.
4. The UI's `?token=` login link (see Security).
5. `docs/DOCKER.md`, linked from README, DEPENDENCIES and REMOTE.

## Decisions (Don, 2026-11-01)

1. Token: Jupyter's pattern (generated, persisted). Amended the same day:
   the login URL comes from `login-url` on request, never the log.
2. cu130 only; cu12 only through a build arg.
3. Build it yourself; no registry publishing yet. Superseded the same day:
   Don chose Docker Hub (`dkackman/diffusers-workflow`), published by a
   `docker` job in ci.yml on each release tag, first with 0.18.0.

---

## Tauri desktop app - notes for the follow-on spec

Not designed here. What the Docker work sets up for it:

- **Same layering, minus Docker.** The app bundles uv as a sidecar
  (`externalBin`). On first run it does what `docker/entrypoint.py` does, with uv
  in place of the image: it picks a Python (uv-managed), installs torch for
  the detected accelerator (install.sh's MPS / cu130 / cu12 / CPU branches,
  ported), installs the pinned dw wheel from the release's constraints file,
  and then the diffusers overlay. The constraints file from stage 3 is the
  shared piece.
- **A thin shell.** `dw.serve` already serves the UI. The app starts it as a
  child process on 127.0.0.1 with a random port and a per-launch token,
  waits for `/api/health`, and points the webview at it. That keeps one
  origin, with no CORS and no second copy of the SPA. It also stops other
  local processes from driving the GPU through the server.
- **Data outside the app.** The user chooses the folder, and the same
  `/data` layout applies (workspace, hf, cache, settings, env), so a
  reinstall or app update never touches models or outputs. An existing HF
  cache can be pointed at instead of re-downloaded.
- **Updates.** The app updater moves the dw wheel (the engine is pinned per
  app version, as in ComfyUI Desktop). diffusers stays on the overlay and
  the Models page, as here. Any engine change needs a server restart; the
  persistent worker only loads new code when it restarts.
- **Platforms.** macOS arm64 first, since that's the gap Docker can't fill,
  then Windows/NVIDIA (WebView2), then Linux, where WebKitGTK quirks need an
  early check against the Svelte UI. macOS also needs `fp4-fp8-for-torch-mps`,
  which pyproject can't express, so it goes in the bootstrap's
  per-platform branch.
- **Signing.** macOS notarization and Windows code signing are release-pipeline
  work and need certificates. Budget for it.
