# Running dw in Docker

A container image for a Linux box with an NVIDIA GPU. It holds Python, torch
(CUDA 13), dw and its web UI. You don't need a venv or `install.sh`. Build
it on the box from this checkout, or run the one published on Docker Hub as
`dkackman/diffusers-workflow` (each release from 0.18.0 on, plus `latest`).

Docker is for Linux and NVIDIA only. On a Mac, Docker runs a Linux VM with
no access to the GPU (MPS), so use `install.sh` there.

## What the host needs

- An NVIDIA driver **580 or newer**, and a Turing or newer card (RTX 20-series
  and up). CUDA 13 dropped Maxwell, Pascal and Volta.
- Docker with the compose plugin, and the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  (`nvidia-ctk runtime configure --runtime=docker`, then restart docker).
- Your user in the `docker` group, or every command below under `sudo`.

Check the GPU reaches a container:
`docker run --rm --gpus all ubuntu nvidia-smi`.

## Start it

```bash
bash docker/init-env.sh                          # writes docker/.env
docker compose -f docker/compose.yaml up -d --build
docker compose -f docker/compose.yaml logs -f    # wait for the server to start
docker compose -f docker/compose.yaml exec dw python /opt/dw/entrypoint.py login-url
```

To use the published image instead of building, uncomment `DW_IMAGE` in
`docker/.env` and run `docker compose -f docker/compose.yaml pull` before
`up -d` (without `--build`). The first build takes a while (torch alone is
several GB), and the first start installs diffusers from GitHub main into
its volume (see below), a few more minutes before the server answers.
`login-url` prints `http://localhost:<DW_HOST_PORT>/?token=...`; opening it
saves the token in the UI.

## Where the data lives: shared with a native install

`docker/init-env.sh` fills `docker/.env` with the directories a native
dw.serve on the same host already uses:

| variable | default | holds |
|---|---|---|
| `DW_WORKSPACE` | `~/diffusers-workspace` | workflows, prompts, assets, outputs |
| `HF_HOME` | `~/.cache/huggingface` | model weights, and the quantized cache (`dw-quantized/`) |
| `TORCHINDUCTOR_CACHE_DIR` | `~/.cache/dw/torchinductor` | `torch.compile` kernels |
| `TRITON_CACHE_DIR` | `~/.triton/cache` | Triton kernels |
| `DIFFUSERS_HELPER_ROOT` | `~/.diffusers_helper` | `settings.json`, the log, the container's API token |
| `DW_ENV_DIR` | `~/.local/share/dw-docker/env` | the diffusers overlay (container only) |

Export any of them before running the script to change it. Each one is
mounted **at the same absolute path** inside the container, and the
container runs as your uid. Run manifests, settings and cache entries
record absolute paths, so a job the container ran opens in a native server
and the other way round. Models download once, and kernels compiled by one
are reused by the other. A symlinked directory, such as an `HF_HOME` that
links to another disk, works: Docker mounts the target at the link's path.

The checkout's own `.env`, which a native dw.serve loads (for example
`DW_QUANTIZED_CACHE=1`), is passed to the container too, so both run with
the same switches. A path in it, such as `DW_QUANTIZED_CACHE_DIR`, has to be
one of the mounted directories above.

### Switching between Docker and systemd

Both use the same port, GPUs and data, so only one should run at a time:

```bash
bash docker/switch.sh docker    # stop the dw-serve user unit, start the container
bash docker/switch.sh native    # stop the container, start the unit
bash docker/switch.sh status
```

The script assumes the user unit from `scripts/dw-serve.service`
(`DW_SYSTEMD_UNIT` names another unit). For the system unit in
`contrib/systemd/`, set `DW_SYSTEMCTL="sudo systemctl"`.
To keep the token your MCP clients already use, set `DW_API_TOKEN` in
`docker/.env` to the one the unit passes.

The examples catalog is one difference: the container offers the copy of
the checkout's `workflows/` that was built into the image, while a native
server offers the live checkout.

## diffusers tracks main, on a volume

Everything in the image is fixed when it's built, except diffusers. New
model pipelines land on diffusers' GitHub main before a release, so the
diffusers the server uses lives in `DW_ENV_DIR`. That directory holds a
small venv layered over the image's Python (`--system-site-packages`), with
only diffusers and any dependency a new commit needs that the image lacks.

- **The first start** installs main HEAD (or `DW_DIFFUSERS_REF`; see below).
- **A restart changes nothing.** diffusers moves only when you ask it to.
- **Update** to HEAD from the web UI's Models page, as on a native install,
  or to HEAD, a commit or the release floor with
  `docker compose -f docker/compose.yaml exec dw python /opt/dw/entrypoint.py update-diffusers [--commit SHA | --revert]`,
  then restart the container (a running job keeps the old import).
- **A rebuilt image** (a new dw) rebuilds the overlay and reinstalls the
  diffusers commit you had, not HEAD.
- **If the overlay doesn't import** (a broken HEAD, or no network on the
  first start), the server starts on the copy of diffusers built into the
  image and says so in the log. Pin a working commit with `update-diffusers
  --commit SHA` (or `--revert`), then restart. An update from the Models
  page in this state goes into the container rather than the volume, so it
  is lost when the container is re-created.

`DW_DIFFUSERS_REF` in `docker/.env`: `head` (the default), a commit hash
(the overlay is held to it), or `image` (no overlay; use the copy built
into the image).

## The API token

Inside the container the server listens on every interface, and it can't
tell how the port is published, so it always has a token: `DW_API_TOKEN`
if you set one, otherwise one generated on first start and saved to
`DIFFUSERS_HELPER_ROOT/api-token`. The token is never written to the log,
so `docker logs` output is safe to paste into an issue; `login-url` prints
it on request. `DW_INSECURE_NO_TOKEN=1` turns this off; do that only with
the port published on localhost (`DW_BIND=127.0.0.1`, the default).

To reach the server from your LAN, set `DW_BIND=0.0.0.0`, then run `up -d`
again to re-create the container; clients connect as
[REMOTE.md](REMOTE.md) describes. A box switching from a systemd unit bound
to `0.0.0.0` needs this, or the container answers only on the box itself.
`DW_HOST_PORT` (default 8765) is the host port it is published on.

## Options

`DW_SERVE_ARGS` holds dw.serve's own arguments (default `--mcp`), for example
`--mcp --devices cuda:0,cuda:1` for one worker per card. The entrypoint adds
`--host 0.0.0.0`, `--port 8765` and the examples catalog unless you give
them. `HF_TOKEN` is passed through for gated models.
`DW_MCP_MAX_WAIT_SECONDS` (written as 1800 by `init-env.sh`, the systemd
unit's value) caps how long an MCP `wait_for_job` may block (55 s when unset).

Build arguments (`docker compose build --build-arg ...`): `TORCH_INDEX`
(default the cu130 index; `https://pypi.org/simple` gives the CUDA 12 wheel
for an older driver) and `DIFFUSERS_REF` (the diffusers built into the
image; default `main`).

flash_attn is not in the image, because it needs the CUDA toolkit to
build. The Hub kernels and comfy-kitchen's prebuilt kernels work as they do
natively.
