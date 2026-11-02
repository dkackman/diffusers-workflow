# syntax=docker/dockerfile:1
# diffusers-workflow server image (Linux, NVIDIA). See docs/DOCKER.md.
#
#   docker compose -f docker/compose.yaml build
#
# The build context is the repository root. Torch and every dependency are
# baked in; diffusers is baked too, as a fallback - the copy the server uses
# lives on the /data/env volume (docker/entrypoint.py), so it can track main.

ARG PYTHON_VERSION=3.13

# --- the wheel: scripts/build_dist.sh, with the SPA built by npm -----------
# Node's version follows ui/.nvmrc (tests/test_docker.py pins the two)
FROM node:24.20.0-trixie-slim AS node

FROM python:${PYTHON_VERSION}-slim-trixie AS wheel
COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -s /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm
WORKDIR /src
COPY ui/package.json ui/package-lock.json ui/
RUN cd ui && npm ci --no-audit --no-fund
COPY . .
RUN bash scripts/build_dist.sh

# --- the runtime ----------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim-trixie

# cu130 needs an NVIDIA driver >= 580 and a Turing or newer card. A CUDA 12
# image for an older driver: --build-arg TORCH_INDEX=https://pypi.org/simple
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu130
# The baked (fallback) diffusers: a branch, tag or commit
ARG DIFFUSERS_REF=main

# git for pip's git+ installs; a compiler for Inductor and Triton, which
# build kernels at run time; tini as PID 1, so signals reach dw.serve and
# its worker processes are reaped
RUN apt-get update \
    && apt-get install -y --no-install-recommends git build-essential tini \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1

# install.sh's order: torch first, so the resolver below sees it satisfied;
# then dw; then diffusers from git, forced without its dependencies (forcing
# those would reinstall torch), and a plain install to add any it needs
RUN pip install torch torchvision torchaudio --index-url "${TORCH_INDEX}"
COPY --from=wheel /src/dist/*.whl /tmp/wheels/
RUN pip install "$(ls /tmp/wheels/*.whl)[server,mcp]" && rm -rf /tmp/wheels
RUN pip install --force-reinstall --no-deps \
        "git+https://github.com/huggingface/diffusers@${DIFFUSERS_REF}" \
    && pip install diffusers

# The catalog the server offers beside the workspace's own (--examples-dir):
# the repo's workflows/, with the prompts/ and loras/ they reference
COPY workflows /opt/dw/examples/workflows
COPY prompts /opt/dw/examples/prompts
COPY loras /opt/dw/examples/loras

COPY docker/entrypoint.py /opt/dw/entrypoint.py
RUN python /opt/dw/entrypoint.py record-image

# Defaults for a container with one /data volume. docker/compose.yaml
# replaces the shared ones with host paths, mounted at the same path.
ENV DW_ENV_DIR=/data/env \
    HOME=/data/env/home \
    DW_WORKSPACE=/data/workspace \
    HF_HOME=/data/hf \
    TORCHINDUCTOR_CACHE_DIR=/data/cache/torchinductor \
    TRITON_CACHE_DIR=/data/cache/triton \
    DIFFUSERS_HELPER_ROOT=/data/settings

# Any uid may run it (compose passes the host user's); nothing below /data
# is created at build time, so the volume's owner owns everything in it, and
# without a volume /data is world-writable (sticky) for whichever uid it is
RUN mkdir -m 1777 /data
WORKDIR /data
EXPOSE 8765
HEALTHCHECK --interval=30s --timeout=15s --start-period=10m --retries=3 \
    CMD ["python", "/opt/dw/entrypoint.py", "healthcheck"]
ENTRYPOINT ["tini", "--", "python", "/opt/dw/entrypoint.py", "serve"]
CMD []
