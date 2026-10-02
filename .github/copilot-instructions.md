# AI Coding Instructions for diffusers-workflow

A declarative workflow engine for HuggingFace Diffusers: image, video and audio pipelines defined in JSON.

- Read `CLAUDE.md` (root) and `docs/ARCHITECTURE.md` first; the conventions for workflow JSON are in `docs/WORKFLOW_GUIDE.md`.
- Security: route every path, URL, variable name and subprocess argument through the validators in `dw/security.py` (`dw/trust.py` is the trust gate), and never use `eval()`, `exec()` or `shell=True`. See `docs/SECURITY.md`.
- Tests: see `docs/TESTING.md`.
