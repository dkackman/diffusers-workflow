# Cold install drill: README to a first image on a clean CUDA box, a clean Mac and the Docker image (#1525)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out,
2026-11-23). Plan v1 approved with every question at its default; v2 moved
the box-side checks the tester couldn't run (Q-S1, Q-S2); v3 records the
ruling on v2, approved with one amendment (Q5: the drill record carries each
arm's command transcript verbatim). Three stages, #1548 (docs and
`install.sh`), #1549 (CUDA and Docker arms on lem) and #1550 (Apple Silicon
arm), all closed. The drill records are comments on #1525: stage B's redo
(issuecomment-7081595581, which supersedes 7080204094) and stage C's. This doc
is the design record around them.

## The idea

An announcement (#1523) sends strangers to the README, and the nearest
neighbours install in one click. dw's path had only been walked on machines
that already had it. The ask: follow the README and `docs/DEPENDENCIES.md` as
written on a box with no dw, no venv and an empty HF cache, through
`dw.serve`, an MCP client and `templates/text-to-image`, on CUDA, Apple
Silicon and the Docker image.

## Decisions

- **Verdict: build.** A desk walk at feba7774 already found three blockers
  (no web UI after `install.sh`, no clone step, the image's location never
  named) and about a dozen missing prerequisites, so the fixes went first and
  the drills measured the fixed text (Q1).
- **Q2:** Don walks the Mac arm. **Q3:** no credentials in any container.
  **Q4:** the mps cost row comes from the Mac record.
- **Q5 [loosen]:** the tester has no shell on lem, so A's installer and `/`
  checks moved into B's drill record, A's README-matches-docs check became
  Don's diff review, and B's independent Docker re-walk was cut, with C as
  the independent walk of the README path. Amendment: verbatim transcripts.
- **Cut:** a one-click installer, Windows, an install CI job (a recurring
  runner is `spend`; it comes back if a drill regression would have been
  caught by one), changing the first-run model.

## What was built

- **#1548 (stage A):** README *Getting started* gains a per-platform
  prerequisites block (git, Python 3.12-3.14 and where to get it on a Mac,
  `python3.X-venv`, NVIDIA driver >=580, Node, the `claude` CLI, disk, why
  sudo prompts), the clone URL, `claude mcp add --scope user`, the `/plugin`
  block labelled "inside Claude Code", and a first-run line (what to ask, the
  SD 1.5 download, where the image lands). `install.sh` builds the UI
  (`npm ci && npm run build`) when npm is present. `DEPENDENCIES.md`'s manual
  path installs `.[server,mcp]`; `DOCKER.md` gains the clone step and the
  same-host `claude mcp add --transport http` line; `REMOTE.md` gains the
  clone URL. `tests/test_install_docs.py` pins the clone URL, the UI build
  and the Python range. Don's README review added two edits (the Claude Code
  setup URL; the step-4 go-ahead wording). Merged 10fa95ad.
- **#1549 (stage B):** both lem arms reached an image through headless
  Claude Code, with `/` rendering the UI. Fixes: the nvm line needs curl on
  a bare Ubuntu (bf80eb57); `DOCKER.md` says how to set `DW_HOST_PORT` in
  `docker/.env` when 8765 is taken (f9c6fe17, merged 4b55d89f). Filed #1580
  (the plan quoted ~44 GB for SD 1.5's 4.0 GB download; fixed).
- **#1550 (stage C):** the Mac arm, walked by Claude from Don's session as a
  fresh standard macOS user on an M4 Pro (Don withdrew his decline and
  delegated the walk). No stalls. The mps row
  `{"device": "mps", "name": "Apple M4 Pro", "vram_gb": 24, "minutes": 0.2}`
  went into `templates/text-to-image.json` (7bf5ce31, merged 95134513).
  Filed #1588 (validate counted a cached SD 1.5 as a 4 GB download because
  the template skips `safety_checker`; fixed in 30a1ceed) and #1589 (the
  README's go-ahead and first-run-time lines; fixed in ec379a32).

## What the drills measured

| Arm | dw commit | Commands (stranger's path) | Wall time to first image | Stalls |
|---|---|---|---|---|
| CUDA, bare `nvidia/cuda:13.0.2-base-ubuntu24.04` | 33a43133 | 12 | about 6 min 1 s | none (curl fixed after the first run) |
| Docker, `docker compose` on lem | 33a43133 | 8 (9 if 8765 is taken) | about 6 min 23 s | none |
| Apple Silicon, M4 Pro, fresh user | bf5d341b | 10 | 3 min 56 s | none |

Downloads dominate: torch through pip (about 151 s CUDA, 81 s for the whole
Mac install), the 266 s image build, and SD 1.5's 4.0 GB inside the first
job's load. diffusers was 750b3be2 (0.44.0.dev0) in every arm.

## Deferred, and what wasn't exercised

- **The CUDA 12 fallback and the no-driver path:** lem's driver is
  580.178.04, so `install.sh` took CUDA 13 only.
- **A desktop Ubuntu:** the container is stricter (no python3, git, curl or
  sudo) but can't show a distro Python or pip config.
- **The Mac's Homebrew path and `xcode-select --install`:** git and
  python.org Python were already system-wide; Node came from nvm.
- **The interactive `/plugin` lines and the workspace-trust dialog:** the
  Mac walk used the `claude plugin` CLI forms and `claude -p`.
- **`master`:** every arm checked out `develop`; a stranger cloning the
  default branch gets the pre-A README until a release merges it. That is
  Don's release call (#1523).
- **A second walker for the Docker arm:** cut under Q5.

## Cost and bounces

- **#1548:** no tester bounce; Don's README review asked for two edits
  before merge. C-F1179-C-F1183 passed first time.
- **#1549:** one bounce. C-F1258: neither transcript showed the `cd` after
  the clone; C-F1259: `/` was reported as an HTTP 200 shell, not a rendered
  UI, with no time. Both arms were re-walked in a logged tmux shell, with a
  Playwright check of `/` after the first image. The re-walk also cost lem's
  pre-drill `diffusers-workflow:local` image (`fbe234b63352`), deleted by
  mistake in the first run's clean-up; its layers are still in the build
  cache and it has not been rebuilt.
- **#1550:** no bounce; Don accepted the record and closed it.
- lem GPU time was under a minute per arm. Session dollar costs were not
  reported on the stages, so none are given.
