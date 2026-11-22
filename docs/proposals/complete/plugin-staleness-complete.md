# Tell the user when the installed dw plugin is older than the server (#1535)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out session,
2026-11-22). Don filed the idea on 2026-11-21, after he verified the notes channel by
hand (#1532, parent #1523). His installed `dw` plugin was 0.9.0-alpha.1 and the server
was 0.22.0. The requests and review-hold skills never loaded, and nothing told him.
Plan v1 is the plan comment on #1535. This file records what was built, what was
deferred and what stays.

## Verdict: build (one stage)

Don approved plan v1 on 2026-11-21 with every question at its default:

- **Q1:** a `plugin` block in the `get_server_info` result, on the MCP side only with no
  REST change, plus one sentence in the tool's description. A skill that states no
  version counts as stale, so old installs are caught as soon as the server deploys.
- **Q2:** the update commands are checked against Claude Code's plugin docs before
  shipping. If `update` isn't a real subcommand, fall back to uninstall plus install.
  A wrong command is worse than none.
- **Q3:** the pre-release floor is `X.(Y-1).0`.

**The design corrects the issue's proposal.** The server never learns the plugin's
version, and it doesn't need to. The agent holds both halves: the release the loaded
skill says it was written for, and the server's `plugin.current_from`.

## What was built (stage A, #1560)

Stage commit `b984f3fc`, merged into `develop` as `8fc5136a`.

- **`plugin` block** (`dw_mcp/workspaces.py`: `plugin_current_from`, `plugin_status`).
  It is merged beside `workspace` and `caches` and holds exactly `current_from` and
  `update`.
  - A release `X.Y.Z` gives `X.Y.0`.
  - A pre-release gives `X.(Y-1).0`, or `X.Y.0` when Y is 0.
  - A version that won't parse, `nightly`, or a missing version leaves the block out.
  - The version comes from `/api/server`, so a remote server's number is used.
  - `/api/server` and the OpenAPI `ServerInfo` schema have no `plugin`.
- **Tool description** (`dw_mcp/tools_catalog.py`). It ends with this rule: if a loaded
  dw skill states no release, or one older than `plugin.current_from`, tell the user
  once and give the `plugin.update` commands. `INSTRUCTIONS` is unchanged.
- **Written-for line** in the 10 skills whose first step is `get_server_info`: "This
  skill is written for dw `<v>`; `get_server_info`'s `plugin` says whether that is
  current." The 10 are flux3-klein, kandinsky-6, long-form, ltx-2-5, minimax-h3,
  minimax-h3-image, minimax-music3, script-to-video, series-episodes and storyboard.
  `cinematography` makes no server call and has no line.
- **`scripts/release.sh`**: `set_version` rewrites those lines. Both the release commit
  and the `--next` commit include them.
- **Docs**: `README.md`, `docs/MCP.md` and `plugins/dw/README.md` give the update
  commands beside the install ones.
- **Tests**:
  - `tests/test_mcp_workspaces.py::TestPluginBlock` covers the floor cases, the
    unparseable cases and the merge.
  - `tests/test_plugin_skills.py` checks that exactly ten skills start at
    `get_server_info`, that each states a version, and that the version equals
    `plugin.json`. Its `test_release_sh_rewrites_the_written_for_lines` runs the
    script's heredoc on temp files.
  - The surface-budget test in `tests/test_mcp_server.py` still passes.

## Deviations from the plan

1. **`plugin.update` holds three commands, not two.** Claude Code has no `/plugin
   update` slash command (code.claude.com/docs/en/discover-plugins); only the shell
   has `claude plugin update`. Following Q2, the list is:
   - `/plugin marketplace update diffusers-workflow`
   - `/plugin uninstall dw@diffusers-workflow`
   - `/plugin install dw@diffusers-workflow`

   The marketplace refresh stays first, because install would otherwise fetch the
   stale listing. Then `/reload-plugins` or a new session applies the update; the docs
   say so.
2. **The rest of the `get_server_info` description got shorter.** It dropped the
   cache-entry field detail and some narrative, to stay under `SURFACE_BUDGET` without
   raising the budget.
3. **Small wording trims** in `long-form`, `minimax-h3` and `script-to-video` keep
   them within their byte caps. Every pinned phrase stays.
4. **`script-to-video` has no *Before anything* section**, so its line sits under
   `## 1. Read the script and the server`, the skill's first instruction.

## Deferred and non-goals

- **Skill edits between releases** land without a version bump, so this check can't
  catch them. Catching them would mean versioning content.
- **Nudging users with no plugin to install it** is out of scope. `docs/MCP.md` says
  every tool works without the plugin.
- **A stdio `dw-mcp` older than its server** (the remote-stdio setup) is a separate
  mismatch and isn't covered.
- **The end-to-end check** is Don's, as approved: the next time he starts a session
  with an old or unversioned skill, it should tell him once. No acceptance case
  covers it.

## Verification and cost

- **Bounces:** stage A had none; it was verified on the first pass on mini-ai, develop
  @ `4db9d8ed`. C-F1251, C-F1252 and C-F1253 pass.
- **C-F1250** passes except for its "exactly two commands" clause. The tester judged
  that the case over-specifies the plan, which never says to drop the marketplace
  refresh, rather than that the code is wrong. The amendment is filed as
  dkackman/harnest#274, and C-F1250 keeps its `pending: #1560` line until it is ruled
  on.
- The build's first hand-off was refused four times by the gate's parallel test run.
  Each time it named a different set of modules outside the diff, and each set failed
  with `[Errno 35] Resource temporarily unavailable` on the loaded Mac. All of them
  passed with `-n0`. A later session repeated the label swap, and the gate passed.
- **Cost:** the stage comments name no `usage:` figures, so none is recorded here.
  The plan's estimate was about $4.
