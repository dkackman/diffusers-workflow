# dw plugin: model-family composition skills — design

Package B of the Part 4 work in
[agent-catalog-legibility-complete.md](../../proposals/agent-catalog-legibility-complete.md).
It answers the open question of how model-specific knowledge reaches an
agent: as a Claude Code plugin of small composition skills that defer the
prompt format to the vendors' own published text. It depends on Package A
([2026-09-07-ltx-h3-catalog-repair-design.md](2026-09-07-ltx-h3-catalog-repair-design.md))
having landed, because the skills describe the repaired catalog.

## Decisions taken

- **Claude Code first, other MCP clients best-effort.** The knowledge lives in
  the plugin. The engine's guide index is unchanged; the two template READMEs
  remain what a non-Claude client can reach.
- **A plugin in this repo, installed once.** Not written into workspaces, not
  a separate repo.
- **Composition only.** A skill teaches what is dw-specific: which template
  fits which shape, the hard numeric rules, cost, and the run-and-judge loop.
  It does not carry a prompt format.
- **Prompt format comes from the vendor.** MiniMax publishes an official
  `h3-prompt-writing` skill and two guides on the model card; Lightricks ships
  the LTX-2.5 training-caption spec as system-prompt constants inside
  diffusers. Both audits found the repo's own copies were the ones that had
  drifted. The skills point at those sources rather than transcribe them.
- **Two families in the first cut**, so the layout is designed against two
  rather than refactored after one.

## Layout

```
.claude-plugin/marketplace.json      # one entry: dw -> ./plugins/dw
plugins/dw/
  .claude-plugin/plugin.json         # name "dw", description, version
  README.md                          # install, what each skill covers
  skills/
    minimax-h3/SKILL.md
    ltx-2.5/SKILL.md
```

- `marketplace.json` names the marketplace `diffusers-workflow` and lists the
  one plugin with a relative source. Install is two commands, shown in the
  repo README's "Drive it from Claude Code" section after the MCP
  registration:

  ```
  /plugin marketplace add dkackman/diffusers-workflow
  /plugin install dw@diffusers-workflow
  ```

- **Version**: `plugin.json`'s version is the dw version. `scripts/release.sh`
  writes it beside `pyproject.toml` in the same bump commit, and a test
  asserts the two agree, so an installed plugin can be matched to the engine
  it was written against.
- **No MCP server declared** by the plugin. A skill's first instruction is to
  call `get_server_info`, so it works against whatever dw server the session
  already has. Remote servers need a URL and token per user, which a plugin
  cannot carry.
- **Nothing under `dw/`, `dw_mcp/` or the wheel changes.** The plugin is
  repository content only. The repo's existing `.claude/` directory holds
  only `settings.json` and is untouched.

## Skill outline

Both skills follow one outline, which is the mould for the next family.

1. **Frontmatter.** `name` and a `description` written for triggering: the
   model's name, the phrases a user says ("a video with sound", "a dialogue
   scene", "a music video", "a clip with its own soundtrack", "extend this
   clip"), and "when a dw MCP server is connected". The description is the
   whole standing cost of the skill, so it is one or two sentences.
2. **Before anything.** Call `get_server_info` for the accelerator and the
   workspace; call `list_workflows` filtered by shape to find the family's
   templates by their current names rather than trusting the ones the skill
   quotes; read `get_workflow` on the one chosen.
3. **Shape decision.** The README's reading-order tables rewritten as choices
   an agent makes from the request. H3: one take up to 14 seconds; a longer
   take by chain and what drift costs per seam; a piece with cuts as fresh
   shots from shared portraits; appearance from an image reference; voice
   from an audio reference; several boards in one generation under one
   score. LTX-2.5: a single clip; first-frame or first-and-last conditioning;
   the three-move two-stage flow for quality; the IC-LoRA upscale and what
   it is not for; extend and chain, and that a single 481-frame pass reaches
   20 seconds before either is needed.
4. **Hard rules.** H3: `17n + 5` frames between 124 and 345 at 24 fps, the
   768 short edge and 32-pixel grid, aspect 1:4 to 4:1, the 544p turbo LoRA
   coupled to 960x544 and nine steps, Music3's duration as a ceiling and its
   six-minute cap. LTX-2.5: `8k + 1` frames, 32-divisible dimensions, 24 fps,
   the distilled sigmas are the schedule and not a knob, guidance off, the
   stage-two sigmas and renoise scale, image conditioning re-compressed at
   CRF 18 and needing a PIL image.
5. **Prompts.** The vendor pointer and nothing else. H3: if the
   `h3-prompt-writing` skill is installed, use it; else fetch the two guides
   from the model card by their raw URLs; else run the family's
   enhance-prompt template, which is the same format from the same guides.
   LTX-2.5: the trained-caption spec, quoted from
   `LTX2_5_T2V_DEFAULT_SYSTEM_PROMPT` with the I2V variant's "describe only
   what changes" rule, and the enhance-prompt template for a one-line idea.
6. **Run and judge.** Validate; quote the listing's `cost` and get a
   go-ahead; run; wait; look at the output with `get_output_image` or the
   gallery URL; the family's failure modes to check for (H3: identity drift
   across shots, a reference portrait imposing its framing, a skipped
   storyboard; LTX-2.5: a scene cut caused by a prompt that contradicts the
   image, softness when stage two was skipped).
7. **Sources.** Each vendor source by URL with the date it was read.

Size: each skill lands near the size of the README it derives from, and a
test caps `SKILL.md` at 12 KB.

The two READMEs keep their tables and gain one line pointing at the skill.

## Drift tests

`tests/test_plugin_skills.py`, run with the suite:

- Every backticked name in a skill that looks like a catalog path
  (`templates/...` or `models/...`) resolves to a file under `workflows/`.
- Every numeric rule the skill states is checked against the source it
  comes from: H3's frame arithmetic and canvas limits against the diffusers
  MiniMax modular-pipeline constants; LTX-2.5's frame and dimension rules,
  the distilled sigma count and the stage-two renoise value against the
  LTX-2 pipeline module. A diffusers upgrade that moves one fails here.
- The LTX-2.5 skill's quoted prompt spec equals
  `LTX2_5_T2V_DEFAULT_SYSTEM_PROMPT`, whitespace aside. The quote is the one
  place the plugin carries vendor text, and it is tied to the library that
  ships it.
- `plugin.json`'s version equals `pyproject.toml`'s.
- Each `SKILL.md` is under the size cap and has a `description` that names
  its model.

The tests import diffusers, so they carry the same skip the existing
pipeline-signature tests use when it is absent.

## Acceptance: the cold drill

Run twice against lem after Package A is merged and the server restarted,
from the scratch directory the earlier drills used:

1. The plugin installed. Prompt: "a short multi-shot video with cuts between
   the shots", the one the earlier probe answered with three unrelated
   LTX-2 shots.
2. The plugin not installed, same prompt, as the control.

Pass, for the first run: the H3 skill fires; the agent chooses the cuts
template with shared portraits; it writes shots in Context-IR from the
vendor guides or the official skill rather than an invented layout; it
quotes cost and waits for a go-ahead; the finished piece holds the cast
across shots. The transcript path and the server-log lines are recorded in
the ledger row beside the earlier probe.

## Ledger

The Part 4 row records the decision, the plugin path, the drill result, and
the mould for the next family: copy a skill, follow the outline, add the
family's rules to the drift test, cite the vendor.
