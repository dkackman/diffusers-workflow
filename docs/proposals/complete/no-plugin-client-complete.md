# A no-plugin MCP client, drilled cold, and the `agent-rules` guide (#1524)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out,
2026-11-22). Plan v1 approved by Don 2026-11-21, every question at its
default. Two stages, #1546 and #1547, both verified 2026-11-22. For the
announcement release (#1523).

## The report

The composition skills ship as a Claude Code plugin (`plugins/dw`). The
announcement will be read by people on Codex, Cursor and other clients, and
nothing had measured what a client without the plugin gets from dw. The ask:
drill three requests (one image, one H3 dialogue clip, one piece longer than
a clip) with only the server instructions, the tools and the served guides;
report what worked, which skill-only rules broke, and whether a guide
`get_guide` can serve could carry them without growing the tool surface.

## Verdict

**Build smaller.** Drill a no-plugin Claude session rather than a non-Claude
model, run the 60 s ask only if its quote is under 30 min, and serve only the
rules the drill showed broken (plus the long-form rules the plan named) as a
docs guide. No MCP prompts, resources or `get_skill`.

## What the plan found

- No route served skill text over MCP. A no-plugin client got the tool
  descriptions, the server instructions and 16 docs guides.
- Guides cost nothing against `SURFACE_BUDGET` (it counts tool text, schemas
  and instructions), so a new `docs/*.md` guide with one `GUIDES` entry was
  free. The instructions sat a few characters under the 2048 limit, so a
  pointer had to replace existing words.
- Serving `SKILL.md` files directly would break: `read_guide` doesn't
  path-check, an install ships only `docs/*.md`, and the skills'
  `references/` links aren't served docs.
- The docs claimed "any MCP client" (`README.md`) with nothing measured.

## Decisions

- **Q1 (trust):** Claude Sonnet via `claude -p`, plugin and user skills
  withheld. The claim names no other client as tested.
- **Q2 (spend):** asks 1 and 2 to completion; ask 3 only under a 30 min
  quote. It quoted about 28 min and ran.
- **Q3 (default):** lem (CUDA). The `target:mini-ai` label was a slip.
- **Q4 (scope):** a served guide, no MCP prompts or resources.
- **Q5:** the guide is `agent-rules` (`docs/AGENT_RULES.md`).
- Don also approved the tester verifying stage A from the record over MCP,
  and stage B shrinking to the claim alone if A found nothing a guide could
  carry. It didn't shrink.

## What was built

### Stage A, #1546: the drill and its record

The record is *Cold drill without the plugin (#1524)* at the end of
`bench/records/drill.md`. Three headless Sonnet sessions (`claude -p
--strict-mcp-config`, dw MCP only, `--setting-sources project,local`, `Bash`
and `Skill` disallowed, each in an empty cwd and its own workspace) on lem,
`develop` @ 87ac105b. Every init event showed `skills: []`.

| Ask | Workflow | Jobs | Result |
|---|---|---|---|
| 1: one image | `templates/flux3/text-to-image` | `76a848eb3ee8` | 1024x1024, 28 s |
| 2: H3 clip, two lines | `templates/minimax/video-with-audio` | `e37acfc896c8` | 8.0 s, turbo kept |
| 3: 60 s, two scenes, score | `music`, `dialogue-short` (`draft`), a hand-authored mix | `cae69d51a4ec`, `67571fdbe0d2`, `f5c2b6ff77da` | 57.5 s, four shots cut over one bed |

**What worked on the tool surface alone:** the right shape from
`list_workflows(shape=...)` every time; validate before every run; looking at
the outputs; cuts rather than chains and one score bed for ask 3; acting on
the server's bleed-tail warning (which carried a series-episodes rule by
itself).

**Rules broken** (nine rows; the record quotes each):

| # | Rule | Skill section | Placement |
|---|---|---|---|
| 1 | short request through `upsample-then-generate` | `flux3-klein` *Which shape is the request* | guide |
| 2 | go-ahead before `run_workflow` | `minimax-h3` *Run and judge* | model knowledge |
| 3 | Context-IR prompts | `minimax-h3` *Prompts* | guide |
| 4 | dialogue drops the turbo LoRA (12 steps, shifts 12.0/3.0) | `minimax-h3` *Hard rules* | validation |
| 5 | staging for two people in frame | `minimax-h3` *Hard rules* | guide |
| 6 | 2.5-4 words/s against frames | `minimax-h3/references/dialogue.md` | validation |
| 7 | a voice description verbatim across shots | `minimax-h3` *Prompts* | validation |
| 8 | one bed via `assemble-and-score` | `long-form` *5. Per piece* | model knowledge |
| 9 | listen and check seams | `long-form` *Run and judge* | model knowledge |

**The client sentence** (the record's For Don, for #1523): a no-plugin
Claude Sonnet session was tested and could find, run and check catalog work
from the served tools and guides alone, but lost the family rules in rows 1
and 3-7, so composing to those rules is supported only in Claude Code with
the dw plugin, and other MCP clients and models are untested.

### Stage B, #1547: the `agent-rules` guide and the client claim

Shipped 2b137a61, merged as 11bc007f.

- **`docs/AGENT_RULES.md`**, served as `agent-rules`, with sections *FLUX 3
  Klein*, *MiniMax H3* and *Longer than one clip*. Each opens by naming the
  Claude Code skill that carries the same rules with their evidence. It
  points only at served guide sections.
- **The server instructions** say "No dw plugin? `get_guide("agent-rules")`
  first.", in space freed by trimming words elsewhere (2046 of 2048).
- **`docs/MCP.md` *Clients*** holds the claim, matching stage A's sentence.
  `README.md`'s "any MCP client" and "other MCP hosts" link to it; no words
  changed.
- **Tests:** `tests/test_server_guides.py::TestAgentRules` (listing, loose
  section match, unknown section, pointers, the instructions, the claim) and
  `tests/test_plugin_skills.py::TestAgentRulesGuide` (each pinned phrase and
  number reads the same in the guide and the skill).

**Deviations from the plan, accepted at verify:**
- Rows 4, 6 and 7, which stage A placed under validation, are in the guide
  too, since no validation stage exists to carry them.
- The guide's dialogue rule says to pass the turbo-drop settings even on a
  template whose defaults keep the LoRA, such as `dialogue-short`. The
  template was not changed.
- Long-form rules the drill didn't break (35 s chains, the 90 s score) are
  included, as the plan's section named them.
- #1519's intent index isn't linked: it lives only in the skills'
  `references/intents.md`, which isn't served.

## Bounces per stage

- **#1546: two.**
  - C-F1166: the record wasn't on the ticket, only linked. A comment with the
    full section was posted.
  - C-F1168 and C-F1170: rows 4, 6 and 7 carried two placement verdicts, and
    row 6 named only ask 3. Each row got one verdict, row 6 names asks 2 and
    3, and the For Don turbo item names the template conflict plainly.
- **#1547: none.** The architecture review passed.

Cost per stage is not recorded: the stage comments name no `usage:` figures.
The plan estimated $8-13 in agent tokens. The three drill sessions cost
$0.149, $0.223 and $0.728.

## Deferred and left open

- **`dialogue-short`'s defaults keep the turbo LoRA**, against the
  minimax-h3 dialogue rule. Either the template or the rule is wrong. Don's
  call, from stage A's For Don.
- **Validation checks for rows 4, 6 and 7** (turbo with `<d>`, words per
  second against `num_frames`, one speaker's voice differing across a
  `for_each`). Each is computable from the arguments and would reach every
  client; warn or refuse is Don's. Not filed.
- **Catalog text for rows 1 and 3** (the `text-to-image` entry naming
  `upsample-then-generate`; H3 entries pointing at `enhance-prompt`).
- **A non-Claude client as tested** (Codex, Cursor). **Trigger:** Don wanting
  the announcement to name one.
- **`dialogue-short`'s estimate** ran 37% short (35.1 vs 25.5 min at 345
  frames, low confidence).
- **`RELEASING.md` rules** (the 35 s drift, the H3 voice rule) are unserved;
  the 35 s rule is now in `agent-rules`.
- **Two copies drift.** The pinning test holds numbers and pinned phrases,
  not prose.
