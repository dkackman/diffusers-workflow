---
name: model-family-onboarding
description: Use when adding a new model family to the dw catalog or re-checking an existing one (H3, LTX-2.5, or whatever comes next) - audit the repo's knowledge against the vendor's published sources, repair the catalog from the audit, then write the family's composition skill for the dw plugin. Covers the whole review, refine, skill lifecycle.
---

# Model-family onboarding: audit, repair, skill

The catalog's knowledge about a model family goes stale two ways: the vendor
moves (new LoRAs, new pipelines, a changed default) and the repo drifts from
what it transcribed (a renamed template, a paraphrase that lost a rule). Both
were found on 2026-09-07 for H3 and LTX-2.5, and the fix was the same each
time: go back to the vendor's own text. This skill is that process, so the
next family gets it in an afternoon rather than a rediscovery.

**Announce at start:** "Using model-family-onboarding for <family>."

## Principles that do not move

- **The vendor's text is canonical.** Model card, the vendor's GitHub repo and
  prompt guides, the technical report, and the diffusers pipeline source (which
  encodes what the pipeline actually accepts). Blog posts, videos and forum
  threads restate the model card; weight a secondary claim only when several
  independent sources repeat it *and* nothing primary contradicts it.
- **Model knowledge is data, never engine code.** Template JSON, stored
  prompts, a builtin's system prompt, README prose, a skill. No per-model
  Python (proposal `docs/proposals/agent-catalog-legibility.md`, "Principle").
- **Do not transcribe a prompt format the vendor publishes.** Point at it. If
  the vendor ships an agent skill (MiniMax does) or a system-prompt constant
  inside diffusers (Lightricks does), the dw skill says "use that" and a test
  ties any quoted text to the library. A copy is the thing that drifts.
- **A number a skill states is pinned by a test to where it comes from**
  (the diffusers module that enforces it), so a library upgrade fails a test
  rather than a cold session.
- **Nothing bundled needs `--trust-workflows`.** No `trust_remote_code`, no
  `custom_pipeline`; `tests/test_catalog_structure.py` refuses them.
- **Every step lands in the ledger row** of the proposal, dated, with the
  audit file it argues from.

## The lifecycle

Work through these in order and create a todo per item.

### 1. Audit

Dispatch one research agent per model on the most capable model available
(cross-checking a spec against sources is judgment work; a cheaper model
takes more turns to do it worse). The prompt is
[references/audit-prompt.md](references/audit-prompt.md): fill in the family,
the repo files that carry its claims, and the primary URLs you already know.
The agent writes a report to the scratchpad and returns a five-line summary.

What a good report has: a sources table with type and date; a claim-by-claim
verdict (CONFIRMED / CONTRADICTED / UNSOURCED, each with a citation) over
every rule the repo states; a dated "missing knowledge" list; and a short
assessment of what a skill could trust as-is. Read the whole report, not the
summary. The two on file are the models to hold a new one against:
`docs/proposals/audits/2026-09-07-minimax-h3-audit.md` and
`docs/proposals/audits/2026-09-07-ltx-2.5-audit.md`.

Save the report as `docs/proposals/audits/<date>-<family>-audit.md` and
commit it with the design work; it is the evidence the ledger points at.

For a family that is new to the catalog there is nothing to verify yet: the
audit's job is then the sources table, the vendor's recommended settings and
prompt format, and the reading-order list of what the templates should teach.

### 2. Assess and decide

From the report, sort findings into three piles:

- **Contradicted** claims in templates, prompts, builtins or docs: these are
  the repair package. A skill that describes a broken template teaches a
  broken thing, so repair lands before the skill is written.
- **Unsourced** claims that are the repo's own engineering (a chain feature,
  an offload layout): keep them, mark them as dw craft rather than vendor
  guidance in whatever prose states them.
- **Missing knowledge**: newer LoRAs, pipelines, parameter ranges. Dated
  follow-ups in the ledger, not part of the repair unless one is blocking.

Then brainstorm (superpowers:brainstorming) with the user on what the repair
covers, and write the two specs the way the first pass did:
`docs/superpowers/specs/<date>-<family>-catalog-repair-design.md` and the
skill's design if the plugin's outline does not already cover the family.
The H3/LTX-2.5 pair are the worked examples:
`docs/superpowers/specs/2026-09-07-ltx-h3-catalog-repair-design.md` and
`docs/superpowers/specs/2026-09-07-dw-plugin-skills-design.md`.

### 3. Repair the catalog

Plan it with superpowers:writing-plans and run it with
superpowers:subagent-driven-development. The shape that worked, from
`docs/superpowers/plans/2026-09-07-ltx-h3-catalog-repair.md`:

- One task per repaired thing, each with a test that would have caught the
  drift: a template's literal equals the library constant it copies; a stored
  prompt meets the trained format's length and carries none of the tag-style
  phrases; a builtin's system prompt contains the corrected rule and not the
  removed line; every README link resolves.
- Descriptions that cite a rule name it without single quotes (the drift
  check treats a single-quoted name as a variable or step).
- A README's first paragraph names the vendor sources and links the audit.
- The last task runs the repaired templates on the GPU box and writes `cost`
  (warm minutes, one decimal rounded up, on the card it was measured on).
- Every task appends one sentence to the Part 4 ledger row.

### 4. Write the composition skill

One `SKILL.md` per family under `plugins/dw/skills/<family>/`, following the
outline in the plugin spec (section "Skill outline"): frontmatter written for
triggering; call `get_server_info` and a shape-filtered `list_workflows`
before trusting any name; the shape decision as choices; the hard numeric
rules; the vendor pointer for prompts and nothing else; validate, quote cost,
run, look, and the family's failure modes; sources with dates. Near the size
of the README it derives from, under the 12 KB cap.

Add the family's numbers to `tests/test_plugin_skills.py`, each checked
against the diffusers module that enforces it, and any quoted vendor text to
the equality test against the library constant.

Copying `plugins/dw/skills/ltx-2.5/SKILL.md` and replacing every section is
the intended way to start.

### 5. Cold drill

The acceptance test is what a fresh agent does unprompted; no unit test
measures it. The drill, from the memory note `cold-session-mcp-tests` and
the proposal's "Cold-session probe" section:

1. The GPU box on the merged branch, `dw.serve --mcp` restarted (a session
   started before the restart tests the old instructions).
2. A fresh Claude Code session in an empty directory with the plugin
   installed, given one open-ended request in the family's shape ("a short
   multi-shot video with cuts between the shots").
3. Read its transcript and the server's access log. Pass: the skill fired, the
   right template was chosen, prompts came from the vendor's text, cost was
   quoted before the run, the output was looked at.
4. Run the same prompt without the plugin as the control.

Record both transcripts' paths and the result in the ledger row.

### 6. Close

Ledger row updated with the audit date, what was repaired, the skill's path,
the drill result, and the dated follow-ups. PR against `master`. The GPU box
back on `master` and restarted.

## What not to do

- Do not "research" by watching videos or reading prompt-guide blogs first.
  They restate the model card, and their repetition measures copying.
- Do not add a guide section to `dw/server/guides.py` for a model family.
  That index is dw-generic and paid for by every session; the plugin is the
  channel for model knowledge (decision recorded 2026-09-07).
- Do not write a template for a flow the vendor does not describe without
  marking it in the description as dw's own recipe.
- Do not skip the repair to get to the skill. The skill is only worth
  writing if what it points at is true.
