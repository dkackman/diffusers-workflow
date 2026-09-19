# Proposal: a `script-to-video` skill that composes what already exists

Status: proposed, 2026-09-19. Raised by Don asking whether "give the agent a
script, have it turn that into a video end to end" is already possible with
`dw`. Written from that conversation, not from a filed issue. No code
changes yet.

## The problem

The request, as posed, was broader than this repo should take on: hand an
agent a script and a high-level goal ("turn this into a video") and have it
autonomously pick models, build the workflow, generate, diagnose failures,
retune settings, and iterate - unsupervised, end to end, including deciding
*how to make a slow model faster*.

Most of that loop already exists in this repo, spread across the MCP surface
and the per-family composition skills. The piece that does not exist is the
first one: nothing turns a script into the `shots` list the templates
expect. Everything downstream of that list is already built. This proposal
scopes a new skill for exactly that gap, and explicitly declines the
open-ended "profile and auto-optimize an arbitrary model" ambition - not
because it is uninteresting, but because it is a different, much harder
problem that does not compose with the rest of this design.

## What already exists (no change proposed here)

| Stage | Mechanism | Where |
|---|---|---|
| Pick a template for a shot's shape | `list_workflows(shape=...)`, the per-family skills | `dw:minimax-h3`, `dw:ltx-2-5`, `dw:minimax-music3` |
| Write a conformant prompt for the chosen family | family prompt-writing skills | `h3-prompt-writing`, `music-caption-rewriter` |
| Cast/voice continuity across shots and episodes | the fixed-asset-reference procedure | `dw:series-episodes` (step 0 + the five-beat cut) |
| Cost estimate before spending | `plan.estimate` (`basis`, `runs`) on `validate_workflow` | `dw/plan.py`, `POST /api/validate` |
| Spend gate | `acknowledged_cost` on `run_workflow`/`rerun_job` | CLAUDE.md, *Acknowledged-cost* |
| Run, inspect, diagnose | `run_workflow` → `wait_for_job` → `get_job_events`/`get_gallery_metadata` | existing MCP tools |
| Structural warnings an agent can act on | `audio_no_headroom`, `audio_clipped`, elision diagnostics | `dw/result.py`, `dw/elision.py` |
| Retry with a fresh seed | `rerun_job(new_seed=True)` | step cache, CLAUDE.md |

None of this needs new engine code. The gap is upstream of all of it: a
script is prose, and every mechanism above starts from a `shots` list that
already has names, per-shot prompts in the target family's structured form,
and reference wiring.

## Design: the new piece is a skill, not an engine feature

### `script-to-video` (new plugin skill, `plugins/dw/skills/`)

Composes, in order:

1. **Read the script and the server.** `get_server_info` for device/workspace
   the way every other skill starts. The script itself is whatever the user
   pastes or attaches - this skill does not parse screenplay formats, it
   reads prose and dialogue the way a person would.

2. **Decompose into a shot list, not shot count.** The user's framing ("split
   into N lines/scenes") is close but the unit that matters is *shots*, not
   lines: a scene may be one shot or several depending on cuts, and H3's
   sweet spot is 4-6 second shots (CLAUDE.md's `17*n+5` frame grid). The
   skill's job is turning script beats into named shot entries - `name`,
   which character(s) appear, what happens, roughly how long - deferring the
   family-specific prompt structure to step 3. This is genuinely new
   reasoning work with no existing mechanism to lean on; it is also the one
   step where "an LLM is good at this" is actually true, unlike model
   auto-tuning.

3. **Cast the recurring characters once, per `dw:series-episodes` step 0.**
   Every character who appears in more than one shot gets a portrait (and
   voice clip, if they speak) drawn once and `keep_output`'d as
   `asset:cast/<name>`, before any shot generates. A script with one
   continuous scene and no recurring cast can skip this and let the family
   template draw its own portraits - the skill states the rule (more than one
   shot needs a fixed asset) rather than always drawing cast up front.

4. **For each shot, pick the shape and write the prompt.** Silent action or
   establishing shots go through a `dw:ltx-2-5` template; anything with
   dialogue, a consistent voice, or a scored montage goes through
   `dw:minimax-h3`'s `dialogue-short`/`music-video`, prompted via
   `h3-prompt-writing`'s structure (`subject_definitions`,
   `retention_analysis`, `detailed_description`, `overall_soundscape`,
   `non_diegetic_music`) rather than the raw script line. This is composition
   of existing skills, stated as an explicit decision tree instead of left
   implicit.

5. **Validate the assembled `shots` argument before queuing.**
   `validate_workflow` against the chosen template with the full `shots`
   list as `arguments`, read `plan.estimate` and `plan.downloads_required`,
   state the cost, and only call `run_workflow` with `acknowledged_cost` once
   the user (or an explicit "proceed" instruction in an unattended run) signs
   off. This is the existing gate, just made a mandatory step in the skill
   rather than something an agent might skip.

6. **Run, read warnings, retry per shot.** After the job completes, read
   `get_job_events` / `get_gallery_metadata` for the warnings the engine
   already emits, and `rerun_job(new_seed=True)` a shot that reads wrong
   (wrong face, no affect in the voice, clipped audio) using the diagnostics
   already surfaced - not by inventing new heuristics.

7. **Cut, score, deliver.** Once every shot exists, hand off to
   `assemble-and-score` (or the `series-episodes` five-beat procedure if the
   script is one episode of a series) for the recut/bed/match_levels/
   normalize/pair pass. No new mechanism - this step already exists and
   already composes.

Nothing above is a new engine capability. It is a skill that states, in one
place, the decision tree an agent currently has to reconstruct from five
separate skills and the MCP tool list every time - the same role
`series-episodes` already plays for its narrower slice (turning generated
shots into a scored episode). This skill sits one level up: turning a script
into the shots in the first place.

## Testing

Skills are audited the way the other `dw` plugin skills are
(`tests/test_plugin_skills.py`-style pinning of every stated number to a real
catalog value), not unit-tested as code. Verification here is a worked
example: a short multi-character script run through the skill end to end on
a real server, checked for - cast consistency across shots (the
`series-episodes` failure mode), correct shape selection per shot, a
`plan.estimate` quoted before the run, and a final cut that plays back
without level jumps. That is a manual acceptance run, not an automated
suite, the same way `series-episodes` itself was validated against two
hand-built episodes (issue #217).

## Not in scope

- **Automatic model speed optimization.** "If a model is slow, research how
  to speed it up, try options, keep what works" is a real engineering task
  (swap `config_type`, add `residency: on_demand`, try a different
  `offload`) but it is not safe to leave to an unsupervised agent loop: the
  search space is large, a wrong choice can silently change output quality
  rather than just fail, and `observed_cost` only tells you a run was slow,
  never *why*. This stays a human-in-the-loop task, informed by
  `observed_cost` and the `ACCELERATION.md`/`QUANTIZATION.md` docs, not
  something this skill attempts.
- **Workflow authoring from scratch per script.** The skill picks among
  existing catalog templates; it does not write new pipeline JSON. A script
  whose shape no template covers is a "propose a new template" conversation,
  not something the skill improvises at run time.
- **ComfyUI / node-graph integration.** Out of scope for this repo, which is
  declarative-JSON-workflow shaped, not node-graph shaped.
- **Full unattended autonomy.** Step 5's cost acknowledgment is a real gate,
  not a formality; this proposal assumes a human (or an explicit
  pre-authorization) confirms spend before a multi-shot script runs.

## Later

- A `cost.per_entry` on the family templates (CLAUDE.md notes neither
  MiniMax cut template has one yet) would let step 5 quote a tighter number
  for a script's actual shot count rather than the template's default list.
- If `select`/`judge` (see `docs/proposals/score-and-select.md`) ships, step
  6's per-shot retry could become deterministic best-of-N per shot rather
  than agent-judged reruns.

## Open question for whoever picks this up

Whether this belongs as a `plugins/dw/skills/` entry (composes over MCP,
travels with the plugin, gets the same numeric-pinning audit as the other
family skills) or as a `docs/` guide referenced by `list_guides` (no pinning
audit, easier to keep prose-loose). Leaning towards a plugin skill, since its
whole value is the decision tree an agent follows at request time, which is
exactly what the existing family skills already do.
