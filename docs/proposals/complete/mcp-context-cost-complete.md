# Proposal: what the `dw` MCP surface costs an agent's context, and what to trim

Status: **measured, not started** - the numbers below are real measurements
taken 2026-09-13 against `dw.serve 0.4.0-beta.3` on `lem` and against the
tool surface `build_server` produces. Nothing here is implemented. Written
for issue #101 by the implementer agent, model `opus` via provider
`anthropic`.

## Method

Token figures are **chars / 4**, the usual English-prose approximation.
JSON packs slightly denser than prose, so treat every number as ±15% and
as a *floor* rather than a ceiling. Comparisons between the numbers are
sound even where the absolute scale is approximate, because every figure
was taken the same way.

- Tool schemas: `build_server(...).list_tools()`, each tool serialized as
  `{name, description, inputSchema}` compactly - what a client renders into
  the model's context on connect.
- Server instructions: `MCPServer(instructions=...)`, also paid on connect.
- Results: the `dw_mcp` handlers called directly against `lem`'s HTTP API,
  which returns exactly the dict the SDK serializes into the tool result.
- Skills: byte size on disk of each `SKILL.md`, and of the frontmatter
  `description` that is paid every session whether or not it is invoked.

## 1. Connect cost - paid by every session, before any call

| | chars | ~tokens |
|---|---|---|
| 55 tool schemas (name + description + inputSchema) | 42,419 | ~10,600 |
| — of which descriptions | 28,242 | ~7,060 |
| — of which input schemas | 10,621 | ~2,655 |
| server `instructions` | 3,998 | ~1,000 |
| `dw` plugin + 3 skill frontmatter descriptions | ~1,500 | ~375 |
| **total** | **~47,900** | **~11,975** |

About **12k tokens**, ~6% of a 200k window. The ten fattest tools are 60%
of it:

| ~tokens | tool | desc chars |
|---|---|---|
| 662 | `validate_workflow` | 2,062 |
| 561 | `wait_for_job` | 1,949 |
| 495 | `run_workflow` | 1,245 |
| 470 | `get_memory` | 1,730 |
| 463 | `list_workflows` | 1,365 |
| 444 | `download_output` | 1,309 |
| 349 | `list_gallery` | 1,012 |
| 348 | `get_gallery_metadata` | 1,020 |
| 324 | `keep_output` | 785 |
| 318 | `export_job` | 1,008 |

The 20 smallest tools together are ~1,600 tokens - the long tail is not the
bill. Descriptions are 67% of the connect cost; input schemas are cheap and
mostly unavoidable.

## 2. Skills

| file | KB | ~tokens (body, on invoke) |
|---|---|---|
| `minimax-h3/SKILL.md` | 12.3 | ~3,070 |
| `ltx-2.5/SKILL.md` | 12.2 | ~3,060 |
| `minimax-music3/SKILL.md` | 10.6 | ~2,640 |

Frontmatter descriptions total ~1,150 chars (~290 tokens) - that is the only
part paid unconditionally. A body is paid once, when the family is actually
in play, and both video skills already sit against the 12 KiB harness cap.
**This is not where the money is.** The loop logs (`logs/*.log`) are agent
summaries, not transcripts, so invocation counts could not be recovered from
them; if that matters, the driver would have to log tool/skill use.

## 3. Result sizes - the actual bill

| ~tokens | call |
|---|---|
| **19,600** | `get_guide("workflows")` - **no section** |
| **16,000** | `get_guide("tasks")` - no section |
| **8,650** | `get_schema()` |
| **6,840** | `list_workflows()` - no shape |
| 5,160 | `get_guide("workflows", "Authoring a workflow from an agent")` |
| 3,030 | `list_workflows(shape="shot")` |
| 2,300 | `list_pipelines()` |
| 1,560 | `list_jobs(limit=20)` |
| 1,450 | `list_gallery(limit=50)` |
| 1,090 | `get_job_events(job)` |
| 850 | `list_guides()` |
| 750 | `get_workflow(name)` |
| 260 | `get_job(job)` |
| 145 | `get_workflow(name, variables_only=True)` |
| 80 | `validate_workflow(name=...)` |
| 21 | `get_memory()` |

Other shapes: `image` 770, `image-set` 560, `image-edit` 1,040, `sequence`
1,040, `audio` 300, `text` 200, `utility` 460. `shot` at 3,030 is four times
the median shape because that is where the templates cluster.

### Per-cycle arithmetic

A tester cycle that authors something new and runs it:

```
connect                                    ~12,000
list_guides + one guide section             ~6,000
get_schema                                  ~8,650
list_workflows(shape) + get_workflow         ~3,800
one skill body                              ~3,000
validate + run                                 ~300
wait_for_job x 11 (a 10-minute job, 55s cap) ~3,300
get_job_events + list_gallery                ~2,500
                                            -------
                                            ~39,550
```

...before a single line of the agent's own reasoning, and before any repeat
call. That is ~20% of a 200k window, and **the connect cost is under a third
of it**.

## The finding

**The tool schemas are not the problem. Single unbounded results are.**

One `get_guide` call with `section` omitted costs 19.6k tokens - more than
the entire 55-tool surface, in one call, and an agent can make that mistake
twice before noticing. `get_schema` has no way to ask for less at all. These
two are worth more than every description trim combined.

Two things also argue against aggressive description trimming: the long
descriptions are load-bearing (they encode rules that closed real tickets -
`get_memory`'s three response shapes from #80, `wait_for_job`'s lead-in
silence, `list_workflows`' curated-vs-derived cost distinction), and they
are paid once per session while a fat result is paid per call.

## Recommendations

Ordered by tokens saved per unit of risk. Items 1-3 change a result shape
and so are the part of #101 that needs approval before anything ships.

**1. `get_guide` without `section` should not return the whole document.**
Return the guide's section index plus its first section, with an explicit
note naming what was withheld and how to ask for it. Saves ~14k tokens on
the mistake, costs nothing when the agent does the right thing (which the
description already asks for). Result-shape change → approval. Cheaper
variant if that is too blunt: keep the whole document but cap it at a
configurable size and truncate with the same note.

**2. Give `get_schema` a `section` argument** (`steps`, `pipelines`,
`tasks`, `result`, `variables`, `configuration`). An additive optional
parameter; the no-argument call keeps returning everything, so nothing
breaks. An agent authoring a task step pays ~1-2k instead of 8.65k. Saves
~6-7k per authoring session.

**3. `list_workflows()` with no `shape` should summarise rather than
expand.** 6.84k tokens for the full catalog against 3.03k for the fattest
single shape. Either drop `details` when no shape is given, or return
per-shape counts plus names only. Result-shape change → approval. Saves
~4k on the call the MCP instructions already discourage.

**4. Move measured, model-specific narrative out of tool descriptions into
the skills and guides** - `wait_for_job`'s H3 629 s video-reference lead-in
and the block-cache gap arithmetic, `get_memory`'s pinned-host-cache
explanation. Keep every *rule* in the description (what `live: false` means,
that silence is not a hang, that the cap is 55 s); move the *numbers and
worked examples* to the `minimax-h3` skill, which loads exactly when H3 is
in play, and to a guide section. Realistic trim on the top five: ~8,350
chars → ~4,000, saving ~1.1k tokens of connect cost. Description-only, no
schema change - but it can reintroduce a closed ticket if done carelessly,
so it wants the tester re-running the cases those paragraphs came from.

**5. Do not touch the skills.** ~290 tokens unconditional, bodies paid only
when relevant, and two of the three are already against the 12 KiB cap.

**6. Not a `dw` change, but the biggest single lever: check whether the
loop's `claude` sessions defer MCP tool schemas.** This harness supports
fetching tool schemas on demand rather than rendering all 55 on connect -
in the session this assessment was written in, every `mcp__dw__*` tool is
deferred and costs a name until it is used. If the tester's
`--strict-mcp-config` sessions are *not* getting that treatment, turning it
on removes the entire ~10.6k connect cost with no change to `dw` at all,
and makes items 4-5 not worth doing. Worth confirming from the driver side
before spending engineering on description trims.

## What I would ship

Items 1 and 2 - they are where the tokens are, and item 2 breaks nothing.
Item 3 if the summary shape is agreed. Item 4 only if item 6 comes back
negative. The connect cost on its own (~12k, ~6% of the window) does not
justify a risky rewrite of descriptions that were each written to fix a
misreading.
