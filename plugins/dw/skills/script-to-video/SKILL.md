---
name: script-to-video
description: Use when a dw MCP server is connected and the user hands over a script or prose and wants it turned into a video end to end - decomposing it into a shot list, casting recurring characters once, picking each shot's template and family, validating cost before spending, running and diagnosing, and handing the finished shots to the cut-and-score pass. Does not cover model auto-tuning or writing new pipeline JSON.
---

# Script to video on a dw server

Turning a script into shots is the one piece none of the other skills
cover. Everything downstream of a shot list already exists: `dw:minimax-h3`
and `dw:ltx-2.5` pick a template per shot's shape, `h3-prompt-writing`
writes the structured prompt, `dw:series-episodes` keeps a cast consistent,
and `templates/assemble-and-score` cuts and scores the result. This skill
is the decision tree that connects them, in order. It does not add anything
to the engine - see "Not in scope" below for what it deliberately declines.

## 1. Read the script and the server

`get_server_info` for device and workspace, the way every dw skill starts.
Read the script as prose and dialogue, the way a person would - this skill
does not parse screenplay formats.

## 2. Decompose into a shot list, not a shot count

The unit that matters is *shots*, not lines or scenes: one scene may be
several shots, and a `dw:minimax-h3` shot runs 5.17 to 14.4 seconds (the
`17*n+5` frame grid, 124 to 345 frames). For each shot, name it, note
which character(s) appear, what happens, and roughly how long it runs.

Fill in this shape before writing any prompt:

- **Cast**: per character, a voice description, sex, and the reference
  asset (`asset:cast/<name>`; step 3).
- **Locations**: each place a shot happens, named once and reused.
- **Per shot**: the above, plus `opening_state` (what is on screen, who
  stands where, what is held, at frame one) and `ending_state` (the same at
  the last frame). A shot's `opening_state` is the previous shot's
  `ending_state`, so the cut between them is written down rather than hoped for.

Whatever the user supplied is locked; copy it, don't improve it. A field
they left blank stays blank - ask, or mark it unspecified - and you invent
no character, place or line to fill it.

## 3. Cast recurring characters once, before any shot generates

Read `dw:series-episodes` step 0 before drawing anything: every character
appearing in more than one shot gets a portrait (and voice clip, if they
speak) drawn once and `keep_output`'d as `asset:cast/<name>`, referenced
from every later shot with `from_file`, never `from_previous_result`. A
script with one continuous scene and no recurring cast can skip this and
let the family template draw its own portrait. This is the step that goes
missing silently - nothing validates "the same character looks the same"
across separately generated shots, so it has to be deliberate.

Large `run_workflow` arguments: `save_prompt(name, {"text": ...})` each shot prompt under a
script-prefixed name (the library is shared across workspaces) and pass
`prompt:<script>/<shot>` references in the list; they resolve in nested
arguments.

## 4. Per shot: pick the shape, write the prompt

Silent action or establishing shots go through a `dw:ltx-2.5` template
(`templates/ltx2/text-to-video`, `templates/ltx2/image-to-video`, or a
chained one for a longer take). Anything with dialogue, a held voice, or a
scored montage goes through `dw:minimax-h3`'s `templates/minimax/dialogue-short`
or `templates/minimax/music-video`, prompted with `h3-prompt-writing`'s
structure (`subject_definitions`, `retention_analysis`,
`detailed_description`, `overall_soundscape`, `non_diegetic_music`) rather
than the raw script line, checking `list_prompts` first for a working
example from the library that already uses these structures rather than
drafting one from nothing. A shot that calls for generating several
candidates and keeping the sharpest one deterministically, with no agent
judgment call at run time, goes through `templates/best-of-n-to-video`
instead of a single draw. Read the chosen family's own skill before writing
a single prompt - the hard rules (frame count, canvas, reference limits)
live there, not here.

## 5. Test-render, validate and quote cost before queuing

Render two to four representative shots first (a dialogue shot, an action
shot, the hardest cut) and judge them before the full batch.

`validate_workflow` against the chosen template with the assembled `shots`
argument, read `plan.estimate` and `plan.downloads_required`, and state the
cost before calling `run_workflow`. Only pass `acknowledged_cost` once a
human (or an explicit pre-authorization) has signed off - this is a real
gate, not a formality, and skipping it here is the one thing this skill
must not do even when running unattended.

## Run and judge

Wait with `wait_for_job`, asking for the plan's estimate plus a margin as
`timeout_seconds`; `timeout_capped` says the server's cap cut it short, and
a `still_running` reply means call again. If the call dies client-side ("timed out") first, call again with `timeout_seconds` 60 and loop; a running job lists `finished_steps`. After the job completes, read `get_job_events` / `get_gallery_metadata` for
the warnings the engine already emits (`audio_no_headroom`, `audio_clipped`,
an elision diagnostic) and `rerun_job(new_seed=True)` a shot that reads
wrong - not by inventing a new heuristic, by reading what is already
surfaced. On an H3 shot with dialogue a new seed also redraws the voice: in
a lem probe (#610) three seeds of one line gave a different speaker each
time without an audio reference, and only a narrower spread with one
(`voice` / `voice-timbre-reference`). So when the voice is right and the
picture is wrong, change the prompt and keep the seed; reach for
`new_seed` only when the voice is pinned by a reference or does not matter.
Give each shot a budget of three failed attempts, then stop and ask the user
rather than spending more. Each shot's files are organized by `subfolder` - typically `final`
for the deliverable video or `intermediate` for test frames; read the
manifest to determine what was written where.

## 6. Cut, score, deliver

Once every shot exists, hand off to `templates/assemble-and-score` (or the
`dw:series-episodes` five-beat procedure, if this script is one episode of
a series) for the recut/bed/match_levels/normalize/pair pass. That skill
already owns this step - do not re-derive it here.

A sung multi-shot piece is reviewed by its lip-sync target: run
`get_guide("tasks", section="Checking the lip-sync target")` on the cut
before delivering it.

To take the project home, call `export_job` once per job of the project, then
fetch each zip's `open_url` and unpack it into `exports/` under the working
directory (per its `next`).

## Not in scope

- **Automatic model speed optimization.** If a model is slow, that is a
  human-in-the-loop task informed by `observed_minutes`/`observed_runs` in a
  workflow listing (or `plan.estimate` with `basis: "observed"`), not
  something this skill attempts unsupervised - the search space is large and
  a wrong choice can silently change output quality.
- **Workflow authoring from scratch.** This skill picks among existing
  catalog templates; a script whose shape no template covers is a
  "propose a new template" conversation, not something to improvise at
  run time.

## Sources

This skill adds no new mechanism of its own - it is the decision tree over
what already exists: `dw:minimax-h3` and `dw:ltx-2.5` (the family template
pickers), `h3-prompt-writing` and the family prompt-writing conventions
(the structured-prompt skills), `dw:series-episodes` (cast consistency and
the recut/bed/match_levels/normalize/pair pass), `plan.estimate` and
`acknowledged_cost` (the cost gate), and the warning kinds `dw/result.py`
and `dw/elision.py` already emit (`audio_no_headroom`, `audio_clipped`, an
elision diagnostic).
