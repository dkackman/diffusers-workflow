# Worked example: a cold-session drill report

From the drill run 2026-09-07, against the shape-first catalog discovery
work (before any model-family skill existed). Kept as an example of what a
useful drill report looks like — what to capture, and how findings turn into
follow-up items rather than just a pass/fail.

## Cold-session probe, 2026-09-07

A fresh Claude Code session (empty directory, no `CLAUDE.md`, no memory) was
asked for "a short multi-shot video with cuts between the shots" against a
server on this branch. What the access log showed, in order:
`list_workflows(shape=sequence)`, `list_workflows(shape=shot)`,
`get_workflow` on `templates/assemble-and-score` and
`templates/ltx2/text-to-video`, `list_assets`. It never fetched the
unfiltered catalog. It shipped three LTX-2 shots cut with assemble-and-score:
two visually related, the third an unrelated nature scene.

What that says about the catalog as built, and what remained to fix:

- **The shape-first entry works cold.** Discovery cost two compact filtered
  listings and two definitions.
- **Without `cost`, an agent picks the shortest path, not the best one.**
  Every `cost` was null, so nothing distinguished `ltx2/text-to-video` (no
  input media, "on a single 24GB card") from the MiniMax H3 keyframe route
  (`chained`, `image-conditioned`, `needs-input-media` — a longer chain with
  invisible prerequisites). Authoring `cost` on the shot baselines was the
  first lever.
- **Traits say what a workflow needs, not when you want it.** The listing
  carried `identity-referenced` and `image-conditioned`, but nothing said
  that cuts between shots imply continuity, so the agent sampled each shot
  fresh. A guide on keeping shots consistent (generate the subject once and
  reference it, or pin keyframes) followed from this.
- **Several summaries were weak as first sentences** because fun-chapter
  descriptions read as prose rather than a summary. A declared `summary`
  fixed that.
- **A rich workspace `CLAUDE.md` pre-empts discovery entirely.** The same
  prompt in a workspace with a film playbook produced a full plan without a
  single catalog call. The compact listing serves agents starting cold; a
  playbook and the catalog need to share the knowledge rather than compete
  for it.
