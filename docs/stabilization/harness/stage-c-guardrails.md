# Harness prompt, stage C: the permanent architecture guardrails

Paste everything below the line into a Claude Code session opened in
`/Users/don/testing/harnest`, after stages A and B are committed. Stage C
turns the freeze's temporary gates into the permanent ones. dw deletes
`docs/stabilization/FREEZE` only after this stage is committed here and its
tests pass.

---

dw (dkackman/diffusers-workflow) has finished its architecture
stabilization. The freeze is about to lift, and features come back. What
must not come back is the drift the stabilization removed: a module per
ticket, and a second copy of a rule that already has an owner. Read these in
the dw checkout this harness uses (`$SOURCE_DIR`):
- `docs/ARCHITECTURE.md`, the seam map: concept, owning module, rule,
  enforced-by;
- `docs/stabilization/ASSESSMENT.md`, "Guardrail principles";
- the root `CLAUDE.md`.

Stages A and B are your own work. Make the harness hold dw's architecture
without the freeze. Requirements:

1. **What the FREEZE switch still controls, and what outlives it.** When
   `docs/stabilization/FREEZE` is gone, `features_pass`, design and
   decompose, and the lead's `build` session all resume. The following
   become independent of FREEZE and stay on for as long as the file they
   read exists on `origin/develop`:
   - the hot-zone refusal (stage A), driven by `docs/stabilization/hot-zone.txt`.
     dw keeps using it for the files a refactor is restructuring. From now on
     read it from `origin/develop` (as stage B reads the metrics script), not
     from the session's commit: a session branched before a hot zone went
     live must still see it;
   - the metrics ratchet (stage B);
   - the new-module rule (item 2).

   Keep the one-helper rule: the driver and `guard.py` read each switch
   through the shared helper. Do not copy a check.

2. **New-module approval replaces the new-file refusal.** This reverses stage
   B's rule 4 ("there is no label that waives the ratchet") on purpose: after
   the freeze a planned new module has to be possible, and an approved rise
   is how. Stage B's merge-base comparison stays the detector. This stage
   adds a check of the waiver's paperwork, and only that check reads
   `baseline.json`, which stage B deliberately never read. After the freeze,
   new workflows, prompts, templates and plugin skills are ordinary feature
   work, so the stage A refusal for new files under `workflows/`, `prompts/`
   and `plugins/` ends with FREEZE. A new Python module under `dw/` or
   `dw_mcp/` is different: it raises stage B's `modules` metric, which the
   ratchet refuses with no waiver today.
   - Make `arch-approved` (Don-only, as now) the one waiver for a ratchet
     rise. It is honoured only when the session's own commits also raise
     `docs/stabilization/baseline.json` by exactly the metrics that rose,
     and the commit that raises it names the rise and why in its message.
   - Without the label, the refusal stands, worded as now: name the metric
     and both values, say what to do, and park the issue for Don.
   - dw's CI (`scripts/arch_metrics.py --check`) passes once `baseline.json`
     matches, and it knows nothing about labels. This hook is the only
     check that ties a raise to the label.

3. **An architecture review on every hand-off that touches `dw/` or
   `dw_mcp/`.** Use the reviewer role that already exists; do not add one.
   Give it one checklist, read from dw's `docs/ARCHITECTURE.md` at run
   time and never copied into the prompt:
   - **A second owner.** The change implements a rule a map row assigns to
     another module (it re-derives a reference prefix, a library search
     order, a media decode, a validation walk, a step-cache key...) instead
     of calling the owner. The finding names the row.
   - **Build vs. buy.** The change hand-rolls something a dependency in
     `pyproject.toml` already does. The finding names the dependency.
   - **The map.** A change that adds an owner (a new module, or a rule
     moving between modules) updates its `docs/ARCHITECTURE.md` row in the
     same change. dw's `tests/test_architecture_map.py` catches a row naming
     something that no longer exists; it cannot catch a missing row.
   - **Context.** Rationale goes in the owning module's docstring, never in a
     `CLAUDE.md` (stage B's `claude_md_lines` ratchet already refuses
     growth).

   A finding of the first two kinds blocks the hand-off until the
   implementer fixes it or Don labels the issue `arch-approved`. The map
   and context findings are fixed in the same session.

4. **A consolidation cadence.** Once a week, the curator runs
   `python scripts/arch_report.py HEAD=HEAD` in dw. Its change-coupling and
   hotspot tables cover a fixed window and set no thresholds of their own.
   - The curator keeps the previous week's tables in the harness's state and
     compares the two.
   - It files at most three issues, labelled `consolidation` and
     `owner:don`. Each is for a pair new to the top ten, or one whose shared
     commits rose by five or more, with both weeks' numbers as evidence.
   - It files nothing when nothing moved, and it does not fix anything:
     consolidation is designed, not drive-by.

   `arch-approved` now covers three things: stage A's new-file waiver while a
   freeze is on, the ratchet-rise waiver (item 2) and the review override
   (item 3). For all three, only Don may add it.

5. **Unpark what the freeze parked.** Stage A parked mid-build features with
   `stabilization` and `owner:don`, to resume when the freeze lifted. When
   FREEZE is gone, the lead lists the open issues carrying `stabilization`.
   For each, it either drops the label and restores the owner it had before
   (from the issue's label history), or leaves it with Don with a one-line
   comment saying why. It does this once, and reports the list in its
   summary.

6. **Prompts: pointers, not prose.** At most five lines in total across the
   role prompts:
   - the implementer is told the ratchet, the new-module rule and the
     architecture review exist, and names the hook and the map;
   - the reviewer is told where the checklist is (dw's map and this
     section);
   - the curator is told the cadence.

   Remove the freeze-only lines stage A added once they no longer apply.
   Do not restate rules the hooks enforce.

7. **Tests,** beside the existing guard tests:
   - FREEZE absent: features resume; the hot-zone and ratchet refusals still
     fire;
   - a new `dw/` module refused without `arch-approved`;
   - with the label but no matching `baseline.json` raise, still refused;
   - with the label and the matching raise, allowed;
   - a new `workflows/` template allowed after FREEZE is gone;
   - the reviewer's checklist loaded from a fixture `ARCHITECTURE.md`,
     never inlined;
   - the curator's cadence filing nothing on a quiet fixture report;
   - the hot zone read from `origin/develop`, not the session's commit;
   - the unpark pass restoring a parked issue's previous owner.

   Use small fixture trees, never the real dw repository. Run the harness
   test suite.

8. **Size discipline.** No new document. One `HARNESS-ROADMAP.md` entry of
   10 lines or fewer, recording that stage C replaced the freeze gates. Do
   not grow `CLAUDE.md` by more than three lines.

Finish with a summary covering:
- every file changed;
- which gates stay on without FREEZE and which end with it;
- how `arch-approved` and the baseline raise are checked together;
- the issues the unpark pass touched;
- the test results;
- anything that would still let a second owner or an unapproved module
  reach `develop`.
