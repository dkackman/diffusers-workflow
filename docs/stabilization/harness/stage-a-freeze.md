# Harness prompt, stage A: stabilization freeze

Paste everything below the line into a Claude Code session opened in
`/Users/don/testing/harnest`. Stage A goes in now; stage B (the permanent
architecture guardrails) comes at dw's Phase 4 gate.

---

dw (dkackman/diffusers-workflow) is entering a stabilization: a staged
refactor to consolidate code that grew one module per ticket. The plan lives
in the dw repo at `docs/stabilization/ROADMAP.md`; read it and
`docs/stabilization/ASSESSMENT.md` in the dw checkout this harness uses
(`$SOURCE_DIR`). Until it ends, dw is under a **hard freeze**: no new
surface of any kind. Testers and regression keep running, and the
implementer keeps fixing the defects they file, but only through existing
code. The refactor itself runs outside this harness and merges to `develop`
at each phase gate.

Make the harness enforce that. Requirements:

1. **One switch, owned by dw.** The freeze is active while
   `docs/stabilization/FREEZE` exists in the dw checkout at the commit the
   session runs against. The freeze lifts when that file is deleted from
   `develop`, with no harness change. Read it through one helper that the
   driver and `guard.py` share, and do not copy the check.

2. **Driver (`run-loop.sh`).** While frozen:
   - skip `features_pass`, so no design or decompose runs;
   - skip the lead's `build` session. A feature that was mid-build when the
     freeze started parks: the lead labels it `stabilization`, sets
     `owner:don`, comments that it is paused for the freeze, and it resumes
     after the freeze lifts;
   - keep implementer fix and triage, lead closeout for work already in
     flight, tester, reviewer, curator and nightly regression unchanged.
   Log one line per skipped pass saying why.

3. **Mechanical gate (`agent-settings/hooks/guard.py`).** While frozen,
   refuse a hand-off (`handoff_gate`), and a `git push` of `develop` if the
   session carries an issue identity the hook can read, when the session's
   **own** commits do either of these.

   Compute "own" as `git fetch` and then
   `git diff --name-only origin/develop...HEAD` (three dots, from the merge
   base; add `--diff-filter=A` for the new-file rule), not `HARNEST_BASE_COMMIT..HEAD`. The
   refactor lands on `develop` while sessions run, so a session that merges
   the newer `develop` into its branch would otherwise be blamed for the
   refactor's hot-zone commits. The existing ruff check uses a two-dot diff
   against the base; do not copy that pattern here.

   The two conditions:
   - It adds a file under `dw/`, `dw_mcp/`, `workflows/`, `prompts/` or
     `plugins/`. That is new surface. The refusal lifts only if the issue
     carries `arch-approved`, and only Don may add that label: extend the
     existing Don-only label rules so agents cannot add it.
   - It changes any path listed in `docs/stabilization/hot-zone.txt` (one
     path or `dir/` prefix per line, `#` comments). These are files the
     refactor is restructuring right now, so there is no label escape.

   Word each refusal to tell the agent what to do instead: label the issue
   `stabilization`, swap its owner to `owner:don`, comment which rule fired
   and on which path, and stop. Create the `stabilization` and
   `arch-approved` labels if they do not exist.

   If the issue identity is not available at push time, enforce at hand-off
   only, and say so in your summary. Do not invent a way to guess it.

4. **Prompts: pointers, not prose.** Role prompts ride in every turn, and
   this harness's own context is already large. Add at most five lines in
   total:
   - `agents/implementer/core.md` ("Park for Don"): while
     `docs/stabilization/FREEZE` exists, fix defects only. A fix that would
     add surface, touch a hot-zone path, or need the same edit in two places
     goes to Don with the `stabilization` label. Never add a bullet to dw's
     `CLAUDE.md`; rationale goes in the owning module's docstring.
   - `agents/implementer/triage.md` step 5: the same test, applied at
     triage.

   Do not restate the rules the hook enforces; name the hook.

5. **Tests.** Add tests beside the existing guard tests for each new refusal:
   - one case refused and one allowed;
   - `arch-approved` lifting the new-file refusal;
   - the hot-zone refusal ignoring that label;
   - both rules off when the FREEZE file is absent;
   - a branch that merged a newer `develop` touching hot-zone paths, with no
     hot-zone change of its own, being allowed.

   Run the harness test suite.

6. **Size discipline.** No new document. At most one short `HARNESS-ROADMAP.md`
   entry (10 lines or fewer) recording that stage A exists and that stage B
   replaces it. Do not grow `CLAUDE.md` by more than three lines.

Finish with a summary covering:
- every file changed;
- the helper's name;
- whether the push-time check could see the issue;
- the test results;
- anything in the harness that would still let new surface reach `develop`
  during the freeze.
