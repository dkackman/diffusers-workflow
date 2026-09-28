# Harness prompt, stage B: the metrics ratchet

Paste everything below the line into a Claude Code session opened in
`/Users/don/testing/harnest`, after stage A is committed. Stage B adds one
mechanical check; the permanent architecture guardrails (seam map,
architecture reviewer, consolidation cadence, build-vs-buy, new-module
approval, a CI ratchet in dw) are stage C, at dw's Phase 4 gate.

---

dw now measures its own architecture: `scripts/arch_metrics.py` prints a
JSON object of lower-is-better counts (modules, modules over 1,000 lines,
functions over 150 lines, functions over cyclomatic complexity 15, import
cycles, modules inside import cycles, duplicate-code blocks,
reference-prefix literals, test `patch("dw...")` targets, CLAUDE.md lines).
Read its docstring and `docs/stabilization/ROADMAP.md` ("Metrics") in the
dw checkout this harness uses (`$SOURCE_DIR`). Make the harness refuse
work that makes any of those numbers worse. Requirements:

1. **What "worse" means.** A session's own commits made a metric worse
   when it is higher at the session's `HEAD` than at its merge base with
   `origin/develop` (`git fetch`, then `git merge-base origin/develop
   HEAD`). Compare against the merge base, not against the committed
   `docs/stabilization/baseline.json`: the refactor improves the numbers on
   `develop` while sessions run, and a session branched before an
   improvement would otherwise be blamed for not having it. This is the
   same "own commits" rule stage A's three-dot diff uses.

2. **One measurement helper.** Measure both trees with the script as it
   stands on `origin/develop` (`git show
   origin/develop:scripts/arch_metrics.py`), run as
   `python <script> --root <tree>`, so both sides count by the same rules
   even when the session's branch predates a metrics change. Measure the
   merge base from an extracted tree (`git archive <sha> | tar -x -C
   <tmpdir>`), never by checking it out in the session's worktree. Both
   the driver and `guard.py` call the helper; do not copy it. The
   comparison is key by key, and a key missing on either side is skipped
   (`regressions()` in the script already does this; import and reuse it
   rather than rewriting it).

3. **Tooling.** The script needs dw's dev extras (grimp via
   import-linter, networkx, pylint, pygount, ruff). Make sure the
   implementer's checkout (`~/src/dkackman/dw-agent`) has them:
   `pip install -e '.[dev]'` in the venv the harness runs dw with, and
   re-run it whenever `pyproject.toml` changes on `develop`. If the script
   cannot run (import error, non-zero exit other than a regression), the
   gate **fails closed** with a message naming the missing tool and the
   install command. A ratchet that skips itself on a broken environment is
   not a ratchet.

4. **Mechanical gate (`agent-settings/hooks/guard.py`).** Refuse a hand-off
   (`handoff_gate`), and a `git push` of `develop` wherever stage A's push
   check can see the session, when any metric regressed. This is
   independent of the FREEZE switch: it stays on after the freeze lifts,
   for as long as `scripts/arch_metrics.py` exists on `origin/develop`.
   Word the refusal to name each regressed metric with its before and
   after value (`complex_functions: 21 -> 22`), then say what to do:
   bring the number back down in this session (split the function, remove
   the duplicate, reuse an existing module instead of adding one, patch
   with `patch.object` or an injected fake instead of a `patch("dw...")`
   string); if that is not possible, label the issue `stabilization`, set
   `owner:don`, comment the metric and the reason, and stop. There is no
   label that waives the ratchet.

5. **Prompts: pointers, not prose.** At most three lines in total, in
   `agents/implementer/core.md`: the hand-off is refused when
   `scripts/arch_metrics.py` gets worse against the merge base; run it
   before handing off. Name the hook; do not restate its rules.

6. **Tests,** beside the existing guard tests:
   - a regression refused, with the metric and both values in the message;
   - an unchanged metric set allowed, and an improvement allowed;
   - a branch that merged a newer `develop` carrying improvements it did
     not make, with no regression of its own, allowed;
   - a branch whose own copy of the script is older than `origin/develop`'s
     measured with `origin/develop`'s;
   - the script failing to import refused with the install message;
   - the check still on when `docs/stabilization/FREEZE` is absent.

   Use small fixture trees, never the real dw repository, so the suite
   stays fast. Run the harness test suite.

7. **Size discipline.** No new document. One `HARNESS-ROADMAP.md` entry of
   10 lines or fewer recording that stage B exists and stage C follows at
   dw's Phase 4. Do not grow `CLAUDE.md` by more than three lines.

Finish with a summary covering:
- every file changed;
- the helper's name;
- how long one gate check takes on the real dw checkout (measure it once);
- whether the push-time check could see the session;
- the test results;
- anything that would still let a regression reach `develop`.
