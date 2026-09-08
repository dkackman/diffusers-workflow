# Proposal: making CodeQL see the path validators

Status: draft, 2026-09-08. No implementation.

## Problem

Every release adds a few `py/path-injection` alerts from code scanning, and
every one so far has had the same shape: a path is joined and then confined
by `validate_path` in `dw/security.py` (or one of the wrappers around it),
and CodeQL flags the `os.*` or `shutil.*` call downstream because it does
not treat the validator as a sanitizer. The count on 2026-09-08 is 28 open
and 14 dismissed as "mitigated", all of that rule, plus the ten on #56 that
were dismissed the same way after a review found one real gap among them
(an `export_directory` job id of `.` named the exports root itself; fixed
in 7ead046 with a shape check).

Dismissing by hand works but scales badly, and it teaches the reviewer to
skim: the one real finding on #56 was flagged in exactly the same words as
the nine that were not.

## What already exists, and why it is not working

`.github/codeql/extensions/dw-models/` is a CodeQL model pack, added with
the workspaces PR (#37, 2026-09-05), that declares the validators' return
values as `path-injection` barriers with the `barrierModel` extensible
predicate:

```yaml
- ["dw.security", "Member[validate_path].ReturnValue", "path-injection"]
```

The approach is right. `barrierModel` and `barrierGuardModel` were added
to models-as-data for Python in CodeQL 2.25.2 (April 2026), the path
injection query's `SanitizerFromModel` reads barriers of kind
`path-injection`, code scanning default setup picks up model packs from
`.github/codeql/extensions/` without a workflow, and the repository is
scanned with CodeQL 2.26.4. Yet every alert listed above was created
*after* the pack landed: 26 to 43 on 2026-09-06 and 07, 44 to 53 on the
08th. The pack is not taking effect, and nothing in the scanning UI says
why.

Hypotheses, most likely first:

1. **The type string does not resolve.** Every caller imports the
   validators relatively (`from ..security import validate_path`,
   `from .security import ...`). The `type` column of a Python model is
   matched through the API graph's module-import resolution, which is
   built for absolute imports of library modules; whether a relative
   import inside the analysed package resolves to `dw.security` is not
   documented and is the first thing to test.
2. **The pack is not being loaded at all.** A malformed
   `codeql-pack.yml`, a name collision, or a default-setup restriction
   the docs do not spell out. Default setup exposes no log for this.
3. **The barrier is on the wrong node.** A barrier on the return value
   stops flow that *passes through* the call. The flow CodeQL reports may
   instead go around it: `validate_path` returns a value computed from
   its argument, and if the analysed body contributes a second path
   (the argument itself flowing to a sink inside `validate_path`, then
   summarised) the return-value barrier would not cut it. Unlikely, but
   the alert's path view would show it.
4. **`base_dir=None` is a real hole in the model.** `validate_path(p,
   None)` resolves and pattern-checks but confines to nothing, so a
   barrier on its return value claims more than the function guarantees.
   That does not explain the alerts, but it is why the model should sit
   on the wrappers that always pass a root (`validate_output_path`,
   `validate_workflow_path`, `validate_prompt_path`) and on
   `validate_path` only when the root argument is present, which the
   model language cannot express. Decide whether to accept the
   over-claim or narrow the model.

## Proposed shape

**Reproduce locally, then fix the model, then let the scanner confirm.**
Default setup gives no feedback loop; the CodeQL CLI does.

1. Install the CodeQL CLI (`gh extension install github/gh-codeql`, which
   also fetches the bundle) on the Mac.
2. `codeql database create --language=python dw-db` at the repo root,
   then `codeql database analyze dw-db codeql/python-queries:codeql-suites/python-code-scanning.qls --model-packs dkackman/dw-models --format=sarif-latest --output=out.sarif`
   with the pack path supplied through `--additional-packs
   .github/codeql/extensions`. Count the `py/path-injection` results.
3. Iterate on `dw-security.model.yml` until the count drops to the
   findings that are real: try an absolute-import form of the type, try
   the wrappers rather than `validate_path`, try `barrierGuardModel` on
   the `SecurityError` branch. Each try is one edit and one analyze.
4. Commit the working model. Code scanning re-runs on the next push and
   the open alerts close as "fixed" on their own, which is the only
   confirmation that matters.
5. Record in `dw/security.py`'s module docstring that the validators are
   modelled, and where, so the next person adding a validator adds a row.

A `scripts/codeql-local.sh` that does steps 2 and 3 in one command is
worth keeping, since the model will need the same loop each time a
validator is added.

## What this is not

- Not a workflow migration. Default setup honours repository model packs,
  and advanced setup is only needed if step 3 shows the fix requires a
  custom query, which nothing so far suggests.
- Not a reason to stop reading the alerts. Once the model holds, an alert
  of this rule means a path that did not go through a validator, which is
  exactly the signal the rule is for.

## Open questions

- Whether the fourteen "mitigated" dismissals and the ten from #56 should
  be reopened once the model works, so the scanner re-evaluates them. They
  would close as fixed if the model is right and stay open if it is not,
  which is a useful check; but reopening is a manual click each.
- Whether `validate_path` with no root should keep returning a path at
  all, or whether every caller should be made to pass one. That is a
  security question independent of CodeQL, and the model's over-claim in
  hypothesis 4 is the argument for looking at it.
