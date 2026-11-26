# The UI rule tools' read and write analysers agree (#1690)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out,
2026-11-26). Plan v2 was approved by the delegate for Don (2026-11-25), with
both questions at their defaults and one gate tightened. The verdict was
"build smaller": the shared analyser module the idea asked for was declined,
and one stage, #1707, fixed the three places where the duplicated analysis
disagreed. It was built, reviewed and closed.

## The idea

The UI refactor review of 2026-11-25 (b2e8ff73..f8af3f02) found three rule
tools that each analyse imports and scopes over the UI source:
`ui/scripts/arch-metrics.mjs`, `ui/src/lib/freshness.test.ts` and
`ui/scripts/design-rules.test.ts`, about 1,500 lines between them. It
proposed one shared analyser module that all three would consume, so a new
Svelte syntax or helper-resolution rule would be taught once.

## What the sweep found (plan v1, develop 7f4760a2)

- **Three parsers, not one analysis copied three times.** arch-metrics uses
  ESLint's parsers and scope manager; freshness uses `svelte/compiler` plus
  the TypeScript compiler API; design-rules uses regexes and already imports
  `sourceFiles` from arch-metrics. Only about 550 of the 1,500 lines are
  machinery, and most of it is parser-specific. The parser-free part a module
  could share is about 40 lines (`resolveImport`/`resolved`, the import-name
  list, `sourceFiles`).
- **A module would break the harnest ratchet,** which copies only
  `arch-metrics.mjs` over the tree (`docs/stabilization/ui/phase-4.md`).
- **No fix had ever been made twice.** The last 15 commits touching the
  three files changed rules and tables, or made one-file fixes.
- **Three real disagreements** between the "does this read" and "does this
  write" analysers:
  1. arch-metrics `declared` took functions and variables but not classes,
     while freshness `declarations` took classes too. An exported class
     whose method wrote through `api` escaped `unguarded_mutations`. None
     wrote at the time, so the count was right by luck.
  2. freshness `readingExports` matched identifier *text*, so `obj.name` or
     a shadowing local counted as a use of a reading helper (a false
     positive, failing toward strictness). arch-metrics followed real scope
     references.
  3. freshness `read` and `strayHelperReads` read only `ast.instance`, so a
     read or helper import in `<script module>` was invisible (failing toward
     permissiveness). Nine components had a module script; none imported a
     reading helper yet.
- Relative import resolution and import-name extraction behaved the same in
  both tools: no fix.

## Decisions

Ruled by the delegate for Don, 2026-11-25:

- **The module: declined.** It stays declined until a reopen trigger fires
  (below).
- **Q1 [none]:** default. The tester verifies at `develop` on a scratch tree
  from `ui/`, without the GPU.
- **Q2 [none]:** default. The four complex functions in `freshness.test.ts`
  (`freshnessOf`, `declarations`, `readingLocals`, `helperCalled`) are left
  alone: arch-metrics doesn't measure test files.
- **Amended gate on fix 2:** if the scope-aware walk would remove any
  freshness-table entry, exception or declared read, the builder stops and
  hands it back as a `loosen` ask instead of editing it out. Added entries
  are added and reported.
- The earlier keep-open condition ("worth doing if #1689 adds a layering
  rule") was met in name only: #1689 stage B reuses arch-metrics' own import
  graph and adds no fourth analyser.

## What was built

**Stage A, #1707** (f31cb57c, merged to `develop` as 2f507d41). Tooling only:
`ui/scripts/arch-metrics.mjs`, `ui/scripts/arch-metrics.test.ts`,
`ui/src/lib/freshness.test.ts`, `docs/ARCHITECTURE.md`.

1. **`declared` takes a named `ClassDeclaration`,** so `writingExports` and
   `closeOver` see class bodies. `unguarded_mutations` on the real tree was 0
   before and 0 after; the baseline was not touched.
2. **`readingExports` is scope-aware** (`topLevelUses`, `scopeNames`,
   `isKey`, over the TS AST, not the type checker). A use is a reference to
   the top-level name only: not the name side of `x.name`, not a property,
   method or destructured key, and not inside a function, block, loop or
   catch that declares the same name. **Gate result:** a dump of every
   helper's reading and watch exports, `bareHelperReaders` and
   `helperReadsOutsideResources` over the real tree was identical before and
   after. Nothing was added or removed, the gate didn't trigger, and the
   table has no edits.
3. **Both Svelte script blocks are read.** One `scripts()` helper returns
   `[ast.module?.content, ast.instance?.content]`, and `read` and
   `strayHelperReads` both use it.
4. **Cross-reference comments.** `declared`'s docstring names freshness'
   `declarations`, and the freshness header names arch-metrics' `declared`
   and says both script blocks are read. The ARCHITECTURE.md "Keeping UI
   views fresh" row says both script blocks are read.

Tests (vitest, each failing on the old analysers): an exported class, and an
unexported class reached through an exported factory, writing through `api`
count in `unguarded_mutations`, while a non-writing class doesn't; a property
access, shadowing parameter, shadowing block `const` and object key don't
mark a helper as reading, while a call through a local and a class method do;
a `<script module>`-only import on a page with no resource, and a stray call
from either block, are both named.

### Deviations

- The plan's `arch-metrics.mjs:575` comment went on `declared`'s own
  docstring instead, because line numbers had drifted and `declared` is the
  walker the comment pairs with.
- The ratchet was already red on `develop` before this stage
  (`files_over_size_ceiling 0 -> 1` for `src/lib/api.ts`, and
  `long_functions 0 -> 1`, from #1701's 796aec95). This stage changes no
  measured file. The tester linked #1713 (#1689 stage A, proxies advisory)
  as the blocker for "`--check` exits 0".

## Verification

- **Build session:** at f31cb57c, `npm test` 239 files / 3068 tests passed,
  lint, `svelte-check` and prettier clean.
- **Tester (2f507d41):** the MCP and UI half of C-F1351 passed (server
  answered; `/` and the gallery loaded with no page or console errors). The
  record checks for C-F1346 to C-F1350 went to a source reader, since the
  tester can't run `npm`. The result was "incomplete", not a bounce.
- **Docs review (7055ee40):** the ARCHITECTURE.md row, the comments and the
  fixtures match what the hand-off claims.
- **Don (7055ee40, clean worktree):** `npm test` exit 0 (241 files / 3090
  tests), `npm run metrics -- --check ../docs/stabilization/ui/baseline.json`
  exit 0. He accepted the "fails on the old analysers" probes on the docs
  reviewer's read and the build report, and closed the stage as reviewed.

**Bounces:** none. **Cost:** no stage comment recorded a `usage:` figure; the
plan estimated $2-3.

## Deferred and why

- **The shared analyser module.** It would share about 40 parser-free lines,
  or force one tool onto the other's AST (about $6-10, plus a change to the
  harnest ratchet's single-file copy). No fix had ever had to be made twice.
- **Splitting the four complex functions in `freshness.test.ts`** (Q2): no
  metric measures test files.
- **design-rules' regex import walk:** it does its own `.svelte` job and
  doesn't need to match the other two.

## What would reopen the module

- A fourth tool that parses imports or scopes (for example, a layering rule
  that builds its own walker instead of reusing arch-metrics' import graph).
- A syntax or resolution fix made in one of these files that then has to be
  made in another: `$lib` aliasing, `index.svelte.ts`, or a new Svelte script
  form.

Either is evidence for a parser-free `ui/scripts/analysis.mjs` holding
`sourceFiles`, `resolveImport` and the import-name list, with the harnest
ratchet copying it too.
