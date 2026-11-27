# UI gallery reads narrowed, one text-filter rule, one grid helper (#1670)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out,
2026-11-26). Plan v1 approved by Don's delegate on 2026-11-25 ("build
smaller"), three stages: #1701 (narrow reads), #1702 (one text filter) and
#1703 (Gallery/Assets grid plumbing), all built, verified on lem and closed.

## The idea

The UI refactor review of 2026-11-25 (b2e8ff73..f8af3f02) found six pages
each fetching the whole gallery listing (`api.gallery` always sent
`limit=100000`) and projecting it client-side, again on every `gallery` feed
move. Its fix direction was a shared resource or server-side projections. It
also named drift: text filters with different rules, and Gallery/Assets grid
plumbing written twice.

## What the sweep found (develop b7150048, checked at 128841e7)

- `_gallery_entries` (`dw/server/routes/gallery.py:108`) walks the whole
  output tree on every call, then filters, sorts and slices. There is no
  listing cache, so `folder=`/`limit=` save transfer and client work, not the
  server walk.
- Pages are routed one at a time, so the six reads never happen at once. Don's
  largest workspace held 1,470 files; nobody had reported a slow gallery.
- Only two pages had an exact narrow read the route already offers: Workflow
  (`folder=<name>&limit=6`) and Compare (2-4 names).
- Three matchers were the same "every word a substring" rule (Gallery
  `matchesText`, `suggest.ts` `matchesWords`, Jobs `jobMatches`); five
  filters matched one substring. The Workflows catalog search is whole-word
  and ranked, on purpose.
- The Gallery/Assets `stray` checks use opposite rules (Gallery counts every
  file, Assets only the shown ones), so they were not a plain duplicate.

## Decisions

- **Verdict: build smaller.** No cache and no new API. UI only.
- **Q1 [default]:** Prompts, Assets, the schema page and the suggestion box
  move to the every-word rule. Results only widen.
- **Q2 [scope]: cut**, each with its trigger:
  - a shared cross-page listing resource: back when #1647/#1648 (`resource`
    reload races) are closed *and* a navigation's listing is measurably slow
    (a field report, or a workspace past ~10k files);
  - server-side proof projections and a listing cache (Workflows
    `latestProofs`, Jobs `proofByRun`, Overview): back on the same evidence,
    as a server design of its own (a cached walk in `_gallery_entries`), not
    more query parameters;
  - bounding Gallery `durations` like Assets': left alone, since it would drop
    durations from older tiles. Back with the cache.

## What was built

### Stage A, #1701: narrow reads (796aec95, merged 9e4091cb; e2e 15692e0d, merged 279f955f)
- `api.gallery` takes `{ folder, limit }`. `folder=` goes first and is sent
  even when empty (a root-level file's folder is `''`).
- `WorkflowPage` reads `/api/gallery?folder=<identity>&limit=6` for its proofs
  strip.
- `ComparePage` makes one `folder=` read per distinct folder among its names
  and looks each name up there. These reads keep the default `limit=100000`,
  because a compared file may be older than a folder's newest few.
- `folderOf(name)` (`ui/src/lib/media.ts`) mirrors `dw/runs.py`
  `split_run_path`, so Compare knows which folder to ask for.
- Watch lists and `freshness.test.ts` are unchanged.
- `ui/e2e/workflow-proofs.spec.ts` drives the real path: list, click, the one
  narrow read, the tile in the strip.

### Stage B, #1702: one text-filter rule (51ade69a, merged b1d349a5)
- Most of it had already landed in #1688 (`aba3a960`), which added
  `ui/src/lib/matchWords.ts` and moved Gallery, Jobs, Prompts, Assets and
  Schema onto it. This stage deleted `suggest.ts`'s duplicate and moved
  `MediaPicker`, `refcomplete` and `Suggest.svelte` onto `matchWords`.
- `docs/ARCHITECTURE.md` has a **List text filters** row: `matchWords` is the
  rule; the catalog's ranked search is the one exception.
- Deviations: the function is `matchWords`, the name #1688 shipped, not the
  plan's `matchesWords`. `prompts.ts` `presetForIntendedModel` is unchanged:
  it pairs a prompt's model with an enhancer preset both ways, is not a list
  filter, and a one-way match would narrow it, which Q1 rules out.

### Stage C, #1703: grid plumbing and Assets sections (5ce33ed1, merged b6a2f003)
- `ui/src/lib/assets/sections.ts` holds AssetsPage's listing logic as pure
  functions (`matchedAssets`, `visibleAssets`, `onPage`, `assetSections`,
  `pickableAssets`, `viewerNames`, `folderGroup`), with `sections.test.ts`.
  It groups with `groupNames`; the Gallery's `gridFiles`/`gridGroups`/
  `inGroupOrder` are typed to `GalleryFile` and didn't fit `AssetFile`.
- `ui/src/lib/mediaGrid.svelte.ts`: `revealOnce` replaces both pages'
  `revealed` effect, `clearOnEscape` both Escape handlers (the Gallery keeps
  its rule through `keepHidden: true`; Assets clears every pick).
- `stray` already had a shared skeleton, `missingFile({ names })`; each page
  keeps its own names. `durations` untouched; `HELPER_EXCEPTIONS` unchanged.
- arch-metrics `duplicate_blocks` lowered from 112 to 110.

## Deferred

- Everything Q2 cut, on the triggers above.
- **For Don, not decided:** should the Gallery and Assets each share one
  Escape rule and one `stray` rule? Stage C kept both pages' rules as they
  were.
- #1653 (a request-narrowed gallery poisons `outputLibrary`) was left to its
  own fix.

## Cost and bounces

Planned at about $3 + $3 + $4 ≈ $10 of agent work and no GPU. The stage
comments name no `usage:` figures, so actual cost is not recorded.

- **#1701: one bounce.** The tester looked for text in the proofs strip, but
  its tiles have no caption (`caption="none"`) and its thumbnails came from the
  browser cache. No code change; the lead added the e2e spec and named where
  the strip sits and how to count its tiles.
- **#1702: none.**
- **#1703: one bounce.** Two case expectations (C-F1331 step 4, C-F1333
  `Name`) described behaviour that was the same before and after the stage:
  `Picks` keeps a filter-hidden pick on both pages, and `Name` sorts folder
  headings while files stay newest first. No code change; the amendment is
  filed as dkackman/harnest#307.
- **Parent final check: two bounces,** fixed forward by #1769 and #1778
  (below).

## Fix-forward after the final check

Added 2026-11-26 by `claude-opus-5-5` via `anthropic` (close-out).

- **The bounce.** The parent's final check failed C-F1322's Workflow half. A
  live note change never showed on the Workflow page, and the screenshot
  showed no proofs strip.
- **#1769, fix-forward stage: no code change (tests d9b8a04d, merged
  ab2cf442).** The Workflow page has never shown notes. Its strip draws tiles
  with `caption="none"`, and did before #1701 too. So the note line plan v1
  wrote for that page was a gap in the plan, not something the build missed.
  The delegate ruled (`loosen`) that the case's Workflow half checks the
  strip's tile and its scoped re-read instead. The note check stays on the
  Compare half, which passed. The case amendment is dkackman/harnest#304. The
  missing strip was also not a build miss: the strip's read is `quiet`, and
  the screenshot was taken before that read answered. #1769 added a
  `WorkflowPage.test.ts` case for a folder with a slash in its name
  (`templates/image-processors`). The tester verified the strip tile and a
  feed-driven `folder=…&limit=6` re-read on lem at ab2cf442.
- **C-F1332** (an opened file is on screen without scrolling) couldn't be
  judged, because `ui-check` reported no viewport. harnest#325 added
  `in_viewport` to `ui-check inspect` (harnest `c92d20e`) and amended the
  case. It is closed.
- **The second bounce.** The next final check judged C-F1332 and failed it
  on both pages. An `open` hash marked the tile `active` but never scrolled
  it into view (Gallery tile at top 9551, Assets at 1910, both
  `in_viewport: false`). Neither page had scrolled before Stage C either, so
  plan v1's Stage C line "opening a file by its `open` hash scrolls it into
  view" was a gap in the plan, not something the build missed.
- **#1778, fix-forward stage: the scroll (a848b8e4, merged d3b88d51).** The
  delegate ruled `default`: build the scroll and keep C-F1332 as written.
  When `mediaGrid` reveals an opened file, the page scrolls that file's tile
  to the middle of the window once it has rendered. It does this once per
  opened name, and leaves a tile alone if any part of it is already on
  screen. Closing the viewer, scrolling away or a reload doesn't pull the
  page back. The rule covers every open, so stepping through the viewer
  also scrolls the grid behind it. Added a `mediaGrid` unit test and an e2e
  spec. The tester verified C-F1332 on lem at d3b88d51 on both pages, with
  `in_viewport: true` at tops 608 and 478.
