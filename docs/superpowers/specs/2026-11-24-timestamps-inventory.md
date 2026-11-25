# Timestamps inventory (#1599)

Task 21 of `docs/superpowers/plans/2026-11-24-ui-followups.md`. For each noun the
glossary names: the times the server returns, the views that show one today, and the
decision for Task 22.

**The rule** (`ui/GLOSSARY.md`, *Formats*): a time is `<Time>`
(`ui/src/lib/ui/Time.svelte`). In a list it is relative, `changed <relative>`, with the
exact time on hover and in `datetime`, and it ages on the shared clock
(`ui/src/lib/clock.svelte.ts`). In a detail it is exact: `Created <exact>`,
`Changed <exact>`. `by the agent` is said only where the server records who. A time
that does not parse (a stored ledger string `Date.parse` cannot read) reads
`unknown time`, with no `datetime`, rather than taking the page down.

**The aim** (Don, 2026-11-24): not exhaustive. A time goes where it helps a person
tell what is recent from what is out of date as they move through the UI: a list they
pick from, or the detail of a thing they are about to reuse. Where a time would be
noise, the last column says "not added" and why.

Sources: `ui/src/lib/generated/api-schema.ts` at 81b526f9, and
`grep -rn "formatMtime\|formatDay\|formatExact" ui/src`. After Task 21 every view
listed under *Shown today* renders through `<Time>` in the format it had (exact
`formatMtime`, or `mode="day"` for `formatDay`). Lists that show an exact time today
move to relative in Task 22, where the decision says so.

## The nouns

| Noun | Times the server returns (field, route) | Shown today (view, format) | Decision for Task 22 |
| --- | --- | --- | --- |
| **workflow** | none (`WorkflowList`, `GET /api/workflows`) | none | **Add**: `mtime` (and `created` where the filesystem has a birth time) through `times=true`. WorkflowsPage rows: relative `changed`. WorkflowPage: exact `Changed` (and `Created` when present). EditorPage file bar: last saved, exact. Newest-first sort on WorkflowsPage. Overview's featured-workflow cards (`OverviewPage.svelte` `featured`): **not added** (a card is a shortcut to one of a few workflows, ordered by whether it has produced something; its caption is the name and its proof, and the rows a person orients by are on WorkflowsPage, one click away). |
| **prompt** | none (`PromptList`/`PromptCard`, `GET /api/prompts`) | none | **Add**: `mtime` through `times=true`. PromptsPage cards: relative `changed`. PromptEditorPage: last saved, exact. Newest-first sort. |
| **asset** | `AssetFile.mtime`, `ShadowedAsset.mtime` (`GET /api/assets`) | AssetAside (the detail): `<size> · <exact>` | **Add**: AssetsPage tiles, relative `changed` in the `OutputTile` `meta` slot; newest-first sort. The aside stays exact. Shadowed copies: **not added** (the row says which copy wins; its age does not change the pick). |
| **output** | `GalleryFile.mtime` (`GET /api/gallery`) | GalleryAside (the detail): `<size> · <exact>`; the gallery's date headings (`gallery/order.ts` `dateGroup`) | **Add**: GalleryPage tiles, relative `changed` in the `meta` slot. The aside stays exact. Compare cells: **not added** (Compare holds outputs side by side the person already chose; age is not the question there). Overview's recent outputs (the strip, `caption="none"`): **not added** (it is the newest few outputs, newest first, so its order already says what is recent; each tile opens the gallery, whose aside gives the time). |
| **workspace** | none (`WorkspaceList`/`WorkspaceInfo`, `GET /api/workspaces`) | none | **Add**, in one place: the Sidebar's workspace list, the one list where a person picks among workspaces. Each expanded row (`Sidebar.svelte` `wsRow`, open and shut) gets a muted relative `changed` beside its size, so a person switching sees which workspace they were last in and which has gone stale. `mtime` comes through `times=true` on `GET /api/workspaces`; Task 22 says what it measures (a folder's own mtime moves only when an entry directly in it changes, so it may need the newest of the folder's runs). **Not added**: the collapsed rail (a dot each, no room); a prefix group's header (it folds names, it is not a pick); the Jobs page's workspace filter and the editor's `WorkflowOrigin` destination (both are `<select>`s, whose `<option>` holds plain text that cannot age or carry a hover, and both choose by name); the Overview (one workspace's own page, not a list of workspaces). **This overrides the plan**: Task 22's file list says "the Sidebar and Overview workspaces" (plan line 733), and an earlier draft of this row put the time on an "Overview workspace list" that does not exist and left the Sidebar out. Task 22 follows this row. |
| **job** | `JobSummary`/`JobDetail`: `created_at`, `started_at`, `finished_at` (`GET /api/jobs`, `GET /api/jobs/{job_id}`) | JobRow (Jobs list): start, exact, hover started/finished/waited; JobHeader (detail): start, exact, hover started/finished; Overview's job rows: last of finished/started/created, exact | **Change**: JobRow and Overview job rows go relative (`<Time>` default mode), keeping the started/finished hover. JobHeader stays exact. The list already sorts newest first. The event log's per-event times: **not added** (they are offsets from the start, `+4m 03s` through `formatDuration`, not dates). |
| **request** | `LedgerSummary.updated_at` (`GET /api/workspaces/{name}/requests`); `OpenRequestSummary.updated_at` (`GET /api/requests`); `LedgerRequest.agent_wrote_at`, `LedgerHistoryEntry.at` + `by` (`GET /api/workspaces/{name}/requests/{request_id}`); `LedgerItem.changed_at`, `review_set_at` | RequestsPage rows: `updated_at`, exact, hover `last written <formatExact>`; RequestHeader: the agent's latest revision, exact, hover the agent's `why`; PlanChanges: each revision, exact; RequestPage: "Showing the plan as of <exact>" | **Change**: RequestsPage rows go relative `changed`, with **no** `by`: neither `LedgerSummary` nor `OpenRequestSummary` carries the last writer, and adding one is a server change Question 2's `times=true` does not cover. RequestHeader and PlanChanges stay exact; PlanChanges already says who per revision. Overview's open requests: relative `changed`. Item `changed_at` / `review_set_at`: **not added** (the item's state chip and the review countdown already say what is current; a second time per item is noise). |
| **note** | `LedgerNote.at` (last changed), `LedgerNote.created` (first saved); no `by`: every note is the person's (`GET /api/workspaces/{name}/requests/{request_id}`) | NoteList via `noteState.noteLine`: "Saved <exact> · edited <exact>." and the agent's later revision, as text | **Change**: the note's line moves onto `<Time>`: `written <relative>` with the exact time on hover, `· changed <relative>` when `at` differs from `created`. "by the agent" stays on the line about the agent's revision (`agent_wrote_at`), the only author the ledger records. **This departs from the plan's wording**, `written <time> by <the agent\|you>` (plan line 733): `LedgerNote` has no `by` field, so a note's line cannot say who wrote it, and every note is the person's, so `by you` would say nothing. Task 22 follows this row. |
| **model** | `ModelRepo.last_job_use`, `last_accessed`, `last_modified`; `ModelRevision.last_modified`; `ModelDownload.finished_at` (`GET /api/models`, `/api/models/downloads`) | ModelsPage: "last used" (`LastUsedCell`, day, `(disk)` fallback) and each revision's `last_modified` (day) | **Keep as is** (`mode="day"`): the column is a sort key for "what can I delete", where the day is the unit; relative would read "40 days ago" for what the column needs as a date. Download `finished_at`: **not added** (it only times the row out). |
| **LoRA** | none (`LoraRow`, `GET /api/loras`) | none | **Not added**: the listing has no time, and Question 2 limits LoRAs to where it does. |
| **disk cache** (Server page) | `DiskCacheInfo.last_used`; `DiskCacheEntry.created`, `last_hit` (`GET /api/caches`) | CachesPanel: last used, created and last hit, exact | **Keep as is**: an operator's table read closely, not a pick list. |

## Times that are not shown

- `NotesChannelState.last_seen` (`GET /api/workspaces/{name}/notes-channel`): whether
  the agent is listening, said as a state by `listening.svelte.ts`. **Not added**: the
  state line already answers the question.
- `DiffusersStatus.finished_at` (`GET /api/system/diffusers`): the update job's end.
  **Not added**: the update result line is shown once, when it finishes.
- `LedgerAcceptedBasis.at`: when a take was accepted. **Not added**: the accepted chip
  is the fact a person needs; its time is in Plan changes.

## Strings that format a time outside markup

These build a sentence, so they keep calling `format.ts` rather than `<Time>`:
`request/standing.ts` (`… by <who> on <day>.`), `request/noteState.ts` (until Task 22
moves the note line), `refcomplete.svelte.ts` (a suggestion's detail text). Task 22's
copy rule allows them because they call `format.ts`. Of the two files the ledger names
for that rule, `request/reviewText.ts` formats a date (`toLocaleTimeString`, the review
deadline) and moves onto `format.ts`; `job/EventLog.svelte`'s `toLocaleString` calls
format event counts, not dates, so the rule should match date formatting
(`Date#toLocale*`, `Intl.DateTimeFormat`) rather than every `toLocale*`.
