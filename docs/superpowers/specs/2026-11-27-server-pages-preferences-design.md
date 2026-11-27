# Server pages and Preferences: design

Date: 2026-11-27. Status: draft for Don's approval.

## Why

Server → Preferences (#1762 stage C, `ui/src/lib/pages/PreferencesPage.svelte`,
`ui/src/lib/ladder/LadderCard.svelte`) works but does not explain itself. A new user
does not know what a preference is, and the page's answer to "what runs here" sits
under an always-open editor. Don's first look: "not immediately obvious how it works";
rework both the explanation and the editing.

The three Server pages answer mixed questions. Status (`StatusPage.svelte` = `ServerPage`
+ the all-workspace `JobsPage`) holds what the server is doing, how to reach it and
what it runs on, in one scroll. Models holds the hub cache and also the dw/diffusers
update controls. Preferences and Models answer two halves of one question ("which
model answers a request, and is it on disk?") and do not link.

## Goals

1. A first-time reader understands Preferences without docs: what a preference is,
   what each tier will run on this machine, and what an edit will change.
2. The page tells the truth admission tells: the same skips and notes a real job's
   resolution gets.
3. Editing shows its consequence before Save (a live dry run).
4. Server pages are organised by the question a visitor came with.
5. The Schema link leaves the sidebar for the top bar, beside OpenAPI.

## Non-goals

- No change to how resolution works (`dw/preferences.py` `resolve`): fallback stays
  host-fit at admission, within one tier, never across tiers, never on runtime failure.
- No dry run over MCP; agents have `validate_workflow` and `get_preferences`.
- Workspace and Shared pages are untouched.
- No "Download now" from a preference (#1781).
- No enforcement of `contracts` at PUT (today they are tested on the shipped file only).

## Anti-goals

- The page must never claim more than the resolver does: no "if it fails, try the
  next" wording; a model after the pick is "fallback, not checked here".
- Editing must stay fast: no hub network call per keystroke or drag.

## Approaches

**Preferences layout** (mockups in `2026-11-27-mockups/`, three options):
A answer-first read view with Edit; B per-tier fallback chains edited in place;
C an interactive "what would happen" strip above an editor. **Picked: A as the
resting view, B as the editor Edit opens.** C's teaching job is done by A's per-tier
answer plus the chain reading left to right.

**Page organisation** (three options): A tidy the three pages in place; B one page per
question (Activity, Models with Preferences folded in, Connect, System); C B's
grouping as tabs under one Server sidebar row. **Picked: B.** A leaves Status
answering three questions; C costs a click to reach Models for no gain.

**Dry-run route:** a separate `POST /api/preferences/{key}/check`, or `PUT ...?dry_run=true`.
**Picked: the separate route.** It writes nothing, so it carries no If-Match, and the
PUT keeps one meaning.

## Design

### 1. Top bar and sidebar

- Top-right nav (`ui/src/App.svelte`, beside the `/docs` OpenAPI link): a Schema icon
  link to `#/server/schema` (internal route, same tab; title "workflow schema").
  `#/server/schema` stays a route; it leaves `SERVER_ITEMS`.
- Sidebar Server group (`ui/src/lib/navItems.ts`): **Activity, Models, Connect, System**.

### 2. Routes (`ui/src/lib/routes.ts`)

| Route | Page |
|---|---|
| `#/server/activity` | Activity |
| `#/server/models` → `#/server/models/preferences` | Models, Preferences tab (default) |
| `#/server/models/downloaded` | Models, Downloaded tab |
| `#/server/connect` | Connect |
| `#/server/system` | System |
| `#/server/schema` | Schema (top bar only) |

Old links keep working: `#/server/status` → activity, `#/server/preferences` →
models/preferences, and the legacy aliases that resolve to `server/status` today
(`jobs`, `server`) go to activity. `Breadcrumb`'s Server crumb points at activity;
`JobError`'s "Open Models" goes to models/downloaded. (The top-bar status button
and `StatusPopover` do not link to the status page today; nothing to change there.)

Router work this needs: `parseView` drops anything after a server section and the
server `RouteView` has no `rest`, so the Models tab needs a sub-path on server views;
and `server/status` / `server/preferences` must reach `legacyRedirect`, which a known
server section never does today.

### 3. Page contents (move, don't rewrite)

The panels exist in `ServerPage.svelte`, `ModelsPage.svelte`, `CachesPanel.svelte`,
`ServerHealthList.svelte`, `JobsPage scope="all"`. Split them:

- **Activity** — "what is it doing?": cards and health (`ServerHealthList`), the queue
  across workspaces (`JobsPage scope="all"`), downloads in flight (from Models'
  download list, active ones only, with cancel), a dw/diffusers update in flight.
  Header: hostname · version · device.
- **Models** — "what runs, and is it here?": tabs Preferences | Downloaded (count, size).
  Downloaded is today's Models page minus the update controls: download form, disk
  meter, cache list.
- **Connect** — "how do I reach it?": Binding, Addresses, Authentication, MCP, Connect
  from another machine.
- **System** — "what is it running on?": Server (hostname, version, device), dw and
  diffusers versions with their update controls (moved from Models), Runtime,
  Directories, Storage, Caches.

`StatusPage.svelte` is removed; `ServerPage.svelte` is split into the Connect and
System pages (or shared panel components they compose).

### 4. Preferences: read view (A)

Intro, plain words, one short paragraph: when a workflow or agent asks for a kind of
output without naming a model, dw picks one from that preference's list. Each list
has three tiers (Draft fastest, Standard, Best highest quality); dw runs the first
model in the tier that this machine can run. Then the default tier sentence.

One card per key:

- **Header:** a readable name from the key (`text_to_image` → "Text to image") with
  the key as a code tag; "as shipped" or "changed on this server"; "used by N
  workflows" (expands to the list, each linking to the workflow); Edit; for a
  changed key, "Reset to shipped" with a confirm naming what it resets to.
- **One row per tier**, the default tier highlighted with a "default" tag and the
  others plain. The default tier is server-wide, so it stays one control for the
  page (builder-check finding): a segmented control at the top, "Requests that
  don't name a tier run at: Draft | Standard | Best", same `setDefaultTier` call
  and 412 handling. Each card highlights that tier.
  - status dot + the picked model + state: **ready**; **downloads N GB on first run**
    (or "downloads its weights on first run" without a size), linking to Models →
    Downloaded; or **nothing fits this machine** with the reason (a refusal, `--bad`).
    No Download now button this round: the existing download route pulls whole repo
    snapshots, not a workflow's exact plan (Don's ruling; exact prefetch is #1781).
  - the models passed over, struck, each with its reason ("needs CUDA", "gated, no
    access to …", "needs 32 GB VRAM, smallest card has 24 GB").
  - the models after the pick, muted: "then z-image-sdnq (fallback, not checked)".
- Skip reasons are rewritten for people from the resolver's strings
  (`cuda-only` → "needs CUDA"); unknown strings pass through as-is.

### 5. Preferences: chain editor (B)

Edit turns that card's tier rows into chains (one card in edit at a time; leaving
the page with an unsaved draft uses the router's leave guard):

- Each tier: chips left to right, "try in order →". Drag to reorder (keyboard:
  focus a chip, Alt+←/→ moves it; the existing up/down buttons are the fallback).
  Each chip has a remove ×. `+ add` at the end opens a popover: workflow picker
  (`Suggest` over the catalog) and an optional "only on" device choice
  (cuda/mps/cpu, labelled "Only on these devices — leave empty for any").
- Target arguments are shown on the chip ("with steps=8"); editing them is out of
  scope (as today).
- **Live dry run:** 300 ms after the last change, `POST /api/preferences/{key}/check`
  with the draft, `full=false`. Each chip shows the answer: runs here / skipped with
  reason / fallback, not checked / error (stale workflow, refused arguments) on the
  chip itself. Save is disabled while the draft has an error.
- A footer line: "Unsaved changes · jobs already queued keep the model they were
  given", Discard and Save. Save uses the existing PUT with If-Match; its response
  (full resolutions) replaces the dry-run answer. 422 and 412 handling stay as today
  (`StaleNotice`, `ErrorLine`).
- An emptied tier reads "No models: requests at this tier are refused" (a ladder
  that leaves a tier out is legal; an empty list is not sent, as `sendable` does today).

### 6. Server changes

- **Page-load truth, opt-in:** the GET's `resolve=true` stays Hub-free (deliberate:
  the run form, `WorkflowPage.svelte`, and MCP `get_preferences` call it). A new
  query flag, `resolve=full`-style (name the builder's), resolves with the admission
  host (`preferences.downloads_gate_check()` as `gate_check`, as
  `dw/server/admission.py:230` does) and adds `used_by` and `contract`. Only the
  Preferences page sends it; MCP and the run form answers are unchanged. The UI's
  `Rung` (`ui/src/lib/preferences.ts`) gains `notes`. *(Builder's-check call.)*
- **Check route:** `POST /api/preferences/{key}/check`, body
  `{"ladder": {...}, "full": bool}`. Never writes. Answers
  `{"tiers": {tier: {"target", "skipped", "notes", "after", "error",
  "target_errors": {name: message}}}}`, where `after` lists the targets after the
  pick (not checked) and `target_errors` carries `check_ladder`'s refusals per
  target instead of one message for the ladder. `full=false` runs the gate check
  with `lookup_sizes=False` and no hub access probe (cache presence only, so a
  missing model notes "not downloaded"); `full=true` is the GET's full check.
  Gated access in the editor: the UI keeps the last full check's per-target access
  verdicts and sends them with the draft (`known_blocked: [names]`), so a target the
  full check skipped as gated stays skipped in the dry run; a target added in this
  edit and not yet fully checked shows "access checked on save". *(My call.)*
- **After Save or Reset**, the UI refetches the full GET (the PUT and DELETE keep
  their Hub-free resolutions, so a save stays fast). *(My call; the earlier draft's
  claim that the PUT returns full resolutions was wrong.)* A key
  not in the store is allowed (a new key being drafted). Path and key validated as
  the PUT does. Response model in `api_models_library.py`, OpenAPI pinned,
  `npm run gen:api`.
- **Used by:** the full read adds `used_by: [workflow names]` per key: the workflows
  the page's workspace can list (its own, shared, builtin) that reach
  `preference:<key>` directly, through a variable default, or transitively through a
  composed workflow (so a workflow calling `templates/preferred/text-to-image`
  counts). Direct-only would show one workflow and read as wrong. *(My call.)*
- **Contract:** the read adds `contract` per key when the shipped file has one
  (`required`, `optional`). The check route warns (a note on the chip, never a
  refusal) when a target does not declare a required variable.

### 7. New preference

"New preference" opens a short form: name (validated with `keyProblem` as today),
one line on what makes it useful ("a workflow uses it by naming
`preference:<name>` as a step's workflow path"), and the first model for the default
tier. It creates the card in edit mode; nothing is saved until Save.

## Constraints

- UI rules in `ui/CLAUDE.md`: `resource()` + `<LoadState>`, `action()` for every
  mutation, destructive confirm via `destructive.ts`, `describeError` copy, colour
  only through the listed tokens, `PageHeader`, copy per `GLOSSARY.md`, layering
  per `LAYER_ALLOWS`; server types generated, never hand-written.
- Server: `dw/security.py` validators on every input; `get_device_type()` for
  backend compares.
- Tests: unit tests for routes, nav, the new pages and the check route; e2e specs
  that visit `#/server/status` updated; arch and freshness ratchets held (raising a
  ratchet needs Don).

## Left to the builder

- Component split of `ServerPage.svelte` into Connect/System panels.
- Drag-and-drop implementation (native HTML drag or a small helper; no new dependency
  without asking).
- Exact copy, within the meanings above.
- Build order (one branch): schema link + reorganisation; server changes; read view;
  chain editor + dry run.
