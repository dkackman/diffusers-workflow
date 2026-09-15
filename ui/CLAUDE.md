# ui

Guidance for the web UI single-page app.

`npm run build` outputs `ui/dist`, which the server serves — the SPA is not
served from source, so a UI change is invisible until it is rebuilt.

Front-end checks, run from `ui/`: `npm run check`, `npm run lint`, `npm test`,
and `npx playwright test` (e2e — it starts its own server, so do not have
`dw.serve` running on the same port).

Every request is scoped to the selected workspace in one place: `scoped()` in
`lib/api.ts` reads `workspace.current` and appends `?workspace=` to request
paths, download URLs and the `/outputs` URLs `outputUrl` builds; a `url` the
server returns (an upload's preview, an asset entry) is used verbatim and the
server is responsible for scoping it, so no page threads a workspace through
its calls. The exception is a file that belongs to a *job*: `outputUrl` and
`fetchOutputText` take an explicit workspace, which wins over the picker, so a
job's files load from where they were written.
`lib/workspace.svelte.ts` holds the selection (restored from localStorage in
`main.ts` before the first request); a page refetches on a switch by reading
`workspace.current` inside its load effect.

## Design system

The look is "contact sheet", and it rests on two rules that `src/app.css`
encodes. Follow them when adding UI.

**Colour means machine state.** The greys are deliberately achromatic
(equal-RGB, not blue-tinted slate) because this app's output is pictures and
any chroma in the surrounding surfaces biases how a generated colour reads.
So the chrome carries no accent: interactive elements are ink (`--accent` is
the *interactive ink*, a near-black in light and a near-white in dark; the
primary button is a solid ink fill, links are ink with an underline).
`--live`, a darkroom safelight amber, is the one signal colour and marks only
what is running, where keyboard focus is, and when VRAM is under pressure -
never decoration. `--good` / `--bad` stay state colours too. Every token pair
passes WCAG AA in both themes; keep it that way.

**Mono is what the engine reads.** `--font-mono` for anything the engine
resolves literally - a workflow or prompt name, a `variable:` / `asset:` /
`prompt:` reference, a path, a seed, a run id, a measurement (add `.num` for
tabular figures). `--font-sans` for anything written for a person -
descriptions, hints, control labels. Headings are mono because they name
things the engine resolves, and they are sentence case: no ALL-CAPS eyebrow
labels and no tracked-out letter-spacing on small labels.

Type scale is `--t-xs` .. `--t-xl` off a 15px base; radius says what a thing
is (`--radius-frame` a picture, `--radius-2` a surface, 999px a chip) rather
than being one global value. One filled button per view: the thing you came
to do. `.quiet` is outlined, `.bare` is text-only.

**Show the proof.** A list of things that produce images shows the images.
The workflows catalog and the workflow page both read `api.gallery()` and
match a workflow's name to an output entry's `folder` - a run writes to
`<outputs>/<identity>/<run id>/` and the gallery reports that identity with
the run id already stripped, so the match is direct equality. Proofs load
separately from the listing so the catalog never waits on them, and a
workflow that has never run gets no frame at all rather than a grey
placeholder (a fresh workspace would otherwise be a wall of empty plates).
Within a folder, workflows that have produced something sort first.

Every picture in the app sits in the global `.frame` (app.css): the media
fills it edge to edge, with no inner padding and no rounding of its own,
which is what makes it read as a proof on a sheet rather than as another
rounded card. Gallery tiles, a workflow's recent outputs and a job's results
all use it; the catalog card's `.cardframe` is the one variant, because it
is a banner inside a card rather than a free-standing frame. Set the size on
the frame - the media inside always fills it.

Where `--live` appears, and nowhere else: the header's running link and VRAM
pressure, a running job's row and status chip, the job page's progress bar
and current step, the flow view's active step, a model download in flight,
and the focus ring. Resting meters (free disk, VRAM under pressure's
threshold) stay `--muted`; selection is the user's state, not the machine's,
so it reads as a heavier ink edge instead.

## Assets

`AssetsPage.svelte` is the input side of the gallery (#165), and deliberately
the same UX: folder groups, a contact-sheet grid, a detail popout. It reads
`GET /api/assets`, which spans the workspace's own library, the shared
`common` one and any `--examples-dir` library, each entry already tagged with
its `origin`, plus the search path itself (`libraries`) and what each root
holds under a name a nearer one has taken (`shadowed`) - so the page never
builds a path, only shows the `asset:` reference a workflow argument carries.
Which library an entry came from is the section it sits in, because "why can
I not delete this" has to be answerable at a glance: an `examples` asset is
read-only and the server answers 403, so the page offers no delete for one at
all.

"The same UX" is literal, and three things were missing from it until they
were fixed: the detail popout is `position: sticky; bottom: 1rem` as the
gallery's is, so it rides the viewport rather than sitting at the end of the
document where a click above the fold scrolls it out of sight; the bulk
actions are the gallery's - a checkbox per tile (shift-click spans a range),
`Select all matching`, and a sticky bar with Download .zip and Delete, the
delete sequential with whatever failed staying ticked; and Escape clears the
selection before it closes the detail. Bulk download is
`POST /api/assets/archive`, the gallery archive's counterpart - the browser
cannot zip on its own.

The selection itself is not this page's: `picks.svelte.ts` holds it (a
`Picks` over a getter for the grid's current order, so a filter changing
under the selection is seen rather than snapshotted), `BulkBar.svelte` is the
sticky bar and the select-all, and the two rules that have to reach a tile
the page lays out - `.cellwrap .pick`, `.cellwrap.picked .cell` - are in
`app.css`. The gallery runs the same three. They were one page's code copied
into the other first, and the copies had already drifted: the assets page's
Escape guard knew that `ConfirmDialog` renders `alertdialog` and the
gallery's did not, so Escape in the gallery's delete confirm closed the
detail behind it. `dialogOpen()` is that check, once. A bulk action must
never touch what the user cannot see, so `Picks.size` counts only the
visible selection - the same set `names` hands to an action - and a ticked
name the filter is hiding is inert until the filter brings it back.

The search path is the page's top level and folders sit inside it: one
section per library, in the order the server resolves them, because which
library a name lives in is what decides whether it can be deleted, what a
delete costs, and what it hides. It used to be one mtime-sorted grid with
an `origin` badge per tile and a library `select` over it - folders then
cut across libraries, and the select could be left pointing at a library
that unmounted on the next workspace. A section header carries the label,
the count, the root it reads and a collapse chevron (persisted under
`collapsed-asset-libraries`; a filter opens everything, as `FolderGroups`
does), and an empty library still shows its header, so an empty workspace
says where an upload would land.

Upload is the section's own button rather than a destination pick, because
which library a file lands in is the one thing about an upload that cannot
be changed afterwards - `Upload` on the workspace section (the page's one
filled button), a `.quiet` `Upload to shared` on the shared one, and a
muted `read-only` where the server would answer 403. For the same reason a
tile from a read-only library carries no checkbox at all: nothing bulk can
do to it. A shared delete says what it costs, singly and in the bulk
confirm, since it goes for every workspace under the root.

`shadowed` is rendered rather than only described. An entry a nearer
library hides gets a dimmed, inert tile under its own library's
`shadowed/` heading - no `url` is served for one, so it is a label rather
than a picture - because "I uploaded it and `asset:` still loads the old
one" is otherwise unanswerable from the page.

See docs/SERVER.md.
