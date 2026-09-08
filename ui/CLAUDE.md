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

See docs/SERVER.md.
