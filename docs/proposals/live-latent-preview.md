# Proposal: live latent preview during generation

Status: design only, no code changes. Responds to audit finding **U1 — no live
preview during generation**: "the single biggest 'where's the ComfyUI thing'
moment" for users coming from node-based tools.

## Where things stand today

The step callback already exists and already receives the latents. In
`dw/pipeline_processors/pipeline.py`, `_with_step_callback` (around
`pipeline.py:503-532`) injects a `callback_on_step_end` into any pipeline call
whose signature accepts one:

```python
def on_step_end(pipe, step_index, timestep, callback_kwargs):
    total = getattr(pipe, "_num_timesteps", None) or num_steps
    run_context.emit("pipeline_step", step=step_index + 1, total_steps=total)
    if total is not None and step_index + 1 >= total:
        emit_phase("decoding")
    run_context.check_cancelled()
    return callback_kwargs
```

`callback_kwargs` is the dict diffusers passes to every `callback_on_step_end`
implementation, and for essentially every pipeline in this family it carries
`callback_kwargs["latents"]` — the current denoised-so-far latent tensor,
still on the accelerator, still in the pipeline's native latent space. Today
the callback reads nothing from it and returns it unchanged. So the audit's
premise is accurate: the raw material for a preview is sitting in the
callback's hand every step, and only `step_index`/`total` ever leave the
callback as `pipeline_step`, plus the coarse `phase` string
(`loading` / `cached` / `generating` / `decoding` / `saving` / `task`, defined
in `dw/events.py`).

**Event path today**: `RunContext.emit()` (`dw/events.py:34`) → the worker
subprocess's `on_event` callback (`dw/worker.py:207`, wired as
`on_event=lambda event: self.result_queue.put({"type": "progress", **event})`)
→ crosses a `multiprocessing.Queue` (pickled) into the server process →
`JobManager` appends it to `job.events` (`dw/server/jobs.py:215`,
`add_event`) → the SSE endpoint `/api/jobs/{id}/events`
(`dw/server/app.py:394-429`) streams new events to the browser as they land →
`ui/src/lib/progress.ts` (`stepProgress`) reduces the event list into a
step/total counter and a phase label → `JobPage.svelte` renders a progress
bar and phase text (`ui/src/lib/pages/JobPage.svelte:166-204`).

Two details of that path matter a lot for the design below and are easy to
miss from the UI side alone:

1. **The callback runs inside the worker subprocess**, not the server
   process. Anything produced from `latents` — decode, resize, JPEG-encode —
   either happens in the worker before the event is queued, or the raw
   tensor would have to cross the `multiprocessing.Queue` itself (pickling a
   GPU/CPU tensor through IPC), which is worse in every dimension than
   encoding first and sending bytes.
2. **`job.events` is not just a live stream — it is what gets persisted.**
   `dw/server/jobs.py` keeps the last `MAX_PERSISTED_EVENTS` events per job
   and writes them into `~/.diffusers_helper/jobs.sqlite` as the job's
   history tail. An event type that carries image bytes must not be treated
   as a normal event for persistence purposes, or the job history database
   silently grows by a JPEG per preview frame per job, forever.

## 1. Decode approach

Three ways to turn `callback_kwargs["latents"]` into something a browser can
show, in order of speed:

**A. A tiny per-family approximate decoder (TAESD-style).** Madebyollin's
`taesd` and `taesdxl` give a near-instant, low-quality RGB approximation
directly from SD/SDXL latents — a few ms on GPU, cheap enough to run every
step. A Flux-compatible tiny decoder (`taef1`) exists too. These are small
(~5MB) separate weight files, loaded once and cached like any other
component.

- *Pro*: fast enough to not matter, works well within the resolution-vs-speed
  trade-off, this is what ComfyUI and most preview implementations actually
  use.
- *Con*: it is a new dependency (or vendored weights) per pipeline **family**,
  not universal. The latent space a tiny decoder was trained against is
  architecture-specific — an SD1.5 TAESD does not decode SDXL latents
  correctly, let alone a DiT-family model's latents. Coverage has to be
  built and verified per family, and it silently produces garbage (not an
  error) on a mismatched family if someone lists the wrong one.

**B. The real VAE at reduced resolution / reduced precision.** Since the
pipeline's own VAE is already loaded for the final decode, the callback
could call `pipeline.vae.decode()` on a downscaled crop or a
`nearest`-interpolated shrink of the latent tensor, then downsize the
resulting image further before encoding.

- *Pro*: no new dependency, no new weights, no per-family coverage question —
  if the pipeline runs at all, this works, because it reuses the exact
  decoder the final image goes through. Simplest to reason about and to
  ship first.
- *Con*: slower. A full VAE decode is the same operation that already shows
  up as the `"decoding"` phase after the denoise loop finishes, and on some
  pipelines (video, high-res image) that step alone is seconds — the
  callback comment even flags it as long enough to leave "the bar sitting at
  100%" for video. Doing that every N steps multiplies that cost by
  `num_steps / N`. Rough order of magnitude on a modern CUDA GPU, a single
  SDXL-size VAE decode is ~100-300ms; on MPS it is commonly 3-6x slower; on
  CPU it can be many seconds. Even decoding a downsampled latent doesn't
  avoid the VAE's own architecture cost (it's convolutional and scales with
  input size, so a 4x-smaller latent decode is meaningfully cheaper, roughly
  proportional to pixel count — call it 4-16x faster than a full decode, but
  still not free, and still on the pipeline's own device, i.e. still
  competing with the denoise loop for the same accelerator).

**C. No preview for pipelines without a fast path; degrade to today's
counter.** Not every pipeline in this repo has a matching tiny decoder or an
easily-introspectable single-frame VAE. `workflows/*.json` and
`dw/workflows/*.json` in this repo currently exercise `StableDiffusionPipeline`,
`StableDiffusion3Pipeline`, `StableDiffusionControlNetPipeline`,
`ZImagePipeline`, `Krea2Pipeline`, and `MochiPipeline` — i.e. mostly
SD-family and DiT-family image pipelines, plus one video pipeline (Mochi).
The engine additionally documents support for exotic families the audit
calls out by name — LTX-2 (video+audio, muxed via PyAV per CLAUDE.md),
MiniMax H3 (dual video/audio latent schedules, on-demand component
residency) — for which no public TAESD-equivalent exists and whose VAE
decode is itself expensive and multi-stage (video VAEs decode a temporal
window, not a single frame; audio has no "frame" to preview at all).

- *Pro*: honest, and it's the correct fallback regardless of which of A/B is
  chosen for the covered families — a `NotImplementedError`-shaped gap is
  fine as long as the UI treats "no preview available" as a normal,
  expected state rather than an error.
- *Con*: none really — this option isn't a real alternative to A/B so much as
  the required behavior at the edges of whichever one is picked.

**Recommendation**: start with **B** (real VAE, reduced resolution) as the
first-version decode path — zero new dependencies, works for every pipeline
that has a VAE attribute and a `callback_on_step_end` parameter, and the
performance question is answered by throttling frequency (§2) rather than by
the decode implementation. Treat **A** as a fast-follow per family, gated on
someone actually measuring the B-path cost as too high for that family
in practice; it's the better long-term answer but shouldn't block v1. **C**
is not optional — it is what every pipeline falls back to when neither A nor
B applies, from day one.

## 2. Performance cost: every step vs. every N steps

Decoding every step maximizes preview smoothness but means the preview cost
is paid `num_inference_steps` times per run, on the same accelerator that is
doing the actual denoising work — there is no separate "preview GPU." That
cost is not symmetric across backends, and CLAUDE.md's device-support notes
explain why:

- **CUDA**: has spare headroom for this most of the time — TF32 matmul,
  cuDNN benchmark mode, and enough throughput that a partial VAE decode
  every few steps is close to free relative to a 20-50 step denoise loop.
- **MPS**: explicitly the platform where the codebase already treats
  "sequential" offload as too expensive and downgrades it to "model"
  offload with a warning, because per-submodule streaming on unified memory
  has no separate CPU/accelerator pools to arbitrage. A preview decode
  competing for the same unified memory and the same execution unit as the
  denoise loop is a strictly worse deal on MPS than on CUDA — no autocast,
  attention slicing already on by default (i.e. already trading speed for
  memory), no torch.compile. Every-step preview here risks visibly slowing
  down the generation it's supposed to be a nicety alongside.
- **CPU**: the existing "CPU is slow, expect a warning" path. A preview
  decode every step here is not a nicety, it's a tax nobody asked for; N
  should default much larger (or previews default off) on CPU.

**Recommendation**: decode on an interval (every N steps, N configurable,
default something like every 3-5 steps or ~every 10% of the run, whichever
is coarser) rather than every step, and pick N adaptively by device type
using the same `get_device_type()` the rest of the codebase already uses to
branch CUDA/MPS/CPU behavior (never `== "cuda"`, per CLAUDE.md). Also always
skip the last one or two steps' preview in favor of just waiting for the
real final decode — a preview that lands 200ms before the real image does
is wasted work. This is a pure frequency knob, not a decode-quality knob:
whichever of A/B from §1 is used, N-step throttling is what actually
protects wall-clock time, and it composes with either.

## 3. Transport: how the preview image reaches the browser

**Option 1: base64 inline in the existing SSE event stream.** Add a new
`preview` event type alongside `pipeline_step`/`phase`, with `data` as a
base64 JPEG, emitted through the same `RunContext.emit()` → worker queue →
`JobManager.add_event()` → SSE path everything else uses.

- *Pro*: reuses 100% of the existing plumbing — no new endpoint, no new
  polling loop in the UI, ordering falls out for free (the SSE stream is
  already ordered and resumable via `seq`/`Last-Event-ID`).
- *Con*: event size becomes proportional to preview frequency × image size,
  and — the detail that's easy to miss from the UI side — **every event
  appended via `job.add_event()` is a candidate for persistence**. `dw/server/jobs.py`
  keeps the trailing `MAX_PERSISTED_EVENTS` per job and writes that tail
  into `jobs.sqlite` as JSON. A `preview` event carrying kilobytes of base64
  would blow that budget out compared to today's few-hundred-byte JSON
  events, and would write image bytes into a SQLite history table that was
  designed for a text/number event tail. This is fixable (see recommendation)
  but is not free by construction the way it looks at first glance.

**Option 2: a separate polling endpoint** (`GET
/api/jobs/{id}/preview` returning the latest frame, or a 204 if none yet),
polled by the UI on an interval (e.g. every 500ms while `running`).

- *Pro*: completely decouples preview traffic from the event/history system —
  nothing about it touches `job.events` or `jobs.sqlite`, so no persistence
  concern at all. Simple to reason about: it's just "what's the latest
  frame," no ordering or replay semantics needed.
- *Con*: a second connection concept alongside SSE (poll timers, not just an
  `EventSource`), and it either always shows the "latest" frame (fine, since
  older previews are worthless anyway) or needs its own tiny sequence number
  if the UI wants to avoid redundant re-renders of the same frame.

**Option 3: write preview frames to a temp file, UI polls a static path.**
Worker writes `~/.diffusers_helper/previews/{job_id}.jpg` (or similar) each
N steps; the UI does `<img src="/api/jobs/{id}/preview.jpg?t={cachebust}">`
on a timer.

- *Pro*: avoids putting image bytes through IPC as event payloads at all,
  and disk I/O for a JPEG is cheap.
- *Con*: adds filesystem lifecycle management that doesn't otherwise exist
  for a job — cleanup on completion/cancellation/crash, a new place path
  traversal / naming needs `validate_path()` treatment per the security
  rules, and multi-worker or multi-job-concurrency considerations (this
  doesn't apply today since there's one worker subprocess, but it's a
  needless new constraint to bake in). It's strictly worse than Option 2 for
  no offsetting benefit here — Option 2 gets the same "just fetch the
  latest thing" simplicity via HTTP response body instead of a file, without
  a new directory to manage or secure.

**Recommendation: Option 2**, a small dedicated polling endpoint, **not**
threading preview frames through the SSE/event-log system. The event stream
is the right place for state that participates in job history (`phase`,
`pipeline_step`, `log`, `job_status` all make sense to see when you reload a
job's page later, or as the persisted tail in `jobs.sqlite`); a preview
frame does not — nobody wants a 10-year-old job history page bringing back
a base64 image of a mostly-noisy step 8/30. Concretely: the worker holds the
latest encoded preview frame (as bytes, in memory, keyed by job id — not
routed through `RunContext.emit`/`job.events` at all, so it never touches
persistence), and a new endpoint on the job manager exposes "give me the
latest frame for this job or 404/204 if there is none yet." This sidesteps
the base64-in-SQLite problem by construction rather than by having to
special-case one event type's persistence behavior. Option 1 is the one to
avoid specifically because of the JobManager/sqlite coupling discovered
above, not because SSE itself is a bad transport for images in general.

## 4. UI surface and the cancellation connection

The natural place is `JobPage.svelte`, next to the progress bar it already
renders (`ui/src/lib/pages/JobPage.svelte:166-204`, the `{#if denoise}`
block with the fill bar and step counter). A preview thumbnail — modest
size, maybe 256-384px on the long edge — sitting above or beside that bar,
updated by the Option-2 poll while `running` is true, is a small, additive
change to a component that already owns the "this job is actively
generating" rendering branch.

This is exactly where the audit's framing connects preview to cancellation
(S6, per the audit's own numbering): the Cancel button already sits right
there (`JobPage.svelte:135-142`, "stop this run at the next step — models
stay cached"). Today a user decides to cancel based on a step counter and an
ETA — abstract numbers. A live preview turns that into an informed decision:
"this composition is wrong, kill it now" instead of waiting out a 30-step
run to find out. The two features multiply each other's value more than
either does alone; this is a good argument for landing them in the same UI
change even though they're separable pieces of work. No cancellation
*semantics* need to change — `run_context.check_cancelled()` already fires
every step in `on_step_end` — this is purely about giving the user something
worth acting on earlier.

## 5. Scope for a first version

Evidence from the repo on which families are actually exercised:
`workflows/*.json` (the runnable top-level examples) and `dw/workflows/*.json`
(the packaged built-ins) together reference `StableDiffusionPipeline` (3),
`ZImagePipeline` (2), `Krea2Pipeline` (2), `StableDiffusionControlNetPipeline`
(1), `StableDiffusion3Pipeline` (1), and `MochiPipeline` (1) — nine workflow
files total, all image pipelines except Mochi, and all standard
UNet/DiT-with-a-VAE architectures with nothing exotic about their latent
space. There is no LTX-2 or MiniMax H3 example workflow in the repo despite
CLAUDE.md documenting support for them — those are the pipelines described
in §1 as needing option C (no preview) regardless of which decode approach
is chosen for the rest.

**v1 scope (recommended)**:
- Decode approach: **B** (real VAE, reduced resolution), gated to pipelines
  whose loaded `pipeline` object exposes a `.vae` with `.decode()` and where
  `callback_on_step_end` is already wired in (i.e. reuse the exact
  `"callback_on_step_end" not in parameters` check `_with_step_callback`
  already does — no new pipeline-capability detection needed).
- Frequency: every N steps, N chosen by `get_device_type()` (§2), never on
  the final 1-2 steps.
- Transport: Option 2, a polling endpoint outside the event/persistence
  path (§3).
- UI: a thumbnail in `JobPage.svelte`'s existing running-job panel, polled
  only while `running` (§4).
- Explicitly out of scope for v1: LTX-2, MiniMax H3, and any other
  video/audio pipeline where a "frame" isn't a well-defined single-step
  concept; TAESD-family fast decoders (tracked as a fast-follow per §1);
  any change to the SSE event stream or `jobs.sqlite` schema.

**Stretch goals**: TAESD/TAESDXL/TAEF1 fast decoders per family once B's
real-world cost is measured and found wanting on a specific family; a
video-pipeline preview (e.g. decode-and-show the first/most-recent frame of
a Mochi latent) once there's a concrete workflow using it in this repo to
validate against; folding preview state into `progress.ts`'s reducer if the
polling model ever needs to become event-driven for some new consumer (the
MCP server's `get_job_events` polling twin, for instance).

## 6. Effort and risk

**Effort**: moderate, not large, if scoped as above.

- Backend: extend `_with_step_callback` to optionally decode-and-cache a
  preview frame every N steps (new code path in `pipeline.py`, guarded by a
  capability check so pipelines without a VAE are unaffected); a small new
  in-memory latest-frame store keyed by job id in `dw/server/jobs.py`
  (parallel to, not part of, `job.events`); one new FastAPI endpoint in
  `dw/server/app.py`. No new dependency for v1 (Option B avoids the TAESD
  question entirely), no schema change to `jobs.sqlite`.
- Frontend: one new poll loop and a thumbnail element in `JobPage.svelte`;
  no changes needed to `progress.ts`'s event reducer since preview state
  isn't an event-stream concern.
- MCP surface: `dw_mcp` wraps the REST API per CLAUDE.md's MCP section; a
  preview endpoint would need a corresponding tool or explicit non-coverage
  note, matching how the SSE stream itself is already excluded there in
  favor of `get_job_events`.

**Risk**:
- *Correctness*: modest — worst case a preview frame is visually wrong for
  a moment (a plain "decode with the real VAE" call is intrinsically
  correct for the color space; there's no new numerical logic to get
  subtly wrong the way a mismatched TAESD variant could).
- *Performance regression*: the main real risk, specifically on MPS/CPU as
  described in §2 — needs a device-aware default and probably a
  user-visible on/off toggle (or an automatic "generation is slow, previews
  auto-disabled" heuristic) so a slow backend doesn't silently get slower
  because of an opt-out-only feature.
- *Concurrency*: today one worker subprocess runs one job at a time, so a
  single "latest frame per job id" store is sufficient; if the worker model
  ever becomes multi-job, the preview store needs the same job-id keying
  discipline the rest of `JobManager` already uses — not a new problem,
  just something to keep consistent.
- *Scope creep*: the biggest practical risk is reaching for TAESD coverage
  or video-pipeline previews in v1 instead of treating them as the stretch
  goals in §5 — B-and-throttle-and-poll is enough to deliver the "oh, it's
  actually painting" moment the audit is asking for, without turning this
  into a per-pipeline-family research project before anything ships.
