# MiniMax-H3-Image as the stills family for H3 video (#1317)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out session,
2026-11-18). Don filed the idea from the diffusers 0.43.0 release notes:
MiniMax-H3-Image shares MiniMax-H3's Qwen3-VL text encoder, so a cast still or
keyframe made with it might hold a face into an H3 clip better than a FLUX 3
Klein still does. The plan (v5) is the plan comment on #1317, and it holds the
full sweep. This file records what was built and measured.

## Verdict: build, in three stages

A bench probe came first (stage A). Then the catalog entries (stage B), and
then the skill and routing (stage C). P-H3-Image and the H3-Image prompt
enhancer (PE) were dropped at plan v1 and never built.

Evidence for the routing (stage A, n=1, relative A/B into H3 i2v): an H3-Image
still scored SFace 0.423 / 0.316 against Klein's 0.243 / 0.292. Its face was
found in 16/16 frames against Klein's 6/16. By eye, the H3-Image edit kept the
grade and costumes and Klein lost both. Record: `bench/records/h3-image.md`.

## Plan history

- **v1 → v2 → v3 (ruled 2026-11-17 13:44).**
  - Q1 [default]: `host_ram_gb: 64` (warm). Each template's text also says a
    cold first run needs about 120 GB (117 GB measured peak).
  - Q2 [scope]: no chained still → H3 template. The skill teaches two jobs
    joined with `output:`.
  - Q3 [default]: only stills and keyframes that feed H3 video route to
    H3-Image, and only on a qualifying CUDA host. Everything else stays on Klein.
  - Q4 [none]: folder `workflows/templates/minimax-h3-image/`, two templates,
    and `min_generation_area` not exposed.
  - Q5 [spend]: about 27 min of lem GPU, with no download.
  - Carried from v1: the encoder comes from `MiniMaxAI/MiniMax-H3`, there is no
    P-H3-Image, and there is no PE or `trust_remote_code`.
- **v3 → v4 (re-plan after stage B's first bounce; ruled 2026-11-17 23:40).**
  - Q6 [scope], yes: `min_length` now counts a list's entries as well as a
    string's characters, and a null counts as empty. This lets the engine
    refuse `images: []`. The change is generic.
  - Q7 [loosen], yes: cost is priced on lem's slower card. The rule is that a
    cached run comes in no more than 10% over `cost.minutes`.
  - Correction: the edit's `image-edit` shape and `needs-input-media` trait
    are derived from `images`, so they aren't declared.
- **v4 → v5.** Don's record of the card gap's cause was folded in. Scope was
  unchanged.
- **Rulings during the build:**
  - 2026-11-16: M-F395 step 3 (a refusal of `trust_remote_code` on an
    untrusted server) is out of stage A's scope.
  - 2026-11-18: the lead runs stage C's cold drill, with nothing else on lem.
  - 2026-11-18 (dkackman/harnest#242): the `RECIPES_24GB.md` section also says
    that a change to cuda:0's PCIe link means re-measuring.

## What was built

### Stage A (#1345): the bench probe

- `bench/workflows/h3-image-*.json` (t2i, a 3-reference edit, and the chain
  into H3 i2v), saved in workspace `bench`, and the record
  `bench/records/h3-image.md`.
- Findings:
  - int8 runs on a 24 GB 3090 with offload, at about 3.7–5.5 GiB reserved.
  - The host-RAM cost is the real one: 64 GB warm and 117 GB peak during the
    cold quantize. A cold first run takes about 38 min.
  - 1024 is no faster than 2048: the offloaded transformer is bound by weight
    transfer.
  - None of dw's H3 video inserts engage.
- Merges: `0365ece6`, `4283fe54`, `44370b83`.

### Stage B (#1424): the catalog entries

- `workflows/templates/minimax-h3-image/text-to-image.json` and
  `multi-reference.json`:
  - variables: `prompt`, `width`/`height` (modulus 32, default 2048), size
    presets, `seed` and `weights_dtype` (`{int8}`). The edit also takes
    `images` (one-based `<Picture N>`, `min_length: 1`);
  - `host_ram_gb: 64`;
  - no steps, guidance, negative prompt or `min_generation_area` variable;
  - the description states the ~120 GB cold first run, that the family is
    CUDA-only, and that there is no PE.
- `workflows/models/minimax-h3-image.json`, the family README, the audit
  `docs/proposals/audits/2026-11-17-minimax-h3-image-audit.md`, a
  MiniMax-H3-Image section in `docs/RECIPES_24GB.md`, and the cold-quantize
  line in `docs/QUANTIZATION.md`.
- Q6: list-length `min_length` in `dw/variable_constraints.py`, with its
  catalog phrase in `dw/server/catalog_shape.py` and the schema and guide text
  to match. Architecture review passed.
- Tests: `tests/test_minimax_h3_image_catalog.py`, which also pins that
  H3-Image's pipeline and transformer classes aren't H3 video's. Also the
  EXPECTED_SHAPES entries, the COMPACT_BUDGET bump and the `min_length` list
  cases.
- **Prices (Q7)**, from a just-loaded run on cuda:0 less its load: the t2i and
  the `models/` entry are **2.3 min**, the edit is **2.7 min**. A fully warm run
  comes in under these. The t2i's 2.3 is built from the phases on record, not
  timed directly on cuda:0 after a load.
- **The card gap:** cuda:0 is on a PCIe x4 link and cuda:1 on x16. This
  offload is bus-bound, so cuda:0 runs it 2–3x slower per pass (about 15 s
  against 4.5–8.6 s). Any offloaded workload on cuda:0 is slowed the same way.
  If the link changes, re-measure and re-price.
- Merges: `4bb60015`, `057b4347`, `f2cd80bb`.

### Stage C (#1425): the skill and routing

- `plugins/dw/skills/minimax-h3-image/SKILL.md`, which covers:
  - the host check (CUDA, ≥ ~70 GB host RAM) and falling back to Klein below
    it;
  - the hard rules, and the cost priced on the slower card;
  - *Quote, then wait*: the estimate and host floor go to the user before any
    run;
  - the two-job route into `templates/minimax/image-to-video` (or
    `first-and-last-frame`) with `image: output:<still>`;
  - the failure modes.
- `cinematography/references/compile-minimax-h3-image.md`, plus routing text in
  `minimax-h3`, `storyboard`, `cinematography`, `long-form` and `flux3-klein`.
  The lists in the plugin README, `CLAUDE.md`, `docs/MCP.md` and `plugin.json`
  were updated.
- The workflows guide gained *A portrait still that feeds an H3 clip:
  MiniMax-H3-Image*, and `README.md`'s Images row was updated.
- Cold drill on lem, with the plugin snapshotted at 940c51d0 (`claude -p
  --model sonnet`):
  - the portrait still went to H3-Image and the 5 s clip to H3 i2v;
  - the storyboard control stayed on Klein, and so did the plain-still
    control;
  - nothing was downloaded. The runner and transcripts are on Don's Mac at
    `~/testing/drill-1425-runner/`.
- Merges: `d60c9d19`, `90c60ded`, `259072ab`.

## Bounces per stage

| Stage | Bounces | What bounced |
| --- | --- | --- |
| A (#1345) | 2 | Workflows not reachable over MCP. Then the chain case and M-F395 step 3, which Don ruled out of scope. |
| B (#1424) | 2 | v3 couldn't refuse `images: []` or meet one cost on both cards, which led to the v4 re-plan. Then the edit's 2.3 price was beaten by a just-loaded run, so both templates were re-priced. |
| C (#1425) | 2 | The drill's session ran without quoting cost or host floor, and fed the clip an `asset:` instead of `output:`. Then the guide's portrait route was not findable. |

Before its drill, stage C also had two "incomplete" verifies, which waited on
the drill evidence and weren't failures.

Cost per stage isn't recorded here: the stage comments don't give `usage:`
figures. The drill sessions cost $0.45, $0.15 and $0.14.

## Deferred, and why

- **P-H3-Image and the PE** (v1 Q3, Q6): P-H3-Image was out of scope, and the PE
  needs `trust_remote_code` and a trust-gate decision.
- **bf16 or two-card runs:** #1319 (H3 tensor parallelism).
- **Per-card cost:** a separate engine issue, if Don wants one. Today the
  catalog over-quotes on cuda:1.
- **An admission rule for host RAM shared across workers:**
  `host_memory_projection.py` warns but doesn't refuse. This is the follow-up
  if two host-heavy jobs on lem collide.
- **An upper bound on `images`, and a `<Picture N>` numbering check:** not in
  the engine.
- **A chained still → H3 template** (Q2), and mps or cpu support.
- **Face-hold evidence beyond n=1:** if a field report shows a regression, C's
  routing reverts, and B's entries stay.
