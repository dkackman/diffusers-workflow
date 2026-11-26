# Default preferences: the shipped text-to-image ladder and its bench (#1625)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out session,
2026-11-26). Plan v2 on #1625 is the full design. This file records what
shipped, what was cut, and why. It belongs to the ladder set: the
`preference:` reference and its resolver are #1626, and the UI and plugin
routing are #1627.

**Result:** `dw/preference_defaults.json` ships one key, `text_to_image`, with
a variable contract and pin tests. `templates/text-to-image` (SD 1.5) gains
`width`/`height`. Every ladder target now has a measured `cost` on each device
it runs on: cuda for all of them, and mps for all of them except the cuda-only
H3-Image.

## The problem

`templates/text-to-image` (SD 1.5) is the smoke test and the first-run
target, but it isn't the model to reach for to make a picture. Twelve
`workflows/models/*` entries declare `configures: templates/text-to-image`, and
nothing ranked them. Several had no measured `cost`, and none had an mps
figure. Those entries are copies of a pipeline, so they don't share a variable
set, and swapping one for another could break a caller.

## Verdict: build smaller

**The plan corrected the issue:**
- **The file's path.** It is `dw/preference_defaults.json`, not under
  `workflows/`. `dw/library.py` lists every `*.json` in a library as a
  workflow, and `workflows/` isn't packaged. `dw/cache_blocks.json` is the
  precedent for data the engine reads.
- **The contract.** `num_images_per_prompt` is optional, because H3-Image
  doesn't take it. The draft target had no `width`/`height`, so stage 1 added
  them.
- **Klein is out of `best`.** Its only ladder route was
  `templates/flux3/upsample-then-generate`, and that takes `request`, not
  `prompt`.
- **Only one key ships.** The plan cut the other four keys. `image_to_image`
  would have had a single rung. `image_edit` has no shared contract
  (`instruction`+`image` against `prompt`+`input_image`). The video keys
  don't fit the H3 Ramp's two cuda rungs, and `templates/minimax/image-to-video`
  has no `tier` preset.

**Rulings.** Don approved plan v2 on 2026-11-25 with the delegate's
recommendations (Q1–Q6), and only stage 1 was cleared to build. Each bench
stage then came back as its own spend ask:
- #1711: Don, 2026-11-26, "approve / gated repo is accepted". Klein was
  included.
- #1712: the delegate ruled for Don on 2026-11-26. It allowed a 30 min cap per
  run, Qwen last, and no downloads.

## What shipped

**Stage 1, #1710.** Commit `6de7ccf7`, merged to `develop` at `2bdad4ca`.
- `dw/preference_defaults.json`, `version` 1, with `contracts`,
  `preferences`, `declines` and `left_out` for `text_to_image`:
  - `draft`: `templates/text-to-image`;
  - `standard`: `models/z-image`, `models/z-image-sdnq`;
  - `best`: `models/minimax-h3-image`, `models/qwen-image-2.1`.
  - `left_out` lists the other eight configuring entries, each with a reason.
- `templates/text-to-image` declares `width`/`height` (default 512, so a
  default render is unchanged), passes them to the pipeline, and lists them in
  `cost_drivers`.
- `tests/test_preference_defaults.py` pins the file:
  - every target resolves through `dw/library.py`;
  - each required variable is declared;
  - each optional variable a target lacks is declined with a reason, and no
    decline is stale;
  - each configuring entry is either in the ladder or in `left_out`, not both;
  - key and tier names are valid;
  - a fixture `{workflow, arguments}` target's argument names are declared
    variables;
  - the file is in the built wheel.
- Docs: a *Default preferences* section in `docs/WORKFLOW_GUIDE.md` and a
  row in `docs/ARCHITECTURE.md`.
- Deviations from the plan:
  - `EXPECTED_SHAPES` needed no change.
  - No canvas-multiple refusal was added for SD 1.5, because
    `dw/pipeline_limits.py` covers only H3. C-F1355 records what happens.

**Stage 2, #1711 (cuda, lem, RTX 3090).** Commits `69b9efa3` + `26dc269c`,
merged to `develop` at `5cb3a9bb`. Each figure is one run at defaults, and
nothing was downloaded:

| entry | job | wall | `cost.minutes` |
|---|---|---|---|
| `models/z-image-sdnq` | `009e1762b903` | 143.3 s | 2.4 |
| `models/flux3-klein` | `ab64739913e9` | 20.1 s | 0.34 |
| `models/krea2` | `832db2e886f8` | 112.7 s | 1.9 |

`models/z-image-sdnq` also gained `cost_drivers`.

**Stage 3, #1712 (mps, mini-ai, Apple M5 Pro).** Commit `99c445d8`, merged to
`develop` at `bf278ae0`. It used 18.6 of the 90 approved mps minutes, with no
downloads:

| entry | job | wall | `cost.minutes` |
|---|---|---|---|
| `models/z-image` | `48b315ca0135` | 141.6 s | 2.4 |
| `models/z-image-sdnq` | `4ab7faa08ae0` | 55.2 s | 0.9 |
| `models/qwen-image-2.1` | `ded23c7bc38e` | 920.6 s | 15.3 |

- The mps rows use `vram_gb` 54, the card's `capacity_gb`, not 64. The
  resolver's fit holds a cost row's VRAM to the host's smallest card. A
  64 GB row would make it skip qwen and z-image-sdnq on the host they were
  measured on.
- `templates/ltx2/text-to-video`'s older mps row (64) has the same latent
  problem, and was left alone.
- `tests/test_preferences.py::test_rung_skips_cuda_only_target_on_mps`
  changed its fixture card from 18 to 54 GB. What the test checks is
  unchanged.

## Cut, and what brings each back

- **`image_to_image`:** a second entry that configures
  `templates/image-to-image`.
- **`image_edit`:** one variable contract across the edit templates.
- **`image_to_video` / `text_to_video`:** a measured rung mapping and a `tier`
  preset on the i2v template, or the H3 Ramp rungs re-keyed.
- **Klein in `best`** (Q3's revisit): the mps bench gave `best` an mps figure
  (Qwen, 15.3 min), so the "no `best` target on mps" input never came up.
  Klein, at 0.34 min on cuda, could join the ladder through a route that takes
  `prompt`. Whether 15 min is acceptable for `best` on mps is Don's call if he
  reopens it.

## Notes for #1626 and later readers

- The issue said "a target with no measured cost on a device is not chosen
  there." The resolver #1626 shipped does the opposite: a target with neither
  a `vram_estimate` nor a `cost` row for the device is *not* skipped
  (WORKFLOW_GUIDE *Model preferences*). The benches make the difference moot
  for this ladder, because every target that runs on a device has a row for
  it.
- A new `configures: templates/text-to-image` entry fails the pins until it is
  placed in the ladder or in `left_out` with a reason. That friction is
  intended.

## Cost and bounces

None of the stage comments gave a `usage:` figure, so this record gives no
dollar cost. GPU time: about 4.6 min on cuda, and 18.6 min on mps plus the SD
1.5 verification runs.

| stage | bounces | notes |
|---|---|---|
| #1710 | 0 | verified on mps the first time |
| #1711 | 0 | verified on lem the first time |
| #1712 | 0 | two build sessions were blocked by a stuck job on the mps card (`872ea59cbdcb`, #1744) until Don restarted `dw-serve`, then verified the first time |
