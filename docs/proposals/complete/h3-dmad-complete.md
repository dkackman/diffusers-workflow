# H3 DMAD step-distillation LoRA probe for dialogue (#1579)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out session,
2026-11-23). Don filed the idea on 2026-11-22 from the WanGP review
(deepbeepmeep/WanGP at f4cfc12, concepts only). The idea was that DMAD's 4- and
8-step H3 LoRAs (`ZhengmingYu/DMAD`, distilled with the audio branch on its own
schedule) might be a fast H3 path that keeps dialogue voices clean. Today the
minimax-h3 skill drops lightx2v turbo on every `<d>` shot, because turbo
digitizes voices. Plan v1 is the plan comment on #1579. Don approved it at its
defaults on 2026-11-22, and it was built as one bench-only stage, #1584. The
full record is `bench/records/h3-dmad.md`.

## Verdict: build smaller (bench only), result: don't adopt

- The A0 load check failed, which is the plan's stop rule. dw loaded the 4-step
  file (pinned `b08c2bf3be8e67039c363781e6eaacc9713340cd`), diffusers mapped
  **0 of its 624 tensors**, and `set_adapters` then failed with `Adapter name(s)
  {'dmad'} not in the list of present adapters: set()`.
- The cause is the file's key format. Keys carry no `transformer.` prefix, and
  one file mixes `.lora.down/.lora.up` (attention) with `lora_A/lora_B`
  (feed-forward). The header metadata says the LoRA targets the T2VA
  `transformer` (`MiniMaxH3Transformer3DModel`), so the partition was not the
  problem.
- The scheduler flag took: `Loading scheduler: MiniMaxH3Scheduler
  (from_config_args: stochastic_sampling)` and the same for `audio_scheduler`
  appear in `get_job_events`.
- A1 (base, dmad4, dmad8, turbo control), A2 (seed spread) and A3 (non-dialogue
  pair) did not run. Every arm's call is **drop**. No arm is a candidate, so
  neither adoption gate (Don's ear, a seeded re-noise path) was reached.
- Nothing ships: no engine, template, catalog, schema, MCP or plugin change.
  The dialogue rule (drop turbo on `<d>` shots) stands.

## Plan history

- **v1 (approved 2026-11-22, at its defaults).**
  - Q1 [spend]: download the two DMAD files (~2.8 GB) and run ~45 min of lem
    GPU, capped at 60.
  - Q2 [trust]: load third-party weights from `ZhengmingYu/DMAD`,
    `.safetensors` only, revision pinned.
  - Q3 [scope]: adoption cut, gated on Don's ear and on a seeded re-noise path
    filed as its own issue.
  - Q4 [none]: cut the sung held-audio shot (`hold_audio` judges no generated
    voice).
  - Timing: filed `parked` + `owner:don` behind the same drained-loop gate as
    #1574. Don unparked it on 2026-11-22.
- **Corrections to the issue found by the sweep:**
  - dw already reaches the re-noise sampler through the full scheduler form
    (`configuration` + `from_config_args: {stochastic_sampling: true}`); #1438
    benched it. A bare `stochastic_sampling` key is silently ignored.
  - The real gap is seeding: diffusers' `minimax_h3` denoise loop passes no
    `generator` to `step()`, so the re-noise is unseeded (Pika H3 Flex's loop
    does pass it).
  - The weights are not split into FL2VA and Ref2VA students. There are three
    H3 files (`4step_full_critic`, `4step_lora_critic`, `8step_full_critic`),
    rank-128 LoRAs of ~1.3 GB each.

## What was built

### Stage A (#1584): DMAD dialogue bench (bench only)

`90c3fb5e` (merged to `develop` as `6ca940fc`), then `ee4717fa` (record text,
merged as `7457cf82`). No deploy.

- `bench/workflows/probe-1579-image-to-video.json`, saved as
  `probe-1579-image-to-video` in workspace `probe-1579`: the
  `templates/minimax/image-to-video` template with both schedulers rebuilt in
  the full form, a `revision` on the LoRA entry, and int8 / seed 1001 defaults.
- `bench/records/h3-dmad.md`: the record, with one call per arm.
- `tests/test_plugin_intents.py`: `h3-dmad.md` added to `EXEMPT`.
- **A0 job** `860478669a1e` (also A1 dmad4, prompt 1, seed 1001): failed after
  78 s in loading, 31.7 GB host peak. GPU spent: about 1.3 min of the 60 min cap.
- **Downloads:** only the 4-step file (1.29 GB). The 8-step file was never
  fetched.

**Deviations:** `euler_update_form` left out of `from_config_args` (the
checkpoint's value is already the class default, `blend`); A1-A3 and the 8-step
download skipped because of the A0 stop.

Cases C-F1285 to C-F1292 were written for this stage. The tester verified it on
lem develop @ `7457cf82` as a stop at A0.

## Cost and bounces

| Stage | Bounces | Cost |
|---|---|---|
| #1584 | 1 (`status:needs-info`: the hand-off summarised the record but neither quoted nor linked it, as C-F1291 requires) | Session cost not recorded on the stage; the plan estimated $3-5. GPU: about 1.3 min of job wall on lem |

## Deferred and open

- **Loading DMAD at all** needs one of: a key conversion (in diffusers or dw
  engine code), a key-map option on dw's LoRA entry (`LORA_ENTRY_KEYS` in
  `dw/trust.py` has none), or an upstream re-save in the standard
  diffusers/PEFT format. Each is a new issue and a Don ruling (`trust`, plus
  `scope` if it is engine code). Not filed.
- **A seeded re-noise path** for `minimax_h3` (pass `generator` to `step()`)
  is still the second adoption gate. It isn't needed until a DMAD file loads
  and Don hears a clean voice. Not filed.
- **A zero-key LoRA load was silent.** The tester found it during verify. It is
  now fixed: a LoRA whose keys match no module fails at load, naming the file
  (#1590).
- **The download planner drops a LoRA's `revision`** (the planned entry shows
  `revision: null`; the loader still used the pinned one). Noted in the record,
  not filed.
- **Left on lem:** the 1.29 GB file in the hub cache and the `probe-1579`
  workspace. The record says how to remove both. They stay until Don closes
  #1579.
- Ref2VA shapes, the sung shot and WanGP's TurboNeat Prime checkpoint stay out
  of scope.
