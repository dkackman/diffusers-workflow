# diffusers floor to 0.43.0: compatibility code and floor bump (#1315)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out session,
2026-11-16). Don filed #1315 from the diffusers 0.43.0 release notes as a
compatibility sweep before dw's floor moved from `>=0.42.0`. The plan went
through five versions. Plan v5 is the plan comment on #1315, and the issue
thread holds the full v1 sweep of every 0.43 behaviour change against dw. This
file records what was built and what was cut.

## Verdict: build smaller

Every host and CI already run diffusers git HEAD (lem and mini-ai are on
`0.44.0.dev0`). So the feature shrank to two things: making develop correct on
0.43 behaviour (stage A), and making the declared floor true (stage B). There
was no version move to compare outputs across, so Don cut every cross-version
parity run (ruling of 2026-11-16).

## Plan history

- **v1 (read-only sweep at develop def442b3).** Two stages: A, the 0.43-ready
  code plus 0.42 baselines on lem; B, the move to 0.43 plus parity runs (H3
  clip, H3 chain, LTX two-stage, Kandinsky 6, stacked LoRA).
- **v2 (5043a2fb)** answered the tester's `spec-questions v1`. **v3 (794a139c)**
  recorded Don's v2 ruling on Q1–Q5. Q1 was the modular None drop, and Q3 was
  guarding the attention-backend workaround instead of removing it.
- **v4 (1b891f98)** re-planned after Don closed stage A with "no 0.42.0
  revert". lem stays on HEAD, so the 0.42 baselines were never taken.
- **v5 (develop a1208a29, approved 2026-11-16):** a floor bump only, CPU-only, on
  mini-ai. Corrections from re-reading develop:
  - `docs/DEPENDENCIES.md` states no diffusers number, so it isn't edited.
  - The backend minimums v4 wanted are dropped, because `ACCELERATION.md`
    already says dw doesn't state them.
  - The `RELEASING.md` `shift_provided_sigmas` note v1 cited no longer exists.
    The floor move gets an `## Unreleased` line instead.

## What was built

### Stage A (#1340): 0.43-ready on 0.42

Merge `43575f35` on `develop` (commit `56b2749b`), deployed to lem.

- **Modular None drop:** `dw/pipeline_processors/modular_arguments.py`. On a
  modular pipeline's top-level call, a `None` argument is dropped, so the
  block default still applies under 0.43's "explicit None is kept". Falsy
  non-None values (`0`, `False`, `""`, `[None]`) are kept. A classic pipeline
  keeps its `None`. It applies at `pipeline.py` `_call_pipeline` and on the
  `h3_tiles.py` refine/text path.
- **MagCache calibration readout:**
  `mag_calibration.watching_unfinished_mag_calibration` reads the ratios
  before 0.43's in-call hook reset and again after the call, which covers 0.42.
- **Attention-backend guard:** `set_component_attention_backend` only pins the
  backend when `_AttentionBackendRegistry._active_backend` is absent. The
  workaround from a9a1298c stays in place, guarded.
- **Availability filter:** `available_attention_backends` drops any backend that
  diffusers' private `_check_attention_backend_requirements` refuses, and filters
  nothing on a build that lacks it. `docs/ACCELERATION.md` says so.

Don closed it on 2026-11-16 with no 0.42.0 revert. The code runs on HEAD and
was never tester-verified against a 0.42.0 release, and the 0.42 baselines
(C-F840–C-F846) were never taken.

### Stage B (#1341): floor bump and LTX2 upsampler relocation

Merge `b7a199fa` (commit `0257a5a9`), then bounce fix merge `2d669aa7` (commit
`49b54290`).

- **Floor:** `pyproject.toml` `diffusers>=0.43.0` and `dw/server/updater.py`
  `FALLBACK_RELEASE_FLOOR = "0.43.0"`, edited by hand.
  `scripts/refresh_dep_floors.py` raises every floor to the local venv's, and
  only the diffusers floor was in scope. The REST/Models-page revert now
  installs 0.43.0.
- **Relocation:** `ltx2/two-stage` and `ltx2/refine-clip` name
  `diffusers.models.latent_upscaler.LTX2LatentUpsamplerModel`. In 0.43.0 the old
  `diffusers.pipelines.ltx2.latent_upsampler` path is a deprecated subclass
  that still loads. `tests/test_security_trust_gate.py` pins both paths, and
  the old-path pin is meant to fail when diffusers 1.0 removes it. The
  `test_step_cache.py` keys for the three affected steps were regenerated, so
  a warm worker reloads the upsampler once after the deploy.
- **Conditions did not move.** In 0.43.0, `LTX2VideoCondition` and
  `LTX2ReferenceCondition` are still under `diffusers/pipelines/ltx2/`. So there
  was no dataclass migration and no `CONSTRUCTIBLE_DATACLASS_PACKAGES` change,
  and SE-F065 didn't apply.
- **Docs:** `LORAS.md`'s `alpha` row says an explicit `alpha` also replaces the
  Kohya text-encoder per-module alphas that diffusers keeps from 0.43.
  `RELEASING.md` `### Unreleased` records the floor move.
- **Proof:** the full suite (CPU, `-n auto`) produced identical 78-failure sets
  in scratch venvs on HEAD `0.44.0.dev0` and on the `0.43.0` wheel, and all 78
  pass serially, so they are contention flakes. No test depends on the diffusers
  version. The normal venv had no new failures.
- **Bounce fix (SE-F067):** `dw/type_references.py` now refuses, at validate,
  a malformed dotted `component_type`/`scheduler_type`/`config_type`: an empty
  segment, a leading or trailing dot, or a segment that starts with a digit.
  Before the fix, such a name validated clean and could never load. The trust
  gate was never bypassed.

## Cost and bounces

No stage comment states a `usage:` figure, so session cost is left out.

| Stage | Bounces | Notes |
|---|---|---|
| #1340 | 0 (never sent to the tester) | Deploy held twice for lem jobs. The revert needed the bearer token, so it was held for Don, then dropped by his ruling |
| #1341 | 1 (SE-F067: malformed dotted name validated clean) | CPU only, on mini-ai. No GPU time |

Cases verified on mini-ai: C-F940–C-F943, SE-F067 and C-F849. The retirements
(C-F840–C-F848, C-F850, C-F782, C-F784, C-F785, C-F788–C-F790, C-F792–C-F796
and C-F791's parity half) are dkackman/harnest#222.

## Deferred and open

Cut by Don's 2026-11-16 ruling, each with what brings it back:

- **GPU parity on exactly 0.43.0** (LTX old/new path, H3 clip and chain,
  Kandinsky 6, stacked LoRA). Brought back by a field report of a break on
  0.43.0 that doesn't reproduce on HEAD. 0.43.0 has never run on a GPU here.
- **The sage_hub / flash_hub leak and network check, and removing the
  attention workaround.** Brought back by a workflow that fails after an
  attention-backend switch.
- **A standing CI floor check (v4 Q7).** Brought back by a second floor bump
  that hits a break in the private internals. Stage A's availability filter
  depends on a private diffusers function.
- **`get_class` for dotted `diffusers.models.X`:** a non-goal.
- `dw/tasks/mmaudio_decoder.py` keeps its "needs diffusers >= 0.42.0" skip. It
  is still true, and dead but harmless now.
- #1316 and #1339 were blocked by #1341 and are now unblocked. #1319 shipped
  without waiting (`h3-bf16-burn-probe-complete.md`).
