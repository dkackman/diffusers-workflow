# Long-form Media Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make long pieces (chained takes, cut scenes, long music and narration) easy for an agent and good by default, by measuring every lever on one bench, building and promoting the winners, and teaching agents the result in one playbook skill.

**Architecture:** A bench (stored long-take workflows, a driver script and a second `dw.serve` on lem) scores every arm the same way. The engine work is confined to the chain (`dw/pipeline_processors/chain.py` and small new modules beside it), the H3 anchor, a new workflow-level `presets` mechanism for the quality dial, and a few tasks. Each idea passes a measured gate before it is built past a spike or promoted to a default. The agent-facing result is a new `long-form` skill plus pointers from the existing skills.

**Tech Stack:** Python 3.12, PyTorch, HuggingFace diffusers modular pipelines (MiniMax-H3, LTX-2.5), the dw workflow engine, aiohttp server, pytest (xdist), the dw Claude Code plugin (Markdown skills), `gh` for GitHub, lem (2x RTX 3090) over ssh.

**Spec:** `docs/proposals/long-form-media.md` (approved 2026-11-03). Executors read both.

## Global Constraints

- Branch: all work on `feat/long-form-media` (stage branches off it are fine). Merge to `develop` only after PR #1198 (0.19.0) merges. Never push to `master`.
- Bench server: a second `dw.serve` on lem from a git worktree of this branch, its own port (8766), its own card, its own workspace (`/home/don/bench-workspace`). The main server stays on `develop` for the harnest tester and the release preflight until #1198 merges.
- Kandinsky 6 is out of scope.
- Licences: GPL, AGPL and no-licence sources contribute concepts only. MIT sources may contribute code with an entry in `docs/ATTRIBUTIONS.md`. Every idea that ships credits its source there.
- Model knowledge lives in `plugins/dw/` and the catalog, never in engine code (CLAUDE.md). Compare backends with `get_device_type()`.
- Security: every new argument that names media goes through `dw/locations.py` (`PIPELINE_MEDIA_ARGUMENTS` for pipeline arguments) and `dw/security.py` validators. Never `eval`, `exec` or `shell=True`.
- Defaults may change where the bench supports it; every knob stays reachable. A renamed or moved knob is labelled `breaking-change` at hand-off.
- `standard` tier: the knee of the quality-per-minute curve, hard ceiling about 6x today's H3 chain default per output minute; if the knee puts a 2-minute take above about 3 hours on one card, stop and show Don the numbers before making it the default.
- Bench speed: cheaper bench settings (shorter chains, draft tier, smaller canvas, both cards, a shared segment 1) are allowed once a check shows the effect being measured reproduces at that setting.
- Run-time re-roll: on at `standard` and `best`, off at `draft`, default budget 1 per segment, quoted in the estimate as a worst case; on exhaustion keep the best-measured attempt, finish, flag the segment. Never restructure the run.
- Audio: Claude never calls a timbre change lossless. Anything that may change timbre goes to Don's ear with job ids and a recommendation.
- Judgement: decide by measurement; ask Don when unsure, unable to decide, or for pure taste, always with a recommendation. Collect those in `bench/records/for-don.md`.
- Skill budgets: `tests/test_plugin_skills.py` pins skills. The H3 skill is at its byte cap and `script-to-video`'s SKILL.md has almost no room: new text goes in the new skill or in `references/` files.
- Tests: `venv/bin/python -m pytest` locally (xdist `-n auto`); name files to run in one process. The full backend suite and the arch ratchet (`scripts/arch_metrics.py --check docs/stabilization/baseline.json`) pass before any merge.
- Commits: `feat(chain): #<issue> - ...` style with a scope, never a bare `fix: #N` headline (it auto-closes parents). End every message with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Approvals: Don's approval of the spec and this plan covers every stage. Stage issues are filed at hand-off (Task 30), not before. This session closes losing ideas itself, with numbers.
- Done: playbook, defaults and run-time behaviours shipped, and every idea in the spec's section 5 table has a measured verdict. A large build that passes its gate may become its own follow-up feature.

## Review Focus

- **A chain with a mid-run re-roll that also carries audio:** the re-rolled segment's audio replaces the failed attempt's on the timeline; nothing doubles, and seam crossfades are recomputed against the kept attempt. Test in Task 13.
- **`duration_seconds` that is not a whole number of segments:** the last segment snaps up to the frame grid and the stitched video is trimmed to the asked duration exactly (frame-rounded), with the soundtrack trimmed with it. Test in Task 12.
- **A `presets` value the caller overrides partly:** a caller passing `tier: "draft"` and `num_inference_steps: 30` gets 30 steps and the rest of the draft bundle. Test in Task 18.
- **A segment-check threshold on a deliberate cut or a static locked-off shot:** a seam whose shot is `hard_cut` is never checked, and a static shot whose typical motion is near zero does not fail its jump check from division by a tiny typical delta (the measure's own floor applies). Test in Task 13.
- **An anchor strength of 0 or 1:** 0 is bit-identical to no anchor, 1 bit-identical to today's full match. Test in Task 6.

---

## Phase 0: The bench

### Task 1: Bench server on lem

**Files:**
- Create: `scripts/bench_server.sh`
- Create: `bench/README.md`

**Interfaces:**
- Produces: a running bench `dw.serve` at `http://lem:8766` (token from `BENCH_TOKEN`), workspace `/home/don/bench-workspace`, serving the worktree `/home/don/dw-bench` checked out at the branch named on the command line. `scripts/bench_server.sh {start|stop|status|deploy <branch>}`.

- [ ] **Step 1: Read lem's current state.** Run `ssh lem 'ps aux | grep -i "dw.serve\|python" | grep -v grep; docker ps; nvidia-smi --query-gpu=index,memory.used --format=csv; free -g'`. Record which server is live (native or the Docker container from 2026-11-01), which `--devices` it uses, and the free host RAM.
- [ ] **Step 2: Write `scripts/bench_server.sh`.** `deploy <branch>`: `git -C ~/diffusers-workflow worktree add ~/dw-bench <branch>` the first time, else fetch and fast-forward the worktree. `start`: from `~/dw-bench`, `nohup ~/diffusers-workflow/venv/bin/python -m dw.serve --host 0.0.0.0 --port 8766 --devices "${BENCH_DEVICES:-cuda:1}" --token "$BENCH_TOKEN" --workspace ~/bench-workspace --examples-dir ~/dw-bench/workflows > ~/dw-bench.log 2>&1 & disown`, then poll `/api/health` for 120 s. `stop`: SIGTERM the process whose command line has `--port 8766`, then SIGKILL after 30 s (lem's server does not always exit on TERM). `status`: health JSON plus the worktree's commit. Running `python -m dw.serve` from inside the worktree imports the worktree's `dw` before the editable install.
- [ ] **Step 3: Free the bench card on the main server if it holds it.** If the main server runs workers on both cards, restart it with `--devices cuda:0` the way it was launched (the Docker container's `DW_DEVICES` in `docker/.env`, or `scripts/deploy.sh` with the systemd unit). Check with `nvidia-smi` that nothing of the main server is resident on cuda:1. If the auto-mode classifier refuses the restart, ask Don once to run it.
- [ ] **Step 4: Check host RAM.** Two servers do not share the host-RAM hold. From Step 1's `free -g` and the `host_ram_need` the main server projects for an H3 chain (`GET /api/jobs/<id>` of a recent chain), confirm both servers' largest jobs fit together. If they don't, the bench runs only while the main server is idle, and `bench/README.md` says so.
- [ ] **Step 5: Deploy and start.** `scripts/bench_server.sh deploy feat/long-form-media && scripts/bench_server.sh start`. Expected: health answers `{"status":"ok", ...}` with `device` cuda and the worktree's commit.
- [ ] **Step 6: Write `bench/README.md`.** What the bench is, the server's port, card, workspace and token variable, how to deploy a branch to it, the host-RAM rule from Step 4, and that the main server is untouched.
- [ ] **Step 7: Commit.** `git add scripts/bench_server.sh bench/README.md && git commit -m "chore(bench): a second dw.serve on lem for the long-form bench"`

### Task 2: Bench driver

**Files:**
- Create: `bench/driver.py`
- Create: `bench/verdict.py`
- Test: `tests/test_bench_driver.py`, `tests/test_bench_verdict.py`

**Interfaces:**
- Consumes: the bench server from Task 1; REST routes `POST /api/jobs`, `GET /api/jobs/{id}`, `POST /api/gallery/compare`, the assess route `api/gallery/{name}/assess` (as `dw_mcp/media.py` calls them).
- Produces:
  - `Arm` dataclass: `name: str`, `scene: str`, `workflow: str` (path under `bench/workflows/`), `arguments: dict`, `seed: int`.
  - `load_arms(path: str) -> list[Arm]`: reads a JSON list of arms.
  - `submit(arm: Arm, server: str, token: str) -> str` (job id, arguments include the seed, `note` names the arm).
  - `wait(job_id, server, token, timeout_s: int) -> dict` (the job detail).
  - `measure(job: dict, scene: Scene, server, token) -> dict`: the measures in the spec section 4 for that scene, keyed by measure name.
  - `Scene` dataclass: `name`, `crop: list[int]` (subject-free box), `moments: list[str]` (`"frame:N"` at segments 1, middle and last), `face_reference: str | None`, `lines: list[str] | None`.
  - CLI: `python bench/driver.py run <arms.json> [--measure] [--out bench/records/<name>.json]` and `python bench/driver.py report <record.json>...` (markdown tables to stdout).
  - In `bench/verdict.py`: `noise_floor(a: dict, b: dict) -> dict` (per-measure absolute difference of two same-seed runs), and `verdict(control: dict, arm: dict, noise: dict, primary: str) -> str`, returning `"wins"`, `"loses"`, `"tie"` or `"mixed"` by the spec's rule: wins when the primary beats control by more than noise and no other measure is worse by more than noise.

- [ ] **Step 1: Write the failing tests.**

```python
def test_verdict_wins_only_past_noise_and_never_worse_elsewhere():
    noise = {"seam_look_mean": 0.2, "texture_fine_seg_last": 0.1}
    control = {"seam_look_mean": 2.0, "texture_fine_seg_last": 1.0}
    assert verdict(control, {"seam_look_mean": 1.5, "texture_fine_seg_last": 1.05}, noise, "seam_look_mean") == "wins"
    assert verdict(control, {"seam_look_mean": 1.9, "texture_fine_seg_last": 1.0}, noise, "seam_look_mean") == "tie"
    assert verdict(control, {"seam_look_mean": 1.5, "texture_fine_seg_last": 1.4}, noise, "seam_look_mean") == "mixed"
    assert verdict(control, {"seam_look_mean": 2.6, "texture_fine_seg_last": 1.0}, noise, "seam_look_mean") == "loses"

def test_noise_floor_is_the_absolute_difference_per_measure():
    assert noise_floor({"a": 1.0, "b": 3.0}, {"a": 1.25, "b": 2.0}) == {"a": 0.25, "b": 1.0}

def test_load_arms_reads_every_field(tmp_path): ...  # name, scene, workflow, arguments, seed round-trip

def test_measure_calls_compare_with_the_scene_crop_and_texture_bands(monkeypatch): ...
    # a fake HTTP layer records the compare payload: names=[output], crop=scene.crop,
    # at=scene.moments, texture_bands=True; and the assess call with probe="analyze_seams"
```

Lower is better for every measure the driver reports; the driver negates any measure where higher is better before it reaches `verdict`.

- [ ] **Step 2: Run them to verify they fail.** `venv/bin/python -m pytest tests/test_bench_driver.py tests/test_bench_verdict.py -v`. Expected: ImportError.
- [ ] **Step 3: Implement `bench/verdict.py` and `bench/driver.py`.** HTTP with `urllib.request` (no new dependency). The driver never fetches media files; every measure comes from the server's compare and assess routes, so the tester can repeat them over MCP.
- [ ] **Step 4: Run the tests.** Expected: PASS.
- [ ] **Step 5: Commit.** `git commit -m "feat(bench): driver and verdict rule for the long-form bench"`

### Task 3: Bench scenes as workflows

**Files:**
- Create: `bench/workflows/s1-static.json`, `s2-arc.json`, `s3-dialogue-walk.json`, `s4-montage.json`, `s5-ltx-action.json`, `s6-narration.json`
- Create: `bench/scenes.json` (one `Scene` per id, Task 2's shape)
- Create: `bench/arms/baseline.json`
- Test: `tests/test_bench_workflows.py`

**Interfaces:**
- Consumes: Task 2's `Scene` and `Arm` shapes.
- Produces: six workflows that validate, each exposing the variables the arms vary: `continuity`, `guide_frames`, `segments` (later `duration_seconds`), `anchor`, `guide_noise`, `seed`, and the tier arguments (`width`, `height`, `num_inference_steps`, `weights_dtype`, LoRA name, `attention_backend`).

- [ ] **Step 1: Write the failing test.** `test_every_bench_workflow_validates`: for each file in `bench/workflows/`, `dw.validate` on it returns no errors; `test_every_arm_names_a_scene_and_a_workflow_that_exist`.
- [ ] **Step 2: Run it.** Expected: FAIL (files missing).
- [ ] **Step 3: Write the scenes.** S1 and S2 copy `probe-1166`'s `drift-chain` workflow, prompts and assets (`GET /api/jobs/755a95b35aea/workflow` on the main server gives the arc's realized copy; the static one is in the first #1175 comparison comment on #1166), so the #1175 numbers are a known baseline. S3 is `templates/minimax/chained-segments` with `continuity: "guide"`, `carry_audio: true`, a cast portrait and a spoken line per segment through `prompts`. S4 is `templates/minimax/dialogue-short`-style cut scenes under one `templates/minimax/music` bed through `templates/assemble-and-score`. S5 is `templates/ltx2/chained-segments`. S6 is several `templates/voice-clone` takes and a Music 3 bed under `templates/animatic` boards. Upload any asset they need to the bench workspace with `POST /api/assets` and reference it as `asset:bench/...`.
- [ ] **Step 4: Write `bench/scenes.json`.** Crops chosen from each scene's own frames so the subject never enters them (S2's from #1175: `[0,0,240,150]`). Moments at segment 1's middle, the middle segment's middle and the last segment's middle.
- [ ] **Step 5: Write `bench/arms/baseline.json`.** The control arm per scene: today's template defaults, seed 1166.
- [ ] **Step 6: Run the test.** Expected: PASS.
- [ ] **Step 7: Commit.** `git commit -m "feat(bench): six long-take scenes and their baseline arms"`

### Task 4: Noise floor and cheap-setting checks (lem run)

**Files:**
- Create: `bench/records/noise-floor.md`, `bench/records/noise-floor.json`
- Create: `bench/arms/noise.json`, `bench/arms/cheap-checks.json`

**Interfaces:**
- Consumes: Tasks 1-3.
- Produces: per scene, the noise floor dict every later `verdict` call uses, and the list of cheap bench settings proven safe.

- [ ] **Step 1: Run each scene's control twice with the same seed.** `python bench/driver.py run bench/arms/noise.json --measure --out bench/records/noise-floor.json`. A plain rerun of a seeded workflow is served from the step cache, so the second run must disable it (a distinct inline `id`, or the cache-off setting the server documents in `docs/WORKER_GUIDE.md`).
- [ ] **Step 2: Compute and record the noise floor.** `noise_floor()` per scene, written as a table in `noise-floor.md`.
- [ ] **Step 3: Check the cheap settings.** For S1 and S2, rerun #1175's arms A, B and C at the cheap settings proposed for the bench (6 segments instead of 10; the draft tier; 960x544). The setting is safe when the ordering of A, B and C on the primary measures matches #1175's ordering at full settings. Record which settings passed in `noise-floor.md`.
- [ ] **Step 4: Commit the records.** `git commit -m "bench: noise floor and cheap-setting checks"`

### Task 5: Baseline round (lem run)

**Files:**
- Create: `bench/records/baseline.md`, `bench/records/baseline.json`
- Create: `bench/records/for-don.md`

**Interfaces:**
- Produces: per scene, the control's measures, wall time per output minute and peak VRAM, plus today's best known arm on S1 and S2 (`anchor: "segment_1"` with `guide_noise` 0.4 on static; `guide_noise` 0.4 on arc). These are the numbers every later arm beats or doesn't.

- [ ] **Step 1: Run the baseline and best-known arms** on all six scenes at the safe settings from Task 4.
- [ ] **Step 2: Write `baseline.md`.** One table per scene: measures, wall time, VRAM, frames at the three moments (job ids and output names, so anyone can open them).
- [ ] **Step 3: Start `for-don.md`** with any taste question the baseline already raises, each with a recommendation.
- [ ] **Step 4: Commit.** `git commit -m "bench: baseline round"`

## Phase 1: Counters on today's chains

Tasks 6 to 10 each build one cheap mechanism, off by default, then Task 11 measures them all together. A mechanism's default changes only in Task 11.

### Task 6: `latent_anchor_strength` (#1190)

**Files:**
- Modify: `dw/pipeline_processors/h3_anchor_stats.py` (`match_channel_statistics`)
- Modify: `dw/pipeline_processors/h3_anchor_steps.py` (reads the new input)
- Modify: `dw/pipeline_processors/h3_rules.py` (`LATENT_ANCHOR_STRENGTH_INPUT = "latent_anchor_strength"`, `anchor_strength_problem(value) -> str | None`)
- Modify: `dw/pipeline_processors/chain.py` (`ChainConfig.anchor_strength`, set on segments 2+ with the anchor)
- Modify: `dw/workflow_schema.json` (chain `anchor_strength`), `dw/guides.py` (validation hook beside `guide_noise_errors`)
- Test: `tests/test_h3_anchor.py`, `tests/test_chain.py`

**Interfaces:**
- Produces: `match_channel_statistics(latents, anchor_mean, anchor_std, strength: float = 1.0)`; H3 call argument `latent_anchor_strength` (float in [0, 1], default 1); chain key `anchor_strength` (same range, only with `anchor: "segment_1"`).

- [ ] **Step 1: Write the failing tests.**

```python
def test_strength_one_is_todays_full_match():
    out = match_channel_statistics(x, mean_a, std_a, strength=1.0)
    assert torch.equal(out, match_channel_statistics(x, mean_a, std_a))

def test_strength_zero_returns_the_latent_unchanged():
    assert torch.equal(match_channel_statistics(x, mean_a, std_a, strength=0.0), x)

def test_half_strength_moves_each_channel_halfway():
    full = match_channel_statistics(x, mean_a, std_a)
    half = match_channel_statistics(x, mean_a, std_a, strength=0.5)
    assert torch.allclose(half, x + 0.5 * (full - x), atol=1e-5)

def test_anchor_strength_problem_refuses_out_of_range_and_non_numbers():
    assert anchor_strength_problem(1.5) and anchor_strength_problem(-0.1) and anchor_strength_problem("a")
    assert anchor_strength_problem(0.0) is None and anchor_strength_problem(1) is None

def test_chain_anchor_strength_needs_anchor_segment_1(): ...  # validate refuses it on a chain without the anchor, naming both keys
def test_chain_sets_strength_on_segments_after_the_first(): ...  # FakePipeline sees latent_anchor_strength 0.5 on segments 2+, none on segment 1
```

- [ ] **Step 2: Run them to verify they fail.** `venv/bin/python -m pytest tests/test_h3_anchor.py tests/test_chain.py -v -k strength`. Expected: FAIL.
- [ ] **Step 3: Implement.** The blend is `x + strength * (matched - x)` in float32, returned in the latent's dtype; strength 0 returns the input object itself (bit-identical). Validation through one rule, `anchor_strength_problem`, called by validate and by the run-time path (the `guide_noise` pattern, `h3_rules.guide_noise_problem`).
- [ ] **Step 4: Run the tests.** Expected: PASS. Then `venv/bin/python -m pytest tests/ -q` for the suite.
- [ ] **Step 5: Docs.** `docs/WORKFLOW_GUIDE.md` H3 anchor section and the chain table; `docs/ARCHITECTURE.md` H3 row unchanged unless an owner moved.
- [ ] **Step 6: Commit.** `git commit -m "feat(h3): #1190 - latent_anchor_strength, a partial statistics match"`

### Task 7: `guide_backoff` (#1189)

**Files:**
- Modify: `dw/pipeline_processors/chain.py` (`GuideContinuity.extract`, `ChainConfig`, a tail trim in `run_chain`)
- Modify: `dw/pipeline_processors/h3_rules.py` (`guide_chain_problems` checks it), `dw/workflow_schema.json`
- Test: `tests/test_chain.py`, `tests/test_guide_chain_validation.py`

**Interfaces:**
- Produces: chain key `guide_backoff` (frames, a multiple of 17, default 0, guide continuity only). With backoff `b`, the carry is `frames[-(guide_frames + b):-b]` (audio sliced the same span), and every segment except the last loses its last `b` frames at assembly. `ChainConfig.tail_trim: int`.

- [ ] **Step 1: Write the failing tests.**

```python
def test_backoff_carries_the_frames_one_block_before_the_end(): ...
    # 124-frame fake segments numbered by frame index; guide_frames 22, backoff 17:
    # the carry holds frames 85..106 of the previous segment
def test_backoff_trims_every_segment_but_the_last(): ...
    # 3 segments: stitched length == 124 + 2*(124-22) - 2*17, and the last segment keeps its tail
def test_backoff_slices_the_held_audio_to_the_same_span(): ...
def test_backoff_must_be_a_multiple_of_17_and_needs_guide_continuity(): ...  # validate refuses 10, and 17 on last_frame
def test_backoff_zero_is_todays_chain(): ...  # identical stitched frames and shots to a chain without the key
```

- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement.** Shot records (`shots[i].num_frames`) reflect the trimmed length so `analyze_seams` measures the real seams.
- [ ] **Step 4: Run the tests and the suite.** Expected: PASS.
- [ ] **Step 5: Commit.** `git commit -m "feat(chain): #1189 - guide_backoff hands over one block before the end"`

### Task 8: `keystone` (#1187, first half)

**Files:**
- Modify: `dw/pipeline_processors/chain.py` (`GuideContinuity.inject` and `LastFrameContinuity.inject` take a keystone frame), `dw/pipeline_processors/h3_rules.py`, `dw/workflow_schema.json`
- Test: `tests/test_chain.py`

**Interfaces:**
- Produces: chain key `keystone`: `"none"` (default) or `"segment_1"`. With `"segment_1"`, every later segment's fl2va keyframe (`segment_argument`) is segment 1's frame 0 instead of the carried frame; the guide (guide continuity) is unchanged, and the keyframe region is already trimmed at assembly, so segment 1's frame never reaches the output. On `last_frame` continuity the keystone would replace the only carry, so it is refused there.

- [ ] **Step 1: Write the failing tests.** `test_keystone_sets_every_later_keyframe_to_segment_1_frame_0`, `test_keystone_leaves_the_guide_as_the_carried_tail`, `test_keystone_is_refused_without_guide_continuity`, `test_keystone_none_is_todays_chain`.
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the tests and the suite.** Expected: PASS.
- [ ] **Step 5: Commit.** `git commit -m "feat(chain): #1187 - keystone: segment 1's frame 0 as every later keyframe"`

The late-locked capstone (#1187's second half) needs a step-gated condition row; it is built only if Task 11 leaves the moving-camera gap open (Task 14).

### Task 9: `carry_pick: sharpest` (#1032)

**Files:**
- Create: `dw/pipeline_processors/chain_pick.py`
- Modify: `dw/pipeline_processors/chain.py` (`LastFrameContinuity.extract`, the head trim of the next segment), `dw/workflow_schema.json`
- Test: `tests/test_chain_pick.py`, `tests/test_chain.py`

**Interfaces:**
- Produces: `sharpness(frame: numpy.ndarray) -> float` (variance of the Laplacian on luma) and `sharpest_index(frames: list, window: int) -> int` (index from the end, 0 = last frame) in `chain_pick.py`; chain key `carry_pick`: `{"sharpest": N}` with N in 1..8 (default absent = the last frame). Picking frame `-1-k` drops the last `k` frames of that segment at assembly (motion continuity: the next segment starts from the picked frame), and the job log names the picked frame per seam.

- [ ] **Step 1: Write the failing tests.** `test_a_blurred_last_frame_loses_to_a_sharp_one_before_it`, `test_window_one_is_always_the_last_frame`, `test_the_segment_is_trimmed_after_the_picked_frame`, `test_the_pick_is_logged_per_seam`.
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement** with numpy only (a 3x3 Laplacian by array slicing).
- [ ] **Step 4: Run the tests and the suite.** Expected: PASS.
- [ ] **Step 5: Commit.** `git commit -m "feat(chain): #1032 - carry_pick sharpest of the last N frames"`

### Task 10: `look_hold`, the pixel drift set (#1191, #1033, #1067)

**Files:**
- Create: `dw/pipeline_processors/chain_look.py`
- Modify: `dw/pipeline_processors/chain.py` (applies it per segment before the carry is extracted), `dw/workflow_schema.json`
- Test: `tests/test_chain_look.py`, `tests/test_chain.py`

**Interfaces:**
- Produces in `chain_look.py`, all on `(n, h, w, 3)` uint8 arrays:
  - `tone(frame) -> numpy.ndarray`: per-channel YCbCr quantiles of a pooled copy (17 points).
  - `pickup_grade(frames, carried) -> frames`: match `frames[1]` to `carried` by quantile LUT, carry the same correction to the last frame, interpolate the LUT per frame.
  - `ease_creep(frames, scene_look, tolerance=0.03) -> frames`: when the last frame's contrast or saturation exceeds the scene's by more than the tolerance, scale it down linearly from 1 at the first frame; never brighten, never touch a darker or paler shot.
  - `ease_texture(frames, scene_texture, tolerance=0.03, max_cut=0.5) -> frames`: subtract up to `max_cut` of the faint, edge-masked luma detail at either end where it exceeds the scene's, interpolated across the shot.
  - `precompensate(frame, boost) -> frame`: divide contrast and saturation by the measured boost (clamped to 1..1.25).
  - Chain key `look_hold`: `true` (all four) or an object `{"pickup": bool, "creep": bool, "texture": bool, "precompensate": bool}`; default absent = off. The scene look and texture are segment 1's (the roughest of its first, middle and last frames for texture); the boost is the running mean of rendered-opening look over fed-frame look.
- Concepts from Smite79/MiniMax-H3-LongVideos (no-redistribution licence): no code, credit in `docs/ATTRIBUTIONS.md`.

- [ ] **Step 1: Write the failing tests** on synthetic frames: `test_pickup_grade_matches_the_opening_to_the_carried_frame` (a frame tinted +10 in Cb comes back within 1 of the carried tone), `test_creep_never_brightens_and_never_touches_a_darker_shot`, `test_creep_brings_the_last_frame_to_the_scene_look_within_tolerance`, `test_texture_ease_leaves_edges_alone` (a sharp step edge survives, faint noise on a flat field drops by at most half), `test_precompensate_is_identity_at_boost_one`, `test_look_hold_off_is_todays_chain`.
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement** with numpy and torch only. Apply to the decoded segment frames before `continuity.extract`, so the carried frame is the corrected one, and to the stitched output.
- [ ] **Step 4: Run the tests and the suite.** Expected: PASS.
- [ ] **Step 5: Commit.** `git commit -m "feat(chain): #1191 - look_hold: pickup grade, creep and texture ease-out, precompensated carry"`

### Task 11: Counter round (lem run)

**Files:**
- Create: `bench/arms/counters.json`, `bench/records/counters.md`, `bench/records/counters.json`
- Modify: `bench/records/for-don.md`

**Interfaces:**
- Consumes: Tasks 4-10 deployed to the bench server (`scripts/bench_server.sh deploy feat/long-form-media && scripts/bench_server.sh stop && scripts/bench_server.sh start`).
- Produces: per shot type (static S1, moving S2, dialogue walk S3), the winning counter set, written as the table the playbook will quote, and the verdict for #1187 (keystone half), #1189, #1190, #1032, #1033, #1067, #1191 and the anti-burn LoRA.

- [ ] **Step 1: Write the arms.** One arm per counter alone on S1, S2 and S3; `latent_anchor_strength` at 0.25, 0.5 and 0.75; the anti-burn LoRA (`minimax-h3/h3-antiburn` at 0.6) on every segment; then the pairwise combinations of every counter that won alone on a scene.
- [ ] **Step 2: Run round 1** (each alone), measure, compute verdicts with the noise floor.
- [ ] **Step 3: Run round 2** (combinations of the round-1 winners per scene) and a second seed for every winner.
- [ ] **Step 4: Confirm at length.** The best set per scene at two minutes.
- [ ] **Step 5: Record.** `counters.md`: a table per scene, the verdict per idea, the job ids, and frames at the three moments. Taste questions (look better against measure better) go to `for-don.md` with a recommendation.
- [ ] **Step 6: Set the defaults the evidence supports.** In `workflows/templates/minimax/chained-segments.json` (and the matched chains where the scene type applies): the continuity default, and the counters per shot type through Task 15's mechanism once it exists. Until then, record the intended defaults in `counters.md`.
- [ ] **Step 7: Commit.** `git commit -m "bench: counter round - verdicts for #1187a #1189 #1190 #1032 #1033 #1067 #1191"`

## Phase 2: Running a long piece

### Task 12: `duration_seconds`

**Files:**
- Modify: `dw/pipeline_processors/chain.py` (`ChainConfig`, a pure `chain_segment_count(chain_definition, arguments) -> int | None`)
- Modify: `dw/workflow_schema.json` (chain `duration_seconds`; exactly one of `segments`, `match_audio`, `duration_seconds`)
- Modify: `dw/plan.py` / `dw/cost_scaling.py` (a chain's per-segment cost times `chain_segment_count`)
- Modify: `workflows/templates/minimax/chained-segments.json`, `workflows/templates/ltx2/chained-segments.json` (a `duration_seconds` variable; `segments` stays)
- Test: `tests/test_chain.py`, `tests/test_plan.py` (or the existing estimate test file)

**Interfaces:**
- Produces: chain key `duration_seconds` (number > 0, or a `variable:` reference); `chain_segment_count(...)` used by the run and the estimate alike.

- [ ] **Step 1: Write the failing tests.** `test_duration_plans_whole_segments_and_trims_to_the_asked_length` (40 s at 24 fps with 124-frame guide segments: the plan's frame total is ≥ 960, the stitched video is exactly 960 frames and its audio exactly 40 s), `test_exactly_one_of_segments_match_audio_duration`, `test_the_estimate_scales_by_the_segment_count_duration_implies`, `test_duration_with_frame_snap_snaps_the_last_segment_up`.
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement** by reusing `plan_segments(total_frames, segment_frames, head_trim, frame_snap)` with `total_frames = round(duration_seconds * fps)`, and the existing `match_audio` trim path for the overshoot.
- [ ] **Step 4: Run the tests and the suite.** Expected: PASS.
- [ ] **Step 5: Commit.** `git commit -m "feat(chain): duration_seconds - ask for a chain by its length"`

### Task 13: Per-segment check and bounded re-roll

**Files:**
- Create: `dw/pipeline_processors/chain_check.py`
- Modify: `dw/pipeline_processors/chain.py` (the segment loop), `dw/workflow_schema.json`, `dw/plan.py` (worst-case retries in the estimate)
- Modify: `dw/events.py` only if a new warning kind needs registering
- Test: `tests/test_chain_check.py`, `tests/test_chain.py`

**Interfaces:**
- Consumes: Task 4's noise floor (to set default thresholds), `dw/tasks/assess.py` seam arithmetic (`_seam_video`, `_seam_look`) and `dw/texture_bands.py` on in-memory frames (`assess._in_memory_thumbs`).
- Produces:
  - `SegmentCheck` dataclass: `jump_ratio_max: float`, `look_ratio_max: float`, `texture_ratio_max: float | None` (fine band against segment 1 on the whole frame), `face: None` (reserved).
  - `check_segment(previous_frames, frames, segment_1_frames, check: SegmentCheck) -> CheckResult` with `CheckResult(passed: bool, measures: dict, failed: list[str])`.
  - Chain keys: `check`: `true` (the default thresholds), `false`, or an object with `SegmentCheck`'s fields; `retries`: integer 0..3. Template defaults: `check: true` and `retries: 1` at `standard` and `best`, `retries: 0` at `draft` (wired through Task 18's presets).
  - Run record: each shot gains `attempts` (int) and, when its check failed on every attempt, `check_failed: [measure names]`; the job emits one `warning` per such segment naming the measures and the kept attempt.
- Re-roll: a failed segment is re-run with a fresh seed (the step's seed plus the attempt number times a fixed prime), from the same carry; the kept attempt is the one with the fewest failed measures, then the lowest jump ratio. Its audio and frames replace the failed attempt's; crossfades are computed against the kept attempt. A seam marked `hard_cut` is never checked.

- [ ] **Step 1: Write the failing tests.**

```python
def test_a_jump_over_the_threshold_fails_and_names_the_measure(): ...
def test_a_static_shot_with_near_zero_motion_does_not_fail_on_jump(): ...  # typical delta under the assess floor
def test_hard_cut_seams_are_never_checked(): ...
def test_a_failed_segment_is_rerolled_with_a_new_seed_from_the_same_carry(): ...  # FakePipeline sees seed+prime, carry unchanged
def test_the_kept_attempts_audio_replaces_the_failed_one(): ...  # stitched audio length and seam crossfade match the kept attempt
def test_budget_exhausted_keeps_the_best_attempt_and_warns(): ...
def test_retries_zero_never_rerolls(): ...
def test_the_estimate_quotes_the_worst_case_retries(): ...  # plan.estimate minutes == per-segment * segments * (1 + retries)
```

- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement.** Default thresholds: the 99th percentile of each measure across the noise-floor runs of S1-S3, plus margin, written as constants in `chain_check.py` with a comment citing `bench/records/noise-floor.md`.
- [ ] **Step 4: Run the tests and the suite.** Expected: PASS.
- [ ] **Step 5: Bench it (lem).** S3 with a segment made to fail (a prompt that breaks staging on segment 4 only, through `prompts`): the check catches it, the re-roll fixes or flags it. Record in `bench/records/check.md`.
- [ ] **Step 6: Commit.** `git commit -m "feat(chain): per-segment check with a bounded re-roll"`

### Task 14: Repair in place (#1169) and the late capstone (#1187, second half)

**Files:**
- Create: `workflows/templates/minimax/repair-segment.json` (if a pure workflow can express it)
- Create: `bench/records/repair.md`
- Modify (only if Step 2 says engine work is needed): `dw/pipeline_processors/h3_rules.py` (a tail guide ending past `num_frames`), `dw/pipeline_processors/h3_guide_steps.py`

**Interfaces:**
- Produces: a template that re-renders segment `k` of a finished chain with segment `k-1`'s last `guide_frames` as a frame-0 guide and segment `k+1`'s first `guide_frames` as a guide ending at `num_frames`, then rejoins with `concat_videos` and `pair_audio` (the guide frames trimmed off both ends). Arguments: `chain_output` (an `output:` of the chain), `segment` (1-based), `prompt`, `seed`.

- [ ] **Step 1: Check whether a pure workflow can do it.** Per-segment files exist with `save_segments` + `keep_segments`; `trim_video`, `concat_videos` and the existing guides cover the rest. `h3_rules.guide_end_problem` already accepts a guide that ends exactly at the end of the render.
- [ ] **Step 2: Build the template** (or, if a guide cannot sit at the tail, the smallest engine change that allows it, with tests in `tests/test_h3_guides.py`).
- [ ] **Step 3: Bench it.** On Task 13's S3 run with a flagged segment: repair it, measure `analyze_seams` at both of the repaired segment's seams against the chain's other seams.
- [ ] **Step 4: The capstone gate.** If Task 11 left S2 (moving camera) without a winning counter, spike #1187's late capstone: a one-frame condition row past the end of the render from segment 1's frame `num_frames - 5`, entering only below a sigma threshold. Record the spike's verdict in `repair.md`; build it only if it wins S2.
- [ ] **Step 5: Commit.** `git commit -m "feat(h3): #1169 - repair one chain segment in place between its neighbours"`

### Task 15: Counters chosen by shot type

**Files:**
- Modify: `dw/pipeline_processors/chain.py` (`ChainConfig`), `dw/workflow_schema.json`
- Modify: the H3 chain templates
- Test: `tests/test_chain.py`

**Interfaces:**
- Produces: chain key `shot`: `"static" | "moving" | "dialogue" | "auto"`; the template maps each to Task 11's winning counter set through its own variables (the mapping is template data, not engine code, per CLAUDE.md). `"auto"` is built only if Task 11 shows that segment 1's measured motion (`analyze_shots`' typical delta) separates S1 from S2 reliably; then the chain measures segment 1 and picks the static or moving set. Explicit counter keys always win over the shot type.

- [ ] **Step 1: Write the failing tests.** `test_shot_static_applies_the_templates_static_counters`, `test_explicit_counter_keys_win_over_the_shot_type`, and, if auto is built, `test_auto_picks_static_for_a_still_segment_1_and_moving_for_a_panning_one` on synthetic frames.
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the tests and the suite.** Expected: PASS.
- [ ] **Step 5: Bench it.** S1, S2 and S3 with `shot` set (and `auto`), against their Task 11 best arms: within noise.
- [ ] **Step 6: Commit.** `git commit -m "feat(chain): shot type picks the drift counters"`

### Task 16: Performer replacement gate (#1194)

**Files:**
- Create: `bench/records/performer-replacement.md`

- [ ] **Step 1: Spike.** On one cut of S4: a per-frame person mask (check the catalog's segmentation models and SAM availability under diffusers/transformers on lem, with licences), a regional edit on ref2va (`templates/minimax/regional-edit`) at `refine_strength` 1.0 inside the mask with the replacement portrait as the identity reference and `hold_audio` the source.
- [ ] **Step 2: Measure.** `face_drift` against the replacement portrait, `compare_outputs` outside the mask (unchanged), the seam at the mask edge by eye.
- [ ] **Step 3: Verdict.** If it works, it becomes a follow-up feature (a person-track task and a cut-detection task) filed at hand-off; if not, close #1194 with the numbers.
- [ ] **Step 4: Commit the record.**

## Phase 3: Structural continuity (gated)

### Task 17: Latent carry family and one long render (#1150, #1044, #1030, #1188)

**Files:**
- Create: `bench/records/structural.md`

- [ ] **Step 1: The gate.** From Tasks 11-15: does the best chain per shot type still leave a seam or drift gap larger than noise against segment 1 on any scene? If not, close #1150, #1044 and #1030 with the numbers, and skip to Step 4.
- [ ] **Step 2: Latent carry spike** (#1150's stages A and B as its plan comment designs them, on a scratch flag): measure against the best chain on the scene with the gap.
- [ ] **Step 3: Verdict for #1150, #1044, #1030.** A winner becomes a follow-up feature with its own plan, filed at hand-off.
- [ ] **Step 4: Hybrid windows spike (#1188).** On the diffusers H3 modular pipeline: can a denoise block run the transformer on a token sub-range (video rows plus their 40 Hz audio rows, rotary positions kept) per step? Build the smallest version that renders S1 at about 40 s in windows, with no keystone, and measure it against the best chain. Record feasibility, VRAM, wall time and the measures.
- [ ] **Step 5: Verdict for #1188.** A win becomes a follow-up feature filed at hand-off; a loss or an infeasible spike closes it with the record.
- [ ] **Step 6: Commit the record.**

## Phase 4: The quality dial and speed

### Task 18: `presets`, one variable that moves several

**Files:**
- Create: `dw/presets.py`
- Modify: `dw/workflow_schema.json` (top-level `presets`), `dw/workflow.py:356` and `dw/validation.py:901` (call `apply_presets` before `set_variables`), `dw/validation.py` (static checks)
- Test: `tests/test_presets.py`

**Interfaces:**
- Produces: top-level workflow key `presets`: `{"<selector variable>": {"<value>": {"<variable>": <value>, ...}, ...}}`. `apply_presets(definition: dict, arguments: dict, variables: dict) -> None` sets, for each selector, the chosen bundle's values on every variable the caller did not pass. Static checks: every bundle key is a declared variable; the selector's default is one of its bundle names; a caller value for the selector outside the bundle names is refused at `arguments.<selector>` naming the allowed values.

- [ ] **Step 1: Write the failing tests.**

```python
def test_the_bundle_sets_the_variables_the_caller_did_not_pass(): ...
def test_a_caller_value_beats_the_bundle(): ...  # tier=draft with num_inference_steps=30 -> 30 steps, rest of draft
def test_the_default_bundle_applies_with_no_arguments(): ...
def test_an_unknown_selector_value_is_refused_naming_the_values(): ...
def test_a_bundle_naming_an_undeclared_variable_is_a_validation_error(): ...
def test_a_workflow_without_presets_is_unchanged(): ...
```

- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the tests and the suite.** Expected: PASS.
- [ ] **Step 5: Docs.** `docs/WORKFLOW_GUIDE.md` (a *Presets* section next to *What a variable is allowed to be*); the MCP surface passes the schema through, so check `scripts/surface_snapshot.py` and the OpenAPI snapshot.
- [ ] **Step 6: Commit.** `git commit -m "feat(workflow): presets - one variable selects a bundle of others"`

### Task 19: Speed levers round (lem run)

**Files:**
- Create: `bench/records/speed.md`

- [ ] **Step 1: The candidates.** For H3 at each tier: `attention_backend` (sparse top-k, already default; sage_hub), the `cache` settings (`first_block` 0.05 at 20+ steps; MagCache 0.06 on a 29-step call), the quantized-weight cache, FastH3 Omni (#980), SelfLift (#1195: spike a mid-schedule half-canvas lift on a scratch flag first), TST (#1196: spike the per-head correction as an attention processor on dense attention first). For LTX-2.5: the levers its skill lists.
- [ ] **Step 2: Lossless test** per the spec: 3 seeds, staging and identity held, PSNR in the "same shot" band or better against the slow run, no drift or seam measure worse than noise. Audio measured as the spec says; a possible timbre change goes to `for-don.md`.
- [ ] **Step 3: Promote.** Lossless levers become template defaults at the tiers they were measured at. Lossy ones are recorded as dial steps with their cost.
- [ ] **Step 4: Verdicts** for #980, #1195 and #1196: built (a spike that won becomes a stage here, with its own tests in the TDD form above) or closed with numbers.
- [ ] **Step 5: Commit.**

### Task 20: Tiers on the long-form templates

**Files:**
- Modify: `workflows/templates/minimax/chained-segments.json`, `chain-video-continuity.json`, `chain-matched-to-audio.json`, `chain-matched-and-aligned.json`, `dialogue-short.json`, `music-video.json`, `workflows/templates/ltx2/chained-segments.json`, `extend-clip.json`
- Create: `bench/records/tiers.md`
- Test: `tests/test_bench_workflows.py` (a template's `tier` bundles name declared variables), the existing template tests

**Interfaces:**
- Consumes: Task 18 `presets`; Tasks 11, 13 and 19's winners.
- Produces: a `tier` variable (`draft`, `standard`, `best`; default `standard`) on each template, through `presets`, setting canvas, weights, LoRA, steps, attention, cache, counters, `check` and `retries`.

- [ ] **Step 1: Measure the quality-per-minute curve** on S1-S3 (and S5 for LTX) across the candidate settings; pick `standard` at the knee.
- [ ] **Step 2: Apply the cost rule.** If `standard` puts a 2-minute take above about 3 hours on one card, stop and put the numbers in `for-don.md` with a recommendation; wait for Don before making it the default.
- [ ] **Step 3: Write the bundles** and update each template's `cost` entry and `cost_drivers` to the new default (measured on the bench).
- [ ] **Step 4: Run the tests.** Expected: PASS.
- [ ] **Step 5: Commit.** `git commit -m "feat(templates): quality tiers on the long-form templates"`

## Phase 5: Audio

### Task 21: Audio-only pass (#1192)

**Files:**
- Create: `dw/pipeline_processors/h3_audio_pass.py` (if the gate in Step 1 passes)
- Modify: `dw/pipeline_processors/h3_rules.py` (`source_video` with neither a mask nor `hold_audio` means the audio pass), `dw/pipeline_processors/h3_mask_steps.py` or a new block for the all-rows-kept case
- Test: `tests/test_h3_audio_pass.py`

**Interfaces:**
- Produces: an H3 call with `source_video` and no `mask_*` and no `hold_audio` keeps every video row (the regional edit's kept-row write-back with an all-kept mask) and generates the audio from the prompt and its `<d>` lines; the output is the source's own pixels with the new track. Optional `listen_scale` (1, 0.5, 0.25) runs the kept rows at a smaller canvas.

- [ ] **Step 1: Gate.** Read #984's plan v3 (kept on the issue) for the locked-video-latent block design and check it against the regional edit blocks. If an all-kept mask through the existing blocks works, build on it; else the smallest new block.
- [ ] **Step 2: Write the failing tests:** `test_the_output_frames_are_the_sources_own`, `test_listen_scale_half_runs_the_kept_rows_at_half_canvas`, `test_a_mask_or_hold_audio_keeps_todays_meaning`.
- [ ] **Step 3: Run them.** Expected: FAIL.
- [ ] **Step 4: Implement.**
- [ ] **Step 5: Run the tests and the suite.** Expected: PASS.
- [ ] **Step 6: Bench it.** S4's silent shots and a re-voice of one S3 segment; `pcm`/level measures plus job ids in `for-don.md` for Don's ear (Claude cannot judge it). A template `templates/minimax/add-sound` if Don's verdict is positive.
- [ ] **Step 7: Commit.**

### Task 22: Long music and long narration

**Files:**
- Create: `workflows/templates/narration-long.json`, `workflows/templates/minimax/music-long.json` (if the gates pass)
- Create: `dw/tasks/text_takes.py` (only if no existing task splits text into takes)
- Create: `bench/records/audio.md`
- Test: `tests/test_text_takes.py`

**Interfaces:**
- Produces: `split_takes(text: str, max_seconds: float, words_per_second: float) -> list[str]` (sentence boundaries, never mid-sentence; a sentence longer than one take is split at its last comma before the limit) as a task `split_takes`; `narration-long` speaks each take with `voice-clone` under one seed and voice and joins them with `concat_audio` (or the existing join task) with a measured gap; `music-long` writes a piece longer than one Music 3 track (300 s) as sections joined with a crossfade at musical boundaries (`analyze_beats`, `join_into_song`).

- [ ] **Step 1: Check what exists.** `list_tasks` on the bench server for text splitting and audio joins; reuse before writing.
- [ ] **Step 2: Write the failing tests** for `split_takes` (no take over the limit; no take ends mid-sentence unless the sentence alone is over it; the takes rejoin to the text).
- [ ] **Step 3: Run them.** Expected: FAIL. **Step 4: Implement.** **Step 5: Run them.** Expected: PASS.
- [ ] **Step 6: Bench S6** (narration over 3 minutes, music over 5): seam level steps, `check-script` on the narration, and job ids in `for-don.md` for listening.
- [ ] **Step 7: Commit.**

## Phase 6: LTX-2.5 parity

### Task 23: LTX long takes

**Files:**
- Create: `bench/records/ltx.md`
- Modify: `workflows/templates/ltx2/chained-segments.json`, `extend-clip.json` (defaults the round supports)

- [ ] **Step 1: Round on S5.** `last_frame` chain against `extend-clip` conditioned on the whole previous segment, the `look_hold` set (Task 10, pixel-only, so family-independent), `carry_pick` (Task 9), the check and re-roll (Task 13), and `duration_seconds`.
- [ ] **Step 2: Defaults.** What wins becomes the LTX templates' defaults through `tier`.
- [ ] **Step 3: Record and commit.**

## Phase 7: The agent layer

### Task 24: The `long-form` playbook skill

**Files:**
- Create: `plugins/dw/skills/long-form/SKILL.md`, `references/levers.md`, `references/quality-and-speed.md`, `references/run-and-repair.md`, `references/audio.md`
- Modify: `plugins/dw/README.md`, `CLAUDE.md` (the plugin paragraph's skill list), `.claude-plugin/marketplace.json` only if it lists skills
- Test: `tests/test_plugin_skills.py` (the new skill's catalog names exist, its references are linked, its budget holds)

**Interfaces:**
- Consumes: every `bench/records/*.md` verdict; Tasks 12, 13, 14, 15, 18, 20, 21, 22, 23 behaviours.
- Produces: the decision per piece, scene, seam and run from the spec's section 1, the tier dial from section 2 with measured costs per output minute, and every rule citing the bench record it came from. Community claims marked as such.

- [ ] **Step 1: Write the failing test** `test_long_form_skill_is_named_in_the_readme_and_claude_md`, and add `long-form` to the skill lists the existing tests pin.
- [ ] **Step 2: Run it.** Expected: FAIL.
- [ ] **Step 3: Write the skill** inside its budget. The description says when it applies: any request longer than one clip, a scene to block, a long take, a series, a music video, long music or narration.
- [ ] **Step 4: Run `tests/test_plugin_skills.py`.** Expected: PASS.
- [ ] **Step 5: Commit.** `git commit -m "feat(plugin): long-form - the playbook for pieces longer than one clip"`

### Task 25: Pointers and planning fields in the other skills

**Files:**
- Modify: `plugins/dw/skills/storyboard/SKILL.md`, `plugins/dw/skills/script-to-video/references/shot-document.md` (per-scene `length_seconds` and `continuity`), `plugins/dw/skills/series-episodes/SKILL.md`, `plugins/dw/skills/minimax-h3/SKILL.md` and `references/chains.md`, `plugins/dw/skills/ltx-2.5/SKILL.md`, `plugins/dw/skills/minimax-music3/SKILL.md`
- Test: `tests/test_plugin_skills.py`

- [ ] **Step 1: Write the failing tests:** the shot document names `length_seconds` and `continuity`; minimax-h3 and ltx-2.5 point to `long-form` for anything longer than one clip; every budget holds.
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Edit the skills.** Replace each family's "longer" guidance with a pointer, keep only family-specific rules, and move the community drift notes in `chains.md` into the playbook with their measured status.
- [ ] **Step 4: Run the tests.** Expected: PASS.
- [ ] **Step 5: Commit.**

### Task 26: Skill claims (#1193)

**Files:**
- Create: `docs/proposals/audits/2026-11-longvideos-claims.md`
- Modify: the skill references each verdict belongs in

- [ ] **Step 1: Run the five claims** as #1193 lists them, two seeds per arm, T2VA/Ref2VA turbo at 960x544, on the bench server.
- [ ] **Step 2: Record and land.** A claim that held on both seeds becomes a rule in its reference; the rest go under *Probed and not a rule*.
- [ ] **Step 3: Commit.**

### Task 27: `export_timeline` (#1197)

**Files:**
- Create: `dw/tasks/timeline_export.py`
- Modify: the task registry (`register_command`, as other tasks do), `docs/TASKS.md`
- Test: `tests/test_timeline_export.py`

**Interfaces:**
- Produces: task `export_timeline` with arguments `video` (a joined output with `shots`), `sources` (optional per-shot source files; default the shots' own spans of `video`), `name`; writes `<name>.otioz` (zip: `version.txt`, `content.otio`, media) next to its outputs. OTIO JSON written directly (`Timeline.1`, `Stack.1`, `Track.1`, `Clip.1`, `Gap.1`, `ExternalReference.1`, `TimeRange.1`, `RationalTime.1`), rates as exact rationals. Concept and schema shape from leonardofrito/Directors-Cut (MIT), credited in `docs/ATTRIBUTIONS.md`.

- [ ] **Step 1: Write the failing tests:** the bundle has the three members; clip `source_range`s and gaps sum to the video's duration; a 23.976 fps video writes rate 24000/1001; a media path is confined by `validate_output_path`.
- [ ] **Step 2: Run them.** Expected: FAIL. **Step 3: Implement.** **Step 4: Run them.** Expected: PASS.
- [ ] **Step 5: Check Resolve import** is out of reach here (no Resolve on lem): ask Don in `for-don.md` to open one bench cut's `.otioz` in Resolve 18.5+.
- [ ] **Step 6: Commit.**

### Task 28: Cold-session drill

**Files:**
- Create: `bench/records/drill.md`

- [ ] **Step 1: Run a fresh Claude session** with only the dw MCP pointed at the bench server and the plugin from this branch (the procedure in memory `cold-session-mcp-tests`), asked for: a 90-second single take of a walk-and-talk; a 3-minute piece of four scenes with a score; a 5-minute narration over boards.
- [ ] **Step 2: Score the transcripts:** did it read `long-form` first, pick the shape per scene the playbook says, quote cost with retries, use the tier dial, and handle a flagged segment? Fix the playbook where it misled, and rerun.
- [ ] **Step 3: Commit the record and any skill fixes.**

## Phase 8: Close-out and hand-off

### Task 29: Merge, deploy and the record

**Files:**
- Create: `docs/proposals/complete/long-form-media-complete.md`
- Modify: `docs/proposals/todo.md`, `docs/ATTRIBUTIONS.md`, `docs/ARCHITECTURE.md` (rows for `chain_check.py`, `chain_look.py`, `chain_pick.py`, `presets.py`)

- [ ] **Step 1: Wait for #1198 to merge** (`gh pr view 1198 --json state`). Then merge `feat/long-form-media` to `develop` with `--no-ff`.
- [ ] **Step 2: Verify:** `venv/bin/python -m pytest tests/ -q`, the UI checks if the UI changed, `scripts/arch_metrics.py --check docs/stabilization/baseline.json`, the OpenAPI and surface snapshots. All green before pushing.
- [ ] **Step 3: Push `develop` and deploy the main lem server** with `scripts/deploy.sh develop`; return the main server to both cards and stop the bench server (`scripts/bench_server.sh stop`; remove the worktree).
- [ ] **Step 4: Copy the bench to the main server** for the tester: the six bench workflows saved into a workspace `bench` there (`save_workflow`), with their assets (`copy_to_workspace` or `upload_asset`).
- [ ] **Step 5: Write the design record** in the house format (`complete/h3-guide-chain-drift-complete.md` is the model): what shipped, the bench's verdicts with job ids, what was cut and what would bring it back.
- [ ] **Step 6: Commit and push.**

### Task 30: GitHub hand-off

- [ ] **Step 1: File the feature issue** "Long-form media" (`feature`, `owner:tester`, `backend:shared`, `target:lem`) linking the design and the record, with the plan as its plan comment (`<!-- harnest:plan v1 -->`) and `status:plan-approved` added by Don's approval of this plan (ask Don to add the label, since only Don may).
- [ ] **Step 2: File one `stage` issue per shipped behaviour** (Tasks 6, 7, 8, 9, 10, 12, 13, 14, 15, 18, 20, 21, 22, 23, 24-25, 27), each with: what was built; how to observe it over MCP (tool calls, workflow names in the `bench` workspace, arguments); the acceptance intent the tester writes cases from; the bench job ids that showed it; `breaking-change` where a default moved. Labels `stage`, `owner:tester`, `status:fixed-pending-verify`, `target:lem`.
- [ ] **Step 3: Close every idea with its verdict.** For each of #1187-#1197, #1030, #1032, #1033, #1044, #1067, #1150, #1169: a comment with the bench numbers and the stage that shipped it (closed as completed by the tester through the stage) or the reason it lost (closed `not planned` now). A winner too large for this effort gets a follow-up feature issue (`idea`, `owner:lead`).
- [ ] **Step 4: Post `bench/records/for-don.md`** as a comment on the feature issue: every taste call left open, each with a recommendation.
- [ ] **Step 5: Save a memory** of the outcome and the hand-off state.
