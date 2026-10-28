# Task weights in `plan.downloads_required` (#975)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out session,
2026-10-23). Filed from #973 (stage A of #956) after `remove_object`'s first
run downloaded about 200 MB that nobody quoted. Plan v1 was designed on
#975. Don's review made v2, and Don approved v2. It was built as two stages,
#977 and #978. The plan comment on #975 stays the full design. This doc
records what shipped.

## The problem

`validate_workflow`'s `plan.downloads_required` walked only pipeline weights
(`from_pretrained_arguments`, `loras`, URL `from_single_file`). Every task
that fetched weights did so with nothing quoted, and nothing in the bound
acknowledgement (#85) covered it. The issue named `upscale`,
`restore_faces` and `remove_object` (0.06–0.2 GB files). The design sweep
found bigger unlisted ones: `remove_background` (RMBG-2.0, ~1 GB),
`segment` (~1.8 GB), `text_generation` (Qwen2.5-1.5B, ~3 GB),
`diffusion_upscale` (16 GB), speech and transcription.

## Verdict: build both stages

Demand was thin (no field report), but the field is a contract: it is what
a bound acknowledgement checks, so a list that silently leaves things out
is worse than none (Don, review). The field had been wrong three times
before in the same way (#382, #894, #151).

Non-goals stayed out: no task's download behaviour changed; facexlib is
reported, not mirrored (#616); `download_model` still fetches whole repos
at `main`; no UI change; zoe_depth, dw_pose and demucs (fetched outside the
HF cache) stay unlisted and documented.

## Plan history

- **v1:** one hook per task, a per-file cache check and HEAD sizing, and
  `(repo, filename, revision)` keying. Stage A was single-file tasks, B the
  whole-repo ones.
- **v2 (approved), Don's four changes:**
  1. facexlib's weights dir is derived with `importlib.util.find_spec`, never
     imported (its `misc.py` imports `cv2` and `torch`).
  2. Each loader reads its source from the same resolution as its
     declaration, so the two can't disagree.
  3. Q4: no H3 VAE subfolder entry (every H3 box lists `MiniMaxAI/MiniMax-H3`
     whole); `subfolder` dropped from the shape.
  4. Q1: re-split. A′ is the hook plus the multi-GB whole-repo tasks; B′ is the
     file-level detail, so if either were cut it would be the smaller one.

Corrections the plan made to the issue's premise: the repo-level cache
check is wrong for single files (a pinned `hf_hub_download` writes no
`refs/main`, and any other cached file of the repo would mask the wanted
one); `_model_info` sizes the whole repo, not one file; dedup was by repo
name only.

## What was built

### Stage A′ (#977): the hook, the plan walk, whole-repo declarations

Commits 36305b36 (code) and b0d0d857 (tests, docs), merged as c6ff498a. The
bounce fix 138bbc42 was merged as a27b61be.
- **Hook:** `required_weights(arguments)`, named on `register_command` and
  looked up by `dw/tasks/registry.py:required_weights_hook`. Helpers are in
  `dw/tasks/weights.py`. `dw/plan.py:_collect_task_weights` walks the
  expanded definition and each composed child. Task repos join the same
  list as pipeline repos, so they share its shape filter, cache test and
  sizing. A shared repo is listed once. A reference or local-path argument
  gives no entry. A hook that raises gives no entry and never fails the
  validate.
- **Declarations:** segment (`model_name` detector, `sam_model_name`
  segmenter), remove_background, diffusion_upscale, transcribe_audio (and
  check_script through it), generate_speech (plus speechbrain x-vector with
  `speaker_embedding`), attribute_voices (ECAPA only), text_generation,
  image_to_text, judge, and the image processors (depth, depth_estimator,
  controlnet_aux detectors, sam, segmentation).
- **Pin:** `tests/test_task_weights.py::TestEveryHubLoadIsDeclared`.
- **Bound acknowledgement fix (the bounce):** `AcknowledgedCost._from_plan`
  in `dw/server/admission.py` reads a validate `plan` sent unedited as the
  acknowledgement. Its `downloads_required` and `estimate.minutes` become
  `downloads` and `minutes`. An explicit `downloads`, even `[]`, wins.
- **Deviations:** judge and image_to_text default to SmolVLM-256M (what
  they load), and Qwen applies only to text_generation without an `image`.
  depth_estimator now passes transformers' default (`Intel/dpt-large` at a
  pin) explicitly, which loads the same weights as before. A
  `system_prompt.from_hub` file is not listed. The first validate of a
  workflow with a task step imports that task's module, which costs a few
  seconds once per server process.
- **Test-only:** `tests/test_server_jobs.py::finished_job` waits for the
  history row (a parallel-load flake that tripped the hand-off gate).

### Stage B′ (#978): file-level entries, single-file tasks, facexlib

Commit 9da24cc3, merged as d5bb9f8b. The map-row fix b72ca04a was merged as
b01f1238.
- **Entries** keyed by `(repo, filename, revision)`. Each file is checked on
  its own with `_cached_file` at `revision or "main"` and sized by a HEAD of
  that file (`_file_info`). A whole-repo entry for the same repo subsumes it.
  `RequiredDownload` gained optional `filename`/`revision`, and the
  OpenAPI and UI types were regenerated. The MCP cost-gate line renders
  `repo/filename`.
- **Declarations** (hook and loader share each task's `*_source`):
  `upscale` (file, or whole repo with no `filename`), `restore_faces`
  (spandrel entry plus facexlib URLs), `remove_object` (pinned only for the
  default pair), `upscale_h3_latents` (the upscaler file at its pin, no VAE
  entry), `crop_face_track` (YuNet, pinned for the default repo),
  `interpolate_frames` (RIFE, no pin).
- **facexlib:** `detection_Resnet50_Final.pth` always, and
  `parsing_parsenet.pth` unless `use_parse` is false. Each is listed as
  `{repo: null, url}` while missing from `<facexlib package>/weights`. The
  directory is found with `find_spec`. If facexlib isn't installed, both are
  listed.
- The pin's B′ allow-list is gone. Three commands are exempt with reasons in
  the test: `separate_stems`, `decode_h3_latents` and `paste_face_track`.
  `tests/test_task_file_weights.py` covers the rest.
- **Deviations:** `gated` is always `null` on a file entry (the file HEAD
  doesn't report the gating mode). A file under ~10 MB reads `gb: 0.01`.
  A filename that fails `validate_weight_name` gives no entry. The
  no-heavy-import test checks `facexlib`/`cv2` only, because `import dw`
  already loads torch. `remove_object` and `upscale_h3_latents` keep
  treating any falsy argument as unset.

Docs updated across both stages: TASKS, SERVER, MCP (including the
`download_model`-fetches-whole-repo note), WORKFLOW_GUIDE, DEPENDENCIES,
ARCHITECTURE (the "Task weights" row), and the RELEASING "Unreleased" note
that a bound acknowledgement from a pre-deploy validate gets 409
`downloads` when the workflow has one of these tasks uncached.

## Verification and bounces

| Stage | Bounces | Cause |
|---|---|---|
| A′ (#977) | 1 (tester) | C-F525 step 5: an unedited validate `plan` sent as the acknowledgement was refused, against the plan's "clients that copy the plan verbatim are unaffected". Fixed by `_from_plan`. C-F519–C-F527 and SE-F051 then passed |
| B′ (#978) | 1 (architecture review) | The "Task weights" map row named `argument_or_default` as the one resolution rule, but three of the six new `*_source` functions don't use it. The row was amended and no code changed. All 11 `pending: #978` cases (C-F528–C-F536, M-F157, SE-F052) then passed |

The stage comments name no `usage:` figures, so this record leaves out cost
per stage. The plan estimated $4–6 for each stage.

Some arms couldn't run on lem because the defaults were already cached:
the real-repo arm of C-F530, the uncached pinned big-lama arm of C-F531, and
the default-file arms of C-F533. The offline arm of C-F536 couldn't run
either. The bogus-repo override arms covered each declaration instead.

## Open after close

- `pending: #977` is still on C-F519, C-F520, C-F522, C-F523, C-F525,
  C-F526 and SE-F051. They pass in substance, but as written they use
  argument names or expectations the plan didn't match. The amendment
  is with Don on dkackman/harnest#139.

## Deferred

- zoe_depth, dw_pose and demucs/`separate_stems` (fetched outside the HF
  cache) and text_generation's `system_prompt.from_hub` file are still
  unlisted. Each is documented in TASKS.md. *Trigger:* a field report of an
  unquoted download from one of them.
- `download_model` can't pre-fetch a pinned single file. *Trigger:* an
  offline box that needs one.
- `gated` on file entries. *Trigger:* a gated single-file task source.
