# Task layer: one registration pattern, one coercion, shared image ops (#692)

Written by model `claude-opus-5-5` via provider `anthropic`, at close-out
2026-10-08, from plan v2 on #692 and the stage threads (#773, #774, #775).
Plan v1 approved by Don 2026-10-08 with every default (D1–D3), recorded as
v2 with no scope change. Built and verified on mini-ai (mps) the same day.

## The ask

The 2026-10-07 develop review (structural item 2) found the ~9 new tasks in
`dw/tasks/` had diverged on four conventions:

- **Registration.** Some tasks self-registered, others went through
  `_handle_*` shims in `task.py`. Each task's rules lived in four tables
  kept by hand, apart from the registry: `TASK_ARGUMENT_DOMAINS`,
  `TASK_ARGUMENT_CHOICES`, the `extra` dict in
  `task_domains.task_argument_errors` and `locations.TASK_MEDIA_ARGUMENTS`.
  Nothing pinned `extra` or the media table to the registry, so a missing
  entry silently dropped a validate-time check. For a media argument, the
  dropped check was path confinement.
- **Coercion.** There were five or more numeric idioms that disagreed.
  `plan_cuts` accepted `"3.0"` and `window_video` refused `3.0`.
  `audio_utils.coerce_number` let `True` through. `beats` truncated
  `"3.5"` to 3.
- **Colour and alpha.** Rec.709 luma weights were defined three times, and
  alpha split/join three times. `lut` imported `finish._split_alpha`, and
  `finish.film_grain` imported `task._per_frame` at call time.
- **Baked model knowledge.** `crop_face_track` hard-coded LTX's 8n+1 and
  the multiple of 32.

## Verdict

**Build smaller, in three stages (A → B → C).** The value was to the agents
that add tasks, plus closing the silent-drop bug class. The cheaper
alternative was pin tests only. It would have caught the silent drop but
left the touch points and the coercion idioms in place. Stage A does that
alternative and also removes the hand-kept tables.

## Design corrections found against the issue

1. There were 12 cross-argument checkers, not 11: `attribute_voices`'s
   `voices_errors` was wired straight into `dw/validation.py`, outside
   `extra`.
2. Adding a task touched seven places, not 4–5.
3. "Touches the task module and the registry decorator only" can't be
   reached without auto-importing every `dw/tasks` module, which would load
   cv2 and torch at registration. The documented rule (D2) is: the module
   with its decorator, one import line in `task.py`, and a `docs/TASKS.md`
   section.
4. The `"3"` / `3.0` / `"3.0"` → 3 acceptance applies to whole-number
   arguments. Float arguments already accepted all three.
5. `LUMA_WEIGHTS` stays float64 and importable from `dw.tasks.lut`
   (`tests/test_lut.py` checks a `< 1e-5` bound against it).

## What was built

### Stage A (#773): the registry owns task metadata

- `register_command` gained `domains`, `choices`, `static_check` and
  `media_arguments`. A `whole_numbers` argument came with stage B.
- `TASK_ARGUMENT_DOMAINS`, `TASK_ARGUMENT_CHOICES` and
  `TASK_MEDIA_ARGUMENTS` are now lazy, read-only `RegistryTable` views
  under their old names and modules. `TASK_STATIC_CHECKS` and
  `TASK_WHOLE_NUMBER_ARGUMENTS` were added the same way.
- Every entry moved onto its registration, `voices_errors` included, and
  `extra` was deleted. Each migrated command was checked against a
  snapshot of the old dicts.
- Pin tests are in `tests/test_task_registry_rules.py`:
  - every derived table is the registry's view;
  - every static check is registered to exactly one command;
  - every media argument is a parameter of its command.
- Docs: *Adding a task* in `docs/TASKS.md`, the `docs/ARCHITECTURE.md` rows
  that now point at the decorator, and the `docs/SECURITY.md` media-argument
  wording.
- **Deviation:** `attribute_voices`'s separate `voices` validation `Check`
  was folded into the shared task-argument check.

### Stage B (#774): one coercion

- `whole_number` and `real_number` in `dw/task_domains.py` share one rule
  (`number_problem`). They are the only numeric coercion in `dw/tasks/`.
  They replaced `cuts._number`/`_integer`, `audio_utils.coerce_number`,
  `fit._whole` and its copies in `windows` and `trim`, the `beats` and
  `loop_bed` `_coerce_arguments`, `video_utils`'s `loop_frames` and
  `_positive_int`, and `image_utils._grid_whole_number`.
- `as_number` and `domain_violation` use the same rule, so validate and run
  agree.
- `Task.run` calls `coerce_arguments` before dispatch, so a handler gets a
  number, never a string. This was the bounce fix (see Bounces).
- **Deviation:** a new `finite` domain for arguments whose only rule was
  "is a number", such as `grade.exposure` and several audio dB/LUFS
  arguments. No range changed.
- Tests: `tests/test_numeric_coercion.py`, built from the registry. It
  covers 38 whole-number and 125 real arguments, and validate/run agreement
  over 13 values each.
- Per D1, the tightenings (`True` refused, `beats` `"22050.5"` refused) are
  bug fixes: no `breaking-change` label and no release note.

### Stage C (#775): shared image ops, de-baked face crop

- `dw/tasks/image_ops.py` holds `LUMA_WEIGHTS` (float64), `luma`,
  `split_alpha`, `join_alpha` and `per_frame`. `grade`, `finish` and `lut`
  use it. `lut.LUMA_WEIGHTS` stays as a re-export, and both private
  cross-module imports are gone.
- Outputs are byte-identical to fixtures captured before the refactor
  (`tests/fixtures/image_ops_outputs.npz`). The float32/float64 risk didn't
  materialise.
- `WHISPER_DEFAULT_MODEL` now lives in `audio_transcription.py`.
- `transcript_problem` moved into `task_domains`, which fixes the inverted
  import.
- `locations.load_json_record` confines the path, then checks the extension
  and the size, then parses. It backs `fit._read_fit` and
  `face_track._read_track`.
- `crop_face_track` takes `modulus` (8), `remainder` (1) and `multiple`
  (32). `padding_to_grid` rounds through `variable_constraints.aligned`, the
  grid's owner. `workflows/templates/ltx2/face-repair.json` declares LTX's
  `8/1/32` explicitly.
- **Deviation (bounce fix):** `fetch_image` gained `keep_alpha`, which the
  image-task loader `task._load_media` sets. Pipeline inputs still load as
  RGB.

## Bounces

| Stage | Bounces | What |
|---|---|---|
| A (#773) | 1 tester | SE-F045: `restore_to_source.fit` and `paste_face_track.track` weren't declared as media arguments, so a `../` or `/etc/hosts` record path validated. The fix declared them, added them to `LOCAL_ONLY_TASK_ARGUMENTS`, and a sweep found and declared `paste_face_track.clip`/`.repaired`, `crop_face_track.clip` and `stabilize_video.clip`. This was exactly the silent-drop class the feature set out to close. The old hand-kept table had missed them too. |
| B (#774) | 1 tester | C-F356: `grade.exposure` had no domain, so validate let `"0.5"`/`true`/`"nan"` through and the handler raised `TypeError`. Fixed with the `finite` domain and coercion at dispatch. |
| C (#775) | 1 architecture review, 1 tester | Review: `padding_to_grid` did grid arithmetic that `variable_constraints.aligned` owns, and the map lacked rows for `image_ops` and `load_json_record`. Tester, C-F361: an RGBA file read from `asset:`/`output:` came back opaque because `load_image` flattened it to RGB before the command ran. The flattening predates the stage, but the stage's acceptance needed it fixed. |

Cost per stage is not recorded: the stage comments name no `usage:` figures.

## Deferred, and why

- **Moving per-task rules out of `task_domains.py`.** 22 test files and 8
  `dw/` modules import from it, so the move would need a shim layer and
  would help no consumer. *Comes back if* the file passes about 2,000
  lines, or a rule's placement causes a real import cycle.
- **Converting the `_handle_*` shims to self-registration.** The decorator
  arguments work the same from a shim. New tasks self-register. *Comes back
  if* the shims get in the way of a later change.
- **The other cross-task private imports.** These are
  `text_generation._DEFAULT_VISION_MODEL`, `upscale._resolve_model_path`,
  `video_utils._frames_of` and `voice_attribution._EMBEDDER_SAMPLE_RATE`.
  They are filed as their own `consolidation` idea, #784 (D3).
- **Variable coercion** (`variables.py`, `variable_constraints.py`) is a
  separate contract (#338). It was a non-goal.

## Notes for later

- `get_gallery_metadata` returns `media: null` for a PNG, so the tester
  could judge alpha only visually and by byte size.
- C-F352 arms e, f and g cited behaviour no case defines. The tester filed
  an amendment on the harness repo.
