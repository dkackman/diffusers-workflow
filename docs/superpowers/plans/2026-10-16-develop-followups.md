# Develop follow-ups after FLUX 3 Klein (#930), 2026-10-15

What is still owed on develop once #930 is merged. Order is by risk, then by what unblocks what.
Owner `don` means the step needs a ruling or a hand on lem; everything else an agent can do.

## 1. Security: the remaining `generate()` kwargs splats (agent, then don)

#930 closed the `custom_generate` + `trust_remote_code` hole in `text_generation`. The same
pattern is still open elsewhere, and has been in every release since a0e8cb15.

- [ ] `dw/tasks/speech_generation.py:190-219` - `forward_params` and `generate_kwargs` reach the
      transformers pipeline unfiltered. Reuse the text_generation filter (`custom_generate`,
      `trust_remote_code` gated on `dw/trust.py`; structural keys and non-scalars refused).
      Test with an untrusted workflow that sets each key.
- [ ] `dw/tasks/image_utils.py:84-124` - `repo_kwargs` / `call_kwargs` / `kwargs` splat into
      `controlnet_aux` `from_pretrained` and the detector call. Audit which keys can load code or
      reach the network (`trust_remote_code`, `cache_dir`, `filename`, a repo id) and allowlist.
- [ ] Sweep for any other `**kwargs` into `from_pretrained`/`generate`/`pipeline(...)` in
      `dw/tasks/` (grep), record the result in docs/SECURITY.md.
- [ ] **don:** GHSA for the released exposure (0.x through 0.14.0)? Fix ships in 0.14.1 or 0.15.0.

## 2. Finish Klein on lem (don deploys, agent runs)

- [ ] **don:** deploy develop to lem (diffusers must stay at or after 7ecb1e1b; restart dw-serve
      after any `update_diffusers`).
- [ ] Task 9: warm `cost` blocks for the five `workflows/templates/flux3/` templates and the
      model entry, from one GPU run each. Check the compact listing stays under 11_400.
- [ ] diffusers main has moved past 7ecb1e1b: the upsampler step (PR #15058) appears merged
      (d42ebf9c touches `modular_pipelines/flux3/prompt_upsampler.py`). The audit and the family
      README still call it open. Check before pinning, and decide whether the builtin upsampler
      should move to it.
- [ ] Fold in the parked doc fix: docs/TASKS.md `generate_kwargs` row still says "merged last,
      overrides the settings above" though `max_new_tokens`/`max_length` are now reserved.
- [ ] Task 11: cold drill with the `flux3-klein` plugin skill, plus a control without it.
- [ ] Delete `.superpowers/sdd/2026-10-15-flux3-klein-catalog/` once 9 and 11 are done.

## 3. Engine gaps found during Klein (agent)

- [ ] `validate_workflow` `downloads_required` is empty for uncached modular-pipeline and task
      repos, then over-counts `variants/` (40.3 GB for Klein). Size from the files the loader
      will actually fetch.
- [ ] `update_diffusers` leaves the running server on a half-old diffusers import (validation
      raises RuntimeError) until dw.serve restarts. Same gap as #663. Either restart the worker
      and refuse admission until it does, or say so in the tool result. **don** picks which.

## 4. #928 roles (don)

- [ ] gilded-echoes' outputs stay `final` until its assembly run consumes them (Don chose to
      wait). Post the mini-ai diagnosis on #928 so the issue says why it looks unfixed.
- [ ] Decide on the `set_role` override (MCP tool + gallery toggle, sidecar beside notes) for
      0.15.0, or close #928 after the assembly run.

## 5. Release housekeeping (don)

- [ ] v0.14.0's GitHub release body is the auto-generated stub. Paste the notes from
      docs/RELEASING.md (`gh release edit v0.14.0 --notes-file ...`).
- [ ] Deploy 0.14.0 to lem if it is not already there (mini-ai was deployed 21:19).

## 6. Parked guided-editing proposal

- `feat/guided-editing` holds `docs/proposals/guided-editing.md` (options only). Pick it up
  when wanted; nothing on develop depends on it.

## 7. Parked, no action unless asked

- `generate_kwargs` refuses lists of lists, so `bad_words_ids` cannot be set.
- Klein README thinking-mode line could cite the pup card.
- Template summary trim candidates (about 40 tokens total).
