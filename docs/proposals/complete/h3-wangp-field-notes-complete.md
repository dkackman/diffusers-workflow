# H3 rules from WanGP's field notes (#985)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out,
2026-10-25). Plan v2 (v1 approved by Don on 2026-10-25, v2 folds in his
approval) was built in two stages, both verified on lem: stage A, #988
(probes, doc only), and stage B, #989 (skill, guide and enhancer), last
shipped at `develop` @ `6997c2c8`.

## The idea

WanGP v18.21's H3 handler (`models/minimax_h3/minimax_h3_handler.py`)
carries four field observations, which are WanGP's and not MiniMax's. The
issue asked for each to be checked on lem before it went into
`plugins/dw/skills/minimax-h3`:

1. lip-sync to a supplied soundtrack works better with the words left out
   of `<d>`;
2. a chained window can snap onto an end image, and discarding about 5 tail
   frames hides it;
3. `input_video_strength` of about 0.5 gives softer keyframes;
4. FL2VA accepts a soundtrack, so a locked first frame might lip-sync
   without the Ref2VA route.

## Verdict and scope (Don, plan v1)

**Build smaller:** probe claims 1 and 4, drop 2, defer 3.

- **Claim 2, dropped.** dw chains have no per-window end image: `last_frame`
  carries a segment's real last frame as the next segment's *first* frame,
  and trimming is head-only (`trim_frames`), so the snap can't happen.
  Reopen on a chain that pins per-segment end images, or on a field report
  of a snap at a dw chain seam.
- **Claim 3, deferred (Q2).** No frame-conditioning strength can be set from
  a workflow: diffusers hard-codes `MiniMaxH3ModularPipeline.keyframe_noise_aug
  = 0.999`, which dw copies (`h3_hold_steps.py`), and the mapping from
  WanGP's `input_video_strength` is unknown. Reopen on a field report of
  hard snaps into an end image or a guide, or when diffusers exposes the
  property.
- **Q1:** about 90 GPU minutes on lem for the probes. **Q3:** the
  `h3_context_ir` enhancer changes if claim 1 is confirmed. **Q4:**
  `staging.md`'s two-singer `<d>` rule is left alone. **Q5:** P1 includes a
  spoken track as well as a sung one.
- **Don's addition:** every probe job runs one pinned configuration, int8
  `weights_dtype` and one turbo-LoRA state, stated in each job's note and in
  the audit doc. #984 had shown int4 + turbo can change the voice, so a
  lip-sync verdict must not secretly be a verdict on quantization.

## What was built

### Stage A, #988: the probes

Workspace `qa-h3-wangp-probes` on lem: int8, turbo LoRA on, 960x544, 124
frames, seeds 1001 and 2002, `hold_audio` on every shot. The record is
`docs/proposals/audits/2026-10-h3-wangp-probes.md`, with the per-arm table
in the #985 results comment.

- **Claim 1: refuted.** With one person in frame, under a held track, the
  mouth opened and shut at the same instants with the words in `<d>` and
  without them, on a sung slice and on a spoken line. H3 didn't voice its
  own words, since the hold replaces the audio.
- **Claim 4: confirmed.** `templates/minimax/image-to-video`'s fl2va step
  with a first frame and `hold_audio` kept frame 0 pixel for pixel, and the
  mouth followed the song on both seeds, at about a fifth of the denoise of
  #852's Ref2VA `[keyframe completion + audio reuse]` route (49.7 s against
  about 233 s). The Ref2VA route keeps only the framing.
- `measure_sync`: lag 0.0 on every arm; confidence 0.979 sung, 0.756 spoken.

### Stage B, #989: the rules

- `plugins/dw/skills/minimax-h3/SKILL.md`: "FL2VA takes no audio" is gone.
  FL2VA with the song held by `hold_audio` is the locked-frame lip-sync
  route; Ref2VA keyframe completion with the track held is the fallback.
  Pinned by `test_the_locked_frame_lip_sync_recipe_names_real_task_types`
  in `tests/test_plugin_skills.py`. SKILL.md is 11,942 B of 12,288.
- `references/dialogue.md`: claim 1 listed as tested and refuted.
- `docs/WORKFLOW_GUIDE.md` *H3: generating to a held soundtrack*: one
  paragraph naming hold on `fl2va` as the locked-frame route, with the #985
  evidence.
- `dw/workflows/h3_context_ir.json` (the `h3` enhancer), system prompt
  only: the `integrated_multimodal_description` field names `<d>` blocks,
  and the `<d>` rule names the language tag and gives a worked `<d>`
  against quoted-prose example. Pinned by
  `tests/test_h3_context_ir.py::test_spoken_lines_are_written_as_d_blocks_not_quoted_prose`.
  This wasn't a claim-1 rule: the scoping case M-F162 showed the enhancer
  had already stopped writing `<d>` at all (see the bounces).
- Not changed, as a refuted claim 1 calls for: no SKILL.md pointer, no
  `workflows/templates/minimax/README.md` rewording, and no held-track rule
  in the enhancer. Nothing for claims 2 or 3.

## Bounces

- **#988: 1.** M-F160 failed: the sung `measure_sync` read confidence 0.17
  against a 118.751 s reference. The `slice` step had written the whole song
  (#990, fixed in f9b96b20). The shots were unaffected, because both of the
  model's audio inputs crop to the shot length, so the 8 sung shots were
  re-measured on cpu (job `85fb684df2ef`), not re-generated.
- **#989: 1.** M-F162 failed: none of nine `h3` enhancements wrote a `<d>`
  block for ordinary dialogue or a generated song. This predated the stage:
  the rule was stated only in the abstract, far from the field definitions,
  and Qwen3-4B wrote supplied lines as quoted prose. The system-prompt fix
  above passed all nine, and a silent idea still got no `<d>`.

## Deferred and left open

- Claim 2 and claim 3, with the reopen triggers above.
- The `h3` enhancer extends a supplied lyric with invented lines (busker
  idea). The original prompt did the same, and a stronger "don't extend"
  clause made it worse. That is a separate fidelity fix if it matters.
- The tester noticed the busker enhancement numbers its shots from
  `[Shot 2]` and puts a shot at the clip's last instant. It looks older
  than this feature and is not filed.

## Cost

The stage comments carry no `usage:` figures, so cost is not recorded here.
Stage A needed three P4 reruns after host-RAM OOMs (two workers at once).
