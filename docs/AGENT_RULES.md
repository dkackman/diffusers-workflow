# Agent rules

Hard rules per model family, and for pieces longer than one clip, for an MCP
client that has no dw plugin loaded. Claude Code with the `dw` plugin gets
these from its composition skills; any other client gets them here. Read the
section for the family you are about to run, before you write a prompt or
pick a template.

Each line is one rule. These are the rules a client without the plugin broke
in a measured drill, plus the decision rules for long pieces that the
workflows guide's mechanics don't state. Each rule's number is the skill's,
and a test holds the two copies equal. They don't replace `validate_workflow`:
validate still refuses what the engine can check.

One rule holds for every family: only the user closes a request. Propose a
close (`set_request(propose_close={state, reason})`); never re-propose after
a decline unless the plan or its takes changed. A request's item fields are
in `get_guide("workspaces", section="Requests")`.

## FLUX 3 Klein

The Claude Code skill `flux3-klein` carries the same rules, with their
evidence.

- A short request ("a man walking his dog") runs through
  `templates/flux3/upsample-then-generate`, not `templates/flux3/text-to-image`:
  the upsampler writes the caption, then Klein draws from it. Use
  `text-to-image` only for a caption that is already written in full.
- 16 steps, no guidance and no negative prompt: leave `num_inference_steps` at
  16, and add no `guidance_scale` or `negative_prompt` to a workflow you write.
- Every component carries an explicit `device`. One left unplaced keeps the
  prompt and the noise on the CPU.
- The weights are gated (FLUX Kommunity License, non-commercial): the server
  needs a Hugging Face login that has accepted it.

## MiniMax H3

The Claude Code skill `minimax-h3` carries the same rules, with their evidence.

- H3 wants Context-IR, MiniMax's prompt format, not prose. `get_prompt` shows
  it, and `templates/minimax/enhance-prompt` (or `-with-image`) writes it from
  an idea. Spoken words go in `<d>[English] ...</d>` blocks.
- Dialogue shots (any `<d>` line) drop the turbo LoRA, which digitizes
  voices: `lora_model_name: null`, `num_inference_steps: 12`,
  `video_shift: 12.0`, `audio_shift: 3.0`. Pass these even on a template whose
  defaults keep the LoRA, such as `templates/minimax/dialogue-short`.
- Size a line to its clip: about 2.5–4 written words/s fills it. At 124 frames
  (5.17 s) that is about 13–21 words; at 345 frames (14.4 s), about 36–57.
- A short line is not stretched to fill the clip: H3 holds silence first, then
  speaks fast. Shorten the clip, or split the line at a sentence break into two
  `<d>` blocks, which removes the lead-in.
- Repeat a speaker's voice description verbatim across shots. A reworded
  description is a different voice.
- Two people in frame can lip-sync the wrong one: spell each spoken line in
  `<d>` tags, credit it to its speaker, and frame the speaker face-on at medium
  close-up.
- A three-reference shot can open on the portraits' backdrop for about 1.5 s:
  check its first frames, and say in `subject_definitions` that the portrait's
  backdrop, pose and framing are not reused.
- `num_frames` is `17n + 5`, from 124 to 345, at 24 fps: one clip is 5.17 to
  14.4 s. Canvas and memory limits per reference count are in
  get_guide("recipes", section="MiniMax-H3").

## Longer than one clip

The Claude Code skill `long-form` carries the same rules, with their evidence.

- Prefer cuts. A cut resets drift; chain only an action, line or camera move
  that must cross the clip ceiling unbroken.
- An H3 chain drifts past about 35 s (a `shot: dialogue` walk was clean to
  35 s and changed face from 45 s). Offer chains of 35 s or less, joined by
  cuts, instead of a longer single take.
- Under cut scenes use one score bed: `templates/assemble-and-score`. If you
  swap it for a mix you wrote, say so.
- A score past about 90 s is `templates/minimax/music-long` sections: the
  lyric, not the ceiling, sets a Music 3 track's length.
- A chain you mean to keep runs with `check: true, retries: 1`. Its worst case
  is N(1 + `retries`) segment renders when it generates its audio; quote
  `plan.estimate` from `validate_workflow`.
- On a 24 GB Mac, `templates/minimax/chained-segments` at its defaults is
  refused; lower the canvas to 768x448.
- Judge the result: listen with `get_output_audio`, and check the seams with
  `assess_output(probe="analyze_seams")` and `get_output_frames(seams=true)`.
  The mechanics are in get_guide("workflows", section="Chained Video Generation").
