# Beats and the beat review

A shot can pass every automated check and still not show what the story
needs: the bear poking the wall instead of shouldering the door. `check-script`
confirms each line is *heard*; this review is how you confirm each beat is
*seen*. It runs on the tools you already have (`get_output_frames`,
`get_output_image`) and adds no engine mechanism.

## Writing a beat

A shot sheet's `beat` names **who, doing what, to what, and where, in a state
one frame proves**. Someone shown a single still, with no script, could say
yes or no to it. `must_not` lists the few things that would make the take
wrong however good it looks: a horse behind the wagon it pulls, extra limbs
or heads, text in frame, the reference portrait itself pasted in.

Four kinds of wording are banned in a beat. They are fine in the generation
prompt, which describes motion and mood; the beat is what you judge against,
and a frame cannot show any of these:

- **Motion over time** ("steadily widening", "rushes toward", "keeps
  rising"). Rewrite: "the bear rushes the door" -> "the door half open, the
  bear's shoulder against it".
- **Intent** ("trying to"). Rewrite: "the girl tries to reach the shelf" ->
  "the girl on tiptoe, one hand on the top shelf's edge".
- **Off-screen facts** (what happened before, who is behind the door,
  what the character knows). Rewrite: "the captain, having heard the news,
  in the cabin" -> "the captain seated at the cabin table, a letter open in
  his hand".
- **Emotion words** ("celebrating", "alarmed"). Rewrite: "the crew
  celebrating" -> "the crew on deck, arms raised, hats in the air".

Most false fails come from a beat a frame could never satisfy, not from a
bad take: in story-forge, 34 of 44 false fails traced back to how the beat
was worded.

## Before anything renders

Cross-check the shot sheet: every shot id in the edit list has a `beat`, and
every `beat` belongs to a shot in the edit list. Do not render while an
edit-list shot has no beat; write it first, or ask.

## The review, before the keep

1. **Pull frames** with `get_output_frames`: **3 evenly spaced** across a
   clip by default; **5 only when the first verdict is split**. For a still,
   the image itself (`get_output_image`).
2. **Describe blind, before the verdict.** Write a plain description of each
   frame (who is there, where, what they hold, what is touching what) before
   re-reading the beat. Where the client can start a fresh subagent, hand it
   the frames without the beat and take its descriptions instead: you wrote
   the beat, so you will see it if you look for it.
3. **A verdict per frame** against `beat` and `must_not`, from the
   description. The shot passes on a **majority** of frames (2 of 3, 3 of 5),
   not unanimity, so one frame mid-motion does not fail a good take.
4. **Exactly one reason per rejected frame**, from this list:
   - `beat_not_shown`: the visible state the beat names is not in frame;
   - `must_not_present`: something on the `must_not` list is;
   - `identity_drift`: a cast member no longer reads as their reference;
   - `text_in_frame`: legible or pseudo text the shot did not ask for;
   - `reference_leak`: the reference portrait (its framing, pose or
     backdrop) shows through.

   A wrong reason teaches the wrong correction, so pick the one that, fixed,
   would have passed the frame.
5. **One concrete correction.** A failing shot yields one change to the next
   attempt's prompt that answers its reason ("the bear's shoulder pressed
   flat against the door, the door half open"), not a list. The rerun counts
   against the shot's three-attempt budget in `SKILL.md`'s "Run and judge";
   after three, stop and ask.

## Audit the auditor

When a human overrules a beat verdict (keeps a take the review failed, or
rejects one it passed), record it in the workspace note
(`set_note("workspace:<name>", ...)`, appending to the note already there
rather than replacing it): the shot, the beat, the verdict, the reason
given, and why it was wrong. The
rule that misfired is then visible the next time the series or script is
picked up, rather than repeated.
