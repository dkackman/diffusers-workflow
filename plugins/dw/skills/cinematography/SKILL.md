---
name: cinematography
description: Use when a dw MCP server is connected and the user directs the camera or the look - a shot size, a camera move (dolly, crane, orbit, whip pan, crash zoom), a lens, lighting, a film era, a pace, or a named colour grade (noir, teal-orange, 70s stock, VHS, bleach bypass) - for an H3, LTX-2.5, Kandinsky 6 or FLUX 3 Klein generation. Gives a closed vocabulary and the one sentence each value becomes in that family's own caption format, and the grades as apply_lut palettes plus film_grain.
---

# Cinematography on a dw server

A director's words for the camera and the look, as a closed set of values,
and what each value becomes in each family's caption. The families already
have a slot for this: LTX-2.5's trained caption must state a shot type, a
camera motion and a viewpoint; Kandinsky 6 wants exactly one camera
sentence; H3 writes camera moves inside its Context-IR shots; FLUX 3 Klein
frames a still. This skill fills that slot the same way every time, so a
shot list can say `move: dolly-in` and mean the same thing on every family.

It does not pick the template, the size or the cost. The family's own skill
does that, and its rules win.

## When to use it

- The user names a camera or look in film words ("slow push-in", "low-key",
  "shot on 16mm", "teal and orange"), or a shot list needs them per shot.
- A piece with several shots should read as one film: fix the values once
  and compile each shot from them.
- The user wants a named grade on a finished clip or image.

## The axes

Six axes, each a closed set; a shot sets at most one value per axis.
Before choosing, read `references/vocabulary.md`, which defines every id.

| Axis | Values |
| --- | --- |
| `shot_size` | `extreme-wide` `wide` `full` `medium` `medium-close` `close-up` `extreme-close-up` |
| `move` | `static` `dolly-in` `dolly-out` `truck-left` `truck-right` `crane-up` `crane-down` `orbit` `tracking` `whip-pan` `push-in` `handheld` `crash-zoom` |
| `lens` | `wide` `normal` `long` `anamorphic` `vintage` |
| `light` | `high-key` `low-key` `golden-hour` `practical` `noir` |
| `era` | `8mm` `16mm` `35mm` `vhs-dv` `digital` |
| `tempo` | `still` `measured` `brisk` `frantic` |

Each value is a decision, not a slider: pick the one that fits, or leave the
axis unset and let the model choose. Map the user's words onto the ids
("arc shot" is `orbit`, "jib up" is `crane-up`, "slow zoom toward her" is
`push-in`) and say which you picked. Name the look by its effect, never by
a camera body or lens brand: a brand only works if the model learned it,
and nothing here has measured that. Genre is not an axis; keep the user's
genre words in the prompt and state the values it implies.

## Compile per family

Each family's table turns every `shot_size` and `move` into words in that
family's caption format, and says where `lens`, `light`, `era` and `tempo`
go. Read the one for the family the shot runs on, alongside that family's
skill:

| Family skill | Table |
| --- | --- |
| `minimax-h3` | `references/compile-h3.md` - Context-IR, move type with amplitude and speed |
| `ltx-2.5` | `references/compile-ltx-2.5.md` - shot type, camera motion and viewpoint in one sentence; name what comes into view |
| `kandinsky-6` | `references/compile-kandinsky-6.md` - shot size up front, exactly one camera sentence |
| `flux3-klein` | `references/compile-flux3-klein.md` - a still: a move becomes framing |

The compiled words go into the prompt the family's own rules build; they do
not replace it. The stored prompts (`list_prompts`, then `get_prompt`) show
each family's format with a camera already in it - `ltx2/`, `kandinsky6/`,
`minimax/fox_dawn_context_ir` - so read one before writing.

## A hint, not a promise

Every value here is a hint, not a promise. The models were trained on
captions that describe camera work; none was trained on this list, and dw
has not measured which named moves each family honours (an LTX clip asked
to stay static has pushed in after five seconds). So:

1. Describe the move in the caption in the family's own words (the table
   does that) rather than naming the id alone.
2. After the run, check it: `get_output_frames(count=12)` and compare the
   first and last frames. A dolly-in should end tighter than it began, a
   crane-up higher, an orbit at a different angle, a static shot unchanged.
3. Tell the user what the frames show. A move that did not land is
   reported as missed, not as done. A reseed, a stronger wording ("with
   large amplitude"), or on H3 a video reference that carries the camera
   (`composable-references`) are the next tries.

## Grades

A grade is colour and grain applied after generation, on the CPU: an
`apply_lut` `palette` (2-16 `#rrggbb` colours, dark to light) and then
`film_grain` (`amount`, `size`, `chroma`). Five are named - `noir`,
`teal-orange`, `70s-stock`, `vhs`, `bleach-bypass` - with their palettes,
strengths, grain settings and a two-step workflow in `references/grades.md`.

Pick a grade by what the user named, else by the `light` and `era` already
chosen (the table there pairs them). `era` is said in the prompt and a grade
is applied afterwards; they agree but are separate. A palette keeps each
pixel's brightness, so a grade cannot add contrast - that comes from
`light`. For a series, one grade on every shot: the `series-episodes` skill's
look recipe takes the palette from here.

## Run and judge

1. Generate with the family's skill and its template (for example
   `templates/ltx2/text-to-video`, `templates/minimax/video-with-audio`,
   `templates/kandinsky6/text-to-video`, `templates/flux3/text-to-image`),
   with the compiled words in its prompt. Its validate, cost quote and
   `acknowledged_cost` rules apply unchanged.
2. Check the camera with `get_output_frames` as above; check a still with
   `get_output_image` against the framing asked for.
3. For a grade, `validate_workflow` the grade workflow (free; it refuses a
   bad palette entry by name), then run it on the `output:` reference of the
   generated file. It needs no GPU and no downloads.
4. A saving step carries a `subfolder`: the graded file is `final`, the
   ungraded step before it `intermediate`, so `list_gallery(subfolder="final")`
   lists only deliverables. Keep both in anything you compose, then give the
   user the gallery `url`.

## Not in scope

- Audio-only families: `minimax-music3` has no camera, and nothing here
  applies to it. A soundtrack's mood is the music skill's.
- Choosing a template or a family, sizes, frame counts and cost: the family
  skill's.
- Recovering the camera from a finished clip automatically, or tuning a
  model to follow moves: not built. Judge by the frames.

## Sources

The axes follow Higgsfield's user-facing preset words, without its
camera-body presets. Ideas, not code, from: Aither (MIT) - enum-sized axes
and "a decision, not a slider"; CreatorOS (Apache-2.0) - a preset list and
pulling a palette from a photo; OpenField (MIT) - director axes and an era
control. The LTX compile rule is from the LTX-2 camera-control LoRA cards
on Hugging Face; H3's from MiniMax's base prompt-writing guide; Kandinsky's
from Kandinsky Lab's beautifier prompts. Audit:
`docs/proposals/audits/2026-11-higgsfield/README.md`.
