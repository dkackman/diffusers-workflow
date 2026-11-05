# The persisted shot document

Part of the `script-to-video` skill; step 2 says when to read this. The shot
list is not kept in the chat. It is saved in the prompt library, so a
revision recompiles only the shots it changed and a later session picks the
project up where it stopped.

## Where it lives

Two kinds of prompt, both under a project prefix (the library is shared
across workspaces, so the prefix is what keeps projects apart):

- `<project>/shots`: the whole document, JSON text, an array of shot objects.
  `save_prompt(name, ...)` it; `get_prompt(name)` reads it back;
  `list_prompts()` finds what a project already holds.
- `<project>/shot-NN`: one compiled prompt per shot, `NN` two digits from
  `01`, in source order. A workflow argument names it as
  `prompt:<project>/shot-NN`.

`<project>` is one name segment matching `[\w][\w.-]*`: letters, digits
and `_`, then `.` or `-` as well - no spaces and no further `/`. A prompt
name holds at most one `/`, and the project prefix uses it, so
`gilded_echoes/shots` is a name and `gilded echoes/shots` or
`series/ep1/shots` is refused.

`<project>/shots` is the source of truth; each `<project>/shot-NN` is what
it compiles to for that shot's family. Never edit a compiled prompt by
hand without writing the change back into the shot object.

## The shot object

| Key | Holds |
| --- | --- |
| `id` | `01`, `02`, ...: the `NN` of its compiled prompt |
| `cast` | the characters in frame, by their `asset:cast/<name>` names |
| `location` | a place named once in the document and reused |
| `opening_state` | what is on screen at frame one |
| `ending_state` | the same at the last frame |
| `beat` | one visible state a single frame can show |
| `must_not` | the few things to rule out |
| `join` | `continue`, `cut`, `cutaway` or `lead_in` |
| `shot_size` | the cinematography axis of that name, or `null` |
| `move` | likewise |
| `lens` | likewise |
| `light` | likewise |
| `duration` | seconds the shot should run (a number), or `null` |

Every shot carries all five cinematography keys - `shot_size`, `move`,
`lens`, `light` and `duration` - so a reader never has to guess whether a
key was forgotten. `shot_size`, `move`, `lens` and `light` take an id from
`cinematography/references/vocabulary.md` and nothing else, or `null`: an
axis set to `null` is left to the model's own choice, not a default. Their
meaning, and the caption words each family gets for them, are that skill's.
`duration` is a number of seconds, not a vocabulary id, or `null` for the
family's default length. It is snapped to the family's own grid when the
prompt is compiled (H3's is in `dw:minimax-h3`), and the document keeps what
the script asked for.

One shot object:

```json
{
  "id": "01",
  "cast": ["asset:cast/priya.png"],
  "location": "the lighthouse kitchen",
  "opening_state": "Priya at the window, back to the room, kettle steaming",
  "ending_state": "Priya turned to face the door, mug in both hands",
  "beat": "she hears the knock and turns",
  "must_not": ["text in frame", "a second figure"],
  "join": "cut",
  "shot_size": "medium",
  "move": "push-in",
  "lens": null,
  "light": "practical",
  "duration": 4.0
}
```

`<project>/shots` is an array of them, in source order. A two-shot
document, saved with `save_prompt("lighthouse/shots", ...)`:

```json
[
  {"id": "01", "cast": ["asset:cast/priya.png"],
   "location": "the lighthouse kitchen",
   "opening_state": "Priya at the window, kettle steaming",
   "ending_state": "Priya facing the door, mug in hand",
   "beat": "she hears the knock and turns", "must_not": ["text in frame"],
   "join": "cut", "shot_size": "medium", "move": "push-in", "lens": null,
   "light": "practical", "duration": 4.0},
  {"id": "02", "cast": ["asset:cast/priya.png"],
   "location": "the lighthouse kitchen",
   "opening_state": "Priya facing the door, mug in hand",
   "ending_state": "Priya's hand on the latch",
   "beat": "she crosses to the door", "must_not": ["text in frame"],
   "join": "continue", "shot_size": null, "move": "tracking",
   "lens": "normal", "light": "practical", "duration": null}
]
```

## Scenes: length and continuity

A scene is one place and one continuous moment, and may hold several shots.
A piece longer than one clip is planned per scene before any shot is
compiled; the `long-form` skill is the playbook for that plan. Save the plan
beside the shots as `<project>/scenes`, an array of scene objects in source
order:

| Key | Holds |
| --- | --- |
| `id` | `s1`, `s2`, ... |
| `shots` | the `id`s of the shots in this scene, in order |
| `length_seconds` | the scene's target length from the story (a number), not from the model |
| `continuity` | `one_clip`, `cuts`, `chain` or `bridge`: how the scene is rendered |

`continuity` is the rendering shape in `long-form`'s *Per scene* table:

- `one_clip`: the scene fits one clip of its family (14.4 s on H3, about
  17 s on LTX-2.5 at 768). It holds one shot.
- `cuts`: the scene changes place, time or angle, or a beat allows a cut.
  Each shot renders fresh and the cut joins them; the shots' own `join`
  keys say how.
- `chain`: one action, line or camera move must cross the ceiling unbroken.
  The scene is one shot whose `duration` is `length_seconds`, rendered as a
  chain; its `move` sets the chain's `shot` preset (`long-form`, *Per
  chain*).
- `bridge`: a cut must land on a fixed frame (a first-and-last-frame or
  last-frame-only shot).

Prefer `cuts`: a cut resets drift and costs nothing at the seam. Choose
`chain` only for what must be unbroken. The sum of the scenes'
`length_seconds` is the piece's length, and the plan's cost is quoted from
it before anything renders.

```json
[
  {"id": "s1", "shots": ["01", "02"], "length_seconds": 9.0,
   "continuity": "cuts"},
  {"id": "s2", "shots": ["03"], "length_seconds": 40.0,
   "continuity": "chain"}
]
```

A changed `length_seconds` or `continuity` changes every shot in that scene.

## Recompiling only what changed

1. Edit the shot object, then `save_prompt` the whole `<project>/shots`.
2. Compile only the changed shot, per `cinematography`, and `save_prompt` its
   `<project>/shot-NN` alone.
3. `rerun_job` that shot's job, or `run_workflow` for that shot alone, and
   quote its cost first. Shots that did not change are not rendered again.

A changed `ending_state` also changes the next shot's `opening_state`: the
shot that has it as its source is a changed shot too. A changed `join`
changes the shots on both sides of the cut (`references/joins.md`).

## Credit

The shape borrows ideas from seriesmith (MIT: a model-free shot list
compiled per target) and OpenReel (MIT: one unified style block and the
actor description repeated verbatim in every prompt). No text was copied.
