# Named grades

Part of the `cinematography` skill; its *Grades* section says when to read
this. A grade is applied after generation, on the CPU, with no model: an
`apply_lut` `palette` followed by `film_grain`. It changes colour and
texture, never framing or motion.

How a palette works (the `apply_lut` task): 2 to 16 `#rrggbb` colours,
ordered dark to light. Each pixel keeps its own brightness and takes its hue
from the palette at that brightness, so shadows lean toward the first
colour and highlights toward the last. `strength` (0..1) blends the result
with the original. `film_grain` takes `amount` (0..1; keep video at or
under about 0.12), `size` (pixels, 1 or above) and `chroma` (0..1, where 0
is brightness-only grain and 1 is colour noise). Its seed is the run's, so
a rerun reproduces the grain.

Because a palette keeps brightness, no grade here adds contrast. Contrast
comes from the `light` axis in the prompt (`low-key`, `noir`).

## The grades

```json
{
    "noir": {
        "palette": ["#000000", "#1c1c1c", "#4a4a4a", "#8c8c8c", "#d4d4d4", "#ffffff"],
        "strength": 1.0,
        "film_grain": {"amount": 0.08, "size": 1.5, "chroma": 0.0}
    },
    "teal-orange": {
        "palette": ["#06222b", "#16505a", "#6f8a86", "#d49a5e", "#ffdcae"],
        "strength": 0.5,
        "film_grain": {"amount": 0.03, "size": 1.0, "chroma": 0.0}
    },
    "70s-stock": {
        "palette": ["#22261a", "#55502f", "#9a7b4b", "#d6ae74", "#f4e3c0"],
        "strength": 0.6,
        "film_grain": {"amount": 0.08, "size": 2.0, "chroma": 0.2}
    },
    "vhs": {
        "palette": ["#140f2a", "#3b3263", "#7d789c", "#c9bfc6", "#f5efdf"],
        "strength": 0.5,
        "film_grain": {"amount": 0.12, "size": 2.5, "chroma": 0.6}
    },
    "bleach-bypass": {
        "palette": ["#0a0c0d", "#33383a", "#7a7f80", "#c3c6c4", "#f4f4ef"],
        "strength": 0.7,
        "film_grain": {"amount": 0.06, "size": 1.2, "chroma": 0.0}
    }
}
```

| Grade | Looks like | Pairs with |
| --- | --- | --- |
| `noir` | Black and white, fine grain | `light` `noir` or `low-key` |
| `teal-orange` | Teal shadows, warm skin and highlights, the blockbuster look | `light` `golden-hour`, `era` `35mm` or `digital` |
| `70s-stock` | Warm, faded, olive shadows, coarse grain | `era` `16mm` or `35mm`, `lens` `vintage` |
| `vhs` | Washed-out, violet shadows, coarse colour noise | `era` `vhs-dv` |
| `bleach-bypass` | Nearly colourless, cool, gritty | `light` `low-key` |

The palettes are starting points chosen by eye, not matched to any film
stock. Lower `strength` for a lighter touch. To pull a palette from a
reference picture instead (CreatorOS's idea): pick 3-6 colours from its
shadows to its highlights, dark to light.

## Applying one

One clip, one grade. `clip` is an `asset:` or `output:` reference; `look`
and the two grain values come from the grade's entry above. For a whole
episode with one look across shots, use the `series-episodes` skill's
`series-episodes/references/look.md` recipe instead, with the palette from
here.

```json
{
    "id": "graded-clip",
    "variables": {
        "clip": "asset:shot.mp4",
        "look": ["#06222b", "#16505a", "#6f8a86", "#d49a5e", "#ffdcae"]
    },
    "steps": [
        {
            "name": "graded",
            "task": {
                "command": "apply_lut",
                "arguments": {"media": "variable:clip", "palette": "variable:look", "strength": 0.5}
            },
            "result": {"content_type": "video/mp4", "subfolder": "intermediate"}
        },
        {
            "name": "grain",
            "task": {
                "command": "film_grain",
                "arguments": {"media": "previous_result:graded", "amount": 0.03, "size": 1.0, "chroma": 0.0}
            },
            "result": {"content_type": "video/mp4", "subfolder": "final"}
        }
    ]
}
```

`validate_workflow` it first: free, and it refuses a bad palette entry by
name. Both tasks keep the clip's size, frame rate, frame count and audio.
An image works the same way, with `image/png` as the `content_type`.
