# dw plugin

Skills that teach an agent to compose a model family's workflows on a
[diffusers-workflow](https://github.com/dkackman/diffusers-workflow) server, and
the `dw` MCP server that gives it the tools. A skill's first move is
`get_server_info`. Six skills cover one model family each; the last five are
cross-cutting shapes that orchestrate them.

`dw setup` installs the plugin in Claude Code
([docs/INSTALL.md](../../docs/INSTALL.md)). The plugin's `dw` server
(`.mcp.json`) launches `${HOME}/.local/share/diffusers-workflow/bin/dw-mcp`, a
shim `dw setup` writes to the `dw-mcp` beside the `dw` that ran it, whether or
not Claude Code is installed (Claude Code and Cowork both start it); on a
machine where setup never ran, that server fails to start, and a `dw` registration you added by hand
(see [docs/MCP.md](../../docs/MCP.md#clients)) is what serves the tools. What
loads where:

| Where | Skills | The plugin's `dw` server |
| --- | --- | --- |
| Claude Code | yes | yes |
| Cowork | yes | yes |
| Claude Desktop chat | yes | no: Desktop chat uses the `dw` entry `dw setup` writes to `claude_desktop_config.json` |

In Claude Desktop the plugin is added by hand, which also makes it available to
Cowork: open Customize > Plugins (or Settings > Plugins), choose Add
marketplace, enter `dkackman/diffusers-workflow`, and install `dw`.

By hand in Claude Code, from inside a session:

```
/plugin marketplace add dkackman/diffusers-workflow
/plugin install dw@diffusers-workflow
```

Update it by refreshing the marketplace and reinstalling, then `/reload-plugins`
(or start a new session). Claude Code has no `/plugin update`; from a shell,
`claude plugin update dw@diffusers-workflow` does the same after the refresh:

```
/plugin marketplace update diffusers-workflow
/plugin uninstall dw@diffusers-workflow
/plugin install dw@diffusers-workflow
```

Each skill that starts at `get_server_info` states the dw release it was written
for (`scripts/release.sh` rewrites the line with `plugin.json`). The server's
`get_server_info` answer carries `plugin.current_from`, the oldest release that
counts as current, and these commands, so a session whose skill states an older
release, or none, tells the user once.

| Skill | Teaches |
| ----- | ------- |
| `minimax-h3` | MiniMax H3 video with audio: one take, a fix to one region of a take, a longer take by chain, a piece with cuts, identity and voice references, music; the frame and canvas rules; what a run costs. Prompts come from MiniMax's own `h3-prompt-writing` skill or the guides on the model card. |
| `minimax-music3` | MiniMax Music 3: a song, an instrumental, a score under a film, the soundtrack a music video is cut to; the ceiling semantics, the tag vocabulary, the caps and the 44.1 kHz output. Captions come from MiniMax's own `music-caption-rewriter` skill. |
| `ltx-2.5` | LTX-2.5 video with a soundtrack: a single clip, first-frame and keyframe conditioning, the three-move two-stage flow, the IC-LoRA upscale, extend and chain; the distilled schedule and the frame and size rules. Prompts follow the trained-caption spec that ships inside diffusers. |
| `kandinsky-6` | Kandinsky 6.0 video with a soundtrack: a five-second clip from text or from a picture on the 3B Lite-distill checkpoint; the distilled schedule, the size and frame rules the pipeline's own defaults get wrong, and why Apple Silicon is parked. Prompts follow Kandinsky Lab's own beautifier system prompts. |
| `flux3-klein` | FLUX 3 Image Klein images: a picture from a short request through BFL's prompt upsampler, from a full caption, an edit of one photo, several references combined, or a layout placed by boxes; the 16-step distilled schedule with no guidance, the 16-pixel grid, the 10-reference training limit the pipeline does not enforce, and the gated license. Prompts come from BFL's upsampler builtins or their prompting guide. |
| `minimax-h3-image` | MiniMax-H3-Image stills: a cast still or keyframe for an H3 clip, drawn at twice the video canvas and passed on as `output:`, a large picture from a long verbatim description, or an edit of several references named `<Picture 1>`; when the host qualifies (CUDA, about 70 GB of host RAM, 120 GB on the cold first run) and when FLUX 3 Klein is the better choice, the fixed six-step sampler with no guidance, the 32-pixel grid, no speedup below 2048x2048, and MiniMax's license. |
| `cinematography` | A director's vocabulary for camera and look - shot size, camera move, lens, light, era, tempo - as closed values, each compiled to the words it becomes in H3, LTX-2.5, Kandinsky 6, FLUX 3 Klein and MiniMax-H3-Image captions; five named grades as `apply_lut` palettes plus `film_grain`. The words are hints the models may not follow, so the skill says how to check a move in the frames. |
| `storyboard` | Boards before video: a scene drawn on FLUX 3 Klein from character and location references (up to eight boards, cited as "image one"), the six-cut coverage set of one moment, a board as the first frame of an H3, LTX-2.5 or Kandinsky 6 shot (an H3 keyframe on MiniMax-H3-Image when the host can run it), and a timed animatic with a scratch voice and music bed. Takes the Popcorn shape from Higgsfield (concept only) and the coverage set from Shot Board (MIT). |
| `script-to-video` | From a prose script to finished video: decompose into shots, cast recurring characters once, pick each shot's template and family, validate cost before spending, run and diagnose, hand off to cut-and-score. The decision tree that turns a script into the shots the other skills know what to do with. |
| `series-episodes` | A series rather than one generation: several episodes over one cast, each cut and scored from its own template runs. The five-beat recut-bed-match_levels-normalize-pair procedure, an optional per-shot look (one shared palette through `apply_lut`, then `film_grain`) before the recut, and the one step that must not be skipped or the cast drifts between episodes. |
| `long-form` | A piece longer than one clip: scenes first, then per scene one clip, cuts or a chain; the chain's `shot` preset (static, static_long, moving, dialogue), the per-segment check and the in-place repair of one segment; narration and music longer than one generation, `pair_audio`, the H3 audio pass, and an OTIO export for an editor. Every rule cites the bench record it came from; what is still being measured is marked pending. |
| `choosing-a-model` | A picture request that names no family, workflow or model: run `templates/preferred/text-to-image`, which composes the server's `preference:text_to_image` ladder, at the tier the user asked for (`draft`, `standard`, `best`) or none, so the store's default applies; read the plan's resolution line to say which model will run on this host before spending; keep `templates/text-to-image` (SD 1.5) for "does it work" smoke tests; a named family, workflow or model wins. |

The two MiniMax skills defer the prompt format to MiniMax's own skills and
fetch the vendor's guides when those are absent, so nothing else is required.
Installing them is optional and worth it for Music 3, whose skill carries a
genre router and 1,000 example captions that a fetch does not reach:

```bash
npx skills add MiniMax-AI/MiniMax-H3 --skill h3-prompt-writing
npx skills add MiniMax-AI/MiniMax-Music3 --skill music-caption-rewriter
```

Name the skill: the H3 repo also ships eight style packs
(`brand-promo-video-generator` and the like) that `--skill '*'` would install
alongside it. When a request wants one of those looks, install that pack the
same way, by name; the `minimax-h3` skill's `references/style.md` lists them.

`tests/test_plugin_skills.py` holds each skill's catalog names and numeric
rules to the catalog and to the diffusers module that enforces them, and
requires every skill directory to be named here and in the repo's `CLAUDE.md`.
A skill's `SKILL.md` stays at or under 12 KiB because it loads whenever the
skill triggers; detail only some requests need is in its `references/`, read
on demand and held to the same tests. A family's `references/intents.md` is
its per-family index from a wanted result to the pattern that writes it and
the record that settled it (`tests/test_plugin_intents.py`). The plugin's version
(`.claude-plugin/plugin.json`) is the engine's; `scripts/release.sh` bumps
both. The marketplace entry is `.claude-plugin/marketplace.json` at the repo
root.

Adding a family: copy a skill, follow its outline, add the family's rules to
the test, cite the vendor, and name it here and in `CLAUDE.md`. The repo's
`model-family-onboarding` skill (`.claude/skills/`) is the full lifecycle.
