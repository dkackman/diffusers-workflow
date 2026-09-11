# dw plugin

Skills for Claude Code that teach an agent to compose a model family's
workflows on a [diffusers-workflow](https://github.com/dkackman/diffusers-workflow)
server. They assume the `dw` MCP server is already registered
(see the repo README, "Drive it from Claude Code"); a skill's first move is
`get_server_info`.

Install once, from Claude Code:

```
/plugin marketplace add dkackman/diffusers-workflow
/plugin install dw@diffusers-workflow
```

| Skill | Teaches |
| ----- | ------- |
| `minimax-h3` | MiniMax H3 video with audio: one take, a longer take by chain, a piece with cuts, identity and voice references, music; the frame and canvas rules; what a run costs. Prompts come from MiniMax's own `h3-prompt-writing` skill or the guides on the model card. |
| `minimax-music3` | MiniMax Music 3: a song, an instrumental, a score under a film, the soundtrack a music video is cut to; the ceiling semantics, the tag vocabulary, the caps and the 44.1 kHz output. Captions come from MiniMax's own `music-caption-rewriter` skill. |
| `ltx-2.5` | LTX-2.5 video with a soundtrack: a single clip, first-frame and keyframe conditioning, the three-move two-stage flow, the IC-LoRA upscale, extend and chain; the distilled schedule and the frame and size rules. Prompts follow the trained-caption spec that ships inside diffusers. |

The two MiniMax skills defer the prompt format to MiniMax's own skills and
fetch the vendor's guides when those are absent, so nothing else is required.
Installing them is optional and worth it for Music 3, whose skill carries a
genre router and 1,000 example captions that a fetch does not reach:

```
npx skills add MiniMax-AI/MiniMax-H3 --skill h3-prompt-writing
npx skills add MiniMax-AI/MiniMax-Music3 --skill music-caption-rewriter
```

Name the skill: the H3 repo also ships eight style packs
(`brand-promo-video-generator` and the like) that `--skill '*'` would install
alongside it.

Each skill quotes catalog names and numeric rules that `tests/test_plugin_skills.py`
holds to the catalog and to the diffusers module that enforces them. The
plugin's version is the engine's; the release script bumps both.

Adding a family: copy a skill, follow its outline, add the family's rules to
the test, cite the vendor. The repo's `model-family-onboarding` skill
(`.claude/skills/`) is the full lifecycle.
