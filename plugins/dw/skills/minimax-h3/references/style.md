# MiniMax H3: a request that wants a look

Part of the `minimax-h3` skill; its *Prompts* section says when to read this.

MiniMax's H3 repo ships eight style packs next to `h3-prompt-writing`. Each
plans a whole piece in one look - style rules, shots, and the prompts for
them. When a request asks for one of these looks, install that pack by name
and let it write the prompts; the templates and hard rules in `SKILL.md`
still pick what runs.

```bash
npx skills add MiniMax-AI/MiniMax-H3 --skill <name>
```

Name the pack. `--skill '*'` installs all eight plus `h3-prompt-writing`.

| Pack | Suits a request for |
|---|---|
| papercraft-stop-motion-explainer | an explainer built from folded, cut paper, animated frame by frame |
| paper-collage-explainer-generator | an editorial explainer of torn-paper and halftone cut-outs |
| 3d-animation-short-generator | a 3D animated short |
| handdrawn-live-video-generator | hand-drawn animation over a live-action scene |
| minimalist-product-ad-generator | a clean, minimal product ad |
| music-video-subtitle-generator | a music video with on-screen lyrics or subtitles |
| brand-promo-video-generator | a brand promo |
| co-op-game-intro-generator | an intro for a co-op game |

The names are MiniMax's and nothing here checks them; if one fails to
install, the pack was renamed, so look in the repo's `skills/` folder.

## Without a pack: name the medium, then what it looks like

A medium named alone tends to render as its smooth CG version. Name it, then
the physical traits that make it that medium: for claymation, fingerprints,
tool marks and stepped, frame-by-frame motion; for paper, fibres, folds and
cut edges; for crayon, wobbling lines and uneven fill.

This is MiniMax's practice, not ours: the papercraft, paper-collage and
hand-drawn packs each prompt this way (read 2026-10-30). It is **unprobed on
dw** - no job here has compared a bare medium with a described one.

It doesn't override `SKILL.md`'s guide rule: a guide clip holds its look as
well as its motion, and no style prompt restyles it.
