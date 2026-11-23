# Wardrobe variants: costume-change guidance, text only (#1591)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out session,
2026-11-22). Don filed the idea on 2026-11-22 from charliewwdev/OpenMontage ("Drama
Studio", AGPL-3.0, concepts only, no code taken). Its character bible gives each character
a list of `wardrobe_variants`, each with its own reference images, and a shot names a
character plus an outfit. Our cast sheet and `dw:series-episodes` held one look per
character. The plan comment on #1591 (v1) holds the full sweep. This file records what was
built and what was cut.

## Verdict: build smaller. Text only, the probe cut

Before this feature the skills contradicted each other once a character changed clothes.
`series-episodes` said to draw each portrait "exactly once" and keep it as
`cast/<character>`. `minimax-h3/references/cuts.md` said to hold wardrobe "by putting it in
the portrait". The record already answered half of it: #1193 (refuted claim 3) found that
one portrait plus a prompt naming new clothes gives the new outfit within a shot, and a
face crop held identity no better. Nobody has measured whether that outfit holds across
shots, or whether one portrait per outfit keeps the face.

No field report asked for this, and the idea came from an external schema. The plan
therefore shipped the measured path, marked the per-outfit portrait as untested, and cut
the probe.

## Plan history

- **v1 (design, 2026-11-22; approved the same day at its defaults by the plan-review
  delegate under Don's 2026-11-21 delegation).** Q1 [scope]: text only, probe cut. Q2
  [none]: outfit portraits are named `cast/<name>-<outfit>`, matching `places/<set>-night`.
- **Corrections to the proposal the sweep found:**
  - "Name the outfit in each shot's item note" doesn't fit the ledger. Items have no
    agent-written note (`notes` are the user's, `dw/server/routes/requests.py`), and
    unknown item fields are refused (`dw/ledger.py`). The outfit goes in the item id
    (`cast-<name>-<outfit>`), the asset name, or the run `note`.
  - "Draw each outfit from the first portrait" is unmeasured, and on MPS the Klein edit
    and multi-reference templates have never run (they hard-code `"cuda"`). The text names
    it as untested, not as the rule.
  - #1400's face-once sheet doesn't cover it. It scored 0.05-0.12 SFace lower than a
    single portrait (`face-once-sheet-complete.md`), so redrawing its body views per outfit
    would rebuild a reference that already lost.

## What was built

### Stage A (#1594): costume-change guidance (plugin only)

Commit 7806141f, merged into `develop` as 630bd616 (branch `feat/1591-A-costume-change`).
Plugin text and tests only, no server restart.

- **`plugins/dw/skills/minimax-h3/references/per-character.md`**, new *A costume change*:
  - The measured path comes first (#1193): keep the one portrait, name the new clothes,
    and say they aren't taken from the picture.
  - It quotes the `cuts.md` wardrobe rule word for word: name the outfit in identical words
    in every shot, and when it drifts use a 22-frame window from an earlier shot in that
    outfit.
  - **Untested (#1591):** one portrait per outfit, edited from the first. On CUDA this goes
    through `templates/flux3/multi-reference` or `templates/minimax-h3-image/multi-reference`.
    On MPS it is unmeasured, and the text says so before spending. Each portrait is kept as
    `cast/<name>-<outfit>` with `shared=True` (item id `cast-<name>-<outfit>`), and drift is
    checked against the portrait the shot used.
- **`script-to-video/references/fix-ladder.md`** *Face drift*: the reference is the cast
  portrait the shot was conditioned on, per outfit when the cast has more than one.
- **`series-episodes/SKILL.md` step 0 and `storyboard/SKILL.md` §1**: one line each linking
  *A costume change*, with the `cast-<name>-<outfit>` item id. The "exactly once" base rule
  stays.
- **`minimax-h3/references/intents.md`**: one `untested` row, § A costume change, #1591.
- **Tests:** `TestCostumeChange` in `tests/test_plugin_skills.py` (8 cases). They pin the
  heading, the order (#1193 before Untested), the quoted `cuts.md` sentence, the asset name,
  the MPS caveat, that the named templates exist, that the links resolve, the fix-ladder
  wording, no `outfit` key in the touched files, and the intents row.
- `script-to-video/SKILL.md` and `cuts.md` are unchanged, and `TestPlaceCard` and the size
  caps still pass.

**Bounces: 0.** The tester verified M-F486 to M-F492 on the mini-ai server (mps) against
`develop` @ 630bd616. All seven passed, including the edge case that a `set_request` with an
`outfit` item field is refused.

## Cost and bounces

| Stage | Bounces | Cost |
|---|---|---|
| #1594 | 0 | Session cost not recorded on the stage. No GPU time |

## Deferred and open

- **The probe (cut).** It would compare outfit portraits with a single-portrait control,
  scoring SFace across outfits. It comes back on a field report of a costume change where
  the face or the outfit didn't hold, or on a series Don starts with a planned costume
  change. It would run on lem: the Klein edit templates are CUDA-pinned, and H3 clips on
  mini-ai take about 9 min each and load only from lem's pre-quantized folders. Estimate
  $8-15. SFace can't score a still directly (`analyze_face_drift` reads a video or frames),
  so a stills-only arm needs a `stills_to_video` wrap.
- **A ledger `outfit` item field (cut).** It comes back if names prove too weak to tell
  which outfit a shot used.
- **A per-outfit #1400 sheet (cut).** It comes back only if a sheet ever beats the single
  portrait.
- Out of scope throughout: engine, MCP, REST, ledger and workflow-syntax changes, new
  templates, and the pinned place-card wording.
