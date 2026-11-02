# Compile Higgsfield's presets into dw's skills

dw should adopt Higgsfield's **control vocabulary and planning shape**, not its model zoo. Most of what makes Higgsfield feel directable is a closed set of named choices: camera move, lens, framing, light, genre, era, tempo and grade. Those choices compile into prompt text before a model ever runs, and an agent plans shots from a script on top of them. dw can build that today as catalog and skill text over H3, LTX-2.5, Kandinsky 6 and FLUX 3 Klein, because every family's caption format already asks for one camera statement. The open-source Higgsfield clones from Comfy's challenge converged on the same architecture dw already has: declarative graph JSON, a server, a UI, an agent surface and a provenance record per take. Their ideas port cleanly even though their ComfyUI graphs do not. The features dw cannot reach on its current families are re-shooting a clip from a new angle, swapping an actor, dubbing existing footage, relighting video, and spectacle effects. Re-shooting and angle stills come within reach of an add-on (the LTX-2.5 Layout-To-Render IC-LoRA, or Qwen-Image-Edit-2511 with a multi-angle LoRA). Actor swap and most effect presets need Wan 2.x as a new base family, which collides with the 24 GB rule. The highest-leverage single move is a camera-and-look vocabulary in the plugin skills plus a persisted shot document, measured before it is promoted. That one move turns `script-to-video` from "write a good paragraph" into "fill a shot card", and it is the shape every serious clone and Higgsfield itself landed on.

## Higgsfield is a preset layer over other people's models

Higgsfield aggregates **30+ third-party models** (Seedance, Kling, Veo, Sora 2, Wan, Nano Banana, Hailuo) behind single-purpose "apps", plus its own Soul image model and Soul ID identity training ([homepage](https://higgsfield.ai/); [help center](https://higgsfield.ai/creator-hub/help-center/getting-started/what-is-higgsfield)). It grew virally in 2025 on one-click camera presets such as Crash Zoom, Bullet Time and Super Dolly, applied to a single image to make a roughly 5 s clip ([X, Mar 2025](https://x.com/higgsfield_ai/status/1906748661322711305); [memons cheat sheet](https://memons.ai/higgsfield-prompt-cheat-sheet)). In late 2026 its flagship surfaces are:

- **Cinema Studio 4.0**, a director's panel.
- **Genjutsu**, video-to-video recast and object swap, launched 1 Sept 2026.
- **Supercomputer**, a chat agent that plans and routes jobs.
- **Popcorn**, an up-to-8-frame storyboard.
- **Canvas**, a node graph.
- **Marketing Studio**, for ads.

A long tail of tools sits beside these: Angles and Shots (re-angled stills), Relight, Upscale, Lipsync Studio, Effects, and Draw-to-Video.

Cinema Studio shows the product philosophy most clearly. Its build blog says: **"Instead of describing a look in words, you build the frame the way a cinematographer would."** Shot settings and metadata are saved with each still, so a still used as a video start frame carries its look forward ([How we built Cinema Studio](https://higgsfield.ai/blog/how-we-built-cinema-studio)).

The control set grew version by version:

- **v1:** camera body, lens and focal length.
- **v1.5:** aperture.
- **v3.0:** genres (General, Action, Horror, Comedy, Noir, Drama, Epic) and eight speed ramps (Linear, Auto, Flash In, Flash Out, Slow-mo, Bullet Time, Impact, Ramp Up). Audio, meaning SFX, speech and music, arrived in the same pass.
- **v3.5:** per-shot camera, lens, focal length and aperture, plus global genre, style, lighting, 8 palettes and 10 "MoveSet" styles. This version added the **AI Director**, a Claude chat panel that breaks a script into shots with camera parameters pre-filled.
- **v4.0:** Era (60s–2020s), Tempo (Single Shot to Chaotic), abstract camera bodies ("35mm Film", "8mm Film", "DV Camcorder"), lens families, 50+ grade templates, an emotion slider, up to 50 references, Forward/Backward Extend to 30 s, and regional Edit Video ([Cinema Studio help](https://higgsfield.ai/creator-hub/help-center/tools/how-do-i-use-cinema-studio)).

The trajectory matters for dw. Higgsfield moved **away from literal hardware names** (ARRI, RED, Cooke) toward abstract looks once multi-shot generation made per-shot rigs less useful. Its blog says camera control "moved up a level to genres" ([How we built Cinema Studio](https://higgsfield.ai/blog/how-we-built-cinema-studio)). Cinema Studio 4.0 runs on Seedance 2.5, so these controls are almost certainly prompt composition plus curated per-model templates rather than model conditioning. That inference comes from the research notes, not from a Higgsfield statement, and it is exactly the layer dw keeps in skills.

Consistency comes in three layers:

- **Trained identity:** Soul ID, from at least 20 photos in about 3 minutes.
- **Named reusable "Elements":** characters, locations and props, injected as numbered references ("image one") into third-party models.
- **Project-locked style:** lighting, optics and tempo held across the project, plus within-sequence chaining in Popcorn, where each clip's last frame starts the next ([Popcorn help](https://higgsfield.ai/creator-hub/help-center/ai-models/how-do-i-use-popcorn)).

Supercomputer, the agent spun out of AI Director, plans a job, routes each step to a model, shows per-step credits live and re-runs only the affected steps on revision. Higgsfield also ships an MCP server for Claude ([Supercomputer help](https://higgsfield.ai/creator-hub/help-center/tools-and-workflows/how-do-i-use-supercomputer)). No usage numbers exist for any app, so "most popular" can only be argued from the viral clips, which were the camera presets and Effects.

### Feature-by-feature: where dw already stands

Status is judged against dw 0.17.0 as read from the repo; nothing was GPU-tested for this report.

| Higgsfield feature | dw state | What dw has, or the gap |
|---|---|---|
| Text/image to shot, first+last frame, extend | **Has** | H3 (22 templates), LTX-2.5 (19), Kandinsky 6 (3) ([minimax-h3 skill](../../../../plugins/dw/skills/minimax-h3/SKILL.md); [ltx-2.5 skill](../../../../plugins/dw/skills/ltx-2.5/SKILL.md)) |
| Multi-shot with cuts in one generation | **Has** | H3 `storyboard`, `dialogue-short`; LTX "a hard cut to" 2-4 shots per paragraph |
| Native dialogue/SFX/music, voice | **Has** | H3 dialogue, Music 3, MMAudio SFX, TTS, voice clone ([TASKS.md](../../../../docs/TASKS.md)) |
| Elements / Soul Cast (named reusable cast, locations, props) | **Has (as procedure)** | `keep_output(..., shared=True)` cast assets + `prompt:<series>/<character>` ([series-episodes skill](../../../../plugins/dw/skills/series-episodes/SKILL.md)) |
| AI Director / Supercomputer (script → shots → route → run) | **Has (as skill)** | `script-to-video` shot list with cast, opening/ending state, beat, `must_not`, join; routes H3 vs LTX ([script-to-video skill](../../../../plugins/dw/skills/script-to-video/SKILL.md)) |
| Colorgrade panel, palettes, film grain | **Has** | `grade`, `apply_lut` with `palette`, `film_grain`, `sharpen` |
| Upscale / restore | **Has (exceeds)** | LTX upscale/refine/restore/colourise/alpha-matte, spandrel, diffusion upscale |
| Montage pacing / beat-cut music video | **Has (exceeds)** | `analyze_beats`, `analyze_phrases`, `plan_cuts`, `music-video-cuts` |
| Edit Video (regional) / face fix | **Has** | H3 `regional-edit`, `face-repair`, `crop_face_track`/`paste_face_track` |
| Named camera-move presets (Crash Zoom, Dolly, Orbit…) | **Missing** | Camera lives only inside each family's caption prose; no preset list (repo grep) |
| Lens / framing / light / era / genre / tempo panel | **Missing** | Same as above; no vocabulary |
| Speed ramps / slow-mo | **Missing** | LTX-2.5 Slow-Motion-Control LoRA exists upstream, not onboarded |
| Popcorn storyboard / animatic deliverable | **Partial** | FLUX 3 Klein `multi-reference`, `layout-boxes`, `consistent-set` exist; no skill composes a board |
| Shots / Angles (multi-angle stills) | **Partial** | Klein multi-reference by prompt only; no trained angle vocabulary |
| Automatic judging between generations | **Partial** | `analyze_shots`/`analyze_seams`/`analyze_sync_drift` probes report places to look, not verdicts; no face-identity check |
| Re-shoot from a new angle (video) | **Missing** | Needs depth + Layout-To-Render or CrossView |
| Genjutsu motion transfer / actor swap | **Missing** | Face round-trip tasks exist; no swap model |
| Lipsync Studio onto existing footage | **Partial** | H3 locked-frame lip-sync to a held track (#985); no dubbing model |
| Relight (photo/video) | **Missing** | No relight model in any family |
| Effects library (Set On Fire, Melt…) | **Missing** | Open versions are Wan community LoRAs |
| Draw-to-Video | **Missing** | Klein `layout-boxes` is the nearest primitive |
| Canvas node graph | **Has (different)** | Web UI FlowView/StepEditor; no real-time collaboration |

The table makes one thing clear: **dw's gaps are mostly vocabulary and presentation, not capability**. The big capability gaps are geometry (re-shoot, angles), identity transfer (swap, dub) and light. Those are the places where Higgsfield is renting Seedance and Kling, and where the clones reach for add-on models.

## The challenge clones converged on dw's own architecture

Comfy's #OpenSourceHiggsfield challenge asked entrants to "Rebuild Higgsfield's popular apps, and more, using ComfyUI" under an open-source licence. It had tracks for Cinematic Studio, Re-shoot and "Openjutsu", plus bonuses for LTX, FLUX 3 and MiniMax H3 as the core model ([comfy.org](https://comfy.org/open-source-higgsfield.md)). That bonus list lines up almost exactly with dw's families.

The CC0 index `jutsuhq/open-source-higgsfield` catalogs **102 entries and 6 older alternatives**, and is the place to start for anything not named here ([jutsuhq list](https://github.com/jutsuhq/open-source-higgsfield)). The entries are overwhelmingly MIT or Apache and open-weights-first: LTX-2.5, H3, FLUX 3/2 Klein, Qwen-Image-Edit-2511, Wan 2.2, SAM 3.1 and SeedVR2. Nearly all share one architecture: a custom web front end and a small server that patches committed ComfyUI API-format JSON per job. Structurally that is dw's declarative workflow, server, UI and MCP stack with a different engine underneath. The research notes infer from this that the clones' UX ideas port to dw even where their graphs do not.

Four architectural patterns recur, and each has a direct dw counterpart.

**Preset → compiled prompt.** Aither Cinema Studio (MIT) states the principle best: "A rig is a decision, not a slider stack. `dolly_in` + `85mm` + `close_up` is a different shot from `static` + `85mm` + `close_up`, and the studio makes the model honour the difference — deterministically, in the prompt text." Its vocabulary has **20 moves, 7 lenses, 3 speeds, 10 framings and 9 looks** ([Aither](https://github.com/Aitherium/aither-cinema-studio)).

Other entries follow the same pattern:

- **OpenField (MIT):** its Director compiles genre, camera, framing, light, film stock, look, grade, an era scrubber and speed ramps into the shot, and its Video app has 28 motion presets ([OpenField](https://github.com/wexloops/openfield)).
- **CreatorOS Studio (Apache-2.0):** a "shot card instead of a prompt", 25 one-click camera/effect presets, seven relights, and a re-shoot angle set ([CreatorOS](https://github.com/thePloggy/CreatorOS-Studio)).

In dw the compiled form belongs in the plugin skills and the stored prompt library, keyed per family. That honours the rule that model knowledge lives in the catalog and skills, never in engine code.

**Plan without render.** Aither's `plan_shot` compiles a rig into the exact prompt with no render. Comfy-Dial has a free "Run dry". Haggs puts the price on every Generate button, and "Nothing is spent on the drawing page" ([Comfy-Dial](https://github.com/ylchen333/comfy-dial); [Haggs](https://github.com/IBFolding/haggs)). dw already has the stronger half of this: `validate_workflow` is free and returns `plan.estimate`, and `acknowledged_cost` is a hard gate ([MCP.md](../../../../docs/MCP.md)). The missing half is a **free view of the compiled prompt per shot**, which `enhance_prompt` approximates only for the families with enhancers.

**Judges between generations.** Several entries check output automatically:

- **CreatorOS** runs a CPU clip check in about 1 s for black frames, frozen video, an in-shot jump, smear, and **how close the face stays to the start image**, using OpenCV YuNet + SFace ([CreatorOS](https://github.com/thePloggy/CreatorOS-Studio)).
- **Body Double (Apache-2.0)** runs seam-ratio and background-drift QC before the next segment starts, and re-rolls a flagged segment ([Body Double](https://github.com/lucataco/open-source-higgsfield)).
- **open-higgsfield-skills (MIT)** has a still-QA judge with a fixed fix ladder (image-edit, relight, re-prompt, seed+1) and hard caps on retries and GPU-minutes ([skills](https://github.com/mojomast/open-higgsfield-skills)).
- **Boson (MIT)** persists each judged round as git-backed state that "the next round must read" ([Boson](https://github.com/0xMisterBailey/Boson)).

dw's `assess_output` probes, `compare_outputs`, `get_output_frames(seams=)` and `set_note` cover the mechanics. What is missing is the **face-identity measurement**: the series skill says outright that "There is no engine check for 'the same character looks the same' across separate runs" ([series-episodes skill](../../../../plugins/dw/skills/series-episodes/SKILL.md)).

**Shot documents.** A few entries make the shot plan its own artifact:

- **seriesmith (MIT)** compiles `series.json` and markdown screenplays into a **model-free `shots.json`**, then compiles that per video target. "Every shot is its own job, so a bad shot is rendered again alone." Its assistant `SKILL.md` is generated from its authoring docs ([seriesmith](https://github.com/bgstratt/seriesmith)).
- **Picky** makes each H3 shot "a structured document with style, overview, timed beats, camera, dialogue, soundscape and music", and adds a **prompt linter that knows H3** for overfull beats, cuts with no new information and missing soundscape ([Picky](https://github.com/chenthedesignmaker/ComfyUI-Picky)). Picky is GPL-3.0, so for dw it is ideas only.

dw's `script-to-video` shot list already has the right fields for continuity: opening and ending state, beat, `must_not` and join. But it lives in the agent's context rather than as a persisted, re-compilable document. Its fields also cover continuity, not cinematography.

| Clone pattern | Best source to mine | dw seam it maps to | Gap to close |
|---|---|---|---|
| Closed-enum rig compiled to prompt | Aither (MIT), CreatorOS (Apache), OpenField (MIT) | Plugin skill `references/` + `prompt:camera/…` library | Vocabulary plus one compiled sentence per family |
| Plan-without-render | Aither `plan_shot`, Comfy-Dial dry run | `validate_workflow` + `plan.estimate`; `enhance_prompt` | Show the compiled caption per shot before spend |
| Judge + fix ladder + caps | open-higgsfield-skills, CreatorOS QC, Body Double | `assess_output`, `compare_outputs`, `set_note`, skill step 6 | Face-drift probe; written fix ladder |
| Model-free shot document → per-target compile | seriesmith (MIT), Picky (GPL, ideas) | `script-to-video` step 2 shot list | Persist it; add camera/lens/light fields |
| Provenance per take | OpenField, CreatorOS, Shot Board | Run manifests, `export_job`, `rerun_job` | None; dw already leads |
| Hardware/quality tiers | ReAngle, Picky, open-higgsfield-skills | H3 draft/final tiers, 24 GB recipes | None material |
| Preflight of models with install hints | OPENjutsu (MIT) | `validate_workflow` downloads | None material |
| Resume after crash | OPENjutsu, Body Double | Open `resume.md` proposal | Already queued |

## Adopt now: vocabulary, storyboards and judges from text alone

Everything in this tier uses the current families and adds only catalog, skill or prompt-library text. The one exception is a small, model-agnostic probe. It is where most of the "agents compose cinematic output" value sits.

**1. A camera-and-look vocabulary per family.** The plugin should gain a shared reference with a closed set of axes:

- **Shot size:** extreme wide to extreme close-up.
- **Camera move:** static, dolly in/out, truck left/right, crane/jib up/down, arc/orbit, tracking, whip pan, push-in, handheld, crash zoom.
- **Lens character:** wide, normal, long, anamorphic, vintage.
- **Light:** high-key, low-key, golden hour, practical, noir.
- **Era/stock:** 8mm, 16mm, 35mm, VHS/DV, digital.
- **Genre and tempo.**

Each family's skill should then state the **one sentence** that value compiles to in that family's caption format. LTX-2.5's trained caption spec already requires "a shot type, a camera motion (say static when none) and a viewpoint" ([ltx-2.5 skill](../../../../plugins/dw/skills/ltx-2.5/SKILL.md)). Kandinsky 6 demands "exactly one camera sentence" ([kandinsky-6 skill](../../../../plugins/dw/skills/kandinsky-6/SKILL.md)). H3's camera language lives inside its Context-IR caption structure ([minimax-h3 skill](../../../../plugins/dw/skills/minimax-h3/SKILL.md)). So this fills an existing slot rather than inventing one.

Mine these sources:

- **Aither:** the enum sizes and the "decision, not a slider" principle.
- **CreatorOS:** the preset list and pull-a-palette-from-a-photo.
- **OpenField:** the Director axes and era scrubber.
- **Higgsfield's preset names:** they are the user-facing words a person will type ([memons](https://memons.ai/higgsfield-prompt-cheat-sheet)).
- **Lightricks' camera LoRA set:** dolly in/out/left/right, jib up/down, static. Its card advises "naming in the prompt what should come into view during the move", which is a good compile rule for LTX ([Dolly-Left card](https://huggingface.co/Galatea777/LTX-2-19b-LoRA-Camera-Control-Dolly-Left)).

Follow Higgsfield's own retreat and **do not ship camera-body brand names** as presets. Abstract looks generalise across families. Brand names only work if the model learned them, and nobody has measured that.

The caveat is measurement. dw's story-forge audit found the "i2v invents camera moves" lesson **did not reproduce** on LTX (0 of 6 clear moves). Yet one 241-frame clip pushed or arced after about 5 s despite "camera remains static" ([story-forge audit](../../../../docs/proposals/audits/2026-11-story-forge-lessons.md)). Whether H3 or LTX honour *named* moves is unprobed. The vocabulary should therefore ship as `trial`, with a small per-family A/B per move judged from `get_output_frames`. Comfy-Dial's approach is the model: recover the camera from each take and score it ([Comfy-Dial](https://github.com/ylchen333/comfy-dial)).

**2. A persisted shot document with cinematography fields.** Extend `script-to-video`'s shot list with `shot_size`, `move`, `lens`, `light` and `duration`. Store it as a workspace artifact, or as a prompt-library entry, so a revision recompiles only the changed shots. That is Supercomputer's "re-runs only affected steps", reached through dw's per-shot jobs.

Mine these sources:

- **seriesmith:** the model-free shot list and per-target compile, `continuous:` chaining queued up front, the STORY_STRUCTURES docs (three-act, Story Circle, Kishōtenketsu), and generating the skill from the authoring doc ([seriesmith](https://github.com/bgstratt/seriesmith)).
- **OpenReel (MIT):** the three continuity anchors. Inject the actor description verbatim into every keyframe prompt, use the first head sheet as the image anchor, and keep one unified style block ([OpenReel](https://github.com/smithysw7/openreel)).
- **Vision Orchestrator (Apache-2.0):** "only approved content flows into generation" ([Vision Orchestrator](https://github.com/stellartune-studio/vision-orchestrator)).

The written shot document is also where an **H3 prompt lint** can live as skill text: flag overfull beats, a cut with no new information, and a missing `overall_soundscape`. Take that idea from Picky without its GPL code.

**3. A storyboard and animatic skill.** Popcorn's shape works on FLUX 3 Klein today:

- up to 8 frames;
- references cited as "image one";
- Manual or Auto mode;
- each frame later becomes a start frame, chained by last frame ([Popcorn help](https://higgsfield.ai/creator-hub/help-center/ai-models/how-do-i-use-popcorn)).

Klein has `multi-reference` (10-reference training limit, not pipeline-enforced) and `layout-boxes` for blocking by bbox, and `consistent-set` produces a subject that differs in one stated way ([flux3-klein skill](../../../../plugins/dw/skills/flux3-klein/SKILL.md)).

Mine **Shot Board (MIT)** for its **six-cut coverage set**: establishing, medium, face close-up, over-the-shoulder, insert and low angle. Pick a hero, redraw the rest from it, record a pass/fail verdict with a note, and export a manifest ([Shot Board](https://github.com/ArcSolver/shot-board)). Klein multi-reference stands in for Shot Board's Qwen-Image-Edit-2511, with weaker angle control.

An animatic is then those boards timed to a TTS scratch track and a Music 3 bed through `assemble-and-score`. This is the cheapest pre-vis tier for sitcom and animated work, and it lets an agent show the cut before spending H3 minutes. One unknown remains: the notes do not say whether an existing task, `loop_frames` for example, can hold a still for a stated duration. If none can, a still-to-clip utility task is the only engine work this tier needs.

**4. A face-drift probe and a written fix ladder.** Add an assessment probe in the style of CreatorOS: YuNet face detection plus SFace embedding similarity of each shot's face against the cast asset ([CreatorOS](https://github.com/thePloggy/CreatorOS-Studio)). It is model-agnostic and CPU-only, and it fills the gap the series skill names. Like the existing probes it should report places to look, not verdicts. The OpenCV model licences are not in the notes and need a check.

Pair the probe with a written fix ladder and retry/GPU-minute caps in `script-to-video`, taken from open-higgsfield-skills ([skills](https://github.com/mojomast/open-higgsfield-skills)). The ladder runs: regional edit or face repair, then re-prompt, then reseed, then ask. Body Double's "QC before the next segment starts" belongs in the chain guidance ([Body Double](https://github.com/lucataco/open-source-higgsfield)). Its five-frame conditioning and per-segment colour match overlap the parked #1033/#1067.

**5. Sitcom and multicam grammar.** One-shot Multicam (MIT) cuts between angles **on word boundaries** from a transcript ([One-shot Multicam](https://github.com/apsolut/one-shot-multicam)). dw already has `transcribe_audio`, `analyze_phrases` and `plan_cuts`. A sitcom section in `series-episodes` should therefore describe:

- a wide master, held throughout;
- per-speaker singles from `dialogue-short`;
- cuts placed on line boundaries.

The skill's own guidance to "frame recurring characters large and keep a wide master" already points this way ([script-to-video skill](../../../../plugins/dw/skills/script-to-video/SKILL.md)).

**6. Named grades and speed language.** Higgsfield offers 50+ grade templates and an era control. dw's `apply_lut` palettes plus `film_grain` can carry a small named set: noir, teal-orange, 70s stock, VHS, bleach-bypass. That set lives as stored palettes, with no engine change. Speed ramps stay prompt-only until the LoRA in tier 2 is measured.

## Add-ons that unlock re-shoot, angles and dubbing

These need a new adapter or model entry on a base dw already runs, or a new utility-task model. Each needs a 24 GB proof and a licence check before it becomes a template.

**LTX-2.5 Layout-To-Render IC-LoRA, for blocking-to-shot and re-shoot.** This is the clones' workhorse:

- **OpenField's Stage** renders greybox blocking to a depth pass, paints a look frame, then renders "a shot that moves like the blocking" with Layout-To-Render ([OpenField](https://github.com/wexloops/openfield)).
- **ReAngle (Apache-2.0)** re-shoots by depth-estimating the clip, warping along a camera path with a z-buffered splat, pre-filling revealed holes from the far background and guiding LTX-2.5, with chunk re-anchoring and LAB colour match for long clips ([ReAngle](https://github.com/Tekholms-LW/ReAngle)).
- **Comfy-Dial** feeds LTX a depth video rendered along an authored path. "The model never sees your camera numbers." It then recovers the camera per take and repairs only the drifted segment ([Comfy-Dial](https://github.com/ylchen333/comfy-dial)).

The LoRA is on the same base as dw's catalogued IC-LoRAs ([Lightricks listing](https://huggingface.co/api/models?author=Lightricks&search=LoRA&limit=100&sort=lastModified)). There is a warning in dw's own record, though. Clean-Plate and Day-To-Night were **not onboarded** because they "don't fit the distilled IC template" and need a guided dev-transformer route ([ltx2 IC-LoRAs record](../../../../docs/proposals/complete/ltx2-ic-loras-complete.md)). Layout-To-Render's recipe must be checked against that constraint first.

The depth model is a licence trap. ReAngle defaults to **Video Depth Anything Small because Base and Large are CC-BY-NC-4.0**. The depth estimator and the reprojection would be new utility tasks. This is the best route to Higgsfield's "Re-shoot" and to camera moves you can actually measure, and it is the most engineering in this tier.

**LTX-2.5 Slow-Motion-Control and Cinemagraph LoRAs.** These are plain LoRAs on the 2.5 base. They give speed-ramp and living-photo presets ([Lightricks listing](https://huggingface.co/api/models?author=Lightricks&search=LoRA&limit=100&sort=lastModified)), and they are dw's first honest "effect presets". Their sampling recipe is unverified.

**LTX-2.3 adapters on 2.5 (probe only).** Three 2.3 adapters are worth probing:

- **Relight** is driven by a light-direction ball composited into the reference's corner, and the docs say it is **not yet validated on LTX-2.5** ([relight card](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Relight); [docs.ltx.io](https://docs.ltx.io/open-source-model/feature-guides/editing-effects/relight)).
- **Motion-Track-Control.**
- **DubIt.** Its card was not read.

An aggregator says 2.3 LoRAs "partly work" on 2.5 ([gradually.ai](https://www.gradually.ai/en/ai-models/ltx-2-5/)). Treat each as a one-afternoon probe, not a plan.

**Qwen-Image-Edit-2511 + fal Multiple-Angles LoRA, for Shots and Angles.** This is the open standard for "9 angles from one image". It covers **96 camera positions** trained on Gaussian-splat renders and is driven by prompts like "Rotate the camera 45 degrees to the left" ([fal LoRA](https://huggingface.co/fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA); [Space prompt grammar](https://huggingface.co/spaces/linoyts/Qwen-Image-Edit-Angles/blob/bbbeff133c3a7714b98a8ab99a61895294dfb4cf/app.py)). CreatorOS and Shot Board both use it.

Two facts from dw's own audit shape the cost:

- The base is Apache-2.0 but **57.7 GB** at full precision, so it needs quantization or offload to meet the 24 GB rule.
- dw's onboarded **Qwen-Image 2.1** is under the non-commercial Qwen Research License and has none of this LoRA ecosystem ([Qwen audit](../../../../docs/proposals/audits/2026-09-20-qwen-image-2.1-audit.md)).

A `for_each` over nine fixed angle prompts makes a "Shots" template. Those stills then feed H3 FL2VA or LTX keyframes. The fal LoRA's licence is unchecked.

**H3 LoRAs from the Genjutsu clones.** Three LoRAs are relevant:

- **comfy-reshoot** pairs MoGe geometry with the **CrossView LoRA for H3** ([comfy-reshoot](https://github.com/Comfy-Org/comfy-reshoot)).
- **H3 Recast** swaps up to three people with the **BFS Character Swap LoRA** ([H3 Recast](https://github.com/Bambushu/h3-recast)).
- **OPENjutsu** runs H3 Ref2VA int8 with a **DMAD 4-step LoRA** ([OPENjutsu](https://github.com/D3voz/openjutsu)).

None has been checked for diffusers loading. OPENjutsu's recipe is the most portable idea:

- a SAM 3.1 track at 1008 px;
- the mask grown 40 px with 10% headroom;
- a grey-mannequin motion reference;
- Ref2VA with the new cast;
- a VLM labelling tracks to roles.

It also documents its failure modes: reflections leak the original, and clothing words leak. That maps onto dw's `segment`, `regional-edit`, `crop_face_track`/`paste_face_track` and Ref2VA. It is the one route to Genjutsu-style object and character swap without a new base. Expect H3 Recast's caveat to hold: "Big, obvious changes land … Subtle ones fall back to the clip."

**LatentSync 1.6 as a dubbing task.** For lip-syncing existing footage, as Lipsync Studio does, LatentSync is the clear commercial path: **Apache-2.0 code and OpenRAIL++ weights at about 18 GB VRAM**, with MuseTalk (MIT, about 4 GB) as the light option ([Creative AI News](https://www.creativeainews.com/articles/open-source-lip-sync-licenses-commercial-use-2026/); [LatentSync](https://github.com/bytedance/LatentSync)). It is post-processing on a clip, so it fits as a utility task rather than a family. One Civitai page labels the weights RAIL-M, which conflicts and is likely a mislabel ([HF discussion](https://huggingface.co/chunyu-li/LatentSync/discussions/2)). For new dialogue, H3's joint audio-video generation remains better than dubbing.

**Location consistency.** 360 Scout to Film (MIT) turns one equirectangular panorama into overlapping perspective location references plus a JSON manifest shared by every shot ([360 Scout](https://github.com/Alvinvision/360-Scout-to-Film-Consistency)). That is a pure image utility over Ref2VA's reference slots, with no new model, and it gives dw a "Cinematic Locations" equivalent.

## Wan 2.x is the price of actor swap and spectacle effects

A large share of what the clones and Higgsfield do sits on Wan 2.1/2.2 14B:

- **Wan2.2-Animate:** character replacement and motion transfer, the core of Genjutsu. diffusers has a `WanAnimatePipeline`, but "integration of these preprocessing steps into Diffusers is planned for a future release" ([diffusers PR #12442](https://github.com/huggingface/diffusers/pull/12442.diff)).
- **VACE, Phantom, MAGREF and Stand-In:** identity-reference options.
- **Remade-AI effect LoRAs:** 360 Orbit and Crash Zoom, with trigger phrases ([Remade-AI](https://huggingface.co/collections/Remade-AI/wan21-14b-480p-i2v-loras)).
- **Uni3C and ReCamMaster:** camera transfer and re-render. ReCamMaster's released weights are a Wan port that "may not match the paper" ([ReCamMaster](https://github.com/KlingAIResearch/ReCamMaster)).
- **InfiniteTalk:** long talking video.

Wan 2.2 itself is Apache-2.0 ([MindStudio](https://www.mindstudio.ai/blog/what-is-wan-2-2-video-open-source)). Adopting it is a real family onboarding, though. Animate needs pose and mask preprocessing (DWPose, SAM2) as new tasks. The conditioning adapters mostly need custom loaders. And dw's parked #1066 notes VACE reframe was "meant for a cloud GPU with 48 GB or more".

The recommendation is **defer**. Wan earns a slot only if a field report asks for actor replacement that the H3 mask-and-Ref2VA route cannot do. If it is onboarded, run it through the `model-family-onboarding` audit, and treat Animate's 24 GB fit as unproven.

The seven named Lightricks camera LoRAs fall in the same "new base" bucket. Their repos say **LTX-2 19B**, while Lightricks' VFX page lists them under LTX-2.3 ([HF](https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-In); [ltx.io](https://ltx.io/model/capabilities/vfx)). They do not target 2.5, so for dw they are vocabulary, not weights.

### Do not adopt

Leave these out:

- **Spectacle effects** (Set On Fire, Melt, Cakeify, Explode). Their open forms are Wan community LoRAs with unchecked licences, or LTX-Video 0.9 relics, and they serve virality rather than cinematic or sitcom composition.
- **A Soul ID-style per-person training step.** dw's fixed-asset cast references are the consistency mechanism and are already proven in the series skill.
- **Supercomputer's connectors, app deployment, Playable, AI-influencer and ad factories.** They are outside dw's purpose.
- **Code from no-licence repos**, including the 5,211-star `wide-trace/open-higgsfield`, HeliosGen, Darius, Conjure, FXjutsu and HiggsOpen. These are concept-only ([jutsuhq list](https://github.com/jutsuhq/open-source-higgsfield)).
- **Code from Picky and Open Cinema (GPL-3.0), or from Parametric Studio and SpEdE (AGPL).** Ideas only. Open Cinema's post-hoc physically based depth of field and rack focus from per-frame depth is a good idea to revisit once a depth task exists ([Open Cinema](https://github.com/LaytonGott/open-cinema)).
- **Wav2Lip.** Commercial use is "strictly prohibited".

Non-commercial weights to flag if they ever appear in a template: Video Depth Anything Base/Large (CC-BY-NC), FLUX.2 (marked non-commercial by OpenField), dw's own Qwen-Image 2.1, and MMAudio (CC-BY-NC 4.0, already shipped). Hunyuan3D 2.1 excludes the EU, UK and South Korea, and the H3 community licence has geographic exclusions too ([ReAngle](https://github.com/Tekholms-LW/ReAngle); [Comfy-Dial](https://github.com/ylchen333/comfy-dial); [OpenField](https://github.com/wexloops/openfield)).

### Unresolved and conflicting

Several sources disagree:

- **Challenge deadline:** Comfy's live page says submissions close 3 Nov with winners on 10 Nov. The ComfyUI Wiki and the jutsuhq list say 30 Oct and 6 Nov ([comfy.org](https://comfy.org/open-source-higgsfield.md); [ComfyUI Wiki](https://comfyui-wiki.com/en/news/2026-10-15-open-source-higgsfield)). Winners are not yet known.
- **Cinema Studio focal lengths:** 12-135 mm versus 8/14/35/50/75 mm ([seedance.tv](https://www.seedance.tv/blog/higgsfield-cinema-studio-camera-settings); [vidpros](https://vidpros.com/higgsfield-cinema-studio/)).
- **Cinema Studio 4.0 resolution:** "up to 720p" conflicts with the homepage's 1080p Seedance option.
- **Genjutsu references:** sources say up to 40 or up to 30.
- **Higgsfield's cease-and-desist** to Comfy rests on Comfy's account alone ([RuntimeWire](https://runtimewire.com/article/comfy-says-higgsfield-sent-a-48-hour-cease-and-desist-over-open-source-challenge)).
- **"Openjutsu"** is the challenge track's name. The product it imitates is Genjutsu.
- **Licence reporting:** GitHub reports `NOASSERTION` for OpenField, open-higgsfield-skills and Vision Orchestrator, whose LICENSE files are MIT, MIT and Apache-2.0.

Several things nobody has verified:

- that CrossView, BFS, DMAD, Layout-To-Render, SCAIL-2 or LTX audio-reference-token freezing load in diffusers;
- the licences of the fal angle LoRA, the Remade LoRAs, ReCamMaster, TrajectoryCrafter, InfiniteTalk and the Lightricks camera LoRAs;
- whether most clones run end to end (open-higgsfield-skills admits "Most have not been run");
- whether H3, LTX or Kandinsky honour named camera moves at all.

## Conclusion

The surprise is how little of Higgsfield is model work. Its moat is a **vocabulary and a planning loop**, and dw already owns the harder parts of that loop: free validation with cost, a hard spend gate, replayable per-shot jobs with manifests, assessment probes, and two orchestration skills that already enforce cast lock and continuity states. Higgsfield's AI Director and Supercomputer are, functionally, dw's `script-to-video` with a shot card bolted on. The challenge entries independently rebuilt dw's architecture on ComfyUI, which is a strong signal the design is right. The work left is to give agents the director's words (closed enums compiled per family), a persisted shot document to fill, and a face-drift measurement to judge with.

The order follows from that. First, vocabulary, the shot document, the storyboard/animatic skill and the face probe, all measured as `trial` before promotion. Second, Layout-To-Render for blocking and re-shoot, Qwen-Image-Edit-2511 angles, and LatentSync, each gated on a 24 GB proof and a licence read. Wan only if a user needs a true actor swap. One thing should be decided before tier 1 lands: whether a named camera move is a promise or a hint. The evidence so far (0 of 6 invented moves, but one clip that drifted after 5 s) says hint. If so, the skills should present presets as hints, the way Higgsfield's own users echo the preset in the prompt and hope, and let the planned verifier turn that hint into a measured claim.
