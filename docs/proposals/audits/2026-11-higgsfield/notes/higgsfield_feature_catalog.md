# Higgsfield AI feature catalog and control surfaces (as of Nov 2026)

Source caveat for the whole file: most detailed material comes from Higgsfield's own help center / blog (vendor claims, but usually accurate on *what controls exist*). `geo.higgsfield.ai` pages are Higgsfield SEO marketing and were used only for feature names. Third-party guides (vidpros, memons, seedance.tv) disagree on numbers in places; conflicts are flagged. No independent hands-on review with measured quality was found. The name "Openjutsu" from the brief was not found anywhere; the real product is **Genjutsu** (Sept 2026).

## What is the full list of Higgsfield apps/features as of late 2026, and which are the most popular?

### Takeaway
Higgsfield is a multi-model aggregator (30+ third-party models: Seedance, Kling, Veo, Sora 2, Wan, Nano Banana, Hailuo) wrapped in preset-driven "apps". Its flagship surfaces in late 2026 are Cinema Studio 4.0, Genjutsu (video-to-video recast/swap), Supercomputer (agent), Marketing Studio, Canvas (node graph), Soul/Soul ID (own image model + identity), plus a long tail of single-purpose apps (Angles, Shots, Relight, Upscale, Effects, Lipsync Studio, Recast, Popcorn, Draw-to-Video). No usage/popularity statistics were found.

### Cited Findings
**Current homepage (fetched 2026-11-01)**
- Homepage features: Higgsfield Genjutsu ("One upload in. Endless new visions out." - takes motion from a source and recasts it with your characters/locations/products, or swaps specific elements), Genjutsu Lite (faster/cheaper), community Genjutsu presets, Higgsfield Playable (mobile game from one prompt), Content Factory (trending styles -> ready-to-post series), Cinema Studio 4.0, AI Influencer Studio, Supercomputer Agent, Higgsfield MCP for Claude; models named Kling 4.0, Seedance 2.5 (1080p option), Sora 2, Kling 3.0, Veo 3.1; "30+ models", "40+ creative AI tools", face swap, virtual try-on; Academy and Global Film Festival — [higgsfield.ai](https://higgsfield.ai/)
  - Note: the homepage text also says Supercomputer is "powered by GPT-6 Astra"; OpenAI customer-story pages reference "Astra" too — [openai.com Astra page](https://openai.com/es-419/index/higgsfield-from-prompt-to-production-with-astra/). Unverified; treat as vendor branding.

**Higgsfield's own top-level product list (help center "What is Higgsfield")**
- Soul (native image model family, cinematic/fashion/culture visuals); Soul ID (train a character from photos, face reused across images and videos); Cinema Studio (camera control, multi-shot scenes, physics-aware motion); Marketing Studio (product URL or image -> finished video ad in formats like UGC, unboxing, tutorial, TV spot, with audio and lip sync); Canvas (node-based board chaining models into reusable pipelines, real-time collaboration); Supercomputer (plain-language request -> plan -> assets); Higgsfield Audio (voiceover, voice cloning, translation); external models Seedance, Kling, Veo, Sora 2, Nano Banana — [Higgsfield help center](https://higgsfield.ai/creator-hub/help-center/getting-started/what-is-higgsfield)

**Per-app catalog (input -> knobs -> output -> model)**

| App | Input | Knobs | Output | Model / notes | Source |
|---|---|---|---|---|---|
| Camera Controls / DoP (2025 origin product) | single image (+ prompt) | pick one camera-move preset (see Q4), aspect 16:9/9:16/1:1 | ~5 s clip | Higgsfield "DoP" model; early Mar 2025 marketing | [memons cheat sheet](https://memons.ai/higgsfield-prompt-cheat-sheet); [X post Mar 31 2025](https://x.com/higgsfield_ai/status/1906748661322711305) |
| Effects library / Effects Mix | image | pick effect(s); Mix stacks two, e.g. "Set On Fire + Action Run", "Melting + Soul Jump", "Crash Zoom In + Invisible" | short clip | Effects Mix was live for all users incl. free trial; blog ~1 yr old (likely 2025) | [Higgsfield blog Effects Mix](https://higgsfield.ai/blog/Higgsfield-Effects-Mix-Lets-You-Tell-Bigger-Stories) (via search summary; page fetch did not surface text) |
| Viral presets | image | named template | short clip | "13 new viral presets" lists exist | [miracamp](https://www.miracamp.com/learn/video-editing/new-higgsfield-viral-presets) |
| Soul / Soul 2.0 | prompt (+ style presets) | aesthetic presets | photo | Higgsfield's own image model | [Soul intro](https://higgsfield.ai/soul-intro) |
| Soul ID | >=20 photos of a person | training (~3 min) | reusable identity for Soul generations | own model | [search summary of Higgsfield help](https://higgsfield.ai/creator-hub/help-center/ai-models/how-do-i-use-popcorn) |
| Popcorn (storyboard) | prompt and/or up to 4 image refs (cited "image one" etc.) | Manual (per-frame subject/setting/style/atmosphere) vs Auto (one prompt + frame count); up to 8 frames; aspect 3:4, 2:3, 3:2, 4:3, 16:9, 1:1, 9:16, Auto; Instadump mode (1 selfie + style pack of up to 15 refs) | consistent frame sequence; frames used as start frames for Kling/Veo, last frame chained to next | model unnamed | [Popcorn help](https://higgsfield.ai/creator-hub/help-center/ai-models/how-do-i-use-popcorn) |
| Shots | one image | select favorites, upscale to 4K | 9 cinematic angles | — | search summary, [Angles app](https://higgsfield.ai/app/angles) |
| Angles / Angles 2.0 | one JPG/PNG | new viewpoint | re-angled image | also shipped as Photoshop + Figma plugins | [Chase Jarvis review](https://chasejarvis.com/blog/higgsfield-angles-2-0-is-here-my-100-honest-review/); [Photoshop plugin](https://higgsfield.ai/plugins/photoshop) |
| Relight | photo (later video) | light-source angle, color temperature, brightness; "understands 3D geometry" | relit image | Jan 2026 photo launch; Aug 2026 "Video Relight and Color Palette" standalone tools | [PetaPixel Jan 2026](https://petapixel.com/2026/01/09/this-ai-app-lets-you-relight-your-photos/); [Relight blog](https://higgsfield.ai/blog/Relight-Director-Style-Cinematic-Lighting) |
| Upscale (video) | video | one-pass deflicker + stabilize + denoise + upscale | upscaled video | — | [Upscale blog](https://higgsfield.ai/blog/Upgrade-the-Quality-of-Old-Videos-using-Upscale) |
| Photoshop plugin | — | Angles, Mockup Studio, Expand, Remove BG, Relight | — | — | [Photoshop plugin](https://higgsfield.ai/plugins/photoshop) |
| Draw-to-Video | sketch/annotations over an image | draw to insert products, outfits, objects into frame | video | — | [UGC/ads blog](https://higgsfield.ai/blog/Create-Selling-Content-Using-Higgsfield) |
| Lipsync Studio / Speak | photo or character + script/audio | engine choice: Speak v2 / Speak 2.0, lipsync-2, InfiniteTalk, Kling AI Avatar (2.0), Kling Lipsync, Veo 3 | talking avatar / redubbed video | Sept 2025 blog; reviewer saw timing drift in close-ups on Kling-based lipsync | [Lipsync Studio blog Sep 18 2025](https://higgsfield.ai/blog/19fPMcKM5T0c8o8HQGdgyW); [best lip sync 2026](https://higgsfield.ai/blog/best-ai-lip-sync-video-generators-2026) |
| UGC Factory (older) -> Marketing Studio (current) | product idea / product URL or image | templates: Unboxing, Product Review, SaaS, Try-On, Tutorial, Localization; one avatar held through clip | creator-style ad with audio + lip sync | — | [UGC Factory blog](https://higgsfield.ai/blog/Higgsfield-UGC-Factory-Explained); [help center](https://higgsfield.ai/creator-hub/help-center/getting-started/what-is-higgsfield) |
| Recast | 5-15 s video + character image | swap subject keeping camera, lighting, voice (voice clone) | recast video | "Recast Studio" under Character, plus simpler Apps version; third-party claims | [perplexityaimagazine](https://perplexityaimagazine.com/?p=1015); [face swap guide](https://higgsfield.ai/blog/AI-Face-Character-Swap-in-Video-Photo-PRO-Guide) |
| Genjutsu (Sept 1 2026) | reference video 3-30 s + up to 40 reference images (another source: up to 30) | modes: Motion Transfer (keep motion/camera/timing/lip sync, rebuild cast + location + look) and Object Swap (change one element: character, wardrobe, product, location); Lite tier; community presets | new video | model unnamed; mixed reception (praise for motion transfer and lip sync; "scam" complaints over cost/access) | [creativeainews](https://www.creativeainews.com/blog/higgsfield-genjutsu-ai-video-motion-transfer-2026/); [postium](https://postium.ru/higgsfield-zapustila-genjutsu/) |
| Wan Animate / character swap | video + character image | — | video | Wan-based | [lilys.ai note](https://lilys.ai/de/notes/higgsfield-ai-20260128/higgsfield-wan-animate-video-magic) |
| Canvas | nodes | chain models, save reusable pipelines, real-time collab | any | — | [help center](https://higgsfield.ai/creator-hub/help-center/getting-started/what-is-higgsfield) |
| Higgsfield Assist | chat | prompt help | — | named only | [memons](https://memons.ai/higgsfield-prompt-cheat-sheet) |
| Supercomputer (agent) | natural-language goal | see Q6 | multi-asset deliverables | see Q6 | [Supercomputer help](https://higgsfield.ai/creator-hub/help-center/tools-and-workflows/how-do-i-use-supercomputer) |
| Faceless channel (Supercomputer skill) | topic | one of 12 visual styles | YouTube video up to 15 min: AI script, voiceover, 2D animation, music, subtitles | — | [Supercomputer help](https://higgsfield.ai/creator-hub/help-center/tools-and-workflows/how-do-i-use-supercomputer) |

- Directory listing of key features: Cinema Studio 4.0, multi-model generation, Soul ID and AI Cast, Canvas, Marketing Studio — [yespress](https://yespress.io/products/higgsfield.md)
- Video length extended from ~1 minute to over ten minutes (OpenAI customer page, Oct 2026) — [openai.com](https://openai.com/hy-AM/index/higgsfield-longer-videos-voice/)
- Pricing tracker: "Supercomputer" tier launched May 2026; Starter/Plus/Ultra prices raised Q2 2026 — [pricingsaas](https://pricingsaas.com/companies/higgsfield/diffs/2026W21) (contains an "[INVALID]" line; low confidence)

### Inferences
- Old/likely-deprecated surfaces: standalone "UGC Factory" appears superseded by Marketing Studio; Recast and Wan-Animate-style swap appear superseded by Genjutsu (the homepage now promotes Genjutsu, not Recast); single-preset "DoP" camera clips are superseded by Cinema Studio's genre/MoveSet controls but the preset library still exists.
- Popularity signals (qualitative only): the 2025 viral growth was driven by the one-click camera presets and Effects (Crash Zoom, Bullet Time, Set On Fire etc. on X/TikTok); 2026 attention centers on Cinema Studio and Genjutsu.

### Gaps
- No usage numbers per app; "most popular" cannot be stated with evidence.
- Effects library size and full effect-name list not retrieved (the Effects Mix blog text did not load).
- Exact status of Speak, Recast, Shots, UGC Factory in Nov 2026 nav was not verifiable (homepage fetch exposes only promoted items).

## For Cinema Studio / director mode: what exact controls (camera body, lens, focal length, aperture, genre, movement) are exposed?

### Takeaway
Cinema Studio is a "build the rig" UI: pick camera body, lens, focal length, aperture per shot; pick genre/style/lighting/palette/MoveSet globally. It has been versioned fast (v1 Dec 2025 -> 4.0 by late 2026) and users can still select older versions (2.0-4.0). The control vocabulary changed between versions, and sources disagree on the exact lists.

### Cited Findings
- Version history: v1 shipped Dec 18 2025 with camera, lens, focal length (movement + slow motion later); v1.5 added aperture and recommended camera presets; v2/2.5 added Elements, 3D Scene, Character Motions, locations, Soul Cinema; v3.0 April 2026 ("turbulent" rollout); v3.5 ten days later added multishot, genre-based camera, video-first generation, projects, AI Director; v4.0 up to 720p, up to 30 s, Montage Pacing, AI Cast, Cinematic Locations, Soul Cinema models, multi-model, "anti-slop camera pipeline" — [How we built Cinema Studio](https://higgsfield.ai/blog/how-we-built-cinema-studio)
- Philosophy quote: "Instead of describing a look in words, you build the frame the way a cinematographer would"; a former DP shaped controls; lenses hand-tuned — "we looked for each lens's character, and we amplified it"; settings + metadata saved with each image so a still used as a video start frame carries the look — [How we built Cinema Studio](https://higgsfield.ai/blog/how-we-built-cinema-studio)
- Version selector 2.0-4.0, 4.0 default. 4.0 runs on Seedance 2.5 — [Cinema Studio help](https://higgsfield.ai/creator-hub/help-center/tools/how-do-i-use-cinema-studio)
- **2.5:** up to 3 Soul Cast characters (saved Elements or new); location in prompt; post "Colorgrade": Color Temperature, Contrast, Saturation, Sharpness, Film Grain, Highlights, Exposure; "Save to Elements" — same source
- **3.0:** up to 9 references (characters, locations, scene details); genres General, Action, Horror, Comedy, Noir, Drama, Epic; speed ramps Linear, Auto, Flash In, Flash Out, Slow-mo, Bullet Time, Impact, Ramp Up; native audio (SFX, speech, music) in same pass; video uploads up to 15 s — same source
- **3.5:** per-shot Camera, Lens, Focal length, Aperture; global Genre, Style, Lighting, Color Palette, Camera MoveSet Style (8 palettes, 10 MoveSet styles "inspired by real-world cinematographers", 6 lighting presets); Elements = Characters, Locations, Props; AI Director (Claude Chat panel) breaks a script into shots with camera params pre-filled; shared projects — same source
- **4.0 Director's Panel:** Film Setup genres (General, Action, Epic, Drama, Comedy, Horror, Noir), Era (60s-2020s), Tempo (Single Shot ... Chaotic); camera bodies "35mm Film", "8mm Film", "DV Camcorder"; lens families clean sharp / anamorphic / vintage; aperture control; moves incl. "POV", "Robot Arm", "Helicopter Shot"; 50+ color-grading templates; lighting 6 presets or custom color/brightness/diffusion/angle; up to 50 references (images, videos, Soul Cast characters); character **emotion slider** (performance expressiveness); clips up to 30 s; modes References / Forward Extend / Backward Extend (up to 30 s) / Edit Video (regional object/face edits) — same source
- 2.0/3.5-era camera list (third party, June 2026): bodies RED V-Raptor, Sony Venice, ARRI Alexa, IMAX, Panavision DXL2; lenses Petzval, JDC Xtal Xpress, Cooke S4, Panavision C-Series, Lensbaby, Zeiss Ultra Prime, Helios, ARRI Signature Prime, Canon K-35, Laowa Macro, Hawk V-Lite; focal lengths 8, 14, 35, 50, 75 mm — [vidpros](https://vidpros.com/higgsfield-cinema-studio/)
- Conflict on focal length: help center (per search summary) says 12-135 mm; Seedance guide and vidpros say 8/14/35/50/75 mm — [seedance.tv](https://www.seedance.tv/blog/higgsfield-cinema-studio-camera-settings); [vidpros](https://vidpros.com/higgsfield-cinema-studio/)

### Inferences
- Trajectory: from literal hardware names (ARRI/RED/Cooke) toward abstract "looks" (35mm Film / 8mm / DV, lens families, era, tempo) as multi-shot made per-shot camera specs less useful — the blog says camera control "moved up a level to genres".
- Under the hood these controls are almost certainly prompt-composition + curated per-model prompt templates (an "enhancer" rewrites sparse input), not model conditioning — consistent with running on third-party Seedance.

### Gaps
- Numeric aperture values and the 10 MoveSet style names were not found.
- Output resolution for 3.x not stated; 4.0 "up to 720p" (blog FAQ) conflicts with homepage "Seedance 2.5 1080p" option.

## How does Higgsfield achieve character/subject consistency across shots (Soul ID, reference images)?

### Takeaway
Three layers: a trained identity (Soul ID, >=20 photos, ~3 min) for its own Soul image model; reusable named "Elements"/"Soul Cast"/"AI Cast" assets (characters, locations, props) that are injected as references into third-party video models (up to 9 refs in 3.0, up to 50 in 4.0); and project-level locked style (lighting, optics, tempo) plus within-sequence consistency in Popcorn.

### Cited Findings
- Soul ID needs >=20 photos, trains ~3 min; holds one face across separate Soul generations; Popcorn instead keeps consistency within one sequence and can take a Soul character as a reference — [Popcorn help](https://higgsfield.ai/creator-hub/help-center/ai-models/how-do-i-use-popcorn)
- Elements (Characters, Locations, Props) saved and reused; Soul Cast characters (3 in 2.5); refs 9 (3.0) -> 50 (4.0) — [Cinema Studio help](https://higgsfield.ai/creator-hub/help-center/tools/how-do-i-use-cinema-studio)
- Projects keep light, optics and tempo consistent across all generations in the project — [How we built Cinema Studio](https://higgsfield.ai/blog/how-we-built-cinema-studio)
- Settings metadata saved with each still so it carries into image-to-video — same source
- Popcorn workflow: last frame of one clip chained as start of next — [Popcorn help](https://higgsfield.ai/creator-hub/help-center/ai-models/how-do-i-use-popcorn)
- Marketing Studio/UGC holds one avatar across a clip — [UGC Factory blog](https://higgsfield.ai/blog/Higgsfield-UGC-Factory-Explained)

### Inferences
- Video-side consistency depends on the underlying model's reference-to-video support (Seedance 2.x / Kling multi-reference); Higgsfield's contribution is asset management + numbered reference citation ("image one") in prompts.

### Gaps
- How Soul ID is implemented (LoRA vs. adapter) is not disclosed.
- An Oct 2026 analysis on character consistency exists but was not read — [aiproductivity.ai](https://aiproductivity.ai/blog/higgsfield-consistent-ai-character/)

## What camera-motion preset vocabulary do they use (names of moves)?

### Takeaway
A named library (counts cited from 50+ to 100+) grouped roughly as zooms, dollies, vertigo, orbit/arc, aerial, signature FX, and time-lapse. Users pick a preset; the cheat-sheet convention is to also echo the preset in the prompt.

### Cited Findings
- Preset names (Aug 2026 cheat sheet): Zoom — Crash Zoom In, Crash Zoom Out, Rapid Zoom In, Rapid Zoom Out, YoYo Zoom; Dolly — Dolly In/Out/Left/Right, Super Dolly In/Out, Double Dolly; Vertigo — Dolly Zoom In/Out; Orbit/Arc — 360 Orbit, Arc Left, Arc Right, Lazy Susan; Aerial — FPV Drone, Crane Up/Down, Jib Up/Down, Overhead; Signature FX — Bullet Time, Snorricam, Robo Arm, Whip Pan, Through Object In/Out, Object POV; Time — Hyperlapse, Timelapse Human, Timelapse Landscape, Timelapse Glam. Duration ~5 s for Higgsfield's own moves; prompt convention "(Preset: Crash Zoom In · 9:16 · 5s)" — [memons cheat sheet](https://memons.ai/higgsfield-prompt-cheat-sheet)
- Other sources confirm Dolly In, Crane Up/Down, Tracking, Whip Pan, Snorricam, Bullet Time, Boom; counts "50+" vs "20+" vs "100+" — [BibiGPT](https://bibigpt.co/en/features/higgsfield-ai-video-generator-explained); [kolbo](https://kolbo.ai/blog/higgsfield-suite-100-camera-presets)
- Crash Zoom In, bullet time, super dollies, robo arms promoted Mar 31 2025 — [X](https://x.com/higgsfield_ai/status/1906748661322711305)
- Cinema Studio 3.0 speed ramps: Linear, Auto, Flash In, Flash Out, Slow-mo, Bullet Time, Impact, Ramp Up; 4.0 moves incl. POV, Robot Arm, Helicopter Shot — [Cinema Studio help](https://higgsfield.ai/creator-hub/help-center/tools/how-do-i-use-cinema-studio)
- Combo motions (Effects Mix pairs camera + effect, e.g. Crash Zoom In + Invisible) — [Effects Mix blog](https://higgsfield.ai/blog/Higgsfield-Effects-Mix-Lets-You-Tell-Bigger-Stories)

### Inferences
- Names mix real grip terms (Snorricam, Jib, Lazy Susan, Robo Arm) with TikTok-native ones (YoYo Zoom, Timelapse Glam) — the vocabulary is a UX product in itself.

### Gaps
- No official, complete, dated preset list; cheat-sheet list is third party.
- Intensity/strength knob per preset: not confirmed in sources.

## How do they handle multi-shot sequences, storyboards, and audio (dialogue, music, SFX)?

### Takeaway
Multi-shot arrived in Cinema Studio 3.5 (April 2026) with genre-driven camera and a "Tempo"/"Montage Pacing" control in 4.0; storyboards via Popcorn (up to 8 frames); continuation via Forward/Backward Extend; audio is native from the underlying model (3.0+) plus a separate Higgsfield Audio suite and Lipsync Studio.

### Cited Findings
- 3.5 added multishot; single continuous camera moves became insufficient so camera control moved to genres — [How we built Cinema Studio](https://higgsfield.ai/blog/how-we-built-cinema-studio)
- 4.0: Tempo (Single Shot -> Chaotic), clips up to 30 s, Forward/Backward Extend, Edit Video — [Cinema Studio help](https://higgsfield.ai/creator-hub/help-center/tools/how-do-i-use-cinema-studio)
- Enhancer example input "generate me a 50-shot battle scene in 15 seconds" — [How we built Cinema Studio](https://higgsfield.ai/blog/how-we-built-cinema-studio)
- 3.0 audio: SFX, speech, background music in same pass — [Cinema Studio help](https://higgsfield.ai/creator-hub/help-center/tools/how-do-i-use-cinema-studio)
- Higgsfield Audio: voiceover, voice cloning, translation — [help center](https://higgsfield.ai/creator-hub/help-center/getting-started/what-is-higgsfield)
- Lipsync engines: Speak v2, lipsync-2, InfiniteTalk, Kling AI Avatar, Kling Lipsync, Veo 3 — [Lipsync Studio blog](https://higgsfield.ai/blog/19fPMcKM5T0c8o8HQGdgyW)
- Popcorn storyboards up to 8 frames, then animate per frame and chain last frame — [Popcorn help](https://higgsfield.ai/creator-hub/help-center/ai-models/how-do-i-use-popcorn)
- Long-form: videos extended to >10 min (Oct 2026) — [openai.com](https://openai.com/hy-AM/index/higgsfield-longer-videos-voice/)

### Inferences
- Multi-shot inside one generation is delegated to the base model's native multi-shot ability (Seedance 2.x); Higgsfield supplies pacing/genre presets around it. Long-form >10 min is presumably agent-stitched.

### Gaps
- Music generation model and SFX controls not identified.

## Are there agent/assistant features that plan shots from a script?

### Takeaway
Yes. Cinema Studio 3.5's AI Director (a Claude chat panel) breaks a script into shots with camera parameters pre-filled; it was spun out into Supercomputer, a chat agent that plans multi-step jobs and routes each sub-task to a model. Also MCP for Claude, a ChatGPT plugin, and a CLI.

### Cited Findings
- AI Director (Claude Chat) breaks a script into shots with camera params pre-filled — [Cinema Studio help](https://higgsfield.ai/creator-hub/help-center/tools/how-do-i-use-cinema-studio); it later became Supercomputer — [How we built Cinema Studio](https://higgsfield.ai/blog/how-we-built-cinema-studio)
- Supercomputer: "agentic workspace"; plans, generates across image/video/audio, re-runs only affected steps on revision; auto-routes generation models (Soul, Cinema Studio, Seedance, Nano Banana); chat LLM selectable (Anthropic, OpenAI, Google, xAI, DeepSeek) or auto; Memory; connectors (Telegram, Slack, TikTok, Gmail, Google Drive, Notion); can build/deploy apps/sites; skills usable via MCP in Claude or Higgsfield Plugin in ChatGPT; per-step credits shown live — [Supercomputer help](https://higgsfield.ai/creator-hub/help-center/tools-and-workflows/how-do-i-use-supercomputer)
- Built on open-source Hermes Agent; memory mostly manual at launch (reviewer); launch date conflicting (May vs later) — [popularaitools review](https://popularaitools.ai/blog/higgsfield-supercomputer-review/); [Higgsfield build blog](https://higgsfield.ai/blog/how-we-built-supercomputer)
- CLI plus agent features cited as signal of a "full AI production studio" (Oct 2026 newsletter) — [mean.ceo](https://blog.mean.ceo/?p=11502)

### Inferences
- UX philosophy overall: presets first (named moves, genres, eras, templates) with an LLM "enhancer" hiding prompt writing; fine control (lens/aperture) offered but increasingly abstracted; consistency via saved asset libraries + projects; and now an agent on top that picks models and composes steps — closely analogous to dw's workflow catalog + MCP agent loop.

### Gaps
- No public detail on how AI Director's shot breakdown is structured (fields per shot) beyond "camera parameters pre-filled".
