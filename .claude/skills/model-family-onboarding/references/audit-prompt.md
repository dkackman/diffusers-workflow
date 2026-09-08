# Audit prompt

Dispatch with the Agent tool, `subagent_type: general-purpose`, the most
capable model available, `run_in_background: true`. Replace every
`<...>`. Keep the structure: the primary-first source order and the
claim-by-claim verdict are what make the report usable.

```
You are verifying whether a repository's knowledge about the <MODEL FAMILY>
model (Hugging Face repo <ORG/REPO>, and <any sibling repos: LoRAs,
upscalers, enhancers>) is correct and current. This is research only. Do not
edit any file in the repository. Write your report to
<SCRATCHPAD>/<family>-research.md and return only a five-line summary.

Today is <DATE>.

The repository's current claims live in:
- <path to the family's templates README> (the reading-order map and its
  conventions).
- The templates in <path>/*.json. Read all of them: note the pipeline
  classes, schedules and step counts, guidance, resolution and frame-count
  rules, fps, quantization and offload choices, any LoRA and its version,
  and how prompts are written (each template's default prompt variable, and
  any enhancer's system prompt).
- <any builtin under dw/workflows/ that carries the family's prompt format>.
- Anything about the family in docs/RECIPES_24GB.md, docs/ACCELERATION.md and
  docs/QUANTIZATION.md (grep for <family name>).
- The diffusers pipeline source installed in the repo's venv: after
  `source <repo>/activate` (if that fails, locate the venv under the repo
  and use its python directly), find the pipeline directory with
  `python -c "import diffusers,os;print(os.path.dirname(diffusers.__file__))"`
  and read the pipeline docstrings, `utils.py` constants and any
  prompt-enhancement helper. That source is a primary source for what the
  pipeline accepts and defaults to.

Then gather sources, PRIMARY first, with URL and date for each:
1. The Hugging Face model card(s): WebFetch https://huggingface.co/<ORG/REPO>
   and the raw README at https://huggingface.co/<ORG/REPO>/raw/main/README.md
   (the raw form may be gated; say so and use the HTML card). Follow any
   docs/ links on the card - prompt-writing guides live there.
2. The vendor's own GitHub repository, its README, any prompting guide or
   agent skill it ships, the recommended inference settings, a CHANGELOG,
   and the technical report or paper if one exists.
3. The diffusers documentation page for the pipeline
   (https://huggingface.co/docs/diffusers/main/en/api/pipelines/<name>).
4. Only then, secondary sources: community prompt guides, blog posts, GitHub
   issues, forum threads. Weight a claim higher when several independent
   secondary sources repeat it AND it does not contradict a primary source.
   Note when secondary sources merely restate the model card.

Report, in markdown:
- **Sources**: a table of every source used: URL, type (primary/secondary),
  date, one line on what it contributes.
- **Claim-by-claim verdict**: walk the repository's claims (every distinct
  rule in any system prompt, every README statement, every template's
  settings and prompt style) and mark each CONFIRMED (cite the source),
  CONTRADICTED (cite, and state what the source says instead), or UNSOURCED
  (no source found either way). Be specific: field names, ordering, exact
  instruction sentences, vocabularies, the frame-count and resolution rules,
  fps, schedules and step counts, guidance values, LoRA versions, the prompt
  style the model was trained on (length, tense, structure, whether audio is
  described separately), the enhancer's role.
- **Missing knowledge**: recommendations the sources make that the
  repository does not carry (new prompting advice, new task modes, parameter
  ranges, known failure modes, newer checkpoints or LoRAs, changed defaults).
  Date each.
- **Assessment**: is the repository's knowledge an accurate transcription of
  the vendor's guidance, a reasonable inference from examples, or partly
  wrong? Which parts would you trust in a skill that teaches an agent to
  compose this family's workflows and write its prompts by hand, and which
  need correction? Under 300 words.

Be rigorous about dates: a source older than the model's current checkpoint
may describe a previous version, and sibling releases (<e.g. LTX-2, 2.3,
2.5>) differ. Do not fabricate URLs; if a fetch fails, say so and move on.
```

For a family new to the catalog, replace the "repository's current claims"
block with "The repository has no entries for this family yet" and add to
the report a fifth section, **What the templates should teach**: the
reading-order list of one-idea-per-template that the family's README will
carry, from the vendor's own examples.
