"""The catalog's two trees, and the invariant each carries.

templates/ teaches a pattern and models/ records a hardware fact. The
distinction is only useful if it is legible from outside the file, so a
template must describe itself and a model config must name the template it
configures.
"""

import glob
import json
import os
import re

import pytest

from dw.server.app import attach_observed, workflow_details
from dw.server.catalog_shape import (
    SUMMARY_LIMIT,
    derive_catalog_metadata,
    project_listing,
)
from dw.workflow_sources import WorkflowSource, listing
from tests.test_examples import BUILTIN_DIR, REPO_ROOT, get_example_files

TEMPLATES = [f for f in get_example_files() if f.startswith("workflows/templates/")]
MODEL_CONFIGS = [f for f in get_example_files() if f.startswith("workflows/models/")]


def test_there_are_templates():
    assert TEMPLATES


@pytest.mark.parametrize("path", TEMPLATES)
def test_every_template_describes_itself(path):
    """A template is read before it is run - by a person choosing one and by an
    agent matching a request's shape. An undescribed template is invisible to
    both, whatever its filename says."""
    definition = json.load(open(os.path.join(REPO_ROOT, path), encoding="utf-8"))

    assert definition.get("description", "").strip(), f"{path} has no description"


def test_there_are_model_configs():
    assert MODEL_CONFIGS


@pytest.mark.parametrize("path", MODEL_CONFIGS)
def test_every_model_config_names_the_template_it_configures(path):
    """A model config is a tuned instance of a pattern. Without the pointer it
    is just another entry in the list, which is the problem this restructure
    exists to fix."""
    definition = json.load(open(os.path.join(REPO_ROOT, path), encoding="utf-8"))
    configures = definition.get("configures", "")

    assert configures, f"{path} has no 'configures'"
    target = os.path.join(REPO_ROOT, "workflows", f"{configures}.json")
    assert os.path.isfile(target), (
        f"{path} configures '{configures}', which is not a workflow ({target})"
    )


def load(path):
    return json.load(open(os.path.join(REPO_ROOT, path), encoding="utf-8"))


# The regression net for the derivation rules: real templates whose shape
# is subtle enough that a rule change should have to answer to them.
EXPECTED_SHAPES = {
    "workflows/templates/text-to-image.json": ("image", []),
    "workflows/templates/minimax/storyboard.json": (
        "shot",
        ["has-audio", "identity-referenced"],
    ),
    "workflows/templates/minimax/dialogue-short.json": (
        "sequence",
        ["has-audio", "identity-referenced"],
    ),
    "workflows/templates/minimax/music-video.json": (
        "sequence",
        ["has-audio", "identity-referenced"],
    ),
    "workflows/templates/minimax/chained-segments.json": (
        "shot",
        ["chained", "has-audio", "image-conditioned", "needs-input-media"],
    ),
    "workflows/templates/ltx2/chained-segments.json": (
        "shot",
        ["chained", "has-audio", "image-conditioned", "needs-input-media"],
    ),
    "workflows/templates/ltx2/keyframes.json": (
        "shot",
        ["has-audio", "image-conditioned", "needs-input-media"],
    ),
    "workflows/templates/image-variation.json": ("image-edit", ["needs-input-media"]),
    "workflows/templates/segment-and-inpaint.json": (
        "image-edit",
        ["needs-input-media"],
    ),
    "workflows/templates/describe-and-regenerate.json": (
        "image-set",
        ["composes-workflows", "needs-input-media"],
    ),
    "workflows/templates/compose-workflows.json": (
        "shot",
        ["composes-workflows", "image-conditioned"],
    ),
    "workflows/templates/generate-speech.json": ("audio", ["has-audio"]),
    "workflows/templates/assemble-and-score.json": ("sequence", ["needs-input-media"]),
    "workflows/templates/image-processors.json": ("utility", []),
}

# Templates that genuinely are utilities - processing with no generation.
# Anything else that derives 'utility' is a rule that missed.
UTILITIES = {
    "workflows/templates/audio-trim-fade.json",
    "workflows/templates/image-processors.json",
    "workflows/templates/recenter-crop.json",
    "workflows/templates/segment.json",
    "workflows/templates/transcribe-audio.json",
    "workflows/templates/upscale-spandrel.json",
}


@pytest.mark.parametrize("path, expected", sorted(EXPECTED_SHAPES.items()))
def test_the_rules_read_these_templates_as_expected(path, expected):
    meta = derive_catalog_metadata(load(path))
    assert (meta["shape"], meta["traits"]) == expected


@pytest.mark.parametrize("path", TEMPLATES)
def test_no_template_falls_through_to_utility(path):
    meta = derive_catalog_metadata(load(path))
    if meta["shape"] == "utility":
        assert path in UTILITIES, (
            f"{path} derived 'utility' - a rule missed it, or add it to UTILITIES"
        )
    else:
        assert path not in UTILITIES, (
            f"{path} is listed as a utility but derives {meta['shape']}"
        )


@pytest.mark.parametrize("path", TEMPLATES + MODEL_CONFIGS)
def test_a_declaration_must_differ_from_the_derivation(path):
    """An override that repeats the rules is noise that will rot when the
    rules or the file change. Declare only what derivation gets wrong."""
    definition = load(path)
    meta = derive_catalog_metadata(definition)
    stripped = {
        k: v for k, v in definition.items() if k not in ("shape", "traits", "summary")
    }
    derived = derive_catalog_metadata(stripped)
    for key in meta["declared"]:
        assert meta[key] != derived[key], (
            f"{path} declares {key}={meta[key]!r}, which derivation already produces"
        )


@pytest.mark.parametrize("path", TEMPLATES)
def test_every_template_has_a_summary_that_fits(path):
    meta = derive_catalog_metadata(load(path))
    assert meta["summary"], f"{path}: description has no first sentence"
    assert len(meta["summary"]) <= SUMMARY_LIMIT
    assert not meta["summary_truncated"], (
        f"{path}: first sentence runs past {SUMMARY_LIMIT} chars - shorten it or declare 'summary': {meta['summary']!r}"
    )


BUILTINS = sorted(
    os.path.relpath(os.path.join(BUILTIN_DIR, name), REPO_ROOT)
    for name in os.listdir(BUILTIN_DIR)
    if name.endswith(".json")
)


def test_workflow_ids_are_unique_across_the_catalog():
    """A duplicate id is a step-cache collision waiting to happen, and it
    makes job history ambiguous about which workflow ran."""
    seen = {}
    for path in TEMPLATES + MODEL_CONFIGS + BUILTINS:
        identity = load(path).get("id")
        assert identity, f"{path} has no id"
        assert identity not in seen, (
            f"{path} and {seen[identity]} share id {identity!r}"
        )
        seen[identity] = path


@pytest.mark.parametrize("path", TEMPLATES + MODEL_CONFIGS)
def test_a_declared_cost_is_well_formed(path):
    cost = load(path).get("cost")
    if cost is None:
        return
    assert isinstance(cost, list) and cost, (
        f"{path}: cost must be a non-empty list or absent"
    )
    for entry in cost:
        assert entry["device"] in ("cuda", "mps", "cpu"), path
        assert isinstance(entry["vram_gb"], (int, float)) and entry["vram_gb"] >= 0, (
            path
        )
        assert isinstance(entry["minutes"], (int, float)) and entry["minutes"] >= 0, (
            path
        )


def per_entry_problems(definition):
    """The checks a per_entry cost has to satisfy: its 'variable' is a
    for_each list this definition actually reads, and 'entries' is that
    list's default length - a stale figure left behind by an edited
    default is otherwise invisible until the wrong minutes gets quoted."""
    from dw.for_each import list_fields

    problems = []
    for entry in definition.get("cost") or []:
        per_entry = entry.get("per_entry")
        if per_entry is None:
            continue
        lists = list_fields(definition)
        variable = per_entry["variable"]
        if variable not in lists:
            problems.append(f"{variable} is not a for_each list")
            continue
        default = (definition.get("variables") or {}).get(variable)
        if not (isinstance(default, list) and len(default) == per_entry["entries"]):
            problems.append(
                f"entries={per_entry['entries']} does not match the default "
                f"list's length ({default!r})"
            )
    return problems


@pytest.mark.parametrize("path", TEMPLATES + MODEL_CONFIGS)
def test_a_per_entry_cost_names_a_list_the_steps_read(path):
    """per_entry is measured over the default list, so the variable it
    names must be one a for_each expands and `entries` must be that
    list's length - an edited default that forgot the cost block fails
    here."""
    definition = load(path)
    problems = per_entry_problems(definition)
    assert not problems, f"{path}: {problems}"


def test_per_entry_problems_reports_a_mismatched_entries_count():
    """Synthetic case so the check above is not vacuous while no bundled
    template carries a per_entry cost yet."""
    definition = {
        "variables": {"shots": [{"name": "a"}, {"name": "b"}]},
        "steps": [
            {
                "name": "shot",
                "for_each": "variable:shots",
                "pipeline": {
                    "configuration": {"component_type": "{Fake}"},
                    "from_pretrained_arguments": {"model_name": "m"},
                    "arguments": {"name": "item:name"},
                },
                "result": {"content_type": "image/jpeg"},
            }
        ],
        "cost": [
            {
                "device": "cuda",
                "vram_gb": 24,
                "minutes": 10,
                "per_entry": {"variable": "shots", "minutes": 5, "entries": 5},
            }
        ],
    }
    problems = per_entry_problems(definition)
    assert problems and "entries=5" in problems[0]


# The catalog's descriptions quote identifiers in single quotes, not
# backticks - JSON strings, where a backtick reads as a stray character.
QUOTED = re.compile(r"'([a-z_][a-z0-9_]*)'")

# Mentions that are deliberately not this workflow's variables: a
# sub-workflow's argument, a step argument, a chain field or a result field.
# One line per entry saying what the name actually is.
LEGITIMATE_MENTIONS = {
    # the argument the composed image-to-video workflow receives
    "workflows/templates/compose-workflows.json": {"image"},
    # a 'result' field, and the point is that this workflow omits it
    "workflows/templates/generate-speech.json": {"sample_rate"},
    # arguments of the image_to_text step, edited into the file rather than passed
    "workflows/templates/image-to-text.json": {"model_name", "prompt"},
    # a field of the step's 'chain' block
    "workflows/templates/minimax/chained-segments.json": {"trim_frames"},
    # fields of a 'shots' list entry, not this workflow's own variables
    "workflows/templates/minimax/dialogue-short.json": {"prompt", "num_frames"},
    # the fl2va sub-workflow's argument, named to say this one leaves it unset
    "workflows/templates/minimax/last-frame-only.json": {"image"},
    # a 'result' field the modular pipeline needs declared
    "workflows/templates/minimax/music.json": {"sample_rate"},
}


def _variable_names_in_catalog():
    names = set()
    for path in TEMPLATES + MODEL_CONFIGS:
        names |= set((load(path).get("variables") or {}).keys())
    return names


def _mentioned(definition):
    return set(QUOTED.findall(definition.get("description", "")))


@pytest.mark.parametrize("path", TEMPLATES + MODEL_CONFIGS)
def test_a_description_names_only_variables_the_workflow_declares(path):
    """A description that says 'num_frames' for a workflow with no such
    variable sends an agent to pass an argument nothing reads. Restricted
    to identifiers that are variable names somewhere in the catalog, so a
    quoted task or type name is not a false positive, and to the mentions
    LEGITIMATE_MENTIONS does not account for."""
    definition = load(path)
    declared = set((definition.get("variables") or {}).keys())
    catalog_variables = _variable_names_in_catalog()
    allowed = LEGITIMATE_MENTIONS.get(path, set())
    undeclared = (_mentioned(definition) & catalog_variables) - declared - allowed
    assert not undeclared, (
        f"{path} describes {sorted(undeclared)} but declares no such variable"
    )


def test_the_drift_check_actually_matches_something():
    """The quoting convention is the whole test: match backticks instead and
    every set is empty and every assertion passes for nothing."""
    matched = [
        path
        for path in TEMPLATES
        if _mentioned(load(path)) & _variable_names_in_catalog()
    ]
    assert matched, (
        "no template description quotes a catalog variable name - the pattern is wrong"
    )


def test_no_stale_entry_in_the_allowlist():
    for path, names in LEGITIMATE_MENTIONS.items():
        assert path in TEMPLATES + MODEL_CONFIGS, (
            f"{path} is allowlisted but not in the catalog"
        )
        assert names <= _mentioned(load(path)), (
            f"{path} no longer mentions {sorted(names - _mentioned(load(path)))}"
        )


# Spec targets, as chars / 4. The listing is the first thing an agent reads;
# these are the ceilings that keep it readable rather than skimmed.
# was 5_500, then 6_000 with the informative MiniMax/LTX-2 summaries;
# raised to 6_400 for the declared variable constraints (#96), which are what
# stop a consumer picking a frame count the model refuses two minutes into a
# run - carried terse (`17*n+5, 124-345, rounds up`), the reason only in the
# full listing; then to 7_100 for `observed_minutes`/`observed_runs` (#93),
# two keys per workflow this box has actually run. Measured 6_276 with no
# history, 6_381 against a real server's (34 named workflows, 106 runs), and
# 7_056 here, where every one of the 65 compact entries carries a figure -
# which is the ceiling the budget has to hold, and why the measurement below
# attaches one. A budget checked against an empty jobs table would pass while
# the running server overran it. Two numeric keys rather than one terse
# phrase (the shape `constraints` uses) costs 179 tokens and is worth them:
# an agent quoting a price should read a number, not parse a sentence, and
# the curated `cost` beside it is structured too. Then to 7_600 for the H3
# checkpoint knobs (#147/#148/#149), measured at 7_484 here: `video_shift`,
# `audio_shift` and `lora_alpha` across seventeen templates, the five adapter
# names the six reference templates gained with the Ref2VA turbo LoRA, and the
# 768p entry. They earn it because they are what makes a checkpoint swap an
# argument rather than a new template - a 768p turbo LoRA on the 544p sigma
# schedule is a silent quality failure that costs a full run to discover, and
# the alpha a file declares is not always the alpha upstream runs it at.
# Worth noting that variable *names* are now the largest single share of this
# listing; if it needs raising again, the question to ask first is whether
# every name belongs in the compact view or only the ones a caller is likely
# to set.
# Then to 7_650 for the bound a list entry's field carries (#145), measured
# at 7_611: `dialogue-short`'s `shots` entries are where a frame count is
# most likely typed by hand, and `17*n+5, 124-345, rounds up` beside
# `num_frames` in the `lists` block is the half of #96 that stops the next
# caller picking 61. It is the only such line in the catalog today, and a
# rule that reaches only an entry field is no longer repeated in the
# top-level `constraints` block of the compact view, so the net cost of the
# feature here is eleven tokens.
# Then to 8_100 for three new LTX-2.5 templates (#151, #152), measured at
# 8_026: `reference-sheet`, `restore-deblur` and `restore-decompression` at
# roughly 125 tokens each. This is the cost of catalog entries existing
# rather than of anything said about them - the listing is what an agent
# reads to find a shape, and before these the LTX-2.5 family had no
# reference or identity route at all and no restoration route that was not
# a re-render.
# Then to 8_250 for best-of-n-to-video, measured at 8_227: the select/judge
# reducer's first template, whose for_each `lists` entry and six variables
# (candidates, num_inference_steps, prompt, rubric, scale, video_prompt)
# cost about 128 tokens on their own.
COMPACT_BUDGET = 8_250
FILTERED_BUDGET = 1_500


def _tokens(payload):
    return len(json.dumps(payload)) / 4


class _every_workflow_observed:
    """An `ObservedCosts` that answers for every entry, so the budget is
    measured against the widest listing the server can produce rather than
    against the empty jobs table a test fixture has."""

    def __init__(self, details):
        self._names = set(details)

    def refresh(self):
        return True

    def observed(self, name, definition, arguments=None, *, fresh=True, workspace=None):
        if name not in self._names:
            return None
        return {
            "device": "cuda",
            "name": "NVIDIA GeForce RTX 3090",
            "runs": 10,
            "comparable": "drivers",
            "cold_minutes": 11.84,
            "cold_runs": 10,
            "cold_range_minutes": [10.03, 14.95],
        }


def test_the_compact_listing_fits_the_budget():
    found = listing(
        [WorkflowSource(os.path.join(REPO_ROOT, "workflows"), "workspace", True)]
    )
    details = workflow_details(found)
    # As the server answers it: every workflow carrying the observed figures
    # it would carry on a box that had run them all
    attach_observed(details, _every_workflow_observed(details))

    compact = project_listing(details, view="compact")
    assert _tokens(compact) <= COMPACT_BUDGET, (
        f"compact listing is {_tokens(compact):.0f} tokens"
    )

    sequences = project_listing(details, view="compact", shape="sequence")
    assert sequences, "no template derives 'sequence'"
    assert _tokens(sequences) <= FILTERED_BUDGET, (
        f"shape=sequence is {_tokens(sequences):.0f} tokens"
    )


def _walk(value):
    if isinstance(value, dict):
        for key, inner in value.items():
            yield key, inner
            yield from _walk(inner)
    elif isinstance(value, list):
        for inner in value:
            yield from _walk(inner)


REMOTE_CODE_KEYS = ("trust_remote_code", "custom_pipeline")


BUILTINS = sorted(
    os.path.relpath(os.path.join(BUILTIN_DIR, name), REPO_ROOT)
    for name in os.listdir(BUILTIN_DIR)
    if name.endswith(".json")
)


@pytest.mark.parametrize("path", TEMPLATES + MODEL_CONFIGS + BUILTINS)
def test_no_catalog_entry_needs_trust_workflows(path):
    """A server started without --trust-workflows refuses either key at load,
    after validation has passed. An entry that carries one runs only where an
    operator lowered that guard, and an agent has no way to find out whether
    this server did - so the catalog carries neither, nor do the packaged
    builtin: sub-workflows a template can compose, and a workflow that needs
    remote code is written elsewhere."""
    definition = load(path)

    found = [key for key, _ in _walk(definition) if key in REMOTE_CODE_KEYS]

    assert not found, f"{path} sets {', '.join(found)}, which needs --trust-workflows"


def test_the_remote_code_check_sees_nested_keys():
    definition = {
        "steps": [{"pipeline": {"from_pretrained_arguments": {"custom_pipeline": "x"}}}]
    }

    assert [k for k, _ in _walk(definition) if k in REMOTE_CODE_KEYS] == [
        "custom_pipeline"
    ]


def _step(definition, name):
    return next(step for step in definition["steps"] if step["name"] == name)


class TestLtxTwoStage:
    """LTX-2.5's distilled two-stage flow is three moves: eight sigmas at half
    size, a 2x latent upsample, then renoise and three more sigmas at full size.
    The renoise scale is the first stage-two sigma, which no reference syntax can
    name, so the template carries the literal and this test ties it to the library."""

    def _definition(self):
        path = os.path.join(
            REPO_ROOT, "workflows", "templates", "ltx2", "two-stage.json"
        )
        return json.load(open(path, encoding="utf-8"))

    def test_the_renoise_scale_is_the_first_stage_two_sigma(self):
        from diffusers.pipelines.ltx2.utils import STAGE_2_DISTILLED_SIGMA_VALUES

        refine = _step(self._definition(), "refine")

        assert (
            refine["pipeline"]["arguments"]["noise_scale"]
            == STAGE_2_DISTILLED_SIGMA_VALUES[0]
        )

    def test_the_refine_pass_runs_the_stage_two_schedule_on_the_upsampled_latents(self):
        refine = _step(self._definition(), "refine")
        arguments = refine["pipeline"]["arguments"]

        assert (
            arguments["sigmas"]
            == "constant:diffusers.pipelines.ltx2.utils.STAGE_2_DISTILLED_SIGMA_VALUES"
        )
        assert arguments["latents"] == "previous_result:upscale.frames"
        assert arguments["audio_latents"] == "previous_result:base.audio"

    def test_the_base_and_refine_passes_share_one_pipeline(self):
        definition = self._definition()
        base = _step(definition, "base")["pipeline"]
        refine = _step(definition, "refine")["pipeline"]

        base_cache_key = {k: v for k, v in base.items() if k != "arguments"}
        refine_cache_key = {k: v for k, v in refine.items() if k != "arguments"}

        assert base_cache_key == refine_cache_key
        assert not _step(definition, "base").get("release_pipeline", False)


LINK_PATTERN = re.compile(r"\]\(([^)]+)\)")
READMES = sorted(
    os.path.relpath(path, REPO_ROOT)
    for path in glob.glob(
        os.path.join(REPO_ROOT, "workflows", "templates", "**", "README.md"),
        recursive=True,
    )
)


@pytest.mark.parametrize("path", READMES)
def test_every_readme_link_resolves(path):
    """A README is a reading-order map over the templates beside it. The
    templates were renamed once and every link in the map broke; this keeps the
    map pointing at files that exist."""
    text = open(os.path.join(REPO_ROOT, path), encoding="utf-8").read()
    base = os.path.dirname(os.path.join(REPO_ROOT, path))

    for target in LINK_PATTERN.findall(text):
        if target.startswith(("http://", "https://", "#")):
            continue
        target = target.split("#", 1)[0]
        assert os.path.exists(os.path.join(base, target)), (
            f"{path} links to {target}, which does not exist"
        )


COSTED = {
    "workflows/templates/minimax/music-video.json": 35,
    "workflows/templates/minimax/dialogue-short.json": 42,
    "workflows/templates/assemble-and-score.json": 0.2,
    "workflows/templates/dissolve-between-shots.json": 0.2,
}


@pytest.mark.parametrize("path,minutes", sorted(COSTED.items()))
def test_the_cut_templates_quote_a_measured_cost(path, minutes):
    """Measured on an RTX 3090 (the cut templates 2026-09-10); without a
    figure an agent cannot quote a price before spending 40 minutes of GPU.

    composable-references (27.4 min) and reference-to-video (8.0) were
    measured at 20 steps against no adapter, and #149 put the Ref2VA turbo
    LoRA on those templates at 9 - so their figures went with the schedule
    they described rather than being scaled, `cost` being measured and never
    derived. The three here kept their step count and their canvas; only
    which adapter loads changed, at the same rank and file size."""
    definition = json.load(open(os.path.join(REPO_ROOT, path), encoding="utf-8"))
    entry = definition["cost"][0]
    assert entry["name"] == "RTX 3090"
    assert entry["minutes"] == minutes
